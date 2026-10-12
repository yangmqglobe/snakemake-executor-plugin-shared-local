"""NUMA locality for scheduled jobs.

Measured on the development machine (2 nodes, 64 logical CPUs and ~257 GiB each,
node distance 10 vs 21) an unpinned allocation lands almost entirely on one
arbitrary node, while an explicitly bound one stays local. Binding is therefore a
performance feature, not a limit: it decides *where* memory and threads live, it
does not restrict how much is used. Resource limiting is the lease's job.

Everything here is implemented with ``ctypes`` against libc rather than by
shelling out to ``numactl``, so the plugin keeps its zero-runtime-dependency
property. Two mechanics make this work for Snakemake's execution model:

* ``set_mempolicy(MPOL_BIND)`` and ``sched_setaffinity`` apply to the *calling
  thread*, and the kernel propagates both to children and grandchildren. A job
  spawned by Snakemake's ``--mode subprocess`` therefore inherits the binding.
* Snakemake asks the executor to run a job from a worker thread; setting the
  policy on that thread covers every process it spawns for that job.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import contextlib
import ctypes
import ctypes.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

SYS_SET_MEMPOLICY = 238  # x86_64
MPOL_BIND = 2
MPOL_PREFERRED = 1
MPOL_DEFAULT = 0

NUMA_NODE_DIR = Path("/sys/devices/system/node")


@dataclass
class NumaNode:
    """A NUMA node and its currently free capacity."""

    index: int
    cpus: list[int]
    mem_total_mib: int
    mem_free_mib: int

    @property
    def cpu_count(self) -> int:
        return len(self.cpus)


@dataclass
class NumaTopology:
    nodes: list[NumaNode] = field(default_factory=list)

    @property
    def available(self) -> bool:
        """Whether binding can do anything useful.

        A single-node machine has no locality to exploit, so callers should treat
        binding as a no-op rather than emit misleading warnings.
        """
        return len(self.nodes) > 1

    def total_cpus(self) -> int:
        return sum(node.cpu_count for node in self.nodes)


def _read_cpu_list(path: Path) -> list[int] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                cpus.extend(range(int(low), int(high) + 1))
            except ValueError:
                return None
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                return None
    return cpus


def detect_topology(node_dir: Path = NUMA_NODE_DIR) -> NumaTopology:
    """Read NUMA topology from sysfs. Returns an empty topology on failure."""
    if not node_dir.is_dir():
        return NumaTopology(nodes=[])

    nodes: list[NumaNode] = []
    for entry in sorted(node_dir.iterdir()):
        name = entry.name
        if not name.startswith("node") or not name[4:].isdigit():
            continue
        index = int(name[4:])
        cpus = _read_cpu_list(entry / "cpulist") or []
        mem_total = _read_meminfo_kib(entry / "meminfo", "MemTotal")
        mem_free = _read_meminfo_kib(entry / "meminfo", "MemFree")
        if not cpus:
            continue
        nodes.append(
            NumaNode(
                index=index,
                cpus=cpus,
                mem_total_mib=mem_total,
                mem_free_mib=mem_free,
            )
        )

    # Keep nodes in index order; iterdir order is not guaranteed.
    nodes.sort(key=lambda node: node.index)
    return NumaTopology(nodes=nodes)


def _read_meminfo_kib(path: Path, key: str) -> int:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if f"{key}:" in line:
                parts = line.split()
                # Format: "Node 0 MemTotal:       263658056 kB"
                for token in reversed(parts):
                    if token.isdigit():
                        return int(token) // 1024
    except OSError:
        pass
    return 0


class _NumaBinder:
    """Thin, lazily-initialised wrapper around libc's NUMA syscalls."""

    def __init__(self) -> None:
        self._libc: ctypes.CDLL | None = None
        self._checked = False

    def _load(self) -> ctypes.CDLL | None:
        if self._checked:
            return self._libc
        self._checked = True
        try:
            name = ctypes.util.find_library("c") or "libc.so.6"
            self._libc = ctypes.CDLL(name, use_errno=True)
        except OSError:  # pragma: no cover - exotic environments
            self._libc = None
        return self._libc

    def bind_memory(self, nodes: list[int]) -> bool:
        """Bind memory allocation to ``nodes`` for the calling thread.

        An empty ``nodes`` list resets the policy to ``MPOL_DEFAULT`` so the
        thread stops being pinned to a specific node.
        """
        libc = self._load()
        if libc is None:
            return False
        if not nodes:
            result = libc.syscall(
                SYS_SET_MEMPOLICY, MPOL_DEFAULT, None, ctypes.c_ulong(0)
            )
            return result == 0
        mask = 0
        for node in nodes:
            mask |= 1 << node
        nodemask = ctypes.c_ulong(mask)
        result = libc.syscall(
            SYS_SET_MEMPOLICY, MPOL_BIND, ctypes.byref(nodemask), ctypes.c_ulong(64)
        )
        return result == 0

    def bind_cpus(self, cpus: list[int]) -> bool:
        """Restrict the calling thread to ``cpus`` and return success."""
        if not cpus:
            return False
        try:
            os.sched_setaffinity(0, set(cpus))
        except (AttributeError, OSError):
            return False
        return True


_BINDER = _NumaBinder()


@dataclass
class Binding:
    """The result of attempting to bind the current thread for a job."""

    applied: bool
    node: int | None = None
    reason: str = ""


def select_node(
    topology: NumaTopology,
    *,
    cores: int,
    mem_mib: int | None,
    used_per_node: dict[int, int] | None = None,
) -> NumaNode | None:
    """Pick the node with the most free capacity that can host the request.

    Returns ``None`` when no single node can hold the job (its memory demand
    exceeds every node), signalling the caller to fall back to unbound
    execution.
    """
    if not topology.nodes:
        return None

    used_per_node = used_per_node or {}
    candidates: list[tuple[int, NumaNode]] = []
    for node in topology.nodes:
        if node.cpu_count < cores:
            continue
        if mem_mib is not None and node.mem_total_mib < mem_mib:
            continue
        # Prefer nodes with the most currently free memory, then the most CPUs.
        free = node.mem_free_mib - used_per_node.get(node.index, 0)
        candidates.append((free, node))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].cpu_count), reverse=True)
    return candidates[0][1]


def bind_current_thread(
    node: NumaNode,
    *,
    cores: int,
    logger: logging.Logger | None = None,
) -> Binding:
    """Bind the calling thread (and thus its children) to ``node``.

    CPU affinity is restricted to ``cores`` CPUs so a job with ``threads: 4`` does
    not get scheduled across all 64 CPUs of the node, which would hurt locality
    for its own data.
    """
    cpus = node.cpus[: max(1, cores)]
    memory_ok = _BINDER.bind_memory([node.index])
    cpus_ok = _BINDER.bind_cpus(cpus)

    if not memory_ok and not cpus_ok:
        if logger is not None:
            logger.debug(
                "NUMA binding to node %s unavailable (memory=%s, cpus=%s); "
                "continuing unbound.",
                node.index,
                memory_ok,
                cpus_ok,
            )
        return Binding(applied=False, node=node.index, reason="syscalls unavailable")

    if not memory_ok and logger is not None:
        logger.debug(
            "Memory binding to node %s failed; CPU binding applied.", node.index
        )
    if not cpus_ok and logger is not None:
        logger.debug(
            "CPU binding to node %s failed; memory binding applied.", node.index
        )

    return Binding(applied=True, node=node.index)


def reset_current_thread() -> None:
    """Undo NUMA binding on the calling thread.

    Called after a job finishes so a thread handed to the next job of the
    executor's thread pool does not inherit another job's placement. The process's
    original CPU affinity is captured once, on first use, and restored verbatim.
    """
    global _ORIGINAL_AFFINITY
    if _ORIGINAL_AFFINITY is None:
        return  # nothing was ever bound
    with contextlib.suppress(AttributeError, OSError):
        os.sched_setaffinity(0, _ORIGINAL_AFFINITY)
    _BINDER.bind_memory([])  # MPOL_BIND with an empty mask resets to default


_ORIGINAL_AFFINITY: set[int] | None = None


def capture_original_affinity() -> set[int] | None:
    """Record the process CPU affinity once, before any job binds a thread."""
    global _ORIGINAL_AFFINITY
    if _ORIGINAL_AFFINITY is None:
        try:
            _ORIGINAL_AFFINITY = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):  # pragma: no cover
            return None
    return _ORIGINAL_AFFINITY
