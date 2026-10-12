"""Hardware detection and resource-pool computation.

The pool is the single source of truth for how much a coordinated group of
Snakemake instances may use together. It is derived from detected hardware
minus a reserve, unless the user pins it explicitly.

Detection deliberately uses only the standard library: ``os.cpu_count`` and
``/proc/meminfo``. Both are stable across the Linux versions this plugin targets,
and neither introduces a dependency such as ``psutil``.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import os
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError

# Fraction of detected resources reserved for the operating system, other users
# and non-Snakemake work. Used only as a default; explicit settings win.
DEFAULT_RESERVE_FRACTION = 0.10


def detect_cpu_count() -> int:
    """Return the number of CPUs usable by this process.

    ``os.process_cpu_count`` (Python 3.13+) and ``os.sched_getaffinity`` both
    honour CPU affinity masks and cgroup restrictions; plain ``os.cpu_count``
    does not. Prefer the most accurate one available.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        detected = process_cpu_count()
        if detected:
            return int(detected)
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return len(affinity)
    detected = os.cpu_count()
    if not detected:
        raise ConfigError("Cannot detect the number of CPUs on this machine.")
    return int(detected)


def detect_mem_total_mib(meminfo_path: Path | str = "/proc/meminfo") -> int:
    """Return total physical memory in MiB by reading ``/proc/meminfo``."""
    try:
        content = Path(meminfo_path).read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigError(
            f"Cannot read {meminfo_path} to detect memory: {err}"
        ) from err

    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                # /proc/meminfo reports kB.
                return int(parts[1]) // 1024
    raise ConfigError(f"Cannot find MemTotal in {meminfo_path}.")


def apply_reserve(
    total: int, reserve_fraction: float = DEFAULT_RESERVE_FRACTION
) -> int:
    """Return ``total`` minus a reserve, never dropping below 1."""
    if total <= 0:
        raise ConfigError(f"Cannot compute a pool from a total of {total}.")
    if not 0 <= reserve_fraction < 1:
        raise ConfigError(
            f"reserve_fraction must be in [0, 1), got {reserve_fraction!r}."
        )
    reserved = int(total * reserve_fraction)
    return max(1, total - reserved)


@dataclass
class Pool:
    """The resource budget shared by all coordinated instances."""

    cores: int
    # ``None`` means memory is not arbitrated at all (no pool memory known).
    mem_mib: int | None

    def as_dict(self) -> dict[str, int]:
        data = {"cores": self.cores}
        if self.mem_mib is not None:
            data["mem_mib"] = self.mem_mib
        return data


def compute_pool(
    pool_cores: int | None = None,
    pool_mem_mib: int | None = None,
    *,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> Pool:
    """Compute the effective pool from explicit values and detection."""
    if pool_cores is not None:
        cores = int(pool_cores)
        if cores < 1:
            raise ConfigError(f"pool_cores must be >= 1, got {cores}.")
    else:
        cores = apply_reserve(detect_cpu_count(), reserve_fraction)

    if pool_mem_mib is not None:
        mem_mib: int | None = int(pool_mem_mib)
        if mem_mib < 1:
            raise ConfigError(f"pool_mem must be >= 1 MiB, got {mem_mib}.")
    else:
        try:
            mem_mib = apply_reserve(detect_mem_total_mib(), reserve_fraction)
        except ConfigError:
            # Memory detection failing must not break CPU coordination.
            mem_mib = None

    return Pool(cores=cores, mem_mib=mem_mib)
