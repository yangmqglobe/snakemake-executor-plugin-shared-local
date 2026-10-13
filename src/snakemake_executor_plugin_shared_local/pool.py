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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError

# Fraction of detected resources reserved for the operating system, other users
# and non-Snakemake work. Used only as a default; explicit settings win.
DEFAULT_RESERVE_FRACTION = 0.10

# The single name under which memory is arbitrated, on both sides of the pool:
# ``Pool.as_dict()`` publishes it and ``lease.normalize_resources()`` produces it.
# It is Snakemake's own resource name, deliberately: it is what jobs declare via
# ``resources: mem_mb=...`` and what ``--default-resources mem_mb=...`` fills in.
#
# This constant exists because the two sides once disagreed (the pool published
# ``mem_mib``, requests used ``mem_mb``). ``policy.fits()`` treats a dimension the
# pool does not declare as *unconstrained*, so that mismatch silently disabled all
# memory arbitration. Any new pool dimension must be spelled identically here, and
# the test suite pins the invariant.
MEMORY_DIMENSION = "mem_mb"


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
    mem_mb: int | None

    def as_dict(self) -> dict[str, int]:
        data = {"cores": self.cores}
        if self.mem_mb is not None:
            data[MEMORY_DIMENSION] = self.mem_mb
        return data


# Spellings of the memory dimension accepted from the outside world. A pool
# declaration written by an older plugin used ``mem_mib``; both must resolve to the
# one dimension actually compared against job requests, or a pre-existing
# ``pool.json`` would look like a conflict (or, worse, be ignored).
_MEMORY_ALIASES = ("mem_mb", "mem_mib", "mem")


def canonicalize_pool(pool: Mapping[str, Any]) -> dict[str, int]:
    """Normalize a pool declaration to canonical dimension names.

    Aliases of memory collapse onto :data:`MEMORY_DIMENSION`, preferring the
    canonical spelling when several are present.

    An unusable value (non-numeric, boolean, zero or negative) raises
    :class:`ConfigError` rather than being dropped. Dropping is the dangerous
    choice: :func:`policy.fits` treats a dimension absent from the pool as
    *unconstrained*, so silently discarding ``cores`` would remove the CPU ceiling
    altogether - the same class of bug as the memory-dimension mismatch this
    module exists to prevent. A declaration nobody can interpret must fail loudly.
    """
    canonical: dict[str, int] = {}
    for name, value in pool.items():
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ConfigError(
                f"Pool dimension {name!r} has unusable value {value!r}: expected a "
                f"positive integer."
            )
        try:
            amount = int(value)
        except (TypeError, ValueError) as err:
            raise ConfigError(
                f"Pool dimension {name!r} has unusable value {value!r}: expected a "
                f"positive integer."
            ) from err
        if amount < 1:
            raise ConfigError(f"Pool dimension {name!r} must be >= 1, got {value!r}.")
        dimension = MEMORY_DIMENSION if name in _MEMORY_ALIASES else name
        if dimension in canonical and name != MEMORY_DIMENSION:
            # The canonical spelling wins over any alias, whatever the dict order.
            continue
        canonical[dimension] = amount
    return canonical


def compute_pool(
    pool_cores: int | None = None,
    pool_mem_mb: int | None = None,
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

    if pool_mem_mb is not None:
        mem_mb: int | None = int(pool_mem_mb)
        if mem_mb < 1:
            raise ConfigError(f"pool_mem must be >= 1 MiB, got {mem_mb}.")
    else:
        try:
            mem_mb = apply_reserve(detect_mem_total_mib(), reserve_fraction)
        except ConfigError:
            # Memory detection failing must not break CPU coordination.
            mem_mb = None

    return Pool(cores=cores, mem_mb=mem_mb)
