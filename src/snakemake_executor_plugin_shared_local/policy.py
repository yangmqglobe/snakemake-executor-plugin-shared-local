"""Fairness policy: deciding which waiting instance may admit the next job.

Ranking is a lexicographic key so the ordering is total and deterministic:

1. ``instance_weight`` descending - operators can prioritise one workflow.
2. ``job_priority`` descending - Snakemake's per-rule priority.
3. ``enqueued_at`` ascending - FIFO tie-break.

Aging is applied to the priority component so a long-waiting job eventually
outranks newer work instead of starving indefinitely.

Note on ``priority``: it is *not* part of the stable executor plugin interface.
Snakemake's ``priority`` is also per-workflow, so equal values in two different
workflows are not comparable. The executor therefore probes for it and degrades
to weight+FIFO when unavailable.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Waiter:
    """A pending admission request, as seen by the fairness policy."""

    instance_id: str
    weight: float
    priority: int
    enqueued_at: float
    # Resource request, used to decide whether a candidate could fit at all.
    resources: Mapping[str, int]

    def effective_priority(
        self, now: float, aging_seconds: int, aging_step: int
    ) -> float:
        """Return priority with an aging bonus applied.

        With ``aging_seconds <= 0`` or ``aging_step <= 0`` aging is disabled and
        the raw priority is returned.
        """
        if aging_seconds <= 0 or aging_step <= 0:
            return float(self.priority)
        waited = max(0.0, now - self.enqueued_at)
        bonus = (waited / aging_seconds) * aging_step
        return float(self.priority) + bonus

    def rank(
        self, now: float, aging_seconds: int, aging_step: int
    ) -> tuple[float, float, float]:
        """Return a sort key where *smaller* means *more deserving*."""
        return (
            -float(self.weight),
            -self.effective_priority(now, aging_seconds, aging_step),
            float(self.enqueued_at),
        )


def fits(
    request: Mapping[str, int], used: Mapping[str, int], pool: Mapping[str, int]
) -> bool:
    """Return whether ``request`` fits into ``pool`` given ``used``.

    Dimensions absent from ``pool`` are unconstrained. A dimension absent from
    ``used`` counts as zero.
    """
    for name, need in request.items():
        capacity = pool.get(name)
        if capacity is None:
            continue
        if used.get(name, 0) + need > capacity:
            return False
    return True


def would_fit(
    waiter: Waiter,
    request: Mapping[str, int],
    used: Mapping[str, int],
    pool: Mapping[str, int],
) -> bool:
    """Return whether another waiter could be admitted once we take ``request``."""
    hypothetical = dict(used)
    for name, need in request.items():
        hypothetical[name] = hypothetical.get(name, 0) + need
    return fits(waiter.resources, hypothetical, pool)


def should_yield(
    self_waiter: Waiter,
    others: list[Waiter],
    *,
    now: float,
    aging_seconds: int,
    aging_step: int,
    used: Mapping[str, int],
    pool: Mapping[str, int],
    self_resources: Mapping[str, int],
) -> tuple[bool, str | None]:
    """Decide whether ``self_waiter`` must let someone else go first.

    Returns ``(should_yield, blocking_instance_id)``.

    We yield when another waiter both (a) outranks us and (b) could actually be
    admitted in our place. Requirement (b) prevents a classic deadlock: if the
    better-ranked job is too large to fit now, blocking on it would stall the
    pool even though our smaller job fits.
    """
    self_key = self_waiter.rank(now, aging_seconds, aging_step)
    for other in others:
        if other.instance_id == self_waiter.instance_id:
            continue
        if other.rank(now, aging_seconds, aging_step) >= self_key:
            continue
        if would_fit(other, self_resources, used, pool):
            return True, other.instance_id
    return False, None


def probe_priority(job: Any, logger: Any = None) -> int:
    """Best-effort read of a job's Snakemake priority.

    Returns ``0`` when the attribute is unavailable, which simply collapses the
    priority component of the ranking. A debug line is emitted once per job by
    the caller, not here.
    """
    try:
        value = job.priority
    except Exception:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
