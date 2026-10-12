"""Tests for the fairness policy."""

from snakemake_executor_plugin_shared_local.policy import (
    Waiter,
    fits,
    probe_priority,
    should_yield,
    would_fit,
)


def make_waiter(instance="a", weight=1.0, priority=0, enqueued=0.0, resources=None):
    return Waiter(
        instance_id=instance,
        weight=weight,
        priority=priority,
        enqueued_at=enqueued,
        resources=resources or {"cores": 1},
    )


class TestFits:
    def test_fits_when_within_capacity(self):
        assert fits({"cores": 2}, {"cores": 1}, {"cores": 4})

    def test_does_not_fit_when_exceeding(self):
        assert not fits({"cores": 4}, {"cores": 1}, {"cores": 4})

    def test_unconstrained_dimension_is_ignored(self):
        # A pool without the dimension cannot constrain it.
        assert fits({"weird": 999}, {}, {"cores": 4})

    def test_missing_used_counts_as_zero(self):
        assert fits({"cores": 4}, {}, {"cores": 4})


class TestAging:
    def test_aging_disabled_returns_raw_priority(self):
        waiter = make_waiter(priority=5, enqueued=0.0)
        assert (
            waiter.effective_priority(now=1000.0, aging_seconds=0, aging_step=1) == 5.0
        )

    def test_aging_increases_with_wait(self):
        waiter = make_waiter(priority=0, enqueued=0.0)
        early = waiter.effective_priority(now=10.0, aging_seconds=60, aging_step=1)
        late = waiter.effective_priority(now=600.0, aging_seconds=60, aging_step=1)
        assert late > early

    def test_aging_is_monotonic(self):
        waiter = make_waiter(priority=0, enqueued=100.0)
        values = [
            waiter.effective_priority(now=t, aging_seconds=30, aging_step=2)
            for t in (100, 130, 200, 500)
        ]
        assert values == sorted(values)

    def test_aging_can_eventually_outrank_newer_high_priority_work(self):
        old_low = make_waiter(priority=0, enqueued=0.0)
        new_high = make_waiter(priority=10, enqueued=9990.0)
        now = 10000.0
        assert old_low.effective_priority(
            now, aging_seconds=10, aging_step=1
        ) > new_high.effective_priority(now, aging_seconds=10, aging_step=1)


class TestRanking:
    def test_higher_weight_ranks_first(self):
        heavy = make_waiter(instance="heavy", weight=10.0)
        light = make_waiter(instance="light", weight=1.0)
        now = 0.0
        assert heavy.rank(now, 0, 0) < light.rank(now, 0, 0)

    def test_weight_beats_priority(self):
        heavy = make_waiter(instance="heavy", weight=5.0, priority=0)
        light = make_waiter(instance="light", weight=1.0, priority=100)
        now = 0.0
        assert heavy.rank(now, 0, 0) < light.rank(now, 0, 0)

    def test_priority_breaks_weight_ties(self):
        high = make_waiter(instance="high", weight=1.0, priority=100)
        low = make_waiter(instance="low", weight=1.0, priority=1)
        now = 0.0
        assert high.rank(now, 0, 0) < low.rank(now, 0, 0)

    def test_fifo_breaks_priority_ties(self):
        first = make_waiter(instance="first", priority=5, enqueued=1.0)
        second = make_waiter(instance="second", priority=5, enqueued=2.0)
        now = 10.0
        assert first.rank(now, 0, 0) < second.rank(now, 0, 0)


class TestShouldYield:
    def _yield(self, me, others, used=None, pool=None, aging=(0, 0)):
        return should_yield(
            me,
            others,
            now=1000.0,
            aging_seconds=aging[0],
            aging_step=aging[1],
            used=used or {},
            pool=pool or {"cores": 8},
            self_resources=me.resources,
        )

    def test_yields_to_better_ranked_waiter_that_fits(self):
        me = make_waiter("me", priority=1)
        other = make_waiter("other", priority=100)
        yield_to, blocker = self._yield(me, [other])
        assert yield_to is True
        assert blocker == "other"

    def test_does_not_yield_to_worse_ranked_waiter(self):
        me = make_waiter("me", priority=100)
        other = make_waiter("other", priority=1)
        assert self._yield(me, [other])[0] is False

    def test_does_not_yield_to_own_instance(self):
        me = make_waiter("same", priority=1)
        other = make_waiter("same", priority=100)
        assert self._yield(me, [other])[0] is False

    def test_does_not_yield_when_other_cannot_fit(self):
        """Avoids the classic deadlock: never block on a job that cannot run."""
        me = make_waiter("me", priority=1, resources={"cores": 1})
        other = make_waiter("other", priority=100, resources={"cores": 100})
        used = {"cores": 1}
        yield_to, blocker = self._yield(me, [other], used=used, pool={"cores": 8})
        assert yield_to is False
        assert blocker is None

    def test_yields_when_no_others(self):
        assert self._yield(make_waiter("me"), [])[0] is False

    def test_aging_lets_a_starved_waiter_win(self):
        me = make_waiter("me", priority=0, enqueued=0.0)
        other = make_waiter("other", priority=50, enqueued=999.0)
        # Without aging the newcomer wins; with aging the long waiter does.
        assert self._yield(me, [other])[0] is True
        assert self._yield(me, [other], aging=(1, 1000))[0] is False


class TestWouldFit:
    def test_accounts_for_requested_resources(self):
        other = make_waiter(resources={"cores": 2})
        # Our 4 cores plus theirs exceeds the pool.
        assert not would_fit(other, {"cores": 4}, {"cores": 0}, {"cores": 5})

    def test_true_when_capacity_remains(self):
        other = make_waiter(resources={"cores": 2})
        assert would_fit(other, {"cores": 1}, {"cores": 0}, {"cores": 5})


class TestProbePriority:
    def test_reads_priority_attribute(self):
        class Job:
            priority = 42

        assert probe_priority(Job()) == 42

    def test_degrades_when_missing(self):
        class Job:
            pass

        assert probe_priority(Job()) == 0

    def test_degrades_on_unconvertible_value(self):
        class Job:
            priority = "high"

        assert probe_priority(Job()) == 0

    def test_degrades_when_property_raises(self):
        class Job:
            @property
            def priority(self):
                raise RuntimeError("not available")

        assert probe_priority(Job()) == 0
