"""Tests for the flock-based lease manager.

These tests exercise the mechanisms the whole plugin rests on: atomic admission,
crash recovery, refusal to hang, group atomicity and pool negotiation.
"""

import fcntl
import json
import multiprocessing
import os
import signal
import stat
import time
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local.lease import (
    INSTANCE_SUFFIX,
    LEASE_SUFFIX,
    POOL_FILE,
    SCHEMA_VERSION,
    AdmissionImpossible,
    LeaseError,
    LeaseManager,
    normalize_resources,
)
from snakemake_executor_plugin_shared_local.namespace import Namespace


def make_namespace(
    tmp_path: Path, mode: str = "user", sticky: bool = False
) -> Namespace:
    """Build a Namespace pointing at an isolated directory."""
    path = tmp_path / f"ns-{mode}"
    path.mkdir(parents=True, exist_ok=True)
    dir_mode = {"user": 0o700, "group": 0o770, "system": 0o777}[mode]
    os.chmod(path, dir_mode)
    if sticky:
        os.chmod(path, dir_mode | 0o1000)
    return Namespace(
        mode=mode,
        path=path,
        dir_mode=dir_mode,
        sticky=sticky,
        uid=os.getuid(),
        gid=os.getgid(),
    )


def make_manager(ns: Namespace, pool=None, **kwargs) -> LeaseManager:
    kwargs.setdefault("instance_id", f"test-{os.getpid()}-{time.time_ns() % 100000}")
    kwargs.setdefault("poll_interval", 0.01)
    return LeaseManager(ns, pool or {"cores": 4, "mem_mb": 8192}, **kwargs)


# --------------------------------------------------------------- normalization


class TestNormalizeResources:
    def test_threads_become_cores(self):
        assert normalize_resources({}, threads=4)["cores"] == 4

    def test_zero_threads_become_one_core(self):
        # A zero-thread job still occupies one slot.
        assert normalize_resources({}, threads=0)["cores"] == 1

    def test_drops_internal_and_derived_keys(self):
        result = normalize_resources(
            {"_nodes": 1, "_cores": 2, "mem_mib": 500, "mem_mb": 600, "mem": "600 MB"},
            threads=2,
        )
        assert result == {"cores": 2, "mem_mb": 600}

    def test_keeps_custom_named_resources(self):
        result = normalize_resources({"foo": 3, "gpu": 1}, threads=1)
        assert result["foo"] == 3
        assert result["gpu"] == 1

    def test_ignores_non_integer_values(self):
        result = normalize_resources({"ratio": 0.5, "txt": "abc", "zero": 0}, threads=1)
        assert result == {"cores": 1}

    def test_memory_excluded_when_disabled(self):
        result = normalize_resources({"mem_mb": 100}, threads=1, with_mem=False)
        assert "mem_mb" not in result

    def test_digit_strings_are_counted(self):
        result = normalize_resources({"foo": "7"}, threads=1)
        assert result["foo"] == 7


# ------------------------------------------------------------------ admission


class TestAdmission:
    def test_admits_when_pool_is_free(self, tmp_path):
        manager = make_manager(make_namespace(tmp_path))
        handle = manager.acquire("j1", {"cores": 2})
        assert handle.path.exists()
        assert json.loads(handle.path.read_text())["resources"] == {"cores": 2}
        manager.release("j1")
        assert not handle.path.exists()

    def test_lease_is_word_readable_for_other_users(self, tmp_path):
        """umask must not make leases invisible: that would break accounting."""
        old = os.umask(0o077)
        try:
            manager = make_manager(make_namespace(tmp_path))
            handle = manager.acquire("j1", {"cores": 1})
        finally:
            os.umask(old)
        mode = stat.S_IMODE(os.stat(handle.path).st_mode)
        assert mode == 0o644, f"lease mode {oct(mode)} would be unreadable by peers"
        manager.release("j1")

    def test_impossible_request_fails_immediately(self, tmp_path):
        manager = make_manager(make_namespace(tmp_path), pool={"cores": 4})
        started = time.monotonic()
        with pytest.raises(AdmissionImpossible, match="requires cores=16"):
            manager.acquire("big", {"cores": 16})
        # Must not wait: waiting here would hang the workflow forever.
        assert time.monotonic() - started < 2

    def test_error_message_names_the_setting_to_change(self, tmp_path):
        manager = make_manager(make_namespace(tmp_path), pool={"cores": 4})
        with pytest.raises(AdmissionImpossible, match="shared-local-pool-cores"):
            manager.acquire("big", {"cores": 8})

    def test_exhausted_pool_blocks_until_release(self, tmp_path):
        ns = make_namespace(tmp_path)
        holder = make_manager(ns, pool={"cores": 2}, instance_id="holder")
        waiter = make_manager(ns, pool={"cores": 2}, instance_id="waiter")
        holder.acquire("h", {"cores": 2})

        import threading

        acquired = threading.Event()

        def wait_for_it():
            waiter.acquire("w", {"cores": 2})
            acquired.set()

        thread = threading.Thread(target=wait_for_it, daemon=True)
        thread.start()
        assert not acquired.wait(0.4), "waiter must not be admitted while pool is full"
        holder.release("h")
        assert acquired.wait(5), "waiter should be admitted once resources free up"
        waiter.release("w")

    def test_snapshot_counts_only_live_leases(self, tmp_path):
        ns = make_namespace(tmp_path)
        a = make_manager(ns, instance_id="a")
        b = make_manager(ns, instance_id="b")
        a.acquire("a1", {"cores": 2, "mem_mb": 100})
        b.acquire("b1", {"cores": 1, "mem_mb": 50})
        used, warnings = a.snapshot()
        assert used["cores"] == 3
        assert used["mem_mb"] == 150
        assert warnings == []

    def test_group_admission_is_atomic(self, tmp_path):
        """All members are reserved in one critical section, or none are."""
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, pool={"cores": 4})
        # A pipe group's aggregated request must be satisfiable as a whole.
        handle = manager.acquire("group", {"cores": 4})
        assert handle.resources == {"cores": 4}
        manager.release("group")

    def test_second_group_cannot_overlap_first(self, tmp_path):
        """A group larger than the remaining pool must wait, not overlap.

        Note the distinction from :meth:`test_impossible_request_fails_immediately`:
        a request that can *never* fit fails at once, while one that merely does not
        fit *yet* waits. Here 2 cores fit once the first group releases.
        """
        ns = make_namespace(tmp_path)
        a = make_manager(ns, pool={"cores": 3}, instance_id="a")
        b = make_manager(ns, pool={"cores": 3}, instance_id="b", admit_timeout=1)
        a.acquire("g1", {"cores": 3})
        started = time.monotonic()
        with pytest.raises(LeaseError, match="Timed out"):
            b.acquire("g2", {"cores": 3})
        # It waited rather than overlapping.
        assert time.monotonic() - started >= 0.9
        used, _ = b.snapshot()
        assert used["cores"] == 3, "the second group must not have been admitted"
        a.release("g1")


# -------------------------------------------------------------- crash recovery


def _hold_lease_and_die(
    directory: str, key: str, ready: str
) -> None:  # pragma: no cover
    """Child process: hold a lease, signal readiness, then sleep until killed."""
    from snakemake_executor_plugin_shared_local.lease import LeaseManager
    from snakemake_executor_plugin_shared_local.namespace import Namespace

    ns = Namespace(
        mode="user",
        path=Path(directory),
        dir_mode=0o700,
        sticky=False,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    manager = LeaseManager(ns, {"cores": 4}, instance_id="crashy")
    manager.acquire(key, {"cores": 4})
    Path(ready).write_text("ready")
    time.sleep(60)


class TestCrashRecovery:
    def test_killed_holder_lease_is_reclaimable(self, tmp_path):
        """The kernel releases flock on death, so a stale lease counts as zero."""
        ns = make_namespace(tmp_path)
        ready = tmp_path / "ready"
        proc = multiprocessing.Process(
            target=_hold_lease_and_die, args=(str(ns.path), "k1", str(ready))
        )
        proc.start()
        try:
            deadline = time.time() + 10
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.05)
            assert ready.exists(), "child failed to take its lease"

            survivor = make_manager(ns, instance_id="survivor")
            used, _ = survivor.snapshot()
            assert used.get("cores") == 4, "live lease must be counted"

            os.kill(proc.pid, signal.SIGKILL)
            proc.join(10)
            assert not proc.is_alive()

            # The lock died with the process, so the lease no longer blocks.
            used_after, _ = survivor.snapshot()
            assert used_after.get("cores", 0) == 0, "dead lease must count as zero"

            # And the survivor can now be admitted into the full pool.
            handle = survivor.acquire("new", {"cores": 4})
            survivor.release("new")
            assert handle is not None
        finally:
            if proc.is_alive():  # pragma: no cover - cleanup
                proc.kill()
                proc.join(5)

    def test_stale_lease_in_sticky_dir_still_counts_zero(self, tmp_path):
        """Counting correctness must not depend on being able to delete."""
        ns = make_namespace(tmp_path, mode="system", sticky=True)
        stale = ns.path / f"other-instance-1-abcdef{LEASE_SUFFIX}"
        stale.write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 4}})
        )
        os.chmod(stale, 0o644)

        manager = make_manager(ns, pool={"cores": 4}, instance_id="me")
        used, warnings = manager.snapshot()
        assert used.get("cores", 0) == 0, "unlocked lease must count as zero"
        assert warnings == []
        # Admitted despite the leftover file.
        handle = manager.acquire("fresh", {"cores": 4})
        manager.release("fresh")
        assert handle is not None


# ------------------------------------------------------------------ accounting


class TestAccountingEdgeCases:
    def test_unreadable_lease_is_fatal_in_shared_namespace(self, tmp_path):
        """Silently ignoring it would under-count and permit oversubscription."""
        ns = make_namespace(tmp_path, mode="system")
        lease = ns.path / f"other-1-abc{LEASE_SUFFIX}"
        lease.write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 1}})
        )

        # Simulate a live-but-unreadable lease held by someone else.
        fd = os.open(lease, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if os.getuid() == 0:  # pragma: no cover - root can always read
                pytest.skip("root bypasses file permissions")
            os.chmod(lease, 0o600)  # ours, so readable by us; force via monkeypatch
            manager = make_manager(ns, instance_id="me")
            import snakemake_executor_plugin_shared_local.lease as lease_module

            original = lease_module._read_lease

            def fake(path):
                return lease_module._Unreadable("simulated unreadable lease")

            lease_module._read_lease = fake
            try:
                with pytest.raises(LeaseError, match="invisible"):
                    manager.acquire("j", {"cores": 1})
            finally:
                lease_module._read_lease = original
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_unreadable_lease_only_warns_in_private_namespace(self, tmp_path):
        ns = make_namespace(tmp_path, mode="user")
        # A lease file must exist, otherwise the scan has nothing to inspect.
        (ns.path / f"other-1-abc{LEASE_SUFFIX}").write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 1}})
        )
        manager = make_manager(ns, instance_id="me")
        import snakemake_executor_plugin_shared_local.lease as lease_module

        original = lease_module._read_lease
        lease_module._read_lease = lambda path: lease_module._Unreadable("simulated")
        try:
            used, warnings = manager.snapshot()
            assert warnings and "simulated" in warnings[0]
        finally:
            lease_module._read_lease = original

    def test_newer_schema_version_fails_loudly(self, tmp_path):
        ns = make_namespace(tmp_path)
        lease = ns.path / f"future-1-abc{LEASE_SUFFIX}"
        lease.write_text(
            json.dumps({"version": SCHEMA_VERSION + 1, "resources": {"cores": 1}})
        )
        fd = os.open(lease, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            manager = make_manager(ns, instance_id="me")
            with pytest.raises(LeaseError, match="newer plugin"):
                manager.snapshot()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_usage_ignores_files_without_extension(self, tmp_path):
        ns = make_namespace(tmp_path)
        (ns.path / "random.txt").write_text("junk")
        manager = make_manager(ns, instance_id="me")
        used, warnings = manager.snapshot()
        assert used == {}
        assert warnings == []


# -------------------------------------------------------------- pool agreement


class TestPoolNegotiation:
    def test_first_instance_publishes_pool(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, pool={"cores": 8})
        manager.negotiate_pool()
        data = json.loads((ns.path / POOL_FILE).read_text())
        assert data["pool"] == {"cores": 8}
        assert data["version"] == SCHEMA_VERSION

    def test_matching_pool_is_accepted(self, tmp_path):
        ns = make_namespace(tmp_path)
        make_manager(ns, pool={"cores": 8}, instance_id="first").negotiate_pool()
        make_manager(ns, pool={"cores": 8}, instance_id="second").negotiate_pool()

    def test_conflicting_pool_errors_by_default(self, tmp_path):
        ns = make_namespace(tmp_path)
        make_manager(ns, pool={"cores": 8}, instance_id="first").negotiate_pool()
        with pytest.raises(LeaseError, match="mismatch"):
            make_manager(ns, pool={"cores": 4}, instance_id="second").negotiate_pool()

    def test_conflicting_pool_can_warn_instead(self, tmp_path):
        ns = make_namespace(tmp_path)
        make_manager(ns, pool={"cores": 8}, instance_id="first").negotiate_pool()
        lenient = make_manager(
            ns, pool={"cores": 4}, instance_id="second", pool_conflict="warn"
        )
        lenient.negotiate_pool()
        assert any("mismatch" in warning for warning in lenient.warnings)

    def test_pool_file_is_word_readable(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, instance_id="me")
        manager.negotiate_pool()
        mode = stat.S_IMODE(os.stat(ns.path / POOL_FILE).st_mode)
        assert mode == 0o644

    def test_pool_file_without_a_pool_mapping_still_counts_as_a_mismatch(
        self, tmp_path
    ):
        """A file that exists but declares nothing must not be silently ignored.

        Skipping the comparison would let every instance run with a pool the other
        never agreed to - the exact failure this negotiation exists to prevent.
        """
        ns = make_namespace(tmp_path)
        (ns.path / POOL_FILE).write_text("{}")
        manager = make_manager(ns, pool={"cores": 8}, instance_id="second")
        with pytest.raises(LeaseError, match="mismatch"):
            manager.negotiate_pool()

    def test_legacy_mem_mib_pool_is_not_a_mismatch(self, tmp_path):
        """An upgrade must not turn a healthy shared namespace into an abort."""
        ns = make_namespace(tmp_path)
        (ns.path / POOL_FILE).write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "host": "otherhost",
                    "pool": {"cores": 8, "mem_mib": 8192},
                    "created_by": 0,
                    "created_at": 0,
                }
            )
        )
        manager = make_manager(ns, pool={"cores": 8, "mem_mb": 8192}, instance_id="me")
        manager.negotiate_pool()
        assert manager.pool == {"cores": 8, "mem_mb": 8192}

    def test_legacy_mem_mib_pool_with_different_cores_is_a_mismatch(self, tmp_path):
        """Canonicalization must not mask a genuine disagreement."""
        ns = make_namespace(tmp_path)
        (ns.path / POOL_FILE).write_text(
            json.dumps(
                {"version": SCHEMA_VERSION, "pool": {"cores": 4, "mem_mib": 8192}}
            )
        )
        manager = make_manager(ns, pool={"cores": 8, "mem_mb": 8192}, instance_id="me")
        with pytest.raises(LeaseError, match="mismatch"):
            manager.negotiate_pool()

    def test_uninterpretable_pool_file_is_a_mismatch_not_a_crash(self, tmp_path):
        """A declaration we cannot read must abort cleanly, not raise KeyError."""
        ns = make_namespace(tmp_path)
        (ns.path / POOL_FILE).write_text(
            json.dumps({"version": SCHEMA_VERSION, "pool": {"cores": "lots"}})
        )
        manager = make_manager(ns, pool={"cores": 8}, instance_id="me")
        with pytest.raises(LeaseError, match="mismatch"):
            manager.negotiate_pool()


# ------------------------------------------------------------- instance claims


class TestInstanceClaims:
    def test_register_and_unregister(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, instance_id="me")
        manager.register()
        assert (ns.path / f"me{INSTANCE_SUFFIX}").exists()
        manager.unregister()

    def test_same_instance_id_cannot_be_claimed_twice(self, tmp_path):
        ns = make_namespace(tmp_path)
        first = make_manager(ns, instance_id="dup")
        first.register()
        second = make_manager(ns, instance_id="dup")
        with pytest.raises(OSError):
            second.register()
        first.unregister()

    def test_active_instances_lists_live_claims(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, instance_id="alive")
        manager.register()
        assert "alive" in manager.active_instances()
        manager.unregister()


class TestGarbage:
    def test_report_garbage_finds_stale_leases(self, tmp_path):
        ns = make_namespace(tmp_path)
        stale = ns.path / f"dead-1-xyz{LEASE_SUFFIX}"
        stale.write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 1}})
        )
        manager = make_manager(ns, instance_id="me")
        assert stale.name in manager.report_garbage()

    def test_live_leases_are_not_reported_as_garbage(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, instance_id="me")
        handle = manager.acquire("live", {"cores": 1})
        assert manager.report_garbage() == []
        manager.release("live")
        assert not handle.path.exists()

    def test_release_all_clears_everything(self, tmp_path):
        ns = make_namespace(tmp_path)
        manager = make_manager(ns, instance_id="me")
        manager.acquire("a", {"cores": 1})
        manager.acquire("b", {"cores": 1})
        manager.release_all()
        assert manager.report_garbage() == []

    def test_release_of_unknown_key_is_harmless(self, tmp_path):
        manager = make_manager(make_namespace(tmp_path), instance_id="me")
        manager.release("never-acquired")  # must not raise
