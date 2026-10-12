"""Tests for the lease manager's coordination paths between instances.

The other lease tests cover admission and accounting. These cover the parts that
only matter when *several* instances contend: advertising an outstanding request,
yielding to a better-ranked peer, and cleaning up after a failed publish.
"""

import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local.lease import (
    INSTANCE_SUFFIX,
    LEASE_SUFFIX,
    PENDING_SUFFIX,
    SCHEMA_VERSION,
    LeaseEntry,
    LeaseError,
    LeaseManager,
    _read_json,
    _read_lease,
    _sanitize,
    _try_lock_read,
    _unlock_close,
    normalize_resources,
)
from snakemake_executor_plugin_shared_local.namespace import Namespace


def make_ns(tmp_path: Path, mode: str = "user", sticky: bool = False) -> Namespace:
    path = tmp_path / f"ns-{mode}"
    path.mkdir(parents=True, exist_ok=True)
    dir_mode = {"user": 0o700, "group": 0o770, "system": 0o777}[mode]
    os.chmod(path, dir_mode | (0o1000 if sticky else 0))
    return Namespace(
        mode=mode,
        path=path,
        dir_mode=dir_mode,
        sticky=sticky,
        uid=os.getuid(),
        gid=os.getgid(),
    )


def manager(ns: Namespace, pool=None, **kwargs) -> LeaseManager:
    kwargs.setdefault("instance_id", f"m-{time.time_ns() % 1000000}")
    kwargs.setdefault("poll_interval", 0.01)
    return LeaseManager(ns, pool or {"cores": 8}, **kwargs)


class TestPendingAdvertisement:
    def test_pending_file_is_written_when_waiting(self, tmp_path):
        ns = make_ns(tmp_path)
        waiter = manager(ns, pool={"cores": 1}, instance_id="waiter", admit_timeout=0)
        # Fill the pool from another instance so the waiter must queue up.
        holder = manager(ns, pool={"cores": 1}, instance_id="holder")
        holder.acquire("h", {"cores": 1})

        import threading

        stop = threading.Event()

        def wait_forever():
            with contextlib.suppress(Exception):
                waiter.acquire("w", {"cores": 1})
            stop.set()

        thread = threading.Thread(target=wait_forever, daemon=True)
        thread.start()
        try:
            deadline = time.time() + 5
            pending = None
            while time.time() < deadline:
                files = list(ns.path.glob(f"*{PENDING_SUFFIX}"))
                if files:
                    pending = files[0]
                    break
                time.sleep(0.02)
            assert pending is not None, "waiter never advertised its request"
            data = json.loads(pending.read_text())
            assert data["instance_id"] == "waiter"
            assert data["resources"] == {"cores": 1}
        finally:
            holder.release("h")
        assert stop.wait(5), "waiter was not admitted after the pool freed up"

    def test_pending_file_is_cleared_after_admission(self, tmp_path):
        ns = make_ns(tmp_path)
        m = manager(ns, pool={"cores": 4}, instance_id="solo")
        m.acquire("j", {"cores": 1})
        assert list(ns.path.glob(f"*{PENDING_SUFFIX}")) == []
        m.release("j")

    def test_waiter_yields_to_a_better_ranked_peer(self, tmp_path):
        """A lower-priority instance must not jump ahead of a waiting peer."""
        ns = make_ns(tmp_path)

        # A peer advertises a high-priority request that fits the free pool.
        peer_pending = ns.path / f"peer{PENDING_SUFFIX}"
        peer_pending.write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "instance_id": "peer",
                    "resources": {"cores": 1},
                    "priority": 100,
                    "weight": 1.0,
                    "enqueued_at": time.time() - 100,
                }
            )
        )

        low = manager(ns, pool={"cores": 4}, instance_id="low", admit_timeout=1)
        # The pool is free, but a better-ranked peer is waiting, so we yield.
        with pytest.raises(LeaseError, match="Timed out"):
            low.acquire("mine", {"cores": 1}, priority=1)
        # We never took the lease.
        assert list(ns.path.glob("*" + LEASE_SUFFIX)) == []

    def test_no_yield_when_peer_request_cannot_fit(self, tmp_path):
        """Yielding to an impossible request would deadlock the pool."""
        ns = make_ns(tmp_path)
        peer_pending = ns.path / f"peer{PENDING_SUFFIX}"
        peer_pending.write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "instance_id": "peer",
                    "resources": {"cores": 999},
                    "priority": 100,
                    "weight": 1.0,
                    "enqueued_at": time.time() - 100,
                }
            )
        )
        m = manager(ns, pool={"cores": 4}, instance_id="me")
        handle = m.acquire("mine", {"cores": 1}, priority=1)
        assert handle.path.exists()
        m.release("mine")

    def test_own_pending_entry_is_ignored(self, tmp_path):
        ns = make_ns(tmp_path)
        m = manager(ns, pool={"cores": 4}, instance_id="self")
        m._publish_pending("x", {"cores": 1}, 100, time.time(), "r")
        assert m._read_other_waiters() == []
        m._clear_pending()

    def test_malformed_pending_entries_are_skipped(self, tmp_path):
        ns = make_ns(tmp_path)
        (ns.path / f"bad{PENDING_SUFFIX}").write_text("{ not json")
        m = manager(ns, pool={"cores": 4}, instance_id="self")
        assert m._read_other_waiters() == []

    def test_clear_pending_is_idempotent(self, tmp_path):
        ns = make_ns(tmp_path)
        m = manager(ns, instance_id="x")
        m._clear_pending()  # nothing to remove; must not raise


class TestPublishFailure:
    def test_lease_is_removed_when_publishing_fails(self, tmp_path, monkeypatch):
        ns = make_ns(tmp_path)
        m = manager(ns, pool={"cores": 4}, instance_id="x")

        def failing_rename(src, dst):
            raise OSError("rename failed")

        monkeypatch.setattr(os, "rename", failing_rename)
        with pytest.raises(OSError, match="rename failed"):
            m.acquire("j", {"cores": 1})

        # No half-written lease or temporary file may be left behind.
        leftovers = sorted(p.name for p in ns.path.iterdir())
        assert not [n for n in leftovers if n.endswith(LEASE_SUFFIX)]
        assert not [n for n in leftovers if n.startswith(".tmp-")]

    def test_lease_records_rule_and_priority(self, tmp_path):
        ns = make_ns(tmp_path)
        m = manager(ns, pool={"cores": 4}, instance_id="x")
        handle = m.acquire("j", {"cores": 1}, priority=7, rule="align")
        data = json.loads(handle.path.read_text())
        assert data["rule"] == "align"
        assert data["priority"] == 7
        assert data["instance_id"] == "x"
        m.release("j")


class TestCollection:
    def test_dead_instance_claims_are_collected(self, tmp_path):
        ns = make_ns(tmp_path)
        dead = ns.path / f"ghost{INSTANCE_SUFFIX}"
        dead.write_text("{}")
        m = manager(ns, pool={"cores": 4}, instance_id="me")
        m.snapshot()
        assert not dead.exists(), "an unlocked instance claim should be collected"

    def test_live_instance_claims_are_preserved(self, tmp_path):
        ns = make_ns(tmp_path)
        live = ns.path / f"alive{INSTANCE_SUFFIX}"
        live.write_text("{}")
        fd = os.open(live, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            m = manager(ns, pool={"cores": 4}, instance_id="me")
            m.snapshot()
            assert live.exists(), "a live instance claim must not be removed"
        finally:
            _unlock_close(fd)

    def test_stale_lease_in_sticky_directory_is_not_deleted(self, tmp_path):
        """Deletion may be impossible; counting must stay correct regardless."""
        ns = make_ns(tmp_path, mode="system", sticky=True)
        ghost = ns.path / f"ghost-1-abc{LEASE_SUFFIX}"
        ghost.write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 2}})
        )
        m = manager(ns, pool={"cores": 4}, instance_id="me")
        used, warnings = m.snapshot()
        assert used.get("cores", 0) == 0
        assert warnings == []

    def test_unregister_removes_the_claim(self, tmp_path):
        ns = make_ns(tmp_path)
        m = manager(ns, pool={"cores": 4}, instance_id="me")
        m.register()
        claim = ns.path / f"me{INSTANCE_SUFFIX}"
        assert claim.exists()
        m.unregister()
        assert not claim.exists()

    def test_unregister_without_register_is_safe(self, tmp_path):
        m = manager(make_ns(tmp_path), instance_id="me")
        m.unregister()

    def test_active_instances_reports_live_and_ignores_dead(self, tmp_path):
        ns = make_ns(tmp_path)
        alive = manager(ns, instance_id="alive")
        alive.register()
        dead_claim = ns.path / f"dead{INSTANCE_SUFFIX}"
        dead_claim.write_text("{}")

        names = alive.active_instances()
        assert "alive" in names
        assert "dead" not in names, (
            "a claim with no live lock is not an active instance"
        )
        alive.unregister()

    def test_report_garbage_ignores_instance_and_pool_files(self, tmp_path):
        ns = make_ns(tmp_path)
        (ns.path / f"somebody{INSTANCE_SUFFIX}").write_text("{}")
        (ns.path / "pool.json").write_text("{}")
        m = manager(ns, instance_id="me")
        assert m.report_garbage() == []


class TestLowLevelHelpers:
    def test_sanitize_replaces_unsafe_characters(self):
        assert _sanitize("a/b c:d") == "a_b_c_d"

    def test_sanitize_truncates_long_names(self):
        assert len(_sanitize("x" * 500)) <= 64

    def test_sanitize_never_returns_empty(self):
        assert _sanitize("") == "job"

    def test_read_json_returns_none_for_missing_file(self, tmp_path):
        assert _read_json(tmp_path / "nope.json") is None

    def test_read_json_returns_none_for_malformed(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert _read_json(path) is None

    def test_read_json_returns_none_for_non_mapping(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]")
        assert _read_json(path) is None

    def test_try_lock_read_returns_none_for_missing_file(self, tmp_path):
        assert _try_lock_read(tmp_path / "nope") is None

    def test_lease_entry_exposes_resources_and_instance(self, tmp_path):
        path = tmp_path / "x.lease"
        path.write_text("{}")
        entry = LeaseEntry(
            path=path,
            data={"resources": {"cores": 2, "ignored": "text"}, "instance_id": "abc"},
        )
        assert entry.resources == {"cores": 2}
        assert entry.instance_id == "abc"

    def test_lease_entry_without_resources_is_empty(self, tmp_path):
        entry = LeaseEntry(path=tmp_path / "x", data={})
        assert entry.resources == {}
        assert entry.instance_id == "?"

    def test_read_lease_returns_none_when_unlocked(self, tmp_path):
        path = tmp_path / "stale.lease"
        path.write_text(
            json.dumps({"version": SCHEMA_VERSION, "resources": {"cores": 1}})
        )
        assert _read_lease(path) is None

    def test_read_lease_reports_unparseable_locked_file(self, tmp_path):
        path = tmp_path / "weird.lease"
        path.write_text("{broken")
        fd = os.open(path, os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            result = _read_lease(path)
            # Locked but unreadable content must not crash the workflow.
            assert result is not None
        finally:
            _unlock_close(fd)


class TestVerboseLogging:
    def test_debug_messages_are_emitted_when_verbose(self, tmp_path):
        class Logger:
            def __init__(self):
                self.messages = []

            def debug(self, msg, *a, **k):
                self.messages.append(str(msg))

            def info(self, msg, *a, **k):
                self.messages.append(str(msg))

            def warning(self, msg, *a, **k):
                self.messages.append(str(msg))

        logger = Logger()
        m = manager(
            make_ns(tmp_path),
            pool={"cores": 4},
            instance_id="v",
            verbose=True,
            logger=logger,
        )
        m.acquire("j", {"cores": 1})
        m.release("j")
        assert any("Admitted" in msg for msg in logger.messages)
        assert any("Released" in msg for msg in logger.messages)

    def test_logger_failures_never_break_coordination(self, tmp_path):
        class ExplodingLogger:
            def debug(self, msg, *a, **k):
                raise RuntimeError("logger broken")

            info = warning = debug

        ns = make_ns(tmp_path)
        m = manager(
            ns,
            pool={"cores": 4},
            instance_id="v",
            verbose=True,
            logger=ExplodingLogger(),
        )
        handle = m.acquire("j", {"cores": 1})
        m.release("j")
        assert not handle.path.exists()

    def test_waiting_is_reported_once(self, tmp_path):
        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, msg, *a, **k):
                self.messages.append(str(msg))

            def debug(self, msg, *a, **k):
                pass

            def warning(self, msg, *a, **k):
                pass

        ns = make_ns(tmp_path)
        holder = manager(ns, pool={"cores": 1}, instance_id="holder")
        holder.acquire("h", {"cores": 1})
        logger = Logger()
        waiter = manager(
            ns,
            pool={"cores": 1},
            instance_id="waiter",
            logger=logger,
            poll_interval=0.01,
        )

        import threading

        def wait_then_release():
            time.sleep(0.3)
            holder.release("h")

        thread = threading.Thread(target=wait_then_release, daemon=True)
        thread.start()
        waiter.acquire("w", {"cores": 1})
        messages = [m for m in logger.messages if "Waiting for shared resources" in m]
        assert len(messages) == 1, f"expected one waiting notice, got {len(messages)}"
        waiter.release("w")


class TestMemoryResourceHandling:
    def test_mem_aliases_collapse_to_mem_mb(self):
        result = normalize_resources({"mem": 500, "mem_mb": 600}, threads=1)
        assert result["mem_mb"] == 600

    def test_negative_values_are_ignored(self):
        result = normalize_resources({"weird": -5}, threads=1)
        assert "weird" not in result

    def test_float_values_are_ignored(self):
        result = normalize_resources({"ratio": 1.5}, threads=2)
        assert result == {"cores": 2}


class TestCrossInstanceContention:
    def test_two_waiters_get_served_eventually(self, tmp_path):
        """Both waiters must make progress, not just the first one."""
        ns = make_ns(tmp_path)
        pool = {"cores": 1}
        holder = manager(ns, pool=pool, instance_id="holder")
        holder.acquire("h", {"cores": 1})

        served: list[str] = []
        import threading

        def run(name):
            m = manager(ns, pool=pool, instance_id=name, poll_interval=0.01)
            m.acquire(name, {"cores": 1})
            served.append(name)
            time.sleep(0.05)
            m.release(name)

        threads = [
            threading.Thread(target=run, args=(name,), daemon=True)
            for name in ("a", "b")
        ]
        for t in threads:
            t.start()
        time.sleep(0.2)
        holder.release("h")
        for t in threads:
            t.join(timeout=15)
        assert sorted(served) == ["a", "b"], (
            f"both waiters must be served, got {served}"
        )
