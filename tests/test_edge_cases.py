"""Tests for defensive branches and error paths across the modules.

These cover the "unhappy path" behaviour that makes the plugin safe to deploy:
misconfiguration must produce a clear error, and platform limitations must degrade
gracefully rather than crash a workflow.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local import cgroup, namespace, numa, pool
from snakemake_executor_plugin_shared_local.config import (
    ConfigError,
    _coerce_float,
    _coerce_int,
    load_config_file,
    parse_size_to_mib,
)
from snakemake_executor_plugin_shared_local.namespace import (
    Namespace,
    _ensure_dir,
    _verify_dir_mode,
    resolve_namespace,
)


class TestCpuDetectionFallbacks:
    def test_uses_process_cpu_count_when_available(self, monkeypatch):
        monkeypatch.setattr(os, "process_cpu_count", lambda: 7, raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2, 3})
        assert pool.detect_cpu_count() == 7

    def test_falls_back_to_affinity_without_process_cpu_count(self, monkeypatch):
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 2, 3, 4})
        assert pool.detect_cpu_count() == 4

    def test_falls_back_to_cpu_count_when_affinity_unsupported(self, monkeypatch):
        monkeypatch.delattr(os, "process_cpu_count", raising=False)

        def unavailable(pid):
            raise AttributeError("no affinity here")

        monkeypatch.setattr(os, "sched_getaffinity", unavailable)
        monkeypatch.setattr(os, "cpu_count", lambda: 12)
        assert pool.detect_cpu_count() == 12

    def test_raises_when_nothing_is_detectable(self, monkeypatch):
        monkeypatch.delattr(os, "process_cpu_count", raising=False)

        def unavailable(pid):
            raise AttributeError("no affinity")

        monkeypatch.setattr(os, "sched_getaffinity", unavailable)
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        with pytest.raises(ConfigError, match="Cannot detect"):
            pool.detect_cpu_count()

    def test_process_cpu_count_returning_none_falls_through(self, monkeypatch):
        monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
        assert pool.detect_cpu_count() == 2


class TestSizeParsingEdgeCases:
    def test_underscores_and_spaces_are_tolerated(self):
        assert parse_size_to_mib("1_000", key="m") == 1000
        assert parse_size_to_mib(" 2 G ", key="m") == 2048

    def test_bool_is_rejected(self):
        with pytest.raises(ConfigError, match="expected a size"):
            parse_size_to_mib(True, key="m")

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ConfigError, match="expected e.g."):
            parse_size_to_mib("12 parsecs", key="m")

    def test_zero_amount_with_unit_is_rejected(self):
        with pytest.raises(ConfigError, match="must be positive"):
            parse_size_to_mib("0G", key="m")

    def test_fractional_suffix_value_with_unit(self):
        assert parse_size_to_mib("0.5G", key="m") == 512

    def test_empty_string_means_unset(self):
        assert parse_size_to_mib("   ", key="m") is None

    def test_kilobyte_suffix(self):
        assert parse_size_to_mib("2048K", key="m") == 2


class TestNumericCoercion:
    def test_coerce_int_rejects_bool(self):
        with pytest.raises(ConfigError, match="expected an integer"):
            _coerce_int(True, key="n")

    def test_coerce_int_rejects_garbage(self):
        with pytest.raises(ConfigError, match="expected an integer"):
            _coerce_int("many", key="n")

    def test_coerce_int_enforces_minimum(self):
        with pytest.raises(ConfigError, match="must be >="):
            _coerce_int(-1, key="n", minimum=0)

    def test_coerce_int_passes_through_none(self):
        assert _coerce_int(None, key="n") is None

    def test_coerce_float_rejects_garbage(self):
        with pytest.raises(ConfigError, match="Invalid number"):
            _coerce_float("fast", key="w")

    def test_coerce_float_rejects_nonpositive(self):
        with pytest.raises(ConfigError, match="must be positive"):
            _coerce_float(0, key="w")

    def test_coerce_float_passes_through_none(self):
        assert _coerce_float(None, key="w") is None


class TestLoadConfigFile:
    def test_none_path_yields_empty(self):
        assert load_config_file(None) == ({}, [])

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Cannot read"):
            load_config_file(tmp_path)

    def test_unreadable_file_raises(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{}")
        os.chmod(config, 0o000)
        if os.getuid() == 0:  # pragma: no cover - root ignores permissions
            pytest.skip("root can read anything")
        try:
            with pytest.raises(ConfigError, match="Cannot read"):
                load_config_file(config)
        finally:
            os.chmod(config, 0o644)


class TestNamespaceErrorPaths:
    def test_non_directory_explicit_path(self, tmp_path):
        target = tmp_path / "file"
        target.write_text("x")
        with pytest.raises(ConfigError, match="not a directory"):
            resolve_namespace("user", explicit_dir=str(target))

    def test_unwritable_explicit_directory(self, tmp_path):
        target = tmp_path / "ro"
        target.mkdir()
        os.chmod(target, 0o500)
        if os.getuid() == 0:  # pragma: no cover - root ignores permissions
            pytest.skip("root can write anything")
        try:
            with pytest.raises(ConfigError, match="not writable"):
                resolve_namespace("user", explicit_dir=str(target))
        finally:
            os.chmod(target, 0o700)

    def test_unowned_directory_missing_shared_bits_is_rejected(
        self, tmp_path, monkeypatch
    ):
        """A group/system scope cannot work on a private directory."""
        target = tmp_path / "private"
        target.mkdir()
        os.chmod(target, 0o700)
        # Pretend someone else owns it, so we may not repair the mode.
        monkeypatch.setattr(namespace, "_owned_by_self", lambda path: False)
        with pytest.raises(ConfigError, match="sudo chmod"):
            _verify_dir_mode(target, 0o770)

    def test_chmod_failure_on_our_own_directory_is_reported(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "ours"
        target.mkdir()
        os.chmod(target, 0o700)

        def failing_chmod(path, mode):
            raise PermissionError("nope")

        monkeypatch.setattr(os, "chmod", failing_chmod)
        with pytest.raises(ConfigError, match="sudo chmod"):
            _ensure_dir(target, 0o770, explicit=False)

    def test_existing_path_that_is_not_a_directory(self, tmp_path):
        target = tmp_path / "afile"
        target.write_text("x")
        with pytest.raises(ConfigError, match="exists but is not a directory"):
            _ensure_dir(target, 0o700, explicit=False)

    def test_stat_failure_during_mode_verification(self, tmp_path, monkeypatch):
        target = tmp_path / "gone"
        target.mkdir()

        real_stat = os.stat

        def failing_stat(path, *args, **kwargs):
            if str(path) == str(target):
                raise OSError("vanished")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", failing_stat)
        with pytest.raises(ConfigError, match="Cannot stat"):
            _verify_dir_mode(target, 0o700)

    def test_unreadable_foreign_directory_reports_owner(self, tmp_path, monkeypatch):
        target = tmp_path / "foreign"
        target.mkdir()

        class FakeStat:
            st_mode = stat.S_IFDIR | 0o700
            st_uid = 4242

        monkeypatch.setattr(os, "stat", lambda path, *a, **k: FakeStat())
        monkeypatch.setattr(namespace, "_owned_by_self", lambda path: False)
        with pytest.raises(ConfigError, match="4242"):
            _verify_dir_mode(target, 0o777)


class TestNamespaceFallbacks:
    def test_xdg_runtime_dir_is_used_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        monkeypatch.setattr(namespace, "CANDIDATE_BASES", (None,))
        ns = resolve_namespace("user")
        assert str(ns.path).startswith(str(tmp_path))

    def test_no_candidates_at_all(self, monkeypatch):
        monkeypatch.setattr(namespace, "CANDIDATE_BASES", ())
        with pytest.raises(ConfigError, match="Could not establish"):
            resolve_namespace("user")

    def test_missing_xdg_runtime_dir_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(namespace, "CANDIDATE_BASES", (None, str(tmp_path)))
        ns = resolve_namespace("user")
        assert str(ns.path).startswith(str(tmp_path))

    def test_describe_identity_handles_missing_passwd_entry(self, monkeypatch):
        def missing(uid):
            raise KeyError(uid)

        monkeypatch.setattr(namespace.pwd, "getpwuid", missing)
        info = namespace.describe_identity()
        assert info["user"] == str(os.getuid())


class TestNumaEdgeCases:
    def test_used_per_node_absent_key_defaults_to_zero(self):
        from snakemake_executor_plugin_shared_local.numa import (
            NumaNode,
            NumaTopology,
            select_node,
        )

        topo = NumaTopology(
            nodes=[
                NumaNode(index=0, cpus=[0, 1], mem_total_mib=1000, mem_free_mib=500),
                NumaNode(index=1, cpus=[2, 3], mem_total_mib=1000, mem_free_mib=400),
            ]
        )
        # A zero-usage entry for an unrelated node must not break selection.
        assert (
            select_node(topo, cores=1, mem_mib=None, used_per_node={9: 100}) is not None
        )

    def test_read_cpu_list_skips_empty_parts(self, tmp_path):
        path = tmp_path / "cpulist"
        path.write_text("0-1,,3")
        assert numa._read_cpu_list(path) == [0, 1, 3]

    def test_read_meminfo_returns_zero_when_key_absent(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text("Node 0 SomethingElse: 100 kB\n")
        assert numa._read_meminfo_kib(path, "MemTotal") == 0

    def test_read_meminfo_handles_unreadable_file(self, tmp_path):
        assert numa._read_meminfo_kib(tmp_path / "nope", "MemTotal") == 0

    def test_bind_cpus_reports_failure_when_affinity_unsupported(self, monkeypatch):
        def unavailable(*args, **kwargs):
            raise OSError("nope")

        monkeypatch.setattr(os, "sched_setaffinity", unavailable)
        assert numa._NumaBinder().bind_cpus([0]) is False

    def test_bind_memory_handles_missing_library(self, monkeypatch):
        monkeypatch.setattr(ctypes_util(), "find_library", lambda name: None)
        binder = numa._NumaBinder()
        monkeypatch.setattr(binder, "_load", lambda: None)
        assert binder.bind_memory([0]) is False

    def test_binding_reports_when_both_syscalls_fail(self, monkeypatch):
        from snakemake_executor_plugin_shared_local.numa import (
            NumaNode,
            bind_current_thread,
        )

        monkeypatch.setattr(numa._BINDER, "bind_memory", lambda nodes: False)
        monkeypatch.setattr(numa._BINDER, "bind_cpus", lambda cpus: False)
        node = NumaNode(index=0, cpus=[0, 1], mem_total_mib=100, mem_free_mib=50)
        result = bind_current_thread(node, cores=1)
        assert result.applied is False
        assert "unavailable" in result.reason

    def test_binding_warns_when_only_cpu_binding_fails(self, monkeypatch):
        from snakemake_executor_plugin_shared_local.numa import (
            NumaNode,
            bind_current_thread,
        )

        monkeypatch.setattr(numa._BINDER, "bind_memory", lambda nodes: True)
        monkeypatch.setattr(numa._BINDER, "bind_cpus", lambda cpus: False)
        node = NumaNode(index=0, cpus=[0], mem_total_mib=100, mem_free_mib=50)
        result = bind_current_thread(node, cores=1)
        # Applied via memory binding alone.
        assert result.applied is True

    def test_reset_is_a_noop_when_nothing_was_captured(self, monkeypatch):
        monkeypatch.setattr(numa, "_ORIGINAL_AFFINITY", None)
        numa.reset_current_thread()  # must not raise


def ctypes_util():
    import ctypes.util

    return ctypes.util


class TestCgroupEdgeCases:
    def test_probe_on_non_posix(self, monkeypatch):
        monkeypatch.setattr(cgroup.os, "name", "nt")
        support = cgroup.probe()
        assert not support.available
        assert "POSIX" in support.reason

    def test_probe_handles_subprocess_error(self, monkeypatch):
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/x")
        monkeypatch.setattr(cgroup.shutil, "which", lambda name: "/usr/bin/systemd-run")

        def explode(*args, **kwargs):
            raise OSError("cannot execute")

        monkeypatch.setattr(cgroup.subprocess, "run", explode)
        support = cgroup.probe()
        assert not support.available
        assert "probe failed" in support.reason

    def test_probe_reports_empty_error_output(self, monkeypatch):
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/x")
        monkeypatch.setattr(cgroup.shutil, "which", lambda name: "/usr/bin/systemd-run")

        class Result:
            returncode = 1
            stdout = ""
            stderr = ""

        monkeypatch.setattr(cgroup.subprocess, "run", lambda *a, **k: Result())
        support = cgroup.probe()
        assert not support.available
        assert "exit status" in support.reason

    def test_missing_xdg_runtime_directory_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/x")
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "absent"))
        support = cgroup.probe()
        assert not support.available
        assert "XDG_RUNTIME_DIR" in support.reason

    def test_controllers_path_is_found_on_cgroup_v2(self):
        if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
            pytest.skip("no cgroup v2")
        assert cgroup._controllers_path() is not None


class TestPoolSizingEdgeCases:
    def test_pool_mem_uses_explicit_value_only(self):
        p = pool.compute_pool(pool_cores=4, pool_mem_mb=1234)
        assert p.mem_mb == 1234

    def test_reserve_rounds_down(self):
        # 10 % of 15 is 1.5; the reserve must not round up and steal a core.
        assert pool.apply_reserve(15, 0.10) == 14

    def test_detect_mem_total_reports_malformed_meminfo(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text("MemTotal: notanumber kB\n")
        with pytest.raises(ConfigError, match="MemTotal"):
            pool.detect_mem_total_mib(path)

    def test_detect_mem_total_reports_short_line(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text("MemTotal:\n")
        with pytest.raises(ConfigError, match="MemTotal"):
            pool.detect_mem_total_mib(path)


class TestNamespaceDataclass:
    def test_lease_path_helper(self, tmp_path):
        ns = Namespace(
            mode="user",
            path=tmp_path,
            dir_mode=0o700,
            sticky=False,
            uid=0,
            gid=0,
        )
        assert ns.lease_path("x.lease") == tmp_path / "x.lease"


def test_python_version_supported():
    assert sys.version_info >= (3, 11)
