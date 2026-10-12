"""Tests for optional cgroup v2 enforcement.

The measured behaviour of this host shapes every assertion here:

* ``CPUQuota`` really throttles.
* ``MemoryMax`` alone does not enforce (swap absorbs the overflow), so
  ``MemorySwapMax=0`` must always be paired with it.
* ``systemd-run --user`` needs a D-Bus session and must degrade to a warning.
"""

import os
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local import cgroup
from snakemake_executor_plugin_shared_local.cgroup import (
    CgroupSupport,
    build_scope_command,
    probe,
    wrap_command,
)


class TestCommandConstruction:
    def test_includes_quota_memory_and_swap_guard(self):
        command = build_scope_command("echo hi", threads=2, pool_cores=8, mem_mib=500)
        joined = " ".join(command)
        assert "--property=CPUQuota=25%" in joined
        assert "--property=MemoryMax=500M" in joined
        # Without this the memory cap is not actually enforced.
        assert "--property=MemorySwapMax=0" in joined
        assert command[-3:] == ["/bin/sh", "-c", "echo hi"]

    def test_quota_is_rounded_up_so_small_jobs_are_not_starved(self):
        command = build_scope_command("x", threads=1, pool_cores=64, mem_mib=None)
        # ceil(1/64*100) = 2, never 1 or 0.
        assert "--property=CPUQuota=2%" in " ".join(command)

    def test_no_memory_limit_when_unknown(self):
        command = build_scope_command("x", threads=1, pool_cores=4, mem_mib=None)
        joined = " ".join(command)
        assert "MemoryMax" not in joined
        # The swap guard is only meaningful together with a memory cap.
        assert "MemorySwapMax" not in joined

    def test_never_uses_cpuset(self):
        """cpuset is not delegated on the target host, so it silently no-ops."""
        command = build_scope_command("x", threads=4, pool_cores=8, mem_mib=100)
        assert not any("cpuset" in part.lower() for part in command)
        assert not any("AllowedCPUs" in part for part in command)

    def test_uses_user_scope_and_collect(self):
        command = build_scope_command("x", threads=1, pool_cores=1, mem_mib=None)
        assert "--user" in command
        assert "--scope" in command
        # --collect prevents failed units from accumulating.
        assert "--collect" in command

    def test_zero_threads_does_not_emit_a_quota(self):
        command = build_scope_command("x", threads=0, pool_cores=8, mem_mib=None)
        assert not any("CPUQuota" in part for part in command)

    @pytest.mark.parametrize(
        "threads,pool,expected", [(4, 8, 50), (8, 8, 100), (2, 8, 25)]
    )
    def test_quota_math(self, threads, pool, expected):
        command = build_scope_command(
            "x", threads=threads, pool_cores=pool, mem_mib=None
        )
        assert f"--property=CPUQuota={expected}%" in command


class TestProbe:
    def test_probe_returns_structured_result(self):
        support = probe()
        assert isinstance(support, CgroupSupport)
        assert isinstance(support.available, bool)
        if not support.available:
            assert support.reason, "an unavailable probe must explain why"

    def test_describe_is_informative(self):
        assert "available" in CgroupSupport(True, "").describe()
        assert "unavailable" in CgroupSupport(False, "no dbus").describe()

    def test_missing_dbus_is_reported(self, monkeypatch):
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        support = probe()
        assert not support.available
        assert "DBUS_SESSION_BUS_ADDRESS" in support.reason

    def test_missing_binary_is_reported(self, monkeypatch):
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/fake")
        monkeypatch.setattr(cgroup.shutil, "which", lambda name: None)
        support = probe()
        assert not support.available
        assert "not found" in support.reason

    def test_missing_cgroup_v2_is_reported(self, monkeypatch):
        monkeypatch.setattr(cgroup, "_controllers_path", lambda: None)
        support = probe()
        assert not support.available
        assert "cgroup v2" in support.reason

    def test_probe_failure_is_reported(self, monkeypatch):
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/fake")
        monkeypatch.setattr(cgroup.shutil, "which", lambda name: "/usr/bin/systemd-run")

        class Result:
            returncode = 1
            stdout = ""
            stderr = "Failed to connect to bus: No medium found\n"

        monkeypatch.setattr(cgroup.subprocess, "run", lambda *a, **k: Result())
        support = probe()
        assert not support.available
        assert "Failed to connect to bus" in support.reason


class TestWrapCommand:
    def test_disabled_returns_plain_shell(self):
        result = wrap_command(
            "echo hi",
            enabled=False,
            support=CgroupSupport(False, "n/a"),
            threads=1,
            pool_cores=1,
            mem_mib=None,
        )
        assert result.command == ["/bin/sh", "-c", "echo hi"]
        assert result.enforced is False

    def test_unavailable_support_warns_but_does_not_fail(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = wrap_command(
                "echo hi",
                enabled=True,
                support=CgroupSupport(False, "no dbus"),
                threads=1,
                pool_cores=8,
                mem_mib=None,
            )
        assert result.command == ["/bin/sh", "-c", "echo hi"]
        assert result.enforced is False
        assert "no dbus" in result.warning
        assert "no dbus" in caplog.text

    def test_available_support_wraps_the_command(self):
        result = wrap_command(
            "echo hi",
            enabled=True,
            support=CgroupSupport(True, "", systemd_run="/usr/bin/systemd-run"),
            threads=2,
            pool_cores=8,
            mem_mib=100,
        )
        assert result.enforced is True
        assert result.command[0] == "/usr/bin/systemd-run"
        assert result.command[-1] == "echo hi"

    def test_available_support_but_no_binary_warns(self):
        result = wrap_command(
            "echo hi",
            enabled=True,
            support=CgroupSupport(True, "", systemd_run=None),
            threads=1,
            pool_cores=1,
            mem_mib=None,
        )
        assert result.enforced is False
        assert result.command == ["/bin/sh", "-c", "echo hi"]


class TestControllersPath:
    def test_returns_none_when_absent(self, monkeypatch):
        monkeypatch.setattr(
            "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )
        assert cgroup._controllers_path() is None


@pytest.mark.skipif(
    not Path("/sys/fs/cgroup/cgroup.controllers").exists(), reason="no cgroup v2"
)
class TestEnforcementIsReal:
    """End-to-end proof that the generated arguments actually constrain a job."""

    def test_cpu_quota_throttles_a_real_process(self):
        support = probe()
        if not support.available:
            pytest.skip(f"cgroup enforcement unavailable: {support.reason}")
        import subprocess
        import tempfile

        # A single-line program avoids quoting pitfalls when passed through the
        # shell layer that systemd-run also uses.
        program = "import time;t=time.time();x=0\nwhile time.time()-t<2: x+=1\nprint(x)"
        # Deliberately not a context manager: the subprocess needs the path to
        # still exist, so the handle is closed and the file removed explicitly.
        script = tempfile.NamedTemporaryFile(  # noqa: SIM115
            "w", suffix=".py", delete=False
        )
        script.write(program)
        script.close()
        try:
            plain = subprocess.run(
                ["python3", script.name], capture_output=True, text=True
            ).stdout.strip()
            wrapped = wrap_command(
                f"python3 {script.name}",
                enabled=True,
                support=support,
                threads=1,
                pool_cores=8,  # ~13% quota
                mem_mib=None,
            )
            limited = subprocess.run(
                wrapped.command, capture_output=True, text=True
            ).stdout.strip()
        finally:
            os.unlink(script.name)
        assert plain, "baseline measurement produced no output"
        assert limited, "throttled measurement produced no output"
        assert int(limited) < int(plain), "CPUQuota did not throttle the process"

    def test_memory_limit_kills_an_overallocating_process(self):
        support = probe()
        if not support.available:
            pytest.skip(f"cgroup enforcement unavailable: {support.reason}")
        import subprocess
        import tempfile

        program = "b=bytearray(600*1024*1024)\nprint('survived')"
        # Deliberately not a context manager: the subprocess needs the path to
        # still exist, so the handle is closed and the file removed explicitly.
        script = tempfile.NamedTemporaryFile(  # noqa: SIM115
            "w", suffix=".py", delete=False
        )
        script.write(program)
        script.close()
        try:
            result = subprocess.run(
                build_scope_command(
                    f"python3 {script.name}",
                    threads=1,
                    pool_cores=1,
                    mem_mib=200,
                    systemd_run=support.systemd_run,
                ),
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(script.name)
        # 137 == SIGKILL, which is what an enforced memory cap produces.
        assert result.returncode != 0
        assert "survived" not in result.stdout
