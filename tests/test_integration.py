"""End-to-end integration tests driving real Snakemake processes.

These are the tests that actually prove the plugin's promise. Everything else can
pass while the plugin still fails to coordinate, because the interesting
behaviour only appears when several Snakemake instances run against one shared
pool. Each test therefore launches real ``snakemake`` subprocesses and inspects
the resulting timing and filesystem state.

Requirements: the ``snakemake`` executable of the running interpreter, and a
POSIX system with ``fcntl``. Tests skip when Snakemake is unavailable.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def snakemake_executable() -> str:
    """Return the ``snakemake`` entry point for the active interpreter."""
    candidate = Path(sys.executable).parent / "snakemake"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("snakemake")
    if found:
        return found
    return ""


SMK = snakemake_executable()
pytestmark = pytest.mark.skipif(not SMK, reason="snakemake executable not available")


def write_workflow(directory: Path, njobs: int, sleep: float, events: Path) -> Path:
    """Write a workflow of ``njobs`` independent jobs sharing a timing log.

    The Snakefile is built by concatenation rather than ``str.format`` so its own
    ``{i}`` / ``{output}`` placeholders are not mistaken for Python format fields.
    """
    directory.mkdir(parents=True, exist_ok=True)
    snakefile = (
        "rule all:\n"
        f"    input: expand('o{{i}}.txt', i=range({njobs}))\n"
        "rule j:\n"
        "    output: 'o{i}.txt'\n"
        "    threads: 1\n"
        "    resources: mem_mb=100\n"
        "    shell: 'echo START $(date +%s.%N) >> "
        + str(events)
        + "; sleep "
        + str(sleep)
        + "; echo done > {output}; echo END $(date +%s.%N) >> "
        + str(events)
        + "'\n"
    )
    (directory / "Snakefile").write_text(snakefile)
    return directory


def run_instance(
    workdir: Path,
    leases: Path,
    pool_cores: int | None,
    options: tuple[str, ...] = (),
    *,
    cores: int = 4,
    log: Path | None = None,
    start: bool = False,
    timeout: float = 180,
) -> subprocess.Popen | subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    # Let coverage measure the subprocess too, otherwise the executor wiring this
    # test is validating looks untested.
    if os.environ.get("COVERAGE_PROCESS_START"):
        sitecustomize_dir = Path(__file__).resolve().parent / "_coverage_boot"
        sitecustomize_dir.mkdir(exist_ok=True)
        (sitecustomize_dir / "sitecustomize.py").write_text(
            "import coverage; coverage.process_startup()\n"
        )
        env["PYTHONPATH"] = str(sitecustomize_dir) + os.pathsep + env["PYTHONPATH"]
    command = [
        SMK,
        "-s",
        "Snakefile",
        "--executor",
        "shared-local",
        "-c",
        str(cores),
        "--shared-local-dir",
        str(leases),
        "--shared-local-numa-bind",
        "off",
    ]
    if pool_cores is not None:
        command += ["--shared-local-pool-cores", str(pool_cores)]
    command += list(options)
    kwargs = dict(
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if log is not None:
        # Held open for the lifetime of the subprocess, then closed by the caller;
        # a context manager would close it before the child finishes writing.
        handle = open(log, "w")  # noqa: SIM115
        kwargs["stdout"] = handle
        kwargs["stderr"] = subprocess.STDOUT
        proc = subprocess.Popen(command, **kwargs)
        proc._sl_log_handle = handle  # type: ignore[attr-defined]
        return proc
    if start:
        return subprocess.Popen(command, **kwargs)
    return subprocess.run(command, **kwargs, timeout=timeout)


def concurrency_from_events(events: Path, sleep: float) -> int:
    """Compute peak concurrency from START timestamps and the job duration.

    Each job started at ``t`` occupies the machine until ``t + sleep``, so peak
    concurrency is the maximum number of overlapping intervals. Sweeping start and
    end points is exact and independent of log line ordering across processes.
    """
    if not events.exists():
        return 0
    starts = [
        float(m.group(1))
        for m in re.finditer(r"^START ([\d.]+)", events.read_text(), flags=re.M)
    ]
    if not starts:
        return 0
    deltas: list[tuple[float, int]] = []
    for start in starts:
        deltas.append((start, 1))
        deltas.append((start + sleep, -1))
    # Process ends before starts at identical timestamps to avoid counting two
    # jobs as concurrent merely because one ended exactly when the other began.
    deltas.sort(key=lambda item: (item[0], item[1]))
    running = 0
    peak = 0
    for _, change in deltas:
        running += change
        peak = max(peak, running)
    return peak


class TestSingleInstance:
    def test_simple_workflow_runs(self, tmp_path):
        events = tmp_path / "events.log"
        work = write_workflow(tmp_path / "wf", 3, 0.2, events)
        leases = tmp_path / "leases"
        result = run_instance(work, leases, pool_cores=8, cores=4)
        assert result.returncode == 0, result.stdout
        assert len(list(work.glob("o*.txt"))) == 3
        # Coordination state must not leak after a clean run.
        leftovers = [p.name for p in leases.glob("*.lease")]
        assert leftovers == [], f"leases leaked: {leftovers}"

    def test_reports_scope_and_pool(self, tmp_path):
        events = tmp_path / "events.log"
        work = write_workflow(tmp_path / "wf", 1, 0.1, events)
        result = run_instance(work, tmp_path / "leases", pool_cores=6, cores=2)
        assert "pool cores=6" in result.stdout
        assert "scope 'user'" in result.stdout

    def test_group_job_does_not_self_deadlock(self, tmp_path):
        """A pipe group must reserve all members at once, not one by one."""
        work = tmp_path / "wf"
        work.mkdir(parents=True)
        (work / "Snakefile").write_text(
            "rule all:\n    input: 'out.txt'\n"
            "rule producer:\n    output: pipe('mid.txt')\n    threads: 2\n"
            "    resources: mem_mb=100\n    shell: 'echo hi > {output}; sleep 1'\n"
            "rule consumer:\n    input: pipe('mid.txt')\n    output: 'out.txt'\n"
            "    threads: 2\n    resources: mem_mb=100\n"
            "    shell: 'cat {input} > {output}'\n"
        )
        result = run_instance(work, tmp_path / "leases", pool_cores=8, cores=4)
        assert result.returncode == 0, result.stdout
        assert (work / "out.txt").read_text().strip() == "hi"

    def test_oversized_job_fails_fast_instead_of_hanging(self, tmp_path):
        """Regression test for the deadlock the prototype exhibited."""
        work = tmp_path / "wf"
        work.mkdir(parents=True)
        (work / "Snakefile").write_text(
            "rule big:\n    output: 'sb.txt'\n    threads: 16\n"
            "    resources: mem_mb=100\n    shell: 'echo x > {output}'\n"
        )
        started = time.monotonic()
        result = run_instance(
            work, tmp_path / "leases", pool_cores=4, cores=32, timeout=60
        )
        elapsed = time.monotonic() - started
        assert result.returncode != 0
        assert "requires cores=16" in result.stdout
        assert "shared-local-pool-cores" in result.stdout
        # Must fail promptly; a hang previously lasted forever.
        assert elapsed < 45, f"took {elapsed:.1f}s, expected a fast failure"

    def test_cgroup_flag_is_accepted(self, tmp_path):
        events = tmp_path / "events.log"
        work = write_workflow(tmp_path / "wf", 1, 0.1, events)
        result = run_instance(
            work,
            tmp_path / "leases",
            pool_cores=4,
            options=("--shared-local-cgroup", "on"),
            cores=2,
        )
        # Either enforced or explicitly degraded; never a hard failure.
        assert result.returncode == 0, result.stdout
        assert "cgroup on" in result.stdout or "cgroup" in result.stdout


class TestMultipleInstances:
    """The core promise: several instances must share one pool."""

    def test_two_instances_respect_the_shared_pool(self, tmp_path):
        sleep = 2.0
        events = tmp_path / "events.log"
        pool = 4
        leases = tmp_path / "leases"
        wf1 = write_workflow(tmp_path / "wf1", 4, sleep, events)
        wf2 = write_workflow(tmp_path / "wf2", 4, sleep, events)

        p1 = run_instance(wf1, leases, pool, cores=4, start=True)
        p2 = run_instance(wf2, leases, pool, cores=4, start=True)
        try:
            out1 = p1.communicate(timeout=180)[0]
            out2 = p2.communicate(timeout=180)[0]
        finally:
            for proc in (p1, p2):
                if proc.poll() is None:  # pragma: no cover - cleanup
                    proc.kill()
        assert p1.returncode == 0, out1
        assert p2.returncode == 0, out2

        built = len(list(wf1.glob("o*.txt"))) + len(list(wf2.glob("o*.txt")))
        assert built == 8, f"only {built}/8 outputs were produced"

        peak = concurrency_from_events(events, sleep)
        assert peak <= pool, f"pool of {pool} was oversubscribed: peak {peak}"

        leftovers = sorted(p.name for p in leases.glob("*.lease"))
        assert leftovers == [], f"leases leaked after both runs: {leftovers}"

    def test_instances_run_both_in_parallel(self, tmp_path):
        """The cap must throttle, not serialise: both instances still make progress."""
        sleep = 1.5
        events = tmp_path / "events.log"
        leases = tmp_path / "leases"
        pool = 4
        wf1 = write_workflow(tmp_path / "wf1", 3, sleep, events)
        wf2 = write_workflow(tmp_path / "wf2", 3, sleep, events)
        p1 = run_instance(wf1, leases, pool, cores=4, start=True)
        p2 = run_instance(wf2, leases, pool, cores=4, start=True)
        out1 = p1.communicate(timeout=180)[0]
        out2 = p2.communicate(timeout=180)[0]
        assert p1.returncode == 0, out1
        assert p2.returncode == 0, out2
        # Total serial time would be 6 jobs * 1.5s = 9s. Parallel within a pool of
        # 4 must finish comfortably sooner.
        text = events.read_text()
        stamps = [
            float(m.group(1)) for m in re.finditer(r"^START ([\d.]+)", text, flags=re.M)
        ]
        assert stamps, "no jobs were recorded"
        assert max(stamps) - min(stamps) < 9.0


class TestCrashRecovery:
    def test_killed_instance_leases_are_reclaimed(self, tmp_path):
        """A SIGKILLed instance must not block the machine forever."""
        events = tmp_path / "events.log"
        leases = tmp_path / "leases"
        work = write_workflow(tmp_path / "wf", 2, 30.0, events)

        proc = run_instance(work, leases, pool_cores=2, cores=2, start=True)
        # Wait until the instance actually holds a lease.
        deadline = time.time() + 30
        while time.time() < deadline:
            if list(leases.glob("*.lease")):
                break
            time.sleep(0.2)
        assert list(leases.glob("*.lease")), "instance never acquired a lease"

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
        # Kill any orphaned job shells from the dead instance.
        subprocess.run(["pkill", "-9", "-f", "sleep 30.0"], check=False)
        time.sleep(1)

        # The stale files remain, but their locks died with the process.
        assert list(leases.glob("*.lease")), "expected stale lease files to remain"

        # A fresh instance sizing itself for the whole pool must still be admitted.
        work2 = tmp_path / "wf2"
        work2.mkdir(parents=True, exist_ok=True)
        (work2 / "Snakefile").write_text(
            "rule all:\n    input: 'fresh.txt'\n"
            "rule fresh:\n    output: 'fresh.txt'\n    threads: 2\n"
            "    resources: mem_mb=100\n    shell: 'echo ok > {output}'\n"
        )
        result = run_instance(work2, leases, pool_cores=2, cores=2, timeout=60)
        assert result.returncode == 0, result.stdout
        assert (work2 / "fresh.txt").exists()
        # The reclamation sweep should have removed the abandoned lease.
        remaining = sorted(p.name for p in leases.glob("*.lease"))
        assert remaining == [], f"stale leases not reclaimed: {remaining}"


class TestPoolNegotiation:
    def test_conflicting_pools_are_refused(self, tmp_path):
        """A pool mismatch would silently void the global cap, so it must fail."""
        leases = tmp_path / "leases"
        events = tmp_path / "events.log"
        wf1 = write_workflow(tmp_path / "wf1", 1, 0.1, events)
        first = run_instance(wf1, leases, pool_cores=8, cores=2)
        assert first.returncode == 0, first.stdout

        wf2 = write_workflow(tmp_path / "wf2", 1, 0.1, events)
        second = run_instance(wf2, leases, pool_cores=4, cores=2)
        assert second.returncode != 0
        assert "mismatch" in second.stdout

    def test_conflicting_pools_can_be_downgraded(self, tmp_path):
        leases = tmp_path / "leases"
        events = tmp_path / "events.log"
        wf1 = write_workflow(tmp_path / "wf1", 1, 0.1, events)
        assert run_instance(wf1, leases, pool_cores=8, cores=2).returncode == 0
        wf2 = write_workflow(tmp_path / "wf2", 1, 0.1, events)
        result = run_instance(
            wf2,
            leases,
            pool_cores=4,
            options=("--shared-local-pool-conflict", "warn"),
            cores=2,
        )
        assert result.returncode == 0, result.stdout
        assert "mismatch" in result.stdout


class TestConfigFile:
    def test_config_file_supplies_the_pool(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"pool_cores": 3, "numa_bind": "off"}))
        events = tmp_path / "events.log"
        work = write_workflow(tmp_path / "wf", 1, 0.1, events)
        result = run_instance(
            work,
            tmp_path / "leases",
            pool_cores=8,
            options=("--shared-local-config", str(config)),
            cores=2,
        )
        assert result.returncode == 0, result.stdout
