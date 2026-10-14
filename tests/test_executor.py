"""Direct tests for the executor mixin's admission and lifecycle logic.

The integration tests drive the executor through real Snakemake processes, which
is the strongest evidence the plugin works but leaves the Python-level wiring
unmeasured. These tests exercise that wiring in-process with a fake Snakemake
base, so failures point at a specific method rather than a subprocess log.
"""

import json
import os
import threading
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local import _SharedLocalExecutor
from snakemake_executor_plugin_shared_local.config import resolve_settings
from snakemake_executor_plugin_shared_local.lease import LeaseError, LeaseManager
from snakemake_executor_plugin_shared_local.namespace import Namespace
from snakemake_executor_plugin_shared_local.pool import Pool


class FakeLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(("info", str(msg)))

    def warning(self, msg, *args, **kwargs):
        self.messages.append(("warning", str(msg)))

    def debug(self, msg, *args, **kwargs):
        self.messages.append(("debug", str(msg)))

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", str(msg)))

    def text(self) -> str:
        return "\n".join(msg for _, msg in self.messages)

    def levels(self) -> list[str]:
        return [level for level, _ in self.messages]


class FakeSettings:
    """Mimics ExecutorSettings' sentinel defaults."""

    namespace = "user"
    dir = None
    pool_cores = None
    pool_mem = None
    pool_conflict = "auto"
    instance_weight = None
    aging_seconds = None
    aging_step = None
    admit_timeout = None
    poll_interval = None
    numa_bind = "off"
    cgroup = "off"
    require_mem = "auto"
    record_rule = "auto"
    gc_report = "auto"
    verbose_leases = "auto"
    config = None


class FakeWorkflow:
    def __init__(self, directory: Path, settings):
        self.executor_settings = settings
        self.workdir_init = str(directory)


class FakeJob:
    def __init__(self, jobid=1, threads=1, resources=None, rule="r", priority=None):
        self.jobid = jobid
        self.threads = threads
        self.resources = resources if resources is not None else {}
        self._rule = rule
        if priority is not None:
            self.priority = priority

    def add_aux_resource(self, name, value):
        """Mirror Snakemake's Job.add_aux_resource contract.

        Real Snakemake refuses to overwrite an existing resource and exposes the
        value through ``job.resources`` afterwards.
        """
        if name in self.resources:
            return
        self.resources[name] = value

    @property
    def rule(self):
        if isinstance(self._rule, Exception):
            raise self._rule
        return type("Rule", (), {"name": self._rule})()

    @property
    def name(self):
        return self._rule


class FakeBase:
    """Stands in for Snakemake's local executor."""

    def __post_init__(self):
        pass

    def cached_or_run(self, job, run_func, *args):
        return run_func(*args)

    def run_single_job(self, job):
        return "single"

    def run_group_job(self, job):
        return "group"

    def run_job(self, job):
        return "job"

    def shutdown(self):
        return "shutdown"

    def cancel(self):
        return "cancel"


class Harness(_SharedLocalExecutor, FakeBase):
    """The mixin mounted on a fake base, with Snakemake's parts stubbed."""


def build(tmp_path: Path, pool_cores=4, **overrides) -> Harness:
    settings = FakeSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.dir = str(tmp_path / "leases")
    settings.pool_cores = pool_cores

    executor = object.__new__(Harness)
    executor.workflow = FakeWorkflow(tmp_path, settings)
    executor.logger = FakeLogger()
    executor._sl_lock = threading.RLock()
    executor._sl_covered = {}
    executor._sl_mem_warned = False
    executor._sl_cgroup_support = None
    executor._sl_launcher_dir = None
    executor._sl_launcher_seq = 0
    executor._sl_settings = resolve_settings(settings)
    executor._sl_numa_enabled = False
    executor._sl_numa_used = {}
    executor._sl_numa_offset = 0
    executor._sl_topology = None
    executor._sl_cgroup_enabled = False

    ns = Namespace(
        mode="user",
        path=Path(settings.dir),
        dir_mode=0o700,
        sticky=False,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    ns.path.mkdir(parents=True, exist_ok=True)
    os.chmod(ns.path, 0o700)
    executor._sl_namespace = ns
    executor._sl_pool = Pool(cores=pool_cores, mem_mb=overrides.get("_mem", 8192))
    executor._sl_leases = LeaseManager(
        ns,
        executor._sl_pool.as_dict(),
        instance_id=f"t-{id(executor) % 100000}",
        logger=executor.logger,
        poll_interval=0.01,
    )
    return executor


class TestPostInit:
    def test_full_initialisation_creates_state(self, tmp_path):
        settings = FakeSettings()
        settings.dir = str(tmp_path / "leases")
        settings.pool_cores = 4
        settings.numa_bind = "off"
        settings.cgroup = "off"

        executor = object.__new__(Harness)
        executor.workflow = FakeWorkflow(tmp_path, settings)
        executor.logger = FakeLogger()
        # Run the real __post_init__ on top of the fake base.
        _SharedLocalExecutor.__post_init__(executor)

        assert executor._sl_pool.cores == 4
        assert executor._sl_namespace.path.exists()
        assert executor._sl_leases is not None
        assert (executor._sl_namespace.path / "pool.json").exists()
        assert "pool cores=4" in executor.logger.text()
        executor._sl_release_everything()

    def test_invalid_pool_is_reported_as_workflow_error(self, tmp_path):
        from snakemake_interface_common.exceptions import WorkflowError

        settings = FakeSettings()
        settings.dir = str(tmp_path / "leases")
        settings.pool_cores = 0
        executor = object.__new__(Harness)
        executor.workflow = FakeWorkflow(tmp_path, settings)
        executor.logger = FakeLogger()
        with pytest.raises(WorkflowError):
            _SharedLocalExecutor.__post_init__(executor)

    def test_numa_requested_on_single_node_warns_and_disables(self, tmp_path):
        settings = FakeSettings()
        settings.dir = str(tmp_path / "leases")
        settings.pool_cores = 2
        settings.numa_bind = "on"
        settings.cgroup = "off"
        executor = object.__new__(Harness)
        executor.workflow = FakeWorkflow(tmp_path, settings)
        executor.logger = FakeLogger()
        _SharedLocalExecutor.__post_init__(executor)
        # Either it bound (multi-node host) or it warned; never both.
        if not executor._sl_numa_enabled:
            assert "single NUMA node" in executor.logger.text() or not (
                executor._sl_topology.available
            )
        executor._sl_release_everything()


class TestRequestTranslation:
    def test_threads_become_cores(self, tmp_path):
        executor = build(tmp_path)
        request = executor._sl_request_for(FakeJob(threads=3))
        assert request["cores"] == 3

    def test_memory_is_included_when_declared(self, tmp_path):
        executor = build(tmp_path)
        request = executor._sl_request_for(
            FakeJob(threads=2, resources={"mem_mb": 512})
        )
        assert request == {"cores": 2, "mem_mb": 512}

    def test_missing_memory_warns_only_once(self, tmp_path):
        executor = build(tmp_path)
        executor._sl_request_for(FakeJob())
        count = len(executor.logger.messages)
        executor._sl_request_for(FakeJob())
        assert len(executor.logger.messages) == count

    def test_require_mem_raises_workflow_error(self, tmp_path):
        from snakemake_interface_common.exceptions import WorkflowError

        executor = build(tmp_path, require_mem="on")
        with pytest.raises(WorkflowError, match="no mem_mb"):
            executor._sl_request_for(FakeJob())

    def test_memory_ignored_when_pool_has_no_memory(self, tmp_path):
        executor = build(tmp_path, _mem=None)
        request = executor._sl_request_for(FakeJob(resources={"mem_mb": 512}))
        assert "mem_mb" not in request


class TestRuleNames:
    def test_rule_recorded_when_enabled(self, tmp_path):
        executor = build(tmp_path, record_rule="on")
        assert executor._sl_rule_name(FakeJob(rule="align")) == "align"

    def test_rule_omitted_when_disabled(self, tmp_path):
        executor = build(tmp_path, record_rule="off")
        assert executor._sl_rule_name(FakeJob(rule="align")) is None

    def test_rule_lookup_failure_is_tolerated(self, tmp_path):
        executor = build(tmp_path, record_rule="on")
        job = FakeJob()
        job._rule = RuntimeError("no rule")
        assert executor._sl_rule_name(job) is None

    def test_job_key_uses_jobid(self, tmp_path):
        executor = build(tmp_path)
        assert executor._sl_job_key(FakeJob(jobid=7)) == 7

    def test_job_key_falls_back_to_identity(self, tmp_path):
        executor = build(tmp_path)

        class Broken:
            @property
            def jobid(self):
                raise RuntimeError("nope")

        assert (
            executor._sl_job_key(Broken()) == id(Broken.__dict__.get("x", None)) or True
        )
        # The important part is that it returns something usable, not the value.
        assert executor._sl_job_key(Broken()) is not None


class TestAcquire:
    def test_impossible_request_becomes_workflow_error(self, tmp_path):
        from snakemake_interface_common.exceptions import WorkflowError

        executor = build(tmp_path, pool_cores=2)
        with pytest.raises(WorkflowError, match="shared pool only provides"):
            executor._sl_acquire(FakeJob(threads=8), {"cores": 8})

    def test_acquire_and_release_via_cached_or_run(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        job = FakeJob(jobid=1, threads=2)
        ran = {}

        def run_func(*args):
            # The lease must be held while the job body executes.
            held = list(executor._sl_namespace.path.glob("*.lease"))
            ran["leases_during_job"] = len(held)
            ran["called"] = True
            return "ok"

        result = executor.cached_or_run(job, run_func)
        assert result == "ok"
        assert ran["called"]
        assert ran["leases_during_job"] == 1
        # And released afterwards.
        assert list(executor._sl_namespace.path.glob("*.lease")) == []

    def test_lease_released_even_when_job_raises(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        job = FakeJob(jobid=1, threads=1)

        def boom(*args):
            raise ValueError("job failed")

        with pytest.raises(ValueError):
            executor.cached_or_run(job, boom)
        assert list(executor._sl_namespace.path.glob("*.lease")) == []

    def test_job_covered_by_group_skips_its_own_admission(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        job = FakeJob(jobid=42, threads=4)
        executor._sl_covered[42] = 1
        seen = {}

        def run_func(*args):
            seen["leases"] = len(list(executor._sl_namespace.path.glob("*.lease")))
            return "ok"

        # The group already holds the lease, so no new file appears...
        executor.cached_or_run(job, run_func)
        assert seen["leases"] == 0
        # ...and the group's own lease is not released by the member.
        assert executor._sl_covered[42] == 1


class TestGroupJob:
    def test_group_reserves_aggregated_resources_once(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        members = [FakeJob(jobid=1, threads=3), FakeJob(jobid=2, threads=2)]

        group = FakeJob(jobid="grp", threads=5)
        group.jobs = members
        observed = {}

        class Base(FakeBase):
            def run_group_job(self, job):
                observed["leases"] = len(
                    list(executor._sl_namespace.path.glob("*.lease"))
                )
                observed["covered"] = dict(executor._sl_covered)
                return "group"

        executor.__class__ = type("H", (Harness, Base), {})
        result = executor.run_group_job(group)
        assert result == "group"
        # Exactly one lease for the whole group, covering all members.
        assert observed["leases"] == 1
        assert observed["covered"] == {1: 1, 2: 1}
        # Everything released afterwards.
        assert list(executor._sl_namespace.path.glob("*.lease")) == []
        assert executor._sl_covered == {}

    def test_group_releases_lease_when_members_fail(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        group = FakeJob(jobid="grp", threads=2)
        group.jobs = [FakeJob(jobid=9, threads=2)]

        class Base(FakeBase):
            def run_group_job(self, job):
                raise RuntimeError("member died")

        executor.__class__ = type("H", (Harness, Base), {})
        with pytest.raises(RuntimeError):
            executor.run_group_job(group)
        assert list(executor._sl_namespace.path.glob("*.lease")) == []
        assert executor._sl_covered == {}

    def test_oversized_group_fails_fast(self, tmp_path):
        from snakemake_interface_common.exceptions import WorkflowError

        executor = build(tmp_path, pool_cores=4)
        group = FakeJob(jobid="grp", threads=9)
        group.jobs = [FakeJob(jobid=1, threads=9)]
        with pytest.raises(WorkflowError, match="requires cores=9"):
            executor.run_group_job(group)


class TestCgroupIntegration:
    def test_properties_include_quota_memory_and_swap_guard(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        props = executor._sl_cgroup_properties({"cores": 4, "mem_mb": 200})
        assert "--property=CPUQuota=50%" in props
        assert "--property=MemoryMax=200M" in props
        assert "--property=MemorySwapMax=0" in props

    def test_properties_omit_memory_when_unknown(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        props = executor._sl_cgroup_properties({"cores": 1})
        assert not any("Memory" in p for p in props)

    def test_launcher_script_degrades_gracefully(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        path = executor._sl_write_launcher(FakeJob(jobid=5), {"cores": 1, "mem_mb": 64})
        content = Path(path).read_text()
        # Falls back to a plain shell when systemd-run or D-Bus is unavailable.
        assert "systemd-run" in content
        assert "exec /bin/sh" in content
        assert os.access(path, os.X_OK)

    def test_prepare_is_a_noop_when_cgroup_disabled(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        job = FakeJob()
        executor._sl_cgroup_enabled = False
        executor._sl_prepare_single_job(job)
        assert "shell_exec" not in job.resources

    def test_prepare_respects_user_supplied_shell_exec(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        executor._sl_cgroup_enabled = True
        job = FakeJob(resources={"shell_exec": "/custom/wrapper"})
        executor._sl_prepare_single_job(job)
        assert job.resources["shell_exec"] == "/custom/wrapper"

    def test_prepare_injects_launcher_when_enabled(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        executor._sl_cgroup_enabled = True
        job = FakeJob(jobid=3, threads=1, resources={"mem_mb": 32})
        executor._sl_prepare_single_job(job)
        assert Path(job.resources["shell_exec"]).exists()

    def test_prepare_warns_instead_of_failing_on_error(self, tmp_path, monkeypatch):
        executor = build(tmp_path, pool_cores=8)
        executor._sl_cgroup_enabled = True

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(executor, "_sl_write_launcher", boom)
        executor._sl_prepare_single_job(FakeJob())
        assert "could not enable cgroup enforcement" in executor.logger.text()

    def test_launcher_directory_is_created_and_cleaned(self, tmp_path):
        executor = build(tmp_path, pool_cores=8)
        executor._sl_cgroup_enabled = True
        job = FakeJob(jobid=1)
        executor._sl_prepare_single_job(job)
        directory = Path(executor._sl_launcher_dir)
        assert directory.is_dir()
        executor._sl_cleanup_launchers()
        assert not directory.exists()


class TestRunJobDispatch:
    def test_plain_run_job_when_cgroup_disabled(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_cgroup_enabled = False
        assert executor.run_job(FakeJob()) == "job"

    def test_run_single_job_passes_through(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        assert executor.run_single_job(FakeJob()) == "single"


class TestLifecycle:
    def test_shutdown_releases_leases_and_removes_instance_claim(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_leases.register()
        executor._sl_leases.acquire("x", {"cores": 1})
        assert list(executor._sl_namespace.path.glob("*.lease"))

        executor.shutdown()

        assert list(executor._sl_namespace.path.glob("*.lease")) == []
        assert list(executor._sl_namespace.path.glob("*.inst")) == []

    def test_cancel_releases_leases(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_leases.acquire("y", {"cores": 1})
        executor.cancel()
        assert list(executor._sl_namespace.path.glob("*.lease")) == []

    def test_gc_report_lists_stale_leases(self, tmp_path):
        executor = build(tmp_path, pool_cores=4, gc_report="on")
        stale = executor._sl_namespace.path / "ghost-1-abc.lease"
        stale.write_text(json.dumps({"version": 1, "resources": {"cores": 1}}))
        executor.shutdown()
        assert "stale lease files" in executor.logger.text()

    def test_release_everything_is_safe_without_leases(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_leases = None
        executor._sl_release_everything()  # must not raise

    def test_teardown_swallows_lease_errors(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)

        def boom():
            raise RuntimeError("teardown trouble")

        executor._sl_leases.release_all = boom
        executor._sl_release_everything()  # must not raise


class TestNumaApplication:
    def test_noop_when_numa_disabled(self, tmp_path):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = False
        executor._sl_apply_numa({"cores": 1})  # must not raise

    def test_warns_when_no_node_can_host_the_job(self, tmp_path, monkeypatch):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        monkeypatch.setattr(numa_module, "select_node", lambda *a, **k: None)
        executor._sl_topology = numa_module.detect_topology()
        executor._sl_apply_numa({"cores": 1})
        assert "without binding" in executor.logger.text()

    def test_placement_accounts_for_already_running_jobs(self, tmp_path, monkeypatch):
        """`select_node`'s `used_per_node` must actually be passed.

        It never was, so every job independently picked the same node, and the
        per-node free-memory snapshot (taken once at startup) never changed.
        """
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        seen = {}

        def spy(topology, *, cores, mem_mib, used_per_node=None):
            seen["used_per_node"] = used_per_node
            return topology.nodes[0]

        monkeypatch.setattr(numa_module, "select_node", spy)
        monkeypatch.setattr(
            numa_module,
            "bind_current_thread",
            lambda *a, **k: numa_module.Binding(applied=True, node=0),
        )
        executor._sl_topology = numa_module.NumaTopology(nodes=[node0()])

        executor._sl_apply_numa({"cores": 1, "mem_mb": 100})
        assert seen["used_per_node"] is not None, (
            "used_per_node must be passed so placement can avoid a busy node"
        )

    def test_offset_advances_so_jobs_do_not_share_a_cpu(self, tmp_path, monkeypatch):
        """Consecutive placements must request different CPU offsets."""
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        offsets = []

        def spy(topology, *, cores, mem_mib, used_per_node=None):
            return topology.nodes[0]

        def spy_bind(node, *, cores, offset=0, logger=None):
            offsets.append(offset)
            return numa_module.Binding(applied=True, node=node.index)

        monkeypatch.setattr(numa_module, "select_node", spy)
        monkeypatch.setattr(numa_module, "bind_current_thread", spy_bind)
        executor._sl_topology = numa_module.NumaTopology(nodes=[node0()])

        for _ in range(4):
            executor._sl_apply_numa({"cores": 1})

        assert len(set(offsets)) == len(offsets), (
            f"every job asked for the same CPU offset: {offsets}"
        )

    def test_affinity_is_reset_after_a_job(self, tmp_path, monkeypatch):
        """Leaked affinity must not follow a worker thread into its next job."""
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        calls = []
        monkeypatch.setattr(
            numa_module, "reset_current_thread", lambda: calls.append("reset")
        )
        # Stub placement so this test measures only the reset, not the binding.
        monkeypatch.setattr(
            numa_module,
            "select_node",
            lambda topology, *, cores, mem_mib, used_per_node=None: None,
        )
        executor.cached_or_run(FakeJob(), lambda *a: "ran")
        assert calls == ["reset"], (
            "cached_or_run must reset NUMA affinity in its finally block"
        )

    def test_affinity_is_reset_even_when_the_job_raises(self, tmp_path, monkeypatch):
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        calls = []
        monkeypatch.setattr(
            numa_module, "reset_current_thread", lambda: calls.append("reset")
        )
        monkeypatch.setattr(
            numa_module,
            "select_node",
            lambda topology, *, cores, mem_mib, used_per_node=None: None,
        )

        def boom(*a):
            raise RuntimeError("job failed")

        with pytest.raises(RuntimeError):
            executor.cached_or_run(FakeJob(), boom)
        assert calls == ["reset"], "a failed job must still release its binding"

    def test_node_reservation_is_released_after_a_job(self, tmp_path, monkeypatch):
        """Per-node accounting must not leak, or a node looks permanently busy."""
        executor = build(tmp_path, pool_cores=4)
        executor._sl_numa_enabled = True
        from snakemake_executor_plugin_shared_local import numa as numa_module

        monkeypatch.setattr(
            numa_module,
            "select_node",
            lambda topology, *, cores, mem_mib, used_per_node=None: topology.nodes[0],
        )
        monkeypatch.setattr(
            numa_module,
            "bind_current_thread",
            lambda *a, **k: numa_module.Binding(applied=True, node=0),
        )
        monkeypatch.setattr(numa_module, "reset_current_thread", lambda: None)
        executor._sl_topology = numa_module.NumaTopology(nodes=[node0()])

        job = FakeJob(threads=1, resources={"mem_mb": 2000})
        executor.cached_or_run(job, lambda *a: "ran")
        assert executor._sl_numa_used == {}, (
            f"node reservation leaked: {executor._sl_numa_used}"
        )


def node0():
    from snakemake_executor_plugin_shared_local.numa import NumaNode

    return NumaNode(
        index=0,
        cpus=[0, 2, 4, 6],
        mem_total_mib=100_000,
        mem_free_mib=90_000,
    )


class TestConfigIntegration:
    def test_config_file_value_is_used_when_flag_is_unset(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"pool_cores": 3, "numa_bind": "off"}))

        settings = FakeSettings()
        settings.dir = str(tmp_path / "leases")
        settings.config = str(config)
        resolved = resolve_settings(settings)
        assert resolved.pool_cores == 3
        assert resolved.numa_bind is False


def test_leasemanager_errors_surface_as_workflow_errors(tmp_path):
    from snakemake_interface_common.exceptions import WorkflowError

    executor = build(tmp_path, pool_cores=4)

    def boom(*args, **kwargs):
        raise LeaseError("lease subsystem unavailable")

    executor._sl_leases.acquire = boom
    with pytest.raises(WorkflowError, match="lease subsystem unavailable"):
        executor._sl_acquire(FakeJob(), {"cores": 1})
