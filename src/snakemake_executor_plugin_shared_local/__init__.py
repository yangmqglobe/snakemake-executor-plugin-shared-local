"""Snakemake executor plugin coordinating several local instances on one machine.

The plugin still runs jobs locally; its only extra job is making concurrent
Snakemake instances share one machine without fighting over it.

Why an executor plugin
----------------------
Only an executor can *hold a job back* until resources are free. A scheduler
plugin can merely return fewer jobs, and returning none while nothing is running
leaves Snakemake blocked forever on ``_open_jobs.acquire()``, so global
arbitration is impossible there.

Where admission happens
-----------------------
Not in :meth:`run_job`. Snakemake's executor contract calls ``run_jobs(jobs)`` as
``for job in jobs: run_job(job)`` from the *scheduling* thread, so blocking there
stalls the whole scheduler: measured with a probe, two independent jobs were
serialised behind one blocked ``run_job``. Admission therefore happens in
:meth:`cached_or_run`, which the local executor invokes on a worker thread of its
pool. Blocking there only occupies that worker, which is exactly the intent.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"


import contextlib
import os
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import AbstractExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecutorSettingsBase,
)

from . import cgroup as cgroup_mod
from . import numa as numa_mod
from .config import AUTO, ConfigError, resolve_settings
from .lease import AdmissionImpossible, LeaseError, LeaseManager, normalize_resources
from .namespace import resolve_namespace
from .policy import probe_priority
from .pool import MEMORY_DIMENSION, compute_pool

# Required by the plugin registry: this plugin executes locally.
common_settings = CommonSettings(
    non_local_exec=False,
    implies_no_shared_fs=False,
    job_deploy_sources=False,
    pass_envvar_declarations_to_cmd=True,
    auto_deploy_default_storage_provider=False,
)

TriState = Literal["auto", "on", "off"]

# Name of the resource Snakemake's shell layer uses to replace the shell binary.
SHELL_EXEC_RESOURCE = "shell_exec"


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    """Settings exposed as ``--shared-local-<name>`` on the Snakemake CLI.

    Every field defaults to ``"auto"``/``None`` so that "not specified" is
    distinguishable from "explicitly set", which is what lets the config file and
    auto-detection fill the gap. Booleans are deliberately modelled as tri-state
    strings: ``argparse_dataclass`` renders ``bool`` defaults as inverted
    ``--no-<flag>`` options, which also loses the plugin's CLI prefix (verified),
    while ``Literal`` yields a prefixed, choice-checked ``--shared-local-<name>``.

    Every field also carries ``"env_var": True``. That flag is what makes Snakemake
    pass ``env_var=<SNAKEMAKE_SHARED_LOCAL_*>`` through to ``configargparse``: the
    CLI generator only forwards it when the metadata opts in, so without it the
    documented environment variables are silently inert. ``configargparse`` then
    applies the precedence CLI > environment > config file > default.
    """

    namespace: Literal["auto", "user", "group", "system"] = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "Coordination scope. 'user' (default) shares resources between this "
                "user's instances only, 'group' between instances of users sharing the "
                "primary group, 'system' between all users on the machine. Shared "
                "scopes coordinate resources; they are not a security boundary."
            ),
        },
    )
    dir: Optional[str] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Directory holding lease files. Defaults to a scope-specific "
                "subdirectory of /dev/shm/snakemake-shared-local."
            ),
        },
    )
    pool_cores: Optional[int] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Total CPU cores shared by all coordinated instances. Defaults to "
                "detected CPUs minus a 10% reserve. This is the hard ceiling; --cores "
                "only limits a single instance and may exceed it."
            ),
        },
    )
    pool_mem: Optional[str] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Total memory shared by all coordinated instances, e.g. '450G' or "
                "'400000'. Suffixes are binary: '450G' means 450 GiB. The value is "
                "compared directly against jobs' mem_mb. Defaults to detected memory "
                "minus a 10% reserve."
            ),
        },
    )
    pool_conflict: Literal["auto", "error", "warn"] = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "What to do when instances sharing a namespace declare different "
                "pools. 'error' (default) refuses to run, because a mismatch silently "
                "voids the global limit. 'warn' proceeds anyway."
            ),
        },
    )
    instance_weight: Optional[float] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": "Relative scheduling weight of this instance versus others.",
        },
    )
    aging_seconds: Optional[int] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Time constant for priority aging, which stops low-priority jobs from "
                "starving. 0 disables aging."
            ),
        },
    )
    aging_step: Optional[int] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": "Priority bonus granted per aging_seconds of waiting.",
        },
    )
    admit_timeout: Optional[int] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Seconds to wait for resources before failing. 0 (default) waits "
                "indefinitely, which is usually right for long jobs."
            ),
        },
    )
    poll_interval: Optional[float] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": "Seconds between admission attempts while waiting.",
        },
    )
    numa_bind: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "Pin each job to the NUMA node with the most free capacity. 'auto' "
                "(default) enables it on multi-node machines, 'on' forces it, 'off' "
                "disables it. Placement only; it does not limit resources."
            ),
        },
    )
    cgroup: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "Wrap jobs in a cgroup v2 scope so the kernel enforces CPU and memory "
                "limits. 'auto' (default) applies the plugin default, which is OFF: "
                "enabling kernel enforcement implicitly would change how failing jobs "
                "fail for existing users. 'on' enables it, degrading to a warning if "
                "cgroup v2, systemd-run or the D-Bus session is unavailable. 'off' "
                "disables it."
            ),
        },
    )
    require_mem: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "Fail when a job declares no mem_mb instead of skipping memory "
                "arbitration for it. Consider --default-resources mem_mb=... first."
            ),
        },
    )
    record_rule: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": (
                "Record rule names in lease files. 'auto' (default) records them "
                "except in the 'system' scope, where other users could read them."
            ),
        },
    )
    gc_report: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": "Report stale lease files that could not be removed.",
        },
    )
    verbose_leases: TriState = field(
        default=AUTO,
        metadata={
            "env_var": True,
            "help": "Log every admission, wait and collection decision.",
        },
    )
    config: Optional[str] = field(
        default=None,
        metadata={
            "env_var": True,
            "help": (
                "Path to a JSON settings file "
                "(default ~/.config/snakemake/shared-local/config.json)."
            ),
        },
    )


def _resolve_local_executor() -> type:
    """Return the core local executor class.

    The plugin registry is still being populated while this module is imported
    (``pkgutil.iter_modules()`` reaches this package before the built-in
    ``local`` plugin is registered), so the lookup must be deferred to
    instantiation time. The registry is preferred because it is the public
    interface; the direct import is only a fallback for unusual deployments.
    """
    try:
        from snakemake_interface_executor_plugins.registry import ExecutorPluginRegistry

        return ExecutorPluginRegistry().get_plugin("local").executor
    except Exception:  # pragma: no cover - fallback path
        from snakemake.executors.local import Executor as LocalExecutor

        return LocalExecutor


_combined_cache: dict[str, type] = {}


def _combined_executor_class() -> type:
    """Build (once) the class combining this plugin's overrides with ``local``."""
    if "cls" not in _combined_cache:
        base = _resolve_local_executor()
        _combined_cache["cls"] = type(
            "Executor", (_SharedLocalExecutor, base), {"__module__": __name__}
        )
    return _combined_cache["cls"]


class _SharedLocalExecutor:
    """Overrides layered on top of Snakemake's local executor.

    Implemented as a mixin so the concrete base class can be resolved lazily at
    instantiation time (see :func:`_resolve_local_executor`).
    """

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore[misc]
        self._sl_lock = threading.RLock()
        self._sl_covered: dict[Any, int] = {}
        self._sl_mem_warned = False
        self._sl_cgroup_support: cgroup_mod.CgroupSupport | None = None
        self._sl_launcher_dir: str | None = None
        self._sl_launcher_seq = 0

        try:
            self._sl_settings = resolve_settings(self.workflow.executor_settings)
        except ConfigError as err:
            raise WorkflowError(str(err)) from err

        for warning in self._sl_settings.warnings:
            self.logger.warning(warning)

        pool = compute_pool(
            self._sl_settings.pool_cores,
            self._sl_settings.pool_mem_mb,
        )
        self._sl_pool = pool

        try:
            namespace = resolve_namespace(
                self._sl_settings.namespace, explicit_dir=self._sl_settings.dir
            )
        except ConfigError as err:
            raise WorkflowError(str(err)) from err
        self._sl_namespace = namespace

        self._sl_leases = LeaseManager(
            namespace,
            pool.as_dict(),
            weight=self._sl_settings.instance_weight,
            aging_seconds=self._sl_settings.aging_seconds,
            aging_step=self._sl_settings.aging_step,
            pool_conflict=self._sl_settings.pool_conflict,
            verbose=self._sl_settings.verbose_leases,
            logger=self.logger,
            poll_interval=self._sl_settings.poll_interval,
            admit_timeout=self._sl_settings.admit_timeout,
        )
        try:
            self._sl_leases.negotiate_pool()
            self._sl_leases.register()
        except LeaseError as err:
            raise WorkflowError(str(err)) from err

        self._sl_topology = numa_mod.detect_topology()
        self._sl_numa_enabled = self._resolve_numa()
        # Per-node in-flight memory, so placement can avoid a node it just filled.
        # The topology's own free-memory figures are read once at startup and never
        # refreshed, so without this every job computes the same answer.
        self._sl_numa_used: dict[int, int] = {}
        # Round-robin position used to spread jobs across a node's CPUs.
        self._sl_numa_offset = 0
        if self._sl_numa_enabled:
            numa_mod.capture_original_affinity()

        self._sl_cgroup_enabled = self._resolve_cgroup()

        self.logger.info(
            f"shared-local: scope '{namespace.mode}' at {namespace.path}; "
            f"pool cores={pool.cores}"
            + (f", mem={pool.mem_mb} MiB" if pool.mem_mb is not None else "")
            + f"; NUMA {'on' if self._sl_numa_enabled else 'off'}"
            + f"; cgroup {'on' if self._sl_cgroup_enabled else 'off'}"
        )

    # --------------------------------------------------------------- settings

    def _resolve_numa(self) -> bool:
        request = self._sl_settings.numa_bind
        if request is False:
            return False
        if not self._sl_topology.available:
            if request is True:
                self.logger.warning(
                    "NUMA binding was requested but only one NUMA node was detected; "
                    "ignoring."
                )
            return False
        return True

    def _resolve_cgroup(self) -> bool:
        """Decide whether to enforce limits in the kernel for this run.

        Resolution already applied the plugin default (off), so this only has to
        deal with an explicit request: enable when the host supports it, and
        otherwise warn and continue, because failing a workflow over an optional
        hardening feature would be worse than running without it.
        """
        if not self._sl_settings.cgroup:
            return False
        support = cgroup_mod.probe(self.logger)
        self._sl_cgroup_support = support
        if support.available:
            return True
        self.logger.warning(
            f"cgroup enforcement was requested but is unavailable: {support.reason}"
        )
        return False

    # ------------------------------------------------------------------ leases

    def _sl_request_for(self, job: Any) -> dict[str, int]:
        """Translate a job's resources into pool dimensions."""
        threads = 1
        # Defensive: a job object that cannot report threads still runs, as a
        # single-core job, rather than aborting the whole workflow.
        with contextlib.suppress(Exception):
            threads = int(job.threads)
        with_mem = self._sl_pool.mem_mb is not None
        try:
            resources = dict(job.resources)
        except Exception:  # pragma: no cover
            resources = {}
        request = normalize_resources(resources, threads=threads, with_mem=with_mem)
        if with_mem and MEMORY_DIMENSION not in request:
            self._warn_missing_mem(job)
        return request

    def _warn_missing_mem(self, job: Any) -> None:
        if self._sl_mem_warned:
            return
        self._sl_mem_warned = True
        if self._sl_settings.require_mem:
            raise WorkflowError(
                f"Job {getattr(job, 'name', job)} declares no mem_mb, but "
                "--shared-local-require-mem is set. Add mem_mb to the rule's "
                "resources or pass --default-resources mem_mb=<value>."
            )
        self.logger.warning(
            "shared-local: some jobs declare no mem_mb, so memory is not arbitrated "
            "for them (CPU still is). Pass --default-resources mem_mb=<value>, or set "
            "--shared-local-require-mem to make this an error."
        )

    def _sl_job_key(self, job: Any) -> Any:
        try:
            return job.jobid
        except Exception:  # pragma: no cover - defensive
            return id(job)

    def _sl_rule_name(self, job: Any) -> str | None:
        if not self._sl_settings.record_rule:
            return None
        try:
            return job.rule.name
        except Exception:
            return None

    def _sl_acquire(self, job: Any, request: dict[str, int]) -> Any:
        key = self._sl_job_key(job)
        try:
            return self._sl_leases.acquire(
                key,
                request,
                priority=probe_priority(job, self.logger),
                rule=self._sl_rule_name(job),
            )
        except AdmissionImpossible as err:
            # Deliberately immediate: waiting would hang the workflow forever.
            raise WorkflowError(str(err)) from err
        except LeaseError as err:
            raise WorkflowError(str(err)) from err

    # ------------------------------------------------------------------- NUMA

    def _sl_apply_numa(self, request: dict[str, int]) -> tuple[int, int] | None:
        """Bind the current (worker) thread to a NUMA node for this job.

        Returns the ``(node_index, mem_mb)`` reservation so the caller can release
        it, or ``None`` when nothing was reserved.
        """
        if not self._sl_numa_enabled:
            return None
        cores = max(1, int(request.get("cores", 1)))
        with self._sl_lock:
            used_per_node = dict(self._sl_numa_used)
            offset = self._sl_numa_offset
        node = numa_mod.select_node(
            self._sl_topology,
            cores=cores,
            mem_mib=request.get("mem_mb"),
            used_per_node=used_per_node,
        )
        if node is None:
            self.logger.warning(
                "shared-local: no single NUMA node can host this job "
                f"({request}); running it without binding."
            )
            return None
        # Advance the round-robin position before binding, so two jobs admitted at
        # the same moment cannot both be handed the same CPU window.
        mem = int(request["mem_mb"]) if request.get("mem_mb") else 0
        with self._sl_lock:
            self._sl_numa_offset += cores
            if mem:
                self._sl_numa_used[node.index] = (
                    self._sl_numa_used.get(node.index, 0) + mem
                )
        binding = numa_mod.bind_current_thread(
            node, cores=cores, offset=offset, logger=self.logger
        )
        if not binding.applied:
            # Nothing was bound, so the optimistic reservation above must be undone
            # or the node would look permanently busier than it is.
            self._sl_release_numa((node.index, mem))
            return None
        return (node.index, mem)

    def _sl_release_numa(self, placement: tuple[int, int] | None) -> None:
        """Undo the per-node memory reservation made by :meth:`_sl_apply_numa`."""
        if placement is None:
            return
        node_index, mem = placement
        if not mem:
            return
        with self._sl_lock:
            remaining = self._sl_numa_used.get(node_index, 0) - mem
            if remaining > 0:
                self._sl_numa_used[node_index] = remaining
            else:
                self._sl_numa_used.pop(node_index, None)

    # ---------------------------------------------------------------- cgroup

    def _sl_launcher_dir_path(self) -> str:
        if self._sl_launcher_dir is None:
            self._sl_launcher_dir = tempfile.mkdtemp(
                prefix="shared-local-", dir=self._sl_snakemake_tmpdir()
            )
        return self._sl_launcher_dir

    def _sl_snakemake_tmpdir(self) -> str | None:
        """Prefer Snakemake's own ``.snakemake`` directory so cleanup is shared."""
        try:
            path = os.path.join(self.workflow.workdir_init, ".snakemake")
            os.makedirs(path, exist_ok=True)
            return path
        except Exception:  # pragma: no cover - fall back to the system tmpdir
            return None

    def _sl_write_launcher(self, job: Any, request: dict[str, int]) -> str:
        """Write a per-job launcher script that runs the command in a cgroup.

        Snakemake's shell layer honours a per-job ``shell_exec`` resource that
        replaces the shell binary while preserving the exact quoting of the user's
        command (verified: the wrapper receives ``-c`` and the full command as two
        argv entries). That makes it a safe interception point, unlike rewriting
        the command string ourselves.
        """
        with self._sl_lock:
            self._sl_launcher_seq += 1
            seq = self._sl_launcher_seq
        launcher_dir = self._sl_launcher_dir_path()
        path = os.path.join(launcher_dir, f"job-{self._sl_job_key(job)}-{seq}.sh")

        properties = self._sl_cgroup_properties(request)
        body = (
            "#!/bin/sh\n"
            "# Generated by snakemake-executor-plugin-shared-local.\n"
            "# Wraps the job command in a cgroup v2 scope so CPU and memory limits\n"
            "# are enforced by the kernel. Falls back to running the command\n"
            "# directly if systemd-run is unavailable.\n"
            "if command -v systemd-run >/dev/null 2>&1 && "
            '[ -n "$DBUS_SESSION_BUS_ADDRESS" ]; then\n'
            "  exec systemd-run --user --scope --quiet --collect \\\n"
            f"    {' '.join(properties)} \\\n"
            '    /bin/sh "$@"\n'
            "else\n"
            '  exec /bin/sh "$@"\n'
            "fi\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.chmod(path, stat.S_IRWXU)
        return path

    def _sl_cgroup_properties(self, request: dict[str, int]) -> list[str]:
        """Build the systemd properties for one job's resource request."""
        properties: list[str] = []
        cores = max(1, int(request.get("cores", 1)))
        pool_cores = max(1, int(self._sl_pool.cores))
        quota = max(1, -(-cores * 100 // pool_cores))  # ceil division
        properties.append(f"--property=CPUQuota={quota}%")
        mem_mib = request.get("mem_mb")
        if mem_mib:
            properties.append(f"--property=MemoryMax={int(mem_mib)}M")
            # Mandatory companion: a memory cap that may swap is not a cap
            # (measured: without this a 600 MiB allocation survived a 200 MiB cap).
            properties.append("--property=MemorySwapMax=0")
        return properties

    def _sl_prepare_single_job(self, job: Any) -> None:
        """Inject the cgroup launcher before the job command is constructed.

        ``run_single_job`` calls ``job_args_and_prepare`` before handing the job to
        ``cached_or_run``, so the resource has to be added here. Adding it later
        proved ineffective (verified: the scope was not applied).
        """
        if not self._sl_cgroup_enabled:
            return
        try:
            if SHELL_EXEC_RESOURCE in dict(job.resources):
                # Respect an explicit user choice.
                return
            request = self._sl_request_for(job)
            path = self._sl_write_launcher(job, request)
            job.add_aux_resource(SHELL_EXEC_RESOURCE, path)
        except Exception as err:  # pragma: no cover - never break the workflow
            self.logger.warning(
                f"shared-local: could not enable cgroup enforcement for this job "
                f"({err}); running it unenforced."
            )

    # ------------------------------------------------------------- execution

    def run_single_job(self, job: Any) -> Any:
        self._sl_prepare_single_job(job)
        return super().run_single_job(job)  # type: ignore[misc]

    def run_group_job(self, job: Any) -> Any:
        """Admit a whole group job atomically, then let its sub-jobs run.

        Pipe and service groups must run their members simultaneously, so taking
        one lease per member in sequence would self-deadlock: the first member
        holds resources while waiting for the second, which cannot be admitted.
        The group therefore reserves all members' resources in a single critical
        section, and the sub-jobs skip admission individually.
        """
        members = _iter_group_members(job)
        request = self._sl_request_for(job)
        key = self._sl_job_key(job)
        self._sl_acquire(job, request)

        member_keys = [self._sl_job_key(member) for member in members]
        with self._sl_lock:
            for member_key in member_keys:
                self._sl_covered[member_key] = self._sl_covered.get(member_key, 0) + 1
        try:
            for member in members:
                self._sl_prepare_single_job(member)
            return super().run_group_job(job)  # type: ignore[misc]
        finally:
            with self._sl_lock:
                for member_key in member_keys:
                    remaining = self._sl_covered.get(member_key, 0) - 1
                    if remaining > 0:
                        self._sl_covered[member_key] = remaining
                    else:
                        self._sl_covered.pop(member_key, None)
            self._sl_leases.release(key)

    def cached_or_run(self, job: Any, run_func: Any, *args: Any) -> Any:
        """Admit the job, run it, then release. Runs on a pool worker thread.

        This is the only place that may block waiting for resources: the
        scheduling thread must never be stalled (see module docstring).
        """
        key = self._sl_job_key(job)
        with self._sl_lock:
            covered_by_group = key in self._sl_covered
        request = self._sl_request_for(job)

        if not covered_by_group:
            self._sl_acquire(job, request)
        placement = self._sl_apply_numa(request)
        try:
            return super().cached_or_run(job, run_func, *args)  # type: ignore[misc]
        finally:
            # Undo the thread binding before the worker picks up another job.
            # sched_setaffinity applies to the calling thread, so without this a
            # worker stays pinned to one job's CPUs for the rest of the run --
            # including for jobs that never bind at all.
            if self._sl_numa_enabled:
                numa_mod.reset_current_thread()
            self._sl_release_numa(placement)
            if not covered_by_group:
                self._sl_leases.release(key)

    # --------------------------------------------------------------- teardown

    def shutdown(self) -> None:
        self._sl_release_everything()
        super().shutdown()  # type: ignore[misc]

    def cancel(self) -> None:
        self._sl_release_everything()
        super().cancel()  # type: ignore[misc]

    def _sl_release_everything(self) -> None:
        leases = getattr(self, "_sl_leases", None)
        if leases is None:
            return
        try:
            if self._sl_settings.gc_report:
                stale = leases.report_garbage()
                if stale:
                    self.logger.warning(
                        "shared-local: stale lease files left on disk (harmless: their "
                        "resources count as zero): " + ", ".join(stale)
                    )
            leases.release_all()
            leases.unregister()
        except Exception as err:  # pragma: no cover - teardown must not raise
            self.logger.debug(f"shared-local: teardown issue: {err}")
        finally:
            self._sl_cleanup_launchers()

    def _sl_cleanup_launchers(self) -> None:
        directory = getattr(self, "_sl_launcher_dir", None)
        if not directory:
            return
        import shutil

        # Best effort: a leftover launcher script is harmless.
        with contextlib.suppress(Exception):
            shutil.rmtree(directory, ignore_errors=True)
        self._sl_launcher_dir = None


def _iter_group_members(job: Any) -> list[Any]:
    """Return the member jobs of a Snakemake group job.

    ``GroupJob`` iterates its members; the interface accessor is tried first so
    this keeps working if the concrete class changes.
    """
    try:
        return list(job.jobs)
    except Exception:
        pass
    try:
        return list(job)
    except Exception:  # pragma: no cover - defensive
        return []


class Executor(AbstractExecutor):
    """Registry-visible executor class.

    The plugin registry validates ``module.Executor`` at collection time, when the
    built-in ``local`` plugin is not yet registered, so the real base class can
    only be resolved later. Instances are therefore created as members of a
    dynamically built class combining :class:`_SharedLocalExecutor` with the local
    executor. ``Executor`` itself is never instantiated directly.
    """

    def __new__(cls, workflow: Any, logger: Any) -> Any:
        combined = _combined_executor_class()
        instance = object.__new__(combined)
        combined.__init__(instance, workflow, logger)  # type: ignore[misc]
        return instance

    # The abstract base requires these to exist for class validation, but they are
    # unreachable because ``__new__`` returns an instance of another class.
    def run_job(self, job: JobExecutorInterface) -> None:  # pragma: no cover
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def cancel(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def handle_job_success(self, job: JobExecutorInterface) -> None:  # pragma: no cover
        raise NotImplementedError

    def handle_job_error(self, job: JobExecutorInterface) -> None:  # pragma: no cover
        raise NotImplementedError
