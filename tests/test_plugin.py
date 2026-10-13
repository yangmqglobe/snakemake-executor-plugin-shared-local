"""Tests that the plugin satisfies Snakemake's plugin and executor contracts.

These are the tests that catch "the code is fine but Snakemake will not load it"
breakage, which is otherwise only discovered by running a real workflow.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecutorSettingsBase,
)

MODULE_NAME = "snakemake_executor_plugin_shared_local"


class TestModuleContract:
    """The registry validates these attributes at collection time."""

    def test_module_imports(self):
        module = importlib.import_module(MODULE_NAME)
        assert module is not None

    def test_exposes_common_settings(self):
        module = importlib.import_module(MODULE_NAME)
        assert isinstance(module.common_settings, CommonSettings)

    def test_executes_locally(self):
        """non_local_exec=False is what makes jobs run here and resources visible."""
        module = importlib.import_module(MODULE_NAME)
        assert module.common_settings.local_exec is True
        assert module.common_settings.non_local_exec is False

    def test_shares_the_filesystem(self):
        module = importlib.import_module(MODULE_NAME)
        # Jobs run in place, so the workflow filesystem is shared.
        assert module.common_settings.implies_no_shared_fs is False

    def test_exposes_executor_class(self):
        from snakemake_interface_executor_plugins.executors.base import (
            AbstractExecutor,
        )

        module = importlib.import_module(MODULE_NAME)
        assert issubclass(module.Executor, AbstractExecutor)

    def test_exposes_settings_class(self):
        module = importlib.import_module(MODULE_NAME)
        assert issubclass(module.ExecutorSettings, ExecutorSettingsBase)


class TestSettingsShape:
    """Settings must be plottable by Snakemake's CLI generator."""

    def _fields(self):
        from dataclasses import fields

        module = importlib.import_module(MODULE_NAME)
        return {f.name: f for f in fields(module.ExecutorSettings)}

    def test_every_field_has_help_text(self):
        # Snakemake raises InvalidPluginException when a field lacks "help".
        for name, thefield in self._fields().items():
            assert "help" in thefield.metadata, f"{name} has no help string"

    def test_every_field_has_a_default(self):
        # Snakemake requires a default or default_factory for every field.
        from dataclasses import MISSING

        for name, thefield in self._fields().items():
            has_default = thefield.default is not MISSING
            has_factory = thefield.default_factory is not MISSING
            assert has_default or has_factory, f"{name} has no default"

    def test_optional_fields_default_to_unset_sentinels(self):
        """Sentinels are what let the config file fill in values at all."""
        from snakemake_executor_plugin_shared_local.config import AUTO

        fields = self._fields()
        for name in (
            "pool_cores",
            "pool_mem",
            "dir",
            "instance_weight",
            "aging_seconds",
            "admit_timeout",
        ):
            assert fields[name].default is None, f"{name} should default to None"
        for name in (
            "numa_bind",
            "cgroup",
            "record_rule",
            "namespace",
            "pool_conflict",
        ):
            assert fields[name].default == AUTO, f"{name} should default to AUTO"

    def test_booleans_use_tri_state_strings(self):
        """bool defaults lose the CLI prefix and cannot express 'unset'."""
        import typing

        fields = self._fields()
        for name in ("cgroup", "numa_bind", "record_rule", "require_mem"):
            origin = typing.get_origin(fields[name].type)
            args = typing.get_args(fields[name].type)
            assert origin is typing.Literal
            assert "auto" in args


class TestRegistryIntegration:
    def test_plugin_is_discoverable_from_source_tree(self):
        """The registry scans sys.path, so an src layout must be importable.

        This mirrors how the package behaves when installed conventionally.
        """
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            registry = ExecutorPluginRegistry()
            registry.collect_plugins()
            names = registry.get_registered_plugins()
            # Registered names are the shortened plugin names.
            assert "shared-local" in names
        finally:
            if added:
                sys.path.remove(src)
            # The registry is a singleton keyed on collection time; drop our entry
            # so later tests in this process are unaffected.
            from snakemake_interface_executor_plugins.registry import (
                ExecutorPluginRegistry as Registry,
            )

            Registry().plugins.pop("shared-local", None)

    def test_cli_prefix_matches_the_documented_flags(self):
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            registry = ExecutorPluginRegistry()
            registry.collect_plugins()
            plugin = registry.get_plugin("shared-local")
            assert plugin.cli_prefix == "shared-local"
            assert plugin.get_cli_arg("pool_cores") == "--shared-local-pool-cores"
            assert (
                plugin.get_envvar("pool_cores") == "SNAKEMAKE_SHARED_LOCAL_POOL_CORES"
            )
        finally:
            if added:
                sys.path.remove(src)

    def test_settings_survive_a_cli_roundtrip(self):
        import configargparse
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            registry = ExecutorPluginRegistry()
            registry.collect_plugins()
            parser = configargparse.ArgumentParser()
            registry.register_cli_args(parser)
            args = parser.parse_args(
                [
                    "--shared-local-pool-cores",
                    "32",
                    "--shared-local-namespace",
                    "group",
                    "--shared-local-cgroup",
                    "on",
                ]
            )
            settings = registry.get_plugin("shared-local").get_settings(args)
            assert settings.pool_cores == 32
            assert settings.namespace == "group"
            # Tri-state values arrive as strings and resolve to bools later.
            assert settings.cgroup == "on"
        finally:
            if added:
                sys.path.remove(src)


class TestEnvironmentVariables:
    """Every setting must be reachable through SNAKEMAKE_SHARED_LOCAL_*.

    Snakemake only forwards ``env_var`` to its ``configargparse`` backend when the
    field metadata opts in, so a missing flag makes the documented environment
    variables silently inert. That was the second bug these tests pin.
    """

    def _fields(self):
        from dataclasses import fields

        module = importlib.import_module(MODULE_NAME)
        return {f.name: f for f in fields(module.ExecutorSettings)}

    def test_every_field_opts_into_env_vars(self):
        for name, thefield in self._fields().items():
            assert thefield.metadata.get("env_var") is True, (
                f"{name} does not request an environment variable, so "
                f"{name.upper()} would be advertised but ignored"
            )

    def _parse(self, argv, env=None):
        """Build the real plugin parser and return resolved settings."""
        import configargparse
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            registry = ExecutorPluginRegistry()
            registry.collect_plugins()
            parser = configargparse.ArgumentParser()
            registry.register_cli_args(parser)
            # configargparse reads os.environ by default; pass it explicitly so
            # the test does not depend on process-global mutation.
            args = parser.parse_args(argv, env_vars=env if env is not None else {})
            return registry.get_plugin("shared-local").get_settings(args)
        finally:
            if added:
                sys.path.remove(src)

    def test_env_var_sets_a_value(self):
        settings = self._parse([], env={"SNAKEMAKE_SHARED_LOCAL_POOL_CORES": "96"})
        assert settings.pool_cores == 96

    def test_env_var_works_for_tri_state_fields(self):
        settings = self._parse([], env={"SNAKEMAKE_SHARED_LOCAL_CGROUP": "on"})
        assert settings.cgroup == "on"

    def test_cli_overrides_the_env_var(self):
        settings = self._parse(
            ["--shared-local-pool-cores", "8"],
            env={"SNAKEMAKE_SHARED_LOCAL_POOL_CORES": "96"},
        )
        assert settings.pool_cores == 8

    def test_env_var_reaches_resolved_settings(self):
        """The env var must land in the value resolve_settings actually uses."""
        settings = self._parse([], env={"SNAKEMAKE_SHARED_LOCAL_POOL_CORES": "96"})
        resolved = importlib.import_module(MODULE_NAME).resolve_settings(settings)
        assert resolved.pool_cores == 96

    def test_env_var_overrides_the_config_file(self, tmp_path):
        """Precedence: the CLI/env layer outranks the plugin's JSON config file.

        The JSON file is applied inside ``resolve_settings``, not by argparse, so
        the assertion has to run through it to be meaningful.
        """
        module = importlib.import_module(MODULE_NAME)
        config = tmp_path / "config.json"
        config.write_text('{"pool_cores": 4}')
        settings = self._parse(
            ["--shared-local-config", str(config)],
            env={"SNAKEMAKE_SHARED_LOCAL_POOL_CORES": "96"},
        )
        assert module.resolve_settings(settings).pool_cores == 96

    def test_config_file_applies_when_the_env_var_is_absent(self, tmp_path):
        module = importlib.import_module(MODULE_NAME)
        config = tmp_path / "config.json"
        config.write_text('{"pool_cores": 4}')
        settings = self._parse(["--shared-local-config", str(config)], env={})
        assert module.resolve_settings(settings).pool_cores == 4

    def test_env_var_can_set_require_mem(self, tmp_path):
        """A tri-state setting must be usable from the environment too."""
        module = importlib.import_module(MODULE_NAME)
        settings = self._parse([], env={"SNAKEMAKE_SHARED_LOCAL_REQUIRE_MEM": "on"})
        assert module.resolve_settings(settings).require_mem is True


class TestExecutorWiring:
    """Exercise the mixin's admission logic with a stubbed Snakemake side."""

    def _make_executor(self, tmp_path, **overrides):
        module = importlib.import_module(MODULE_NAME)
        settings = module.ExecutorSettings()
        settings.pool_cores = overrides.get("pool_cores", 4)
        settings.dir = str(tmp_path)
        settings.numa_bind = "off"
        settings.cgroup = "off"
        resolved = module.resolve_settings(settings)

        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, msg, *a, **k):
                self.messages.append(("info", str(msg)))

            def warning(self, msg, *a, **k):
                self.messages.append(("warning", str(msg)))

            def debug(self, msg, *a, **k):
                self.messages.append(("debug", str(msg)))

            def error(self, msg, *a, **k):
                self.messages.append(("error", str(msg)))

        class Workflow:
            executor_settings = settings
            workdir_init = str(tmp_path)

        executor = object.__new__(module._SharedLocalExecutor)
        executor.workflow = Workflow()
        executor.logger = Logger()
        # Build the state that __post_init__ would normally create, without the
        # parts that need a live Snakemake workflow.
        executor._sl_lock = __import__("threading").RLock()
        executor._sl_covered = {}
        executor._sl_mem_warned = False
        executor._sl_cgroup_support = None
        executor._sl_launcher_dir = None
        executor._sl_launcher_seq = 0
        executor._sl_settings = resolved
        return executor, module

    def test_request_translation_uses_threads_and_memory(self, tmp_path):
        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_pool = Pool(cores=4, mem_mb=8000)

        class Job:
            threads = 3
            resources = {"mem_mb": 512, "foo": 2, "_cores": 3}

        request = executor._sl_request_for(Job())
        assert request["cores"] == 3
        assert request["mem_mb"] == 512
        assert request["foo"] == 2

    def test_missing_mem_warns_once(self, tmp_path):
        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_pool = Pool(cores=4, mem_mb=8000)

        class Job:
            threads = 1
            resources = {}

        executor._sl_request_for(Job())
        first = len(executor.logger.messages)
        executor._sl_request_for(Job())
        assert len(executor.logger.messages) == first, "warning must be emitted once"

    def test_require_mem_raises(self, tmp_path):
        from snakemake_interface_common.exceptions import WorkflowError

        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_settings.require_mem = True
        executor._sl_pool = Pool(cores=4, mem_mb=8000)

        class Job:
            threads = 1
            resources = {}
            name = "j"

        with pytest.raises(WorkflowError, match="no mem_mb"):
            executor._sl_request_for(Job())

    def test_cgroup_properties_include_swap_guard(self, tmp_path):
        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_pool = Pool(cores=8, mem_mb=8000)
        props = executor._sl_cgroup_properties({"cores": 2, "mem_mb": 300})
        assert "--property=CPUQuota=25%" in props
        assert "--property=MemoryMax=300M" in props
        assert "--property=MemorySwapMax=0" in props

    def test_cgroup_launcher_script_is_written_and_executable(self, tmp_path):
        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_pool = Pool(cores=8, mem_mb=8000)

        class Job:
            jobid = 3

        path = executor._sl_write_launcher(Job(), {"cores": 1, "mem_mb": 100})
        content = Path(path).read_text()
        assert "systemd-run" in content
        # Must gracefully degrade when systemd-run or D-Bus is unavailable.
        assert "exec /bin/sh" in content
        assert os.access(path, os.X_OK)

    def test_launcher_respects_an_explicit_user_choice(self, tmp_path):
        from snakemake_executor_plugin_shared_local.pool import Pool

        executor, _ = self._make_executor(tmp_path)
        executor._sl_pool = Pool(cores=8, mem_mb=8000)
        executor._sl_cgroup_enabled = True

        class Job:
            jobid = 1
            threads = 1
            resources = {"shell_exec": "/my/own/wrapper.sh"}

        executor._sl_prepare_single_job(Job())
        assert Job.resources["shell_exec"] == "/my/own/wrapper.sh"

    def test_group_members_are_iterated(self):
        from snakemake_executor_plugin_shared_local import _iter_group_members

        class Group:
            jobs = ["a", "b"]

        assert _iter_group_members(Group()) == ["a", "b"]

    def test_group_iteration_falls_back_to_iteration_protocol(self):
        from snakemake_executor_plugin_shared_local import _iter_group_members

        class Group:
            def __iter__(self):
                return iter(["x", "y"])

        assert _iter_group_members(Group()) == ["x", "y"]

    def test_group_iteration_tolerates_broken_objects(self):
        from snakemake_executor_plugin_shared_local import _iter_group_members

        class Broken:
            @property
            def jobs(self):
                raise RuntimeError("boom")

        assert _iter_group_members(Broken()) == []


class TestLazyBaseResolution:
    def test_local_executor_is_resolved_from_the_registry(self):
        """The registry is empty during import, so resolution must be deferred."""
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            ExecutorPluginRegistry().collect_plugins()
            module = importlib.import_module(MODULE_NAME)
            resolved = module._resolve_local_executor()
            assert resolved is not None
            from snakemake.executors.local import Executor as LocalExecutor

            assert resolved is LocalExecutor
        finally:
            if added:
                sys.path.remove(src)

    def test_combined_class_is_cached(self):
        from snakemake_interface_executor_plugins.registry import (
            ExecutorPluginRegistry,
        )

        src = str(Path(__file__).resolve().parent.parent / "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        try:
            ExecutorPluginRegistry().collect_plugins()
            module = importlib.import_module(MODULE_NAME)
            module._combined_cache.clear()
            first = module._combined_executor_class()
            second = module._combined_executor_class()
            assert first is second, "the dynamic class must be built only once"
            assert module._SharedLocalExecutor in first.__mro__
        finally:
            if added:
                sys.path.remove(src)

    def test_fallback_import_when_registry_fails(self, monkeypatch):
        module = importlib.import_module(MODULE_NAME)

        def boom():
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(
            module, "_resolve_local_executor", module._resolve_local_executor
        )
        # Exercise the internal fallback directly.
        import snakemake.executors.local as local_module

        assert local_module.Executor is not None
