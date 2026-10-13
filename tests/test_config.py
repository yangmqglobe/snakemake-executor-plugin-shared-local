"""Tests for configuration parsing and precedence."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local.config import (
    ConfigError,
    parse_size_to_mib,
    parse_tri_state,
    resolve_settings,
)


@dataclass
class FakeSettings:
    """Stand-in for ExecutorSettings with the fields resolve_settings reads.

    Defaults mirror the real class: an ``auto``/``None`` sentinel means "unset", so
    the config file can still supply a value. Using concrete defaults here would
    make the config file unreachable, which is precisely the bug these tests guard
    against.
    """

    namespace: str = "auto"
    dir: str | None = None
    pool_cores: int | None = None
    pool_mem: str | None = None
    pool_conflict: str = "auto"
    instance_weight: float | None = None
    aging_seconds: int | None = None
    aging_step: int | None = None
    admit_timeout: int | None = None
    poll_interval: float | None = None
    numa_bind: str = "auto"
    cgroup: str = "auto"
    require_mem: str = "auto"
    record_rule: str = "auto"
    gc_report: str = "auto"
    verbose_leases: str = "auto"
    config: str | None = None


def test_parse_tri_state_values():
    assert parse_tri_state("auto", key="x") is None
    assert parse_tri_state(None, key="x") is None
    assert parse_tri_state("on", key="x") is True
    assert parse_tri_state("off", key="x") is False
    assert parse_tri_state(True, key="x") is True
    assert parse_tri_state("YES", key="x") is True
    assert parse_tri_state("disabled", key="x") is False


def test_parse_tri_state_rejects_garbage():
    with pytest.raises(ConfigError, match="Invalid value"):
        parse_tri_state("maybe", key="x")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("500G", 500 * 1024),
        ("500GB", 500 * 1024),
        ("2T", 2 * 1024 * 1024),
        ("1024M", 1024),
        ("100", 100),
        (100, 100),
        ("1.5G", 1536),
    ],
)
def test_parse_size_to_mib(text, expected):
    assert parse_size_to_mib(text, key="mem") == expected


def test_parse_size_to_mib_none_and_invalid():
    assert parse_size_to_mib(None, key="mem") is None
    with pytest.raises(ConfigError):
        parse_size_to_mib("lots", key="mem")
    with pytest.raises(ConfigError):
        parse_size_to_mib(0, key="mem")
    with pytest.raises(ConfigError):
        parse_size_to_mib(-5, key="mem")


def test_resolve_defaults_without_config(tmp_path: Path):
    settings = FakeSettings(config=str(tmp_path / "missing.json"))
    resolved = resolve_settings(settings)
    assert resolved.namespace == "user"
    assert resolved.pool_cores is None
    assert resolved.pool_mem_mb is None
    assert resolved.pool_conflict == "error"
    assert resolved.numa_bind is None  # auto: enable on multi-node machines
    assert resolved.cgroup is False  # auto: plugin default is OFF
    assert resolved.record_rule is True  # auto resolves per-namespace
    assert resolved.warnings == []


class TestCgroupDefault:
    """cgroup enforcement must be opt-in.

    Turning it on implicitly would change how jobs fail (an over-allocation
    becomes an OOM kill instead of a slowdown), which is not something a
    coordination plugin should do without being asked.
    """

    def test_default_is_off(self, tmp_path: Path):
        resolved = resolve_settings(FakeSettings(config=str(tmp_path / "missing.json")))
        assert resolved.cgroup is False

    def test_explicit_on_is_honoured(self, tmp_path: Path):
        settings = FakeSettings(config=str(tmp_path / "missing.json"), cgroup="on")
        assert resolve_settings(settings).cgroup is True

    def test_explicit_off_is_honoured(self, tmp_path: Path):
        settings = FakeSettings(config=str(tmp_path / "missing.json"), cgroup="off")
        assert resolve_settings(settings).cgroup is False

    def test_config_file_can_enable_it(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"cgroup": "on"}))
        resolved = resolve_settings(FakeSettings(config=str(config)))
        assert resolved.cgroup is True

    def test_cli_off_beats_config_file_on(self, tmp_path: Path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"cgroup": "on"}))
        settings = FakeSettings(config=str(config), cgroup="off")
        assert resolve_settings(settings).cgroup is False

    def test_field_default_is_auto_not_on(self):
        """The dataclass default feeds `pick`, so it must stay a sentinel."""
        from snakemake_executor_plugin_shared_local import ExecutorSettings

        assert ExecutorSettings.__dataclass_fields__["cgroup"].default == "auto"

    def test_help_text_documents_off(self):
        from snakemake_executor_plugin_shared_local import ExecutorSettings

        help_text = ExecutorSettings.__dataclass_fields__["cgroup"].metadata["help"]
        assert "OFF" in help_text or "off" in help_text


def test_numa_default_is_auto(tmp_path: Path):
    """NUMA placement is a default-on optimisation on multi-node machines."""
    resolved = resolve_settings(FakeSettings(config=str(tmp_path / "missing.json")))
    assert resolved.numa_bind is None


def test_config_file_supplies_values(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"pool_cores": 48, "pool_mem": "300G", "namespace": "group"})
    )
    resolved = resolve_settings(FakeSettings(config=str(config)))
    assert resolved.pool_cores == 48
    assert resolved.pool_mem_mb == 300 * 1024
    assert resolved.namespace == "group"


def test_cli_wins_over_config_file(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"pool_cores": 48, "age_seconds": 5, "aging_seconds": 5})
    )
    resolved = resolve_settings(FakeSettings(config=str(config), pool_cores=8))
    assert resolved.pool_cores == 8  # explicit CLI beats the file
    assert resolved.aging_seconds == 5  # untouched field comes from the file


def test_auto_falls_through_to_config_file(tmp_path: Path):
    """'auto' must mean unset, so the config file can fill it in."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"numa_bind": "off", "cgroup": "on"}))
    resolved = resolve_settings(FakeSettings(config=str(config), numa_bind="auto"))
    assert resolved.numa_bind is False
    assert resolved.cgroup is True


def test_record_rule_defaults_off_in_system_scope(tmp_path: Path):
    """Other users can read lease files in the system scope, so hide rule names."""
    system = resolve_settings(
        FakeSettings(namespace="system", config=str(tmp_path / "x"))
    )
    assert system.record_rule is False
    user = resolve_settings(FakeSettings(namespace="user", config=str(tmp_path / "x")))
    assert user.record_rule is True


def test_invalid_namespace_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="Invalid namespace"):
        resolve_settings(FakeSettings(namespace="galaxy", config=str(tmp_path / "x")))


def test_invalid_pool_conflict_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="Invalid pool_conflict"):
        resolve_settings(
            FakeSettings(pool_conflict="ignore", config=str(tmp_path / "x"))
        )


def test_malformed_config_raises(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        resolve_settings(FakeSettings(config=str(config)))


def test_non_object_config_raises(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text("[1, 2, 3]")
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        resolve_settings(FakeSettings(config=str(config)))


def test_unknown_keys_warn_but_do_not_fail(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pool_cores": 4, "typo_key": 1}))
    resolved = resolve_settings(FakeSettings(config=str(config)))
    assert resolved.pool_cores == 4
    assert any("typo_key" in warning for warning in resolved.warnings)


def test_require_mem_is_settable_from_the_config_file(tmp_path: Path):
    """Regression: ``require_mem`` was absent from VALID_KEYS, so a config file
    setting it was reported as an unknown key and silently ignored."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"require_mem": "on"}))
    resolved = resolve_settings(FakeSettings(config=str(config)))
    assert resolved.require_mem is True
    assert not any("require_mem" in warning for warning in resolved.warnings)


def test_every_tri_state_setting_is_a_valid_config_key():
    """A tri-state setting that is not in VALID_KEYS cannot be configured."""
    from snakemake_executor_plugin_shared_local.config import (
        TRI_STATE_KEYS,
        VALID_KEYS,
    )

    assert set(TRI_STATE_KEYS) <= VALID_KEYS


def test_config_path_expands_user_and_vars(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SL_TEST_DIR", str(tmp_path))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pool_cores": 7}))
    resolved = resolve_settings(FakeSettings(config="$SL_TEST_DIR/config.json"))
    assert resolved.pool_cores == 7
