"""Configuration loading and precedence handling for the shared-local plugin.

Precedence (highest first):
    1. Explicit CLI values
    2. Configuration file
    3. Environment variables (handled by Snakemake's plugin settings layer)
    4. Auto-detection

Because Snakemake hands the plugin a fully populated ``ExecutorSettings``
instance (its dataclass defaults already baked in), we cannot distinguish
"user passed the default explicitly" from "user passed nothing". To keep the
precedence rules above meaningful, every overridable setting is declared as a
tri-state ``Literal["auto", "on", "off"]`` or as ``Optional[...]`` whose default is
``None``/``"auto"``; an "unset" sentinel is then resolved against the config file.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTO = "auto"

DEFAULT_CONFIG_PATH = "~/.config/snakemake/shared-local/config.json"

# Namespace modes.
NAMESPACE_USER = "user"
NAMESPACE_GROUP = "group"
NAMESPACE_SYSTEM = "system"
NAMESPACES = (NAMESPACE_USER, NAMESPACE_GROUP, NAMESPACE_SYSTEM)

# Names of the tri-state settings that resolve through AUTO_ON_OFF.
# ``require_mem`` belongs here: without it the key is not in VALID_KEYS, so a
# config file setting it would be reported as unknown and silently ignored.
TRI_STATE_KEYS = (
    "numa_bind",
    "cgroup",
    "record_rule",
    "verbose_leases",
    "gc_report",
    "require_mem",
)

TRUEISH = {"on", "true", "yes", "1", "enabled"}
FALSEISH = {"off", "false", "no", "0", "disabled"}

VALID_KEYS = {
    "namespace",
    "dir",
    "pool_cores",
    "pool_mem",
    "pool_conflict",
    "instance_weight",
    "aging_seconds",
    "aging_step",
    "admit_timeout",
    "poll_interval",
    "config",
    *TRI_STATE_KEYS,
}


class ConfigError(Exception):
    """Raised for unusable configuration input."""


@dataclass
class ConfigWarning:
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def parse_tri_state(value: Any, *, key: str) -> bool | None:
    """Resolve a tri-state value into ``True``/``False``/``None``.

    ``None`` means "unset" (caller falls back to the next precedence level).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == AUTO:
        return None
    if text in TRUEISH:
        return True
    if text in FALSEISH:
        return False
    raise ConfigError(
        f"Invalid value {value!r} for {key!r}: expected one of "
        f"'auto', 'on', 'off' (or true/false)."
    )


def parse_size_to_mib(value: Any, *, key: str) -> int | None:
    """Parse ``"500G"``/``"400000"``/``400000`` into MiB (int).

    Plain numbers are interpreted as MiB, matching Snakemake's ``mem_mb``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"Invalid value {value!r} for {key!r}: expected a size.")
    if isinstance(value, int | float):
        if value <= 0:
            raise ConfigError(f"Value for {key!r} must be positive, got {value!r}.")
        return int(value)

    text = str(value).strip().replace("_", "").replace(" ", "")
    if not text:
        return None

    units = {
        "k": 1 / 1024,
        "kb": 1 / 1024,
        "kib": 1 / 1024,
        "m": 1,
        "mb": 1,
        "mib": 1,
        "g": 1024,
        "gb": 1024,
        "gib": 1024,
        "t": 1024**2,
        "tb": 1024**2,
        "tib": 1024**2,
    }
    lowered = text.lower()
    for suffix in sorted(units, key=len, reverse=True):
        if lowered.endswith(suffix):
            number = lowered[: -len(suffix)]
            try:
                amount = float(number)
            except ValueError as err:
                raise ConfigError(
                    f"Invalid size {value!r} for {key!r}: cannot parse {number!r}."
                ) from err
            if amount <= 0:
                raise ConfigError(f"Value for {key!r} must be positive, got {value!r}.")
            return int(amount * units[suffix])

    try:
        amount = float(lowered)
    except ValueError as err:
        raise ConfigError(
            f"Invalid size {value!r} for {key!r}: expected e.g. "
            "'500G', '400000' or 400000."
        ) from err
    if amount <= 0:
        raise ConfigError(f"Value for {key!r} must be positive, got {value!r}.")
    return int(amount)


def _coerce_float(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"Invalid number {value!r} for {key!r}.") from err
    if result <= 0:
        raise ConfigError(f"Value for {key!r} must be positive, got {value!r}.")
    return result


def _coerce_int(value: Any, *, key: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"Invalid value {value!r} for {key!r}: expected an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as err:
        raise ConfigError(
            f"Invalid value {value!r} for {key!r}: expected an integer."
        ) from err
    if result < minimum:
        raise ConfigError(f"Value for {key!r} must be >= {minimum}, got {value!r}.")
    return result


def load_config_file(
    path: str | os.PathLike[str] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Load a JSON config file.

    Returns ``(data, warnings)``. A missing file yields an empty mapping and no
    warning; unreadable or malformed files raise :class:`ConfigError` because
    silently ignoring them would change resource accounting in surprising ways.
    """
    if path is None:
        return {}, []
    expanded = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not expanded.exists():
        return {}, []
    try:
        raw = expanded.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigError(f"Cannot read config file {expanded}: {err}") from err
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ConfigError(f"Config file {expanded} is not valid JSON: {err}") from err
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Config file {expanded} must contain a JSON object, "
            f"got {type(data).__name__}."
        )

    warnings: list[str] = []
    unknown = sorted(set(data) - VALID_KEYS)
    if unknown:
        warnings.append(
            f"Unknown keys in config file {expanded} are ignored: {', '.join(unknown)}."
        )
    return {k: v for k, v in data.items() if k in VALID_KEYS}, warnings


@dataclass
class ResolvedSettings:
    """Fully resolved plugin settings after applying precedence rules."""

    namespace: str
    dir: str | None
    pool_cores: int | None
    pool_mem_mb: int | None
    pool_conflict: str
    instance_weight: float
    aging_seconds: int
    aging_step: int
    admit_timeout: int
    poll_interval: float
    numa_bind: bool | None
    cgroup: bool | None
    record_rule: bool | None
    verbose_leases: bool
    gc_report: bool
    require_mem: bool
    config_path: str
    warnings: list[str]


def resolve_settings(settings: Any) -> ResolvedSettings:
    """Merge an ``ExecutorSettings`` instance with its config file.

    Precedence is: explicit CLI value, then config file, then built-in default.
    This only works because every overridable field in ``ExecutorSettings``
    defaults to the ``AUTO`` sentinel (or ``None``): Snakemake hands us an already
    populated dataclass, so a field's dataclass default is indistinguishable from
    a user-supplied value. A sentinel is what makes "unset" representable.
    """
    config_path = getattr(settings, "config", None) or DEFAULT_CONFIG_PATH
    file_data, warnings = load_config_file(config_path)

    def is_unset(value: Any) -> bool:
        if value is None:
            return True
        return isinstance(value, str) and value.strip().lower() == AUTO

    def pick(name: str, default: Any = None) -> Any:
        """Return the effective raw value for ``name`` honouring precedence."""
        cli_value = getattr(settings, name, None)
        if not is_unset(cli_value):
            return cli_value
        if name in file_data and not is_unset(file_data[name]):
            return file_data[name]
        return default

    namespace = pick("namespace", NAMESPACE_USER)
    if namespace not in NAMESPACES:
        raise ConfigError(
            f"Invalid namespace {namespace!r}: expected one of {', '.join(NAMESPACES)}."
        )

    pool_conflict = pick("pool_conflict", "error")
    if pool_conflict not in ("error", "warn"):
        raise ConfigError(
            f"Invalid pool_conflict {pool_conflict!r}: expected 'error' or 'warn'."
        )

    record_rule = parse_tri_state(pick("record_rule"), key="record_rule")
    if record_rule is None:
        # In the system-wide namespace other users can read lease files, so
        # avoid disclosing rule names by default. Callers may force either way.
        record_rule = namespace != NAMESPACE_SYSTEM

    # Plugin defaults. `auto` means "no explicit choice", so it resolves to the
    # documented default rather than a hidden behaviour change. In particular
    # cgroup enforcement is OFF by default: silently switching jobs to kernel-
    # enforced limits (where an over-allocation becomes an OOM kill) would change
    # the failure mode for users who never asked for it.
    cgroup_request = parse_tri_state(pick("cgroup"), key="cgroup")
    cgroup = False if cgroup_request is None else cgroup_request

    return ResolvedSettings(
        namespace=namespace,
        dir=pick("dir"),
        pool_cores=_coerce_int(pick("pool_cores"), key="pool_cores", minimum=1),
        pool_mem_mb=parse_size_to_mib(pick("pool_mem"), key="pool_mem"),
        pool_conflict=pool_conflict,
        instance_weight=_coerce_float(pick("instance_weight"), key="instance_weight")
        or 1.0,
        aging_seconds=_coerce_int(pick("aging_seconds"), key="aging_seconds", minimum=0)
        if not is_unset(pick("aging_seconds"))
        else 60,
        aging_step=_coerce_int(pick("aging_step"), key="aging_step", minimum=0)
        if not is_unset(pick("aging_step"))
        else 1,
        admit_timeout=_coerce_int(pick("admit_timeout"), key="admit_timeout", minimum=0)
        if not is_unset(pick("admit_timeout"))
        else 0,
        poll_interval=_coerce_float(pick("poll_interval"), key="poll_interval")
        if not is_unset(pick("poll_interval"))
        else 0.05,
        numa_bind=parse_tri_state(pick("numa_bind"), key="numa_bind"),
        cgroup=cgroup,
        record_rule=record_rule,
        verbose_leases=bool(
            parse_tri_state(pick("verbose_leases"), key="verbose_leases")
        ),
        gc_report=bool(parse_tri_state(pick("gc_report"), key="gc_report")),
        require_mem=bool(parse_tri_state(pick("require_mem"), key="require_mem")),
        config_path=str(config_path),
        warnings=warnings,
    )
