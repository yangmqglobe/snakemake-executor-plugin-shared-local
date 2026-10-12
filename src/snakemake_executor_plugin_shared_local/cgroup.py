"""Optional kernel-enforced limits via cgroup v2.

Leases are advisory: an instance that ignores the protocol is not stopped, and a
job that allocates far more memory than it declared is not stopped either. When
kernel enforcement is wanted, this module wraps a job in a systemd user scope so
the kernel caps CPU and memory for real.

Measured behaviour on the development machine (cgroup v2, user delegation active):

* ``CPUQuota=25%`` really throttles: 24.9M loop iterations in 3s became 4.6M.
* ``MemoryMax=200M`` alone does **not** enforce: a 600 MiB allocation survived
  because the cgroup could swap. Adding ``MemorySwapMax=0`` made the same program
  die with SIGKILL (exit status 137, ``oom_kill`` reported).
* ``AllowedCPUs`` silently did nothing, because ``cpuset`` is not delegated on
  this host and writing ``+cpuset`` to ``cgroup.subtree_control`` is denied. CPU
  placement is therefore handled by :mod:`numa` instead, and this module never
  relies on cpuset.
* ``systemd-run --user`` needs the user D-Bus session. Without it (e.g. under a
  cron job) it fails with "Failed to connect to bus". That must degrade to a
  warning rather than breaking the workflow.

Given the above, enforcement here is strictly best-effort and opt-in.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass

SYSTEMD_RUN = "systemd-run"
# How long to wait for the one-time capability probe.
PROBE_TIMEOUT = 20

# Fallback logger so warnings raised before Snakemake's logger is threaded through
# (and in tests) still reach the user instead of vanishing.
_MODULE_LOGGER = logging.getLogger(__name__)


@dataclass
class CgroupSupport:
    """Result of probing whether cgroup v2 enforcement is usable here."""

    available: bool
    reason: str = ""
    systemd_run: str | None = None

    def describe(self) -> str:
        if self.available:
            return f"cgroup enforcement available via {self.systemd_run}"
        return f"cgroup enforcement unavailable: {self.reason}"


def _has_user_session() -> tuple[bool, str]:
    """Check for the environment systemd-run --user needs."""
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return False, "DBUS_SESSION_BUS_ADDRESS is not set"
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and not os.path.isdir(runtime_dir):
        return False, f"XDG_RUNTIME_DIR {runtime_dir} does not exist"
    return True, ""


def _controllers_path() -> str | None:
    """Return the cgroup v2 path of this process, if cgroup v2 is in use."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    return parts[2]
    except OSError:
        return None
    return None


def probe(logger: logging.Logger | None = None) -> CgroupSupport:
    """Determine whether cgroup v2 enforcement can be used.

    Any missing prerequisite is reported rather than raised: callers treat this
    as an optional feature and fall back to advisory-only coordination.
    """
    if os.name != "posix":
        return CgroupSupport(False, "not a POSIX platform")

    if _controllers_path() is None:
        return CgroupSupport(False, "cgroup v2 unified hierarchy not detected")

    binary = shutil.which(SYSTEMD_RUN)
    if binary is None:
        return CgroupSupport(False, f"{SYSTEMD_RUN} not found in PATH")

    session_ok, session_reason = _has_user_session()
    if not session_ok:
        return CgroupSupport(False, session_reason)

    # A real invocation is the only reliable test: D-Bus may be present yet the
    # user manager unreachable, and properties may be rejected by an older
    # systemd. Probe with the smallest possible scope.
    try:
        result = subprocess.run(
            [
                binary,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--property=CPUQuota=1%",
                "--property=MemorySwapMax=0",
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return CgroupSupport(False, f"probe failed: {err}")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit status {result.returncode}"
        return CgroupSupport(False, f"probe failed: {reason}")

    if logger is not None:
        logger.debug("cgroup v2 enforcement probe succeeded.")
    return CgroupSupport(True, "", systemd_run=binary)


def build_scope_command(
    base_command: str,
    *,
    threads: int,
    pool_cores: int,
    mem_mib: int | None,
    systemd_run: str = SYSTEMD_RUN,
) -> list[str]:
    """Build the ``systemd-run`` wrapper for one job command.

    ``threads`` and ``pool_cores`` are turned into a percentage quota of the
    pool. ``mem_mib`` maps to ``MemoryMax``; when it is unknown, no memory limit
    is imposed rather than inventing one. ``MemorySwapMax=0`` is always included:
    without it the memory cap is not actually enforced (see module docstring).
    """
    args = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit=shared-local-{os.getpid()}-{os.urandom(4).hex()}",
    ]

    if pool_cores > 0 and threads > 0:
        quota = max(1, math.ceil(threads / pool_cores * 100))
        args.append(f"--property=CPUQuota={quota}%")

    if mem_mib is not None and mem_mib > 0:
        args.append(f"--property=MemoryMax={int(mem_mib)}M")
        # Mandatory companion: a memory cap that may swap is not a cap.
        args.append("--property=MemorySwapMax=0")

    args.extend(["/bin/sh", "-c", base_command])
    return args


@dataclass
class CgroupResult:
    """Outcome of wrapping a command."""

    command: list[str]
    enforced: bool
    warning: str = ""


def wrap_command(
    base_command: str,
    *,
    enabled: bool,
    support: CgroupSupport,
    threads: int,
    pool_cores: int,
    mem_mib: int | None,
    logger: logging.Logger | None = None,
) -> CgroupResult:
    """Wrap ``base_command`` in a cgroup scope when possible.

    Never raises: if enforcement is requested but unavailable, the command is
    returned unwrapped together with a warning, because failing the workflow
    over an optional hardening feature would be worse than running without it.
    """
    if not enabled:
        return CgroupResult(command=["/bin/sh", "-c", base_command], enforced=False)

    if not support.available or support.systemd_run is None:
        warning = (
            f"cgroup enforcement was requested but is unavailable ({support.reason}); "
            "running with advisory coordination only."
        )
        # Always emit through some logger: silently dropping an opt-in hardening
        # warning would leave the user believing limits are enforced.
        (logger or _MODULE_LOGGER).warning(warning)
        return CgroupResult(
            command=["/bin/sh", "-c", base_command], enforced=False, warning=warning
        )

    command = build_scope_command(
        base_command,
        threads=threads,
        pool_cores=pool_cores,
        mem_mib=mem_mib,
        systemd_run=support.systemd_run,
    )
    return CgroupResult(command=command, enforced=True)
