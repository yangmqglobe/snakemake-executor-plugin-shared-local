"""POSIX-only facilities, guarded so a missing module cannot break startup.

Snakemake discovers executor plugins by scanning ``sys.path`` with
``pkgutil.iter_modules()`` and importing every ``snakemake_executor_plugin_*``
package it finds. That scan runs on *every* Snakemake invocation, on every
platform, and the registry does not catch ``ImportError``. An unguarded
``import fcntl`` at module scope therefore did not merely make this plugin
unavailable on Windows -- it aborted Snakemake itself, for every user, including
those who never asked for this executor::

    ImportError: No module named 'fcntl'

``fcntl``, ``pwd`` and ``os.getuid`` are the POSIX-only facilities this plugin
needs: liveness detection is a ``flock``, and the ``group``/``system`` scopes
compare Unix ownership. None has a Windows equivalent.

So the imports below are resolved defensively. When a module is missing, the name
is bound to a stub that raises a clear :class:`WorkflowError` only at the moment
the feature is actually used -- discovery succeeds, and an unsupported platform
gets an explanatory message instead of a traceback from deep inside Snakemake.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import importlib
import os
from types import ModuleType
from typing import Any

from snakemake_interface_common.exceptions import WorkflowError

_MESSAGE = (
    "The 'shared-local' executor plugin requires a POSIX system (Linux or macOS), "
    "but this platform does not provide {missing}. The plugin coordinates "
    "concurrent local Snakemake instances through flock-based leases and Unix "
    "ownership checks, which have no equivalent on Windows. Use Snakemake's "
    "built-in local executor instead."
)


class _MissingModule(ModuleType):
    """Stand-in for a POSIX-only module on an unsupported platform.

    Any attribute access -- ``fcntl.flock``, ``pwd.getpwuid`` -- raises rather
    than returning ``None``, so the failure surfaces where the plugin is used
    instead of as a confusing ``AttributeError`` somewhere downstream.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._missing_name = name

    def __getattr__(self, attribute: str) -> Any:
        raise WorkflowError(
            _MESSAGE.format(missing=f"the {self._missing_name!r} module")
        )


def _load(name: str) -> ModuleType:
    """Import ``name``, substituting a raising stub when it is unavailable."""
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - Windows only
        return _MissingModule(name)


#: ``fcntl`` on POSIX; a stub that raises ``WorkflowError`` elsewhere.
fcntl: Any = _load("fcntl")

#: ``pwd`` on POSIX; a stub that raises ``WorkflowError`` elsewhere.
pwd: Any = _load("pwd")


def is_posix() -> bool:
    """Whether every POSIX facility this plugin relies on is present.

    Deliberately checks for :class:`_MissingModule` rather than ``ModuleType``:
    the stub *is* a ``ModuleType`` subclass, so a plain isinstance check on the
    module type would report a stubbed-out platform as supported.
    """
    return not isinstance(fcntl, _MissingModule) and hasattr(os, "getuid")


def require_posix() -> None:
    """Fail with an explanatory error on a platform the plugin cannot serve.

    Called from the executor's constructor and from the coordination-layer entry
    points, so both ``--executor shared-local`` and direct API use report the
    real problem. Kept separate from :data:`is_posix` so callers that only need
    to branch (tests, documentation) can avoid raising.
    """
    if is_posix():
        return
    missing = []
    if isinstance(fcntl, _MissingModule):
        missing.append("'fcntl'")
    if not hasattr(os, "getuid"):
        missing.append("'os.getuid'")
    raise WorkflowError(
        _MESSAGE.format(missing=" or ".join(missing) or "required POSIX facilities")
    )
