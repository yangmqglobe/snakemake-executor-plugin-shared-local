"""Namespace (lease directory) resolution and permission handling.

Three coordination scopes are supported:

``user``
    Only this user's instances coordinate. Directory mode ``0700``.
``group``
    Instances of users sharing the primary group coordinate. Directory mode
    ``0770``.
``system``
    All users on the machine coordinate. Directory mode ``0777`` and *not*
    sticky, so any user can unlink abandoned entries.

Two permission details are correctness-critical and are therefore handled
explicitly rather than left to the process umask:

1. Directories created by :func:`os.makedirs` inherit the umask. With the common
   ``umask 077`` a ``0777`` directory would silently become ``0700`` and other
   users could not participate at all. We therefore ``chmod`` every created
   level.
2. Lease files that are unreadable to other users make their resource usage
   invisible, which would silently under-count and allow oversubscription. We
   therefore ``fchmod`` files to ``0644`` at creation time (see ``lease.py``).
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import os
import pwd
from dataclasses import dataclass
from pathlib import Path

from .config import (
    NAMESPACE_GROUP,
    NAMESPACE_SYSTEM,
    NAMESPACE_USER,
    NAMESPACES,
    ConfigError,
)

BASE_DIRNAME = "snakemake-shared-local"

# Preferred locations, in order. /dev/shm is a tmpfs shared by all users on this
# machine and is not backed by disk, which is exactly what we want for
# frequently-rewritten coordination state.
CANDIDATE_BASES = (
    "/dev/shm",
    None,  # placeholder for $XDG_RUNTIME_DIR
    "/var/tmp",
)

DIR_MODES = {
    NAMESPACE_USER: 0o700,
    NAMESPACE_GROUP: 0o770,
    NAMESPACE_SYSTEM: 0o777,
}


@dataclass
class Namespace:
    """A resolved lease directory and the policy attached to it."""

    mode: str
    path: Path
    dir_mode: int
    sticky: bool
    uid: int
    gid: int

    @property
    def is_shared(self) -> bool:
        """Whether instances of other users may legitimately appear here."""
        return self.mode in (NAMESPACE_GROUP, NAMESPACE_SYSTEM)

    @property
    def may_remove_foreign_entries(self) -> bool:
        """Whether we may unlink entries owned by other users.

        A sticky directory forbids this (``/tmp``, ``/dev/shm`` semantics), in
        which case abandoned entries are counted as dead but never deleted.
        """
        return not self.sticky

    def lease_path(self, name: str) -> Path:
        return self.path / name


def _base_candidates() -> list[Path]:
    bases: list[Path] = []
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    for candidate in CANDIDATE_BASES:
        if candidate is None:
            if runtime_dir:
                bases.append(Path(runtime_dir))
            continue
        bases.append(Path(candidate))
    return bases


def _namespace_dirname(mode: str, uid: int, gid: int) -> str:
    if mode == NAMESPACE_USER:
        return f"u{uid}"
    if mode == NAMESPACE_GROUP:
        return f"g{gid}"
    return "shared"


def _ensure_dir(path: Path, mode: int, *, explicit: bool) -> None:
    """Create ``path`` (and parents) and enforce ``mode``.

    When ``explicit`` is set the user named the directory themselves, so we do
    not chmod it: they own that decision and it may be a deliberately strict
    location. We only verify it is usable.
    """
    if path.exists():
        if not path.is_dir():
            raise ConfigError(f"Lease path {path} exists but is not a directory.")
    else:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise ConfigError(f"Cannot create lease directory {path}: {err}") from err

    if explicit:
        return

    # Undo umask effects. The container directory stays traversable for everyone
    # so several namespace subdirectories can coexist; the namespace directory
    # itself gets the scope's mode. Only chmod what we own, since pre-existing
    # directories owned by root may legitimately reject this.
    for level, wanted in ((path.parent, 0o755), (path, mode)):
        if not level.exists() or not _owned_by_self(level):
            continue
        try:
            current = os.stat(level).st_mode & 0o7777
        except OSError:
            continue
        if current != wanted:
            try:
                os.chmod(level, wanted)
            except OSError:
                if level == path:
                    raise ConfigError(
                        f"Lease directory {level} has mode {current:04o} but "
                        f"{wanted:04o} is required for the '{path.name}' coordination "
                        f"scope. Ask an administrator to run: "
                        f"sudo chmod {wanted:04o} {level}"
                    ) from None

    _verify_dir_mode(path, mode)


def _verify_dir_mode(path: Path, mode: int) -> None:
    """Ensure the namespace directory really carries the required permissions.

    Some filesystems and ACL setups silently drop what ``chmod`` asked for, and a
    scope that *looks* configured while nobody can actually participate would
    defeat the whole point. Verification therefore depends on what the scope
    needs: shared scopes require the group/other bits, private scopes only
    require that we can use it.
    """
    try:
        actual = os.stat(path).st_mode & 0o7777
    except OSError as err:  # pragma: no cover - races
        raise ConfigError(f"Cannot stat lease directory {path}: {err}") from err

    owned = _owned_by_self(path)
    if owned:
        # We are authoritative: the mode must match exactly.
        if actual != mode:
            raise ConfigError(
                f"Lease directory {path} has mode {actual:04o}, but {mode:04o} is "
                f"required for the '{mode}' coordination scope. Adjust it with: "
                f"chmod {mode:04o} {path}"
            )
        return

    # Not ours (e.g. pre-created by an administrator). Only insist on the bits
    # that make coordination possible; a stricter mode than required is fine for
    # private scopes but fatal for shared ones.
    if actual & mode != mode:
        raise ConfigError(
            f"Lease directory {path} is owned by uid {os.stat(path).st_uid} and has "
            f"mode {actual:04o}, which lacks the {mode:04o} permissions required for "
            f"the '{mode}' coordination scope. Ask an administrator to run: "
            f"sudo chmod {mode:04o} {path}"
        )


def _owned_by_self(path: Path) -> bool:
    try:
        return os.stat(path).st_uid == os.getuid()
    except OSError:  # pragma: no cover - races
        return False


def resolve_namespace(
    mode: str,
    *,
    explicit_dir: str | None = None,
    uid: int | None = None,
    gid: int | None = None,
) -> Namespace:
    """Resolve the lease directory for ``mode``.

    With ``explicit_dir`` the caller's path wins and its permissions are left
    untouched (only usability is verified).
    """
    if mode not in NAMESPACES:
        raise ConfigError(
            f"Invalid namespace {mode!r}: expected one of {', '.join(NAMESPACES)}."
        )

    uid = os.getuid() if uid is None else uid
    gid = os.getgid() if gid is None else gid

    dir_mode = DIR_MODES[mode]

    if explicit_dir:
        path = Path(os.path.expanduser(os.path.expandvars(explicit_dir))).absolute()
        _ensure_dir(path, dir_mode, explicit=True)
        if not os.access(path, os.W_OK | os.X_OK):
            raise ConfigError(f"Lease directory {path} is not writable by this user.")
        return Namespace(
            mode=mode,
            path=path,
            dir_mode=dir_mode,
            sticky=_is_sticky(path),
            uid=uid,
            gid=gid,
        )

    errors: list[str] = []
    for base in _base_candidates():
        path = base / BASE_DIRNAME / _namespace_dirname(mode, uid, gid)
        try:
            _ensure_dir(path, dir_mode, explicit=False)
        except ConfigError as err:
            errors.append(str(err))
            continue
        if not os.access(path, os.W_OK | os.X_OK):
            errors.append(f"Lease directory {path} is not writable by this user.")
            continue
        return Namespace(
            mode=mode,
            path=path,
            dir_mode=dir_mode,
            sticky=_is_sticky(path),
            uid=uid,
            gid=gid,
        )

    raise ConfigError(
        "Could not establish a usable lease directory. Tried:\n  "
        + "\n  ".join(errors)
        + "\nUse --shared-local-dir to point at a writable directory."
    )


def _is_sticky(path: Path) -> bool:
    try:
        return bool(os.stat(path).st_mode & 0o1000)
    except OSError:  # pragma: no cover - races
        return False


def describe_identity() -> dict[str, object]:
    """Return identifying information about this user for diagnostics."""
    try:
        gecos = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:  # pragma: no cover - unusual NSS setups
        gecos = str(os.getuid())
    return {"uid": os.getuid(), "gid": os.getgid(), "user": gecos}
