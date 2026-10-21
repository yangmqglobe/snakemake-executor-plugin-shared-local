"""flock-based resource leases.

Design
------
There is no daemon. Coordination state lives in a directory of small JSON files,
one per admitted job (or group job). The rules are:

* The **owner** of a lease holds an exclusive ``flock`` on it for the whole job.
* The kernel releases that lock when the owning process dies *for any reason*,
  including ``SIGKILL`` and power loss. Liveness therefore needs no heartbeat and
  no timeout guessing.
* A **waiter** tries ``LOCK_EX | LOCK_NB`` on every lease file. Success means the
  former owner is gone, so the entry describes nobody and its resources count as
  zero. Failure means the entry is live.

This makes liveness detection and crash recovery the same mechanism: a stale
lease is simply one whose lock can be taken. Consequently, counting stays correct
even when a stale file *cannot be deleted* - which is the normal situation in a
sticky, shared directory such as ``/dev/shm``.

Two further details matter for correctness:

* Files are published atomically (write temp, ``flock``, ``fsync``, ``rename``)
  so a waiter never observes a half-written entry.
* Files are explicitly ``fchmod``-ed to ``0644``. Under ``umask 077`` the default
  ``0600`` would make them unreadable to other users, whose usage would then be
  invisible and oversubscription would follow silently.
* A file we may not write is opened read-only instead (see
  :func:`_open_for_lock`). ``flock`` needs no write permission, so a peer's
  ``0644`` lock file is still a working mutex; requiring ``O_RDWR`` made shared
  scopes usable only by whoever created the files first.
"""

__author__ = "ymq"
__copyright__ = "Copyright 2026, ymq"
__email__ = "ymq@gzfezx.com"
__license__ = "MIT"

import contextlib
import errno
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .config import ConfigError
from .namespace import Namespace
from .policy import Waiter, fits, should_yield
from .pool import MEMORY_DIMENSION, canonicalize_pool
from .posix_support import fcntl

SCHEMA_VERSION = 1

ADMIT_LOCK = "admit.lock"
POOL_FILE = "pool.json"
INSTANCE_SUFFIX = ".inst"
PENDING_SUFFIX = ".pending"
LEASE_SUFFIX = ".lease"

FILE_MODE = 0o644
# Flags that must never be part of a resource request: they are internal
# scheduler bookkeeping rather than consumable capacity.
INTERNAL_RESOURCES = frozenset({"_job_count", "_nodes", "_cores"})
# Resources derived from others; counting them as well would double-count.
DERIVED_SUFFIXES = ("_mib",)


class LeaseError(ConfigError):
    """Raised when a lease cannot be honoured."""


class AdmissionImpossible(LeaseError):
    """Raised when a request can never fit into the pool."""


@dataclass
class LeaseEntry:
    """A live lease as read from disk."""

    path: Path
    data: Mapping[str, Any]
    # True when this process holds the flock, i.e. nobody else can.
    locked_by_us: bool = False

    @property
    def resources(self) -> Mapping[str, int]:
        raw = self.data.get("resources") or {}
        return {k: int(v) for k, v in raw.items() if isinstance(v, int | float)}

    @property
    def instance_id(self) -> str:
        return str(self.data.get("instance_id", "?"))


@dataclass
class LeaseHandle:
    """An admitted lease held by this process."""

    path: Path
    fd: int
    resources: Mapping[str, int]


def normalize_resources(
    resources: Mapping[str, Any], *, threads: int, with_mem: bool = True
) -> dict[str, int]:
    """Convert a Snakemake resource mapping into lease dimensions.

    ``threads`` becomes the ``cores`` dimension, matching the pool's core budget.
    Derived/aliased entries (``mem_mb``/``mem_mib``/``mem``) and internal scheduler
    keys are dropped so a single request is never counted twice.

    Memory is emitted as :data:`MEMORY_DIMENSION`, the same name ``Pool.as_dict()``
    publishes. Both sides must agree, because ``policy.fits()`` treats a dimension
    absent from the pool as unconstrained - the mismatch that once silently
    disabled memory arbitration entirely.
    """
    request: dict[str, int] = {"cores": max(1, int(threads))}
    for name, value in resources.items():
        if name in INTERNAL_RESOURCES:
            continue
        if name.endswith(DERIVED_SUFFIXES):
            continue
        if name in (MEMORY_DIMENSION, "mem", "mem_mib") and not with_mem:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value <= 0:
                continue
            dimension = MEMORY_DIMENSION if name in (MEMORY_DIMENSION, "mem") else name
            request[dimension] = int(value)
        elif isinstance(value, str) and value.isdigit():
            dimension = MEMORY_DIMENSION if name in (MEMORY_DIMENSION, "mem") else name
            request[dimension] = int(value)
    return request


class LeaseManager:
    """Coordinates resource leases for one Snakemake instance."""

    def __init__(
        self,
        namespace: Namespace,
        pool: Mapping[str, int],
        *,
        instance_id: str | None = None,
        weight: float = 1.0,
        aging_seconds: int = 0,
        aging_step: int = 0,
        pool_conflict: str = "error",
        verbose: bool = False,
        logger: Any = None,
        poll_interval: float = 0.05,
        admit_timeout: int = 0,
        gc: bool = False,
    ) -> None:
        self.namespace = namespace
        # Canonicalize on the way in so an alias spelling (a pool declaration
        # written by an older plugin, or a hand-written ``mem_mib``) still lands on
        # the dimension job requests are expressed in.
        self.pool = canonicalize_pool(pool)
        self.instance_id = (
            instance_id
            or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:4]}"
        )
        self.weight = float(weight)
        self.aging_seconds = int(aging_seconds)
        self.aging_step = int(aging_step)
        self.pool_conflict = pool_conflict
        self.verbose = verbose
        self.logger = logger
        self.poll_interval = float(poll_interval)
        self.admit_timeout = int(admit_timeout)
        self.gc = gc
        self.warnings: list[str] = []

        self._held: dict[str, LeaseHandle] = {}
        self._lock = threading.RLock()
        self._instance_fd: int | None = None

    # ------------------------------------------------------------------ helpers

    def _debug(self, message: str) -> None:
        if self.verbose and self.logger is not None:
            # A misbehaving logger must never break coordination.
            with contextlib.suppress(Exception):
                self.logger.debug(message)

    def _info(self, message: str) -> None:
        if self.logger is not None:
            with contextlib.suppress(Exception):
                self.logger.info(message)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.logger is not None:
            with contextlib.suppress(Exception):
                self.logger.warning(message)

    def _path(self, name: str) -> Path:
        return self.namespace.path / name

    # ----------------------------------------------------------- instance claim

    def register(self) -> None:
        """Claim an instance slot so this process counts as an active instance.

        Holds a ``flock`` on ``<instance_id>.inst`` for our lifetime. Another
        process cannot claim the same id, and the claim disappears automatically
        if we die.
        """
        path = self._instance_path()
        fd = _open_locked(path)
        self._instance_fd = fd
        self._debug(f"Registered instance {self.instance_id} ({path}).")

    def unregister(self) -> None:
        if self._instance_fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self._instance_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._instance_fd)
        self._instance_fd = None
        # Remove our claim so the directory does not accumulate dead entries.
        # Safe because we just dropped the lock: if anything re-took it in
        # between, the unlink would fail or remove someone else's live claim, so
        # verify liveness by re-acquiring before deleting.
        try:
            probe = _try_lock_read(self._instance_path())
        except Exception:  # pragma: no cover
            probe = None
        if probe is not None:
            _unlock_close(probe)
            # A sticky directory or a race may make this fail; harmless either way.
            with contextlib.suppress(OSError):
                os.unlink(self._instance_path())

    def _instance_path(self) -> Path:
        return self._path(f"{self.instance_id}{INSTANCE_SUFFIX}")

    def active_instances(self) -> list[str]:
        """Return the ids of instances that currently hold their claim.

        A claim whose lock we *can* take has no live owner, so it is not active.
        The probing descriptor must be closed to avoid leaking one fd per call.
        """
        found: list[str] = []
        for path in self._scan(INSTANCE_SUFFIX):
            fd = _try_lock_read(path)
            if fd is None:
                # The claim is locked, so its owner is alive.
                found.append(path.name[: -len(INSTANCE_SUFFIX)])
            else:
                _unlock_close(fd)
        return found

    # ------------------------------------------------------------------- pool

    def negotiate_pool(self) -> None:
        """Publish/verify the pool declaration for shared namespaces.

        Several users must agree on the pool, otherwise each of them believes the
        machine is bigger than it is and the global cap is fiction. The first
        instance writes the file atomically; later ones verify or complain.
        """
        path = self._path(POOL_FILE)
        payload = {
            "version": SCHEMA_VERSION,
            "host": socket.gethostname(),
            "pool": dict(self.pool),
            "created_by": os.getuid(),
            "created_at": time.time(),
        }
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, FILE_MODE)
        except FileExistsError:
            existing = _read_json(path)
            # Compare canonicalized declarations: a file written by an older plugin
            # spells memory ``mem_mib``, which is the same pool as our ``mem_mb``.
            # Comparing raw dicts would turn an upgrade into a spurious abort.
            #
            # A file that exists but carries no usable ``pool`` mapping must still
            # count as a mismatch (yielding ``{}``), never as "nothing to compare":
            # skipping the check would let each instance run with a pool the other
            # never agreed to, which is exactly what this negotiation prevents. The
            # same applies to a declaration we cannot interpret at all.
            raw = existing.get("pool") if isinstance(existing, Mapping) else None
            try:
                on_disk = canonicalize_pool(raw) if isinstance(raw, Mapping) else {}
            except ConfigError as err:
                on_disk = {}
                unreadable = str(err)
            else:
                unreadable = None
            if existing is not None and on_disk != self.pool:
                shown = dict(raw) if isinstance(raw, Mapping) else raw
                message = (
                    f"Pool declaration mismatch in {path}: "
                    f"on disk {shown} vs configured {dict(self.pool)}. "
                    + (f"{unreadable} " if unreadable else "")
                    + "All instances sharing a namespace must agree on the pool, "
                    "otherwise the global limit is not enforced. Align "
                    "--shared-local-pool-cores/-mem, or use a separate namespace."
                )
                if self.pool_conflict == "error":
                    # Deliberate replacement: the FileExistsError is expected and
                    # would only add noise to a configuration error message.
                    raise LeaseError(message) from None
                self._warn(message)
            return
        except OSError as err:
            raise LeaseError(f"Cannot publish pool declaration {path}: {err}") from err

        try:
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.fchmod(fd, FILE_MODE)
        finally:
            os.close(fd)
        self._debug(f"Published pool declaration {payload['pool']} to {path}.")

    # ------------------------------------------------------------ accounting

    def _scan(self, suffix: str) -> list[Path]:
        try:
            names = os.listdir(self.namespace.path)
        except FileNotFoundError:
            return []
        except OSError as err:
            raise LeaseError(
                f"Cannot list lease directory {self.namespace.path}: {err}"
            ) from err
        return sorted(self.namespace.path / n for n in names if n.endswith(suffix))

    def snapshot(self) -> tuple[dict[str, int], list[str]]:
        """Return ``(used_resources, warnings)`` across all live leases.

        Stale entries count as zero. Unreadable entries are reported; the caller
        decides how severe that is (it is fatal in shared namespaces because
        silently under-counting would break the cap).
        """
        used: dict[str, int] = {}
        warnings: list[str] = []
        for path in self._scan(LEASE_SUFFIX):
            entry = _read_lease(path)
            if entry is None:
                # No lock held by anyone => dead. Counts as zero.
                self._maybe_collect(path)
                continue
            if isinstance(entry, _Unreadable):
                warnings.append(entry.message)
                continue
            for name, amount in entry.resources.items():
                used[name] = used.get(name, 0) + amount
        self._collect_dead_instances()
        return used, warnings

    def _collect_dead_instances(self) -> None:
        """Remove instance claims whose owner died, so the directory stays tidy."""
        for path in self._scan(INSTANCE_SUFFIX):
            if path.name == f"{self.instance_id}{INSTANCE_SUFFIX}":
                continue
            fd = _try_lock_read(path)
            if fd is None:
                # Lock held => owner alive; leave it alone.
                continue
            _unlock_close(fd)
            self._maybe_collect(path)

    def _maybe_collect(self, path: Path) -> None:
        """Delete a provably-dead lease when the directory permits it."""
        if not self.namespace.may_remove_foreign_entries:
            try:
                if os.stat(path).st_uid != os.getuid():
                    return
            except OSError:  # pragma: no cover - raced
                return
        try:
            path.unlink()
            self._debug(f"Collected stale lease {path.name}.")
        except FileNotFoundError:  # pragma: no cover - raced
            pass
        except PermissionError:
            # Sticky directory owned by someone else. Counting stays correct.
            self._debug(f"Stale lease {path.name} could not be removed (sticky dir).")
        except OSError:  # pragma: no cover
            pass

    def report_garbage(self) -> list[str]:
        """Return the names of stale lease files currently on disk."""
        stale: list[str] = []
        for path in self._scan(LEASE_SUFFIX):
            if _read_lease(path) is None:
                stale.append(path.name)
        return stale

    # -------------------------------------------------------------- admission

    def acquire(
        self,
        key: str,
        request: Mapping[str, int],
        *,
        priority: int = 0,
        rule: str | None = None,
    ) -> LeaseHandle:
        """Block until ``request`` fits into the pool, then hold a lease.

        ``key`` is a stable identifier for this admission (job id, or group id)
        and is used only for bookkeeping and error messages.
        """
        for name, need in request.items():
            capacity = self.pool.get(name)
            if capacity is not None and need > capacity:
                raise AdmissionImpossible(
                    f"Job {key}"
                    + (f" (rule {rule})" if rule else "")
                    + f" requires {name}={need}, but the shared pool only provides "
                    f"{name}={capacity}. Increase --shared-local-pool-"
                    + (
                        "cores"
                        if name == "cores"
                        else "mem"
                        if name == "mem_mb"
                        else name
                    )
                    + ", reduce the job's requirement, or lower the reserve."
                )

        deadline = (
            time.monotonic() + self.admit_timeout if self.admit_timeout > 0 else None
        )
        enqueued_at = time.time()
        waited_reported = False

        while True:
            admitted = self._try_admit(key, request, priority, rule, enqueued_at)
            if admitted is not None:
                if waited_reported:
                    self._info(
                        f"Admitted job {key} after {time.time() - enqueued_at:.1f}s "
                        f"of waiting."
                    )
                return admitted
            if deadline is not None and time.monotonic() > deadline:
                raise LeaseError(
                    f"Timed out after {self.admit_timeout}s waiting for resources for "
                    f"job {key} (needs {dict(request)}, pool {dict(self.pool)})."
                )
            if not waited_reported:
                self._info(
                    f"Waiting for shared resources to run job {key} "
                    f"(needs {dict(request)}, pool {dict(self.pool)})."
                )
                waited_reported = True
            time.sleep(self.poll_interval)

    def _try_admit(
        self,
        key: str,
        request: Mapping[str, int],
        priority: int,
        rule: str | None,
        enqueued_at: float,
    ) -> LeaseHandle | None:
        lock_fd = _open_locked(self._path(ADMIT_LOCK), blocking=True)
        try:
            used, unreadable = self.snapshot()
            self._handle_unreadable(unreadable)
            if not fits(request, used, self.pool):
                self._publish_pending(key, request, priority, enqueued_at, rule)
                return None

            others = self._read_other_waiters()
            self_waiter = Waiter(
                instance_id=self.instance_id,
                weight=self.weight,
                priority=priority,
                enqueued_at=enqueued_at,
                resources=dict(request),
            )
            now = time.time()
            yield_to, blocker = should_yield(
                self_waiter,
                others,
                now=now,
                aging_seconds=self.aging_seconds,
                aging_step=self.aging_step,
                used=used,
                pool=self.pool,
                self_resources=request,
            )
            if yield_to:
                self._debug(f"Yielding admission of {key} to {blocker}.")
                self._publish_pending(key, request, priority, enqueued_at, rule)
                return None

            handle = self._publish_lease(key, request, priority, rule, enqueued_at)
            with self._lock:
                self._held[key] = handle
            self._clear_pending()
            return handle
        finally:
            _unlock_close(lock_fd)

    def _handle_unreadable(self, unreadable: list[str]) -> None:
        """React to leases we cannot read.

        In a shared namespace an unreadable lease means someone's usage is
        invisible, so the cap cannot be honoured. Silently ignoring it would
        cause oversubscription, which is exactly what this plugin exists to
        prevent, so we fail loudly instead.
        """
        if not unreadable:
            return
        message = (
            "Cannot read "
            + ", ".join(unreadable)
            + ". Resource usage of those leases is invisible, so the shared limit "
            "cannot be guaranteed. The files must be world-readable (mode 0644); "
            "ask their owner to remove them or fix permissions."
        )
        if self.namespace.is_shared:
            raise LeaseError(message)
        self._warn(message)

    def _publish_lease(
        self,
        key: str,
        request: Mapping[str, int],
        priority: int,
        rule: str | None,
        enqueued_at: float,
    ) -> LeaseHandle:
        safe_key = _sanitize(key)
        final = self._path(
            f"{self.instance_id}-{safe_key}-{uuid.uuid4().hex[:6]}{LEASE_SUFFIX}"
        )
        tmp = self._path(f".tmp-{uuid.uuid4().hex}")
        payload = {
            "version": SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "host": socket.gethostname(),
            "jobid": key,
            "rule": rule,
            "resources": dict(request),
            "priority": int(priority),
            "weight": self.weight,
            "enqueued_at": enqueued_at,
            "started_at": time.time(),
        }
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR, FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(fd, json.dumps(payload).encode("utf-8"))
            # Explicitly widen permissions: under umask 077 the default would be
            # 0600 and other users could neither read nor lock the file.
            os.fchmod(fd, FILE_MODE)
            os.fsync(fd)
            os.rename(tmp, final)
        except BaseException:
            try:
                os.close(fd)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            raise
        self._debug(f"Admitted {key}: {dict(request)} -> {final.name}")
        return LeaseHandle(path=final, fd=fd, resources=dict(request))

    def release(self, key: str) -> None:
        """Release a lease held for ``key``."""
        with self._lock:
            handle = self._held.pop(key, None)
        if handle is None:
            return
        # The lock is what actually matters: dropping it frees the resources even
        # if the file cannot be removed (a sticky directory, or a race with a peer).
        with contextlib.suppress(OSError):
            os.unlink(handle.path)
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(handle.fd)
        self._debug(f"Released lease for {key}.")

    def release_all(self) -> None:
        with self._lock:
            keys = list(self._held)
        for key in keys:
            self.release(key)

    # ------------------------------------------------------------- pending info

    def _pending_path(self) -> Path:
        return self._path(f"{self.instance_id}{PENDING_SUFFIX}")

    def _publish_pending(
        self,
        key: str,
        request: Mapping[str, int],
        priority: int,
        enqueued_at: float,
        rule: str | None,
    ) -> None:
        """Advertise our highest-priority outstanding request.

        Other instances read this to decide whether to yield. Publishing
        atomically avoids readers seeing a truncated file.
        """
        final = self._pending_path()
        tmp = self._path(f".tmp-pending-{uuid.uuid4().hex}")
        payload = {
            "version": SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "uid": os.getuid(),
            "jobid": key,
            "rule": rule,
            "resources": dict(request),
            "priority": int(priority),
            "weight": self.weight,
            "enqueued_at": enqueued_at,
            "updated_at": time.time(),
        }
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR, FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.fchmod(fd, FILE_MODE)
            os.rename(tmp, final)
        finally:
            _unlock_close(fd)

    def _clear_pending(self) -> None:
        with contextlib.suppress(OSError):
            os.unlink(self._pending_path())

    def _read_other_waiters(self) -> list[Waiter]:
        waiters: list[Waiter] = []
        for path in self._scan(PENDING_SUFFIX):
            data = _read_json(path)
            if data is None:
                continue
            if data.get("instance_id") == self.instance_id:
                continue
            resources = {
                str(k): int(v)
                for k, v in (data.get("resources") or {}).items()
                if isinstance(v, int | float)
            }
            waiters.append(
                Waiter(
                    instance_id=str(data.get("instance_id", "?")),
                    weight=float(data.get("weight", 1.0)),
                    priority=int(data.get("priority", 0)),
                    enqueued_at=float(data.get("enqueued_at", 0.0)),
                    resources=resources,
                )
            )
        return waiters


@dataclass
class _Unreadable:
    message: str


def _sanitize(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
    return safe[:64] or "job"


def _open_locked(path: Path, *, blocking: bool = False) -> int:
    """Open ``path`` and take an exclusive flock.

    A file that cannot be opened at all becomes a :class:`LeaseError` naming it,
    rather than a bare ``PermissionError`` escaping from the middle of admission.
    The fallback in :func:`_open_for_lock` handles every file this process may
    read, so what reaches here is state an operator has to fix.
    """
    try:
        fd = _open_for_lock(path, create=True)
    except PermissionError as err:
        raise LeaseError(_lock_failure_message(path, err)) from None
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, flags)
    except OSError:
        os.close(fd)
        raise
    return fd


def _lock_failure_message(path: Path, err: PermissionError) -> str:
    """Explain an unopenable lock file, and what to do about it."""
    if not path.exists():
        return (
            f"Cannot create lock file {path}: {err}. Its directory must be writable "
            "by this process; ask an administrator to fix its permissions."
        )
    owner, mode = _uid_of(path), _mode_of(path)
    message = (
        f"Cannot open lock file {path} for locking: {err}. It is owned by uid "
        f"{owner} with mode {mode}, and this process cannot open it at all."
    )
    if owner == os.getuid():
        message += (
            f" Remove it and it is recreated with mode {FILE_MODE:04o}: rm {path}"
        )
    else:
        message += (
            " Ask its owner to remove it (0.1.3 created these as 0600 under "
            f"umask 077), or remove it yourself if the directory is writable: rm {path}"
        )
    return message


def _open_for_lock(path: Path, *, create: bool = False) -> int:
    """Open ``path`` for locking, falling back to read-only when not writable.

    Lock files are created with :data:`FILE_MODE` (``0644``) so that peers can
    read them, which means a *peer's* file is not writable by us. Opening it
    ``O_RDWR`` would then fail with ``EACCES`` -- that is the case in the ``group``
    and ``system`` scopes, where the first user to run creates ``admit.lock`` and
    everybody else gets ``PermissionError``.

    ``flock`` needs no write access (unlike POSIX ``fcntl`` locks), so retrying
    ``O_RDONLY`` keeps the lock fully functional while leaving the published
    modes untouched. Without this fallback a shared scope only ever worked for
    the user who happened to create the file first.

    Two consequences are worth knowing, and neither is a silent degrade:

    * A file this process cannot open *at all* still maps to "held" in
      :func:`_try_lock_read`. Its content is then reported as :class:`_Unreadable`
      (loud, and fatal in a shared scope), so it can never look like a phantom
      live lease that quietly withholds capacity.
    * On filesystems whose ``flock`` is emulated through POSIX record locks (some
      NFS mounts), a read-only descriptor may be refused. That fails explicitly
      instead of locking incorrectly. The default namespaces live on tmpfs.
    """
    flags = (os.O_CREAT if create else 0) | os.O_RDWR
    try:
        fd = os.open(path, flags, FILE_MODE)
    except PermissionError as err:
        # Either the file exists and we may not write it (a peer published it
        # 0644), or -- when creating -- the directory is not writable. A read-only
        # open separates the two without a second stat: it succeeds for a peer's
        # file and fails for a missing one. Re-raising the *original* error keeps
        # "the directory is not writable" from being reported as FileNotFoundError.
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            raise err from None
    if create:
        # ``os.open``'s mode argument is masked by the umask, so under ``umask 077``
        # the lock file would be ``0600`` and peers could neither read nor lock it.
        # Files created here are the shared lock files (``admit.lock``, instance
        # claims), so widen them explicitly like every other published entry. This
        # also repairs a 0600 file we own left behind by 0.1.3.
        with contextlib.suppress(OSError):
            os.fchmod(fd, FILE_MODE)
    return fd


def _unlock_close(fd: int) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)


def _try_lock_read(path: Path) -> int | None:
    """Try to take an exclusive lock; return the fd on success, ``None`` if held."""
    try:
        fd = _open_for_lock(path)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        os.close(fd)
        if err.errno in (errno.EACCES, errno.EAGAIN):
            return None
        return None
    return fd


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _read_lease(path: Path) -> Union[LeaseEntry, "_Unreadable"] | None:
    """Read a lease, distinguishing live, dead and unreadable entries.

    Returns
    -------
    ``None``
        The entry consumes no resources: either nobody holds the lock (a stale
        file), or the file no longer exists at all.
    ``_Unreadable``
        The lock is held by a live owner but we may not read the content, so the
        usage is unknown.
    ``LeaseEntry``
        A live lease whose content we could read.
    """
    fd = _try_lock_read(path)
    if fd is not None:
        # We were able to take the lock, so no live owner exists.
        _unlock_close(fd)
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        # The file vanished between _scan()'s listdir() and this open, which is the
        # normal consequence of another thread's release() unlinking it. No entry
        # means no resources, so this must be treated exactly like a stale file --
        # never reported as unreadable. Note this is *not* a rare condition: the run
        # that exposed it emitted 39 such warnings, all for files that had already
        # been released successfully. Reporting them as unparseable both blamed
        # permissions and, in a shared namespace, aborted the workflow at random.
        return None
    except PermissionError:
        return _Unreadable(
            f"{path.name} (owner uid {_uid_of(path)}, mode {_mode_of(path)})"
        )
    except (OSError, json.JSONDecodeError):
        # Locked but unparseable: treat as dead-but-undeletable rather than
        # crashing the workflow. Its resources are unknown; a shared namespace
        # will surface this through the unreadable path instead.
        return _Unreadable(f"{path.name} (unparseable)")
    version = data.get("version", 0)
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise LeaseError(
            f"Lease {path} was written by a newer plugin (schema version {version} > "
            f"{SCHEMA_VERSION}). Upgrade this plugin or use a separate namespace."
        )
    return LeaseEntry(path=path, data=data)


def _uid_of(path: Path) -> int:
    try:
        return os.stat(path).st_uid
    except OSError:  # pragma: no cover
        return -1


def _mode_of(path: Path) -> str:
    try:
        return oct(os.stat(path).st_mode & 0o777)
    except OSError:  # pragma: no cover
        return "?"
