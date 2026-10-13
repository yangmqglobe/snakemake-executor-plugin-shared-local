# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases are managed
by [Release Please](https://github.com/googleapis/release-please) from
conventional commits.

## [Unreleased]

### Fixed

- **Memory was never arbitrated.** The pool published its memory dimension as
  `mem_mib` while job requests were normalized to `mem_mb`. Because a dimension
  absent from the pool is treated as unconstrained, the mismatch did not raise — it
  silently disabled memory arbitration entirely, so a pool of 100 MiB admitted a job
  asking for 26000 MiB. Both sides now use Snakemake's own `mem_mb`, pool
  declarations are canonicalized on read (so a `pool.json` written by the older
  plugin is still understood rather than reported as a conflict), and a test pins
  the invariant that no pool dimension can be a name a job request cannot produce.
- **`SNAKEMAKE_SHARED_LOCAL_*` environment variables had no effect.** Snakemake only
  forwards `env_var` to its `configargparse` backend when a settings field opts in
  via metadata; no field did, so the documented variables were inert. Every field
  now opts in, giving the documented order CLI flag > environment variable > config
  file > auto-detection.
- **`require_mem` was ignored in the JSON config file.** It was missing from the
  set of valid keys, so `{"require_mem": "on"}` was reported as an unknown key and
  silently dropped.

## [0.1.0]

### Added

- Executor plugin (`--executor shared-local`) that coordinates several local
  Snakemake instances through a shared resource pool.
- flock-based leases with crash recovery: a lease whose owner died is reclaimed
  automatically, because the kernel releases the lock when the process exits.
- Three coordination scopes: `user` (per-UID), `group` (per-GID) and `system`
  (all users on the machine), with explicit permission handling so a restrictive
  `umask` cannot silently disable a shared scope.
- Pool declaration negotiated through an atomically claimed `pool.json`; a
  mismatch between instances aborts by default, since it would silently void the
  global limit.
- Priority-then-FIFO scheduling with aging to prevent starvation.
- NUMA placement of jobs via `ctypes` (`set_mempolicy` / `sched_setaffinity`), so
  no `numactl` dependency is required.
- Optional cgroup v2 enforcement of CPU and memory limits, always pairing
  `MemoryMax` with `MemorySwapMax=0` because a swappable memory limit is not a
  limit. Degrades to a warning when D-Bus or systemd is unavailable.
- Jobs whose resource request can never fit the pool fail immediately with an
  explanatory message instead of waiting forever.
- Group (`pipe`/service) jobs reserve all members' resources atomically, avoiding
  self-deadlock.
- Test suite covering configuration, pool arithmetic, fairness, namespaces,
  cgroups, executor wiring, plus integration tests that run real concurrent
  Snakemake processes.
