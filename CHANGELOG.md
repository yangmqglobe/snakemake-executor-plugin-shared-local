# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases are managed
by [Release Please](https://github.com/googleapis/release-please) from
conventional commits.

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
