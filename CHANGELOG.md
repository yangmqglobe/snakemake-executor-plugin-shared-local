# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases are managed
by [Release Please](https://github.com/googleapis/release-please) from
conventional commits.

## [0.1.3](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/compare/v0.1.2...v0.1.3) (2026-10-15)


### Bug Fixes

* prepare for PyPI and bioconda release ([#9](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/issues/9)) ([8239dbe](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/commit/8239dbe30ebe16e81a76f4a0f8dd5835f70355fd))

## [0.1.2](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/compare/v0.1.1...v0.1.2) (2026-10-14)


### Bug Fixes

* spread NUMA-bound jobs across a node instead of one core ([#7](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/issues/7)) ([20aedb3](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/commit/20aedb3064c39636d7b9768432798fece9893488)), closes [#5](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/issues/5)
* stop reporting released leases as unparseable ([#6](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/issues/6)) ([cd45a46](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/commit/cd45a46574a0c72659c2680b6408aab63786ae3e)), closes [#4](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/issues/4)

## [0.1.1](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/compare/v0.1.0...v0.1.1) (2026-10-13)


### Bug Fixes

* arbitrate memory through one shared dimension and honor env vars ([ed11884](https://github.com/yangmqglobe/snakemake-executor-plugin-shared-local/commit/ed11884df43e328407dd11843ab652960cb9aa2d))

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
