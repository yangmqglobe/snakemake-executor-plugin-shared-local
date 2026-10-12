# snakemake-executor-plugin-shared-local

Run several Snakemake instances on **one machine** without them fighting over it.

Normally, every `snakemake -c N` believes it owns the whole computer. Start three
of them on the same workstation and you get `3 × N` processes competing for the
same cores and memory, and everything runs slower than running them one at a
time. This plugin keeps execution **local** — no cluster, no container, no daemon
— but makes participating instances share a single resource pool.

```console
$ snakemake -s Snakefile --executor shared-local -c 16
```

---

## What it does

- **Shares a CPU/memory pool between instances.** The pool is the machine's
  capacity minus a reserve, or whatever you configure. It is the *global*
  ceiling; `--cores` only limits one instance and may freely exceed the pool.
- **Schedules by priority, then FIFO.** Within equal priority, whoever asked
  first wins. Waiting jobs age so they cannot starve.
- **Pins jobs to NUMA nodes** by default on multi-node machines, which is a
  measurable win: an unpinned 400 MiB allocation on the reference machine landed
  almost entirely on one arbitrary node (`N0=1222 / N1=103615`), while a bound one
  stayed local (`N0=104369 / N1=473`).
- **Optionally enforces limits in the kernel** with cgroup v2, for jobs that
  misbehave or workflows that under-declare their needs.

---

## Install

```console
pip install snakemake-executor-plugin-shared-local
```

Then select it with `--executor shared-local`. There are **no Python
dependencies** beyond Snakemake's own plugin interfaces — everything is standard
library.

> **Developing the plugin itself?** Snakemake discovers executor plugins by
> scanning `sys.path` with `pkgutil.iter_modules()` for packages named
> `snakemake_executor_plugin_*`. A regular or editable install both work: hatchling
> (this project's backend) and setuptools defaults write a `.pth` file containing
> the real source path, which `pkgutil` can see. If `--executor shared-local`
> reports the package as not installed, you are most likely running from a tree
> whose `src/` is not importable — put `src/` on `PYTHONPATH`, or run
> `pip install .` — and note that a stale `snakemake_executor_plugin_shared_local`
> in a *different* interpreter's `site-packages` will shadow what you expect.

---

## How it works

There is no daemon and no heartbeat protocol. Coordination state is a directory of
small JSON files, and **liveness is a `flock`**:

- A running job holds an exclusive lock on its own lease file for the job's
  lifetime. When the process dies — including `SIGKILL` or power loss — the kernel
  releases that lock.
- A waiting job scans the directory and tries a non-blocking lock on each file. If
  it **succeeds**, the owner is gone, so that entry consumes nothing. If it fails,
  the entry is live.

Two consequences fall out of this design for free: crash recovery *is* liveness
detection, and accounting stays correct even when a stale file cannot be deleted
(the usual case in a sticky directory such as `/dev/shm`).

Admission happens on a worker thread, never on Snakemake's scheduling thread.
That detail is load-bearing: Snakemake's executor contract calls
`run_jobs(jobs)` as `for job in jobs: run_job(job)` *from the scheduling loop*, so
blocking inside `run_job` serialises the whole scheduler — with a probe, two
independent jobs were observed to execute strictly one after another behind a
single blocked `run_job`.

---

## Coordination scope

Pick who shares the pool with `--shared-local-namespace`:

| Scope | Directory (default) | Mode | Meaning |
|---|---|---|---|
| `user` *(default)* | `/dev/shm/snakemake-shared-local/u<uid>` | `0700` | Only your own instances coordinate. |
| `group` | `/dev/shm/snakemake-shared-local/g<gid>` | `0770` | Instances of users sharing your primary group coordinate. |
| `system` | `/dev/shm/snakemake-shared-local/shared` | `0777` | Every user on the machine coordinates. |

In a shared scope all participants **must agree on the pool**. The first instance
writes `pool.json`; a later instance that disagrees aborts by default, because a
mismatch means each process believes the machine is a different size and the
global limit quietly becomes fiction. Use `--shared-local-pool-conflict warn` to
downgrade that to a warning.

For `system`, it is worth creating the directory up front so its permissions are
explicit rather than inherited from whoever happens to run first:

```console
sudo mkdir -p /dev/shm/snakemake-shared-local/shared
sudo chmod 1777 /dev/shm/snakemake-shared-local/shared
```

### This is coordination, not a security boundary

Participants report their own resource usage, so this is a **cooperative
protocol among trusted users**, not isolation. A participant can under-declare
what it needs, and lease filenames reveal other users' rule names and PIDs
(`record-rule` defaults to off in the `system` scope for that reason). If you need
enforcement against uncooperative workloads, that is what the cgroup option and
ordinary Unix permissions are for.

---

## The pool

By default the pool is detected hardware **minus a 10 % reserve** for the
operating system and non-Snakemake work:

```console
$ snakemake -s Snakefile --executor shared-local -c 16 --shared-local-pool-cores 96
```

`--cores` is *your instance's* concurrency limit. It plays no part in computing the
pool, and every instance passing `--cores 128` is fine — the plugin is what keeps
the machine from being oversubscribed. You do not have to rename or re-tune your
existing `--cores` values to cooperate.

If a job needs more of a resource than the pool can ever provide, the plugin
**fails immediately** with an explanation instead of waiting forever:

```
Job 0 (rule big) requires cores=16, but the shared pool only provides cores=4.
Increase --shared-local-pool-cores, reduce the job's requirement, or lower the
reserve.
```

This matters because Snakemake itself will not catch it. Snakemake only clamps
resource requests against `--cores` / `--resources`, which are per-instance:

```
$ snakemake -s Snakefile -c 8      # rule with threads: 16
Rules claiming more threads will be scaled down.    # job.threads becomes 8, no error
```

so a pool smaller than a single job's request is invisible to it.

---

## Scheduling and fairness

Admission order is a total order:

1. `--shared-local-instance-weight` (default `1.0`), descending
2. Snakemake's rule `priority`, descending
3. arrival time, ascending (FIFO)

Waiting jobs receive an aging bonus (`--shared-local-aging-seconds`, default 60) so
a low-priority job is not starved indefinitely. An instance may exceed its equal
share while nobody else is waiting, and is held to its share once someone is.

Two caveats about priority, both deliberate:

- Snakemake's `priority` is **per-workflow**. A rule with `priority: 50` in
  workflow A has no defined relationship to a rule with `priority: 50` in
  workflow B, so priority is a tie-breaker *within* the configured weight.
- `priority` is not part of Snakemake's stable executor plugin interface. The
  plugin reads it when present and silently falls back to weight + FIFO when it is
  not, rather than breaking on a future Snakemake release.

---

## Memory

Memory is arbitrated through `mem_mb`, the same resource Snakemake uses. Note that
**`mem_mb` does not exist unless you declare it** — with no `resources: mem_mb=...`
and no `--default-resources`, a job's resources are just `{tmpdir, _nodes,
_cores}`, and the plugin can only arbitrate cores.

The plugin therefore warns once if any job lacks `mem_mb` and suggests:

```console
$ snakemake -s Snakefile --executor shared-local -c 16 --default-resources mem_mb=8000
```

Set `--shared-local-require-mem` to turn that warning into a hard error.

Be aware that Snakemake clamps `mem_mb` against `--resources mem_mb=...`, so if you
use `--resources mem_mb`, keep it consistent with `--shared-local-pool-mem`.

---

## NUMA placement

On a machine with more than one NUMA node, each job is pinned to the node with the
most free capacity that can hold it (`--shared-local-numa-bind`, `auto` by
default; `on` forces, `off` disables). The binding uses `set_mempolicy(MPOL_BIND)`
and `sched_setaffinity` through `ctypes`, so `numactl` is **not** required, and it
is inherited by job subprocesses.

A job whose memory requirement exceeds every single node runs unbound with a
warning rather than failing. On a single-node machine the feature disables itself.

Binding is a *placement* decision, not a limit — it does not restrict how much a
job may use.

---

## Optional kernel enforcement (cgroup v2)

By default coordination is advisory. With `--shared-local-cgroup on` (or
`"cgroup": "on"` in the config file),
each job is wrapped in a systemd user scope so the kernel enforces the limits:

```console
$ snakemake -s Snakefile --executor shared-local -c 16 --shared-local-cgroup on
```

What this actually does on a typical modern host, as measured for this plugin:

- `CPUQuota=25%` genuinely throttles (24.9M loop iterations in 3 s became 4.6M).
- `MemoryMax=200M` **alone does not enforce anything** — a 600 MiB allocation
  survived, because the cgroup could swap. Adding `MemorySwapMax=0` made the same
  program die with `SIGKILL` (exit 137). The plugin therefore always pairs them;
  without the swap guard a memory limit is decoration.
- `AllowedCPUs` silently did nothing, because `cpuset` is not delegated on that
  host. The plugin never relies on cpuset; CPU placement is `numa.py`'s job.

Because `systemd-run --user` needs the user D-Bus session, enforcement is
**best-effort**: if D-Bus, `systemd-run`, or cgroup v2 is unavailable (a cron job,
a container, a non-systemd host), the job runs unwrapped and you get a warning.
This feature never fails your workflow.

---

## Deployment: make it the default

The plugin can only coordinate instances that *use* it. Someone running plain
`snakemake -c 64` is invisible to the pool, and so is non-Snakemake work. To
coordinate a whole account, put it in a profile so nobody has to remember the
flags:

```yaml
# ~/.config/snakemake/sharedlocal/config.yaml
executor: shared-local
shared-local-namespace: group
shared-local-pool-cores: 96
shared-local-pool-mem: 450G
```

```console
$ snakemake -s Snakefile --profile ~/.config/snakemake/sharedlocal -c 32
```

Settings may also come from a JSON config file
(`~/.config/snakemake/shared-local/config.json`), overridable by CLI flags:

```json
{
  "namespace": "group",
  "pool_cores": 96,
  "pool_mem": "450G",
  "numa_bind": "on",
  "cgroup": "off"
}
```

Precedence is: explicit CLI flag, then config file, then auto-detection.

---

## Settings

| Flag | Default | Purpose |
|---|---|---|
| `--shared-local-namespace` | `auto` → `user` | `user`, `group` or `system` scope |
| `--shared-local-dir` | scope-dependent | Override the lease directory |
| `--shared-local-pool-cores` | detected − 10 % | Global CPU ceiling |
| `--shared-local-pool-mem` | detected − 10 % | Global memory ceiling (`450G`, `400000`) |
| `--shared-local-pool-conflict` | `error` | `error` or `warn` on pool mismatch |
| `--shared-local-instance-weight` | `1.0` | This instance's share relative to others |
| `--shared-local-aging-seconds` | `60` | Starvation-avoidance time constant (`0` off) |
| `--shared-local-aging-step` | `1` | Priority bonus per aging interval |
| `--shared-local-admit-timeout` | `0` (wait forever) | Give up waiting after N seconds |
| `--shared-local-poll-interval` | `0.05` | Seconds between admission attempts |
| `--shared-local-numa-bind` | `auto` | `auto`, `on`, `off` |
| `--shared-local-cgroup` | `auto` → **off** | `auto`, `on`, `off`; see below |
| `--shared-local-require-mem` | `auto` | Fail when a job omits `mem_mb` |
| `--shared-local-record-rule` | `auto` | Record rule names in leases (off in `system`) |
| `--shared-local-gc-report` | `auto` | Report stale leases that could not be removed |
| `--shared-local-verbose-leases` | `auto` | Log every admission/wait/collection decision |
| `--shared-local-config` | `~/.config/snakemake/shared-local/config.json` | JSON settings file |

Every setting also has a `SNAKEMAKE_SHARED_LOCAL_*` environment variable.

---

## What it does not do

- **It cannot see uncooperative workloads.** Instances launched without this
  executor, and non-Snakemake processes, are not accounted for. The reserve is the
  only allowance made for them.
- **It does not limit resources unless you enable cgroup enforcement.** Leases are
  an agreement.
- **No GPU support, no disk-space arbitration, no cross-machine coordination, no
  job preemption, no daemon.**
- **It does not replace a cluster scheduler.** If this machine is a batch node with
  SLURM, submitting through SLURM gives you accounting and enforcement you cannot
  get from a cooperative protocol; this plugin is for the common case where a
  workstation or login node is driven by several ad-hoc Snakemake runs.

---

## Development

```console
$ pip install -e .
$ pip install pytest pytest-cov pytest-timeout
$ python -m pytest tests -q
```

The suite has two layers. Unit tests cover configuration, pool arithmetic,
fairness, namespaces/permissions and the cgroup command construction. Integration
tests launch **real, concurrent `snakemake` processes** and assert the properties
that matter: a shared pool is never exceeded, a `SIGKILL`ed instance's leases are
reclaimed, oversized jobs fail fast instead of hanging, group (`pipe`) jobs do not
self-deadlock, and pool conflicts are refused.

## License

MIT
