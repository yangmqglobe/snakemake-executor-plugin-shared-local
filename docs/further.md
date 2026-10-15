### How it works

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

Admission happens on a worker thread, never on Snakemake's scheduling thread. That
detail is load-bearing: Snakemake's executor contract calls `run_jobs(jobs)` as
`for job in jobs: run_job(job)` *from the scheduling loop*, so blocking inside
`run_job` would serialise the whole scheduler.

### Coordination scope

`--shared-local-namespace` selects who shares the pool:

| Scope | Directory (default) | Mode | Meaning |
|---|---|---|---|
| `user` *(default)* | `/dev/shm/snakemake-shared-local/u<uid>` | `0700` | Only your own instances coordinate. |
| `group` | `/dev/shm/snakemake-shared-local/g<gid>` | `0770` | Instances of users sharing your primary group coordinate. |
| `system` | `/dev/shm/snakemake-shared-local/shared` | `0777` | Every user on the machine coordinates. |

In a shared scope all participants **must agree on the pool**. The first instance
writes `pool.json`; a later instance that disagrees aborts by default, because a
mismatch means each process believes the machine is a different size and the global
limit quietly becomes fiction. Use `--shared-local-pool-conflict warn` to downgrade
that to a warning.

### This is coordination, not a security boundary

Participants report their own resource usage, so this is a **cooperative protocol
among trusted users**, not isolation. A participant can under-declare what it needs,
and lease filenames reveal other users' rule names and PIDs (`record-rule` defaults
to off in the `system` scope for that reason). If you need enforcement against
uncooperative workloads, that is what the cgroup option and ordinary Unix
permissions are for.

### The pool

By default the pool is detected hardware **minus a 10 % reserve** for the operating
system and non-Snakemake work. `--cores` is *your instance's* concurrency limit and
plays no part in computing the pool, so every instance passing `--cores 128` is
fine — the plugin is what keeps the machine from being oversubscribed.

If a job needs more of a resource than the pool can ever provide, the plugin **fails
immediately** with an explanation instead of waiting forever. This matters because
Snakemake itself will not catch it: it only clamps resource requests against
`--cores` / `--resources`, which are per-instance, so a pool smaller than a single
job's request is invisible to it.

### Scheduling and fairness

Admission order is a total order: `--shared-local-instance-weight` (descending),
then Snakemake's rule `priority` (descending), then arrival time (FIFO). Waiting jobs
receive an aging bonus so a low-priority job is not starved. An instance may exceed
its equal share while nobody else is waiting, and is held to its share once someone
is.

Two caveats about priority, both deliberate. Snakemake's `priority` is
**per-workflow**, so a `priority: 50` rule in workflow A has no defined relationship
to a `priority: 50` rule in workflow B. And `priority` is not part of Snakemake's
stable executor plugin interface, so the plugin reads it when present and silently
falls back to weight + FIFO when it is not.

### Memory

Memory is arbitrated through `mem_mb`, the same resource Snakemake uses. Note that
**`mem_mb` does not exist unless you declare it** — with no `resources: mem_mb=...`
and no `--default-resources`, a job's resources are just `{tmpdir, _nodes, _cores}`,
and the plugin can only arbitrate cores. It therefore warns once if any job lacks
`mem_mb`; set `--shared-local-require-mem` to turn that warning into a hard error.

`--shared-local-pool-mem` accepts a plain number (MiB) or a suffix, and **suffixes
are binary**: `450G` means 450 GiB. The value is compared directly against jobs'
`mem_mb`, so no rescaling happens in between.

### NUMA placement

On a machine with more than one NUMA node, each job is pinned to the node with the
most free capacity that can hold it (`--shared-local-numa-bind`, `auto` by
default). The binding uses `set_mempolicy(MPOL_BIND)` and `sched_setaffinity`
through `ctypes`, so `numactl` is **not** required, and it is inherited by job
subprocesses. Within the chosen node the CPUs are handed out round-robin.

A job whose memory requirement exceeds every single node runs unbound with a warning
rather than failing, and on a single-node machine the feature disables itself.
Binding is a *placement* decision, not a limit.

### Optional kernel enforcement (cgroup v2)

By default coordination is advisory. With `--shared-local-cgroup on`, each job is
wrapped in a systemd user scope so the kernel enforces the limits. What this
actually does on a typical modern host, as measured for this plugin:

- `CPUQuota=25%` genuinely throttles (24.9M loop iterations in 3 s became 4.6M).
- `MemoryMax=200M` **alone does not enforce anything** — a 600 MiB allocation
  survived, because the cgroup could swap. Adding `MemorySwapMax=0` made the same
  program die with `SIGKILL` (exit 137). The plugin therefore always pairs them.
- `AllowedCPUs` silently did nothing, because `cpuset` is not delegated on that
  host. The plugin never relies on cpuset; CPU placement is the NUMA layer's job.

Because `systemd-run --user` needs the user D-Bus session, enforcement is
**best-effort**: if D-Bus, `systemd-run`, or cgroup v2 is unavailable (a cron job, a
container, a non-systemd host), the job runs unwrapped and you get a warning. This
feature never fails your workflow.

### Platform support

**Linux is the supported platform. macOS is expected to work but is not covered by
CI.** This plugin is POSIX-only: liveness detection is an `flock`, and the `group`
and `system` scopes compare Unix ownership.

**Windows is not supported and is not intended to be.** Note what happens there:
Snakemake discovers executor plugins by importing every `snakemake_executor_plugin_*`
package on `sys.path`, and that scan runs on *every* invocation. An unguarded
`import fcntl` would therefore break Snakemake itself, for every user, even those
who never select this executor. The POSIX-only imports are consequently resolved
defensively: discovery always succeeds, and only selecting
`--executor shared-local` reports a clear error naming the missing facility.

On macOS, `/dev/shm` does not exist, so the coordination directory falls back to
`$XDG_RUNTIME_DIR` and then `/var/tmp`. NUMA placement (`/sys/devices/system/node`)
and cgroup enforcement (systemd) both disable themselves with a warning.

### Configuration

Settings come from the CLI, from `SNAKEMAKE_SHARED_LOCAL_*` environment variables, or
from a JSON file (`~/.config/snakemake/shared-local/config.json`). Precedence is:
explicit CLI flag, then environment variable, then config file, then
auto-detection.

```json
{
  "namespace": "group",
  "pool_cores": 96,
  "pool_mem": "450G",
  "numa_bind": "on",
  "cgroup": "off"
}
```

To coordinate a whole account without anyone remembering the flags, put it in a
profile:

```yaml
# ~/.config/snakemake/sharedlocal/config.yaml
executor: shared-local
shared-local-namespace: group
shared-local-pool-cores: 96
shared-local-pool-mem: 450G
```

### What it does not do

- **It cannot see uncooperative workloads.** Instances launched without this
  executor, and non-Snakemake processes, are not accounted for. The reserve is the
  only allowance made for them.
- **It does not limit resources unless you enable cgroup enforcement.** Leases are
  an agreement.
- **No GPU support, no disk-space arbitration, no cross-machine coordination, no job
  preemption, no daemon.**
- **It does not replace a cluster scheduler.** If this machine is a batch node with
  SLURM, submitting through SLURM gives you accounting and enforcement you cannot
  get from a cooperative protocol.
