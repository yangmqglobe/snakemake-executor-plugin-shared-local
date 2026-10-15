Run several Snakemake instances on **one machine** without them fighting over it.

Every `snakemake -c N` normally assumes it owns the whole computer. Start three of
them on the same workstation and you get `3 × N` processes competing for the same
cores and memory, so everything ends up slower than running them one after another.
This plugin keeps execution **local** — no cluster, no container, no daemon — but
makes the participating instances share a single resource pool:

```console
$ snakemake -s Snakefile --executor shared-local -c 16
```

It is aimed at the common case of a workstation or login node driven by several
ad-hoc Snakemake runs. Use it when you want those runs to cooperate; use a cluster
executor when you want accounting and enforcement that a cooperative protocol
cannot provide.
