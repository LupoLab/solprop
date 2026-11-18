`run_scan` no longer takes `stale_age`: a `Local` or `Pool` scan holds its file's lock
for its whole session, so a `running` point it finds is always taken up again, and the
keyword never reached `Queue` or `Slurm`. Remove it from calls; a durable queue's lease
is `Queue(stale_age=...)`, and now `Slurm(stale_age=...)` for the queue its tasks share.
`python -m solprop.scan --stale-age` now reaches the tasks of `--submit` and
`--emit-slurm`, and is refused without `--queue`, `--chunk` or a Slurm mode, where it
did nothing.
