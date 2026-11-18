`solprop.sol` exports `StatsSpec`, which a session passes to `stats=`, and no longer
exports `resume`: that name read a file's resumable state and never continued a run,
which is `resume=True` on a run or `solprop resume`. The reader is
`solprop.output.read_resume_state`.
