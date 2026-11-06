An output file no longer grows with the square of its planes. `HDF5Output` rewrites
`/meta/outcome` in place instead of deleting and recreating it at every plane, a saved
plane builds and validates its run record once instead of nine times, and the step
driver keeps only the error norms its controller reads, not one per accepted step: a
Multistep run saved step by step (`step_planes=1`) costs 19.5 kB a plane at 707 planes
and 19.3 kB at 3479, where it cost 23.8 kB and 31.5 kB. To get there a Multistep run's
call ledger (`logical_calls` and `execution.response_calls` in the run's outcome,
`/meta/outcome` in an HDF5 file) records its predictor orders run-length encoded, as
`[order, count]` pairs, where 0.1.0 wrote one integer per accepted step. Code that read
that list should expand the pairs, or read the record through `CallLedger.from_record`,
which gives the orders step by step. A file written by a development build of output
schema 2 that holds the flat list is refused by name on resume; rerun the case, or read
it with that build ([#80](https://github.com/LupoLab/solprop/issues/80)).
