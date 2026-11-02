An output file no longer grows with the square of its planes. `HDF5Output` rewrites
`/meta/outcome` in place instead of deleting and recreating it at every plane, a saved
plane builds and validates its run record once instead of nine times, and the
Multistep call ledger's orders are run-length encoded: a Multistep run saved step by
step (`step_planes=1`) costs 19.5 kB a plane at 707 planes and 19.3 kB at 3479, where
it cost 23.8 kB and 31.5 kB. The step driver keeps only the error norms its controller
reads, not one per accepted step
([#80](https://github.com/LupoLab/solprop/issues/80)).
