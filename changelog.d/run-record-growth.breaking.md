A Multistep run's call ledger (`logical_calls` and `execution.response_calls` in the
run's outcome, `/meta/outcome` in an HDF5 file) records its predictor orders run-length
encoded, as `[order, count]` pairs, instead of one integer per accepted step;
`CallLedger.to_record` writes them and `CallLedger.from_record` reads them back step by
step. `/meta/outcome` is rewritten in place at every plane rather than deleted and
recreated. The output schema stays 2, the version 0.2.0 is the first to release: a file
written by an earlier development build of schema 2 that holds the flat list of orders
is refused by name on resume; rerun the case, or read it with that build
([#80](https://github.com/LupoLab/solprop/issues/80)).
