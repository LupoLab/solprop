The output schema is 3: a Multistep run's call ledger (`logical_calls` and
`execution.response_calls` in `/meta/outcome`) records its predictor orders run-length
encoded, as `[order, count]` pairs, instead of one integer per accepted step;
`CallLedger.to_record` writes them and `CallLedger.from_record` reads them back step by
step. A file of schema 2 is refused by name, so a run started before this change cannot
be resumed by it: finish such a run, or read its file, with the version that wrote it
([#80](https://github.com/LupoLab/solprop/issues/80)).
