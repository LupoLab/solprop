A refused argument to `SnapshotController` (`interval_seconds`, `max_snapshots`) or to
`TDSEArchive` (`path`, `outputs`) raises the new
`solprop.exceptions.OutputRequestError`, a `SolpropError` and a `ValueError`, as a
refused `progress` or `resume` raises a `DriverError`. They raised `OutputError` (a
`RuntimeError`) and bare `TypeError` or `ValueError` before. Catch `OutputRequestError`
or `ValueError` in place of `OutputError` or `TypeError` around these constructors;
`OutputError` still reports a file that cannot be written or read.
