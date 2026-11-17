The progress refusals of `solprop.log` (`set_default_progress`, `default_reporter` and
`BarReporter`'s `period`) raise a `DriverError`, which is still a `ValueError`, instead of
a bare `ValueError`; and `configure_logging(file=...)` writes the log file as UTF-8
whatever the locale.
