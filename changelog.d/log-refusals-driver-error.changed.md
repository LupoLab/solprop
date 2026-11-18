The progress refusals of `solprop.log` (`set_default_progress`, `default_reporter` and
`BarReporter`'s `period`) raise a `DriverError` instead of a bare `ValueError`
(`DriverError` is now a `ValueError`, so `except ValueError` still catches them); and
`configure_logging(file=...)` writes the log file as UTF-8 whatever the locale.
