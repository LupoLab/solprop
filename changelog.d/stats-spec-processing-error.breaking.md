A malformed `StatsSpec` (an unknown, repeated or reserved name, a bad window or region, a
periodic subset that is not a subset or names a statistic no backend evaluates between
planes, a period below 1, a missing far-field table) now raises `ProcessingError`, a
`ValueError`, instead of `OutputError`, a `RuntimeError`: it is an argument error. Catch
`ProcessingError` (or `ValueError`) where `OutputError` was caught.
