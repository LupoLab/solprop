A malformed `StatsSpec` (an unknown, repeated or reserved name, a bad window or region,
a periodic subset that is not a subset or names a statistic no backend evaluates between
planes, a period below 1, a missing far-field table) or `stats.ApertureSet`, an unknown
profile or weighting passed to `stats.check_aperture_choices`, and a spec that does not
fit a line grid (an aperture centred off `y = 0`, a half-plane bounded in y or an
ellipse, or a spectral region whose `kappa_y` interval excludes zero) now raise
`ProcessingError`, a `ValueError`, instead of `OutputError`, a `RuntimeError`: they are
argument errors. Catch `ProcessingError` (or `ValueError`) where `OutputError` was
caught, around building a spec or its apertures and around a line run's statistics
tables.
