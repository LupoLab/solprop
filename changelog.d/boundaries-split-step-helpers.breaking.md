Removed `solprop.boundaries.split_step_factor`, `apply_split_step` and `apply_band_mask`,
which no backend ran: the `absorber_scheme="split_step"` path is the backends' own. To
reproduce them, multiply a time-domain field by `numpy.exp(-0.5 * temporal_rate * step)`
and mask a spectrum with `numpy.where(band_mask, spectrum, 0.0)`.
