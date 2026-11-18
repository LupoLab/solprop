The plots take their default ranges from the run instead of fixed windows that clipped
picosecond fibre pulses and mid-infrared spectra: with no `spec_range`, `prop_2d`,
`spec_1d` and `spectrogram` show the band the grid was declared over
(`plotting.propagation.default_spec_range`), and with no `time_range`, `prop_2d`,
`time_1d` and `spectrogram` show where the drawn light is
(`plotting.propagation.default_time_range`: 0.05 % to 99.95 % of each trace's energy,
widened by a quarter on each side). `prop_2d` defaults to `axis="wavelength"` like every
other plot. `spec_1d(log_min=1e-6)` is now `db_min=-60.0`, in dB below the peak, as on
`prop_2d` (`spectrogram` names its floor `db_min` too, at -40 dB by default). `DEFAULT_WAVELENGTH_RANGE`, `DEFAULT_SLICE_WAVELENGTH_RANGE`
and `DEFAULT_TIME_RANGE` are removed.
