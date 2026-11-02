On a radial, Cartesian, line or planar output, the processing readers that read each
transverse column as a modal amplitude in sqrt(W) now refuse the file by name with an
`OutputError` that names the spatial functional to use, instead of returning wrong
numbers silently (a 3 nJ launch read back as 4.8e-15 J). They are `energy`,
`peak_power`, `get_power`, `get_spectral_density`, `fwhm_t`, `fwhm_f`, `time_bandwidth`,
`arrival_time`, `field_autocorrelation`, `intensity_autocorrelation` and
`coherence_time`, the plots `prop_2d`, `time_1d`, `spec_1d` and `plot_energy`, and a
plane observer's `power` and `spectral_density`. Use `solprop.spatial.spatial_energy`,
`spatial_power` and their neighbours on a saved plane, or the run's `stats/energy`,
which is unchanged; `get_spectrum`, `get_field` and `get_phase` still return the stored
columns. `solprop.capabilities.output` gains `refused_readers`, which states these
refusals with the readers' own text, while it still admits a spatial file's `energy`
statistic. Modal files read as before (#147, #406).
