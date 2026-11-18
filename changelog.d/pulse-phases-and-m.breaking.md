The pulse builders now spell their parameters as the drivers do. The Taylor spectral
phase of `GaussianPulse`, `SechPulse`, `CustomPulse`, `DataPulse` and `TimeDataPulse`
(and of `DataPulse.from_spectral_density`) is `phases=`, the name `prop_gnlse`,
`prop_capillary` and `prop_waveguide` use, rather than `phase=`; and the super-Gaussian
order of `SpatialPulse` and `PlanarPulse` is `m=`, as on `GaussianPulse`, rather than
`order=`. The meaning is unchanged. A recipe's pulse tables take `phases` and `m`.
Migration: rename `phase=` to `phases=` on the pulses and `order=` to `m=` on the beams.
