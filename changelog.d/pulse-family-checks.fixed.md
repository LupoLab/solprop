`GaussianPulse` now refuses a super-Gaussian order `m` that is not an integer of at least
1, as `SpatialPulse` does; `PulseSum` and `SpatialPulse` refuse a beam given as a temporal
pulse with a `PulseError` at construction instead of a `TypeError` at build time; and
`PlanarPulse`'s printed form and the planar-grid refusal of a `SpatialPulse` name the
`waist` keyword, not a `waist_y` that does not exist.
