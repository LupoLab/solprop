`solprop.capabilities.output` now states a spatial file's field in the run's stored unit
(V m radially, V/m on a Cartesian or line grid, sqrt(W/m) per y frequency on a planar
one) instead of sqrt(W), and no longer states a modal power, energy or spectral density
for it, whose readers refuse a spatial file (#425).
