`solprop.tools.pressure_zdw` and `solprop.tools.pressure_rdw` now run the search of
`solprop.tools.solve_parameter`. `pressure_zdw` samples the pressure range rather than
taking one Brent bracket over it: where the dispersion has one zero in the range it finds
the same root, to within the noise of the differenced dispersion (a few parts in 1e8); a
sign change through a resonance is no longer taken for a zero; and with several zeros it
returns the lowest. When no pressure in the range meets the target, both now say so as
`solve_parameter` does, naming the range of the quantity the range gives.
