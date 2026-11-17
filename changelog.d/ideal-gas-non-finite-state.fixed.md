`solprop.physdata.ideal_gas_density`, and `density` and `pressure` with `ideal=True`,
now refuse a NaN or infinite pressure, number density or temperature with a
`MaterialError`, as the real-gas path already did, rather than returning NaN or
infinity.
