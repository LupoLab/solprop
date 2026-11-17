The arrays of one trajectory carry no leading axis of one anywhere (SEP 52): the
`PropagationProblem` operator's value tables (`LinearOperator.phi_coefficients` is
`(n_intervals, n_deg, n_modes, n_pol, n_omega)`, `phi_constant` and an expansion's
`terms` lose the same axis), the stage terms (`StageTerms.phi`), the separable operator's
`value_shape`, the boundary monitor's and the split collar's readings
(`BoundaryMonitor.power`, `SplitCrossing.occupancy`), the design check's launch, the
carrier frame's chord (a scalar per position) and both backends' private state. No field
is added or removed and no value changes; code that indexed `[0]` into these arrays drops
the index.
