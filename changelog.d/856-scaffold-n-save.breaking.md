`solprop.driver.scaffold.propagate_fixed_rk4`'s `n_save` now means what it means on every
`Simulation` route: the number of evenly spaced saved planes over `[0, z_end]`, both ends
included, at least 2 (default 2). `n_steps` must be a multiple of `n_save - 1`. A call
that passed `n_save=4` now passes `n_save=5` for the same planes (#856).
