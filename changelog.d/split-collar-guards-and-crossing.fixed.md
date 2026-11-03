Under `absorber_scheme="split_step"`, the default on the spatial routes, the boundary
monitor judges the time window's guards on the power each step's split finds, halfway
through its factor, instead of the state it leaves, whose guards it had damped by about
`e^-10`: a pulse walking out of the window is now stopped where the in-stage scheme
stops it (#127). Steps are also held to the split collar's crossing limit, half the
collar's width over the walk-off of the fastest light holding more than `1e-6` of the
energy, so that no step carries light through the collar and round the window undamped
(issues #149 and #440). It is read from the launch at translation, which holds `dz_max`
and refuses a longer fixed schedule, and again from the state at every check, which caps
the steps after it and stops a fixed schedule it falls below. See [absorbing
boundaries](docs/guide/numerics/boundaries.md#the-two-temporal-schemes).
