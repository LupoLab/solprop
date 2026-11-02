Under `absorber_scheme="split_step"`, the default on the spatial routes, the boundary
monitor judges the time window's guards on the power each step's split finds, halfway
through its factor, instead of the state it leaves, whose guards it had damped by about
`e^-10`: a pulse walking out of the window is now stopped where the in-stage scheme
stops it (#127). Translation also holds adaptive steps to the split collar's crossing
limit, half the collar's width over the fastest group-delay walk-off against the frame,
so that no step carries light through the collar and round the window undamped, and
refuses a fixed schedule past it, naming the limit (#149). See [absorbing
boundaries](docs/guide/numerics/boundaries.md#the-two-temporal-schemes).
