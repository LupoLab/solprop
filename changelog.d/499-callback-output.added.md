`solprop.output.CallbackOutput`, an output handler built from a callable: it gives each
saved plane to the callable and keeps nothing, so a reducing scan on `SimulationRunner` is
a few lines and costs no volume beyond the plane's own. `check_cache`, `resume` and
`resumability` now refuse any output handler by name (#499).
