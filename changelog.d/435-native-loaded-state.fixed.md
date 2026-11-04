The native backend no longer keeps the last host state it loaded alive after
`set_initial`: a temporary, such as an input noise sum, a resumed plane or a delay
scan's input, was one host field volume held until the next load, outside
`memory_ledger()` and `estimate_memory`. A resident Adams march no longer keeps a host copy
of the state for the backend's life either; solprop's host engine, which reads one,
lists it as `adams_host_state` (#435).
