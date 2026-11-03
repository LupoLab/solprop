A `Simulation` built from an array now copies it at construction, so changing the
caller's array before the run no longer changes the run, its checks or its recorded
configuration (#434). `Simulation.initial` is that read-only complex128 copy; the
translated problem holds the same copy rather than a second one, and a
`numpy.broadcast_to` input still copies only the block it repeats. An array input of
another dtype is now recorded as complex128.
