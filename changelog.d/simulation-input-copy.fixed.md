A `Simulation` built from an array input now copies the array at construction, so
changing it before the run no longer changes the run, its construction-time checks or
its recorded configuration (#434). `Simulation.initial` is a read-only view of that copy
as the caller gave it, so an input of another dtype is recorded unchanged and older
files resume. On free-space and spatial routes the translated problem holds the same
copy; on a modal route it holds the converted launch beside it, two states, which the
memory admission counts as `simulation_input`. A `numpy.broadcast_to` input copies only
the block it repeats.
