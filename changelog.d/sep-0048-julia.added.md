Add explicit setup and status commands for the optional Julia engine, with pinned bridge
versions, immutable source copies and verified offline artifact inventories, as part of
[SEP 48](docs/dev/seps/sep-0048-julia-backend.md). Its numerical sessions and device
qualification remain in progress on the implementation branch.

Add owned `SimulationSettings.backend_options`, forwarded through execution options
and spawned workers without changing scientific hashes. Backends refuse unsupported
option keys before workspace construction.
