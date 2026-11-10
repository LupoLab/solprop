Add explicit setup and status commands for the optional Julia engine, with pinned bridge
versions, immutable source copies and verified offline artifact inventories, as part of
[SEP 48](docs/dev/seps/sep-0048-julia-backend.md). Its numerical sessions and device
qualification remain in progress on the implementation branch.

Add owned `SimulationSettings.backend_options`, forwarded through execution options
and spawned workers without changing scientific hashes. Backends refuse unsupported
option keys before workspace construction.

Backend selection now exposes session ownership independently of the legacy numerical
operation surface. Single runs and fixed-schedule groups use their backend's own
factories, so the optional Julia adapter can keep all numerical marching in Julia.
Failed group construction preserves entrance fields and caller-owned backends for
retry ([#623](https://github.com/LupoLab/solprop/issues/623)).

The binary64 Julia session runs scalar modal fields and envelopes through the
shared event and output services, with adaptive or fixed Runge–Kutta and landed
exponential Adams stepping,
bounded attempt delivery, cooperative stops, snapshots and validated destination
exports. Kerr, Raman and plasma responses retain independent represented profiles;
operators, prefactors and power weights support constant, table and expansion forms.
The source owns bounded node scratch and refuses overlapping caller buffers before
numerical writes. Broader route, method, observation and device support remains under
implementation.

Adams preserves shared control policy, exact logical budget charges, actual source-call
counts and compact accepted-order records across event yields and resumes. Portable
binary64 snapshots transfer raw accepted state and full history stores losslessly;
compatible Julia/NumPy/native CPU continuation adds no startup charge. Field-only resets preserve
cumulative budgets and record incompatible history explicitly. The remaining restart
and device qualification matrix is still in progress.
Cumulative counters refuse before overflow or unrecordable work, preserving their
saved values ([#658](https://github.com/LupoLab/solprop/issues/658)).

The fixed low-storage RK4IP session owns its register updates and prescribed itinerary.
Its record separates attempted from committed source calls, and in-flight failures
poison the field without discarding actual work counts. Matched-history RK and Adams
comparisons use checked discrete-stage and retained-source bounds, including the effect
of windowing ([#656](https://github.com/LupoLab/solprop/issues/656)).
