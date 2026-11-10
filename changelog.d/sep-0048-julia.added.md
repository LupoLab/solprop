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
counts and compact accepted-order records across event yields. Budget refusal retains
the accepted field; portable Adams history continuation remains under implementation.
