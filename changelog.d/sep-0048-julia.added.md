Add explicit setup and status commands for the optional Julia engine, with pinned bridge
versions, immutable source copies and verified offline artifact inventories, as part of
[SEP 48](docs/dev/seps/sep-0048-julia-backend.md). Its numerical sessions and device
qualification remain in progress on the implementation branch.

Enable `backend="julia"` for the admitted binary64 CPU routes. Node budgets account
for source and observation array payloads together, bind to each actual run request,
and report their fixed-storage and opaque-plan exclusions explicitly. Prepared requests
share immutable numerical data while retaining independent chunk capacities.

Use the common `prepared.start` return boundary for ordinary benchmark repetitions
on every execution owner, with an explicit timing-boundary identifier and separate
preparation-inclusive wall time. Actual source counters come from the same session;
logical-only counts remain unavailable as invocation measurements.

Add owned `SimulationSettings.backend_options`, forwarded through execution options
and spawned workers without changing scientific hashes. Backends refuse unsupported
option keys before workspace construction.

Backend selection now exposes session ownership independently of the legacy numerical
operation surface. Single runs and fixed-schedule groups use their backend's own
factories, so the optional Julia adapter can keep all numerical marching in Julia.
Failed group construction preserves entrance fields and caller-owned backends for
retry ([#623](https://github.com/LupoLab/solprop/issues/623)).

Julia fixed-schedule groups now keep member scheduling in Julia and publish complete
common windows through the shared service interface. Interrupted construction preserves
the original starts; failed publication retains chunks for retry without repeating
numerical work. Stops and failures preserve actual member positions and one shared
primary error. Mixed Runge–Kutta tableaux refuse before consuming starts
([#676](https://github.com/LupoLab/solprop/issues/676),
[#677](https://github.com/LupoLab/solprop/issues/677)).

The binary64 Julia session runs scalar modal fields and envelopes through the
shared event and output services, with adaptive or fixed Runge–Kutta and landed
exponential Adams stepping,
bounded attempt delivery, cooperative stops, snapshots and validated destination
exports. Kerr, Raman and plasma responses retain independent represented profiles;
operators, prefactors and power weights support constant, table and expansion forms.
The source owns bounded node scratch and refuses overlapping caller buffers before
numerical writes. Broader route, method, observation and device support remains under
implementation.

Scalar modal plane and periodic statistics execute inside Julia, with request-owned
catalogues, profile tables and windows. Event copies are transactional, preserve rows
after cleanup and retry undelivered chunks without numerical replay. Missing embedded
quadrature diagnostics remain NaN; closed sessions release their request preparation
([#653](https://github.com/LupoLab/solprop/issues/653),
[#662](https://github.com/LupoLab/solprop/issues/662)).

Julia also owns temporal monitoring and its warning and stop decisions. Closed
events carry detached records; failed copies can retry without duplicating rows.
Resumed runs retain the original launch, removal totals and monitor history, with
the first supplied continuation taking precedence over an output-record fallback.

Binary64 Julia input-noise sessions add the selected per-start draw only at a fresh
launch, retain it for resumed monitor floors, and export detached cumulative phase
arrays at the selected observation coordinate. Modified and binary32 noise remain
refused. Draw application protects the active-mask storage before any candidate write
([#678](https://github.com/LupoLab/solprop/issues/678)).
Field and phase caches retain their original ticket and owner across Python callbacks,
so a completed copy cannot repopulate a replacement or closed session
([#680](https://github.com/LupoLab/solprop/issues/680)).

Adams preserves shared control policy, exact logical budget charges, actual source-call
counts and compact accepted-order records across event yields and resumes. Portable
binary64 snapshots transfer raw accepted state and full history stores losslessly;
compatible Julia/NumPy/native CPU continuation adds no startup charge. Field-only resets preserve
cumulative budgets and record incompatible history explicitly. The remaining restart
and device qualification matrix is still in progress.
Adaptive Adams dense planes retain their own selected field and coordinate while the
committed march remains ahead. Power conversion and statistics use the observation
coordinate; interior fields refuse future endpoint history, and snapshots wait for
all owed planes before copying a committed checkpoint.
Cumulative counters refuse before overflow or unrecordable work, preserving their
saved values ([#658](https://github.com/LupoLab/solprop/issues/658)).

The fixed low-storage RK4IP session owns its register updates and prescribed itinerary.
Its record separates attempted from committed source calls, and in-flight failures
poison the field without discarding actual work counts. Matched-history RK and Adams
comparisons use checked discrete-stage and retained-source bounds, including the effect
of windowing ([#656](https://github.com/LupoLab/solprop/issues/656)).
