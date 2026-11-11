# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`; draft [PR 611](https://github.com/LupoLab/solprop/pull/611).
Proposal 609 merged at `04e72192`; Part I is frozen. Leave the completed implementation
ready for maintainer review, without merging it. Latest inspected CI head: `118e600c`.

**Completion baseline, amended by the maintainer on 11 November:** binary64 CPU and
CUDA. Metal, ROCm/HIP and unfinished binary32/DF32 qualification are deferred. The
reviewed RK dense engine attachment is complete; numerical work now targets CUDA.
See [DEFERRED.md](DEFERRED.md) for exact status; Part II records the scope amendment.
New issues are only for work this PR will leave unresolved. Track in-PR defects here.

## Current plan and status — 11 November 2026

| Phase | Current status | Remaining qualification |
|---|---|---|
| 1: package and runtime | Immutable setup, offline artifacts, bridge and ownership implemented | Deferred runtime failures stay visible; final installed-package and supervisor matrix |
| 2: binary64 CPU | Complete scalar modal/spatial session, services, history, groups, providers and public CPU selection | Final conformance/CI and isolated startup/worker measurements |
| 3: binary32 and DF32 | Substantial internal implementation and reviewed RK dense engine attachment preserved | Remaining qualification is deferred outside the baseline |
| 4: devices | CUDA execution/FFT foundation tested; storage and complete sessions in progress | CUDA full sessions, vendor quality, measurements; Metal and ROCm/HIP deferred |
| 5: documentation and Final | Manual/design records evolving with implementation | Complete measurements, final integration review, required green gates, archival/Final commit |

No implementation phase is declared complete while its required evidence is missing.
Binary32 and GPU public admission remain closed. Every binary32 route refuses noise;
scientific defaults and hashes are unchanged. Julia 1.13.3 and bridge 0.9.37 remain pinned.

The complete preliminary binary64 CPU matrix covers 12 cases at one and 64 cores,
three paired rounds and three repetitions per arm: 144 samples and 432 repetitions.
All results are finite; accepted/rejected counts match. Timings, actual short windows,
preparation-inclusive wall and limitations are in the
[benchmark ledger](../docs/dev/benchmarks.md#julia-cpu-entrance-windows-complete-preliminary-matrix).
Results are mixed and do not support a blanket speedup claim. Issues 711 and 729
retain endpoint-reuse and radial/Cartesian profiling follow-ups.

## Active ownership

Every author uses a separate worktree. Reviewers do not review their own work.
Shared interfaces and integration remain master-owned.

| Owner | Exclusive active scope |
|---|---|
| Master | Shared interfaces, integration, CI and records; CUDA bridge, exports and services |
| Independent reviewer | Complete CPU/CUDA integration; read-only service interface planning |
| Numerical agent | CUDA Raman/plasma composition, narrow validated coefficient-copy hooks, source tests/note |
| Device agent | A40 storage qualification; binary64 Adams method/dense mirrors and tests |

The RK dense leaf `990871fb` is independently approved and integrated as `ad5ab855`,
with its reviewed engine attachment at `98f6af3e`. Python DF32 dense admission remains
closed. Adams dense is an unqualified deferred draft at `973fc351`, outside this branch;
no numerical test completed before deferral. Its intended retained polynomial and
rate-provenance design is not qualified evidence.

Reviewed CUDA storage/readback, capture guards and RK orchestration are integrated at
`75d6849a`, `ba632416` and `9e41cdb1`. Host execution-model tests pass; actual A40
storage qualification is queued, and device RK qualification remains pending.
The reviewed Kerr source, low-storage method and transactional export are now
integrated at `23fed2ab`, `900ce053`, `bbdf0eda` and `f007e1b7`. The master connects
execution selection through the bridge in a separate worktree. Actual device
qualification and the rest of the CUDA session catalogue remain required.

The RK/low-storage DF32 bridge attachment is independently approved: ten files covering
`backend.py`, `_session.py`, `_wide.py`, PythonCall launch/wide attachment, its focused
tests and `DF32_EXECUTION.md`, plus retained independent boundary and group witnesses.
It has no public admission change; shared integration remains master-owned.

CPU32 qualification is independently approved and integrated at `13835cf1` and
`5fa7294c`. All
59 distinct short cases have successful latest results; the corrected 32-case physical
spectral-band subset passes at unchanged envelopes. Issue 734 preserves the original
metric defect and the distinction from historical SEP 33 measurements.
The master adds 46 exact-rational physical-band assertions and ten managed all-route
integration cases on the new bridge, all passing. The DF32 Adams leaf is integrated
at `dc90b8ef`, independently approved for its conditional internal CPU scope. Review
passes 26,755 distinct assertions, including all-order factors, MPFR enclosures,
failure preservation, actual FFT/Kerr, chunk invariance and emitted arithmetic.
Warmed allocation assertions are zero; exact-final Aqua/JET passes 16 checks with
2,211 definitions analyzed, none cached, plus 12 concrete calls. Bridge attachment,
dense/spatial and device execution remained separate work at that leaf milestone.
The bridge/history attachment is now independently approved. A 51-case managed run
passes 23 new Adams cases and 28 existing history/RK/low-storage/group regressions;
two additional Adams group cases and six independent factory-boundary cases pass.
The groups include different plain/compensated members and refusal before a seed
source call. Completed compensated history and accepted sample/landing slack are
corrected (747, 748), without changing active numerical policy or portable bytes.
Targeted Julia checks pass 998 assertions. Final CPU Aqua/JET passes 16 checks with
all 2,217 definitions analyzed and none cached; pure boundary/CI checks pass 204 cases.
Spelling, formatting, types, SEP consistency and the Sphinx build pass. The first
source-identity and group-fixture failures remain in the record; their corrected
checks do not relax numerical envelopes. Public admission remains closed.

The six-case Adams-versus-RK campaign is committed at `e1d7b460`: argon field passes
both tolerances; helium initially failed shared translation (737), and Julia binary32 argon
envelope exhausts cycle quadrature (738). All four identical NumPy binary32 envelope
controls complete. These remain ordinary qualification failures. The capped binary64
reference pairs agree exactly but still need an independent step-size refinement.
The independently reviewed 737 fix uses the existing cancellation-free refractivity
with `log1p`, without changing physical inputs or interpolation tolerances. Its
46 changed-module cases and 743 independent Decimal/validity/table probes pass;
the unchanged helium propagation comparison still needs to be repeated.
The reduced-log plasma correction is integrated at `e690ef30` after independent
review of 37,243 distinct bounds-enabled assertions, including MPFR comparisons,
unchanged quadrature criteria and zero warmed allocations. The physical failing
neighborhood passes 1,038 checks. ADR-080 states the CPU elementary-function API
boundary and the separate device obligation (751). The repeated six-case comparison
passes four cases; both helium rows finish within every absolute envelope but fail
the unchanged twice-RK rule (753). Independent 16/32-step binary64 references differ
by at most `2.514e-15`. Candidate windows, metrics and envelopes are unchanged.
Identical NumPy binary32 helium controls pass all four method/tolerance combinations.
The separate population ownership fix (749), integrated at `cd409ee3`, refuses
foreign-header overlap and non-injective workspace destinations before writes.
Independent review passes 299 bounds-enabled assertions with zero warmed allocation;
the author also passes 22,725 directly affected assertions and CPU Aqua/JET.

CUDA foundation review corrected raw pointer capture (730), complex FFT alignment
(732) and active cuFFT workspace accounting (735). The integrated foundation at
`dfa8f656` passes 418 actual A40 assertions in eight testsets, under network isolation.
Unfiltered quality passes Aqua's 11 checks but fails core/extension JET and all six
concrete-call analyses (741); no full-device qualification is inferred. First-entry
foreign-context coexistence (733) and released vendor handle-cache retention (736)
remain explicit qualification work. Apple M5 Max probes find 20 subnormal failures
in 136 arithmetic checks despite precise compiler settings (739). Base Metal execution
and FFT work is independently reviewed and integrated at `035ee5ac`: 308 actual device
assertions pass, including 48 independent ownership, queue and transform checks.
Metal Aqua passes 11 checks, while unfiltered JET and all six concrete analyses fail
(745). Native graph-cache/workspace retention remains unmeasured (743). Independent
probes confirm normal-input cancellation and FMA residuals can flush to zero through
both owned kernel launch forms. DF32 Metal is not qualified.

## Latest validation

- CUDA source review passes 858 host assertions, including 194 independent
  composition/session/ownership checks. Low-storage review passes 179 assertions,
  including 64 independent extreme-norm, early-refusal and analytic-phase checks;
  its author passes 2,322 affected CPU assertions and Aqua/JET. Export review passes
  362 assertions after correcting explicit power weights overlapping staging storage.
  Its author passes 2,469 affected CPU assertions and Aqua/JET. No actual GPU result
  is implied by these host checks. The inherited RK positional alias fixture now
  includes its readback field; 431 affected assertions pass.
- The combined source/method/export tree passes 1,700 targeted assertions across
  41 testsets. CPU Aqua/JET passes 16 checks with all 2,294 definitions analyzed and
  none cached. The complete gate still runs in CI; device tests remain pending.
- The amended baseline selector passes 79 committed tool tests and 104 independent
  inventory/verifier checks. Actual opt-in collection selects 1,374 baseline cases
  across 55 modules and exactly 118 deferred precision cases across seven modules;
  all ten required Julia CPU conformance names remain in the baseline collection.
  Missing, skipped, failed or errored mandatory tests still fail either command.
- The integrated RK dense attachment passes 2,419 bounds-enabled assertions across
  40 testsets. The subsequent device storage/RK integration passes 1,353 targeted
  assertions; CPU Aqua/JET passes 16 checks with all 2,266 definitions analyzed and
  none cached. These are CPU checks, not CUDA hardware evidence.
- CI at `118e600c` reports 15 qualification-test typing errors, corrected and
  independently reviewed at `5a7d4300`, plus Linux package-mirror DNS failures (757).
  Local types have zero errors and three unresolved-native-stub warnings; the local
  command therefore exits nonzero. No full green CI result is claimed.
- Integrated Julia Core: 156,695 assertions in 484 testsets pass, bounds enabled,
  two threads. Aqua/JET: 16 checks pass, all 2,067 definitions analyzed, none cached.
- Reviewed binary32 composition: 84 actual managed retained-owner cases plus 27 pure
  cases pass; independent review adds 178 PythonCall factory and seven registry checks.
- Reviewed DF32 bridge: 101 typed pure tests and ten actual managed tests have passing
  latest results, including refined binary64 references, phase refusals before source
  work, completed starts and mid-plane field restarts. Six narrow-state group checks
  also pass, including both wide types and independent member failure state. Independent
  review repeats all 22 managed cases and adds the retained factory boundary witnesses.
- CPU-only CUDA artifact setup and isolated-network A40 package loading, broadcast,
  GEMM and FFT primitives pass in both precisions. These precede extension qualification.
- CI at `984444ce`: Linux/macOS Julia Core and Aqua/JET pass. macOS Python passes
  1,275 tests with 20 declared skips; verifier correction 731 passes 67 independently
  repeated tests. Linux Python passes 1,274 with one shutdown-finalization failure,
  followed by a post-summary segmentation fault (669).
- Other inspected CI lanes pass except the combined coverage report's two missing
  device-refusal lines (725, corrected without exclusions) and an unrelated Windows
  GUI event-log sharing violation after 424 passes (724). Fresh full CI runs on
  `c35e35bd`. That run exposes stale internal/public admission test expectations (740),
  corrected and independently reviewed at `f1589c25`: 55 typed targeted tests pass,
  without changing production admission. Windows GUI fails its resume test again;
  the Linux Julia job passes Core and Aqua/JET, then reaches its two-hour job limit
  during the still-progressing adapter suite (712). macOS completes 1,339 adapter
  tests with 20 declared skips and the three stale spatial-admission failures already
  corrected under 740. The next batch separates the complete engine and adapter
  stages, each with a bounded three-hour allowance. No check is omitted and no
  cancelled or failed gate is claimed green.

Local checks are targeted; full gates, slow lanes and package coverage run in CI.
Previous failed fixtures, timeouts and harness invocations remain in the evidence.
The [historical record](IMPLEMENTATION_HISTORY.md) retains detailed earlier reviews,
source identities, validation counts and ownership handoffs.

## Next actions

1. Run the integrated CUDA source/method/export host regressions and the next actual
   device snapshot; no further binary32 campaign is on the baseline critical path.
2. Complete binary64 CUDA modal source, low-storage/Adams methods, observations,
   export and shared services; integrate the selected-device bridge and registry.
3. Qualify actual CUDA kernels, ownership and complete session behavior on A40,
   including vendor quality and CPU regression checks.
4. Finish binary64 CPU/CUDA, startup, setup/depot and spawned-worker measurements
   on short windows. No AMD rental or Metal campaign is required for this baseline.
5. Reconcile final CI and SEP 47 interfaces, finish manual/design/ADR/release records,
   complete integration review, then prepare Final and mark the one PR ready.

The maintainer deferred rare runtime investigation until core implementation is done.
Issues 610, 620 and 669 retain the accepted boundaries and remaining failures; this
changes work order and does not turn a failed qualification into a pass.

## Decisions

- The maintainer narrowed completion to binary64 CPU/CUDA on 11 November. Defer
  Metal, ROCm/HIP and unfinished binary32 execution with either wide arithmetic.
  DF32 is a wide-intermediate choice for binary32 state, not a whole-state precision.
  Deferred evidence stays explicit and public admission remains closed.
- The accepted SEP's scope is retained with binary64-only input noise. Every binary32
  path, including Metal, refuses noise. No physics default changes.
- Start with Julia 1.13.3 and matching JuliaCall/PythonCall 0.9.37. Sources and
  manifests identify immutable setups; runtime execution downloads nothing.
- `SimulationSettings.backend_options` forwards `wide` and `setup_id` through the
  existing execution options. Device/options are outside science and resume hashes.
- Every Python entry into Julia is main-thread-only; Julia's own synchronous parallel
  work and spawned worker processes remain supported. The maintainer accepted this
  amendment after worker GC deadlocks were reproduced on macOS and Linux (#620).
  ADR-077 and SEP Part II record it. Runtime initialization remains serialized.
  Preserve actual Python
  SIGINT without restoring Julia's SIGINT thread mask. The maintainer accepts that,
  after Julia starts, SIGTERM may terminate the whole process without final output,
  including with custom or ignored handlers and while no session advances. Fatal
  shutdown may stall; supervisors must enforce an external SIGKILL deadline. Keep
  Julia 1.13.3; revisit a supported solution with stable 1.14 in #610. Normal
  cooperative cancellation and actual SIGINT handling retain their strict contracts.
- The ordered project stack and effective preferences are immutable setup identity.
  Adding a device publishes another identity. Only that stack and standard libraries
  enter `LOAD_PATH`; user startup files and artifact overrides cannot alter it.
- A plain Julia `@cfunction` is unsuitable for a stop listener: first entry from a
  foreign thread may allocate and wait for GC. Use a C-compatible atomic-store leaf
  without Julia runtime entry. The Python bridge owns the C byte, and Julia borrows
  its address; the standalone engine can own its own byte. The listener retains no
  Julia wrapper, whose final destruction could enter Julia on the stopping thread.
  Captured callbacks retain the byte after detachment; session close must reject or
  wait for active readers. The process runtime retains the compiled leaf's code.
- Process identity includes resolved depot, engine, executable and ordered project
  locations, not merely the content ID. Otherwise selection in a relocated depot
  could validate different files from those the running runtime will load.
- A loaded foreign `libjulia` is refused before import. Actual Julia load/depot paths,
  active project and thread pools are checked before loading the engine. Failed
  startup or device loading poisons the process; a fork after startup refuses.
- Registry handles retain Julia wrappers and borrowed Python buffers until main-thread
  release. Foreign finalizers enqueue integer IDs; shutdown releases every owner before
  JuliaCall's hook. Raw wrappers stay inside narrow bridge helpers. Error formatting,
  exception-graph clearing and unwound-frame cleanup happen on main before a fresh
  Python-only failure escapes. Recursive preparation, bridge entry and owner release
  refuse. Interrupted startup and cleanup are covered explicitly.
- Set `PYTHONDONTWRITEBYTECODE=1` before JuliaCall import for Julia precompilation
  children that embed Python. The initialized host's bytecode policy stays unchanged;
  managed PythonCall source trees must remain byte-identical after runtime use.
- Runtime keeps CPU first in the ordered environment stack. Strict precompilation
  temporarily puts the active device environment first, after proving identical global
  and per-package effective preferences, then restores the runtime order. This permits
  weak-extension dependency lookup without changing the published runtime identity.
- CUDA setup on Linux aarch64 refuses until identity distinguishes Jetson/L4T from
  SBSA. The pinned runtime and driver selectors choose different artifacts there;
  the current OS/ISA identity cannot establish portability. CPU aarch64 setup remains
  supported but unqualified, as SEP 48 specifies.
- Julia consumes only `RunRequest` policy through the generic preparation boundary.
  It never inherits the legacy driver or uses a Python-over-Julia numerical loop.
- DF32 admission bounds actual exponent spans, including replay and restored Adams
  history, over the complete represented operator interval.
- Numerical comparisons use SEP 25/33 classes, never blanket cross-backend bit identity.
  Export structure and exact controller decisions retain their exact contracts.
- Statistics consume explicit numerical inputs for the actual committed or dense field;
  they never reuse an unrelated RHS trial. Fixed packed output and notice buffers belong
  to the session. Spectral centroid and Parseval weights remain separate. In addition to
  the periodic-capable catalogue, support exactly the existing spatial plane defaults
  `beam_radius`, `fwhm_r` and `on_axis_spectrum`, preserving their plane-only eligibility.
  This small scope clarification avoids refusing ordinary admitted spatial defaults.
- Every completed trial records its actual attempt before a proposal-floor failure,
  including an Adams rejection. The existing reference/native omission is tracked in
  [#651](https://github.com/LupoLab/solprop/issues/651); controller decisions, charged
  calls and accepted state remain unchanged. Preserve the historical trace corpus.
- Binary64 Adams coefficients and transported compensation are generated in Julia.
  History retains actual predictor samples and their represented operator rates. Rate
  evaluation differentiates the same cumulative operator with explicit knot-side choice.
  The Adams norm scales by the old state; it reuses only the protected sum-of-squares
  primitives from the Runge–Kutta leaf. Carrier-frame and interaction formulations remain
  subject to the accepted admission table; no additional route is inferred from kernels.
- Measure preparation-inclusive `wall_seconds` beside kernel-window timing. Full
  gates run in CI; local qualification uses targeted, short-window tests.
