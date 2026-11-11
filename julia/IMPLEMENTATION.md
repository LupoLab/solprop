# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`.
Draft implementation [PR #611](https://github.com/LupoLab/solprop/pull/611).
Proposal [#609](https://github.com/LupoLab/solprop/pull/609) merged at `04e72192`.
The completed implementation is left ready for maintainer review, not auto-merged.

## Current status

| Phase | Status | Evidence or prerequisite |
|---|---|---|
| Acceptance | Complete; Part I frozen | Final head `da2e699f` passed GitHub CI and required documentation preview; merged `04e72192` |
| 1: package and runtime | Setup/package integrated; remaining runtime qualification deferred | Maintainer directs core implementation to proceed; failures remain tracked |
| 2: CPU binary64 session | Scalar modal/spatial sessions, services, providers, FFT persistence, public selection and budgeted preparation are integrated; combined materials pass | Initial CPU measurements and remaining CI corrections precede completion |
| 3: CPU binary32 | DF32 primitives, conservative function leaf and operand analysis reviewed; full execution pending | Whole-expression and device admission remain unqualified |
| 4: devices | Pending | CPU gates and one qualification pass per vendor |
| 5: records and Final | Pending | All required capability, CI and measurement evidence |

SEP 47's contract milestone is merged at `2e2490c2`. The completed Part B head,
`e9105523` from [#607](https://github.com/LupoLab/solprop/pull/607), is integrated at
`83da0ad5` under the maintainer's authorization to proceed before its CI fixes merge.
Its actual interfaces have been read: the descriptor/request split and generic
`Backend.prepare(problem, request).start(...)` are present. `assemble` returns
`Translation(problem, request)`, validation is `validate_run`, and preparation checks
`foreign_request`. Group construction now dispatches through an execution owner's factory, independently
of legacy numerical operations. The Julia owner now uses that contract directly;
public registry selection now exposes the admitted binary64 CPU composition. No new backend-name
branch belongs in the service/session dispatcher.
The subsequent Part B CI corrections at `f6921fae` are integrated at `10be2eff`.
The additional CI corrections at `55fdc423` are also integrated: they restore
request-owned arrays to the mixture baseline and remove nonportable hash literals.
The final documentation/skip corrections at `21b5808c` are integrated as well.
The final contract/coverage corrections at `f351d0a3` are integrated at `4dade535`.
PR #607 has now merged at `ae65df6a`; its complete reviewed head was already integrated.
The targeted merged selection passes 528 tests with 56 declared capability skips.
`RunSession.march_export_refusal()` is now explicit; Julia already implements it.
The legacy callers of this branch's new execution-owner protocol are corrected;
whole-package basedpyright reports zero errors and three existing native-stub warnings.
Explicit legacy selections retain their stronger type; general selection keeps the
session-only owner contract. No legacy Julia methods were added.

## Ownership

The master owns architecture, shared transfer records and session lifecycle, shared
Python integration, CI and packaging hooks, and this record. Implementation agents
receive disjoint files in separate worktrees after these rules are committed.
Assignments and ownership handoffs are recorded here before overlapping work begins.
An independent reviewer reviews each package they did not author before integration.

The maintainer explicitly moved remaining runtime investigation off the critical path
on 2026-11-10. Continue core implementation using the experiment's numerical work,
retaining the accepted main-thread boundary and visible runtime failures. This changes
the work order, not the accuracy requirements or a claim of completed qualification.

| Assignment | Exclusive files | State |
|---|---|---|
| Numerical transform/Adams agent | New `transverse_observations.jl` and focused tests; transverse decision stages in `monitor.jl`; subsequent Julia service/transport hooks in `observation_session.jl`, `monitor_transfer.jl`, narrow launch hooks in `transfer.jl`, and new `ext/transverse.jl` | Reviewed attachment integrated at `ea4b8731`, first-selection correction at `70fa4d8e`; files return to master |
| Numerical services agent | New `trial_probe.jl`, `ext/trial_probe.jl` and focused tests | Reviewed leaf integrated at `d71d29cf`; Python adapter independently reviewed; files return to master |
| FFT attachment agent (rotated from review) | `execution.jl`, `transforms.jl`, the Raman planning block in `composition.jl`, focused new tests and the small explicit-planning precompile workload; one local `windows` narrowing in `observations.jl`; a typed scalar/rank barrier in `fft_plans.jl` | Reviewed attachment integrated at `2b2ef88c`; runtime/cache integration independently reviewed; files return to master |
| Master | All shared Python interfaces, Julia module/test inclusion, transfer records, integrators/session lifecycle, repository records and remote campaign | Integration and phase 2 shared interfaces |
| Independent reviewer | Production code read-only; independent transaction tests returned to master | Group, dense scheduler/adapter and corrected input-noise leaf approved; independent integration review continues |
| Wide arithmetic agent | New `wide_operators.jl`, its focused tests and `WIDE_ASSEMBLY.md` | Phase 3 whole-expression assembly and range/error certificate; shared interfaces require master agreement before dependent work |
| CPU provider agent | New `cpu_blas.jl`, `_accelerate.py` and focused tests | Reviewed leaf integrated at `5795fe65`; runtime/lease attachment reviewed, managed checks passed; files return to master |
| Preparation agent | New `_node_plan.py`, focused Python/Julia tests and `NODE_MEMORY.md`; narrow owned-window reshape correction in `observations.jl` | Derive Julia-specific chunk storage bounds and authoritative budget admission; master owns shared adapter integration |
| Source precision agent | `precision.jl`, `composition.jl`, `spatial_composition.jl`, `responses.jl`, `raman.jl`, `plasma.jl` and their focused tests | Binary32 source preparation/kernels under the shared `PRECISION.md` interface; no Python, transfer, session or observation edits |
| Observation precision agent | `observations.jl`, `spatial_observations.jl`, `transverse_observations.jl`, `statistics.jl`, `monitor.jl`, `observation_precision.jl`, `power_weights.jl` and focused tests | Binary32 pointwise work, protected reductions and host physical scaling; no source, Python or scheduling edits |
| Device-controls/review agent | Python execution settings/options, registry, host/native guards, recipe execution transport, runner compatibility and focused tests | Public CPU preparation independently approved; implementing device controls with Julia devices still refused; then review master march precision |

The precompile workload uses only owned, small CPU arrays and existing numerical
constructors. It covers serial CPU session/source/observation dispatch without loading
Python or devices, retaining no global mutable workspace and publishing no calibration
files. Parallel specialization and cold-cache evidence are qualified separately; the
workload never changes the process's thread policy. Module inclusion remains master-owned.

Phase 3's wide-operator slice will use paired Clenshaw differences on a single
represented interval. This evaluates the same cumulative polynomial difference while
retaining its correlation across endpoints, so a forced small step does not divide
two unrelated cumulative-error bounds by the step length. Actual rounded endpoint
distance remains distinct from the prescribed step. Cross-knot jumps and restored
sampled-rate provenance remain explicit certificate inputs. The first certificate
implementation is a conservative CPU operation check, not public DF32 or device
qualification; its per-cell cost and range limits require separate evidence.

Spatial observations reuse the existing statistics and temporal-monitor reductions.
Their separate flux descriptor retains the actual represented material and protected
longitudinal-root inputs. Cartesian reconstruction uses transforms, never a dense
node-to-node map. Coarse electric/closure pairs are allocated only for selected needs;
fine population scratch is bounded by the node chunk. The master retains ownership
of Python lowering, preparation factories and shared session-service hooks.

The temporary CI assignments for `restore_control!`, CPU allocation and source/monitor
constructors are complete and integrated after independent review. Their files return
to master ownership; further leaf changes require an explicit assignment.
The portable history leaf, sample tags and profile lowering also return to master
ownership after review. Observation preparation is request-specific: its catalogue,
monitor policy and profile context do not enter a problem-only preparation cache.
The dense-observation leaf is independently approved and integrated at `2660ce42`;
shared scheduler attachment remains master-owned. The history bridge milestone
`12613fc5` is pushed to the draft PR for full CI.
The DF32 scalar primitive commit `dff18fd1` is integrated as `1e9bc6a4` after
independent source and algorithm review. The reviewer passed 33,651 assertions,
including additional exact split/tie cases and CPU IR checks. These qualify only
the scalar operations under their stated finite-intermediate hypotheses, not a
propagation domain or vendor arithmetic. The monitor continuation fix `2216949e`
is integrated as `f54aff0b`; independent bounds-checked testing passed 3,884
assertions, including recorded Python readings and zero-allocation checks. Missing
legacy quiet counts preserve the new launch settlement; explicit zero restores it.
These two filesets return to master ownership after review.

The request-specific statistics bridge has passed integration and independent review.
Its first runtime pass found a PythonCall automatic-array-conversion dispatch gap;
a distinct Python copy entry now transposes the packed destination explicitly.
Transactional retries, buffer alias/admission and retained-array cleanup now pass.
Admitted spatial routes, preparation and measurements still block phase 2 completion.
The subsequent phases continue
without another approval step; the completed implementation PR remains unmerged.
The reviewer-authored statistic-bound helper is integrated at `8e8aa1c6` after
master review. It uses exact rational reduction bounds, a declared CPU FFT envelope,
and certified width brackets/slopes. Twenty-eight independent witnesses cover its
admission and 1,024 width perturbation corners. It does not bound material populations
or propagated-field errors. The master-owned group hooks are committed at `81b0bdde`;
the services agent independently passed all 55 new bounds-enabled assertions.
The group coordinator keeps Julia in charge of member advancement; Python assembles
only complete transport windows. Fresh entrance starts are required by the shared
group contract; portable-history groups are not admitted.
The numerical group leaf `cb8ace0e` is integrated at `340ea43c`. Independent review
passed 639 bounds-enabled assertions, including original witnesses for nonprogressing
near-coincident planes (#666) and zero-trial deadline turns (#667). Warmed advancement
allocates zero; the author's package/concrete quality checks pass. Its public compatibility,
transactional original-start adoption and complete Python windows have since passed
the integration milestone below.
The DF32 function API admits only the documented `abs(real(z)) <= 16`,
`abs(imag(z)) <= 2^20` domain with a separate `u/8` input-error budget. Protected
exponential, phasor and phi helpers implement that derivation;
neither whole-run admission nor device qualification is inferred from this decision.
The protected function leaf `5db7d4de` is integrated at `93ba2e53` after independent
review. The exact committed source passed 66,336 bounds-enabled assertions at two
threads, zero warmed allocations and 16 Aqua/JET checks over 1,063 definitions.
Independent qualification passed 157,543 assertions, including 78 emitted-code
checks and the separately reviewed finite-underflow proof. These results establish
the documented conservative CPU arithmetic domain only. Actual replay spans,
restored Adams history and represented-operator operand bounds are the next
certificate inputs; public binary32 admission and each device remain unqualified.
Operand analysis `1ea43509`, `9ed0d2a4` and `59252a3f` is integrated at
`1f0c3912`, `9ced7ba5` and `410442e0` after independent review. Sixty-five committed
tests and three independent rational suites pass, including 60 signed large-node
replay geometries checked through both APIs, 810 adaptive stage endpoints, and exact
polynomial/rate/jump bounds. Positive restored sample offsets and multiply/add stage
rounding are included (#675). This is preparation-time operand analysis only; full
input assembly, Clenshaw/subtraction/compensation and vendor arithmetic obligations
remain explicit. It does not enable binary32.
The Python group adapter is independently approved on the reviewed Julia coordinator. Julia
chooses every member turn; Python copies bounded transports and joins each member
chunk exactly once. Core group acknowledgment leaves member barriers held until the
next Julia turn. Starts and owners transfer only after all member construction succeeds.
Dense Adams attachment is independently approved with explicit observed-field/coordinate
selectors while committed state stays monotone. Endpoint periodic rows accompany their
accepted attempt before dense planes; snapshots follow all owed planes. The common
conformance suite exposed and corrected the former row deferral in #687.
Shared replay-plus-interpolation refusal remains
unchanged.
The temporal monitor transport `a7d77050` is integrated at `8c625d0b`, with its
Python publication/service adapter independently approved. Julia owns decisions and
totals; Python retains detached rows and formats the existing record/messages.
The first continuation is committed atomically with its numerical restore. A fallback
after launch acknowledgment retains the last published scalars until the next valid
generation. External attempt/start callbacks can request a snapshot while recursive
advance, acknowledgment and close still refuse. Exact regressions cover #672 and #673.
The new managed monitor/conformance selection passes 35 cases: eleven monitor cases
and 24 common contract cases. Thirty skips name unfinished dense/transverse capabilities
or inapplicable bitwise repetition; they do not qualify those missing capabilities.
The integrated bounds-enabled Core passes 110,066 assertions in 224 sets. The 240-test
typed affected selection, scoped types, docstrings, spelling and Sphinx pass. Integrated
Aqua/JET passes all 16 checks over 1,169 definitions. All 53 previous real core, history
and statistic bridge regressions also pass. The numerical monitor leaf's separate
Aqua/JET and bridge checks passed.
Linux CPU qualification of `23f38f03` passes all 53 bridge/history/statistic cases at
one thread (215.32 seconds) and eight threads (1,011.55 seconds) on an eight-core
allocation of AMD EPYC 7513 hardware with Julia 1.13.3 and GCC 16.2. The allocation
is shared; these preparation/compilation-inclusive test durations are correctness
evidence, not comparative kernel benchmarks. The job completed successfully and its
evidence was retrieved. Cloud spend remains zero.
The previous pushed head `5c18bb90` still fails Julia Linux and combined coverage in
GitHub CI. Coverage is down to four statements and four branches in group-owner
refusals; targeted regressions are next. No full-gate success is claimed.

The foundation is committed at `540794bd`. Setup commits `6c569742`, `fadae531` and
`52269788` were independently reviewed and integrated as `9fe74c26`, `5cec895c` and
`5264160b`. Julia package commits `28fb628e` and `201d91d5` were independently reviewed
and integrated as `ea847fcd` and `773bd539`. Updated main is merged at `e2df6c7c`.
Issue [#610](https://github.com/LupoLab/solprop/issues/610)
tracks the accepted signal-handler limitation and intermittent SIGTERM-during-GC
shutdown hang. The maintainer accepted an external hard-kill deadline and then deferred
further runtime investigation until after core implementation. Earlier failures remain
preserved. The supervisor tests now exercise the accepted TERM-then-KILL deadline;
successful forced termination is recorded separately from voluntary Julia shutdown.

## Decisions

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

## Next actions

1. Finish independent review of request-specific node preparation and collect the
   common benchmark window's actual Julia smoke evidence.
2. Retrieve the one/eight-thread Linux qualification, address required CI findings,
   and complete the initial CPU startup, preparation and short-window timing record.
3. Complete binary32 execution and whole-expression DF32 bounds, then each vendor's
   public selection/qualification and the phase 5 documentation/evidence gates.
4. Revisit deferred runtime issues after core implementation; preserve all failed
   evidence and complete the accepted supervisor qualification before claiming it passed.

## Verification log

- Acceptance revision: changed-document codespell and `git diff --check` pass;
  `pytest tests/test_seps.py`: 349 passed, 3 expected process-SEP skips;
  Sphinx builds with warnings as errors in a lightweight environment.
- Independent acceptance review completed. Its requested setup-selection tests and
  backend-options phase assignment are included in `5fb24689`.
- Independent Phase 1 architecture review identified environment-stack leakage,
  artifact-selection hooks, partial initialization poisoning and the foreign-thread
  stop-setter constraint above. The macOS arm64 standalone LLVM leaf probe executes
  a release store correctly; full lifecycle, signal and threaded-GC probes remain.
- Separate macOS arm64 subprocess probes with Julia 1.13.3 and JuliaCall 0.9.37:
  foreign-thread atomic cancellation during threaded GC passed, as did restored
  custom Python SIGINT. Ignored and custom SIGTERM, restored either immediately or
  after threaded work, all terminated the process. Julia's kqueue subscription
  receives SIGTERM before ordinary signal disposition. Disabling Julia handlers
  with two threads crashed during shutdown. The maintainer explicitly accepted the
  SIGTERM limitation and kept 1.13.3; no signal-disabling workaround is adopted.
- Python setup: 175 targeted typed tests, complete owned statement/branch coverage,
  ruff, basedpyright, numpydoc and spelling passed before integration. The final
  symlinked-home correction passed its seven affected tests. Independent review
  corrected loader aliases, workspace redirection, publication races and wheel pairing.
- Julia package candidate `28fb628e`: 2,299 Core assertions, Aqua/JET, Runic and bridge
  ownership checks passed. An empty CPU depot installed 83 packages and 19 artifacts;
  offline repeat, JuliaCall use and another verification produced identical inventories.
  CPU-plus-Metal strict precompilation initially failed on LLVM's BFloat16s weak
  extension. The reviewed setup-only ordering correction passed 2,315 Core assertions,
  Aqua/JET, Runic and 17 bridge checks. A fresh CPU/Metal stack precompiled 110
  dependencies; its Metal inventory contains 87 packages and 19 artifacts. Inventories
  stayed identical after offline JuliaCall use. GPU execution is not yet qualified.
- Master runtime: 35 targeted typed tests and basedpyright passed, covering initialization
  races, poisoning, identity/location pinning, actual load paths, and stop lifetime.
  Packaged subprocess, native-coexistence and device tests remain outstanding.
- Integrated Python setup/runtime: 220 targeted tests passed. CLI, recipe and resume
  integration: 78 passed with two expected skips. Julia Core tests including the atomic
  stop byte passed with bounds checking; runtime docstrings passed. Native coexistence
  awaits a matching native build; no source-digest guard is bypassed.
- Foundation checks: ruff and whole-package basedpyright pass; the targeted Python
  setup/runtime and SEP consistency selection passes 569 tests with three expected
  process-SEP skips. This is not numerical parity evidence.
- Independent wheel review passed direct and source-archive builds, installed source
  pairing and lazy import/status. Its incidental-file inclusion finding is corrected
  with a shared declared-input list and a regression; independent injected-file direct
  and source-archive wheel rebuilds now exclude all sentinels and have identical Julia
  payloads. This smoke used an isolated no-dependency install with existing dependencies;
  fresh dependency resolution remains in CI. The public setup API page fixes three unresolved
  function references; Sphinx now passes with warnings as errors.
- Julia CI commits `48714ac3` and `ebd239bf` were independently approved and integrated
  as `8e584d24` and `90453af2`. Their runner tests, static checks, actionlint and change
  selection scenarios passed. Existing device-filter robustness is tracked separately
  in [#612](https://github.com/LupoLab/solprop/issues/612).
- After integrating SEP 47 Part B, 268 targeted Julia/CI-runner/request tests passed.
  A fresh native build completed; whole-package types remain blocked by the dependency
  errors above, pending that PR's CI fixes.
- Packaged macOS runtime gate: the first 12 subprocess cases pass with no skips,
  covering main-thread startup, concurrent initialization, one/two-thread GC and GIL
  progress, raw foreign-thread stopping, last-owner reclamation, shutdown and offline
  preflight. A further 14 signal/spawn cases and four native-coexistence cases passed.
  Native-before-Julia and Julia-before-native both retained real FFT plans at one and
  two threads through clean shutdown against SEP 47 Part B.
- Worker-first Metal loading timed out at 180 seconds. Native stack sampling places
  the worker in Julia's package-image global safepoint while Python's main thread is
  joining it. A GIL-free loader alone also hangs at one and two threads. This is a
  blocking runtime qualification failure, separate from the accepted SIGTERM policy;
  ordinary worker computation/GC and any proposed fix require independent probes.
- Independent CPU-only GC probes reproduced the same worker deadlock at one and two
  Julia threads on macOS. Internal main-thread GC parking still deadlocked under GIL
  contention and is rejected. The maintainer accepted the main-thread boundary; 60
  targeted owner/runtime/stop tests pass, including failure graphs, interrupted hook
  registration, recursive entry and cleanup failures. Real runtime requalification
  remains in progress; no worker support is claimed.
- The main-thread boundary passed independent code review and its deterministic
  failure reproductions. All 597 targeted Julia/SEP tests pass, with three expected
  process-SEP skips; targeted types, formatting, docstrings and spelling pass. The
  current Sphinx build is blocked by two references in unchanged SEP 47 Part B files
  (`driver_session.foreign_request` and a malformed `spatial` run-request attribute
  link), alongside that dependency's outstanding type fixes. No Julia documentation
  warning remains; integrate the dependency's corrections before qualification.
- Linux x86-64 package qualification at `83da0ad5` passed fresh installation, native
  coexistence prerequisites, Runic, 2,328 bounds-enabled Core assertions and 13 Aqua/JET
  checks. Hardware: AMD EPYC 7513, four physical CPUs with SMT disabled; GCC 16.2.0,
  x86-64-v3 native target, Julia 1.13.3 with generic cached code. A separate two-thread
  CPU diagnostic completed its main call in 0.183398 seconds but timed out its worker
  call after ten seconds, corroborating #620. These are runtime probes, not benchmarks.
- Polyester 0.7.19's macro can confuse a local `values` argument with `Base.values`:
  two-thread macOS arm64 execution fails in standalone Julia and JuliaCall. Renaming
  the local to `buffer`, or using an explicit singleton callback, passes at one and two
  threads. [#621](https://github.com/LupoLab/solprop/issues/621) records the twelve-case
  probe matrix and requires a production-loop regression when that wrapper lands.
- Read-only donor audit reproduced a non-THG multimode Raman out-of-bounds padding
  read under bounds checking. [#613](https://github.com/LupoLab/solprop/issues/613)
  tracks its correction and regression in the production port; the prototype is not
  accepted as proof of correctness.
- Additional planning review requires conditioning-aware bounds near zeros of phi
  functions and corrects Metal's portable sine/cosine allowance to 4 ulp. Experimental
  witnesses remain measurements; algorithm and admission constants are not qualified.
- The primary DF32 papers and authors' formal supplement were checked. The planned
  family is AccurateDWPlusDW, DWTimesFP3, DWTimesDW3 (including low-by-low), DWDivFP3
  and DWDivDW3. The last algorithm's naming typo is resolved by its formal proof;
  underflow/overflow guards and transcendental bounds still need qualification.
- No propagation or GPU qualification is claimed yet.
- Main-thread runtime milestone `fda416dc` is committed. Independently reviewed
  subprocess probes `2fedad6b` were integrated as `a01efd1f`; their 48-case matrix keeps
  every requirement active. The first revised macOS run passed 18 cases before a
  one-thread, ignored-SIGTERM-during-GC case hung for 180 seconds and needed supervisor
  SIGKILL. The original failure remains visible.
  Julia's pinned fatal-signal source explicitly anticipates asynchronous shutdown
  stalls. A control with the Python owner-cleanup hook unregistered also stalled and
  exited after a second SIGTERM; five companion controls exited on the first signal.
  The maintainer explicitly accepted external hard-kill supervision (ADR-078), then
  deferred further runtime investigation until after the core implementation.
- The revised main-thread macOS non-SIGTERM matrix passed all 36 cases, with no skips,
  in 163.40 seconds. It covers one/two threads, cancellation during Julia and Python GC,
  owner destruction, sanitized failures, shutdown, exact SIGINT counts, native loading
  in both orders, actual Metal import, offline preflight and spawned workers. This does
  not qualify numerical parity or the revised SIGTERM supervisor policy.
- Owned backend-option transport and recipe metadata pass 483 directly affected
  tests, including spawn/pickle, post-translation replacement, unchanged scientific
  hashes, early unsupported-option refusal and recipe/Python export. Targeted types,
  ruff, formatting, changed-file spelling and diff checks pass. SEP consistency passes
  349 tests with three expected process-SEP skips; the incremental Sphinx build passes.
  This does not clear the separately recorded whole-package dependency failures.

- Owned backend options at `d1feaa15` passed independent review, 483 targeted tests,
  targeted typing and formatting. Scientific hashes and recipe/spawn transport retain
  their intended separation; unsupported option keys refuse before workspace allocation.
- Execution-owner and transactional group construction passed independent review and
  170 targeted session/group/preparation/frozen-interface tests at one native thread,
  with no skips. Targeted Python and native adapter typing pass. Issue #623's regressions
  preserve entrance fields and caller-owned backends on failed construction, including
  a second native child allocation failure followed by a successful retry.
- The historical Linux packaged runtime job at `a01efd1f` completed with 273 passed,
  23 failed and zero skipped tests. Logs and XML are retained; remaining runtime
  investigation is deferred as directed. No additional runtime job was submitted.
- CPU execution and packed-array protections are integrated at `3cfd932b`,
  `d93590f9` and `ba7bee7a`; transforms/tables/operators at `92b81eba` and `037f20e8`.
  Independent review approved their one/four-thread bounds checks, reference DFTs,
  polynomial/operator witnesses and zero warmed allocations. Issues #624 and #626
  record the packed-storage and temporal-buffer corrections.
- Reviewed Kerr/Raman/plasma leaves are integrated at `94803513` through `fd313d8f`.
  Their 282 assertions passed at one/two threads; independent actual NumPy comparisons
  passed 23 assertions with maximum scaled difference 2.73e-16. Response quality passed
  12 Aqua/JET availability checks and nine concrete JET error analyses. The Raman
  padding and plasma scratch regressions address #613 and #622.
- The independently reviewed binary64 Runge–Kutta leaf passes 163 bounds-enabled
  assertions at two threads, including convergence, stable norms and zero allocations.
  Scalar control passes 55 unit assertions and 418 same-host exact recorded-reading
  assertions, including all three transactional floor failures. History restoration
  now retains its total accepted count explicitly.
- The first constant scalar envelope source composition passes independent DFT and
  short nonlinear-phase convergence witnesses, with zero warmed allocations. Independent
  review also compared the actual NumPy passes at one/two threads with fourfold padding,
  two modes, three nodes, nonunit folded taper and Kerr density: 40 assertions pass.
  This composition alone does not admit the remaining routes or session services.
- Integrated package Aqua/JET passes 13 checks using the prepared shared dependency
  cache. Earlier attempts with an incomplete isolated cache failed to load matching
  dependency/stdlib images; these were tooling failures, not suppressed quality findings.
- The first complete binary64 scalar Kerr/RK composition passes nine managed-runtime
  integration tests on macOS arm64 at two Julia threads. Fixed replay uses an
  operation-derived rounding budget; adaptive output meets a fixed global accuracy
  envelope against a refined NumPy reference. The 128/256-step reference difference
  is 9.04e-13 relative over two centimetres; a separate 256/512 refinement gives
  1.35e-14. Each reference window finishes in seconds. Shared output service tests
  cover both retained and field-free output, without legacy Julia methods.
- Session-boundary and lowering tests pass 36 cases, including receipt/generation
  refusal, exact-once undelivered chunks, retained exports, recursive callbacks,
  interrupted acknowledgment and cleanup failures. The standalone march/transfer
  tests pass 181 assertions. Export conversion passes 16 independent exact NumPy
  recipe witnesses, including signed-zero preservation (#633).
- Adams and low-storage leaves pass independent actual-NumPy witnesses: 144 Adams
  assertions over orders one through six, and 32 low-storage assertions over one/eight
  steps and task budgets one/two. Focused allocation tests pass 278 Adams and 69
  low-storage assertions. Temporal monitoring passes 94 component assertions and
  1,552 identical recorded-reading assertions. Review corrected immutable bucket
  aliasing (#634); integrated precompilation corrected a helper-name collision (#636).
- The execution-owner type integration passed 928 affected typed tests and 13 helper
  cases; final legacy-selector deltas passed 129 and 44 cases. Independent review
  found that the session's started notification follows launch delivery and therefore
  cannot replace the benchmark's earlier load timestamp. The established benchmark
  window is preserved; execution-only backend timing remains an explicit pending
  capability tracked in [#638](https://github.com/LupoLab/solprop/issues/638).
  The directly affected CI/benchmark selection passes 74 tests. Public and
  internal adapter docstrings, scoped types, formatting and spelling pass. The refreshed
  Sphinx build passes with warnings as errors. Independent review approved this first
  CPU slice; the restored legacy benchmark tests pass 34 cases and both shared output
  modes pass again. Final boundary/runner selection passes 77 cases; scoped typing has
  zero errors or warnings. Complete integrated package quality now passes all 13
  Aqua/JET checks after using prepared, matching dependency images. Earlier child
  precompilation errors remain recorded as tooling failures.
- Transverse monitoring is integrated at `187504ec` and `cae3e377` after independent
  review. The final bounds-enabled two-thread monitor selection passes 191 assertions;
  the combined numerical and actual-oracle witnesses pass 304 and 4,345 assertions.
  Mismatched face metadata now refuses before mutation in both directions (#637).
- Modal observation interfaces are frozen before implementation: separate coarse
  analytic statistics, coarse real/envelope monitoring, and fine physical population
  reconstructions. Split monitoring captures the masked pre-split candidate, then
  consumes it after split damping, final masking and commit. Numerical observation
  work owns scratch only; session glue owns bounded rows, event cursors and delivery.
- The shared real power-weight evaluator is integrated at `92bb693a`: 61 bounds-enabled
  assertions, independent review, zero warmed allocation, Aqua and concrete JET checks
  pass. Destination exports and statistics use this one represented-value evaluator.
- General scalar modal sources are integrated at `12cc9f41` and `76331571`, after
  independent review. The final one/two-thread selections pass 1,027/1,981 assertions;
  240 actual NumPy witnesses pass, including field/envelope, each response, independent
  profiles, three prefactor forms, chunk remainders and in-stage absorption. Aqua and
  eight concrete JET checks pass. Review corrected operand overlap (#639) and mixed-
  profile allocations (#640), with zero warmed allocation preserved. A cancellation
  regression verifies the strengthened absolute-input contribution envelope.
- Full modal Python lowering/bridge integration passed independent review. The former narrow
  envelope source is retired; its defining-DFT, ownership and nonlinear-phase tests now
  exercise the general source. The carried source/march/transfer selection passes 309
  bounds-enabled assertions. Pure lowering/session-boundary selection passes 41 tests;
  scoped typing has zero errors or warnings. Nineteen managed-runtime tests pass on
  macOS arm64 with two Julia threads, including all three scalar response families on
  both grids, represented varying operators/prefactors, and actual-position power
  exports. Short material runs compare eight steps with sixteen/thirty-two-step NumPy
  references under explicit accuracy and refinement envelopes. No long propagation
  is part of this qualification. Sphinx passes with warnings as errors.
- Independent review corrected destination overlap with any retained numerical buffer
  before weight evaluation (#642), and duplicate accounting of Julia-owned stop
  storage (#643). Views and reinterpreted arrays are covered; borrowed stop storage
  remains counted. The reviewer passed 111 transfer assertions and the original three
  reproducer assertions after the fixes.
- CI at `5ae085f2` exposed an unqualified test package's `conftest` collision and
  Linux package-wide JET findings. Adding a package identity resolves the three-file
  fixture collection; the narrow runtime selection collects all 31 tests. Owned JET
  corrections are in progress and remain blocking. The earlier local Aqua/JET pass
  does not qualify the failing Linux lane.
- The modal bridge milestone is committed at `4dfa7ab2`. Pure Adams policy is
  integrated at `71289bf2` and `b4df31bf`: 3,754 bounds-enabled checks and 33,003
  exact assertions on captured reference decisions passed independent review. The
  shortened-run admission retains later operator kinks without forcing later steps
  (#644). The session composition, rather than the policy leaf, remains pending.
- Reviewed CI corrections at `1047779a` and `0903c874` constrain CPU allocation and
  parallel dimensions, preserve overlapping controller-history restoration, and bind
  transverse-state construction to prepared face metadata (#645). Their focused
  four-thread and two-thread selections pass 585 and 195 assertions respectively.
  Remaining package-wide JET findings are still being corrected. Quality runners
  must import JET before the engine so definition tracking is active; the direct
  runner now follows the same order as CI.
- Prepared factories now report their owned run-transfer arrays once and sessions
  retain only the execution owner after copying policy into Julia (#646). Independent
  review and the 43-test boundary/lowering selection cover multiple factories,
  dropped factories and nonempty shared controller weights.
- Shared CI integration corrections (#648) preserve lightweight parser/calibration
  imports, validate legacy choices in native measurement scripts, reflect inherited
  methods and Julia modules in structure diagrams, and register exact justified skip
  reasons. The 76-test CLI/calibration/collection selection and 364-test figure
  selection pass. Native script types and focused Python types report zero errors;
  Sphinx passes. Windows subprocess confirmation remains a GitHub CI check.
- The shared internal landed-session lifecycle is frozen in `SESSION_DESIGN.md`.
  Independent review approved the `_MarchSession` dispatch and unchanged launch
  validation extraction; 309 source/march/transfer assertions pass with bounds and
  zero warmed advancement allocation. The numerical agent owns the next independent
  Adams session composition; the master retains shared lifecycle and Python integration.
- Modal observation composition is integrated at `cdc657bb` and `a6a357e9` after
  independent review: 1,269 focused assertions include actual NumPy comparisons and
  warmed allocation checks. Population scans use only rates and preionisation,
  independently of current/loss algebra (#647), and reject raw nonfinite exposure
  before flooring (#649). Periodic rows and monitor event transport remain the next
  session integration, not a completed public capability.
- The combined package, including observations and Adams policy, passes all 16 blocking
  Aqua/JET checks on macOS arm64. JET analyses all 864 definitions. The quality runner
  now imports JET before the engine and rejects empty definition tracking explicitly;
  earlier local passes without this evidence do not substitute for this check. Runic
  passes on the changed integration files. Linux confirmation remains in GitHub CI.
- Latest dependency skip/documentation integration passes 37 targeted tests with three
  expected skips. Whole-package basedpyright reports zero errors and three existing
  native-stub warnings; the final structure/session-boundary selection passes 391
  tests. The full CI rerun awaits the next batch push.
- After publishing a fresh immutable setup containing the combined package, all
  nineteen real bridge tests pass again in 51.05 seconds on macOS arm64 with two
  Julia threads. This rechecks actual source construction, advance and export after
  the JET corrections and shared lifecycle integration.
- The common scheduler now resumes warnings at launch, accepted endpoints, plane
  preparation and finish without repeating numerical work or losing a pending plane
  or snapshot. Candidate finishing receives the actual landing endpoint between its
  initial and final masks. The 425-assertion source/march/transport selection passes;
  independent review adds a seventeen-assertion consecutive-warning witness (442
  total). Zero warmed no-service advancement allocation is preserved. Combined
  Aqua/JET passes sixteen checks, analysing all 874 definitions. Real monitor bridge
  attachment is still pending; this is lifecycle and numerical-hook evidence.
- The batch at `aaec5e92` is pushed. GitHub's static lane confirms typing but finds
  missing docstrings on two overload stubs, where Ruff D418 forbids docstrings. Narrow
  GL08 exceptions explain that boundary and leave the implementation fully checked;
  both scoped validators now pass. Other full CI lanes are still running.
- Independently reviewed Adams session `fc121ef5` and observation services `86a39425`
  are integrated at `ce20d587` and `4a1171ca`. The reviewer ran 648 bounds-enabled
  assertions, including actual Adams with split damping and all warning, deferred-row,
  plane, snapshot and completion continuations. The combined witness retained zero
  warmed advancement allocation. Separate NumPy split witnesses also pass.
- The Python Adams integration now copies the complete shared policy once, preserves
  actual replay steps beyond the adaptive ceiling, exports cumulative logical and
  actual work counts, and copies each pending accepted-order chunk before acknowledgment.
  A single immutable accounting publication binds its generation and compact order
  record, so interruption or failed-copy retries cannot duplicate the chunk. Every
  retained order buffer appears in the memory ledger. Portable Adams continuation is
  explicitly refused until its history transfer is complete.
- A fresh immutable setup passes 24 real Python/Julia bridge tests in 65.84 seconds on
  macOS arm64 with two Julia threads. New witnesses cover adaptive Adams against a
  converged independent RK reference and exact call-budget failures for both active
  and structurally absent sources, at work limits one and 256. The focused Python
  lowering/boundary selection passes 60 tests with runtime type checks, including
  interruption immediately before and after atomic publication; changed adapter types
  report zero errors.
  The combined Julia source/march/Adams/transfer selection passes 911 assertions;
  all 16 Aqua/JET checks pass with 968 definitions analysed and none cached. Sphinx,
  scoped docstrings, Runic, changed-file spelling and adapter types pass. Independent
  bridge review is complete with no remaining code blocker. Broader fixed-history
  parity witnesses remain a subsequent qualification slice. The 364 structure-figure
  assertions also pass after the new private accounting record.
- Modal statistics audit finds all 17 publicly admitted periodic names represented.
  Remaining work is lowering and delivery, the accepted-source quadrature alarm and
  monitor restore distinctions. Unavailable quadrature diagnostics must preserve NaN
  without permitting nonfinite physical fields (#653). A resumed monitor with no old
  record and an old record without quiet counts need distinct restoration inputs.
- GitHub confirms native CUDA/HIP compilation, sanitizers, tidy and three wheel targets
  at `aaec5e92`; the full matrix remains incomplete. Its native Python surface test
  exposed a missing `start_group` contract entry (#654). The entry and missing-member
  regression now pass seven targeted tests; the shared frozen signature was already
  correct. The registry overload-docstring correction also remains in the next push.
- Adams bridge milestone `50b369f9` and native contract correction `441b47fe` are
  pushed to the draft PR. The subsequent native quality lane passes. GitHub found one
  test-only type-narrowing error on `ResponseBudgetError.field`; an explicit ndarray
  assertion now preserves that API check. Whole-package typing reports zero errors
  and three existing native-extension source warnings; the scoped adapter check has
  zero errors or warnings. GitHub uses the freshly built extension for its full gate.
- Reviewed low-storage `5cfb3577`, unavailable-diagnostic correction `0739d9b8` and
  stable quadrature leaf `78ffd086` are integrated at `b29849cc`, `a54b54ab` and
  `b0a024d5`. Independent qualification passes 428 low-storage assertions, 144 focused
  statistic assertions and 685 quadrature assertions, including a 256-bit norm oracle.
  The alarm leaf is available; accepted-source scoring and notice attachment remain
  master integration work.
- Low-storage bridge review is complete. A fresh managed setup passes all 33 real
  bridge tests in 72.50 seconds on macOS arm64 with two Julia threads, including an
  actual first-source overflow with attempted count one, committed count zero and no
  field ticket. Sixty-two typed boundary/lowering tests pass. The combined numerical
  selection passes 2,410 bounds-enabled assertions; all 16 Aqua/JET checks pass with
  994 definitions analysed and none cached.
- Fixed-history Adams comparisons at orders one, five and six now use input-only
  positive polynomial-product envelopes, separate retained-source error propagation,
  and a verified Wiener-norm radius. Review found that the earlier RK helper assumed
  an analytic soliton peak without bounding discrete stages (#656). Both RK and RK4IP
  now use absolute-tableau stage majorants and a verified radius with explicit
  transform, combination and phase-formation budgets. All ten affected comparisons
  pass with unchanged precomputed bounds; adaptive runs keep their independent
  converged-reference envelopes. The time-window multiplier is included explicitly.
- Portable Adams sample identity means accepted versus predicted, not charged-call
  ordinal. The next transfer slice uses explicit integer tags mapped to the existing
  strings, exact full store copies, oldest-first selected positions and zero-based
  portable slots. Only a fresh unadvanced session can restore; compatibility refusals
  precede all writes. Python retains history-key, precision, noise and ledger policy.
- Portable history leaf `f6849fa0` is independently reviewed and integrated at
  `aa06f03d`, following the reviewed sample-kind correction. Its two-thread,
  bounds-enabled selection passes 1,034 assertions, including plain/compensated
  continuation, full inactive-row copies, signed zeros, alias/lifecycle refusals and
  zero warmed resumed-advance allocations. Its quality run passes all 16 checks with
  997 definitions analysed and none cached.
- Profile-statistic lowering `b8e3bd10` is independently reviewed and integrated at
  `d7fe0ee7`. Thirty-eight focused typed tests pass independently; the author's broader
  descriptor selection passes 54. Profiles retain their actual units and measured
  approximation records without retaining originating callables. Extreme-range shared
  Chebyshev transform overflow remains tracked in #657.
- The Python portable-history bridge passes twelve actual managed-runtime tests:
  Julia-to-Julia and both directions involving NumPy/native CPU, exact raw/full-row
  transfer, foreign-store resets, overlapping destination refusal, and restored
  budget exhaustion without a trial. Twenty-two pure history tests and the existing
  64 typed boundary/lowering tests pass. A trillion-step compact order record restores
  in one pair without per-step expansion. Broader dense/restart qualification
  remains pending; no binary32 history is admitted by this binary64 composition.
- Independent bridge review found that a restored int64-limit counter could wrap on
  its next increment (#658). Checked cumulative arithmetic and pre-trial capacity
  checks now preserve the final representable trial and refuse subsequent work before
  source evaluation. Independent witnesses pass 32 additional assertions. The final
  numerical aggregate passes 3,103 bounds-enabled assertions; all 16 Aqua/JET checks
  pass with 1,014 definitions analysed and none cached. Forty-two real core/history
  tests pass together in 79.29 seconds, with the two native continuation witnesses and
  compensated-history transfer passing in subsequent focused runs. The offset-view
  alias regression also passes. The combined typed Python/CI-runner/SEP selection
  passes 514 tests with three expected process-SEP skips; docstrings, spelling, Runic,
  scoped types and the Sphinx build pass. The full GitHub matrix remains outstanding.
- The statistics bridge milestone passes independent production and test review.
  Eight managed-runtime tests pass on macOS arm64 with two Julia threads: six
  integrator/yield cases compare identical raw fields using operation-derived absolute
  bounds, and two material cases validate catalogue transport and population identities.
  All 45 existing core/history bridge cases pass after attachment. The bounds-enabled
  Core aggregate passes 42,802 assertions in 193 sets, including the reviewed DF32,
  dense, monitor-continuation and group-hook leaves; all 16 Aqua/JET checks pass with
  1,076 definitions analysed. The 170 typed boundary/lowering/CI-runner tests pass,
  as do Runic, scoped types, docstrings, spelling, whitespace, 349 SEP checks (three
  expected process-SEP skips) and Sphinx with warnings as errors. The earlier pushed
  head's CI reports Julia Linux and combined coverage failures; their logs are being
  investigated, and this local milestone does not claim a green full gate.
- CI run `42916416591` at `12613fc5` completed with failures in Julia Linux and the
  combined coverage report. The Julia suite passed 465 cases but failed two known
  first-SIGTERM-exit probes (#610) and a status test that incorrectly assumed its
  parent process had never initialized Julia (#668). The status assertion now runs
  in a bounded cold subprocess; four typed tests and independent review pass.
  A subsequent pytest-process SIGSEGV is retained separately in #669 without a
  claimed cause or fix. Runtime investigation remains deferred as directed.
  The latest SEP 47 merge addresses several reported coverage gaps; the next CI
  report must establish the remaining package-wide coverage result.
- The fixed-schedule Python group adapter passes independent production review and
  16 real managed-runtime tests, including matched solo trajectories, warning/stopping
  outcomes, numerical budget failures and failed copies with Adams order/monitor rows.
  Seven independently authored typed transaction cases cover deferred interrupts,
  repeated publication failures and constructor rollback; three ledger publication
  cases pass alongside them. The broader affected typed selection passes 132 tests.
  The bounds-enabled numerical group selection passes 659 assertions, including zero
  warmed allocation; all 16 Aqua/JET checks pass over 1,199 definitions, none cached.
  Scoped types, docstrings, spelling, Ruff, Runic and Sphinx pass. Missing runtime gates
  remain visible; no whole-package CI result is inferred from these checks.
  The shared alias/layout correction is now exercised through the actual managed
  PythonCall bridge in the mandatory Julia lane. Review and current runtime checks pass.
- Adaptive Adams dense scheduler `c527047a` is integrated as `6da9b9b7` after an
  independent 1,327-assertion numerical review. The adapter uses authoritative observed
  field/coordinate selectors for export and power conversion; no caller can request
  committed history while a dense plane behind it is selected. Snapshots drain owed
  observations before saving the endpoint. The independent attachment review passes
  2,291 bounds-enabled assertions, including forced-terminal coordinate changes and
  an affine-weight export bound. All 85 real core/history/statistic/monitor/group/dense
  cases pass, including an interior field-only restart and independent RK refinement.
  The integrated Core suite passes 110,764 assertions in 244 sets. Aqua/JET passes all
  16 checks over 1,217 definitions; the affected typed selection passes 149 tests.
  Replay/interpolation and split/interpolation refusals remain unchanged.
- Binary64 input noise is attached to all three session methods. Each start selects
  its actual problem/override draw, copies it into owned Julia storage, and adds it
  only at a fresh launch before observations or history. Resumes keep their field
  and the draw's monitor floor. Phase exports use the observed coordinate and the
  absolute cumulative operator, including direct/expanded offsets. Modified and
  binary32 noise still refuse. Independent review passed 1,340 bounds-enabled
  assertions and eight typed callback-publication tests. The callback tests exposed
  and fixed #680: copied fields/phases cannot populate replacement or closed caches.
  The separately reviewed noise-floor comparison bound propagates absolute FFT
  and energy errors through both numerator and denominator; its 19 tests pass.
  All 100 real core/history/statistic/monitor/group/dense/noise cases pass on macOS
  arm64 with two Julia threads. The affected typed selection passes 149 tests.
  Core numerical tests plus the isolated inventory rerun pass 111,597 assertions
  in 251 sets; the first inventory invocation lacked `--startup-file=no`, and an
  isolated retry needed its existing `Pkg`/`TOML` imports. Those command errors did
  not require source changes. Aqua/JET passes 16 checks over 1,236 definitions,
  none cached. Scoped types, docstrings, Ruff, Runic, spelling and Sphinx pass.
  Whole-package CI, standard performance records and remaining routes are pending.
- DOPRI5 dense leaf `f4cf2ed4` and conditioning correction `844ac356` are integrated
  as `0e9ff4ae` and `3e395988` after 779 independent bounds-enabled assertions.
  Shared extension weights are copied from the actual tableau; endpoint rows and
  dense planes retain their own coordinates. Common conformance exposed #687:
  endpoint rows must accompany the accepted attempt before the first dense plane
  drains it. The repaired source preserves warning continuation and snapshot order.
  Core passes 112,291 assertions in 264 sets, including explicit diagnostic reset
  without new source work. Aqua/JET passes 16 checks over 1,253 definitions, none
  cached. The managed dense/monitor/common selection passes 53 cases and 22 declared
  skips; all 23 split-collar cases pass. Initial split assertions incorrectly
  assumed low-storage skips its zero source and that a quiet verdict meant actual
  collar omission; tests now verify the actual call counts and continued damping.
  No production workaround was required for those test assumptions. Independent
  adapter review and 12 typed admission checks pass. The shared adapted-request
  admission correction #682 is committed as `af3edeb9`. The combined typed selection
  had 263 passes and one stale CI skip fixture; the corrected CI fixture's 56 tests
  pass. Formatting, scoped types and documentation checks accompany this milestone.

- Representative precompile prerequisite `6c50a522` and workload `d7b29ca1` were
  independently approved and integrated as `b26309fc` and `3cc78d05`. The prerequisite
  selection passed 601 bounds-enabled assertions; the workload review passed 106,
  including preservation of a nondefault planner policy. Full author quality passed
  16 Aqua/JET checks with 1,225 definitions analyzed and none cached, plus a concrete
  allocating-workload analysis. The three owned serial sessions exercise field RK,
  envelope Adams and low-storage dispatch, responses, observations and monitoring.
  A fresh native-target engine-cache diagnostic measured a 22.67 s cache build and
  67.54 MB image; first representative execution was 6.93 ms versus 20.26 s with only
  the workload macro disabled. These are local Apple arm64 cache diagnostics, not
  quiet-machine propagation benchmarks or complete public-path compilation claims.
  Two-thread dispatch still compiled a small specialization. Generic-target CI and
  the required startup/worker campaign remain pending.
- The latest pushed `dca5615e` completed all Python/native static, default, slow and
  combined coverage checks successfully, including every coverage shard and the
  coverage report. Both Julia jobs passed their engine tests and 774 Python checks,
  with 30 declared skips, but each failed one first-SIGTERM-exit diagnostic through
  the already recorded shutdown timeout. This remains visible under #610 and the
  maintainer's deferred runtime work; no completed full gate is claimed.
- Scalar spatial session attachment passes 104 managed checks across the affected
  selections: 73 spatial cases, 23 modal split cases and eight modal observation
  regressions. Eighteen initial comparison tests mistakenly requested a stage-only
  quadrature diagnostic from the saved-field reference collector; the corrected
  reference selection passes all 18, with the actual unavailable diagnostic checked
  separately. The production implementation needed no adjustment for that test error.
  The full two-thread bounds-enabled Core passes 116,924 assertions in 294 sets.
  Integrated Aqua/JET passes all 16 checks over 1,409 definitions, none cached.
  The new physical temporal-split leaf passes 64 assertions, including zero warmed
  allocations; constant spatial noise phase passes three. Pure spatial transfer and
  width-conditioning checks pass 21 cases. Independent source/observation reviews
  cover transform order, flux normalization, aliasing and conditioned width bounds.
  Actual bridge regressions cover nonzero launch phase, material-free physical
  absorption and foreign-preparation export rejection (#690). Additional preparation,
  transverse monitoring and per-route material/service witnesses remain outstanding.
- Shared BLAS lifetime attachment is independently approved. Fresh discovery after
  runtime loading joins `_process.acquire_host_blas`, including libraries absent from
  an outer construction scope. The loaded native backend's private BLAS is excluded.
  Main-thread registry cleanup owns release, and interrupted publication retains one
  complete preparation or rolls back the unpublished resources. A foreign restoration
  failure keeps a Python-only retry owner and preserves the primary failure. The
  127-test affected typed selection passes; independent review repeated 46 BLAS/runtime
  cases, including both acquisition/release orders and abandoned-owner cleanup.
  Actual controller widths are metadata. The separately planned macOS Accelerate
  forwarding remains unimplemented; no provider-selection qualification is inferred.
- Linear-flow and propagator outputs now reject hidden foreign-header overlap and
  inputs aliasing retained operator/exponent storage before any write (#691).
  Focused bounds-enabled tests pass 40 flow and 15 propagator assertions, alongside
  the existing operator/source checks. Independent original probes pass 28 assertions
  at task budgets one and two, including offset foreign headers, negative strides,
  exact in-place operation, failure preservation and zero warmed allocations.
- Transverse service attachment and explicit continuation selection are independently
  approved at `ea4b8731` and `70fa4d8e`. The latter fixes #693: an explicitly missing
  record is a selection, while constructor-only resumption may still receive its first
  later fallback. The independent core/bridge review passed 1,180 distinct assertions;
  the final 447-assertion selection and 16 Aqua/JET checks passed, with 1,477 definitions
  analyzed and none cached. Master Python record lowering passed 42 independent typed
  tests, including the #692 face-stop first-warning coordinate correction. Subsequent
  factory/session review passed 118 typed Python checks and 139 actual PythonCall
  assertions across three methods and geometries. Managed integration remains separate.
- The timing-probe leaf `d71d29cf` is independently approved: 472 bounds-enabled
  assertions, including additional counter-capacity/failure cases and zero warmed
  allocations. Author witnesses also include 94 NumPy comparisons. The master adapter
  retains preparation through its probe owner, consumes a start only after construction,
  returns owning read-once fields and closes live probes with the execution owner.
  Its separate review identified #696, where a cleanup exception replaced an initiating
  error; cleanup now preserves the original exception with a diagnostic note.
- FFT call-site attachment `2b2ef88c` is independently approved. The exact final author
  selections passed 686 assertions at one thread and 866 at two, all warmed allocation
  checks zero. Aqua/JET passed 16 checks plus two complete-workload checks, with 1,455
  definitions analyzed and none cached. Independent testing added odd-length transforms,
  both precisions and nondefault global planner settings: 396 assertions passed.
  Master bridge integration refreshes offers per start (#694), makes optional cache-write
  failure nonfatal while cleaning unexpected adoption failures (#695), and releases all
  preparation-only wisdom maps after detaching facts. Calibration clearing also removes
  compatible Julia files and forgets process offers without importing either runtime.
- The `09fb770a` CI run passed all Python/native default, slow, documentation, wheel,
  device-compile and combined coverage jobs inspected so far. Static checking exposed
  test-only optional-type narrowing and an ambiguous bounds import, corrected locally.
  Linux Julia passed 911 Python checks with 22 declared skips, but three SIGTERM cases
  stalled in GC. The diagnostic still expects SIGTERM alone; it is being aligned with
  the already accepted external SIGKILL deadline, retaining observed escalation facts
  and failure for unverified startup or unsuccessful supervision. No runtime fix is
  claimed, and #610 remains open.
- Combined managed integration passes 34 new cases: 21 transverse sessions, 11 timing
  probes and two real cache publication/reuse transactions. The numerical Core selection
  passes 119,300 assertions in 323 sets; the final inventory module passes 34 assertions
  in seven sets separately with its required startup flags and imports. Earlier command
  omissions are retained as failed harness attempts, not hidden or patched in production.
  Integrated Aqua/JET passes 16 checks over 1,544 definitions, zero cached. The affected
  Python/SEP selection passes 455 tests with three declared process-SEP skips. Sphinx
  builds successfully. Whole-package Python checking has zero errors and three existing
  local native-stub warnings; scoped changed-module checking is clean.
- The SIGTERM supervisor correction is integrated at `eb4835ff` and `ef0687e4` after
  independent review and 36 focused tests. A controlled real child proves forced-kill
  escalation; three actual managed-runtime smoke probes exited on TERM in roughly
  42–47 ms, so those probes did not reproduce the intermittent stall. The supervisor
  records which signal completed shutdown and still fails unverified startup or an
  unsuccessful hard deadline. This implements the accepted policy, not a runtime fix.
- The shared conformance and affected spatial/monitor/noise/group regression selection
  passes 147 tests, with 20 explicit numerical-versus-bitwise skips and two stale
  transverse capability skips. Both newly enabled transverse conformance cases then
  pass separately. CI now requires them and the spatial, transverse, timing-probe and
  FFT managed modules; only the explicit bitwise-repeatability skips remain allowed.
- Verified CPU provider attachment passes 133 independent bounds-enabled Julia
  assertions and four actual managed preparation/cache checks. Both integer ABIs,
  real/complex products and missing-symbol fallback are checked. Missing Apple
  threading capabilities retain OpenBLAS; actual provider and calling-thread scope
  are metadata. ADR-079 records the macOS 15 control boundary. Regenerated manifests
  add only the Libdl dependency edge. Independent wiring review passes 113 typed
  checks, including the corrected #697 acquisition/rollback ownership and orderly
  main-thread retry. The final affected cleanup selection passes 78 tests.
- The combined material matrix passes all 20 managed cases. It covers field and
  envelope on five routes, plural plasma/Raman, independent modal gradients and
  uncoupled tapers, and alternate march families. Independently reviewed eight-step
  accuracy envelopes require separately refined 16/32-step reference settlement;
  no formal convergence-order result is inferred. The material module and actual
  provider checks are now mandatory in the Julia lane.
- Wide assembly leaf `d6687bd7` is independently approved and integrated at
  `1a653768`: 25,055 independent exact-rational, domain, ownership and IR assertions.
  Author final numerical checks pass 9,960 at one thread and 18,930 at two, including
  zero warmed allocations. This certifies conditional CPU operations only; public
  binary32 execution, stored-history integration, device certification and performance
  remain separate work. Combined provider/wide/window package quality passes all
  16 Aqua/JET checks over 1,644 definitions, none cached.
- Window ownership correction #698 is independently inspected and integrated at
  `3138fa4c`; 52 repeated bounds-enabled assertions verify one storage owner, actual
  window results and zero warmed allocation. The shared planner's 23 pure and 38
  managed inventory cases accompany its pending adapter integration.
- The `162da950` CI static lane identified seven missing private-helper docstrings.
  Scoped validation passes after `72ee3e4c`. Documentation builds successfully;
  the SEP selection passes 349 tests with three declared process-SEP skips. Full
  CI and the current one/eight-thread Linux correctness campaign remain in progress.
- Public CPU selection and request-specific node binding pass 38 actual retained-array
  inventory cases, three public integrator cases and an independent-capacity reuse case.
  Fourteen standalone Julia assertions verify immutable input sharing and range refusal.
  A complete public simulation and the common benchmark window pass separately.
  The corrected focused settings fixture passes; the combined pure selection passes
  180 tests with five declared managed-runtime skips. Scoped types have zero errors or
  warnings; documentation builds successfully and changed-file spelling passes.
- The benchmark slice is independently reviewed and integrated at `dd94898d`.
  Generic `prepared.start` return is the versioned ordinary-run timing boundary;
  preparation-inclusive wall time and same-session actual counters remain separate.
  Author validation passes 106 typed tests and 23 affected session cases; master
  integration includes actual Julia public-run smoke. No timing ratio is claimed yet.
- Required CI at `162da950` reports 100% package coverage. Windows collection failure
  #699 is independently reviewed, integrated at `6c211acf`, and passes all 38 affected
  tests including cold imports with absent signal constants. The unrelated service
  scheduling assumption is tracked in #700. A macOS shard passes 7,229 tests but its
  artifact upload fails DNS resolution; the next push must re-establish every gate.
- Source range/coefficient helpers are independently reviewed at `dd38b8bf` and
  integrated at `25b9c775`. All 159 authored bounds assertions and 128 additional
  extreme-scale, signed-zero, coefficient-order and transactional-refusal assertions
  pass on the exact detached review tree. These are preparation helpers, not yet
  public binary32 execution. The original shared arithmetic record stays unchanged.
- Service test correction #700 is reviewed and integrated at `43fb2b7c`. It waits on
  the actual stop-file watcher at a selected saved plane and separately tests a stop
  at the final plane. Both independent cases pass, preserving every saved array.
  The author additionally passes all 50 affected service/helper-consumer checks.
  Production behavior and signal policy are unchanged. A new metadata helper's missing
  parameter/return docstrings were caught and corrected by the local fast check.

- Independent public CPU review approves `f85d3b49`: 55 typed checks plus actual
  reverse chunk adaptation, detached metadata, preparation/handle lifetime and
  request-added monitors on three spatial geometries. No blocker remains within
  the declared numerical node-array payload scope.
- Protected observation helpers `3c8e4333` are integrated at `93050175` after
  independent review. The primary Ogita–Rump–Oishi paper's Algorithm 4.1, its
  equivalence to 4.4 and Proposition 4.5 (including underflow) were checked. All 50
  authored and 399 independent exact-sum/scale assertions pass. A private harness
  initially failed to expect deliberate overflow refusals; its correction changes
  no production behavior.
- Protected plasma `d5e48bda` is independently approved and integrated. In addition
  to the authored regression selection, 77 independent scaled-population and saturated
  exposure assertions pass. #701 remains visible until the implementation merge.
  Existing binary64 composition tests and actual NumPy-single prefix witnesses pass;
  full binary32 material/session and vendor qualification remain pending.
- The eight-thread Linux correctness selection reached its 45-minute stage deadline
  without completion. The one-thread selection passed 267 tests with 22 declared skips
  at source `4ab7e15a`; two of those skips are already removed and tested locally.
  The deadline is not a passing multithread qualification. The comparative CPU
  campaign uses source `08721c9d`, one/64 physical cores on an otherwise idle AMD
  EPYC 7763, three alternating rounds and three repeats of 0.001 entrance fractions.
  It exposed missing default plasma statistics (#702) and low-storage flow-table
  support (#704), now assigned. Failed windows are retained; no complete phase 2
  measurement or performance conclusion is claimed.
- Master binary32 RK/low-storage leaf now retains narrow state arithmetic and wide
  coordinates/factors, with protected normalized norms. The focused leaf and existing
  binary64 operator/RK/low-storage/source regressions pass; the final new march leaf
  passes 85 assertions and the combined plasma selection passes 88. Warm RK, norm
  and low-storage calls allocate zero. Aqua/JET passes 16 package checks over 1,711
  definitions, none cached, plus three concrete binary32 calls before the plasma
  leaf is combined. Independent review and combined quality remain pending; public
  single precision, complete session scaling and Adams are not enabled by this leaf.
