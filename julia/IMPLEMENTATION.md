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
| 2: CPU binary64 session | Scalar modal RK, landed Adams, low-storage and statistics pass; remaining services in progress | Fifty-three real bridge cases include portable histories and same-field statistics |
| 3: CPU binary32 | Scalar DF32 foundation reviewed; full execution pending | Finite-expression and transcendental admission remain unqualified |
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
public registry selection awaits the broader CPU composition. No new backend-name
branch belongs in the service/session dispatcher.
The subsequent Part B CI corrections at `f6921fae` are integrated at `10be2eff`.
The additional CI corrections at `55fdc423` are also integrated: they restore
request-owned arrays to the mixture baseline and remove nonportable hash literals.
The final documentation/skip corrections at `21b5808c` are integrated as well.
The final contract/coverage corrections at `f351d0a3` are integrated at `4dade535`.
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
| Numerical transform/Adams agent | New `julia/DF32_BOUNDS.md`, `julia/src/double_float_functions.jl` and its test | Conservative function domain/API approved; implementation and independent qualification pending |
| Numerical services agent | New `julia/src/group_session.jl` and its test | Julia-owned group coordinator under independent review |
| Master | All shared Python interfaces, Julia module/test inclusion, transfer records, integrators/session lifecycle, repository records and remote campaign | Integration and phase 2 shared interfaces |
| Independent reviewer | Production code read-only | Statistics transport approved pending same-field runtime evidence; reviewing groups |

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
Monitoring, dense output, groups, noise
and broader routes still block phase 2 completion. The subsequent phases continue
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
The initial DF32 function API admits only a proposed `abs(real(z)) <= 16`,
`abs(imag(z)) <= 2^20` domain with a separate `u/8` input-error budget. Protected
exponential, phasor and phi helpers are being implemented against that derivation;
neither whole-run admission nor device qualification is inferred from this decision.

The foundation is committed at `540794bd`. Setup commits `6c569742`, `fadae531` and
`52269788` were independently reviewed and integrated as `9fe74c26`, `5cec895c` and
`5264160b`. Julia package commits `28fb628e` and `201d91d5` were independently reviewed
and integrated as `ea847fcd` and `773bd539`. Updated main is merged at `e2df6c7c`.
Issue [#610](https://github.com/LupoLab/solprop/issues/610)
tracks the accepted signal-handler limitation and intermittent SIGTERM-during-GC
shutdown hang. The maintainer accepted an external hard-kill deadline and then deferred
further runtime investigation until after core implementation. Existing failing results
remain preserved; the old first-SIGTERM-exit tests have not been changed or passed.

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

1. Owned execution options and recipe metadata are integrated at `d1feaa15`; execution
   ownership and transactional groups are integrated at `a6f30aba`.
2. Connect the general modal source, Adams/low-storage methods, statistics and monitors
   to the tested Julia-owned session, using `SESSION_DESIGN.md`.
3. Complete dense output, restart histories, binary64 input noise and member groups;
   then CPU conformance and preparation-inclusive short-window measurements.
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
