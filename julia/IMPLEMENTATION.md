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
| 2: CPU binary64 session | In progress | SEP 47 Part B integrated; disjoint numerical packages and shared owner integration |
| 3: CPU binary32 | Pending | Phase 2 and source-checked DF32 bounds |
| 4: devices | Pending | CPU gates and one qualification pass per vendor |
| 5: records and Final | Pending | All required capability, CI and measurement evidence |

SEP 47's contract milestone is merged at `2e2490c2`. The completed Part B head,
`e9105523` from [#607](https://github.com/LupoLab/solprop/pull/607), is integrated at
`83da0ad5` under the maintainer's authorization to proceed before its CI fixes merge.
Its actual interfaces have been read: the descriptor/request split and generic
`Backend.prepare(problem, request).start(...)` are present. `assemble` returns
`Translation(problem, request)`, validation is `validate_run`, and preparation checks
`foreign_request`. Group construction now dispatches through an execution owner's factory, independently
of legacy numerical operations. The Julia owner and registry entry remain to be added;
no new backend-name branch belongs in the service/session dispatcher.
The current whole-package type check reports 93 errors in 23 files identical to that
dependency head; no Julia file is among them. Track and integrate #607's CI corrections
before qualification; do not suppress those errors or duplicate its agent's fixes.

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
| Numerical transform/Adams agent | `julia/src/execution.jl`, `transforms.jl`, `operators.jl`, `tables.jl`, `phi_functions.jl`, `adams.jl`; corresponding tests; dependency declarations/generated environments and module imports | Reviewed transforms; binary64 exponential Adams and represented operator rates next |
| Numerical services agent | `julia/src/responses.jl`, `raman.jl`, `plasma.jl`, `statistics.jl`, `monitor.jl`; corresponding Julia tests | Reviewed responses integrated; statistics and monitors next |
| Master | All shared Python interfaces, Julia module/test inclusion, transfer records, integrators/session lifecycle, repository records and remote campaign | Integration and phase 2 shared interfaces |
| Independent reviewer | Read-only reviews and numerical probes | Reviews each implementation package and shared interfaces |

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
2. Integrate reviewed transforms, responses and Runge–Kutta leaves. Implement binary64
   Adams and statistics/monitors in parallel, using `SESSION_DESIGN.md`.
3. Build the complete Julia-owned session against SEP 47 Part B and integrate its CI
   corrections. Reuse reviewed experimental kernels without its Python march adapter.
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
