# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`.
Draft implementation [PR #611](https://github.com/LupoLab/solprop/pull/611).
Proposal [#609](https://github.com/LupoLab/solprop/pull/609) merged at `04e72192`.
The completed implementation is left ready for maintainer review, not auto-merged.

## Current status

| Phase | Status | Evidence or prerequisite |
|---|---|---|
| Acceptance | Complete; Part I frozen | Final head `da2e699f` passed GitHub CI and required documentation preview; merged `04e72192` |
| 1: package and runtime | Setup/package integrated; accepted main-thread boundary under qualification | macOS and Linux package gates pass; revised subprocess gate remains |
| 2: CPU binary64 session | Pending | Runtime gate; final SEP 47 signatures available in updated handoff |
| 3: CPU binary32 | Pending | Phase 2 and source-checked DF32 bounds |
| 4: devices | Pending | CPU gates and one qualification pass per vendor |
| 5: records and Final | Pending | All required capability, CI and measurement evidence |

SEP 47's contract milestone is merged at `2e2490c2`. The completed Part B head,
`e9105523` from [#607](https://github.com/LupoLab/solprop/pull/607), is integrated at
`83da0ad5` under the maintainer's authorization to proceed before its CI fixes merge.
Its actual interfaces have been read: the descriptor/request split and generic
`Backend.prepare(problem, request).start(...)` are present. `assemble` returns
`Translation(problem, request)`, validation is `validate_run`, and preparation checks
`foreign_request`. Group preparation and the fixed registry still need generic Julia
integration; no new backend-name branch belongs in the service/session dispatcher.
The current whole-package type check reports 93 errors in 23 files identical to that
dependency head; no Julia file is among them. Track and integrate #607's CI corrections
before qualification; do not suppress those errors or duplicate its agent's fixes.

## Ownership

The master owns architecture, shared transfer records and session lifecycle, shared
Python integration, CI and packaging hooks, and this record. Implementation agents
receive disjoint files in separate worktrees after these rules are committed.
Assignments and ownership handoffs are recorded here before overlapping work begins.
An independent reviewer reviews each package they did not author before integration.

Phase 1 divides into three packages: the Julia package, environments and installer
script; Python setup, identity, artifact preflight and commands; and the master's
runtime, cancellation, wheel/CLI integration and CI. Session implementations wait
for the runtime probes. Numerical source review may proceed independently.

| Assignment | Exclusive files | State |
|---|---|---|
| Julia package agent | `tests/backends/julia/runtime_probe.py`, `test_runtime_subprocess.py` | Real packaged subprocess qualification; package ownership returned to master |
| Distribution agent | Private Linux campaign checkout, environment and its own Slurm jobs | DMOG runtime qualification; reviewed CI ownership returned to master |
| Master | Rules, this record, integrated Julia package/setup, runtime/session/transfer, CI, wheel hooks, CLI and documentation | Integration and phase 2 shared interfaces |
| Independent reviewer | Read-only reviews and isolated lifecycle/numerical probes | Main-thread ownership boundary review; donor audit complete |

The foundation is committed at `540794bd`. Setup commits `6c569742`, `fadae531` and
`52269788` were independently reviewed and integrated as `9fe74c26`, `5cec895c` and
`5264160b`. Julia package commits `28fb628e` and `201d91d5` were independently reviewed
and integrated as `ea847fcd` and `773bd539`. Updated main is merged at `e2df6c7c`.
Issue [#610](https://github.com/LupoLab/solprop/issues/610)
tracks the accepted signal-handler limitation. A newly observed SIGTERM-during-GC
shutdown hang is under investigation and remains a qualification blocker; no timeout
has been accepted as a passing result.

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
  including with custom or ignored handlers and while no session advances. Keep
  Julia 1.13.3; revisit a supported solution with stable 1.14 in #610. Threaded GC,
  bounded Ctrl-C, cooperative cancellation and lifecycle probes remain blocking.
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
- Measure preparation-inclusive `wall_seconds` beside kernel-window timing. Full
  gates run in CI; local qualification uses targeted, short-window tests.

## Next actions

1. Finish independent review and the real subprocess matrix for the accepted main-thread
   ownership boundary; retain the original failed worker evidence in #620.
2. Run the same accepted runtime gate on the DMOG Linux checkout.
3. Integrate reviewed subprocess tests and publish a batch for the full CI gates.
4. Pass the phase 1 probes before implementing the complete numerical session against
   the final SEP 47 signatures; integrate its subsequent CI corrections.

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
  SIGKILL. The failure remains visible while plain-JuliaCall and Linux controls run.
  Julia's pinned fatal-signal source explicitly anticipates asynchronous shutdown
  stalls; this has not silently extended the maintainer's accepted signal policy.
