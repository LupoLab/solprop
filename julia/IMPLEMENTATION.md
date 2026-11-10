# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`. One draft implementation pull request.
Proposal [#609](https://github.com/LupoLab/solprop/pull/609) merged at `04e72192`.
The completed implementation is left ready for maintainer review, not auto-merged.

## Current status

| Phase | Status | Evidence or prerequisite |
|---|---|---|
| Acceptance | Complete; Part I frozen | Final head `da2e699f` passed GitHub CI and required documentation preview; merged `04e72192` |
| 1: package and runtime | Setup/package integrated; real runtime qualification active | CPU and CPU/Metal installation pass; 220 targeted Python tests pass; subprocess gate remains |
| 2: CPU binary64 session | Pending | Runtime gate; final SEP 47 signatures available in updated handoff |
| 3: CPU binary32 | Pending | Phase 2 and source-checked DF32 bounds |
| 4: devices | Pending | CPU gates and one qualification pass per vendor |
| 5: records and Final | Pending | All required capability, CI and measurement evidence |

SEP 47's contract milestone is merged at `2e2490c2`. The maintainer's updated handoff
on 2026-11-10 confirms its finished native-session branch now contains the descriptor/
request split and generic `Backend.prepare(problem, request).start(...)`. The maintainer
authorizes proceeding against those final signatures while
[#607](https://github.com/LupoLab/solprop/pull/607) completes its push and CI fixes.
Read the actual code at that commit before adapter integration. `assemble` returns
`Translation(problem, request)`, validation is `validate_run`, and preparation checks
`foreign_request`. Group preparation and the fixed registry still need generic Julia
integration; no new backend-name branch belongs in the service/session dispatcher.

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
| Distribution agent | Julia jobs/change flag in `.github/workflows/ci.yml`; new Julia workflow, `tools/check_julia.py`, `tests/test_check_julia.py` | CI implementation and targeted runner tests |
| Master | Rules, this record, integrated Julia package/setup, runtime/session/transfer, wheel hooks, CLI and documentation | Runtime/CLI foundation tested; opening draft implementation PR |
| Independent reviewer | Read-only reviews and isolated lifecycle probes | Setup, Julia package, runtime and DF32 derivation reviewed; wheel boundary and CI review active |

The foundation is committed at `540794bd`. Setup commits `6c569742`, `fadae531` and
`52269788` were independently reviewed and integrated as `9fe74c26`, `5cec895c` and
`5264160b`. Julia package commits `28fb628e` and `201d91d5` were independently reviewed
and integrated as `ea847fcd` and `773bd539`. Updated main is merged at `e2df6c7c`.
Issue [#610](https://github.com/LupoLab/solprop/issues/610)
remains a follow-up, not a blocker under the maintainer's revised signal policy.

## Decisions

- The accepted SEP's scope is retained with binary64-only input noise. Every binary32
  path, including Metal, refuses noise. No physics default changes.
- Start with Julia 1.13.3 and matching JuliaCall/PythonCall 0.9.37. Sources and
  manifests identify immutable setups; runtime execution downloads nothing.
- `SimulationSettings.backend_options` forwards `wide` and `setup_id` through the
  existing execution options. Device/options are outside science and resume hashes.
- Runtime initialization is main-thread-only and serialized. Preserve actual Python
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
- Set `PYTHONDONTWRITEBYTECODE=1` before JuliaCall import for Julia precompilation
  children that embed Python. The initialized host's bytecode policy stays unchanged;
  managed PythonCall source trees must remain byte-identical after runtime use.
- Runtime keeps CPU first in the ordered environment stack. Strict precompilation
  temporarily puts the active device environment first, after proving identical global
  and per-package effective preferences, then restores the runtime order. This permits
  weak-extension dependency lookup without changing the published runtime identity.
- Julia consumes only `RunRequest` policy through the generic preparation boundary.
  It never inherits the legacy driver or uses a Python-over-Julia numerical loop.
- DF32 admission bounds actual exponent spans, including replay and restored Adams
  history, over the complete represented operator interval.
- Numerical comparisons use SEP 25/33 classes, never blanket cross-backend bit identity.
  Export structure and exact controller decisions retain their exact contracts.
- Measure preparation-inclusive `wall_seconds` beside kernel-window timing. Full
  gates run in CI; local qualification uses targeted, short-window tests.

## Next actions

1. Open the single draft implementation PR and record its link here.
2. Qualify installed-wheel and editable identities and the packaged subprocess gate.
3. Integrate independently reviewed Julia CI and run the Linux runtime gate.
4. Pass the phase 1 probes before implementing the complete numerical session against
   the final SEP 47 signatures; reconcile the actual pushed source before integration.

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
- Additional planning review requires conditioning-aware bounds near zeros of phi
  functions and corrects Metal's portable sine/cosine allowance to 4 ulp. Experimental
  witnesses remain measurements; algorithm and admission constants are not qualified.
- The primary DF32 papers and authors' formal supplement were checked. The planned
  family is AccurateDWPlusDW, DWTimesFP3, DWTimesDW3 (including low-by-low), DWDivFP3
  and DWDivDW3. The last algorithm's naming typo is resolved by its formal proof;
  underflow/overflow guards and transcendental bounds still need qualification.
- No propagation or GPU qualification is claimed yet.
