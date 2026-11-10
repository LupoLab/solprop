# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`. One draft implementation pull request,
opened after proposal [#609](https://github.com/LupoLab/solprop/pull/609) merges.
The completed implementation is left ready for maintainer review, not auto-merged.

## Current status

| Phase | Status | Evidence or prerequisite |
|---|---|---|
| Acceptance | CI passed; merge held for signal-policy decision | `5fb24689`; 349 SEP tests pass, 3 expected skips; spelling, whitespace, Sphinx and CI pass |
| 1: package and runtime | Rules prepared; signal-policy conflict found | macOS SIGTERM restoration fails; session work remains gated |
| 2: CPU binary64 session | Pending | Merged SEP 47 descriptor/request split and generic preparation |
| 3: CPU binary32 | Pending | Phase 2 and source-checked DF32 bounds |
| 4: devices | Pending | CPU gates and one qualification pass per vendor |
| 5: records and Final | Pending | All required capability, CI and measurement evidence |

SEP 47's contract milestone is merged at `2e2490c2`. Its native-session PR
[#607](https://github.com/LupoLab/solprop/pull/607), inspected at `6a94b3cc`, still
needs the descriptor/request split and production `Backend.prepare` entry point.
Inspect merged code before integrating; prospective phase reports are not evidence
that an interface exists. Package/setup work is independent of that prerequisite.

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

## Decisions

- The accepted SEP's scope is retained with binary64-only input noise. Every binary32
  path, including Metal, refuses noise. No physics default changes.
- Start with Julia 1.13.3 and matching JuliaCall/PythonCall 0.9.37. Sources and
  manifests identify immutable setups; runtime execution downloads nothing.
- `SimulationSettings.backend_options` forwards `wide` and `setup_id` through the
  existing execution options. Device/options are outside science and resume hashes.
- Runtime initialization is main-thread-only and serialized. Preserve actual Python
  handlers; use an atomic stop flag. The agreed signal policy must pass its probes.
- The ordered project stack and effective preferences are immutable setup identity.
  Adding a device publishes another identity. Only that stack and standard libraries
  enter `LOAD_PATH`; user startup files and artifact overrides cannot alter it.
- A plain Julia `@cfunction` is unsuitable for a stop listener: first entry from a
  foreign thread may allocate and wait for GC. Use a C-compatible atomic-store leaf
  without Julia runtime entry, and retain its code and byte owner through captured
  callbacks, including callbacks that outlive detachment. Verify this in subprocesses.
- Julia consumes only `RunRequest` policy through the generic preparation boundary.
  It never inherits the legacy driver or uses a Python-over-Julia numerical loop.
- DF32 admission bounds actual exponent spans, including replay and restored Adams
  history, over the complete represented operator interval.
- Numerical comparisons use SEP 25/33 classes, never blanket cross-backend bit identity.
  Export structure and exact controller decisions retain their exact contracts.
- Measure preparation-inclusive `wall_seconds` beside kernel-window timing. Full
  gates run in CI; local qualification uses targeted, short-window tests.

## Next actions

1. Resolve the macOS signal-policy conflict with the maintainer before merging #609;
   bring the revised, merged main into this worktree.
2. Mark SEP 48 In progress, open the single draft PR, and record its link here.
3. Assign package/environment and setup/runtime work with disjoint ownership; freeze
   their bridge contract and independently review both packages.
4. Pass the phase 1 probes before implementing the complete numerical session.

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
  with two threads crashed during shutdown. This blocks the agreed signal policy;
  the maintainer has been asked for a direction. No production workaround is adopted.
- The primary DF32 papers and authors' formal supplement were checked. The planned
  family is AccurateDWPlusDW, DWTimesFP3, DWTimesDW3 (including low-by-low), DWDivFP3
  and DWDivDW3. The last algorithm's naming typo is resolved by its formal proof;
  underflow/overflow guards and transcendental bounds still need qualification.
- No propagation or GPU qualification is claimed yet.
