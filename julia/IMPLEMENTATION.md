# SEP 48 implementation

Branch: `feature/sep-0048-julia-backend`; draft [PR 611](https://github.com/LupoLab/solprop/pull/611).
Proposal 609 merged at `04e72192`; Part I is frozen. Leave the completed implementation
ready for maintainer review, without merging it.

**Paused checkpoint — 11 November 2026.** Binary64 CPU is available on this branch.
CUDA is partly integrated and remains closed to public selection. No new implementation
slice or compute campaign starts during the maintainer's requested pause. The last
code integration before this record is `8c7596cb`; CI results below identify the
revision actually checked. A checkpoint is not Final qualification.

## Agreed scope and phase status

The completion baseline is **binary64 CPU and CUDA**. Metal, ROCm/HIP and unfinished
binary32/DF32 qualification are deferred. DF32 supplies selected wide intermediates
for binary32 state and FFTs; it is not a whole-state backend mode. See
[DEFERRED.md](DEFERRED.md) for retained implementation and failed evidence.

| Phase | Delivered | Still needed for this baseline |
|---|---|---|
| 1: package/runtime | Immutable setup identities, offline artifact checks, lazy bridge, ownership and packaging | Final installed-package/supervisor matrix; accepted runtime limitations remain explicit |
| 2: binary64 CPU | Public scalar modal, radial, Cartesian and line sessions; responses, methods, services, history and groups | Final complete conformance/CI; isolated startup and worker measurements |
| 3: binary32/DF32 | Substantial internal implementation, tests and numerical analysis retained | Unfinished qualification deferred; public selection stays closed |
| 4: devices | CUDA execution/FFT, storage, RK, low-storage, modal source and export integrations; initial actual A40 evidence | Adams integration, complete observations/monitors, spatial composition, end-to-end bridge/history/groups, vendor quality and measurements |
| 5: documentation/Final | Manual, design, decisions, release fragment and preliminary benchmark ledger | Final evidence, independent integration review, required green gates, archival/Final commit |

Binary64 CPU supports scalar field/envelope Kerr, Raman and plasma, adaptive or replay
RK/Adams and fixed low-storage stepping, represented operators, dense observations,
statistics, temporal/transverse monitoring, portable Adams history, snapshots, input
noise and fixed-schedule groups. The [manual](../docs/guide/performance/julia.md)
records exact admission and refusals. Scientific defaults and hashes are unchanged.
Julia 1.13.3 and JuliaCall/PythonCall 0.9.37 remain pinned.

## CPU use at this checkpoint

From a checkout of this branch with the normal native build prerequisites:

```sh
uv sync --extra julia
uv run solprop julia setup
uv run solprop julia status
```

Use `SimulationSettings(backend="julia", device="cpu", ...)` in an existing scalar
binary64 example. If status lists multiple compatible identities, select the intended
one with `backend_options={"setup_id": identity}`. Re-run setup after engine source
changes; begin a fresh Python process after selecting another runtime identity.
Julia entry belongs on Python's main thread. The accepted SIGTERM limitation requires
an external hard-kill deadline; ordinary cancellation remains cooperative.

## Integration and review checkpoint

All completed, independently reviewed integrations are on the one PR branch:

- Storage/readback, capture guards and RK orchestration: `75d6849a`, `ba632416`,
  `9e41cdb1`.
- Modal Kerr, low-storage and transactional export: `23fed2ab`, `900ce053`,
  `bbdf0eda`, `f007e1b7`; selected-execution bridge: `2a1fdd4a`.
- Modal Raman/plasma and execution-aware factories: `2002886f`, `5b5ac0d1`.
  Review adds 150 independent witnesses; the integrated engine passes 4,942
  assertions, and 16 actual managed CPU material/route cases pass.
- Shared RK dense observation: `eb0a3e69`. Author checks pass 3,508 affected
  CPU/spatial/DF32 assertions and 334 model/RK assertions. Independent review passes
  475 assertions, including 141 original ownership, represented-operator and failure
  witnesses. CPU Aqua/JET passes 16 checks, all 2,310 definitions analyzed, none cached.
- CUDA retained-payload test correction: `8c7596cb`. Independent source review
  verifies pinned CUDACore four-byte allocation padding; no production code changes.
  The corrected actual-device rerun remains pending.

The final combined checkpoint passes 10,541 bounds-enabled Julia assertions across
42 testsets and six actual managed CPU replay/dense cases after publishing a fresh
setup. Documentation checks pass: spelling, whitespace, 349 SEP consistency tests
(three declared skips) and the Sphinx build. Two earlier private test-runner invocation
errors are retained in the evidence; the corrected aggregate exits successfully.

Against current `main` (`ae65df6a`), cloc 2.10 counts 34,180 added production code
lines, 76 modified and 46 removed: a net increase of 34,134. This excludes blanks,
comments/docstrings, separate documentation, tests and manifests. It includes retained
deferred SEP 48 work, but excludes already merged SEP 47 work. This is a lexical
physical-line count, not a statement count or the minimum size of a CPU-only backend.

Unintegrated work is deliberately **not** counted as delivered:

| Frozen work | Checks already run | Exact next action |
|---|---|---|
| Binary64 device Adams method/dense mirrors | 10,896 distinct bounds-enabled CPU assertions; package and concrete JET | Independent review, then master integration and actual CUDA execution |
| Shared statistics/temporal readings draft | 540 bounds-enabled assertions including existing zero-allocation checks | Test workspace attachment, finish split/quadrature, then full affected regressions and independent review |
| Spatial source/operator attachment | Interface reviewed; implementation not started | Reuse existing generic source/operator recipes and selected storage after resumption |

Every future author uses a separate worktree. The master owns architecture, shared
interfaces, bridge/registry, transfer records and integration. Services own numerical
statistics/monitor preparation; numerical work owns source/spatial composition; device
work owns Adams/storage and hardware qualification. Assign exclusive files again when
resuming; no agent is currently implementing another slice.

## Hardware and CI evidence

The latest actual A40 combined run reports **2,663 passes, nine assertion failures
and three runner errors**. March (167), modal source (1,595), export (157) and 58
independent storage/march checks pass. The nine failures used logical array sizes
where the ledger correctly includes CUDA allocation padding. The three errors were a
missing module binding in the private runner. Reviewed test corrections and a corrected
runner exist; the original evidence remains failed until an actual rerun passes.
Unfiltered vendor quality also fails: Aqua passes 11 checks, core/extension JET report
76/38 possible errors, and all nine concrete analyses fail. These are qualification
blockers, not waived checks. No Adams, complete services or spatial CUDA result is
implied. Both A40 jobs have ended; no rental exists and campaign spend is US$0/US$50.

At inspected CI head `de6111e9`, static checks (including types, docstrings, spelling,
licences and changelog), no-native tests and native wheel jobs pass. Several Linux
lanes fail during package installation before tests: removing the Azure mirror did
not resolve DNS failures for the official Ubuntu archive. Required CI is not green;
the remaining full matrix and the checkpoint push must be inspected after resumption.
Earlier Linux Julia shutdown failure #669 and accepted runtime boundaries stay visible.
The precise historical runs, failures and reviews remain in
[IMPLEMENTATION_HISTORY.md](IMPLEMENTATION_HISTORY.md).

The preliminary CPU benchmark matrix covers 12 short cases at one and 64 physical
cores on AMD EPYC 7763: 144 samples, 432 repetitions, finite fields and matching step
counts. Results are mixed; there is no blanket speedup claim. The
[benchmark ledger](../docs/dev/benchmarks.md#julia-cpu-entrance-windows-complete-preliminary-matrix)
records windows and preparation-inclusive wall time. Isolated startup, spawned-worker,
CPU-to-GPU package loading and final CUDA measurements remain mandatory.

## Resume in this order

1. Read this record, `julia/AGENTS.md`, the manual, deferred scope and SEP Part II.
   Inspect current PR/CI heads before editing. Preserve all unintegrated work until its
   review and exact ownership are understood.
2. Resolve the current CI infrastructure failure and inspect the complete binary64 CPU
   gate. Keep CPU usable while adding device transport; do not reopen deferred precision.
3. Review/integrate the frozen Adams work. Complete shared observations/monitors,
   population/spatial services and spatial composition; connect history/groups and the
   remaining factories. Keep public CUDA closed until the full path is qualified.
4. Run a new immutable, reviewed A40 snapshot with corrected runner/accounting tests.
   Triage every unfiltered JET result; validate real GPU sessions, failure handling,
   retained memory and CPU regressions. Do not reuse old job results for new sources.
5. Finish required short-window measurements and installed-package checks, then final
   documentation, independent integration review and green gates. Only then prepare
   Final and mark PR 611 ready. **Do not merge the implementation PR.**

## Architectural decisions to preserve

- One numerical routine and operation sequence per feature serves CPU and GPU.
  Backend specialization belongs in allocation, launch, transforms, copies and compact
  readback. Shared arithmetic helpers do not justify duplicated numerical recipes.
- Source, RK, low-storage and Adams audits confirm shared arithmetic. The pending
  statistics draft now has one catalogue dispatcher and common family routines;
  workspace/service attachment is still unfinished. AcceleratedKernels is imported
  but unused in owned source. Reconcile that deliberately; do not rewrite protected
  reductions merely to introduce a library call.
- Python lowers descriptors and services events; Julia owns the march, controller,
  history, statistics, monitors, snapshots and export generations. Follow the merged
  SEP 47 preparation/session contract; no native lowering or Python trial loop.
- Runtime identity includes immutable source, bridge, ordered environments, effective
  preferences, depot and executable locations. Setup can download; runtime artifact
  verification precedes Julia loading and execution performs no downloads.
- All Julia-facing Python operations, including cleanup and exception formatting,
  run on the main thread. Only the lifetime-owned atomic stop setter runs elsewhere.
  Keep the accepted SIGTERM/external-deadline policy and ADR-077/078 boundaries.
- `device`, `wide` and `setup_id` are execution choices outside scientific/resume hashes.
  Input noise is binary64 only, added once at a fresh launch and never again on resume.
- Preserve SEP 25/33 comparison classes, strict arithmetic protections, zero warmed
  CPU advancement allocation and complete retained-buffer accounting. Real vendor
  allocation padding is distinct from logical transfer size and allocator reservations.
- New issues cover only work this PR will not resolve. Keep in-PR fixes in this record;
  consolidate the earlier issue inventory when finalizing. Historical failures remain
  visible and do not become passes through scope changes or documentation edits.
