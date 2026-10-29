# AGENTS.md

Contributor guidelines for `solprop-native`, the compiled native backend for solprop,
in `native/`. They apply to people and to coding agents alike, beside the repository's
[`AGENTS.md`](../AGENTS.md), which governs everything outside `native/`. Tested code and
build configuration are authoritative: when this file, `README.md` or the build guide
(`docs/guide/performance/building-native.md`) disagrees with them, correct the
document. solprop's contributor guidelines and numerical documents govern the backend
contract and its conventions; where both apply, follow the stricter rule.

## Product and architecture

- Distribution `solprop-native`; import package `solprop_native`. It is the second
  distribution of the solprop repository, built from `native/`: one checkout, one uv
  workspace and one `uv.lock` (`uv sync --extra native` at the root installs it
  editable; a plain `uv sync` removes it).
- Both distributions take one version from the repository's git tags, and solprop's
  `native` extra requires this package at that same version.
- The public names are `NativeBackend`, `NativeBatchBackend`, the frozen `BuildInfo`,
  `build_info`, `check_build`, `__version__`, and the contract tuples
  `DRIVER_SURFACE`, `LINEAR_OPERATOR_FIELDS`, `TRANSFORM_FIELDS`, `RESPONSE_FIELDS`,
  `SEPARABLE_OPERATOR_FIELDS`, `STAGE_TERMS_FIELDS`, `STAGE_SCALARS_FIELDS`,
  `MODAL_TABLES_FIELDS`, `REGIONAL_RESPONSE_FIELDS`, `COUPLED_FLOW_FIELDS` and
  `DISPERSIVE_KERR_FIELDS`. The
  compiled `_core` module and the C++ API are private.
- The layers are solprop -> Python backend adapter -> nanobind seam -> a Python-free
  C++20 core. Native code never imports Python or solprop concepts and never owns
  user-facing policy.
- solprop imports this package only lazily, through its backend registry. The wheel
  does not depend on solprop at run time; the contract is solprop's kernel-contract
  schema plus the members named by the tuples above, which `check_build()` checks at
  import, so mismatched halves (a stale extension beside newer sources) are refused by
  name.
- Kokkos runtime ownership is process-wide and locked. Initialise at most once, destroy
  plans and views before an owned finalisation, and never call Python during native
  teardown.

Changes to the public API, numerical conventions, precision, layout, backend support,
dependencies, runtime lifetime or ownership need discussion in an issue first. The
approved dependencies are Kokkos, Kokkos Kernels (BLAS component) with a vendor BLAS
behind it (the distribution build's BLAS on Linux and in the Windows wheel is OpenBLAS
built with OpenMP from pinned source; a Windows build without it falls back to
`scipy-openblas32`, a build requirement; each is carried in the wheel), Kokkos-FFT with
FFTW or the device FFT, SLEEF (the host vector maths, static, on x86-64 CPU builds;
solprop's ADR-059), nanobind
and scikit-build-core; GoogleTest is test-only. Adding any other dependency needs approval.

Major changes follow solprop's enhancement proposal process (SEPs,
[`docs/dev/seps/`](../docs/dev/seps/), SEP 0). A SEP is implemented `numpy_ref` first,
then `numba_cpu`, then here. A phase that touches both halves is one branch named for
the SEP (`feature/sep-NNNN-…`) and one pull request, whose commits cite the SEP number;
its phase report — deviations, decisions, verification, open questions — is appended to
the SEP. Until the native phase lands, the backend refuses the new problem by name.

## Scope

The backend executes the problems solprop hands it through the seam and refuses every
other problem by name, before allocating anything. When solprop adds a problem class,
the backend either executes it with tests against solprop's reference backend or
refuses it explicitly. Admission of the general modal route and the coupled flow is
solprop's shared list (`solprop.backends.numpy_ref.admission`), run under the name
`native`. Never claim accelerator support from a successful configuration or host
compilation alone; a device path is supported once its tests pass on that device.

## The gate

The native gate runs from the repository root:

```console
uv run poe check-native   # native/scripts/check.sh
uv run poe check-all      # solprop's gate (poe check), then the native gate
```

`check-native` covers this package's Python format, lint and strict typing, C++ format,
clang-tidy and warnings, native tests, ASan+UBSan native tests, an editable build with
pytest, a distribution wheel build and a clean wheel smoke test; per-file copyright and
licence (`reuse lint`) is in solprop's `poe check`, which covers `native/` too. A change
under `native/` is finished when `poe check-all` is green, so that solprop's test suite
has passed with this backend freshly built. Report the compiler, build type, execution
space, thread settings and any check that was not run.

CI (the repository's `.github/workflows/ci.yml`) runs on every push and pull request:
the `native-quality` jobs run `native/scripts/check.sh` on Linux, one group of its stages
each (`SOLPROP_NATIVE_CHECK_STAGES`). The `native-wheel` jobs build the CPU distribution
wheel on Linux, macOS and Windows, cached under a hash of `native/`, and solprop's
default lane runs on it on macOS and Windows with the native backend selected, and in
the coverage lane on Linux. GPU builds are validated by hand
(`.github/workflows/cuda.yml` on a self-hosted runner, or `pytest -m gpu` on a machine
with a device).

New behaviour needs a test; a bug fix needs a regression test. Never disable a test,
widen a tolerance after seeing a failure, lower warnings or hide a diagnostic to pass
the gate.

## Native implementation

- Standard C++20 only: no extensions, raw ownership, fast-math, RTTI, exceptions or I/O
  in device code, hidden default execution spaces, or backend-dependent field layout.
- Use RAII, Kokkos views, `Kokkos::complex`, device-callable math, named kernels and
  allocations (`solprop_native.*` labels), explicit `LayoutRight` and explicit
  execution-space instances.
- Allocate field-sized views, scratch and FFT plans once, when a state is constructed.
  A trial step performs no allocation, planning, logging, Python call or unplanned
  fence.
- Keep data resident in the execution space until it is observed; `field_to_host` is
  the explicit synchronous boundary. Every persistent allocation appears in
  `memory_ledger()` with its real size in bytes.
- Use `KokkosFFT::Normalization::none` and keep solprop's explicit scaling. Do not
  hand-roll transforms, reductions, scans, sorts, random number
  generators or other facilities that Kokkos, Kokkos Kernels or Kokkos-FFT supply.
- **Reproducibility to the error bound** (solprop SEP 25). Independent floating-point
  computations are compared by numerical error, not bits: no bitwise agreement is
  required across backends, builds, machines, thread counts, batch membership, scans or
  repeated runs. Matched discrete computations use rounding budgets derived before the
  comparison; adaptive runs and replay use accuracy envelopes. Exact structural
  contracts (lossless copies, integer accounting, identity metadata, hard-mask zeros,
  explicit no-ops) stay exact.
- Kernels may reorder, vectorise, thread and contract. Contraction is on by default
  (`-ffp-contract=fast` for GCC, `-ffp-contract=fast-honor-pragmas` for Clang and
  `icpx`: plain `fast` makes Clang ignore the pragmas). A protected block (a stable
  increment, a small-argument function, a defect or a compensated sum) stays strict
  with `SOLPROP_NATIVE_STRICT_FUNCTION` and `SOLPROP_NATIVE_STRICT_BLOCK`
  (`include/solprop_native/fp.hpp`; GCC ignores `#pragma STDC FP_CONTRACT`) and spells
  its complex arithmetic with the `strict_*` helpers, since `Kokkos::complex`'s
  operators keep their header's contraction when inlined. Reassociation is scoped to
  the loops that need it. No umbrella fast-math, `nnan`/`ninf`, `nsz`, approximate
  functions or reciprocals, or flush-to-zero; the `__FAST_MATH__` refusal stays.
  CUDA and HIP builds leave the strict blocks empty and follow their compilers'
  contraction defaults, held to the measured device envelopes (solprop's ADR-049,
  amended) until device round-to-nearest helpers land.
- A host kernel's hot transcendentals go through `include/solprop_native/vmath.hpp`
  (solprop's ADR-059): form a block's arguments into a buffer, call the batch function
  once over it, and finish each element from the results, behind
  `if constexpr (vmath::vectorised)` where the per-element form is faster with the
  system libm. Device code keeps the `Kokkos::` functions. The layer's functions keep
  their own error bounds (SLEEF's u10, 1.0 ulp); a protected block that uses one
  takes its result as an input.
- Reductions and scans take whatever order measures fastest within their rounding
  bound; the native-order GEMM (`native_order_gemm`) is held to the GEMM rounding
  bound against `KokkosBlas::gemm`, not to its bits.
- On the host, a team's vector loop vectorises only when its views are captured by
  value (use unmanaged copies) and each loop writes one output row.

## Python and nanobind seam

- The contract with solprop is its kernel-contract schema plus the `StepDriver`
  members and bundle fields listed in the contract tuples. `check_build()` refuses a
  schema mismatch or a missing member by name and records solprop's version without
  comparing it. Add a name to the tuples whenever the adapter starts reading another
  member; never pin solprop's version.
- Accept the documented NumPy dtype, rank, shape and C-contiguity exactly and refuse
  hidden O(n) conversions. Read-only input is valid. Returned host arrays own their
  memory.
- Validate extents and size arithmetic before allocating. Translate failures into
  specific exceptions carrying the operation, backend and shape.
- Release the GIL around long native work and hold it while touching Python objects.
  Never call Python from a Kokkos or OpenMP worker, or during teardown.
- Public Python has complete annotations, NumPy-style docstrings, `__all__`, `py.typed`
  and a complete stub for the private extension. Install a `NullHandler`; never
  configure the root logger or print diagnostics from library code.
- Thread settings are read once, at import. Never change process thread settings (or
  a BLAS thread pool) after import.

## Numerical and performance discipline

Document units, axes, signs, FFT normalisation, boundary conditions, precision and
NaN/Inf behaviour before code depends on them. Analytic results are the first oracle,
then invariants, then solprop's NumPy reference backend. Derive tolerances from machine
epsilon, operation counts, conditioning and step counts before running the tests.

Performance means end-to-end time to a correct answer. Warm up, measure a stable
baseline, change one thing, rerun the correctness tests, and report distributions,
hardware, compiler, backend, threads, problem shape, synchronisation and memory. Time
short windows (about ten accepted steps, or a propagation that completes in seconds)
and do not extrapolate them to a complete run. On a shared machine, compare builds
only as interleaved runs in the same session. Never add native-architecture flags to
a distributable build (build it with `SOLPROP_NATIVE_PROFILE=DISTRIBUTION`, whose
documented CPU baseline, x86-64-v3, Neoverse N1 or Apple M1, the package checks before
it loads the extension, and whose BLAS runs on every such machine), and never tune the
step-size controller to gain speed.

## Git and worktrees

- Create your own topic branch and `git worktree` for every change without asking,
  named for the topic: `fix/…`, `feature/…`, `docs/…`, and `feature/sep-NNNN-…` for a
  SEP phase.
- All development happens in a worktree, never in the main checkout. Each worktree
  has its own environment: `uv sync --extra native` at its root builds both halves
  from that worktree.
- Merge, push and tag only when asked. Never run a destructive or history-rewriting
  command (`git reset --hard`, `git checkout .`, `git restore` over uncommitted changes,
  `git clean`, `git commit --amend`, `git rebase`, `git push --force`): the work it
  destroys may not be yours.

## Repository hygiene

Read the files, tests, callers and build configuration before editing, and preserve
unrelated work. Write small commits with imperative subjects under 72 characters and
stage explicit paths. Build directories, wheels, fetched dependency trees, profiler
traces and local benchmark outputs are never committed. Record user-visible changes
in the repository's `CHANGELOG.md`.

Code is `GPL-3.0-or-later` and documentation `CC-BY-4.0`. Every code file starts with
the SPDX header that `uv run reuse annotate --copyright "John C. Travers and
contributors" --year 2026 --license GPL-3.0-or-later <file>` writes (above
`#pragma once` and any shebang); the repository's `REUSE.toml` covers the rest.
