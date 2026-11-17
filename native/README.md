# solprop-native

`solprop-native` is the compiled backend of
[solprop](https://solprop.readthedocs.io/en/stable/), a Python package for ultrafast
nonlinear pulse propagation, and a required dependency of it. It implements solprop's
propagation kernels in C++20 on
[Kokkos](https://github.com/kokkos/kokkos), bound to Python with
[nanobind](https://github.com/wjakob/nanobind) and built with
[scikit-build-core](https://github.com/scikit-build/scikit-build-core). One build runs
either on the CPU (OpenMP, with FFTW and a vendor BLAS) or on a GPU (CUDA with cuFFT and
cuBLAS, or HIP with rocFFT and rocBLAS). solprop's default `backend="auto"` selects it,
with no fallback. solprop's NumPy reference backend is the numerical oracle this package
is tested against.

The distribution is `solprop-native` and the import package `solprop_native`. It is
built from the `native/` directory of the
[solprop repository](https://github.com/LupoLab/solprop) and released at solprop's own
version.

## Installing

```sh
pip install solprop
```

installs the precompiled wheel on macOS on Apple silicon (osx-arm64), Linux on x86-64
(linux-64), Linux on aarch64 (linux-aarch64), Windows on x86-64 (win-64) and Windows on Arm
(win-arm64), with no extra; the x86-64 wheels need an x86-64-v2 processor (Intel Nehalem or
AMD Bulldozer and newer) and run faster on x86-64-v3 (AVX2), the aarch64 wheel needs a
Neoverse N1 or newer. On any other platform the installer finds no matching distribution.
A build from source is the way past a wheel's limits, for NVIDIA (CUDA) or AMD (HIP) GPUs
above all; a new release replaces a source build with its CPU wheel, so rebuild on upgrading.
The documentation covers both:

- [Installation](https://solprop.readthedocs.io/en/stable/introduction/installation.html)
- [Building the native backend](https://solprop.readthedocs.io/en/stable/guide/performance/building-native.html):
  requirements, CPU builds on Linux and macOS, build profiles and every CMake option,
  the CUDA and HIP builds, and OpenMP runtimes beside other packages
- [The native backend](https://solprop.readthedocs.io/en/stable/guide/performance/native.html):
  what it runs and running on a GPU
- [Performance tuning](https://solprop.readthedocs.io/en/stable/guide/performance/tuning.html):
  threads, BLAS settings and profiling with Kokkos Tools

## Development

The package is a member of the repository's uv workspace and shares its environment and
lockfile. At the repository root,

```sh
uv sync
```

builds it from `native/` and installs it, editable, beside solprop; a contributor with no
compiler has `uv run --no-project tools/setup_light.py`, which leaves it out. The editable
install does not rebuild itself: after changing C++ sources or CMake options, run
`uv sync --reinstall-package solprop-native`, passing options through
`CMAKE_ARGS` (a device build is made the same way, with the CUDA or HIP options). The
contributor rules for this directory are in [`AGENTS.md`](https://github.com/LupoLab/solprop/blob/main/native/AGENTS.md), beside the
repository's [`AGENTS.md`](https://github.com/LupoLab/solprop/blob/main/AGENTS.md).

### Source layout

| Directory | What lives there |
|---|---|
| `include/solprop_native/runtime/` | Execution spaces and launches, the floating-point policy (`fp.hpp`), vector maths, the CPU thread policy, crossings, field tables, refusals, the host source hook |
| `include/solprop_native/transforms/` | FFT plans and FFTW wisdom, the Raman FFT, the GEMM kernels, the slab transform (header-only) |
| `include/solprop_native/problem/` | `Problem`, the tables fixed for a workspace |
| `include/solprop_native/response/` | The material responses: the plasma band scans, envelope plasma, dispersive Kerr, the general modal source, the far field, the coupled flow |
| `include/solprop_native/march/` | The steppers: the Adams and Runge-Kutta marches, low-storage RK4IP, dense output, the engines, stage and z tables |
| `include/solprop_native/session/` | The native execution session (solprop SEP 47) and what it consults: the boundary watch, the transverse edge, the statistics plan |
| `include/solprop_native/general/` | `GeneralState`, the general route's state; its implementation is private to `src/general/` |
| `include/solprop_native/workspace/` | `Workspace`, with its lease, marches, session and probe; the run records (the set-once `Setup` and what a run starts from) |
| `src/{runtime,problem,response,march,session,workspace}/` | The definitions behind the headers of the same name |
| `src/general/` | The general workspace: refusals, setup, statistics, its kernel helpers (`kernels.hpp`, an umbrella over topic headers such as `views.hpp`, `traversal.hpp` and `stage_terms.hpp`), and the hot set, one translation unit (`trial.cpp`) in topic fragments (`trial_*.inc`) |
| `src/bindings/` | The nanobind seam: the `_core` module and its `_testing` submodule, one file a bound concept |
| `src/solprop_native/` | The Python adapter: `backend.py`, `_lowering.py`, `_load.py`, `_march.py`, `_probe.py`, `_device.py`, `_build.py`, `session.py`, … |
| `tests/native/` | The GoogleTest unit tests of the C++ library (`ctest`) |
| `tests/` | The Python tests, through the built extension |

Each layer of `runtime` ← `transforms`, `problem` ← `response`, `march` ← `session`,
`general` ← `workspace` ← `bindings` ← the Python adapter includes only those to its
left, with two exceptions among the headers. `general/general_state.hpp` sits low,
beside `problem.hpp`: the marches, the flow functions and the session's statistics
include it, and only `src/general/` sits high. The session's helper headers
(`watch.hpp`, `transverse.hpp`, `session_resident.hpp`, `session_dense.hpp`, …) sit
among the marches, which include them, and `readings.hpp` lower still, under
`general_state.hpp`; only `session.hpp` is above `march/`. `workspace/` sits above the
marches, the session and the general workspace, since the workspace owns them. The
strict floating-point files (`-ffp-contract=off`, see `AGENTS.md`) are those of
`src/march/` but `engines.cpp`, and every file of `src/session/`.

### Tests

From `native/`, in that environment:

```sh
uv run pytest                                  # Python tests
cmake --preset cpu-debug -DPython_EXECUTABLE="$PWD/../.venv/bin/python"
cmake --build --preset cpu-debug --parallel
ctest --test-dir build/presets/cpu-debug --output-on-failure   # native GoogleTest
```

On macOS add `-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++` to the
`cmake --preset` command. `cpu-sanitize` builds the native tests with ASan and UBSan
(`--target solprop_native_tests`). A wheel built with `-Ccmake.define.BUILD_TESTING=ON`
leaves the native tests in its build directory for `ctest --test-dir build/<wheel tag>`.

The Python suite compares this backend with solprop's reference backend within derived
bounds and runs thread-count probes in subprocesses; set `SOLPROP_NATIVE_THREADS` and
`OMP_PROC_BIND` as for a normal run. On a GPU build the same suite runs on the device,
and host-only tests skip themselves. `pytest -m gpu` selects the focused numerical tests
(`tests/test_numerics.py`), small enough to run under Compute Sanitizer, and the band
rule's device qualification (`tests/test_band_plasma.py`). To compare the band rule's
two halo reads (`SOLPROP_NATIVE_BAND_STAGING`), run `python tests/band_thread_probe.py`
under each build and compare the arrays it writes within the band test's rounding bound.
To qualify the GPU's tiled band column on a CPU build, run `tests/test_band_plasma.py`
and `tests/test_band_stress.py` with `SOLPROP_NATIVE_BAND_TILES=4`; on a GPU,
`SOLPROP_NATIVE_BAND_TILES=0` runs the team column instead.

### Quality gate

```sh
uv run poe check-native   # at the repository root; runs native/scripts/check.sh
uv run poe check-all      # solprop's, this and the front end's gates (gui extra)
uv run poe fmt-native     # format every tracked C++ file in place
uv run poe tidy-native    # clang-tidy on the C++ files changed from main, or on those named
```

`poe check-native` runs everything a change to `native/` must pass: Ruff formatting and
lint, strict Pyright, the pinned `clang-format` from the `dev` group over every tracked
C++ file, the pinned clang-tidy (whole check families, `.clang-tidy`) with strict
warnings, the native tests, the native tests under ASan and UBSan, an editable
local build with the Python tests, a distribution wheel build, and an import smoke test
of that wheel in a clean environment (on Linux it checks that the wheel's OpenBLAS loads
from the package). It syncs the repository's environment without removing the native
build, honours `CMAKE_ARGS`, and needs Node.js (`npx`) for Pyright. `SOLPROP_NATIVE_CHECK_STAGES`
picks stages (`lint`, `tidy`, `sanitize`, `pytest`, `wheel`). The slower checks run
weekly in `.github/workflows/native-deep.yml` and report without gating: source
coverage, the extension under ASan and UBSan, and an include-cleaner report.
`SOLPROP_NATIVE_QUALIFICATION_PYTHON=/path/to/python` installs the wheel into that
environment and runs the Python tests there instead. A change to the backend also passes
solprop's own suite with this package installed (`poe check`, or `uv run poe test`
with `SOLPROP_TEST_BACKEND=native`).

### The kernel contract

This source implements solprop's kernel contract, whose fields solprop's frozen tables
hold (`tests/backends/test_driver.py`). A build pins no solprop version. In an editable
checkout `check_build()` compares `build_info().source_digest`, the digest of what the
extension was compiled from, with this directory's sources. Rebuild with
`uv sync --reinstall-package solprop-native` when it refuses; a wheel carries no sources,
and solprop's metadata pins it to solprop's own version. The version recorded at configure time
(`solprop_version_built_against`: the repository's version, or `unknown` for a direct
CMake configure) is provenance only. The TDSE electronic response runs on CPU builds
only (solprop SEP 49); a device build refuses it by name before anything is allocated.

### The binding

The private extension `_core` is a thin seam (solprop SEP 52 section 5.3):

- **One workspace, one run record.** A backend builds one `_core.Problem`, which
  carries every table, guard and setting fixed for the workspace, and one
  `_core.Workspace` from it for its life. Each run starts from one record
  (`Workspace.start(run)`), and the timing probe from the run's start
  (`Workspace.probe(start)`); `Workspace.diagnostics()` returns every fact the
  metadata reads. A session or probe leases the workspace until it closes.
- **Records by field table.** Each C++ record declares its fields once, beside its
  definition (`include/solprop_native/runtime/fields.hpp`), and one converter
  (`src/bindings/records.hpp`) reads a dataclass, mapping or tuple into it, refusing a missing
  or unknown field, a wrong dtype, rank or extent relation, a strided array or an
  enumeration given by anything but its name, by the field's path before copying
  anything. `_core.schemas()` exports every table, and a test holds each Python record
  to its table. Input arrays are copied in, except a run's initial field and noise,
  which the start reads where they are, for its load only; returned arrays own their
  storage.
- **Bindings only marshal.** A binding converts, releases the interpreter lock and
  forwards. The lease, the coupled flow's guard, the marches' construction and the
  probe live in the C++ library (`native::Workspace`, `src/workspace/workspace.cpp`), where the
  native tests reach them.
- **`_testing` promises nothing.** The test-only entry points live in the submodule
  `_core._testing`, shipped in the wheel for solprop's own tests; they take the
  workspace first and may change with any commit. `_testing.engines(workspace)` leases
  the workspace's Runge-Kutta (or low-storage) and Adams engines, which solprop's
  tests run through the same engine protocols as the reference's.
- **Generated stubs.** `_core.pyi` and `_core_testing.pyi` are generated whole by
  nanobind's stubgen (`scripts/problem_stub.py`; `--check` in the lint stage). A binding
  that takes or returns a record names its Python type with `nb::sig`, so the stubs
  stay typed: edit the binding and regenerate, never the stub.

### Native source coverage

Build coverage in a separate build directory and Python environment. On Linux select
Clang and its matching OpenMP runtime explicitly; the usual GCC build cannot provide
Clang source coverage. On macOS the configured Homebrew LLVM toolchain supplies the
compiler and tools. With solprop installed in that environment, from `native/`:

```sh
uv build --wheel -Ccmake.define.SOLPROP_NATIVE_ENABLE_COVERAGE=ON \
    -Ccmake.define.CMAKE_CXX_COMPILER=/path/to/llvm/bin/clang++ \
    -Cbuild-dir=build/coverage
```

This instruments the core, the bindings and the instantiated headers at `-O1` with
`-fprofile-instr-generate -fcoverage-mapping -fprofile-update=atomic`, with IPO
disabled. `build_info().coverage_instrumented` identifies it, and `coverage_source_hash`
binds the build to its ordered C++ sources, uncommitted edits included. Its wall times
are not performance evidence.

Set `LLVM_PROFILE_FILE` before importing the extension, using a new directory for each
case and `%p-%m.profraw` filenames so that processes and instrumented modules do not
overwrite each other's profiles. Use `llvm-profdata` and `llvm-cov` from the same LLVM
installation as the compiler, keep the extension hash and source revision with every
collection, and never combine profiles from different binaries. Filter reports to
`native/src/` and `native/include/solprop_native/`: templates from Kokkos, Kokkos-FFT,
Kokkos Kernels and other dependencies are not project coverage.

The coverage build also installs `solprop_native_labels`, a host-only Kokkos Tools
module that counts parallel-for/reduce/scan labels and profile regions without
requesting fences. Set `KOKKOS_TOOLS_LIBS` to this module and `SOLPROP_NATIVE_LABEL_FILE`
to a new JSON-lines output path for a separate label pass. Reports, raw profiles and binary snapshots belong
in ignored local `.benchmarks/` storage, not in git. An uninstantiated template or
compiled-out path has no counter and is not an executed path with a zero count; CPU
coverage has no target percentage.

## AI-assisted development

The C++ code and build infrastructure here are slightly gnarly, and developing and
maintaining them is enormously aided by AI coding agents working under the lead
developer's direction, as in solprop. Verification rests on solprop's 100% test
coverage and systematic comparisons of this backend against its validated NumPy
reference backend: the results agree within derived floating-point error bounds. See
solprop's
[account of AI-assisted development](https://solprop.readthedocs.io/en/stable/introduction/about.html#use-of-ai)
for the wider approach to derivation, testing and review.

## Licence

solprop-native is free software: you can redistribute it and modify it under the terms
of the GNU General Public License, version 3 or (at your option) any later version
(`GPL-3.0-or-later`), like solprop; see [LICENSE](https://github.com/LupoLab/solprop/blob/main/native/LICENSE), a copy of the repository's
for the wheel. The documentation is under [CC BY 4.0](https://github.com/LupoLab/solprop/blob/main/LICENSES/CC-BY-4.0.txt). Every
file states its copyright and licence, in an SPDX header or in the repository's
[`REUSE.toml`](https://github.com/LupoLab/solprop/blob/main/REUSE.toml), following the [REUSE specification](https://reuse.software/);
`uv run poe reuse` at the repository root checks it, as part of `poe check`. Start a new
code file with the header that `uv run reuse annotate --copyright "John C. Travers and
contributors" --year 2026 --license GPL-3.0-or-later <file>` writes.

The fetched dependencies (Kokkos, Kokkos Kernels, Kokkos-FFT, SLEEF, GoogleTest) and the
libraries a build links (FFTW, the BLAS, the CUDA or ROCm libraries) carry their own
licences.
