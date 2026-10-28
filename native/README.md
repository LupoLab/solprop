# solprop-native

`solprop-native` is the optional compiled backend for
[solprop](https://solprop.readthedocs.io/en/latest/), a Python package for ultrafast
nonlinear pulse propagation. It implements solprop's propagation kernels in C++20 on
[Kokkos](https://github.com/kokkos/kokkos), bound to Python with
[nanobind](https://github.com/wjakob/nanobind) and built with
[scikit-build-core](https://github.com/scikit-build/scikit-build-core). One build runs
either on the CPU (OpenMP, with FFTW and a vendor BLAS) or on a GPU (CUDA with cuFFT and
cuBLAS, or HIP with rocFFT and rocBLAS). When it is installed beside solprop, solprop's
default `backend="auto"` selects it. It is never needed for correctness: solprop's
NumPy reference backend is the numerical oracle, and this package accelerates the
problem classes it supports.

The distribution is `solprop-native` and the import package `solprop_native`. It is
built from the `native/` directory of the
[solprop repository](https://github.com/LupoLab/solprop) and released at solprop's own
version.

## Installing

```sh
pip install "solprop[native]"
```

installs the precompiled wheel on macOS on Apple silicon (osx-arm64), Linux on x86-64
(linux-64), Linux on aarch64 (linux-aarch64) and Windows on x86-64 (win-64); the x86-64
wheels need an x86-64-v3 processor (AVX2: Intel Haswell or AMD Excavator and newer), the
aarch64 wheel a Neoverse N1 or newer. Elsewhere,
and for NVIDIA (CUDA) or AMD (HIP) GPUs, build it from source. The documentation
covers both:

- [Installation](https://solprop.readthedocs.io/en/latest/introduction/installation.html)
- [Building the native backend](https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html):
  requirements, CPU builds on Linux and macOS, build profiles and every CMake option,
  the CUDA and HIP builds, and OpenMP runtimes beside pyFFTW
- [The native backend](https://solprop.readthedocs.io/en/latest/guide/performance/native.html):
  what it runs and running on a GPU
- [Performance tuning](https://solprop.readthedocs.io/en/latest/guide/performance/tuning.html):
  threads, BLAS settings and profiling with Kokkos Tools

## Development

The package is a member of the repository's uv workspace and shares its environment and
lockfile. At the repository root,

```sh
uv sync --extra native
```

builds it from `native/` and installs it, editable, beside solprop; a plain `uv sync`
leaves it out and removes a copy installed by hand. The editable install does not rebuild
itself: after changing C++ sources or CMake options, run
`uv sync --extra native --reinstall-package solprop-native`, passing options through
`CMAKE_ARGS` (a device build is made the same way, with the CUDA or HIP options). The
contributor rules for this directory are in [`AGENTS.md`](https://github.com/LupoLab/solprop/blob/main/native/AGENTS.md), beside the
repository's [`AGENTS.md`](https://github.com/LupoLab/solprop/blob/main/AGENTS.md).

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
uv run poe check-all      # solprop's gate, then this one
```

`poe check-native` runs everything a change to `native/` must pass: Ruff formatting and
lint, strict Pyright, the pinned `clang-format` from the `dev` group, clang-tidy with
strict warnings, the native tests, the native tests under ASan and UBSan, an editable
local build with the Python tests, a distribution wheel build, and an import smoke test
of that wheel in a clean environment (on Linux it checks that the wheel's OpenBLAS loads
from the package). It syncs the repository's environment without removing the native
build, honours `CMAKE_ARGS`, and needs Node.js (`npx`) for Pyright.
`SOLPROP_NATIVE_QUALIFICATION_PYTHON=/path/to/python` installs the wheel into that
environment and runs the Python tests there instead. A change to the backend also passes
solprop's own suite with this package installed (`poe check-all`, or `uv run poe test`
with `SOLPROP_TEST_BACKEND=native`).

### The kernel contract

This source implements solprop's kernel-contract schema 34. A build pins no solprop
version: it requires the schema version (`build_info().schema_version`) and the solprop
members listed in the exported tuples `DRIVER_SURFACE`, `STAGE_TERMS_FIELDS`,
`STAGE_SCALARS_FIELDS` and the other `*_FIELDS` tuples, which `check_build()` checks at
import, naming any that are missing. Rebuild when solprop's schema version changes or
`check_build()` names a missing member. The version recorded at configure time
(`solprop_version_built_against`: the repository's version, or `unknown` for a direct
CMake configure) is provenance only. The TDSE electronic response is refused by name
before anything is allocated; TDSE propagation uses `backend="numpy_ref"`.

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
[account of AI-assisted development](https://solprop.readthedocs.io/en/latest/introduction/about.html#use-of-ai)
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

The fetched dependencies (Kokkos, Kokkos Kernels, Kokkos-FFT, GoogleTest) and the
libraries a build links (FFTW, the BLAS, the CUDA or ROCm libraries) carry their own
licences.
