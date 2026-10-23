# Releasing and the native wheels

solprop is released as two distributions from one repository, at one version taken from
the git tag ([SEP 29](seps/sep-0029-compiled-backend-distribution.md)):

- **`solprop`**, pure Python: a source distribution and one universal wheel.
- **`solprop-native`**, the compiled backend built from `native/`: **wheels only**, one
  per supported platform. No source distribution is published, so that where no wheel
  matches, `pip install "solprop[native]"` fails to resolve instead of starting a C++
  build; source builds are made from the repository
  ([Building the native backend](../guide/performance/building-native.md)).

The wheels are built, repaired, qualified and published by
`.github/workflows/wheels.yml`. cibuildwheel's configuration is the
`[tool.cibuildwheel]` table of `native/pyproject.toml`; the libraries each wheel bundles
are built from pinned, checksummed sources by the scripts in `native/ci/`, each of which
also runs by hand.

## The wheels

| | Linux x86-64 | Linux aarch64 | macOS arm64 | Windows x64 |
|---|---|---|---|---|
| CI runner | `ubuntu-24.04` | `ubuntu-24.04-arm` | `macos-15` | `windows-2025` |
| build environment | `manylinux_2_28` container, GCC 14 | `manylinux_2_28` container, GCC 14 | Homebrew's Clang, deployment target 14.0 (macOS Sonoma) | Clang on the MSVC ABI (MSVC STL and CRT, `/MD`) |
| CPU baseline | x86-64-v3 | Neoverse N1 | Apple M1 | x86-64-v3 |
| host execution space | Kokkos OpenMP | Kokkos OpenMP | Kokkos OpenMP | Kokkos OpenMP |
| OpenMP runtime | GCC's `libgomp` | GCC's `libgomp` | LLVM `libomp`, built for the target | LLVM `libomp` from the LLVM installation |
| BLAS | OpenBLAS 0.3.34, OpenMP, `DYNAMIC_ARCH`, `solprop_` prefix (built by the configure step) | the same | Accelerate | OpenBLAS 0.3.34, OpenMP, `DYNAMIC_ARCH`, static |
| FFTW 3.3.11 | shared, SSE2 and AVX codelets | shared, cycle counter | shared, cycle counter | static, SSE2 and AVX codelets |
| repair | `auditwheel` | `auditwheel` | `delocate` | `delvewheel` (mangled names) |

Every bundled library runs on the same OpenMP runtime as Kokkos, so a process holds one
thread pool for the backend. The wheel carries the licence text of every library compiled
into it or copied in by the repair (`LICENSE-*.txt` beside the extension). CPython 3.14
only; free-threaded CPython is not a target.

## Building a wheel locally

Every wheel is the distribution profile (`SOLPROP_NATIVE_PROFILE=DISTRIBUTION`), never
a build for the build machine's own architecture.

**macOS arm64.** cibuildwheel needs python.org's framework Python, which it installs only
on CI. `native/ci/build-wheel-macos.sh` runs the same steps with uv: it builds `libomp`
and FFTW for the deployment target (`native/ci/build-deps.sh`), builds the wheel and
repairs it with `delocate`.

```sh
brew install llvm cmake ninja
native/ci/build-wheel-macos.sh /tmp/wheels   # the repaired wheel lands in /tmp/wheels
```

Keep the script and `[tool.cibuildwheel.macos]` in step: they set the same deployment
target, CMake arguments and `delocate` check.

**Linux** (x86-64 or aarch64, on a machine of that architecture with Docker):

```sh
uvx cibuildwheel --platform linux --output-dir wheelhouse native
```

**Windows x64**, from a Visual Studio developer prompt (the x64 MSVC tools and SDK in
the environment), with LLVM, CMake and Ninja installed:

```sh
uvx cibuildwheel --platform windows --output-dir wheelhouse native
```

To build one platform's dependencies by hand: `native/ci/build-deps.sh PREFIX` (Linux,
macOS) or `native/ci/build-deps-windows.ps1 -Prefix PREFIX`.

## Qualifying a wheel

The workflow's test jobs install each repaired wheel beside solprop from the same
checkout and run, in order:

1. `python -I native/ci/smoke.py --version VERSION`: the package imports and passes
   `check_build()`, is the distribution build at the release's version, loads the FFT,
   BLAS and OpenMP libraries from inside the wheel, and carries their licences;
2. `python -I -m solprop_native`: the build record;
3. `python -I native/ci/coexistence.py`: NumPy, SciPy, pyFFTW and Numba imported before
   and after `solprop_native`, each order in a fresh process, every package's thread
   pool started, and a short propagation run on the native backend, on `numba_cpu` and on
   the native backend again in the same process;
4. `uv run --no-sync poe test` with `SOLPROP_TEST_BACKEND=native`: solprop's default
   lane on the native backend.

On Linux x86-64, the `cpu-floor` job runs the wheel under Intel's Software Development
Emulator: the smoke test on a Haswell (x86-64-v3); a refusal before any native code runs
on a Nehalem (x86-64-v2, no AVX) and a Sandy Bridge (AVX without AVX2, FMA or BMI), naming
the missing features; and, on the Sandy Bridge, `backend="auto"` resolving to
`numba_cpu` with its one warning. It runs when the repository variables `SDE_URL` and
`SDE_SHA256` name an SDE release. The other platforms' checks (Windows through
`IsProcessorFeaturePresent`, Linux aarch64 through `/proc/cpuinfo`'s `Features`) are
unit-tested with stand-in processors in `native/tests/test_cpu_baseline.py`; no CI job
runs them on a processor below the baseline.

**macOS and pyFFTW.** pyFFTW's macOS wheel carries its own `libomp`, which it initialises
when it is imported. LLVM's OpenMP runtime refuses, or crashes on, a second initialised
copy, so the native package raises at import when pyFFTW is already loaded
([OpenMP runtimes and pyFFTW](../guide/performance/building-native.md#openmp-runtimes-and-pyfftw)).
solprop imports both backends lazily, so a script that only calls solprop is unaffected.
`coexistence.py` expects that refusal on macOS; `--bypass-guard` adds a report-only
scenario that hides pyFFTW from the guard, to test whether it is still needed.

## Making a release

1. On `main`, with CI green, check that the changelog's `Unreleased` section describes
   the release, then move it under the new version.
2. Tag the release, `git tag -a vX.Y.Z`, and push the tag. The wheels workflow builds
   and tests every wheel and builds solprop's distributions.
3. Publishing runs on the tag once the repository variable `PUBLISH` is `true`, through
   PyPI's trusted publishing from the `pypi` environment: solprop's sdist and wheel,
   and solprop-native's wheels. Each PyPI project (`solprop`, `solprop-native`) names
   this repository, the workflow `wheels.yml` and the environment `pypi` as its trusted
   publisher.
4. Install the published release in a fresh environment, `pip install "solprop[native]"`
   on each platform, and check that `backend="auto"` resolves to `native`.

The actions in the workflows are pinned by commit; when updating one, take the commit of
its release tag and keep the version in the trailing comment.
