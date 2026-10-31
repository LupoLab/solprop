# Building the native backend

The [native backend](native.md) is the `solprop-native` package, built from the
`native/` directory of the solprop repository at solprop's own version. On the four
platforms with precompiled wheels — macOS on Apple silicon (osx-arm64), Linux on x86-64
(linux-64), Linux on aarch64 (linux-aarch64) and Windows on x86-64 (win-64) —
`pip install "solprop[native]"` installs it and nothing on this page is needed
([installation](../../introduction/installation.md)). Build it from source to:

- run on a **GPU**: NVIDIA with CUDA or AMD with HIP. There are no GPU wheels;
- get the **local** build, compiled for the machine that runs it with that machine's
  vendor BLAS, which is a few per cent faster than the wheel on most cases and more on
  some;
- run on a platform without a wheel, or work on the backend itself.

The instructions cover Linux and macOS. One build is either a CPU build or a GPU build;
the arithmetic is double precision throughout, and no build enables fast-math.

## Requirements

| requirement | version and notes |
|---|---|
| Python | 3.14, with NumPy 2.3 or newer, and solprop at the same version installed in the same environment |
| CMake | 3.23 or newer (with build isolation, scikit-build-core fetches one from PyPI if none is found) |
| build tool | Ninja is recommended; scikit-build-core uses it when available |
| C++20 compiler, macOS | Clang 14 or newer; Homebrew LLVM is recommended and selected by default |
| C++20 compiler, Linux CPU | GCC 14 or newer, IntelLLVM (`icpx`) 2022 or newer, or Clang 14 or newer / AMD AOCC |
| C++20 compiler, CUDA | GCC 14 or newer as the host compiler, with `nvcc` from CUDA 12.8 or newer (qualified on CUDA 13) |
| C++20 compiler, HIP | ROCm's Clang (`hipcc` or `amdclang++`) |
| OpenMP (CPU builds) | the compiler's runtime; on macOS Homebrew's `libomp` |
| FFT (CPU builds) | FFTW 3 with its threaded libraries, in double and single precision (or MKL's FFTW interface) |
| BLAS (CPU builds) | optional but recommended: Accelerate (macOS), OpenBLAS, MKL or another LP64 BLAS |
| GPU libraries | cuFFT and cuBLAS (CUDA), or rocFFT and rocBLAS (HIP) |
| network | configuring fetches the pinned dependencies below from GitHub (see [offline builds](#offline-builds)) |
| git | the source must be a git clone of the repository: configuring records the commit, and the version comes from the repository's tags |

CMake fetches and builds these at pinned versions; nothing needs to be installed for
them:

| dependency | version | use |
|---|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | 5.2.1 | execution spaces, views and parallel patterns |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) | 5.2.0 | dense GEMM (BLAS component only), dispatching to the vendor BLAS |
| [Kokkos-FFT](https://github.com/kokkos/kokkos-fft) | 2.0.0 | FFTs over FFTW, cuFFT or rocFFT |
| [SLEEF](https://github.com/shibatch/sleef) | 3.9.0 | vector `exp`, `log` and `sincos` for the CPU kernels on x86-64 (see [vector maths](#vector-maths)); built static as its own CMake project, and not fetched where the system libm is used |
| [GoogleTest](https://github.com/google/googletest) | 1.17.0 | the native tests only, when `BUILD_TESTING=ON` |

The Python build requirements are scikit-build-core 0.11 and nanobind 2.9 or newer. CPU
builds contract multiply-adds (`-ffp-contract=fast` with GCC,
`-ffp-contract=fast-honor-pragmas` with Clang and `icpx`, after `-fno-fast-math` and
`-fp-model=precise`) except in a few protected blocks, as
[SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md) sets out;
`build_info().compiler_flags` and `optimization_policy` record the setting. GPU builds
follow their compilers' defaults throughout ([GPU builds](#gpu-builds)).

## A CPU build

Clone the repository and install both packages from it into one environment.
`/path/to/env/bin/python` stands for the Python of that environment.

**macOS** (Apple silicon or Intel), with [Homebrew](https://brew.sh):

```sh
brew install llvm libomp fftw cmake ninja pkg-config
git clone https://github.com/LupoLab/solprop
cd solprop
uv pip install --python /path/to/env/bin/python . ./native
```

`native/pyproject.toml` selects Homebrew's Clang (`/opt/homebrew/opt/llvm/bin/clang++`
on Apple silicon, `/usr/local/opt/llvm/bin/clang++` on Intel), and CMake finds Homebrew's
keg-only FFTW and `libomp` by itself. The build links Apple's Accelerate. If the Xcode
licence state blocks the Command Line Tools, select their SDK explicitly and use a fresh
build directory:

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export CMAKE_ARGS="-DCMAKE_OSX_SYSROOT=$SDKROOT"
```

For an Apple silicon performance build, the pinned dependency recipe enables FFTW
3.3.11's NEON codelets in **both precisions**. Compiling scalar code with `-mcpu`
does not enable those codelets. The recipe also retains the Arm cycle counter needed
by measured planning, the Apple M1 distribution baseline and the macOS 14 deployment
target. Its FFTW threading libraries and the native extension share one source-built
LLVM OpenMP runtime; Accelerate remains the BLAS provider. The Linux aarch64 wheel's
recipe enables the same NEON codelets, on its Neoverse N1 baseline.

From the repository root, use a fresh, owned dependency prefix:

```sh
native_deps="$PWD/native/build/neon-dependencies"
MACOSX_DEPLOYMENT_TARGET=14.0 native/ci/build-deps.sh "$native_deps"
export CMAKE_ARGS="-DFFTW_ROOT=$native_deps -DSOLPROP_NATIVE_LIBOMP_LIBRARY=$native_deps/lib/libomp.dylib -DSOLPROP_NATIVE_LIBOMP_INCLUDE_DIR=$native_deps/include -DSOLPROP_NATIVE_BUNDLED_LICENCES=$native_deps/licences"
uv sync --extra native --reinstall-package solprop-native
uv run --no-sync python -m solprop_native
```

Check the reported `fft_library` and `openmp_library` before benchmarking. The
[pinned upstream configuration](https://github.com/FFTW/fftw3/blob/fftw-3.3.11/configure.ac)
supports `--enable-neon` for double and single precision on AArch64. The installed
libraries identify their codelet family in `fftw_version` and `fftwf_version`, which
both end in `-neon` for this recipe. Keep these prefixes separate from a comparison's
baseline libraries. A pyFFTW build used in the same process must also share this
runtime. To build the existing pyFFTW dependency against that same owned prefix:

```sh
pyfftw_version=$(uv run --no-sync python -c 'from importlib.metadata import version; print(version("pyfftw"))')
CC=/opt/homebrew/opt/llvm/bin/clang \
CXX=/opt/homebrew/opt/llvm/bin/clang++ \
PYFFTW_FFTW_PREFIX="$native_deps" \
LDFLAGS="-L$native_deps/lib -Wl,-rpath,$native_deps/lib" \
uv pip install --python .venv/bin/python --no-deps --no-binary pyfftw \
  --reinstall --no-cache "pyfftw==$pyfftw_version"
```

The uncached source build prevents reuse of a wheel linked to another runtime.
Keep the project's locked pyFFTW version when qualifying a comparison. See
[OpenMP runtimes](#openmp-runtimes-and-pyfftw) below for import ordering and the
limits of mixing independently bundled wheels.

For a distributable wheel, `native/ci/build-wheel-macos.sh OUT_DIR DEPS_PREFIX`
builds the distribution profile, repairs its library references with delocate, and
includes the FFTW and OpenMP licences. Use a fresh dependency prefix when changing
the recipe: that helper intentionally reuses an existing prefix. Validate the repaired
wheel in a clean environment with `python -I -m solprop_native`, its linked libraries,
the native tests and the import-order/coexistence checks before distributing it.

**Linux**, on Debian or Ubuntu. Ubuntu 24.04's default compiler is GCC 13, which the build
refuses; install GCC 14 from the distribution's archive:

```sh
sudo apt install g++-14 cmake ninja-build pkg-config libfftw3-dev libopenblas-dev
git clone https://github.com/LupoLab/solprop
cd solprop
CXX=g++-14 uv pip install --python /path/to/env/bin/python . ./native
```

`libfftw3-dev` provides the OpenMP and pthreads FFTW libraries in both precisions. For
MKL instead of OpenBLAS, source oneAPI first (`source /opt/intel/oneapi/setvars.sh`): a
set `MKLROOT` makes the automatic search choose MKL. Other compilers and libraries are
in [Linux compilers and libraries](#linux-compilers-and-libraries).

`python -m pip install . ./native` works the same way in an activated environment.
Compilation takes several minutes; `uv` and `pip` hide the build output, and `-v` shows
it, including the configure summary line described [below](#build-profiles).

## A development build

In a development checkout, one command builds the native package from `native/` and
installs it editable into the repository's environment beside solprop:

```sh
uv sync --extra native
```

A plain `uv sync` leaves it out, and removes a copy installed by hand. The editable
install does not rebuild itself: run `uv sync --extra native --reinstall-package
solprop-native` after a native change, with any build options in `CMAKE_ARGS`
([below](#passing-cmake-options)); a device build is made the same way. `uv run poe
check-native` runs the native gate and `uv run poe check-all` both gates. A build stays
valid across solprop changes until solprop's kernel schema (`SCHEMA_VERSION`) changes or
`check_build()` names a member the installed solprop lacks. Running solprop's test lanes
against the build is in the developer notes on
[testing](../../dev/testing.md#tiers-and-backends), and the native package's own tests in
`native/README.md`.

## Checking a build

```sh
python -m solprop_native | python -m json.tool
python -c "import solprop_native as sn; sn.check_build(); print(sn.build_info().build_profile)"
```

`python -m solprop_native` prints the build record, `build_info()`, as one sorted JSON
object suitable for a run log:

- **provenance:** `package_version`, `git_commit` (with `+dirty` for a modified tree),
  `schema_version` (the kernel-contract version the build implements),
  `solprop_version_built_against`;
- **dependencies:** `kokkos_version`, `kokkos_kernels_version`, `kokkos_fft_version`,
  `nanobind_version`;
- **profile and BLAS:** `build_profile`, `cpu_baseline`, `gemm_provider`,
  `tpl_blas_enabled`, `blas_backend`, `blas_selection`, `blas_thread_policy`,
  `blas_threads`, `blas_threading`, `blas_library`;
- **execution:** `execution_space`, `host_execution_space`, `device_backend`,
  `device_arch`, `device_count`, `openmp_enabled`, `openmp_library`;
- **FFT:** `fft_backend`, `fft_planner`, `fft_normalization`, `fft_library`,
  `raman_fft_option`, `raman_fft_library`;
- **threads:** `requested_threads`, `runtime_threads`, `thread_count`, `fftw_threads`,
  `raman_fft_threads`, `raman_tail_fft_threads`, `threaded_fft_min_samples`
  ([Performance tuning](tuning.md#the-native-backends-pools));
- **compilation:** `compiler`, `compiler_family`, `compiler_flags`, `cxx_standard`,
  `build_type`, `native_arch_enabled`, `aggressive_vectorization_enabled`,
  `ipo_enabled`, `fast_math_enabled`, `optimization_policy`, `precision`, `layout`;
- **host:** `python_version`, `numpy_version`, `platform_machine`.

The library paths (`fft_library`, `blas_library`, `openmp_library`) are the files
actually loaded, the reliable way to confirm which FFT and BLAS a process uses.
`raman_fft_option` identifies the compiled Raman wrapper, which can select FFTW for
individual plans even when its value is `ACCELERATE`. The backend's execution
metadata records the selected full-chunk `raman_fft_provider` and full/tail task
counts; selection also accounts for the actual tail width.
`native_arch_enabled` reports the package's option only; architecture flags passed by
hand through `CXXFLAGS` appear in `compiler_flags`.

`check_build(problem=None)` raises `solprop.exceptions.BackendError` when the installed
solprop lacks a driver member or bundle field the build relies on (it names them);
when `problem` declares a different kernel-contract schema; when the build enables
fast-math; when the execution-space record is inconsistent; when a GPU build sees no
device; when the GEMM provider is unknown or its BLAS is missing or unidentified; when
a reference build carries a vendor BLAS or machine-specific code, or a distribution
build carries native code or no BLAS; and when a GPU build's BLAS is not the device's.
Then, in solprop:

```python
import solprop
from solprop.backends.registry import resolve_name

print(resolve_name("auto"))  # "native" when solprop_native is installed
solprop.configure_logging("INFO")  # each run logs "Using native backend"
```

`resolve_name` only probes whether the package can be found; the log line reports the
backend actually constructed.

## Build profiles

A plain build is the **local** profile, the fast build for the machine it is built on:

- a **vendor BLAS** behind Kokkos Kernels, found automatically: Accelerate on macOS; on
  Linux MKL when `MKLROOT` (or `SOLPROP_NATIVE_MKL_ROOT`) names it and its LP64 libraries
  are found, otherwise OpenBLAS; cuBLAS or rocBLAS in a GPU build;
- the `HYBRID` dense-transform policy, which sends the eligible multimode transforms to
  the vendor GEMM;
- code for **the build machine's own architecture** (`Kokkos_ARCH_NATIVE`, for example
  `-march=native`) in CPU builds, and checked interprocedural optimisation (IPO/LTO) in
  Release builds.

If the search finds no BLAS, a local build still succeeds with a configure warning and
runs the native-order GEMM. A vendor BLAS matters most for the dense modal transforms of
multimode problems; single-mode and free-space problems depend on it much less.

The **distribution** profile, `SOLPROP_NATIVE_PROFILE=DISTRIBUTION`, is the build the
wheels are made with, for machines other than the build machine. It is a CPU profile:

- a **CPU baseline** instead of the build machine's architecture: `-march=x86-64-v3`
  (AVX2, FMA, BMI1/2: Intel Haswell, AMD Excavator and newer) on x86-64,
  `-mcpu=apple-m1` on Apple silicon, `-mcpu=neoverse-n1` on Linux aarch64. For an older
  x86-64 processor (no AVX2), build with `SOLPROP_NATIVE_CPU_BASELINE=x86-64-v2`
  (SSE4.2 and POPCNT): on the benchmark core it costs up to 2 to 4 %, and on the general
  modal route about a quarter at one thread and 7 to 12 % at full width, against
  `x86-64-v3`. The bundled FFTW and OpenBLAS choose their kernels at run time either
  way. The package records its baseline and refuses to import on a processor below it,
  naming the missing features (from `/proc/cpuinfo` on Linux, from
  `IsProcessorFeaturePresent` on Windows, where AVX2 stands for the other x86-64-v3
  features Windows does not report); solprop's `backend="auto"` then uses `numba_cpu`
  with a warning;
- a **BLAS** that every such machine has and that shares the extension's threads:
  Accelerate on macOS; on Linux an OpenBLAS 0.3.34 that the configure step builds with
  OpenMP and the extension's own compiler (`native/cmake/OpenMPOpenBLAS.cmake`), so that
  it runs on the extension's OpenMP runtime, selects its kernels at run time and carries
  `solprop_`-prefixed symbols that never collide with NumPy's copy. It is built once per
  build tree; `SOLPROP_NATIVE_OPENBLAS_ROOT` names a prefix that already holds one. The
  Windows wheel links the same OpenBLAS release built static with OpenMP on LLVM's
  `libomp`, the runtime of its Kokkos and FFTW, named by `SOLPROP_NATIVE_OPENBLAS_ROOT`
  with `SOLPROP_NATIVE_DISTRIBUTION_BLAS=OPENMP_OPENBLAS`; a Windows distribution build
  without them links the [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/)
  OpenBLAS through its `scipy_` symbols. The BLAS's licence is copied into the package.

```sh
cd native
uv build --wheel -Ccmake.define.SOLPROP_NATIVE_PROFILE=DISTRIBUTION \
  -Cbuild-dir="build/distribution-{wheel_tag}"
```

On the measured capillary and general-route cases a Linux distribution build ran within
about 5 % of a local build. The **reference** profile, `SOLPROP_NATIVE_PROFILE=REFERENCE`,
links no vendor BLAS and generates no machine-specific code. It is the gate's
sanitizer and clang-tidy build, slower than the other two, and holds no privileged
numerical answers.

The three profiles agree with solprop's reference backend within the same derived bounds;
they differ from each other in the last bits, because a vendor BLAS sums in its own
order. Keep each profile in its own build directory; switching the profile in one tree
rebuilds everything. Configuring prints one summary line, for example

```text
-- solprop-native build profile: local; BLAS Accelerate (automatic: Accelerate (macOS)); GEMM provider HYBRID; native architecture ON; CPU baseline none; IPO ON (Release builds)
```

and `build_info()` reports the same choice in `build_profile`, `cpu_baseline`,
`blas_backend`, `blas_selection`, `blas_library`, `native_arch_enabled` and
`ipo_enabled`.

## Passing CMake options

Pass options through the `CMAKE_ARGS` environment variable or scikit-build-core's `-C`
settings; both work with `pip`, `uv pip` and `uv build`:

```sh
CMAKE_ARGS="-DSOLPROP_NATIVE_BLAS_VENDOR=OPENBLAS" uv pip install --python /path/to/env/bin/python ./native
uv pip install --python /path/to/env/bin/python ./native -Ccmake.define.SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF
```

In the repository's own environment, pass them to `uv sync` and force the rebuild:

```sh
CMAKE_ARGS="-DSOLPROP_NATIVE_BLAS_VENDOR=OPENBLAS" \
  uv sync --extra native --reinstall-package solprop-native
```

Select the compiler with `CXX` or `-DCMAKE_CXX_COMPILER=...`. CMake caches compilers and
library paths, so use a fresh build directory (`-Cbuild-dir=...`) whenever you change
the compiler, the profile or a library provider. The default build directory is
`native/build/{wheel_tag}`, and the build type is `Release`.

## Option reference

The options are read by `native/CMakeLists.txt` and `native/cmake/HostLibraries.cmake`.

**Profile and optimisation**

| option | default | values and effect |
|---|---|---|
| `SOLPROP_NATIVE_PROFILE` | `LOCAL` | `LOCAL`, `DISTRIBUTION` (CPU builds only) or `REFERENCE`. The distribution and reference profiles choose their own BLAS and cannot be combined with `SOLPROP_NATIVE_HOST_ARCH=ON`, an explicit `SOLPROP_NATIVE_BLAS_VENDOR` or `SOLPROP_NATIVE_BLAS_LIBRARIES`; distribution cannot take `SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF`, reference cannot take `ON` |
| `SOLPROP_NATIVE_HOST_ARCH` | `AUTO` | `AUTO`, `ON` or `OFF`: code for the build machine's architecture. `AUTO` is `ON` for a local CPU build and `OFF` otherwise |
| `SOLPROP_NATIVE_CPU_BASELINE` | `AUTO` | the oldest processor a non-native CPU build runs on: `AUTO` (`x86-64-v3`, `apple-m1` or `neoverse-n1` in a distribution build, none otherwise), `NONE`, or an `-march` (x86-64) or `-mcpu` (Arm) value such as `x86-64-v2`. Not with the host architecture, a reference or a GPU build |
| `SOLPROP_NATIVE_ENABLE_IPO` | `ON` | checked IPO/LTO for CPU Release builds; never for CUDA or HIP |
| `SOLPROP_NATIVE_ENABLE_AGGRESSIVE_VECTORIZATION` | `ON` | sets `Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION` |
| `SOLPROP_NATIVE_ENABLE_COVERAGE` | `OFF` | a Clang CPU build with source coverage, for development only; disables IPO, and its timings are not performance evidence |
| `SOLPROP_NATIVE_VECTOR_MATH` | `AUTO` | the CPU kernels' transcendental functions ([vector maths](#vector-maths)): `AUTO` (SLEEF on x86-64 CPU builds, the system libm on Arm and in GPU builds), `SLEEF` (also on aarch64, with its AdvSIMD functions) or `SCALAR` (the system libm). Not `SLEEF` in a GPU build |
| `SOLPROP_NATIVE_VECTOR_MATH_AVX512` | `OFF` | `ON` takes SLEEF's AVX-512F functions where the target has AVX-512F, instead of its AVX2 ones: faster on AMD Zen 5, slower on Intel's Ice Lake generation ([vector maths](#vector-maths)) |

**Execution space**

| option | default | values and effect |
|---|---|---|
| `SOLPROP_NATIVE_ENABLE_OPENMP` | `ON` | the OpenMP host execution space and threaded FFTW. `OFF` in GPU builds, whose host space is Serial |
| `SOLPROP_NATIVE_ENABLE_THREADS` | `OFF` | the Kokkos Threads (C++ threads) host space with FFTW's threads library, for a build that adds no OpenMP runtime (a fallback for Windows builds); requires `SOLPROP_NATIVE_ENABLE_OPENMP=OFF`. Kernels then run at the runtime's thread count, without per-operation sizing |
| `SOLPROP_NATIVE_ENABLE_CUDA` | `OFF` | the CUDA execution space with cuFFT (Linux, GNU host compiler) |
| `SOLPROP_NATIVE_ENABLE_HIP` | `OFF` | the HIP execution space with rocFFT (Linux, ROCm Clang); exclusive with CUDA |
| `SOLPROP_NATIVE_CUDA_ARCH` | `HOPPER90` | the Kokkos architecture suffix, `Kokkos_ARCH_<value>` ([CUDA](#cuda)) |
| `SOLPROP_NATIVE_HIP_ARCH` | `AMD_GFX942` | the Kokkos architecture suffix, `Kokkos_ARCH_<value>` ([HIP](#hip)) |
| `SOLPROP_NATIVE_DEVICE_TEAM_SIZE` | `256` | the maximum GPU response and scan team width: 64, 128, 256, 512 or 1024 (the `cuda` presets use 512) |
| `SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE` | `512` | the GPU deterministic-reduction block width: 64, 128, 256, 512 or 1024 |
| `SOLPROP_NATIVE_BAND_STAGING` | `ON` for HIP, else `OFF` | how the field plasma column's band rule reads its halo: `ON` stages each band filter's input row in team scratch, a tile of 1,024 samples at a time, instead of reading it directly from the row. The arithmetic is the same; each default is the faster on its vendor's devices |
| `SOLPROP_NATIVE_DEVICE_GRAPH` | `ON` | on a CUDA or HIP build, replay an Adams trial's source evaluation as a device graph while its launches repeat from trial to trial; `OFF` launches them one by one on the stream. Results are the same either way; host builds ignore it |

**BLAS and dense transforms**

| option | default | values and effect |
|---|---|---|
| `SOLPROP_NATIVE_ENABLE_TPL_BLAS` | `AUTO` | `AUTO` links the profile's BLAS (a found vendor BLAS in a local build, none in a reference build); `ON` fails if none is found; `OFF` keeps the native-order GEMM |
| `SOLPROP_NATIVE_DISTRIBUTION_BLAS` | `AUTO` | the distribution build's BLAS on Linux and Windows: `AUTO` (`OPENMP_OPENBLAS` on Linux with OpenMP, `SCIPY_OPENBLAS` otherwise), `OPENMP_OPENBLAS` (with OpenMP; on Windows from a prebuilt static library) or `SCIPY_OPENBLAS` |
| `SOLPROP_NATIVE_OPENBLAS_ROOT` | the environment variable of that name | the prefix of a prebuilt OpenMP OpenBLAS for the distribution build: on Linux `lib/libsolprop_openblas.so` and `LICENSE` (empty builds it in the build tree), on Windows the static `lib/openblas.lib` and `LICENSE` (required) |
| `SOLPROP_NATIVE_BLAS_VENDOR` | `AUTO` | `AUTO`, `ACCELERATE` (macOS), `MKL` (Linux) or `OPENBLAS`. An explicit vendor that is not found fails. Host builds only |
| `SOLPROP_NATIVE_BLAS_ROOT` | empty | a prefix searched first for the BLAS |
| `SOLPROP_NATIVE_BLAS_LIBRARIES` | empty | an explicit `;`-separated library list (not together with `SOLPROP_NATIVE_BLAS_VENDOR`) |
| `SOLPROP_NATIVE_MKL_ROOT` | `$MKLROOT` | where the LP64 MKL libraries are looked for |
| `SOLPROP_NATIVE_GEMM_PROVIDER` | `HYBRID` | `HYBRID` (vendor GEMM for eligible transforms, custom kernels for the rest), `KOKKOSBLAS_TPL` (requires a BLAS), `KOKKOSBLAS_NATIVE` or `CUSTOM`; the alternatives are for comparison and validation |

**FFT**

| option | default | values and effect |
|---|---|---|
| `SOLPROP_NATIVE_FFT_PROVIDER` | `FFTW` | `FFTW`, or `MKL_FFTW` (MKL's FFTW3 interface, Linux). Host builds only |
| `SOLPROP_NATIVE_FFTW_THREADING` | `OPENMP` | `OPENMP` (`libfftw3_omp`) or `PTHREADS` (`libfftw3_threads`); IntelLLVM with FFTW requires `PTHREADS` |
| `SOLPROP_NATIVE_RAMAN_FFT_PROVIDER` | `ACCELERATE` on macOS CPU builds, else `KOKKOSFFT` | construction-time Raman selection: with NEON FFTW, vDSP handles short parallel power-of-two rows (up to 16,384 samples); serial and longer rows use Kokkos-FFT. Without NEON, only short serial rows use Kokkos-FFT. `KOKKOSFFT` forces that provider; `ACCELERATE` is macOS CPU only |
| `FFTW_ROOT` | the Homebrew prefix on macOS | a non-default FFTW installation |

**Development and diagnostics**

| option | default | values and effect |
|---|---|---|
| `BUILD_TESTING` | `OFF` in pip and uv builds, `ON` in the presets | builds the GoogleTest native tests and registers them with CTest |
| `SOLPROP_NATIVE_WARNINGS_AS_ERRORS` | `ON` | `-Werror` on the package's own sources |
| `SOLPROP_NATIVE_ENABLE_CLANG_TIDY` | `OFF` | runs clang-tidy while compiling the package's sources |
| `SOLPROP_NATIVE_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer and UBSan on the package's targets; CPU builds only |
| `SOLPROP_NATIVE_BUILD_BENCHMARKS` | `OFF` | builds the `solprop_native_gemm_benchmark` microbenchmark (never installed) |
| `SOLPROP_NATIVE_BUNDLED_LICENCES` | the environment variable of that name | a directory of `LICENSE-*.txt` files, installed beside the extension: the licences of the libraries a wheel repair copies in (FFTW, the OpenMP runtime) |

The installed module of a local build keeps the directories of a BLAS it linked outside
the system library path in its runpath, so an MKL build loads without sourcing oneAPI at
run time; a Linux distribution build's runpath is the package directory, where its
OpenBLAS is.

### Presets

`native/CMakePresets.json` defines development builds, all with `BUILD_TESTING=ON`:
`cpu-debug`, `cpu-tidy` and `cpu-sanitize` (reference profile, Debug), `cpu-benchmark`
(local, Release, IPO and benchmarks), the CUDA builds `cuda` (hybrid GEMM with cuBLAS),
`cuda-tpl`, `cuda-native` and `cuda-custom` (the other GEMM providers, for comparison),
and `hip`. The presets do not choose a compiler: pass `-DCMAKE_CXX_COMPILER=...` (on
macOS, Homebrew's `clang++`) and `-DPython_EXECUTABLE=...`.

### Offline builds

The pinned dependencies are fetched with CMake's `FetchContent`. To build without network
access, point it at source trees of the same versions:

```sh
-DFETCHCONTENT_SOURCE_DIR_KOKKOS=/path/to/kokkos-5.2.1
-DFETCHCONTENT_SOURCE_DIR_KOKKOSKERNELS=/path/to/kokkos-kernels-5.2.0
-DFETCHCONTENT_SOURCE_DIR_KOKKOSFFT=/path/to/kokkos-fft-2.0.0
-DFETCHCONTENT_SOURCE_DIR_SLEEF=/path/to/sleef-3.9.0           # only with SLEEF vector maths
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest-1.17.0   # only with BUILD_TESTING
```

### Vector maths

The CPU kernels form their transcendentals (the stage and step propagators, the
exponential Adams method's flows, phi-functions and compensation, the ionisation rates
and fractions) in passes over blocks of elements through one layer,
`native/include/solprop_native/vmath.hpp` ([ADR-059](../../dev/decisions.md)). On x86-64
it calls [SLEEF](https://sleef.org)'s vector functions, each within 1.0 ulp of the
correctly rounded value over its whole domain: AVX2 where the build's target has it (the
wheels' x86-64-v3 baseline and any local build on a processor from 2013 or later), SSE2
otherwise. `SOLPROP_NATIVE_VECTOR_MATH_AVX512=ON` takes the AVX-512F functions on a
target with AVX-512F: on one thread they ran the tracking cases about 4 % faster than
the AVX2 ones on an AMD EPYC 9B45 (Zen 5) but 7 % slower on an Intel Xeon of the Ice
Lake generation, whose cores lower their clock under 512-bit work. glibc's scalar `exp`, `cos` and `sin` cost a complex
exponential 10 to 30 ns on an AMD EPYC 7763; SLEEF's AVX2 functions take it 3.3 to 3.6
times faster at the same worst error on the kernels' own arguments. On Apple silicon
the system libm is already about 3 ns, SLEEF's AdvSIMD functions gain nothing, and the
build keeps the libm and each element's own calls (`SOLPROP_NATIVE_VECTOR_MATH=SLEEF`
selects SLEEF there too). `build_info().vector_math` records the route, for example
`"SLEEF 3.9.0 u10, AVX2"` or `"scalar libm"`. Results differ between the routes only
by rounding (the functions' own ulp), within the backend's derived rounding budgets
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).

SLEEF is configured and built at build time as its own CMake project, with its own
`-ffp-contract=off` (its double-double steps need every product and sum rounded) and
the C driver beside the build's C++ compiler, which SLEEF requires to be the same
version.

## Linux compilers and libraries

Any of these compilers builds the CPU backend; Kokkos, Kokkos Kernels and Kokkos-FFT stay
pinned whichever is used.

| compiler | notes |
|---|---|
| GCC 14 or newer | the reference route, with GNU OpenMP (`libgomp`). GCC 13 is refused: it compiles the strict floating-point helpers through the stack, and coupled Multistep steps took 1.25 to 1.45 times as long as with GCC 14 |
| IntelLLVM (`icpx`) 2022 or newer | compiled with `-fp-model=precise` (its default is fast floating point) and then `-ffp-contract=fast-honor-pragmas`; requires `SOLPROP_NATIVE_FFTW_THREADING=PTHREADS` with FFTW, because distribution OpenMP FFTW libraries link GNU OpenMP; uses the compiler's own archiver for LTO |
| Clang 14 or newer, AMD AOCC | compiled with `-fno-fast-math`, then `-ffp-contract=fast-honor-pragmas` |

| provider | how to select it |
|---|---|
| OpenBLAS | the automatic choice when MKL is not configured; `-DSOLPROP_NATIVE_BLAS_VENDOR=OPENBLAS` forces it, and `-DBLAS_openblas_LIBRARY=/path/to/libopenblas.so` names a specific library |
| MKL BLAS | `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), `-DSOLPROP_NATIVE_MKL_ROOT=...` or `-DSOLPROP_NATIVE_BLAS_VENDOR=MKL`. LP64 interface; GCC links MKL's GNU threading layer, IntelLLVM its Intel layer |
| AOCL-BLIS and other BLAS | `-DSOLPROP_NATIVE_BLAS_LIBRARIES="/path/libblis-mt.so;/path/libflame.so"` (BLIS alone lacks complex rotations that Kokkos Kernels' BLAS layer links; AOCL's LP64 libFLAME supplies them and needs `libaoclutils` on the loader path) |
| FFTW | the default; `-DFFTW_ROOT=...` for a non-system installation, such as AOCL's FFTW |
| MKL FFT | `-DSOLPROP_NATIVE_FFT_PROVIDER=MKL_FFTW`: MKL's FFTW3 interface, independent of the BLAS choice. MKL ignores FFTW's planning flags. Sourcing oneAPI never selects it by itself |

**FFTW threading.** `OPENMP` links FFTW's OpenMP library, which must use the same OpenMP
runtime as the rest of the build: GNU `libgomp` for distribution packages built with
GCC. `PTHREADS` links FFTW's pthreads library instead; it is required for IntelLLVM and
is the safer choice whenever a second OpenMP runtime could otherwise be loaded. With
pthreads FFTW, start with `OMP_PROC_BIND=false`: a bound OpenMP main thread can confine
the FFTW worker threads it creates to its own core.

**OpenBLAS threading.** A pthreads OpenBLAS (the Debian and Ubuntu default) keeps its own
thread pool beside the backend's OpenMP pool and, at the full core count, oversubscribes
the cores: the modal and wall cases ran 2 to 75 times slower than with one BLAS thread.
At start-up the package therefore holds a pthreads OpenBLAS to one thread and caps an
OpenMP OpenBLAS, which shares the runtime, at the backend's thread count, unless
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` or `OMP_NUM_THREADS` is set;
`build_info().blas_thread_policy` records which. An OpenMP OpenBLAS on the same runtime
(the Linux distribution build, or Ubuntu's `libopenblas0-openmp`) is the one to prefer
for a local Linux build. `build_info()` reports the loaded library's `blas_threading`
(`pthreads`, `OpenMP` or `serial`) and `blas_threads`.

A compiler and library combination that configures is not thereby validated: run the
native and Python tests after changing either (`native/README.md`).

## GPU builds

There are no GPU wheels: an NVIDIA or AMD GPU needs a build from source, on Linux. A GPU
build replaces the OpenMP host space with Serial, FFTW with the device FFT and the host
BLAS with the device BLAS. `SOLPROP_NATIVE_ENABLE_OPENMP=OFF` is required, the host
BLAS and FFT options do not apply, and the host architecture and IPO are off. The local
profile links the device BLAS; the reference profile does not; the distribution profile
is CPU-only.

A GPU build keeps no strict floating-point blocks: the few protected helpers that a CPU
build compiles without contraction (stable increments, small-argument functions,
defects and compensated sums) follow the device compiler's own default there, and nvcc
and HIP's Clang both fuse multiply-adds. `build_info().optimization_policy` then says
"compiler-default contraction". A GPU build's answers are held to the device envelopes
of its tests (`pytest -m gpu`) against the reference backend
(ADR-049 in the [decision records](../../dev/decisions.md)).
Round-to-nearest device arithmetic in those helpers is planned.

### CUDA

Requirements: CUDA 12.8 or newer with cuFFT and cuBLAS, and GCC 14 or newer as the C++
compiler. GCC 14 is the floor of every Linux build; CUDA 12.8 is the first toolkit that
accepts it as the host compiler (12.6 stops at GCC 13.2), and CMake refuses either below
its floor. The device build is qualified on CUDA 13 with GCC 14; CUDA 13 needs an NVIDIA
driver of the 580 series or newer and no longer targets Maxwell, Pascal or Volta, so a
V100 builds with CUDA 12.8 or 12.9. Kokkos compiles the device code through its `nvcc_wrapper`, which uses
`NVCC_WRAPPER_DEFAULT_COMPILER` as the host compiler. CMake finds `nvcc` from
`CUDA_ROOT` or `CUDAToolkit_ROOT` when given, otherwise on `PATH` or under
`/usr/local/cuda` or `/opt/cuda`.

`SOLPROP_NATIVE_CUDA_ARCH` names the GPU generation as a Kokkos architecture suffix:
`VOLTA70`, `TURING75`, `AMPERE80`, `AMPERE86`, `ADA89`, `HOPPER90` (the default),
`BLACKWELL100` or `BLACKWELL120`, among others in the Kokkos documentation.

```sh
git clone https://github.com/LupoLab/solprop
cd solprop
export CUDA_ROOT=/usr/local/cuda
export CXX=g++-14 NVCC_WRAPPER_DEFAULT_COMPILER=g++-14
export CMAKE_ARGS="-DSOLPROP_NATIVE_ENABLE_CUDA=ON -DSOLPROP_NATIVE_ENABLE_OPENMP=OFF \
  -DSOLPROP_NATIVE_CUDA_ARCH=AMPERE80"
uv pip install --python /path/to/env/bin/python . ./native
```

The `cuda` preset configures the same build for development, with the native tests and
the GEMM benchmark (`cmake --preset cuda -DPython_EXECUTABLE=... -DCUDA_ROOT=...`, from
`native/`).

### HIP

Requirements: ROCm with its Clang (`hipcc` or `amdclang++`), and the HIP runtime, rocFFT
and rocBLAS development packages (headers and CMake configuration files; the package
names vary between ROCm releases and distributions, for example `hip-dev`, `rocfft-dev`
and `rocblas-dev`). Point CMake at ROCm with `CMAKE_PREFIX_PATH`.

`SOLPROP_NATIVE_HIP_ARCH` names the GPU as a Kokkos architecture suffix, for example
`AMD_GFX90A`, `AMD_GFX942` (the default), `AMD_GFX950`, `AMD_GFX1100` or `AMD_GFX1201`.

```sh
git clone https://github.com/LupoLab/solprop
cd solprop
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc -DCMAKE_PREFIX_PATH=$ROCM_PATH \
  -DSOLPROP_NATIVE_ENABLE_HIP=ON -DSOLPROP_NATIVE_ENABLE_OPENMP=OFF \
  -DSOLPROP_NATIVE_HIP_ARCH=AMD_GFX942"
uv pip install --python /path/to/env/bin/python . ./native
```

If the ROCm installation lacks `clang-scan-deps`, add `-DCMAKE_CXX_SCAN_FOR_MODULES=OFF`
(the package uses no C++ modules). The `hip` preset configures the same build for
development.

### After a GPU build

`check_build()` refuses a GPU build that sees no device, and `build_info()` reports
`device_backend`, `device_arch`, `device_count` and the loaded cuFFT/rocFFT and
cuBLAS/rocBLAS libraries. Running on the device — one device per process, which problems
belong on the CPU, reproducibility — is in
[the native backend](native.md#running-on-a-gpu). Use the vendors' tools (Compute
Sanitizer, `rocprof`) to check device memory; the CMake sanitizer option is for CPU
builds. GPU builds are qualified with CUDA 13 and a ROCm 7 stack; other toolkit versions
are expected to work but are untested.

## OpenMP runtimes and pyFFTW

A process can safely load only one OpenMP runtime. Two different runtimes (LLVM's
`libomp` and GNU's `libgomp`, or two copies of one) each start their own thread pools,
oversubscribe the cores, and can crash when the second initialises. A CPU build
therefore links every native part — Kokkos, FFTW's threaded library, an OpenMP BLAS —
against one runtime: on macOS, either Homebrew's `libomp` with its FFTW, or the one
runtime in the owned dependency prefix built above.

solprop's `numba_cpu` backend uses [pyFFTW](https://github.com/pyFFTW/pyFFTW), whose
published wheels bundle their own OpenMP runtimes, so running both backends in one
process loads a second runtime.

- **On macOS, `solprop_native` must be imported before anything imports pyFFTW.** The
  package raises `RuntimeError` at import if pyFFTW is already loaded. solprop takes
  care of this itself: when the native backend is installed, `numba_cpu` imports it
  before pyFFTW, so both backends can run in one process in either order. A script
  that imports pyFFTW itself, or a package that does, imports solprop (or
  `solprop_native`) first.
- Import order alone does not remove pyFFTW's bundled runtimes. When one process must run
  both the native and the Numba backends, build pyFFTW from source against the same
  FFTW, in the same environment.
- pyFFTW then shares FFTW's process-wide planner with Kokkos-FFT. The package finalises
  the Kokkos runtime it owns at the very end of interpreter shutdown, after Python has
  released pyFFTW's plans.

macOS, against Homebrew's FFTW and `libomp`:

```sh
CC=/opt/homebrew/opt/llvm/bin/clang \
CXX=/opt/homebrew/opt/llvm/bin/clang++ \
PYFFTW_FFTW_PREFIX=/opt/homebrew/opt/fftw \
LDFLAGS="-L/opt/homebrew/opt/libomp/lib -Wl,-rpath,/opt/homebrew/opt/libomp/lib" \
  uv pip install --python /path/to/env/bin/python \
  --no-deps --no-binary pyfftw --reinstall pyfftw
```

Linux, against the system FFTW with pthreads, so that pyFFTW adds no OpenMP runtime:

```sh
CC=gcc CXX=g++ PYFFTW_USE_PTHREADS=1 PYFFTW_FFTW_PREFIX=/usr \
  uv pip install --python /path/to/env/bin/python \
  --no-deps --no-binary pyfftw --reinstall pyfftw
```

For an FFTW installed elsewhere, set `PYFFTW_FFTW_PREFIX` to its prefix; pyFFTW needs the
single-precision FFTW libraries as well as the double-precision ones. On macOS also
change both library paths in `LDFLAGS` to that prefix's OpenMP library directory. With AOCL's FFTW,
build pyFFTW with AOCC and without `PYFFTW_USE_PTHREADS`, so that it uses AOCL's OpenMP
interface; do not combine a standard FFTW pthreads library with AOCL's FFTW core.

## Wheels and redistribution

A wheel meant for other machines is built with `SOLPROP_NATIVE_PROFILE=DISTRIBUTION`.
It carries its BLAS (on Linux and Windows the OpenMP OpenBLAS, BSD-3-Clause) with its
licence, and the licence texts of Kokkos, Kokkos Kernels, Kokkos-FFT, nanobind,
robin_map and, on x86-64, SLEEF (BSL-1.0), which are compiled in. FFTW and the OpenMP
runtime are bundled by the wheel
repair (`auditwheel repair` on Linux, `delocate` on macOS, `delvewheel` on Windows), and
their licence texts by `SOLPROP_NATIVE_BUNDLED_LICENCES`; FFTW (GPL-2.0-or-later) is
compatible with the package's GPL-3.0-or-later. The published wheels' recipe is in the
developer notes ([Releasing and the native wheels](../../dev/releasing.md)). CUDA and
HIP builds link the vendors' libraries and are built from source only.

The build profiles, CPU baselines and wheel recipes on this page were proposed and
qualified by [SEP 29](../../dev/seps/archive/sep-0029-compiled-backend-distribution.md).
