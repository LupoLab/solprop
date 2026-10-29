# Experimental CPU backend

`solprop-cpunative` is the CPU-only OpenMP/FFTW experiment in solprop. Select it
explicitly with `backend="cpunative"`. It does not depend on Kokkos or
`solprop-native`. Its supported subset and measurements are documented in the
solprop manual. It is not selected by `backend="auto"`.

Build with a C++23 compiler (GCC 14 or newer, or recent LLVM Clang), OpenMP and
double-precision FFTW including its OpenMP library. Set `FFTW_ROOT` through
`CMAKE_ARGS` when FFTW is not on the normal search path. Apple silicon builds
require FFTW compiled with its NEON codelets and a matching LLVM OpenMP runtime.

From the repository root: `uv sync --extra cpunative`. Both backends can be
installed with `uv sync --extra cpunative --extra native`.

Compiler experiments can set `SOLPROP_CPUNATIVE_MATH_PROFILE` through `CMAKE_ARGS`:
`PROTECTED` retains the qualified defaults, `ERRNO_FREE` adds `-fno-math-errno`,
`NONTRAPPING` also adds `-fno-trapping-math`, and `FORTRAN_COMPLEX` additionally
enables `-fcx-fortran-rules`. These are ablations, not an accuracy waiver. All
retain the protected contraction policy and omit umbrella fast-math. The build
checks flag support and reports the selected profile in `build_info()`.

For a compiler-only comparison, `SOLPROP_CPUNATIVE_FFTW_THREADING=PTHREADS` allows
all compilers to share one FFTW build without loading a second OpenMP runtime.
Plans still use one FFTW thread and independent rows execute through OpenMP.
Keep BLAS and its threading fixed too, then compare provider choices separately.
Archive `compile_commands.json` and `compile-options-Release.txt` from each CMake
build alongside the measured results; configure `CMAKE_EXPORT_COMPILE_COMMANDS=ON`.
