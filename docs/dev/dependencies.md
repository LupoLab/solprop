# Dependencies

Adding a dependency is a decision, not an implementation detail: propose it before adding
it (`AGENTS.md`). This page lists what is approved, why, and the candidates that could be
evaluated. Versions are lower bounds except where pinned; `uv.lock` is the resolution.

**How a floor is chosen.** A lower bound is a claim that solprop works with that
release, so it is the oldest release that (a) publishes CPython 3.14 wheels for Linux,
macOS and Windows and (b) provides every API solprop calls, unless a stricter reason is
recorded in the table. It is not the version that happened to be current. The `oldest`
CI job installs the package with `uv pip install --resolution lowest-direct` and runs
the default lane, so a floor that stops holding fails there. Raise a floor only for a
reason written here. The recorded states under `tests/` are compared at a tolerance, not
pinned to a platform or a library build ([testing.md](testing.md) section 5), so the floors
need no exception for them.

## Core (always installed)

| package | purpose | notes |
|---|---|---|
| numpy ≥ 2.5 | arrays | the array type at every layer; the APIs used are NumPy 2.0's (`concat`, `trapezoid`, `copy=`) and 2.3.2 is the first release with 3.14 wheels on all three platforms, but under 2.3 and 2.4 the free-space reference runs produce NaN and about 200 free-space and dispersive tests fail; the cause is not yet identified, so the floor stays at 2.5 until it is |
| scipy ≥ 1.18 | `scipy.fft` (reference backend), `special`, `optimize`, `interpolate`, `integrate.quad_vec`, `signal`, `constants` | root finding, special functions, interpolation and adaptive quadrature throughout; the newest API used is `signal.ShortTimeFFT` (1.12), but optmat requires 1.18, the release its stored fits reproduce from |
| matplotlib ≥ 3.10.7 | plotting | earlier 3.10 releases call pyparsing APIs that current pyparsing deprecates, which the test suite's warnings-as-errors refuses |
| h5py ≥ 3.15 | HDF5 output | first release with 3.14 wheels on all three platforms |
| numba ≥ 0.67 | CPU kernels (ADR-002, ADR-023) | the first release that supports NumPy 2.5 (0.63 is the first with Python 3.14, but caps NumPy below 2.4, and 0.64–0.66 below 2.5); the internals `solprop._numba_cache` uses were checked down to 0.63; `cache=True` except for parallel and dispatcher-capturing closures, and SEP 25's floating-point licence (contraction by default, reassociation per kernel, nothing else) ([backends.md §3.2](backends.md)) |
| pyfftw ≥ 0.15.1 | FFTW plans executed in place (ADR-023) | wisdom handled explicitly; 0.15.1 has the 3.14 wheels (none for Windows on Arm) |
| optmat[gas] ≥ 0.9.1 | built-in material models: gas dispersion, Kerr, equation of state and transport; solids; gas and silica Raman responses; gas ionisation rates ([SEP 3](seps/archive/sep-0003-optmat-materials.md)) | the same maintainers' curated-data package; never imported at `import solprop`, since its registries take about a second to import; 0.9 reads only the data a model needs and says which gases each rate model covers |
| threadpoolctl ≥ 3.1 | inspect and control the backend-owned BLAS pool | pool provenance and limits are recorded per run |
| jaxtyping ≥ 0.3.11 | array annotations, evaluated at import time in every module | pure Python; the checker (beartype) is dev-only; under earlier releases the test session's import hook wraps the compiled kernels, which numba then cannot compile or cache |
| tqdm ≥ 4.67 | the progress bar (`solprop.log.BarReporter`) | pure Python; imported when the first bar is drawn, never at `import solprop`; `ipywidgets` is optional for the notebook widget and is not a dependency |

## Optional atomic-response engine

Pyroraptor (`pyroraptor`) provides the helium atomic responses of the
[TDSE response](../guide/media/tdse.md): the single-active-electron model of
[SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md) and the haCC models of
[SEP 20](seps/archive/sep-0020-hacc-helium-tdse.md). It is not yet publicly available, so
solprop declares no dependency or extra for it, and imports it only when a TDSE response
is prepared; ordinary imports and non-TDSE runs never initialise its runtime. A missing
provider, an unsupported interpreter (the provider supports Python 3.14 only) or a
missing capability is refused with a `BackendError`.

solprop pins one provider source revision, `solprop.driver.tdse._PROVIDER_REVISION`
(currently `876b72d10b5286478125b842ea45548a5051b033`);
the binding checks the installed provider against it, and the source identity and the
loaded native build identity are recorded separately with every run. A device build
(CUDA or HIP) is selected explicitly through `TDSEExecution.device` and checked against
the build's execution space, with no fallback.

## The Apple-GPU backend (`mlx` extra)

MLX (`mlx`, MIT licence, compatible with GPL-3.0-or-later) is the optional extra of the
`mlx` backend ([SEP 33](seps/archive/sep-0033-single-precision-mlx.md)), declared with the marker
`sys_platform == "darwin" and platform_machine == "arm64"` and pinned to a tested minor
version, `mlx>=0.32.4,<0.33`; the macOS floor is MLX's wheels'. A pin is raised only in
a pull request that runs the GPU lane and the arithmetic probes (Q9). Nothing outside
`solprop.backends.mlx_gpu` imports it; availability is probed with
`importlib.util.find_spec`.

## The native backend package

`solprop-native` is the repository's second distribution, built from `native/` and
installed through solprop's `native` extra, which pins it to solprop's own version. Its
native build pins and fetches Kokkos 5.2.1, KokkosKernels 5.2.0 (BLAS component only) and
Kokkos-FFT 2.0.0 and uses nanobind. A local build (the default) links the system
Accelerate, oneAPI's MKL or OpenBLAS through KokkosKernels, whichever it finds, and
neither fetches nor bundles a BLAS library. A distribution build
(`SOLPROP_NATIVE_PROFILE=DISTRIBUTION`, ADR-034) links Accelerate on macOS and, on Linux
and in the Windows wheel, OpenBLAS 0.3.34 built with OpenMP from its pinned, checksummed
source release and carried in the wheel; a Windows build without that library links the
`scipy-openblas32` OpenBLAS from PyPI, a build requirement it copies into the wheel. The
release wheels also bundle FFTW 3.3.11 and their OpenMP runtime, built from pinned
sources by `native/ci/` ([Releasing and the native wheels](releasing.md)). A reference
build links no BLAS. Its CUDA build uses the CUDA toolkit's cuFFT and cuBLAS through Kokkos-FFT
and KokkosKernels, and its HIP build rocFFT and rocBLAS; it neither fetches nor bundles
device libraries. Both distributions resolve in the root workspace's one `uv.lock`;
solprop-native has no runtime dependency on solprop.

## Development

The benchmark controller uses **psutil ≥ 7.1.0** for the CPU allocation, core
count, memory size and the busy-machine reading. This floor includes the macOS C
safety fixes and Windows ARM64 wheels recorded in the
[upstream 7.1.0 changelog](https://github.com/giampaolo/psutil/blob/master/docs/changelog.rst).
Collection records the installed version. Unsupported or sandbox-denied counters
remain unavailable with a reason; `threadpoolctl` supplies library-pool introspection.
psutil is a development dependency and does not enter the simulation API.

basedpyright (pinned exactly), scipy-stubs, beartype, sympy (the symbolic Marcatili
derivation in `tests/data/`, run by hand), pytest, pytest-xdist, pytest-randomly,
pytest-regressions, pytest-cov, hypothesis, poethepoet, pre-commit,
codespell, reuse (the per-file copyright and licence check of `poe reuse`), numpydoc, sphinx, myst-parser, pydata-sphinx-theme, sphinx-gallery, ruff, and
drawsvg (the structure figures of `docs/dev/figures/build.py` and the geometry figures of
`docs/guide/figures/geometries.py` only, run by hand with `poe figures`; the SVGs it
writes are committed, so the documentation builds without it).

## Candidates (not approved; only evaluate for a selected use case)

| package | for | status |
|---|---|---|
| rocket-fft | transforms callable inside `@njit`, if a measured need for a compiled march emerges | not needed; the shared march is Python ([backends.md §3.5](backends.md)) |
| mkl_fft | Intel FFT on Linux nodes | no evaluation planned |
| jax | array-compiler backend | evaluated; not selected (ADR-002, [backends.md §5](backends.md)) |
| sphinx-design, sphinx-copybutton | docs presentation | no evaluation planned |

## Rejected

| package | why | alternative |
|---|---|---|
| refractiveindex | A 9 KB shim that downloads an unversioned database to `~/.refractiveindex.info-database` on first use (breaks the no-network rule, cannot be pinned, caches outside the project) and returns only `n`, so the analytic derivative of ADR-006 would be lost | For glasses, crystals and metals: read the refractiveindex.info database **offline**, pinned to a database commit, with a generator script that writes provenance-tracked resources under `src/solprop/physdata/data/` (the route by which coefficients, reference states and validity ranges are verified) |
