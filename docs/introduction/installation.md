# Get and install solprop

You need Python 3.14 and an environment in which you can install Python packages.
Python 3.15 is not yet supported: solprop-native, pyFFTW and h5py publish no wheels
for it yet, so an installation there falls back to building them from source, which
usually fails.

On the four platforms with prebuilt wheels, install solprop together with its
compiled native backend, the fastest way to run it; elsewhere, solprop's included CPU
backends run every model, and the native backend can be built from source and added
later without changing the physical model. The installation commands
bring in the dependencies: NumPy, SciPy, Matplotlib, h5py, Numba, pyFFTW (FFTW
transforms for the Numba backend), [optmat](https://github.com/LupoLab/optmat) with its
`gas` extra (the
curated material library every built-in gas, solid, Raman response and ionisation
rate comes from, with its equations of state through CoolProp), threadpoolctl (BLAS
thread control), jaxtyping (array annotations) and tqdm (progress bars). solprop
keeps no material data of its own: see [materials](../guide/media/materials.md) for
how model data are selected.

## Installing solprop

Install solprop from the Python Package Index, with its native backend. With
[uv](https://docs.astral.sh/uv/) (recommended):

```sh
uv venv --python 3.14
uv pip install "solprop[native]"
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

or with pip into an existing Python 3.14 environment:

```sh
python -m pip install "solprop[native]"
```

The `native` extra adds `solprop-native`, the compiled native backend, at solprop's own
version. It is a precompiled wheel on four platforms:

| platform | system |
|---|---|
| osx-arm64 | macOS on Apple silicon |
| linux-64 | Linux on x86-64 |
| linux-aarch64 | Linux on aarch64 |
| win-64 | Windows on x86-64 |

With it installed, `backend="auto"` uses the native backend
([The native backend](../guide/performance/native.md) describes what it runs).

**Processor requirements.** The wheels are compiled for a minimum processor:

| platform | minimum processor |
|---|---|
| osx-arm64 | any Apple silicon Mac (M1 or newer) |
| linux-64, win-64 | x86-64-v3: AVX2, FMA and BMI2, that is Intel Haswell (2013) or AMD Excavator (2015) and newer |
| linux-aarch64 | Arm Neoverse N1 (Armv8.2-A with dot product and half precision): AWS Graviton2, Ampere Altra and newer |

Low-power Intel Atom, Celeron and Pentium chips without AVX are below x86-64-v3 even
when recent, as are older cluster nodes. On such a processor the native package refuses
to load, naming the features it lacks, before any of its compiled code runs.
`backend="auto"` then uses the `numba_cpu` backend instead and warns once, saying so;
every model still runs, at the included backend's speed. An explicit
`backend="native"` raises the refusal. To use the native backend on such a machine,
build it from source for it ([Building the native backend](../guide/performance/building-native.md)),
and please [open an issue](https://github.com/LupoLab/solprop/issues) so that the
maintainers know such processors are in use.

**Other platforms.** Install solprop alone, `uv pip install solprop` or
`python -m pip install solprop`. This gives two backends, the NumPy reference backend and
the compiled Numba CPU backend (see [Choosing a backend](../guide/performance/backends.md)),
which run every model; consult [backend support](../guide/performance/backends.md) when
choosing a representation or response, since some combinations require a particular
backend. The native backend can then be built from source.

**GPUs and source builds.** For an NVIDIA GPU (CUDA) or an AMD GPU (HIP), and on any
platform without a wheel, build the native backend from source:
[Building the native backend](../guide/performance/building-native.md) gives the
requirements and the commands. A source build can also be compiled for the machine that
runs it, which is a few per cent faster than the wheel.

For the development version, install from the repository instead
(`python -m pip install "git+https://github.com/LupoLab/solprop.git"`); to work on
solprop itself, use the [contributor setup](../dev/contributing.md).

## Checking the installation

Use Python in the activated environment, or a notebook kernel using that same
interpreter. Run the following code; it prints the package versions and the
backends available to this interpreter.

```python
import optmat

import solprop
from solprop.backends.registry import available, resolve_name

print("solprop", solprop.__version__, "with optmat", optmat.__version__)
print("backends:", available(), "- auto means", resolve_name("auto"))
```

`available()` lists the backends that are installed and `resolve_name("auto")` names the
one a run uses by default: `native` when the native backend is installed. The first run
of the `numba_cpu` backend compiles its
kernels. This can make the first calculation slower than subsequent runs; many
compiled kernels are cached on disk. Measure performance after that initial setup.

## Make a first short run

This small linear-envelope calculation checks that propagation and output work.
The nonlinear coefficient is zero and the dispersion coefficients are zero, so
the pulse should keep its energy. The propagation is only 1 mm; it is an
installation check, not a demonstration of nonlinear dynamics.

```python
import numpy as np

from solprop.driver import prop_gnlse
from solprop.processing import energy

run = prop_gnlse(
    0.0,
    1e-3,
    [0.0, 0.0, 0.0],
    wavelength=800e-9,
    fwhm=50e-15,
    energy=1e-9,
    trange=400e-15,
    wavelength_limits=(600e-9, 1.2e-6),
    shock=False,
    noise=None,
    n_save=3,
    progress="none",
)
energies = energy(run, modes="sum")
print("saved positions (m):", run["z"])
print("input and exit energy (nJ):", energies[[0, -1]] * 1e9)
assert np.isclose(energies[-1], energies[0], rtol=1e-8, atol=0.0)
```

An approximately 1 nJ input and exit confirm this check. Continue with
[what solprop can do](capabilities.md) and
[choosing a spatial description](spatial-description.md), then run the
[dispersive-wave lesson](lessons/plot_01_rdw.rst) for a complete physical example.
For package import or backend errors, check that the interpreter is the one in
which you installed solprop and inspect [backend selection](../guide/performance/backends.md).

## Platforms

solprop is developed and tested on macOS (Apple silicon), Linux (x86-64) and Windows
(x86-64). Single runs and scans work on all three with the included backends. The native
backend has wheels for the four platforms above, builds from source on Linux and macOS,
and builds for GPUs on Linux. The `Queue` and `Slurm` scan executors need a POSIX system and refuse to start
on Windows; `Local` and `Pool` run there. On Windows a scan's readers take its lock one
at a time, where POSIX lets them read together, so concurrent reads of one scan file
wait for each other.

The precompiled native wheels and the `solprop[native]` extra were proposed and qualified by
[SEP 29](../dev/seps/archive/sep-0029-compiled-backend-distribution.md).
