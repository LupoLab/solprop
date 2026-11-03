# Get and install solprop

You need Python 3.14 and an environment in which you can install Python packages.
Python 3.15 is not yet supported: h5py publishes no wheels for it yet, so an installation
there falls back to building it from source, which usually fails.

solprop runs where its compiled native backend installs. `pip install solprop` installs
the two together, with no extra, and the platforms and processors below are what is
supported. The installation commands also bring in the other dependencies: NumPy, SciPy,
Matplotlib, h5py, [optmat](https://github.com/LupoLab/optmat) with its `gas` extra (the
curated material library every built-in gas, solid, Raman response and ionisation
rate comes from, with its equations of state through CoolProp), threadpoolctl (BLAS
thread control), jaxtyping (array annotations) and tqdm (progress bars). solprop
keeps no material data of its own: see [materials](../guide/media/materials.md) for
how model data are selected.

## Installing solprop

Install solprop from the Python Package Index. With
[uv](https://docs.astral.sh/uv/) (recommended):

```sh
uv venv --python 3.14
uv pip install solprop
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

or with pip into an existing Python 3.14 environment:

```sh
python -m pip install solprop
```

Either brings in `solprop-native`, the compiled native backend, at solprop's own version,
as a precompiled wheel. No extra is needed. These five platforms are the ones solprop
supports:

| platform | system |
|---|---|
| osx-arm64 | macOS 14 or newer on Apple silicon |
| linux-64 | Linux on x86-64, glibc 2.28 or newer |
| linux-aarch64 | Linux on aarch64, glibc 2.28 or newer |
| win-64 | Windows on x86-64, Windows 10 version 2004 or newer |
| win-arm64 | Windows on Arm |

`backend="auto"` uses the native backend
([The native backend](../guide/performance/native.md) describes what it runs).

On Windows on Arm, leave `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `GOTO_NUM_THREADS`
unset: small values can crash or hang NumPy's own linear solves
([BLAS and vendor libraries](../guide/performance/tuning.md#blas-and-vendor-libraries)).

**Processor requirements.** The wheels are compiled for a minimum processor:

| platform | minimum processor |
|---|---|
| osx-arm64 | any Apple silicon Mac (M1 or newer) |
| linux-64, win-64 | x86-64-v2: SSE4.2 and POPCNT, that is Intel Nehalem (2008) or AMD Bulldozer (2011) and newer |
| linux-aarch64 | Arm Neoverse N1 (Armv8.2-A with dot product and half precision): AWS Graviton2, Ampere Altra and newer |
| win-arm64 | any processor that runs Windows 11 on Arm |

An x86-64 wheel holds two builds of the backend and loads the best one the processor
runs: x86-64-v3 (AVX2, FMA and BMI2; Intel Haswell, 2013, or AMD Excavator, 2015, and
newer), and x86-64-v2 for the rest, including low-power Intel Atom, Celeron and Pentium
chips without AVX even when recent, and older cluster nodes. The v2 build takes about 20
to 25 % longer by geometric mean, up to 50 % on one thread. On a processor below the
minimum the native package refuses to load, naming the features it lacks, before any of
its compiled code runs. `backend="auto"` and `backend="native"` both raise that refusal:
solprop does not switch to another backend. If nothing can establish the processor's
features, the package refuses in the same way rather than risk an illegal instruction; on
x86-64, `SOLPROP_NATIVE_CPU_VARIANT` then names a build on your word about the processor.
A source build compiled for such a machine may work, with no support promise
([Building the native backend](../guide/performance/building-native.md)). Please
[open an issue](https://github.com/LupoLab/solprop/issues) so that the maintainers know
such processors are in use; the refusal gives the same link.

**Any other platform.** Intel macOS, macOS before 14, ppc64le, musl Linux, an older glibc
and any other platform have no native wheel, and the native backend publishes no source
distribution, so the installer stops there. pip says:

```text
ERROR: Could not find a version that satisfies the requirement solprop-native==X.Y.Z (from solprop) (from versions: none)
ERROR: No matching distribution found for solprop-native==X.Y.Z
```

and uv that `solprop-native==X.Y.Z has no wheels with a matching platform tag`. This is
not a broken installation: it means that solprop does not support this platform, and none
of solprop's code has run. An installer that cannot satisfy the newest release may also
try older ones, and solprop 0.1.0 did not require the native backend, so if it starts
building Numba or pyFFTW, ask for a current release (`pip install "solprop>=0.2"`) to see
the error above.

To report the platform, [open an issue](https://github.com/LupoLab/solprop/issues) and
include what this prints:

```sh
python -c "import platform, sys; print(sys.version); print(platform.platform(), platform.machine(), platform.libc_ver())"
```

**GPUs and source builds.** For an NVIDIA GPU (CUDA) or an AMD GPU (HIP), build the native
backend from source: [Building the native backend](../guide/performance/building-native.md)
gives the requirements and the commands. A source build can also be compiled for the
machine that runs it, which is a few per cent faster than the wheel, and it is the way past
a wheel's limits: on a platform without a wheel it may work, with no support promise.

For the development version, build both packages from a clone of the repository
([a CPU build](../guide/performance/building-native.md#a-cpu-build)), since a development
solprop requires the native backend at its own version; to work on solprop itself, use the
[contributor setup](../dev/contributing.md).

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
one a run uses by default, `native`. If the native package is missing or the processor is
below its minimum, it raises instead, saying why and what to do.

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
(x86-64 and Arm). The native backend has wheels for the five platforms above, builds from
source on Linux and macOS, and builds for GPUs on Linux. The `Queue` and `Slurm` scan executors need a POSIX system and refuse to start
on Windows; `Local` and `Pool` run there. On Windows a scan's readers take its lock one
at a time, where POSIX lets them read together, so concurrent reads of one scan file
wait for each other.

The precompiled native wheels were proposed and qualified by
[SEP 29](../dev/seps/archive/sep-0029-compiled-backend-distribution.md), and
[SEP 43](../dev/seps/sep-0043-retire-numba-cpu.md) made the native backend a required
dependency.
