# solprop

[![CI](https://github.com/LupoLab/solprop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LupoLab/solprop/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/solprop/badge/?version=latest)](https://solprop.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/LupoLab/solprop/graph/badge.svg)](https://codecov.io/gh/LupoLab/solprop)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright](https://img.shields.io/badge/basedpyright-checked-42b983)](https://docs.basedpyright.com)
[![PyPI](https://img.shields.io/pypi/v/solprop.svg)](https://pypi.org/project/solprop/)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Licence: GPL-3.0-or-later](https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg)](https://github.com/LupoLab/solprop/blob/main/LICENSE)
[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](https://reuse.software)

**Ultrafast nonlinear pulse propagation in Python** — from few-cycle pulses in
gas-filled hollow-core fibres to multimode solid-core fibres and free-space beams.

**Everything, fast, with rigour.** solprop aims to be one unified, highly accurate code
for nonlinear pulse propagation, scaling from a laptop to a GPU cluster, with every model
derived, cited and verified against analytic and independent references
([About solprop](https://solprop.readthedocs.io/en/latest/introduction/about.html)).

It solves the unidirectional pulse propagation equation (UPPE) and the generalised
nonlinear Schrödinger equation (GNLSE), for the full electric field or its envelope, in
waveguides and in free space. Use it for soliton self-compression to sub-cycle pulses,
resonant dispersive-wave emission into the vacuum and deep ultraviolet, supercontinuum
generation, multimode and graded-index beam dynamics, filamentation, photoionisation and
plasma, and Raman frequency combs.

![Vacuum-ultraviolet dispersive-wave emission in a helium-filled capillary](https://raw.githubusercontent.com/LupoLab/solprop/main/docs/_static/readme-vuv-rdw.png)

*A 7.5 fs pulse at 800 nm self-compresses in a helium-filled capillary with a pressure
gradient to vacuum and emits a resonant dispersive wave at 120–150 nm: spectral and
temporal evolution along the 1.5 m fibre, and the exit spectrum. Four coupled modes,
ionisation and plasma included; about half a minute on a laptop.*

```python
from solprop.driver import prop_capillary
from solprop.plotting import prop_2d

run = prop_capillary(
    125e-6,  # core radius, m
    1.5,  # length, m
    "He",
    (0.8, 0.0),  # pressure gradient, bar: entrance to exit
    wavelength=800e-9,
    fwhm=7.5e-15,
    energy=275e-6,
    modes=4,  # the first four HE_1m modes
    trange=400e-15,
    wavelength_limits=(90e-9, 4e-6),
)
prop_2d(run, "wavelength")
```

## What it does

- **Waveguides**
  - Gas-filled hollow capillaries, one or many modes, with tapers, pressure gradients
    and gas flow; a coherent wall that couples the modes and converts between them.
  - Antiresonant hollow-core fibres (the Vincetti model); rectangular and planar
    guides; modes tabulated from any mode solver.
  - Step-index and graded-index fibres with full vector modes, through a general modal
    route for any guide that supplies its own electric and magnetic fields — including
    guides that change along their length.
- **Free space**: beams in bulk gases and solids on radial, Cartesian and line grids,
  with the exact (non-paraxial) diffraction operator, absorbing spatial boundaries, and
  the Kerr, plasma and Raman responses of the guided routes: self-focusing,
  filamentation and ionisation in cells, jets and windows, and planar guides resolved
  across their width.
- **Media and nonlinear responses**: noble and molecular gases and their mixtures,
  glasses and metals, all from the curated material library
  [optmat](https://github.com/LupoLab/optmat); Kerr and
  third-harmonic nonlinearity; photoionisation with optmat's atomic and molecular rate
  models, and plasma on field and envelope grids; rotational and vibrational
  Raman responses, scalar and vector; quantum shot noise; multi-colour inputs.
  An optional [helium TDSE atomic response](https://solprop.readthedocs.io/en/latest/guide/media/tdse.html), for research,
  replaces Kerr and plasma by a microscopic atomic current; its native backend is not
  yet publicly available.
- **Accurate numerics**: two adaptive integrators, a low-call exponential Adams method
  (the default of capillary, radial and planar runs) and an embedded Runge–Kutta pair
  (the default of GNLSE, waveguide and Cartesian runs;
  [the defaults](https://solprop.readthedocs.io/en/latest/guide/numerics/choosing-a-stepper.html#the-defaults)), with the linear propagation
  integrated exactly, absorbing boundaries as rates, and every backend tested against a
  reference implementation within derived error bounds.
- **Workflows**: HDF5 output with resume, statistics along the fibre, parameter scans on
  process pools and Slurm clusters, processing and plotting, soliton and dispersive-wave
  design tools, and pulse compressors (Taylor phase, chirped mirrors, materials,
  gratings and prisms). One import, `import solprop.sol as sol`, sets up an interactive
  session.

## Fast CPUs and GPUs: the native backend

solprop runs out of the box on a NumPy reference backend and a compiled Numba CPU
backend. For speed on large problems — multimode and general modal guides, broadband
supercontinuum grids, free-space beams — add the optional **native backend**,
solprop-native, built from [`native/`](https://github.com/LupoLab/solprop/tree/main/native)
in this repository: C++ kernels built with [Kokkos](https://kokkos.org), using OpenMP,
FFTW and a vendor BLAS on multi-core CPUs, or CUDA and HIP on NVIDIA and AMD GPUs. Once
it is installed, solprop uses it automatically; your scripts do not change. See
[The native backend](https://solprop.readthedocs.io/en/latest/guide/performance/native.html) and
[Performance tuning](https://solprop.readthedocs.io/en/latest/guide/performance/tuning.html).

## Installation

solprop needs Python 3.14 (3.15 is not yet supported: some dependencies have no wheels
for it) and runs on Linux, macOS and Windows:

```sh
pip install "solprop[native]"   # with the compiled native backend (recommended)
pip install solprop             # elsewhere: the NumPy and Numba backends only
```

The `native` extra installs a precompiled wheel of the native backend, the fastest way to
run solprop, on macOS on Apple silicon (osx-arm64), Linux on x86-64 (linux-64), Linux on
aarch64 (linux-aarch64) and Windows on x86-64 (win-64). On other platforms, and for GPUs,
build it from source (below).
[Installation](https://solprop.readthedocs.io/en/latest/introduction/installation.html) covers environment checks and a
first run.

## Building from source

Both packages build from one clone of the repository:

```sh
git clone https://github.com/LupoLab/solprop.git
cd solprop
pip install . ./native          # solprop and the native backend, for this machine's CPU
```

The native build needs a C++20 compiler, CMake, FFTW and, for speed, a BLAS. A GPU
build (NVIDIA with CUDA, AMD with HIP, on Linux) is always compiled from source, with the
device enabled through CMake options. [Building the native
backend](https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html) gives the requirements, the
commands for Linux, macOS, CUDA and HIP, and every build option. To work on solprop
itself, see [Contributing](https://solprop.readthedocs.io/en/latest/dev/contributing.html).

## Documentation

The manual is at [solprop.readthedocs.io](https://solprop.readthedocs.io/en/latest/):

- [Introduction](https://solprop.readthedocs.io/en/latest/introduction/index.html) — installation, capabilities, spatial
  descriptions, and first propagation calculations.
- [User guide](https://solprop.readthedocs.io/en/latest/guide/index.html) — simulation setup, waveguides and free space,
  materials and nonlinear responses, numerical methods, results, workflows, and
  performance, including the detailed equations and approximation limits.
- [Examples](https://solprop.readthedocs.io/en/latest/examples/index.html) — introductory lessons, focused demonstrations,
  and full worked simulations, organised by what they teach.
- [Developer’s guide](https://solprop.readthedocs.io/en/latest/dev/index.html) — contributing, architecture, implementation
  contracts, testing, tooling, proposals, and project history.
- [API](https://solprop.readthedocs.io/en/latest/api/index.html) — the complete reference for functions, classes, and modules.

Users of solprop's predecessor, [Luna.jl](https://github.com/LupoLab/Luna.jl), can
start with [noteworthy differences from Luna](https://solprop.readthedocs.io/en/latest/guide/workflows/luna.html).

## About

solprop continues more than two decades of pulse-propagation codes by John Travers and
colleagues, most recently [Luna.jl](https://github.com/LupoLab/Luna.jl); [About solprop](https://solprop.readthedocs.io/en/latest/introduction/about.html) tells the story. solprop
itself is developed almost entirely by AI coding agents under the lead developer's
direction, within a strict harness of derivations, reference comparisons, convergence
checks and complete test coverage; the same page explains how, and what that means for
users. The code is free software under the [GNU General Public License, version 3 or
(at your option) any later version](https://github.com/LupoLab/solprop/blob/main/LICENSE), and the documentation is under
[CC BY 4.0](https://github.com/LupoLab/solprop/blob/main/LICENSES/CC-BY-4.0.txt). Contributions are welcome: see
[CONTRIBUTING.md](https://github.com/LupoLab/solprop/blob/main/CONTRIBUTING.md). If you use solprop in published work, please cite
the repository and the version you used.
