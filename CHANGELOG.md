# Changelog

All notable changes to solprop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags.

<!-- start changelog -->

## Unreleased

### Changed

- The README, `CITATION.cff` and the citing page name solprop's Zenodo DOI,
  [10.5281/zenodo.23854248](https://doi.org/10.5281/zenodo.23854248), which covers
  every release.

## [0.1.0] - 2026-10-31

The first public release of `solprop` and of its compiled backend, `solprop-native`,
which is released at the same version. Releases are 0.x for now: a minor release may
break the API, the output schema or the resume key, and lists every such break under
"Changed (breaking)" (ADR-058).

### Added

- **Propagation equations.** The unidirectional pulse propagation equation (UPPE) and
  the generalised nonlinear Schrödinger equation (GNLSE), for the real electric field or
  a complex envelope, in waveguides and in free space.
- **Waveguides.**
  - Gas-filled hollow capillaries, with resolved modes (the default) or mode-averaged
    (`mode_average=True` on `prop_capillary` and `prop_waveguide`). Tapers,
    pressure gradients and gas flow are supported, as is a coherent wall that couples
    the modes.
  - Antiresonant hollow-core fibres (the Vincetti model).
  - Step-index, graded-index, rectangular and planar guides.
  - Modes tabulated from arrays or from external mode solvers.
  - A general modal route for any guide that supplies its own electric and magnetic
    mode fields, including guides that change along their length.
- **Free space.** Radial, Cartesian and line grids with the exact (non-paraxial)
  diffraction operator; planar guides can be resolved across their width. Launches,
  shot noise, energies, fluences and intensities measure the medium's axial Poynting
  flux at the plane, so a requested energy is the physical energy there
  ([SEP 34](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0034-medium-flux.html)).
  Far-field and aperture statistics.
- **Media and responses.**
  - Gases, gas mixtures, glasses and metals from the [optmat](https://github.com/LupoLab/optmat)
    material library.
  - Kerr and third-harmonic nonlinearity, including a dispersive electronic response.
  - Photoionisation with atomic and molecular rate models, and plasma on field and
    envelope grids.
  - Rotational and vibrational Raman responses, scalar and vector.
  - Quantum shot noise, on by default
    ([SEP 37](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0037-noise-and-stepper-defaults.html)):
    one photon per mode with a random phase, added to the input field (`noise="input"`)
    and recorded with its seed. A resumed run draws the same realisation again, and a
    scan draws each point's from a seed of its own.
    `noise=None` switches it off and `noise="modified"` keeps the realisation apart
    from the field. The Cartesian, line and hybrid planar grids, slab mode sets, the
    TDSE response and single precision carry none by default and record why. Spectral
    densities can subtract the one-photon floor (`noise_floor="subtract"`).
  - An optional helium TDSE atomic response for research. Its native backend is not
    yet publicly available.
- **Integrators and their defaults.**
  - Two adaptive integrators: a low-call exponential Adams method (`Multistep()`) and
    an embedded interaction-picture Runge–Kutta pair (`RungeKutta()`), with a
    low-storage Runge–Kutta variant and fixed schedules. The linear propagation is
    integrated exactly.
  - Each interface has its own default integrator and tolerances, chosen by
    measurement: `RungeKutta()` for `prop_gnlse` (`rtol` 1e-7), `prop_waveguide` (1e-5)
    and `prop_free` (1e-4); `Multistep()` for `prop_capillary` (1e-6), `prop_radial`
    and `prop_planar` (1e-5). The
    [choosing-a-stepper page](https://solprop.readthedocs.io/en/latest/guide/numerics/choosing-a-stepper.html)
    tabulates them and says when to choose another.
  - `Multistep(carrier_frame="scalar")`, an opt-in carrier frame for the Adams history
    that is the fastest method measured on dense, multimode, Raman-active runs such as
    a comb-to-continuum capillary
    ([SEP 35](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0035-carrier-frame-exponential-adams.html)).
  - Backends agree with each other to bounded, rounding-level differences.
- **Time-window boundaries**
  ([SEP 21](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0021-absorbing-boundaries.html)).
  Every interface absorbs at the edges of the time window and of the spectrum. The
  temporal collar lies outside the requested window: the grid widens to
  `trange / (1 - 2 collar_fraction)` (`collar_fraction` 0.1), so the whole request is
  flat, and transform sizes default to `fft_size="auto"`, the cheapest 2-3-5-smooth
  size up to the next power of two. `absorber=False` switches the soft collars off and
  keeps the same grid. A boundary monitor, on by default (`boundary_check=` on every
  `prop_*` function and `Simulation.from_*` constructor), warns (`BoundaryWarning`) and
  then stops the run (`BoundaryError`) when light above the shot-noise floor reaches
  the window's edge, and records the energy the collar removed.
- **Coupling simulations**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html)).
  `solprop.coupling` hands a field from one simulation to the next. A `FieldPlane`
  holds a coherent field at one plane with its reference and provenance; `couple` maps
  it into another simulation's `InputPort` under a declared `CouplingModel` and returns
  an energy report; `transport` carries a free-space plane through lenses, apertures,
  masks, mirrors, interfaces and spectral filters; `Outcouple` synthesises a guide's
  exit field. An incident beam launches into a general modal guide with the share of
  its energy that the guide captures. Each step is admitted against an error budget and
  refuses what it cannot certify.
- **Backends.**
  - `numpy_ref`, the NumPy reference, against which every other backend is tested
    within derived error bounds, and `numba_cpu`, compiled with Numba. The backends come
    in tiers: new features land on `numpy_ref` and `native`; `numba_cpu` is mature and
    takes a new feature only where its design says so; `mlx` is experimental.
  - The optional [native backend](https://solprop.readthedocs.io/en/latest/guide/performance/native.html),
    the solprop-native distribution: C++20 kernels on Kokkos for multi-core CPUs
    (OpenMP) and for NVIDIA and AMD GPUs (CUDA, HIP). `pip install "solprop[native]"`
    installs a precompiled CPU wheel on macOS on Apple silicon, Linux on x86-64 and
    aarch64, and Windows on x86-64. `backend="auto"` selects it when installed, except
    on a processor below the wheel's CPU baseline, where it uses `numba_cpu` and warns
    once ([SEP 29](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0029-compiled-backend-distribution.html)).
    On the native backend the exponential Adams march runs in compiled code, many steps
    per call
    ([SEP 39](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0039-native-adams-march.html)).
  - One thread budget per process, which every pool of both CPU backends stays within:
    the smallest of the CPU affinity mask, the physical cores that hold it, the
    control group's CPU quota and `SOLPROP_THREADS` (ADR-052).
  - Single precision
    ([SEP 33](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0033-single-precision-mlx.html)):
    `precision="single"` holds the state, the transforms and the responses in binary32,
    with the controller in binary64. `numpy_ref` runs it as the reference, and the `mlx`
    backend (`pip install "solprop[mlx]"`) on the Apple GPU. A
    [survival record](https://solprop.readthedocs.io/en/latest/guide/performance/single-precision.html)
    states which problems keep their accuracy.
- **Running and recording.**
  - HDF5 output with periodic snapshots and provenance, and resuming an interrupted
    run, which continues the Adams history.
  - A summary at the start of every run of what it is about to compute (the code, the
    machine, the backend, the grid, the medium, the responses, the noise, the
    integrator, the absorber and the output), and at its end the wall time, CPU time
    and peak memory. Each execution records its machine and what it spent in the file.
  - Statistics along the propagation.
  - Parameter scans on process pools and Slurm clusters.
- **Analysis and tools.**
  - Processing and plotting. Spectral ranges and band-pass windows are given in the
    units of any spectral axis (`spec_range`, `Passband`).
  - Ensemble coherence by spectral band, and replay of a run's accepted steps
    (`FixedSchedule.from_history`), to compare two precisions or backends on one
    discretisation.
  - Soliton and dispersive-wave design tools.
  - Pulse compressors: Taylor phase, chirped mirrors, materials, gratings and prisms.
  - An interactive session through `import solprop.sol as sol`.
- **Performance tracking** for developers
  ([SEP 38](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0038-performance-tracking.html)):
  `python -m benchmarks track` times about a dozen representative workloads, with each
  interface's defaults, at one thread and at the production width, and records them by
  hardware; `python -m benchmarks ab` compares two builds.
- **Documentation.** An introduction with executable lessons, a user guide, a
  long-form primer on the physics and numerics, focused tutorials and worked examples,
  the API reference, and a developer's guide with the design records (SEPs and ADRs).

### Changed (breaking)

Nothing was released before 0.1.0. For scripts and files from a development checkout:

- **Output files of development versions are not read.** The output schema restarts at
  1 (`solprop.output.SCHEMA_VERSION`); keep the development install that wrote such a
  file to read it. The kernel contract's version is `KERNEL_SCHEMA_VERSION`, separate
  from it.
- **Defaults changed late in development**, so a development script's results change:
  shot noise is on; each interface takes its own integrator and tolerances; the
  temporal collar lies outside the requested window, which widens the grid
  (`collar_placement="inside"` rebuilds the former geometry); transform sizes are
  `fft_size="auto"`; and the boundary monitor is on.
- **Renamed or removed.** `prop_gnlse`, `prop_capillary`, `prop_waveguide` and their
  `Simulation.from_*` constructors take `wavelength` and `shape`, not `lambda0` and
  `pulseshape`, and `responses_for` takes `wavelength`, as the pulses and the design
  tools already did. The plots' `wavelength_range` is `spec_range`, in the units of
  their axis, and `solprop.plotting.style.spectral_limits` is gone
  (`wavelength_to_axis` converts a wavelength range); `bandpass` takes `limits`, not
  `wavelength_limits`; `poynting_spectrum` is gone (`beam_spectrum` is the axial-flux
  spectrum); the `same_basis` function is gone (use `couple`); and `solprop.driver` no
  longer exports `propagate_fixed_rk4`, the tests' fixed-step oracle. `Multistep()`
  refuses the `PIController` settings it does not read.

### Known limitations

- **The `mlx` backend is experimental.** Its Metal kernels are tested by hand on a
  Metal device, not in continuous integration, and its coverage is not measured there.
  The single-precision survival record is the reference implementation's (on `mlx` for
  two cases). On the default `fft_size="auto"` grids MLX's GPU transform runs four to
  eight times slower than on powers of two, so a run can be slower than binary64
  `native`: build single-precision grids with `fft_size="pow2"`. The guide's speed table
  predates the current grids and is marked so.
- **Free space has no accurate transverse absorber.** By default a radial grid's edge is
  a hard wall and a Cartesian or line grid wraps around; the optional real-space collar
  (an absorber given to a spatial run, `Absorber.spatial`) removes only part of the light that reaches it, at a large cost
  in steps, and the boundary monitor watches the time window only. Size the transverse
  window so that the light stays clear of its edge.
  [SEP 40](https://solprop.readthedocs.io/en/latest/dev/seps/sep-0040-transverse-absorbing-boundaries.html)
  (a draft) rebuilds these boundaries.
- **A run is resumed by the release that wrote it.** A run's configuration hash covers
  every recorded field, so a release that adds one changes every hash, and within 0.x
  the output schema may change too: finish a run with the release that started it.
- **GPU builds are made from source.** The wheels are for CPUs; CUDA and HIP builds of
  solprop-native are built from the repository
  ([building the native backend](https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html)).
- **CPython 3.14 only.** solprop-native, pyFFTW and h5py have no wheels for 3.15 yet.
- **macOS: import solprop before pyFFTW.** pyFFTW's macOS wheel starts its own OpenMP
  runtime, so the native backend refuses to load in a process that imported pyFFTW
  first; solprop itself imports them in the working order.
