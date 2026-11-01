# Changelog

All notable changes to solprop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags.

<!-- start changelog -->

## Unreleased

### Added

- A published-install check after each PyPI release, on all four wheel platforms:
  fresh CPython 3.14 environments resolve the native extra with pip and verify its
  version, dependencies and default backend. The same check can be run by hand for a
  named release, with an optional run of its tagged README example and retained logs.

- The first performance-history rows (`benchmarks/history/`), recorded at 0.1.0 on
  six CPU and three GPU hardware sets, with numba runs beside the native ones on three
  of them; `docs/dev/benchmarking.md` lists how each is run. The history takes any
  backend: each row names it, and rows compare only within one backend; a run the
  history recorded no longer makes the next run in that checkout dirty. The benchmark
  harness names an AMD GPU from its PCI IDs where ROCm reports only
  "AMD Radeon Graphics".

- `resume="force"` continues a file across changed physics, in the same file, while
  the stored arrays' layout agrees, the statistics fit and work is left: the Adams
  history restarts, the session's execution record marks the file, and later sessions
  and readers respect the mark (SEP 42, ADR-063). `RunMetadata.compare` and
  `solprop.output.compatibility` hold the comparison; `RunOutcome.resumes` and
  `RunOutcome.forced_positions` read back how each session continued its file.

### Changed

- A resume compares the stored and requested run records path by path instead of
  requiring equal configuration hashes (SEP 42, ADR-063). The layout and the physics
  must agree, numbers to 1e-11 per element, so a run resumes after an upgrade that adds
  a field with a declared neutral value, after an optmat release that changed no number
  it uses, or on another platform; a changed tolerance or integrator continues an
  unfinished run, with a warning and a record of it, and a finished file is returned
  only under the numerics every session ran. optmat models are compared by the digest
  of their numbers, so a changed optmat version, model name or status no longer refuses
  a resume. A refusal names what differs by class and part of the record, and says what
  to do instead. An Adams history continues across a changed call budget or safety
  factor, an Adams file continues under Runge-Kutta, and the interpolated-plane refusal
  is decided by the file, whatever `interpolate_planes` the resuming run asks for.
  `source=` in `field_plane` is checked by the same comparison.
- A scan rerun reuses a finished point whose record differs only in rounding, a neutral
  value or provenance, and refuses one whose numerics or physics changed, naming what;
  the `Queue` and `Slurm` executors keep exact identity (SEP 42). The output schema is
  bumped only for a change an older reader would misread (ADR-063).
- `backend="auto"` now warns, once per process, whenever it chooses `numba_cpu`: when
  `solprop-native` is not installed as well as on a processor below its wheel's baseline.
  The warning says how to install the native backend; an explicit `backend="numba_cpu"`
  does not warn. The README and manual now recommend the native backend by default and
  describe `numba_cpu` as its slower fallback, with the 0.1.0 measurements. They no longer
  advise `numba_cpu` for scans of one-thread workers, which the history does not support,
  and they now say that only GPU builds refuse z-tabulated problems.
- Planned work has moved to [GitHub issues](https://github.com/LupoLab/solprop/issues),
  labelled by type, priority and area: the roadmap now lists only SEPs (ADR-062). Issue
  templates and `CONTRIBUTING.md`
  say what an issue or a pull request should hold and what to check before opening one.
- The README, `CITATION.cff` and the citing page name solprop's Zenodo DOI,
  [10.5281/zenodo.23854248](https://doi.org/10.5281/zenodo.23854248), which covers
  every release.

### Changed (breaking)

- `resume` takes `False`, `True`, `"interpolated"` or `"force"`, and any other value is a
  `ValueError`: a mistyped word used to read as `True`. An appending `HDF5Output` holds
  the file's own record as its `metadata` after a resume, not the one offered, and
  compares the two instead of requiring equal hashes (SEP 42).
- The output schema is 2: a `DataPulse` records its `spectral_phase` (below), and every
  field of a recorded dataclass is recorded, at its default too. This release refuses
  a file written by 0.1.0, and every configuration hash moves, so a 0.1.0 run cannot be
  resumed; read such a file with 0.1.0.
- `DataPulse.from_time` is replaced by the class `TimeDataPulse`: write
  `TimeDataPulse(t=t, field=field, energy=energy)`. It holds the time record and has no
  `omega`, `spectrum` or `spectral_phase`. The method kept the spectrum at the record's
  own bin spacing and splined its magnitude and phase, which cannot follow the fringes
  of pulses far apart: pulses at 100 and 600 fs in a record of 800 fs, of relative
  amplitude −0.9, came out 85 % wrong with no warning. The class evaluates the record's
  transform exactly at the grid's frequencies, 3e-12 from the direct sum on that case,
  less its mean: a baseline is an offset of the instrument, not part of the field.

### Fixed

- A scan derived from one cached simulation, the scans guide's own recipe, now gives every
  point its own noise seed. `with_input(noise="keep")` and `with_mixture` keep the ability
  to draw unseeded noise again, and the draw replaces only the initial noise of the shared
  translation. Before, such points shared the base's draw, depended on the worker under
  `Pool`, could not be rerun in a new process and failed under `base_seed`. Resuming a
  derived, unseeded run now takes the file's seed, as a constructor's does, instead of
  being refused.
- One object returned by a scan builder for every point no longer collapses a noise
  ensemble under `Batch`, where every member ran on the last point's noise while the file
  recorded each point's own seed. The draw now goes on a copy, and the builder's object is
  left as it is. A file written by such a scan holds the wrong noise for its recorded
  seeds and a rerun does not detect it: run those points again.
- A scan file whose points' run records declare another output schema, a 0.1.0 scan
  among them, is refused by name, in the words a run file is refused with, when it is
  read (`ScanOutput`, `read_point`) or resumed or collected (`run_scan`, a queue's
  collector). Before, only the scan container's own version was checked, and that
  stayed 1, so such a scan was read as if its layout were this release's.
- The native backend's HIP build compiles again under its default
  warnings-as-errors: ROCm's Clang refused a thread-local variable that only an OpenMP
  build uses (0.1.0 needed `-DSOLPROP_NATIVE_WARNINGS_AS_ERRORS=OFF`).
- Resuming a file that holds only its launch plane, from an `input`-noise run stopped
  before its first saved plane, no longer adds the noise draw to the stored field a
  second time, which had doubled the noise amplitude for the rest of the run on every
  backend. `set_initial` takes `resume=True` for a stored field, which moves the kernel
  schema to 40: rebuild solprop-native. It moves neither the output schema nor the
  resume key.
- `DataPulse.from_spectral_density` interpolates the phase it is given. It used to wrap
  the phase into the complex spectrum and unwrap it again, which cannot follow a change
  of more than pi between samples, so a strongly chirped pulse was built wrongly with no
  warning: 20 000 fs² of group-delay dispersion on a 0.4 nm axis at 800 nm came out 13.5 %
  away from the converged spectrum, on a 1.6 nm axis 100 %. A `DataPulse` given a complex
  `spectrum` still unwraps it, and its docstring states that limit.
- A measured field in time puts a pulse where it sat in the record (`TimeDataPulse`,
  above). `DataPulse.from_time` unwrapped the phase, and a pulse more than half the
  record's length from `t = 0` changed its phase by more than pi per bin, so it was moved
  by one record length with no warning: at 500 fs in a record from 0 to 800 fs it came out
  at −300 fs.

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
