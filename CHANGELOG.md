# Changelog

All notable changes to solprop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags.

<!-- start changelog -->

## Unreleased

The first public release. It contains:

### Added

- **Propagation equations.** The unidirectional pulse propagation equation (UPPE) and
  the generalised nonlinear Schrödinger equation (GNLSE), for the real electric field or
  a complex envelope, in waveguides and in free space.
- **Waveguides.**
  - Gas-filled hollow capillaries, mode-averaged or with resolved modes. Tapers,
    pressure gradients and gas flow are supported, as is a coherent wall that couples
    the modes.
  - Antiresonant hollow-core fibres (the Vincetti model).
  - Step-index, graded-index, rectangular and planar guides.
  - Modes tabulated from arrays or from external mode solvers.
  - A general modal route for any guide that supplies its own electric and magnetic
    mode fields, including guides that change along their length.
- **Free space.** Radial, Cartesian and line grids with the exact (non-paraxial)
  diffraction operator. Planar guides can be resolved across their width. Absorbing
  spatial boundaries are included, and far-field and aperture statistics.
- **Physical flux in free space**
  ([SEP 34](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0034-medium-flux.html)).
  Spatial launches, shot noise, energies, fluences, intensities and statistics measure
  the axial Poynting flux of the medium at the plane (`SpatialFlux`), so a requested
  energy is the physical energy there; local flux uses the complex admittance and is
  signed, and `peak_field` names the physical-field diagnostic. Accessors take
  `(field, flux)`; `SpatialPulse.build` and `spatial_noise` take the medium and plane.
  `critical_power` takes the carrier `index=` and returns the physical power, `n0` times
  the vacuum-measure value. A radial gas gradient carries the forward amplitude term
  `-ln(Y_z(z)/Y_z(0))/2` and is admitted under a real-index adiabatic certificate
  (`AdiabaticAccuracy`, default 1e-4 on field error and reflected energy;
  `SimulationSettings.adiabatic_accuracy`). `poynting_spectrum` is removed:
  `beam_spectrum` is the axial-flux spectrum. Kernel schema 35 (`flux`) and output
  schema 23 (`flux_model`, `flux_plan`).
- **A carrier frame for the exponential Adams history**
  ([SEP 35](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0035-carrier-frame-exponential-adams.html),
  phases 1–4). `Multistep(carrier_frame="scalar")` interpolates, inside each step, the
  source with the carrier's phase removed and integrates it against correspondingly
  shifted moments, exactly for any shift, with each step capped at one radian of
  carrier phase (`CARRIER_PHASE_CAP`); on field grids and envelope grids with the third
  harmonic. On dense, multimode, Raman-active runs such as a comb-to-continuum
  capillary it is the fastest method measured, about half the time of the default and
  of IP-DOPRI5; elsewhere it saves less or nothing, and a weak band still forming is
  less tightly controlled, so it is opt-in and `"off"` stays the default. The manual's
  decision guide says when to use it. It runs on `numpy_ref`, `numba_cpu` and native
  CPU builds; native device builds and the free-space routes refuse it by name. The
  benchmark suite's `ctc_reduced` case moves to a 2 ps window (revision 2). Kernel schema 37 (`Multistep.carrier_frame`); every `Multistep`
  representation names the policy, so a march snapshot taken before it restarts its
  history.
- **Incident beams into general modal guides**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html),
  phases 1 and 3). `from_waveguide(provider=..., incident=SpatialPulse(...),
  coupling=CouplingModel("vector_paraxial", model_error=...))` and `prop_waveguide`
  launch the share of an incident beam that the guide's channels capture. The beam's
  energy is met over its whole plane in the core medium, and the light that misses is
  lost rather than restored. The coupling's `CouplingReport` is recorded in
  `configuration["launch"]`, and `SpatialPulse.incident` samples a beam normalised over
  its own plane. The node-sampled `transverse=` profile, its rescaling to `energy=`, the
  3 % uncaptured warning, `LAUNCH_UNCAPTURED_WARNING` and `IncidentLaunch` are removed;
  a strongly guiding core is refused by the vector model's contrast screen.
- **Plane exchange between simulations**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html),
  phase 2). `solprop.coupling` holds `FieldPlane`, a coherent field at one plane with
  its reference, geometry, metric, basis identity and provenance; `regrid`, the direct
  finite-time Fourier remap between temporal grids with its loss report and no
  renormalisation; and `InputPort`. `processing.field_plane` reads a saved plane back
  (refusing a modified-noise run's, whose auxiliary noise phase it does not reconstruct),
  `Simulation.input_port()` describes a simulation's entrance and `with_input` takes a
  launch bound to it, recording `configuration["coupling"]` and drawing no new noise.
  Output schema 24 records each run's plane-reference record (`RunMetadata.reference`).
- **Coupling models**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html),
  phase 3). `couple(plane, port, model=CouplingModel(kind, model_error=...))` is the one
  path from a field plane to a launch bound to a port, returning the launch and an
  immutable `CouplingReport`. `same_basis` copies an identical basis, applies a declared
  energy-preserving `BasisChange` or recovers a general modal plane by the flux pairing;
  `scalar_paraxial` fits Gaussian, analytic or sampled beams into a capillary's HE1m
  modes, one capillary's exit into another of any radius, and one weakly guiding
  step-index fibre into another, in the common paraxial measure; `vector_paraxial`
  couples into a general modal guide with the forward H closure. Each is admitted
  against its energy budget: angular tails, the weak-guidance contrast screen or a
  recorded pair envelope (`CAPILLARY_ENVELOPE`, `STEP_INDEX_ENVELOPE`), metric defects,
  the H closure, a passive-map certificate and numerical errors; uncaptured energy is
  reported only where a common positive metric exists. `FieldPlane.from_analytic`
  samples an analytic field with its support and domain checks, `FieldPlane.from_pulse`
  takes a reference `index=`, `field_plane(run, source=...)` carries an ordinary mode
  set's scalar profiles (`ScalarModeBasis`), and `Outcouple` synthesises a guide's exit
  field on a spatial grid under a declared interface model. Ports state the models they
  admit. The phase-2 `same_basis` function is removed in favour of `couple`.
- **Transport between planes**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html),
  phase 4). `coupling.transport` carries a radial or Cartesian `FieldPlane` through an
  ordered sequence of linear elements -- `FreeSpace` (exact angular-spectrum
  propagation, with an explicit `"angular"` or `"fresnel"` output plane), `ThinLens`,
  `Aperture`, `TransmissionMask`, `Alignment`, `PolarisationTransform`, `Mirror`,
  `PlanarInterface` and `SpectralTransfer` (any `optics` transfer function or `Chain`)
  -- and returns the plane with its reference, geometry, medium, metric and provenance
  updated and a sequential ledger in joules. Thin screens carry their angular-flux
  certificate; undersampled phases, wrapped rays and evanescent content are refused
  unless their removal is declared, and a same-grid field filling the window's outer
  band is refused beyond `FreeSpace(edge_tolerance=...)` (default 1e-4). Refusals raise `CouplingError`.
- **Coupled workflows**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html),
  phase 5). The eight compositions of the coupling chapter -- capillary into capillary,
  a nonlinear gas gap into a capillary, fibre into fibre, Gaussian, analytic and
  Cartesian beams into a guide, a guide's far field, and SPM, compression and relay
  optics into a second capillary -- run as short integrated tests that assert their
  energy ledgers, with sources and receivers on different CPU backends. A plane's path
  certificate (`FieldPlane.certificate`, `CouplingReport.certificate`,
  `TransportReport.source_certificate`) is the rebuilt `AdiabaticAdmission`, not its raw
  record. `CAPILLARY_ENVELOPE` now bounds the scalar model against Marcatili's leaky core
  field (bound 6e-3 at its range's top, delta = 1e-2, scaled by the pair's delta) rather
  than against the scalar profile itself; bins outside an envelope's range, including a
  fibre's V across the band, are counted in full; the modal overlap quadrature enters
  the budget as `2 delta + delta^2`. `scalar_paraxial` also admits a radial or
  Cartesian beam into a weakly guiding step-index fibre (LP01, V 1.5 to 2.4, contrast
  to 0.01) within `STEP_INDEX_BEAM_ENVELOPE`, the scalar overlap against the beam's
  full-vector flux coupling into the exact HE11 pair maximised over every field in the
  admitted cone (bound 1e-2 at contrast 0.01, scaled by the fibre's contrast).
  `same_basis`'s flux recovery of a general modal plane allows the plane's own metric
  defect as well as the port's when it checks that the field lies in the port's span,
  and reports both.
  the budget as `2 delta + delta^2`.
  `VECTOR_STEP_INDEX_ENVELOPE` bounds `vector_paraxial` into a step-index guide
  against an independent full-vector mode-matching solution of the entrance facet,
  maximised over every incident field in the admitted cone (bound 3.0e-3 at contrast
  0.05, twice the Fresnel ceiling a mode near cutoff approaches, V 1.5 to 8, scaled by
  the squared contrast); its item is counted and a guide outside its range is refused
  by name. A step-index
  port carries `InputPort.normalised_frequency`; a graded-index guide keeps an
  uncounted item without a value.
- **Media and responses.**
  - Gases, gas mixtures, glasses and metals from the [optmat](https://github.com/LupoLab/optmat)
    material library.
  - Kerr and third-harmonic nonlinearity, including a dispersive electronic response.
  - Photoionisation with atomic and molecular rate models, and plasma on field and
    envelope grids.
  - Rotational and vibrational Raman responses, scalar and vector.
  - Quantum shot noise.
  - An optional helium TDSE atomic response for research. Its native backend is not
    yet publicly available.
- **Numerics.**
  - Two adaptive integrators: a low-call exponential Adams method (the default) and an
    embedded interaction-picture Runge–Kutta pair. A low-storage Runge–Kutta variant is
    included.
  - The linear propagation is integrated exactly, and absorbing boundaries act as rates.
  - Fixed and adaptive schedules.
  - Bounded, rounding-level reproducibility across backends.
- **Backends.**
  - A NumPy reference backend and a compiled Numba CPU backend.
  - The optional [native backend](https://solprop.readthedocs.io/en/latest/guide/performance/native.html)
    (`backend="native"`), the solprop-native distribution built from `native/` in this
    repository at solprop's own version: C++20 kernels on Kokkos, Kokkos Kernels and
    Kokkos-FFT, bound with nanobind, for multi-core CPUs (OpenMP) and for NVIDIA and AMD
    GPUs (CUDA, HIP). `pip install solprop[native]` installs a precompiled CPU wheel on
    macOS on Apple silicon, Linux on x86-64 and aarch64, and Windows on x86-64; other
    platforms and GPUs build it from source. A CUDA build needs GCC 14 or newer as the
    host compiler and CUDA 12.8 or newer (qualified on CUDA 13). `backend="auto"` selects it when installed,
    except on a processor below the installed build's CPU baseline, where it uses
    `numba_cpu` and warns once, naming the missing features; `backend="native"` raises
    there
    ([SEP 29](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0029-compiled-backend-distribution.html)).
  - One thread budget per process (ADR-052): the smallest of its CPU affinity mask, the
    tightest control-group CPU quota on its path (versions 1 and 2, rounded down) and
    `SOLPROP_THREADS`. Every pool of both CPU backends is sized within it, explicit
    counts included; `numba_cpu`'s kernels are also held to `NUMBA_NUM_THREADS`, so a
    large problem no longer fails at construction when that is below the CPUs, and the
    native backend no longer takes every CPU of the machine. `SOLPROP_OVERSUBSCRIBE=1`
    lets an explicit count exceed the CPUs for thread-count tests. The budget and its
    source are recorded in run metadata; the `Pool` and Slurm scan executors set
    `SOLPROP_THREADS` per worker instead of forcing every pool to the worker's count.
  - Each backend is tested against the reference implementation within derived error
    bounds.
  - A short start-up: a run on the native backend imports neither Numba nor
    `scipy.signal`, plotting imports `matplotlib.pyplot` only when it draws, and an
    optmat rate builds only the model it uses (optmat 0.9.1, which reads only the data a
    model needs). A five-centimetre capillary run with ionisation is under way in about
    3 s on an Apple M5 Max, from 5.5 s.
- **Single precision**
  ([SEP 33](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0033-single-precision-mlx.html)).
  `precision="single"` on `SimulationSettings` and on `prop_gnlse`,
  `prop_capillary` and `prop_waveguide` holds the state, the transforms and the
  responses in binary32, with the controller and the propagator's exponent in binary64,
  every narrowed table at its own power-of-two exponent and the state at an exact scale.
  `numpy_ref` runs it as the single-precision reference (oracle) on the fibre, radial and
  Cartesian routes; `numba_cpu` and `native` refuse it. Its default `rtol` is the
  floor `1e-6`, below which an explicit tolerance is refused, and its default integrator
  `RungeKutta()`; noise models and the Adams integrator are refused. A
  [survival record](https://solprop.readthedocs.io/en/latest/guide/performance/single-precision.html)
  states which problems keep their accuracy. Every run records its precision in its
  configuration, so a resume in the other precision is refused (kernel schema 36,
  output schema 25).
  - The `mlx` backend (`pip install solprop[mlx]`, macOS on Apple silicon) runs
    single-precision capillary, fibre, radial and Cartesian free-space runs, the
    envelope-grid plasma among them, on the Apple GPU through MLX, validated against
    the reference backend's single-precision path; `backend="auto"` chooses it under
    `precision="single"`. Its propagators are formed in double-float arithmetic in a
    custom Metal kernel, its matrix products are checked not to be TF32, and each
    envelope sample's cycle moments are one kernel thread.
  - Under single precision a replayed step (or an adaptive floor) whose propagator
    exponent passes binary32's range, `exp(88.7)`, is refused at translation, and one
    within a tenth of it is warned of.
- **The native backend.**
  - Native CPU optimisation: a coherent NEON FFTW build for Apple silicon and Linux
    aarch64 (the wheels' smoke test requires it on Arm), long-row
    Raman planning with a compact FFTW/vDSP selection rule, and deferred embedded
    transverse diagnostics for ordinary DOPRI5 modal trials. Accelerate BLAS, kernel
    schema 35, frontend signatures, protected arithmetic and state layouts are
    unchanged. See the [CPU measurements](https://solprop.readthedocs.io/en/latest/dev/benchmarks.html#native-cpu-fft-and-diagnostic-scheduling)
    for the lightweight validation scope and rejected traversal and sizing leads.
  - `NativeBackend` runs one trial step per native call, with states, views and FFT
    plans resident in the execution space; `NativeBatchBackend` runs fixed-schedule
    batches with fused accepted updates.
  - Every route: mode-averaged, multimode and general modal guides (moving guides and
    coupled flows included), radial, Cartesian, line and hybrid planar free space, on
    field and envelope grids; Kerr, third-harmonic, field and envelope plasma (both
    running-integral rules), scalar, vector and tensor Raman, gas mixtures, the
    dispersive electronic response, noise, absorbing boundaries, statistics including
    the far-field and on-axis plasma statistics.
  - All three integrators: the Runge–Kutta march, the resident low-call Adams method in
    both formulations (with dense output and coupled sections), and the low-storage
    march. The resident Adams history is copied to the host for periodic snapshots and
    kept resume histories, and installed again on a resume, which continues it (SEP 29).
    The CF4 coupled-flow method and the TDSE response are refused by name.
  - The Chebyshev z table on CPU builds (SEP 29): its coefficients are held resident,
    and each native call evaluates the table at its positions into term slots that
    the stage weights select, so a moving dispersion, three or more species on
    independent density paths (modal and radial), a multimode taper without its
    coupling and a forced table run under every integrator and in batches, and
    `backend="auto"` runs them. GPU builds refuse the table by name, giving its size
    and naming the CPU backends.
  - Execution spaces: OpenMP, or Kokkos Threads (`SOLPROP_NATIVE_ENABLE_THREADS`), with
    FFTW or MKL's FFTW interface on CPUs; CUDA with cuFFT and cuBLAS; HIP with rocFFT
    and rocBLAS. Device builds are qualified on a data-centre NVIDIA GPU and an AMD
    Instinct GPU.
  - Build profiles (`SOLPROP_NATIVE_PROFILE`): `LOCAL`, the default, compiled for the
    build machine with a vendor BLAS found at configure time; `DISTRIBUTION`, for wheels,
    compiled for a CPU baseline (`SOLPROP_NATIVE_CPU_BASELINE`; x86-64-v3, Apple M1 or
    Neoverse N1, and x86-64-v2 for an older x86-64 processor from source) checked before
    the extension loads (`/proc/cpuinfo` on Linux, `IsProcessorFeaturePresent` on
    Windows), with Accelerate on macOS or an
    OpenBLAS built with OpenMP and carried in the wheel; and `REFERENCE`, with no vendor
    BLAS or machine-specific code. Multiply-adds are contracted outside a few protected
    blocks, and builds agree to rounding.
  - Linux CPU builds from source need GCC 14 or newer (or IntelLLVM or Clang). GCC 13
    is refused at configure time: it compiled the strict floating-point helpers through
    the stack, and coupled Multistep steps took 1.25 to 1.45 times as long as with
    GCC 14.
  - Threads: a process cap (`SOLPROP_NATIVE_THREADS`) and kernel, FFT and Raman pools,
    read once at import, with per-operation widths chosen by problem size; a host
    OpenBLAS is fitted to the same pool.
  - Introspection: `build_info()` and `python -m solprop_native` report provenance,
    providers, libraries and thread settings; `check_build()` checks the build against
    the installed solprop's kernel-contract schema and named members.
  - The per-stage Chebyshev rows of the scalar tables and the Adams basis weights run as
    strict C++ transcriptions of solprop's own, so a native run never calls Numba. Each
    backend supplies these evaluations through `StepDriver.helpers` (kernel schema
    34): the NumPy reference its uncompiled Python, `numba_cpu` the same functions
    compiled.
  - Profiling: named kernels, regions and allocations (`solprop_native.*`) for Kokkos
    Tools, a per-state memory ledger, and an optional Clang coverage build
    (`SOLPROP_NATIVE_ENABLE_COVERAGE`).
  - Wheels carry the licence texts of the libraries compiled into them or copied in by
    the wheel repair (FFTW, the OpenMP runtime; `SOLPROP_NATIVE_BUNDLED_LICENCES`).
- **Workflows.**
  - HDF5 output, including periodic snapshots, provenance, and resuming an interrupted
    run, which continues the Adams history on every backend; a history the resuming
    backend cannot continue restarts and says why, and a method that keeps none warns.
  - Statistics along the propagation.
  - Parameter scans on process pools and Slurm clusters.
  - Processing and plotting.
  - Soliton and dispersive-wave design tools.
  - Pulse compressors: Taylor phase, chirped mirrors, materials, gratings and prisms.
  - An interactive session through `import solprop.sol as sol`.
- **Documentation.** An introduction with executable lessons, a user guide, a
  long-form primer on the physics and numerics, focused tutorials and worked examples,
  the API reference, and a developer's guide with the design records (SEPs and ADRs).

### Fixed

- **Spectral ranges of the plots in the units of their axis.** `prop_2d`, `spec_1d` and
  `spectrogram` took their spectral window only as `wavelength_range`, so a frequency
  axis could not be given a range in hertz, nor one starting at zero. They now take
  `spec_range`, in m, Hz or rad/s as `axis` says, like the processing accessors;
  `wavelength_range` and `solprop.plotting.style.spectral_limits` are gone, and
  `wavelength_to_axis` converts a wavelength range to any axis.
- **Third harmonic on an envelope grid.** With `thg=True` the envelope frame measured
  its group delay from the carrier, which also removed a constant phase
  β₁ω₀z that the harmonic term does not share. The third harmonic was
  driven with a spurious phase mismatch of 2β₁ω₀ (about 1e7 rad/m in a
  gas) and came out orders of magnitude too weak, with very small steps. A frame that
  keeps the carrier phase is now a pure delay on both grids
  (`solprop.linear.frame_origin`), and envelope runs with the harmonic agree with
  field-grid runs. Results of envelope runs with `thg=True` change.
- **Reading the planes of a run with periodic step statistics.** The in-memory store
  learnt the statistic names only at the first plane with rows, and gave the planes
  before it no empty piece in those columns, so `planes()` (and `to_file`) of any run
  with a periodic subset failed with "step statistics columns must have equal row
  counts". The HDF5 file was not affected.
- **Native CPU cost cliffs at 2-3-5-smooth grid sizes.** Two causes made some smooth
  sizes 1.6-1.8 times slower per source evaluation than the neighbouring power of two
  on the native CPU backend. The node-chunk budget took `budget // per_node`, so 33
  resolved-mode nodes under a limit of 32 ran a 32-node pass and then a one-node tail on
  one thread; the nodes are now spread evenly over the same number of passes (17 and
  16). And the threading thresholds for stage kernels (8192 elements) and modal FFTs
  (32,768 samples) sat just above the four-mode RDW's sizes at n_t 4000, so its stage
  and transforms ran serially; on Linux they are now 4096 and 16,384. On an 18-core
  Cascade Lake socket the RDW case costs 1.32 ms per evaluation at n_t 4000 (was 2.47;
  4096: 1.24-1.36) and 2.54 ms at 8640 (was 3.55; 8192: 2.28-2.31); on Zen 3 n_t 4000
  is 24-27 % faster (`native/benchmarks/size_cliffs.py`). macOS keeps 8192 and 32,768:
  on an Apple arm64 laptop the serial stage and transforms are the faster ones at those
  sizes, and its node-chunk gain at n_t 8640 is 26 %.
