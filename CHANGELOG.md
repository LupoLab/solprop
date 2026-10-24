# Changelog

All notable changes to solprop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags.

<!-- start changelog -->

## Unreleased

The first public release. It contains:

### Added

- **Experimental Julia CPU backend.** Explicit `backend="julia"` in prepared source
  checkouts runs constant scalar field-grid resolved and mode-averaged capillaries with Kerr/THG,
  IP-DOPRI5, both temporal absorbers and normal output, using schema 34 unchanged.
  PythonCall/JuliaCall bridges to a private Julia 1.13.2 environment. GPU support and
  production packaging are deferred; see the Julia backend manual and benchmark ledger.
  Cartesian Kerr beams add planned transverse FFTs, chunked temporal transforms,
  compact separable operators, independent spatial absorption and default host
  spatial statistics, with the existing frontend and schema unchanged.
  FFT planning effort is selectable; Cartesian stages reuse equal-distance flow.
  Line free-space field/envelope propagation uses the same FFT path with the
  existing per-unit-width normalization.
  CPU tuning adds constant-step propagator reuse, contiguous SIMD loops, operation-sized
  task launches and shape-dependent temporal/Raman FFT caps, with serial BLAS for
  mode averaging. Optional Accelerate and MKL providers
  and alternating short-window benchmarks cover macOS arm64 and Linux x86_64.
  Mode averaging shares the numerical core and the existing effective-area contract,
  with default modal statistics. Mode-averaged envelopes add GNLSE Kerr, scalar
  Raman and self-steepening through the existing translated coefficients.
  Radial Kerr beams in constant media use the translated Hankel maps, independent
  spatial absorption and default spatial statistics through the same schema.
  The SEP 30 core extension adds single-channel field plasma, longitudinal stage
  expansions, constant scalar Raman, resident exponential Adams with transported
  compensation, and Cartesian envelopes with the existing low-storage RK4IP scheme.
  Cartesian and line fields also evaluate the scalar plasma current in physical-space
  chunks and report accepted-field populations. Parallel large copies and inverse
  FFT normalization, an exact real-root branch and omission of exactly absent
  absorber transforms improve the retained strided Cartesian implementation.
  Modal and radial sources also omit transforms for exactly absent additive losses.
  The standard benchmark CLI admits Julia explicitly and records Julia source and
  package identities. Julia computes accepted-field plasma populations and integration
  diagnostics in one observation pass, with shared host reductions and other statistics;
  GPU execution and the remaining physics remain deferred.
  CPU plasma filtering exposes independent samples to SIMD and skips only exact
  zero support, preserving reflected endpoints and the protected running integral.
  Columns with exactly zero rates and no initial electrons also omit the current
  integration, clearing reused outputs without introducing a small-rate cutoff.
  Adams prepares only the stage bundles it consumes; duplicate DOPRI offsets share
  the existing pure bundle evaluation without changing the schema.
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
  ([SEP 34](https://solprop.readthedocs.io/en/latest/dev/seps/sep-0034-medium-flux.html)).
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
    there.
  - Each backend is tested against the reference implementation within derived error
    bounds.
- **The native backend.**
  - Native CPU optimisation: a coherent NEON FFTW build for Apple silicon, long-row
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
