# solprop architecture

This is the current architecture of solprop, the reference that changes to the design
are checked against. Companion documents:
[conventions.md](conventions.md) (units, signs, shapes), [stepper.md](stepper.md) (the
solver), [backends.md](backends.md) (execution), [kernel-contract.md](kernel-contract.md)
(the seam), [decisions.md](decisions.md) (why) and [roadmap.md](roadmap.md) (what is
next). What this page describes in prose, [structure.md](structure.md) draws: six figures
of the layers, the data flow, the contract, one stage, the backends and the output side.

## 1. What solprop is

solprop models ultrafast nonlinear pulse propagation: the unidirectional pulse propagation
equation (UPPE) for the full electric field or for a complex envelope, in a single spatial
mode, in many modes of a waveguide, and in free space (radially symmetric and full 3-D).
It began as a Python rewrite of [Luna.jl](https://github.com/LupoLab/Luna.jl): it keeps
Luna's scope and its most valuable design choices, and is designed around the following
(the user-facing differences are listed in [Luna and solprop](../guide/workflows/luna.md)):

- exponential treatment of the represented diagonal linear flow, with adaptive
  estimates for the integrated source ([stepper.md](stepper.md));
- absorbing boundaries defined as rates, with explicit and split-step treatments;
- a specified one-photon input-noise model normalised for dispersive media;
- field and envelope propagation with explicit supported combinations and limitations;
- polarisation as an ordinary array axis, at no cost when unused;
- a hot loop designed for threads and GPUs from the start, behind one seam;
- lean memory: bounded buffers, node chunking, no materialised stages;
- an in-process batch mode for many small or several large problems;
- a clean, fast-loading Python package that plays well with NumPy, SciPy, matplotlib and h5py.

**Priorities, in order:** accuracy and reliability; then as much CPU performance as can be
had; then GPU performance (ADR-020). Double precision by default: realistic modelling always
carries noise, so the stepper's job is efficiency at sufficient accuracy, not extended
precision. Single precision is an explicit per-run choice, `precision="single"`, under
SEP 33's contract (ADR-054): the state, stages, transforms and responses in binary32,
the controller, host scalars and propagator exponents in binary64, every narrowed table
with its own power-of-two exponent and the state at an exact scale. Mixed precision,
`precision="mixed"` (SEP 36, ADR-055), keeps everything that touches the state in
binary64 -- the state, every row stored in the interaction variable, the propagators,
the once-per-step boundary treatments and the norms -- and evaluates the source in
binary32 on the state narrowed once at the same exact scale. `numpy_ref`'s single and
mixed paths are the oracles of the two modes; `numba_cpu` computes in binary64 only, and
`native` until its build compiles the reduced modes (SEP 36 phases 2 and 3).

## 2. The model

In the frequency domain the propagation equation for every mode `m` (and polarisation
component `p`) is

```
∂z Ẽ_m(ω, z) = D_m(ω, z) Ẽ_m(ω, z) + N_m[Ẽ](ω, z)
```

- `D_m(ω, z) = −i(β_m(ω, z) − β₀(z) − β₁(z)(ω − ω₀)) − α_m(ω, z)/2 − α_abs(ω)/2` is the
  **linear operator**: dispersion and loss of the mode in a co-moving frame, plus the
  spectral absorbing collar as a rate. It is diagonal in ω, which is the one property the
  whole solver design rests on: `D(ω, z₁)` and `D(ω, z₂)` commute, so
  `exp(∫_{z₁}^{z₂} D dz')` *is* the linear flow, exactly. Which frame terms apply depends
  on the grid: an envelope grid removes `β₀` unless a third-harmonic term is present; a
  field grid keeps it, because a constant phase on the half-spectrum of a real field
  rotates its carrier-envelope phase, and measures the `β₁` term from zero frequency
  (`docs/guide/numerics/linear-operator.md`, "Which terms the frame removes").
- `N_m` is the **nonlinear term**: the projection onto mode `m` of the nonlinear
  polarisation evaluated in the time domain at points of the fibre cross-section (Kerr,
  Raman, plasma, and later χ⁽²⁾ and spectral-domain responses such as two-photon absorption),
  multiplied by the UPPE prefactor `−iω/(2ε₀c Re n_eff(ω))` (field) or its envelope
  equivalent, with the temporal absorbing collar as a rate. Only the real part of the
  index enters the prefactor: the loss belongs to `D`, and carrying it here would give the
  nonlinear term a spurious gain component.

Two grids: a **field grid** (real `E(t)`, half-spectrum state, r2c/c2r transforms, oversampled
fine grid for the cubic response) and an **envelope grid** (complex `A(t)` about `ω₀`,
full-spectrum state, c2c transforms, optional oversampling for third-harmonic terms). Both are
first class, with parity tests for shared physics on narrowband pulses. Supported
combinations are explicit: plasma supports scalar and Cartesian vector fields on
both grids, using a separately derived cycle-averaged envelope model. Explicitly
declared tensor Raman supports both grids on NumPy, Numba CPU and native;
unlabelled custom kernels and the default silica model remain scalar-only.
Envelope plasma with THG is refused. The native CPU/CUDA/HIP paths
have their own qualification. The [running guide](../guide/setup/running.md)
and [backends.md](backends.md) record the qualified scope.

**Optional atomic response** ([SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md)).
A scalar helium `TDSEGasResponse` owns
the gas nonlinearity. Its immutable specification becomes a `TDSEDescriptor`;
the driver lazily owns native plans and stable radial chunks and passes a narrow
host binding explicitly to NumPy. The optical march uses Adams. Atomic device
execution is independent of the optical backend. Numba and the native backend refuse this
response by name. The kernel schema includes separate polarisation and current
factors: current enters with `-1/(2 epsilon_0 c Re(n_eff))`, without division by
frequency. The provider's raw discrete shadows are subtracted before density
and calibration. Native retention subtracts only the first-order current and
applies its explicitly declared positive scale to the whole nonlinear current;
its polarisation contribution is zero. Its source is always gated and edge-free
([SEP 26](seps/archive/sep-0026-edge-free-tdse-source.md)): the backend multiplies the already subtracted remainder by its gate,
`grids.t_window_fine` or the atomic window's, before source projection/FFT, and
subtracts the gate's edge term `F[g' S]/(i omega)` on the active bins. The raw provider result, atomic
drive and separate in-stage field absorber are unchanged. This is a recorded
finite-window approximation, not a model of absorption, walk-off or phase mismatch.
Cubic-replacement policies also subtract the third-order current and supply their
separately specified polarisation; their sources are never gated.

The TDSE binding uses the shared Adams hooks and the step and plane policies of the
optical march. Every descriptor has one layout, `solprop.tdse-descriptor.v5`, which binds
the closure, model, window and grid collar geometry into its identity. The saved output
records the absolute launch, the TDSE identity, the outcome and any terminal checkpoint
beside the step limits and the dense and step-plane policies. The full-band sensitivity to
the window and the atomic cut error are unresolved
([TDSE accuracy and limitations](../guide/media/tdse.md#accuracy-and-limitations)). The
current versions are kernel schema 37 and saved-output schema 26
([kernel contract](kernel-contract.md)).

**haCC helium** ([SEP 20](seps/archive/sep-0020-hacc-helium-tdse.md)). The same route carries a second atomic model, the
two-electron haCC atom (`TDSEGasResponse.model="hacc"` with a `HaccBasis`), at a
fixed current scale of one. Its descriptor may carry an atomic window
whose gate replaces the grid collar as the source gate; the binding slices and
gates the drive to the window and pads the currents back to the full lattice, so
the seam's arrays keep their shapes. The weak-field consistency check lives in the
driver layer (`solprop.driver.tdse_consistency`), runs before the first propagation
call on the run's open binding, and is cached on disk; its policy never enters an
identity. An optional raw response archive (`output/tdse_archive.py`) records every provider evaluation in a separate append-only HDF5 file: the
driver opens its session before any atomic work, the binding writes each
evaluation's input before its first provider call and each raw result before
validation, and the reference backend reports each Adams trial's outcome to the
binding from its step record. It changes no numerical step and adds no call; the
optical file records the archive's run UUID and relative path.

Plan lifetime, allocation admission and attempted-call budgets
belong to the driver/binding; modal synthesis, P/J projection and spectral
assembly belong to the reference backend. See the [TDSE manual](../guide/media/tdse.md).

Geometry: mode-averaged propagation is the single-node case of multimode propagation (the
effective-area normalisation replaces the quadrature); the GNLSE is the mode-averaged
envelope equation with a **dispersion-specified mode** (β from Taylor coefficients or a
tabulated β(ω), γ or n₂ + A_eff, Raman fraction), so it is an ordinary problem rather than a
hack. Free-space propagation (radial and 3-D) uses the same equation with a transverse
spectral axis in place of the mode index; nothing in the layering below prevents it, and the
kernel layout carries the transverse columns in its mode axis (see §12).

## 3. Layers and their rules

| layer | contents | may import | performance rules |
|---|---|---|---|
| **P** physics / convenience | grids, materials, modes, profiles, pulses, noise, linear-operator assembly, response *definitions*, ionisation and Raman models, boundaries, normalisation, tools | numpy, scipy, standard library, other P modules | none: clarity first; never in a hot loop |
| **T** translation | turns P objects into validated kernel-ready arrays and the `KernelProblem` contract | P, numpy, scipy | may be expensive once per problem; results cached and validated |
| **B** backends | state buffers, transform plans, kernels, the inner z-loop, statistics kernels | numpy, numba, pyfftw; the native package in `native/` (Kokkos, Kokkos-FFT and nanobind); from P only the types and constants the seam names — never a physics object | all hot-path rules of [backends.md](backends.md); in-place and `out=` allowed here only |
| **D** driver | `Simulation`, run loop over save planes, batch runners, `prop_capillary`/`prop_gnlse`, output handlers, statistics specs | P, T, B (through the seam only), h5py | none |
| **X** processing / plotting | analysis of outputs, standard plots, design tools | numpy, scipy, matplotlib, D (outputs), T (reference tables and their identities, for `processing.modal`; the recorded plane measure, for `processing.mixtures` and `processing.planes`) — never B | none; may call backend statistics kernels through the D layer when a computation belongs on the device |

**Transverse rules are built in P.** `modes/quadrature.py` constructs its rules
numerically -- Laurie's Kronrod recursion, and the field-scale rules of
[SEP 19](seps/archive/sep-0019-field-scale-quadrature.md) by the Lanczos
process and the Golub–Welsch eigenvalue method, certified on construction -- once per
problem, and caches them. It is the one numerical construction kept in P: it yields
nodes and weights for T to fold into the transform, and no compiled code, per-stage
evaluation or hot loop enters P with it.

**The X → T edges.** `solprop.processing.modal` imports
`solprop.translate.modal_tables.ModalTransformTables` and
`solprop.translate.general.guide_identity`, and that is inherent to the output schema
rather than incidental. A general modal file records what its E/H reference **is** --
the table identity, the guide identity -- and not the reference's tens of megabytes,
so the reference is rebuilt in **T** by `general_modal_route` and read in **X**, which
checks it against the stored identity, path by path, before it synthesises anything.
**X** takes the record type and the identity functions, never an execution map, never a
`KernelProblem` and never a backend. Two further modules read what a file records the
same way. `processing.mixtures` and `processing.planes` rebuild a spatial plane's
measure from the record with `translate.flux.flux_from_record` and
`translate.spatial.axial_admittance` (SEP 34), and `processing.planes` reuses
`processing.modal`'s checked tables and evaluates a general basis Gram with
`translate.coupled.metric_at`. These edges are deliberate (SEP 31): the measure a file
records is rebuilt by the code that built it for the run, not by a second copy in **X**.
No other module of **X** imports **T**.

**Coupling crosses the layers without breaking them** (SEP 31). The field plane, the
input port, the remap and `couple` are **P**; the driver's `Simulation.input_port()`
asks **T** (`translate.coupling`) to fill the port from the translated problem -- its
metric, reference index, the scalar profiles of its modes (built in **P** from the
modes, `coupling.models.ScalarModeBasis`) or a general guide's E/H and contrast -- and
extraction is **X** (`processing.field_plane`, which builds the same profiles from a
`source=` simulation's modes), because **P** must not import **X**. A coupling computed
in **P** therefore never imports **T** or reads a file. The driver's incident-beam
launch calls `couple` and installs its launch through `with_input`.

Rules: imports flow P → T → B; B never builds a physics object — it does not know how to make
a mode or a gas, and imports P only for the grid, mode and profile types its statistics are
handed at construction, the constants, and the Raman drive form;
X never imports B directly. Every public module declares `__all__`. No `utils` modules:
organise by domain concept. Hand-rolled numerics live in T (`translate.chebyshev`) and B
only; everything else uses NumPy/SciPy/h5py/matplotlib. A value type in P may carry what T
designed and apply it with NumPy alone: the plasma column's
`nonlinear.ionisation.RunningIntegral` holds taps that `translate.plasma_integration`
fits and certifies, and filters and sums with them
([SEP 18](seps/archive/sep-0018-band-exact-plasma-integrals.md)).

## 4. Package layout

The public package is organised by domain.
[structure.md](structure.md) figure 1 lists the modules individually and is checked
against `src/solprop` by `tests/test_structure_figures.py`.

| Location under `src/solprop/` | Responsibility | Layer |
|---|---|---|
| `__init__.py`, `exceptions.py`, `constants.py` | Lazy public imports, package logging, exception hierarchy and SI constants | package / P |
| `grid.py`, `fourier.py`, `hankel.py`, `spatial.py`, `adiabatic.py` | Temporal and spatial grids (radial, the Cartesian plane, and `LineGrid`, its x axis uniform in y per unit width), Fourier conventions, raw Hankel transforms, spatial media, the axial-flux measure of a plane (`SpatialFlux`, SEP 34) and the adiabatic path certificate of a z-varying medium | P |
| `profiles.py`, `gasflow.py`, `pulses.py`, `noise.py`, `polarisation.py`, `optics.py`, `tools.py` | Longitudinal paths, steady gas flow in a hollow fibre (the taper- and temperature-aware, choked, regime-checked fill, its flow report and calculators), temporal/spatial inputs, noise, polarisation algebra, linear optical elements as transfer functions, and the soliton and phase-matching design calculators | P |
| `coupling/` | Coupling simulations through a coherent field at a reference plane (SEP 31): the field plane with its reference, geometry, metric, basis identity and provenance (`plane.py`), the input port of a receiving simulation and its bound launch (`port.py`), temporal remapping (`remap.py`), the coupling models, their scalar mode profiles and the recorded pair envelopes (`models.py`, `envelopes.py`), `couple` with its admission budget and report (`coupler.py`), and the synthesis of a guide's exit field (`outcouple.py`), and the transport core: the linear elements between planes (`elements.py`) and the ordered transport of a plane with its ledger (`transport.py`); NumPy, SciPy and other P modules only | P |
| `physdata/` | Species, mixtures, equations of state, transport properties (viscosity, sound speed, mean free path), optical material data, metals, nonlinear and Raman data | P |
| `modes/` | Mode protocols and derived quantities; capillary, antiresonant, tabulated, step-index, rectangular and planar (slab, uniform in y, per unit width) modes, with `PlanarGrid` for the planar guide resolved in y; shared gas fill, quadrature and collections; full-vector modal records, material laws and regions, bound step-index and radial graded-index providers, and the guide that pairs a provider with a mode collection, and the connection of a basis that moves along z | P |
| `linear.py`, `boundaries.py`, `nonlinear/` | Linear assembly, absorbing rates, response definitions, ionisation, Raman, normalisation and the GNLSE mapping | P |
| `translate/` | Validated `KernelProblem`; parameter expansions and z tables; modal/radial/Cartesian transforms and the hybrid planar one (slab modes in x × FFT in y, `translate/planar.py`); dense tracked modal tables, the general modal route and the coupled-flow and metric records (`translate/general.py`, `translate/coupled.py`); the spatial plane measure (`translate/flux.py`, the `FluxPlan` of schema 35); the persisted frame integral and the filled input port of a problem's entrance (`translate/coupling.py`); response tables, masks and stage bundles | T |
| `backends/` | Registry and protocols, shared step and batch drivers, scalar bodies, spatial and population statistics | B |
| `backends/numpy_ref/`, `backends/numba_cpu/` | Reference and compiled CPU execution | B |
| `stepper/`, `log.py`, `stats.py` | Tableaux, controllers, schedules, progress reporters and statistics specifications | D |
| `driver/`, `driver/executors/`, `scan.py` | Simulation and convenience interfaces, scan/batch orchestration, Local/Pool/Queue/Slurm execution and scan CLI | D |
| `output/` | Memory and HDF5 outputs, provenance/resume, scan storage and artifacts | D |
| `processing/`, `plotting/`, `sol.py` | Field, spectrum, time, coherence, compression, mixture and scan analysis; a saved plane read back as a field plane (`processing/planes.py`); propagation/statistics plots and spectrograms; the interactive session namespace, the one module whose import has side effects | X |

`solprop-native` is the backend's second distribution, built from `native/` in this
repository at solprop's own version and installed through the `native` extra. It depends
on the versioned seam; solprop does not depend on its native build.

`driver/scaffold.py` is an independent fixed-step IP-RK4 reference used by tests and
its tutorial; it is not the production stepper. It propagates a `GnlseProblem`
directly, without a contract, a backend or an output handler, which is what makes it a
cross-check of the reference backend. `processing/windows.py` is a candidate
module, drawn dashed in figure 1, to be written only for an agreed use case.
`_display.py` is private: the formatter behind every class's printed block, imported by
any layer and importing none. `_numba_cache.py` is private in the same way: numba's
`njit` with a disk cache that concurrent processes cannot cross-wire, through which
every kernel in **T**, **D** and **B** compiles ([backends.md §3.2](backends.md)).

The layout does not mirror Luna's module tree: what `NonlinearRHS.jl` does spans
response normalisation, translation and backend passes; `Maths.jl` mostly becomes SciPy
calls; the cumulative-operator stepper takes the place of `RK45.jl`; scans use the
driver and executor layers above.

## 5. Data flow of one run

```
user script / prop_capillary(...)                                  (D)
   │  builds physics objects: Grid, Modes, ZProfiles, Responses, Pulses, Noise,
   │  Boundaries, StepperSettings, Output
   ▼
Simulation(...)                                                    (D)
   │  .translate() → KernelProblem  (validated arrays, schema_version)   (T)
   │  .backend = backends.select(name, problem)                     (B)
   ▼
run loop over save planes and stats periods                        (D)
   │  backend.set_initial(field)                                    host → device once
   │  for z_save in planes: backend.propagate_to(z_save)            shared step loop
   │                        obs = backend.observe()                 small host arrays
   │                        stats = backend.stats(spec)             one fused pass
   │                        output.save(z_save, obs, stats)
   ▼
MemoryOutput / HDF5Output  ──►  processing / plotting               (X)
```

The inner loop (many adaptive or scheduled steps between two save planes) is the
shared Python `StepDriver`. NumPy evaluates reference passes, Numba launches compiled
kernels and library transforms, and the native backend executes each trial through one native
call. The outer driver visits save planes. Device arrays never cross the seam;
`observe()` returns host arrays of the field at the plane
and the step history; `stats()` returns a small dictionary of host scalars/vectors.

The outer loop is `solprop.driver.run.run_problem`, which takes a translated problem, a
backend that already holds its initial state, and an output handler. Nothing in it is
specific to a `Simulation`; it is the piece a batch runner or a scan calls. What the
driver does not do is as much a part of the design as what it does:

- It **never conjugates a spectrum**. The backend works in the FFT convention, the output
  layer converts on the way to disk, and those are the only two places a sign changes.
- It **never touches the state between planes**. Everything between two save planes
  happens inside the backend's march.
- It **never computes physics**. The mapping from `gamma` to a susceptibility is
  `solprop.nonlinear.gnlse`, the statistics are `solprop.stats`, the storage is
  `solprop.output`; the driver is the wiring.

A general modal guide adds a chain of host-side records in front of the
`KernelProblem` ([vector fields and projection](../guide/waveguides/modal-projection.md)):

```text
ModalProvider  ->  DenseModalTable  ->  ModalTransformTables  ->  KernelProblem
 E, H, beta at      tracked knots in     S and Q on the run's      the three
 nodes, one         frequency, gated     coarse grid, the dense    general-modal
 (omega, z)         on its sampled       E/H and the node rule     leaves and the
                    Gram                                           diagonal operator
```

## 6. The physics layer

### 6.1 Grids (`grid.py`, `fourier.py`)

`FieldGrid` and `EnvelopeGrid` implement one protocol: time axis `t` (`n_t` points, spacing
`δt`), coarse frequency axis `omega` (`n_omega`: `n_t/2+1` for the field grid, `n_t` for the
envelope grid in FFT order), the oversampled fine axes `t_fine`/`omega_fine` (factor `ffac`:
6 for a cubic field response with third-harmonic content, 4 for `|E|²E`, 1 or 2 for
envelopes), the in-band mask (physical frequencies inside the wavelength limits), the
spectral and temporal collar profiles, and the reference frequency `ω₀`. Sizes are powers of
two or 2-3-5-smooth on request. The invariants between coarse and fine grids (Luna
`Grid.jl:107–108, 244–250`) are asserted in the constructors and tested. `fourier.py` owns the
sign convention ([conventions.md](conventions.md)), the r2c/c2c helpers, the analytic-signal
and oversampling operations used outside the hot loop, and the crop/pad scaling factors that
the translation layer hands to kernels.

### 6.2 Modes (`modes/`) and profiles (`profiles.py`)

The `Mode` protocol: `neff(omega, z)` (complex), `dneff_domega(omega, z)` (analytic where the
model allows; a validated finite-difference fallback otherwise, ADR-006), `field(points, z)`
(transverse field components at points), `dimlimits(z)`, `aeff(z)`, and the traits
`z_constant`, `scale_invariant`, `azimuthal_order`. Derived quantities (β, α, β₁, β₂, ZDW,
loss length) are protocol-level functions. `MarcatiliMode` (hollow capillary, full and reduced
models, loss as a Boolean, scale factor or callable of ω) shares gas-fill machinery
with the tabulated and rectangular hollow modes; the core index is a
`ZProfile`-aware object so gradients and tapers are ordinary cases. `n_eff − 1` is computed
directly, never as `(1 + x) − 1`.

`ZProfile` (`profiles.py`) represents a scalar function of z — gas density, core radius,
pre-ionisation fraction — with its breakpoints (stage boundaries are hard kinks), analytic
derivative where available, and evaluation on arrays. It also declares its jumps
(a subset of the breakpoints) and encloses its variation over any interval as a
`PathVariation`: pieces with one-sided limits, value bounds and a variation bound,
exact on pieces certified monotone. An arbitrary callable must declare its turning
points to have one; sampling is not an enclosure. The adiabatic certificate
(`adiabatic.py`, SEP 34 §4.5) builds on it. Pressure gradients use the steady-flow
profile `p(z) = sqrt(p₀² + (z/L)(p₁² − p₀²))`; fills must not start or end at exactly zero
pressure (a vacuum-terminated fill caps every scheme at order ~3, [stepper.md](stepper.md)
§10; the interface substitutes a small floor and logs it). A steady-flow fill of a
hollow fibre (`gasflow.py`) follows the bore and chokes at its vacuum end (ADR-032).

`modes/quadrature.py` builds the fixed transverse rules (Gauss–Legendre radial with Kronrod
extension for the embedded estimate; periodic trapezoid in θ, exact for trigonometric
polynomials of degree ≤ n_θ − 1, with the `4·h_max + 1` warning; Cartesian Gauss for
rectangular modes; a one-dimensional Gauss–Kronrod rule in x with unit y weight for
slab modes; for one Marcatili HE₁₁ mode, Gauss and Gauss–Radau rules in the field scale
$\eta = J_0(u_{01}r/a)$ that carry radial *area* weights) and the
synthesis/projection matrices with weights folded in (Luna `TransModalFixed` §3.3–3.4;
ADR-010). `radial_rule` selects the family; a one-node rule is an ordinary modal
transform, and only `mode_averaged_transform` makes `Transform.kind == "mode_averaged"`.

### 6.3 Materials (`physdata/`)

solprop keeps no material database of its own ([SEP 3](seps/archive/sep-0003-optmat-materials.md)).
Every built-in material is optmat's, adapted at the boundary: a gas's polarizability
(`OptmatSusceptibility`), equation of state and transport (`OptmatGas`) and electronic
Kerr model at the carrier (`OptmatKerr`), its first ionisation energy, and the solids'
complex indices (`OptmatGlass`) and fused silica's nonlinear bundle (`SilicaBundle`),
each with its exact derivative, its qualified intervals, its status and an identity that
the run's material manifest records. A `MaterialSelection` names optmat's models, and
the solid factories' `model=` names a solid's. What `physdata/` keeps is solprop's physics around the data
— linear density scaling, the mixture rule (`mixtures.py`), the χ⁽³⁾/n₂ conversions — and
the types a caller's own data take: the gas formula types (`sellmeier.py`), a custom
`Species` with `TabulatedSusceptibility` or `IdealGas`, `SellmeierGlass`,
`TabulatedGlass` and `CompositeGlass`, and the molecular Raman constants of
`raman_types.py`. Air's molar mass is read from CoolProp through `pyfluids`, the one
direct use left (see [dependencies.md](dependencies.md)).

### 6.4 Pulses, noise, optics

`pulses.py` builds initial fields on either grid (Gaussian, sech, custom shape, measured
data, CW), scaled to energy or peak power, with Taylor spectral phase; `noise.py` draws
shot-noise realisations for the input model and the modified model (one photon per mode per
frequency bin, amplitudes per [conventions.md](conventions.md), masked to the in-band
region, seeded with `np.random.default_rng`); `optics.py` applies linear optical
elements (Taylor phase, bulk material, chirped mirrors, grating and prism pairs, and
chains of them) as spectral transfer functions to a field outside the propagation.

### 6.5 Linear operator assembly (`linear.py`) and boundaries (`boundaries.py`)

`linear.py` evaluates `D_m(ω, z)` from the modes and the frame (β₀(z), β₁(z) from analytic
derivatives at ω₀) and folds in the spectral absorber rate; it knows nothing about Chebyshev
tables. For Marcatili modes with `ZProfile` core indices it exposes the three-scalar structure
(a(z), ρ(z), β₁(z)) that lets the translation layer sample D cheaply at many z (the idea behind
Luna's `MarcatiliLinop`). `boundaries.py` defines the collar geometry (spectral edges beyond
the wavelength limits; temporal collar as a fraction of the window, guaranteed non-degenerate)
and converts profiles to rates with a reference length `ℓ`: `α = −2 log(W)/ℓ` for the
spectral collar and the bounded `12.6·(1 − W)/ℓ` for the temporal one (ADR-007).

### 6.6 Responses (`nonlinear/`)

There is one definition per response, not one per layout or per device: a frozen
dataclass that carries its coefficients, declares four traits (`ResponseTraits`,
ADR-011) and provides a NumPy `evaluate` that is the **reference implementation** — the
oracle every backend kernel is tested against, written for clarity rather than speed.

| trait | values | Kerr | dispersive Kerr | plasma | envelope plasma | Raman | silica tensor Raman |
|---|---|---|---|---|---|---|---|
| `domain` | `pointwise`, `memory` (sequential in t), `convolution`, `spectral` | `pointwise` | `convolution` | `memory` | `memory` | `convolution` | `convolution` |
| `grids` | `field`, `envelope`, both | both | both | `field` | `envelope` | both | `field` |
| `polarisation` | `scalar`, `vector_isotropic`, `tensor` | `vector_isotropic` | `vector_isotropic` | `vector_isotropic` | `vector_isotropic` | `tensor`, or `scalar` for a scalar-only declaration | `tensor` |
| `z_profiles` | names of the scalar paths it consumes | `density` | `density` | `density`, `preionisation` | `density`, `preionisation` | `density` | none |

A response bound to a mixture component names its paths with the label, `density:N2`
([labelled mixture responses](../guide/media/nonlinear.md#labelled-mixture-responses)).
The translation layer reads the traits and compiles the declared set into one fused
time-domain traversal per column. Batched variants do not exist as types: batching is the
kernel layout. Adding a response means adding a definition, a scalar body and a test.
Envelope plasma uses the same traits but exposes `sources` rather than one `evaluate`
polarisation: its temporal population scan produces a Drude drive and a directional loss
current, which keep their own frequency factors through projection and the temporal
transforms ([envelope plasma sources](../guide/media/ionisation.md#envelope-plasma-sources)).

`ionisation.py` holds the rate protocol (`IonisationRate`, `EvaluableRate`), its
optical-cycle conventions, the energy binding of the plasma loss, the rate ceiling
`MAX_RATE` (clipping the rate so a diverged step gives no NaN) and the Keldysh and
barrier-suppression diagnostics, and no rate formula; `optmat_rates.py` adapts optmat's
rate models to the protocol (`OptmatRate`); `raman.py` the response models
(damped oscillator, rotational/vibrational models of a supplied species, silica's
sampled bundle shape and tensor response); `normalisation.py` the UPPE and
envelope prefactors and the effective-area normalisation for mode-averaged runs; `gnlse.py`
the mapping from (γ or n₂/A_eff, β coefficients, f_R, shock) to a mode-averaged envelope
problem. `optmat_raman.py` adapts optmat's gas Raman responses onto the same terms:
one isotropic and one rank-two row per damping group, its decay law optmat's own, and a
mixture's one response driven by the summed density of its labels.

`nonlinear/envelope_plasma.py` defines an independent Cartesian-cycle oracle for
envelope plasma. Translation compiles instantaneous-rate tables, validated angular
rules and separate Drude/loss frequency factors. Backends recompute each channel's
cycle moments and temporal population for every RHS input, then project and
transform both current streams before adding their equivalent polarisations to
the ordinary source. The [envelope plasma manual](../guide/media/envelope-plasma.md)
derives the directional loss current, source signs and averaging limits; the
[kernel contract](kernel-contract.md)
defines the numeric data crossing into a backend.

`RamanResponse` stores immutable `RamanTensorTerm` declarations. Translation
preserves each term's density and decay law, then supplies numeric trace and
traceless weights. The local Cartesian response retains delayed off-diagonal
coherence. Scalar and isotropic-only vector responses use one real history;
rank-two vector responses use three histories and two effective kernel spectra.
The [Raman manual](../guide/media/raman.md) derives the contraction and its scalar and
circular limits. Circular launch still uses the existing real modal basis.

## 7. The translation layer (`translate/`)

The translation layer produces the `KernelProblem` ([kernel-contract.md](kernel-contract.md)):
plain arrays with fixed dtypes, shapes and axis names, validated (finite, contiguous, sizes
consistent), immutable, versioned. What it computes, once per problem:

| product | from | representation |
|---|---|---|
| Φ table | `linear.py` + `ZProfile`s | piecewise Chebyshev in z, coefficients `(n_intervals, n_deg, n_batch_or_1, n_modes, n_pol, n_omega)` as in `kernel-contract.md` §3, breakpoints, hard kinks; constant-D fast path (degree 1: one interval, or one per stretch between the kinks of a response path) |
| unitary phase for noise | Im Φ | same table, imaginary part |
| quadrature | `modes/quadrature.py` | nodes, weights, coarse weights, synthesis and projection matrices; taper scaling rule |
| response spec | `nonlinear/responses.py` | compiled ordering of the fused pass, coefficients, ionisation-rate spline tables (log-spaced, converged, clip limits), Raman kernels on the doubled grid in the frequency domain, normalisation prefactors |
| masks and rates | `grid.py`, `boundaries.py` | in-band mask, spectral rate (already in D), temporal rate, hard band mask, crop/pad scaling; analytic fine-grid collar derivative and steady-current certificate for field plasma |
| initial state and noise realisation | `pulses.py`, `noise.py` | canonical layout |
| stepper settings | `stepper/` | tableau, tolerances, controller parameters, schedule, save planes, caps |
| statistics requests | `stats.py` | which statistics, which windows |
| spatial plane measure | `spatial.py`, `adiabatic.py`, `translate/flux.py` | the `FluxPlan`: separable coefficients whose root gives each column's axial admittance, the material square's z table for a varying radial medium, and its adiabatic admission (SEP 34) |
| batch parameters | `driver/batch.py` | per-member scalars or per-member table indices |

Validation is not optional: a backend calls `problem.validate()` and refuses anything
inconsistent. Tables are cached on the `Simulation` and rebuilt only when a physics object
changes.

## 8. The solver, in one page

Details and evidence are in [stepper.md](stepper.md). The anatomy of one stage `i` of the
interaction-picture DOPRI5 step from anchor `z_n`, for a multimode field-grid problem, as the
backend executes it (each numbered line is one pass over its data; the FFT counts are the
cost metric):

1. **stage combine + to lab frame + noise + pad**: `u = exp(ΔΦ_i)·(y + h Σ_j a_ij k_j) + N(z_n + c_i h)`, written zero-padded into the fine-grid spectral buffer — one fused pass over `(n_modes, n_pol, n_omega)`.
2. **batched inverse FFT** (fine grid), `n_modes·n_pol` columns.
3. **synthesis GEMM** onto a chunk of quadrature nodes: `(nodes·n_pol) × n_modes` by `n_modes × n_to`.
4. **fused response pass** per column: `|E|²` → ionisation rate (table) → two plasma running integrals (exposure and free current) → physical current `J`. Kerr and Raman accumulate separate non-plasma polarisation. Raman adds one or three real histories per physical node, transformed in a doubled-grid batch. The column writes three complete streams: collared non-plasma polarisation `gP`, `gJ` and `g′J`.
5. **projection GEMM** for each stream back onto modes, accumulated over chunks.
6. **batched forward FFT** (fine grid) for each stream, followed by `DFT[gP] + DFT[gJ]/(iω) − DFT[g′J]/(iω)²` on active bins. Inactive bins are exactly zero and no division is evaluated there.
7. **crop + prefactor + spectral window + band mask + back to the interaction frame + accumulate**: `k_i = exp(−ΔΦ_i)·p(ω)·W(ω)·crop(source)`, `acc_sol += b_i k_i`, `acc_err += e_i k_i` — one fused pass; on the last stage the weak error norm is reduced in the same pass (a two-stage reduction, its order free within its rounding bound).

Without field plasma there is one projected source stream. The held-current source adds
two temporal transforms to that modal route. Radial propagation instead transforms and
combines the three streams per node, crops, then projects once in frequency. Cartesian
and line propagation combine before the transverse FFT; planar propagation projects
in x before the temporal transforms and combines before the y transform. These orders
and their buffer budgets are part of [SEP 27](seps/archive/sep-0027-held-plasma-current.md).

These are passes over data, not launches.

Per step: six new nonlinear evaluations with FSAL (seven without it), with the modal
batched transforms above plus the doubled-grid pairs for Raman, 6 `exp(ΔΦ_i)`
evaluations (or 5 with FSAL and memoisation), one controller decision (lagged, with rejection allowed by default;
`reject_free` is explicit), one accept pass (`y ← exp(ΔΦ)·(y + h·acc_sol)` — the stages live in the interaction variable anchored at the step start, so the whole sum is carried by one exponential, and the embedded difference takes the same factor — rebase the FSAL stage, apply
the temporal absorber per ADR-025 when active), one hard-mask multiply. Steps land on save
planes and hard kinks (ADR-009). Where the linear flow couples channels (§16), each
`exp(±ΔΦ_i)` multiply becomes the action of a small per-bin matrix on the mode axis, and a
frozen-residual section adds its explicit residual to the stage source
([stepper.md](stepper.md) §2).

**Memory budget** (field-sized unit `F = n_batch·n_modes·n_pol·n_omega·16 B`): state `y`,
stages `k₁…k₆`, two accumulators, up to five propagator factors (tabulated) or none
(recomputed) — 9–14 F by design, more when both propagator signs are tabulated
(`recompute` halves that share); fine-grid spectral
and modal time-domain buffers — 2–3 F·ffac;
node-chunk buffers `(chunk, n_pol, n_to)` for `E` and `P`, one or three doubled
real Raman histories per node and their spectra/convolved values — bounded by
the chunk size, not by the node count. Field plasma also charges its current and
product streams, projected streams, transforms and provider plans. Planar node
admission reserves at least one complete x row. Effective kernels and provider workspace
are counted separately in the backend ledger. Examples:

| case | F | solver registers | node buffers (chunk 16) | Luna's node buffers (all nodes) |
|---|---|---|---|---|
| RDW VUV: n_t 4096, 4 modes | 0.13 MB | ~2 MB | ~3 MB | ~8 MB |
| CtC: n_t 32768, 6 modes, Raman | 1.6 MB | ~20 MB | ~45 MB | ~210 MB |

**The low-storage march** ([SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md),
ADR-050) is RK4 in the midpoint interaction picture, replay-only and accepted in place:
each slope is folded into a running sum as it is produced, so a step holds the state, an
accumulator and a stage register beside the evaluation's scratch, plus the half-step flow
`exp(hD/2)` when tabulated — four or five `F` where the Runge–Kutta march's ledgers hold
13 to 16. It makes four source evaluations a step and keeps no estimate.

A backend exposes a `memory_ledger()` naming every buffer and its size; the ledger is tested.
The table shows the design's scale, not a per-configuration allocation promise: read the
constructed backend's ledger for its actual buffers.

## 9. Backends, in one page

Details in [backends.md](backends.md). The seam is at the level of *operations on the
propagated state*, never at the array-namespace level
(`solprop.backends.protocol.Backend`; signatures abbreviated):

```python
class Backend(Protocol):
    name: str
    max_batch: int

    def set_initial(self, field, noise, *, z: float = 0.0) -> None: ...
    def set_progress(self, reporter) -> None: ...  # an observer, never an input
    def propagate_to(self, z: float) -> StepSummary: ...  # shared step loop
    def observe(self, *, field: bool = True) -> Observation: ...  # host arrays only
    def stats(self, spec) -> dict[str, np.ndarray]: ...  # one fused pass
    def metadata(self) -> dict[str, object]: ...  # what the execution was
    def restore_call_ledgers(self, logical, atomic=None) -> None: ...  # after a resume
    def available_device_memory(self) -> int | None: ...
    def memory_ledger(self) -> dict[str, int]: ...
    def close(self) -> None: ...
```

- `numpy_ref`: the readable oracle. Plain NumPy + `scipy.fft`, no fusion, no in-place
  tricks, one function per pass. Every other backend is validated against it.
- `numba_cpu`: fused stage kernels (`@njit(cache=True, nogil=True)` through
  `solprop._numba_cache.njit`), pyfftw plans executed in place on preallocated aligned
  buffers (`FFTW_MEASURE` + wisdom on a quiet machine), node-parallel kernels and controlled
  FFTW/BLAS threads, with two-stage reductions and the shared Python step loop. Its
  floating-point licence is [SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)'s:
  contraction by default, strict opt-outs for the protected helpers (stable increments,
  small-argument functions, defects, compensated sums), reassociation only on the loops
  of a kernel that asks for it, and no other fast-math flag
  ([backends.md §3.2](backends.md)), applied by the `njit` wrapper. Fixed fibre batches have a compiled accept
  update over members; other numerical work remains member-owned. An all-compiled march
  is a deferred experiment, subject to a measured dispatch-cost case.
- `native` (the `solprop-native` distribution, `native/`): C++20/Kokkos kernels and persistent Kokkos-FFT plans
  behind nanobind. It runs mode-averaged and multimode scalar/two-component problems,
  Kerr, plasma and declared tensor Raman, weighted control, embedded quadrature and
  backend-owned statistics with one native call per trial, and on its CPU build also the
  general modal route and the coupled flow. Raman uses doubled-grid plans, actual-width
  tails and memoisation by decay rates and mixture amplitude densities. KokkosKernels
  supplies the portable GEMM seam; measured local builds may dispatch eligible whole
  transforms to Accelerate, OpenBLAS or cuBLAS. The CUDA build uses time-parallel
  response teams, deterministic plasma scans and device reductions and is qualified on
  data-centre NVIDIA GPUs; the HIP build is qualified on AMD Instinct GPUs, with the HIP
  sanitizer run still outstanding. Fixed-schedule fibre batching is implemented through
  `select_batch`. The private state is exposed to the shared driver only through its
  protected state type and the `_observed_field` host-copy hook.
- JAX: evaluated in a measured comparison (ADR-002); not a planned backend unless the
  data reverse the decision.

[backends.md §9](backends.md) records which route each backend admits and where each is
qualified, and §11 its current limits.

## 10. Output and statistics

`MemoryOutput` (everything in RAM: fast, small problems, batch scans) and `HDF5Output`
(streaming to disk, chunked per saved plane, optional compression, cache/resume) implement one
save protocol and one metadata schema (`output/schema.py`): grid, modes, simulation type,
the full configuration that built the `Simulation` (serialised physics objects), package and
dependency versions, git provenance of the running code when available, the noise seed and
realisation, the stepper settings and the step schedule actually taken (z, h, error norm,
rejections, wall time), the statistics, and the rows a periodic subset of them recorded
between planes. Fields are stored in the canonical layout
`(n_z, n_modes, n_pol, n_omega)` (batch: leading `n_batch`), in the **optics** sign convention
with the attribute `spectral_convention = "optics"` (ADR-017: the output layer conjugates on
save and on resume); processing accessors squeeze trivial axes and accept `convention="fft"`.
A TDSE run may also write its raw provider evaluations to a separate archive
(`output/tdse_archive.py`, schema `solprop.tdse-archive.v1`), which the optical record
references and never needs ([the archive page](../guide/media/tdse.md)).

`output/snapshots.py` coordinates optional immutable disk snapshots at the existing
accepted-step commit boundary. It persists retained march state and execution facts,
suspends the optical and optional archive handles while their advisory writer locks
remain held, copies and verifies the pair, publishes a manifest last, then reopens
the same writers. Backend and atomic resources are untouched; no additional observe,
landing plane or response evaluation occurs. The store bounds completed and incomplete
generations and leaves transport/retirement to an external controller. The manifest
has its own version, outside the scientific configuration and kernel schema. See the
[manual](../guide/results/output.md#periodic-snapshots-while-computing) for admission,
restoration, ownership and interruption rules.

Statistics (energy per mode and window, peak power and intensity, FWHM, electron density and
peak ionisation rate, density and pressure, core radius, ZDW, the quadrature error estimate,
`ω₀`) belong to saved planes (ADR-012). They are computed by the backend in one fused pass with
two-stage reductions and returned as small host arrays; an optional **periodic** mode
computes a subset every N accepted steps for step-level diagnostics.

The periodic mode is the shared step loop's, not a backend's: after every N-th accepted
step the march calls the backend's own `stats` with the subset and appends one row, which
`observe` drains into `Observation.step_stats` at each plane, so there is one
implementation of every statistic ([backends.md §1](backends.md#1-the-seam), "The
statistics channel between planes"). The catalogue's `between_planes` trait says which
statistics may be in a subset, and `StatsSpec` refuses the rest by name — a root search or
a mode-profile rebuild per accepted step is what cost Luna a fifth of its step. The native backend
reduces and packs the statistics without observing the complete field. The rows are stored
beside the step history and in the same ragged shape.

## 11. Scans and batch mode

[Scans](scans.md) run above the backend seam. `Scan` gives each point a stable global
index, parameters and seed. `run_scan` selects an executor; `Local` is the API default,
`Pool` uses spawned processes for independent CPU runs, and `Queue`/`Slurm` provide
durable distributed execution. A controller owns the collected HDF5 scan file.

Fixed-schedule fibre batches use `BatchBackend` and `select_batch`.
A batch adopts independently translated and loaded member backends, each with its
own tables, state and noise; it advances them under one compatible explicit schedule.
The accepted update is fused on compiled backends, while response and transform work
remains member-owned. `observe_members` preserves each member's history and statistics.
Admission uses memory ledgers and a free-device-memory query with headroom.

Every backend has exact member-versus-singleton identity checks under the same
schedule. That does not imply a throughput improvement on every workload: whether a
batch beats independent runs depends on the problem size and the device. Adaptive
lockstep is not implemented. A group-level failure fails every member of that group;
recovering the surviving members is an open question ([roadmap](roadmap.md)).

Independent simulations use spawned processes, normally one scientific thread per
worker, rather than threads, because thread pools, FFTW plans and the OpenMP runtime are
process-global (ADR-013). Each member may use its own adaptive schedule outside a fixed
batch.

## 12. Parallelism and the GPU model

Where parallelism lives at each scale. Construction-time pool choices use measured
size rules with explicit overrides ([backends.md §3.4](backends.md)); independent
process counts belong to the scan executor:

| problem | inside one step | across steps | across problems |
|---|---|---|---|
| small single | serial kernels and small transform pools by measured size rules | sequential | spawned processes |
| large single | `prange` over node chunks; FFTW/BLAS threads on batched plans | sequential | processes |
| compatible fibre problems on GPU | member-owned stage work; fused accepted updates | fixed lockstep | backend batch |
| large on GPU | one thread per element | sequential | one problem per device |

The plasma cumulative integral is sequential in value but not necessarily in execution. On
the CPU it is the inner loop of the fused pass. The CUDA path uses one team per node: rate
evaluation is time-parallel and a fixed deterministic scan tree forms exposure, fraction and
currents before the response finish. Response teams are 256 below 8,192 time samples and 512
at and above it; deterministic reduction blocks are 512. Fixed batches are
implemented; their measured crossover depends on workload
([backends.md](backends.md)). Public selection prefers the installed native backend,
otherwise Numba CPU (ADR-029); the installed native build determines its execution space.

## 13. Memory discipline

- Every buffer is named, preallocated at backend construction and listed in `memory_ledger()`.
- The Runge–Kutta march holds every stage row its tableau and dense output read, and
  running accumulators for the solution and the error estimate; the low-storage march
  folds each slope into its registers as it is produced (ADR-009 as amended by ADR-050).
- Space-domain work runs over node chunks; the chunk size is chosen from a memory budget
  (default: the whole node set when it fits a per-problem budget, else the largest chunk that
  does).
- `HDF5Output` never holds more than the current plane, in datasets preallocated for the
  planned planes with an `n_saved` count; `MemoryOutput` is allocated for the planned
  planes, grows by doubling past them, and is charged by the memory admission.
- A run keeps its fields on every plane, the named planes or none
  (`FieldRetention`,
  [SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)): a plane without its field is not copied, converted or
  stored, the stores allocate field storage for the kept planes only, and the admission
  charges those alone (its `output` group). The backend is asked for the field only on
  the kept planes: `StepDriver.observe(field=False)` transfers nothing from a device,
  scans nothing for finiteness, skips the collar diagnostic (counted in the execution
  record) and forms no noise-phase volume, so a field-free plane costs its statistics
  alone. Under the low-storage march a field-free plane holds one state volume so that a
  run stopped in flight can resume (`held_plane`, in the ledger and as its own admission
  group; on the native backend in the memory space).
- The Cartesian and planar collar diagnostic reconstructs the physical field by block of
  frequency, at most a node chunk's worth of samples at a time, never the whole volume.
- A `SimulationRunner` keeps one backend's buffers and plans across many inputs, and is
  admitted once; `StepDriver.set_initial` restores everything a march reads, so a reused
  backend repeats a fresh one's run. An input array of more than 2²⁰ elements is
  recorded by its SHA-256 digest, not inline, so the run record stays small.
- Φ tables are shared where possible; the constant-D fast path stores one array.
- Fine-grid buffers use the smallest `ffac` the declared responses need.

## 14. Logging and progress

`logging.getLogger(__name__)` per module; the package installs a `NullHandler`;
`solprop.configure_logging(level, file)` for scripts. Debug logs are plentiful in P, T and D
(translation summaries, table sizes, schedule statistics, memory ledger) and guarded with
`logger.isEnabledFor(logging.DEBUG)` when formatting costs anything. Nothing logs inside
compiled kernels or per-stage code. Progress is reported by the driver through one
`ProgressReporter` protocol (`solprop/log.py`): `started` once, `step` from inside the step
loop on every attempted step, `plane` per save plane and `finished` at the end. The default
`LogReporter` turns those into three kinds of INFO line -- a periodic one on a wall-clock
period (`progress_period`, 10 s, `0` to disable) carrying position, counts, current step,
elapsed and an ETA; one per plane with the step count and mean h; and a closing one with the
achieved-against-requested tolerance ratio and how far off the run's own ETAs were. The
period is wall time, not a step count, so the log volume of a run does not depend on how fast
its steps are. Where a person is watching -- standard error a terminal, or a Jupyter
kernel -- the default is instead a `BarReporter`: a `LogReporter` whose
periodic event redraws a `tqdm` bar carrying the same content and the same measured ETA,
with the plane line demoted to `DEBUG`; redirected or captured output always gets the
lines. A reporter that implements `interrupted` (`InterruptibleReporter`) is told when
the loop raises, before the error propagates, so a bar can give the terminal back.

## 15. Extension points

- **Modes:** implement the `Mode` protocol and declare the profile geometry, frequency
  support, derivatives and z dependence. Capillary, antiresonant, tabulated,
  step-index and rectangular implementations provide current examples. A full-vector
  E/H source implements the separate `ModalProvider` protocol instead, declaring its
  material regions, its domains and a serializable identity; the bound step-index and
  radial graded-index providers are the current examples. A
  cross-section is `polar`, `cartesian` or `slab`; a new one needs a `quadrature_rule`
  branch, `taper_scale`'s scaling law and `on_axis_rule`'s node. A mode uniform in one
  transverse direction declares `transverse_measure = "per_unit_width"`, which the run
  records and processing and plotting follow for units; the kernel never sees it.
  A grid uniform in y (`LineGrid`) is the same measure, recorded from the
  free geometry's single y row. A spatial kind whose columns are not its nodes
  (`"slab"`) says so through `SpatialGeometry.n_columns`; its transform
  carries the x half of the map in `synthesis` and `projection` and the y half as FFT axes.
- **Materials and responses:** species and mixtures carry optical and nonlinear data;
  the translation layer compiles supported Kerr, scalar/tensor Raman and
  field/envelope plasma terms.
  A new tensor or spectral response needs a written design with its physical
  derivation, grid/backend scope and contract changes, and independent tests.
- **Spatial propagation:** radial and Cartesian transforms already share the seam.
  Additional geometry or varying Cartesian media are separate extensions to qualify
  against the existing spatial contracts.
- **Execution:** scans separate point identity and output collection from the executor.
  New execution policies must preserve that identity and be measured on complete jobs.

The [roadmap](roadmap.md) records selected work and optional extensions.

## 16. Coupled linear flows, and non-goals

**The linear operator need not be diagonal in the modes.** Where channels couple
linearly — a capillary's coherent wall, the taper coupling of a moving multimode guide
(the connection of a basis that moves along z), a declared unitary basis of a split
block on the general modal route — the problem carries the optional
`linear.coupled_flow` leaf (`translate/coupled.py`) beside the diagonal Φ table. Its
generator does not commute with itself at different z, so the flow is built per
section of the operator table's breakpoints, in one of three ways:

- `"constant_schur"`: a constant block per section, whose exponential is formed from its
  Schur decomposition and is exact over any step inside the section;
- `"frozen_residual"`: the block frozen per section inside the exponential, and the rest of
  the generator — its z variation and the connection — as an explicit residual that the
  Runge–Kutta source carries, so the embedded pair sees its error;
- `"cf4"`: a fourth-order commutator-free exponential of the whole generator at Gauss
  nodes of each stage interval, whose flow error the embedded pair does not see
  ([stepper.md](stepper.md) §2); admitted on `numpy_ref` only and never selected by
  default.

The coupled flow runs on `numpy_ref`, `numba_cpu` and the native CPU build, which form
the per-bin flow actions in their own kernels and agree with the reference to a derived
rounding bound; a certificate records each flow's conditioning. A multimode capillary
taper carries the taper coupling by default; `taper_coupling=False` is the explicit
opt-out to the adiabatic route
([the taper coupling](../guide/waveguides/taper-coupling.md#the-taper-coupling-without-a-wall)). The general
modal route and its moving guides are described in
[general modal propagation](../guide/waveguides/general-modal.md).

**Non-goals.** Extended precision; adaptive spatial quadrature; Julia interoperability.
Queue and Slurm executors are implemented; more general distributed task graphs are
outside the scan model.
