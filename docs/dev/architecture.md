# solprop architecture

This is the current architecture of solprop, the reference that changes to the design
are checked against. Companion documents:
[conventions.md](conventions.md) (units, signs, shapes), [stepper.md](stepper.md) (the
solver), [backends.md](backends.md) (execution), [kernel-contract.md](kernel-contract.md)
(the seam), [decisions.md](decisions.md) (why) and the [issues](https://github.com/LupoLab/solprop/issues) (what is
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
had; then GPU performance (ADR-020). Double precision throughout, on every backend
(ADR-077): realistic modelling always carries noise, so the stepper's job is efficiency at
sufficient accuracy, not extended precision.

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
declared tensor Raman supports both grids on NumPy and native;
unlabelled custom kernels and the default silica model remain scalar-only.
Envelope plasma with THG is refused. The native CPU/CUDA/HIP paths
have their own qualification. The [running guide](../guide/setup/running.md)
and [backends.md](backends.md) record the qualified scope.

**Optional atomic response** ([SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md)).
A scalar helium `TDSEGasResponse` owns
the gas nonlinearity. Its immutable specification becomes a `TDSEDescriptor`;
the driver lazily owns native plans and stable radial chunks and passes a narrow
host binding explicitly to NumPy. The optical march uses Adams. Atomic device
execution is independent of the optical backend. The native backend refuses this
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
saved output has a schema version and the kernel contract has none
([kernel contract](kernel-contract.md), ADR-068).

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
belong to the driver/binding, which runs the provider in a worker process of its own
(`driver/_source_worker.py`, provider-agnostic, with `driver/_pyroraptor.py` as the
provider's adapter): framed pipes, no pickle, cancellation while waiting; modal synthesis to the nodes, the assembly of the
macroscopic sources and their projection belong to `backends/tdse_source.py`, which
imports neither backend and which a backend calls from its source evaluation
([SEP 49](seps/archive/sep-0049-tdse-on-native.md)). See the [TDSE manual](../guide/media/tdse.md).

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
| **T** translation | turns P objects into validated kernel-ready arrays and the `PropagationProblem` contract | P, numpy, scipy | may be expensive once per problem; results cached and validated |
| **B** backends | state buffers, transform plans, kernels, the inner z-loop, statistics kernels | numpy; the native package in `native/` (Kokkos, Kokkos-FFT and nanobind); from P only the types and constants the seam names — never a physics object | all hot-path rules of [backends.md](backends.md); in-place and `out=` allowed here only |
| **D** driver | `Simulation`, run loop over save planes, batch runners, `prop_capillary`/`prop_gnlse`, output handlers, statistics specs | P, T, B (through the seam only), h5py | none |
| **X** processing / plotting | analysis of outputs, standard plots, design tools | numpy, scipy, matplotlib, D (outputs), T (reference tables and their identities, for `processing.modal`; the recorded plane measure, for `processing.mixtures` and `processing.planes`) — never B | none; may call backend statistics kernels through the D layer when a computation belongs on the device |

**Transverse rules are built in P.** `modes/quadrature.py` constructs its rules
numerically -- Laurie's Kronrod recursion, and the field-scale rules of
[SEP 19](seps/archive/sep-0019-field-scale-quadrature.md) by the Lanczos
process and the Golub–Welsch eigenvalue method, certified on construction -- once per
problem, and caches them. It is the one numerical construction kept in P: it yields
nodes and weights for T to fold into the transform, and no compiled code, per-stage
evaluation or hot loop enters P with it.

**Construction's host resources.** The private process policy in `_process` holds
NumPy's and SciPy's controllable BLAS pools at one thread (or `SOLPROP_BLAS_THREADS`
under ADR-052) around guide construction, standalone graded-index solves, translation
and coupling. These coarse scopes change no physical algorithm. They restore the
previous limits after the last nested or overlapping scope exits, since the limits
are process-wide. Construction and the native adapter share one private
lease coordinator: each library takes the smallest active request, even when owners
start and finish in different orders, and only the last release restores its original
width. Native's adapter supplies the lease hook explicitly; the standalone native
package keeps its local hold and imports no solprop code. Construction excludes the
native backend's own BLAS and does not import a backend; an already loaded native
package supplies its library identity through `build_info`. See the
[BLAS guide](../guide/performance/tuning.md#blas-and-vendor-libraries).

**The X → T edges.** `solprop.processing.modal` imports
`solprop.translate.modal_tables.ModalTransformTables` and
`solprop.translate.general.guide_identity`, and that is inherent to the output schema
rather than incidental. A general modal file records what its E/H reference **is** --
the table identity, the guide identity -- and not the reference's tens of megabytes,
so the reference is rebuilt in **T** by `general_modal_route` and read in **X**, which
checks it against the stored identity, path by path, before it synthesises anything.
**X** takes the record type and the identity functions, never an execution map, never a
`PropagationProblem` and never a backend. Two further modules read what a file records the
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
| `profiles.py`, `gasflow.py`, `pulses.py`, `noise.py`, `polarisation.py`, `optics.py`, `tools/` | Longitudinal paths, steady gas flow in a hollow fibre (the taper- and temperature-aware, choked, regime-checked fill, its flow report and calculators), temporal/spatial inputs, noise, polarisation algebra, linear optical elements as transfer functions, and the design calculators: pulse shape numbers (`tools/pulse.py`), soliton numbers (`soliton.py`), beam numbers (`beam.py`), the fill and zero-dispersion search a design evaluates (`waveguide.py`), dispersive-wave phase matching (`phase_matching.py`), the parameter solvers (`design.py`) and the soliton parameters and designs of a capillary (`capillary.py`), all re-exported from `solprop.tools` | P |
| `coupling/` | Coupling simulations through a coherent field at a reference plane (SEP 31): the field plane with its reference, geometry, metric, basis identity and provenance (`plane.py`), the input port of a receiving simulation and its bound launch (`port.py`), temporal remapping (`remap.py`), the coupling models, their scalar mode profiles and the recorded pair envelopes (`models.py`, `envelopes.py`), `couple` with its admission budget and report (`coupler.py`), and the synthesis of a guide's exit field (`outcouple.py`), and the transport core: the linear elements between planes (`elements.py`) and the ordered transport of a plane with its ledger (`transport.py`); NumPy, SciPy and other P modules only | P |
| `physdata/` | Species, mixtures, equations of state, transport properties (viscosity, sound speed, mean free path), optical material data, metals, nonlinear and Raman data | P |
| `modes/` | Mode protocols and derived quantities; capillary, antiresonant, tabulated, step-index, rectangular and planar (slab, uniform in y, per unit width) modes, with `PlanarGrid` for the planar guide resolved in y; shared gas fill, quadrature and collections; full-vector modal records, material laws and regions, bound step-index and radial graded-index providers, and the guide that pairs a provider with a mode collection, and the connection of a basis that moves along z | P |
| `linear.py`, `boundaries.py`, `nonlinear/` | Linear assembly, absorbing rates, response definitions, ionisation, Raman, normalisation and the GNLSE mapping | P |
| `translate/` | Validated `PropagationProblem` (`translate/problem.py`) and its route union (`translate/routes.py`), assembled by `assemble` (`translate/assembly.py`), with a capillary's coherent wall and taper coupling in `translate/walls.py` and the refusals and advisories assembly applies in `translate/validation.py`; parameter expansions and z tables; modal/radial/Cartesian transforms and the hybrid planar one (slab modes in x × FFT in y, `translate/planar.py`); dense tracked modal tables, the general modal route and the coupled-flow and metric records (`translate/general.py`, `translate/coupled.py`); the spatial plane measure (`translate/flux.py`, the `FluxPlan` of schema 35); the persisted frame integral and the filled input port of a problem's entrance (`translate/coupling.py`); what the general route and the coupled flow execute, checked once (`translate/admission.py`); response tables, masks and stage bundles | T |
| `backends/` | Registry and protocols, sessions, the Adams call ledger both backends keep (`ledger.py`), node chunks, scalar bodies, spatial and population statistics, and the TDSE response's host source assembly (`tdse_source.py`) | B |
| `backends/numpy_ref/` | Reference execution, its Adams kernels and tables (`multistep.py`) and its on-axis scan (`populations.py`) | B |
| `backends/host/` | The reference's own step driver, Adams and low-storage marches, fixed-schedule group, run session and group session ([SEP 47](seps/archive/sep-0047-run-ownership.md) phase 2); the parts the Python sessions compose are in `backends/session_core.py` | B |
| `stepper/`, `log.py`, `stats.py` | Tableaux, controllers, schedules, progress reporters and statistics specifications | D |
| `driver/`, `driver/routes/`, `driver/executors/`, `scan.py` | The `Simulation` object (`simulation.py`) with its settings and their admission (`settings.py`), its input, launch, noise and peak field (`inputs.py`), its run record (`metadata.py`), resume (`resume.py`), many-input runner (`runner.py`) and TDSE run glue (`tdse_run.py`); the keyword routes behind its constructors, one module per family (`routes/`: GNLSE, hollow guides, general modal guides, spatial grids, and the material selection and keyword attribution they share); the convenience interfaces, scan/batch orchestration, Local/Pool/Queue/Slurm execution and scan CLI | D |
| `calibration/` | The native backend's machine calibration ([SEP 46](seps/archive/sep-0046-machine-calibration.md)): reading, showing, exporting, importing and clearing profiles in the per-user store, through solprop-native's standard-library modules loaded by path so that the native runtime never starts (`calibration/_native.py`), the measuring controller, its child processes, the synthetic problems they time, the operation-grain candidates and the selection rule (`calibration/controller.py`, `child.py`, `problems.py`, `grain.py`, `selection.py`), the refinement of one simulation (`calibration/refine.py`, stage 2, which `Simulation.calibrate` runs), and the `solprop calibrate` command (`calibration/command.py`) | D |
| `output/` | Memory and HDF5 outputs, provenance/resume, scan storage and artifacts; the run record in five modules: serialising configuration objects (`serialisation.py`), what executed a run (`provenance.py`), the record types of a saved run and its planes (`records.py`), `RunMetadata` and the resume key (`metadata.py`), and the save protocol, `PlaneStore` and field retention (`store.py`) | D |
| `processing/`, `plotting/`, `sol.py` | Field, spectrum, time, coherence, compression, mixture and scan analysis; a saved plane read back as a field plane or a fluence map (`processing/planes.py`), and a field plane's file form (`processing/plane_file.py`); propagation/statistics plots and spectrograms; the interactive session namespace, the one module whose import has side effects | X |
| `service/` | Front-end support, above every other layer: the messages between a front end and solprop, the kernel process that answers a front end's checks, previews and readings of a results file as plain data, each run as a job that keeps its record in its folder, and the helper and connection that reach a machine's runs over SSH ([the front end's design](gui/design.md), sections 13.3 and 13.4). Not part of the user API and outside the stability promise; nothing else in solprop imports it | X |

`solprop-native` is the backend's second distribution, built from `native/` in this
repository at solprop's own version, and a required dependency of solprop pinned to that
version ([SEP 43](seps/archive/sep-0043-retire-numba-cpu.md)). It depends on the seam,
which has no version number: the pin and `check_build` guard it (ADR-068).

`solprop-gui`, the desktop front end, is the third, built from `gui/` at solprop's
version and installed through the `gui` extra ([its design](gui/design.md)). It sits
above every layer here and uses only solprop's public interface, running it in child
processes through `solprop.service`; nothing in solprop imports it. The service lives in
solprop rather than beside the window so that a machine which only runs simulations needs
solprop alone, with no Qt.

`driver/scaffold.py` is an independent fixed-step IP-RK4 reference used by tests and
its tutorial; it is not the production stepper. It propagates a `GnlseProblem`
directly, without a contract, a backend or an output handler, which is what makes it a
cross-check of the reference backend. `processing/windows.py` is a candidate
module, drawn dashed in figure 1, to be written only for an agreed use case.
`_display.py` is private: the formatter behind every class's printed block, imported by
any layer and importing none. `_validation.py` is private too: the plain refusals
(`positive`, `finite`, `choice`, `shape`) in one wording, each raising the caller's own
exception class, imported by any layer and importing only `exceptions.py`.
`_ownership.py` is private in the same way: it records which arrays solprop holds,
so that an array solprop has just made is frozen where it stands (`hold`) and anything
a caller hands in is copied before it is frozen (`frozen_copy`), in every layer. So is
`_identity.py`: the canonical JSON text of a record and its SHA-256, which every stored
identity is (a run's resume key, a scan's common record, a coupling plane's basis, a
TDSE basis or descriptor, a material population), and `_durable.py`: a small file
replaced whole, which the scan queue's state, the front end's run records and the TDSE
consistency cache are written with.

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
   │  .translate() → PropagationProblem  (validated arrays)                   (T)
   │     beside it the RunRequest: stepper settings, planes, statistics
   │  .backend = backends.select(name, problem, request)            (B)
   ▼
run loop over save planes and stats periods                        (D)
   │  session = backend.prepare(problem, request.for_run(...)).start(start)
   │     the start loads once, host → device
   │  while not session.finished: batch = session.advance()         the session's marches
   │     ObservationReady(z, stats, ticket)                          small host arrays
   │     session.export(ticket, recipe) → output.save(...)          one fused pass a plane
   │     session.acknowledge(batch.receipt)
   ▼
MemoryOutput / HDF5Output  ──►  processing / plotting               (X)
```

The inner loop (many adaptive or scheduled steps between two save planes) belongs to
the backend's session. The NumPy reference runs its own Python step driver
(`backends/host/`, SEP 47 phase 2) over its reference passes; a native run is its
native execution session's (`solprop_native.session.NativeSession`), which owns the
march, the plane schedule and the statistics natively and returns to Python only at an
event. The outer driver visits save planes. Device arrays never cross the seam;
`observe()` returns host arrays of the field at the plane
and the step history; `stats()` returns a small dictionary of host scalars/vectors.

The outer loop is `solprop.driver.run.run_problem`, which takes a translated problem, a
constructed backend, an output handler and a `Start` -- the entrance, or a resumed
state with its call ledgers and method history -- which the run's session loads into
the backend (SEP 47). Nothing in it is
specific to a `Simulation`; it is the piece a batch runner or a scan calls. What the
driver does not do is as much a part of the design as what it does:

- It **never conjugates a spectrum**. The backend works in the FFT convention, the output
  layer's recipe converts on the way to disk -- applied by the session as it exports a kept
  field into the store ([backends.md §1](backends.md#1-the-seam), "Export into a
  destination") -- and those are the only two places a sign changes.
- It **never touches the state between planes**. Everything between two save planes
  happens inside the backend's march.
- It **never computes physics**. The mapping from `gamma` to a susceptibility is
  `solprop.nonlinear.gnlse`, the statistics are `solprop.stats`, the storage is
  `solprop.output`; the driver is the wiring.

A general modal guide adds a chain of host-side records in front of the
`PropagationProblem` ([vector fields and projection](../guide/waveguides/modal-projection.md)):

```text
ModalProvider  ->  DenseModalTable  ->  ModalTransformTables  ->  PropagationProblem
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
spectral and temporal collar profiles, and the reference frequency `ω₀`. The temporal collar
lies outside the requested window: the realised window is at least
`trange / (1 − 2 collar_fraction)`, `collar_fraction` 0.1 by default (ADR-057). Sizes are even
and 2-3-5-smooth: by default (`fft_size="auto"`) the cheapest by a transform-cost model up to
the next power of two, or the next power of two or smooth size on request. The invariants
between coarse and fine grids (Luna `Grid.jl:107–108, 244–250`) are asserted in the
constructors and tested. `fourier.py` owns the
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
the wavelength limits; temporal collar outside the requested window, at least a fraction of
the realised window, guaranteed non-degenerate)
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

The translation layer produces the `PropagationProblem` ([kernel-contract.md](kernel-contract.md)):
one trajectory's plain arrays with fixed dtypes, shapes and axis names, validated (finite,
contiguous, sizes consistent) and immutable, with no version number (ADR-068). Its
shared groups are common to every route; what differs between routes is one variant of a
closed union, `problem.route` (`translate/routes.py`: `ModalRoute`, `GeneralRoute` or
`SpatialRoute` of kind radial, Cartesian, planar or line), each with the fields its
executor needs, which every backend visits explicitly and refuses by name before it
allocates ([SEP 47](seps/archive/sep-0047-run-ownership.md) §6.2). Beside the problem it
returns the run's `RunRequest` (`translate/run_request.py`): the stepper settings with
the route's defaults filled and the collars' limits applied, the save planes and the
statistics, which are how the problem is run and not what it means (SEP 47 §5.2;
`assemble` returns `Translation(problem, request)`). Four modules carry it:
`translate/problem.py` the contract's dataclasses and their own checks,
`translate/assembly.py` `assemble` and the route builders (mode set, general modal guide,
spatial geometry, the collars' step limits), `translate/walls.py` the coupled flows of a
capillary's mode set (the coherent wall and the taper coupling), and
`translate/validation.py` what assembly refuses and warns about beyond the contract's own
checks. What it computes, once per problem:

| product | from | representation |
|---|---|---|
| Φ table | `linear.py` + `ZProfile`s | piecewise Chebyshev in z, coefficients `(n_intervals, n_deg, 1, n_modes, n_pol, n_omega)` as in `kernel-contract.md` §3, breakpoints, hard kinks; constant-D fast path (degree 1: one interval, or one per stretch between the kinks of a response path) |
| unitary phase for noise | Im Φ | same table, imaginary part |
| quadrature | `modes/quadrature.py` | nodes, weights, coarse weights, synthesis and projection matrices; taper scaling rule |
| response spec | `nonlinear/responses.py` | compiled ordering of the fused pass, coefficients, ionisation-rate spline tables (log-spaced, converged, clip limits), Raman kernels on the doubled grid in the frequency domain, normalisation prefactors |
| masks and rates | `grid.py`, `boundaries.py` | in-band mask, spectral rate (already in D), temporal rate, hard band mask, crop/pad scaling; analytic fine-grid collar derivative and steady-current certificate for field plasma |
| initial state and noise realisation | `pulses.py`, `noise.py` | canonical layout |
| stepper settings (the request's) | `stepper/` | tableau, tolerances, controller parameters, schedule, save planes, caps |
| statistics requests (the request's) | `stats.py` | which statistics, which windows |
| spatial plane measure | `spatial.py`, `adiabatic.py`, `translate/flux.py` | the `FluxPlan`: separable coefficients whose root gives each column's axial admittance, the material square's z table for a varying radial medium, and its adiabatic admission (SEP 34) |
| route | `translate/routes.py` | the variant, its transform and its own leaves: the TDSE source, the plane measure, the transverse boundary |

Validation is not optional: a backend calls `validate_run(problem, request)` where it is
built, before it allocates, and refuses anything inconsistent; `problem.validate()` is the
part of it that needs no request. Tables are cached on the `Simulation` and rebuilt only when a physics object
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

**Memory budget** (field-sized unit `F = n_modes·n_pol·n_omega·16 B`): state `y`,
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

    def prepare(self, problem, request) -> PreparedExecution: ...  # a run's session
    def prepare_group(self, members, problems, requests) -> PreparedGroup: ...
    def timing_probe(self, start) -> TimingProbe: ...  # an estimate's trial
    def metadata(self) -> dict[str, object]: ...  # what the execution was
    def available_device_memory(self) -> int | None: ...
    def memory_ledger(self) -> dict[str, int]: ...
    def close(self) -> None: ...

    # The reference's host stepping seam (set_initial, propagate_to, observe, stats, the
    # monitor and the march's snapshot) is a protocol of its own, HostStepping, which
    # its session drives (backends.md section 1).
```

- `numpy_ref`: the readable oracle. Plain NumPy + `scipy.fft`, no fusion, no in-place
  tricks, one function per pass. Every other backend is validated against it.
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
  sanitizer run still outstanding. Fixed-schedule fibre batching runs as the native group
  session, one native session per member (§11). The private state is read only by the backend's own
  session and timing probe. Its floating-point licence is
  [SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)'s: contraction by default,
  strict opt-outs for the protected helpers (stable increments, small-argument functions,
  defects, compensated sums), reassociation only on the loops of a kernel that asks for it,
  and no other fast-math flag ([backends.md §3.2](backends.md)).
- JAX: evaluated in a measured comparison (ADR-002); not a planned backend unless the
  data reverse the decision.

[backends.md §9](backends.md) records which route each backend admits and where each is
qualified, and §11 its current limits.

## 10. Output and statistics

`MemoryOutput` (everything in RAM: fast, small problems, batch scans) and `HDF5Output`
(streaming to disk, chunked per saved plane, optional compression, cache/resume) implement one
save protocol (`output/store.py`) and one metadata schema (`output/metadata.py`, its record types in `output/records.py`): grid, modes, simulation type,
the full configuration that built the `Simulation` (serialised physics objects), package and
dependency versions, git provenance of the running code when available, the noise seed and
realisation, the stepper settings and the step schedule actually taken (z, h, error norm,
rejections, wall time), the statistics, and the rows a periodic subset of them recorded
between planes. Fields are stored in the canonical layout
`(n_z, n_modes, n_pol, n_omega)` (a batched scan stores each member separately), in the **optics** sign convention
with the attribute `spectral_convention = "optics"` (ADR-017: the output layer conjugates on
save and on resume); processing accessors squeeze trivial axes and accept `convention="fft"`.
A TDSE run may also write its raw provider evaluations to a separate archive
(`output/tdse_archive.py`, schema `solprop.tdse-archive.v1`), which the optical record
references and never needs ([the archive page](../guide/media/tdse.md)).

A kept field reaches a store once, already converted ([SEP 47](seps/archive/sep-0047-run-ownership.md)
§5.4): the run's session exports it in the stored form straight into the room the store
gives (`PlaneStore.field_destination`) -- `MemoryOutput`'s own block row, or a new array
that `HDF5Output` writes as it is -- and `PlaneStore.save_exported` stores the plane. The
conversion is the one `save` makes, so the stored bits do not depend on the route.

`CallbackOutput` (`output/callback.py`) stores nothing: it implements `Output` and
`RetainingOutput` and hands each plane to a callable, unconverted, its field a fresh array
in the internal convention that the callable may keep. It is deliberately not a
`PlaneStore`, whose readers would have nothing to read, and not a `CheckpointOutput`,
which promises a state a resume can continue.

`output/snapshots.py` coordinates optional immutable disk snapshots at the existing
accepted-step commit boundary. It persists retained march state and execution facts,
suspends the optical and optional archive handles while their advisory writer locks
remain held, copies and verifies the pair, publishes a manifest last, then reopens
the same writers. Backend and atomic resources are untouched; no additional observe,
landing plane or response evaluation occurs. The store bounds completed and incomplete
generations and leaves transport/retirement to an external controller. The manifest
has its own version, outside the scientific configuration and the kernel contract. See the
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

Fixed-schedule fibre batches run as a group (`solprop.backends.session.GroupSession`,
started by `solprop.backends.sessions.start_group`; SEP 47 section 6.7): a sequence of
independently translated single-trajectory member problems — no problem carries a batch
axis — each started from its own `Start`, advanced
one common window (save plane) at a time, each member reporting its own events. The
batched scan's loop (`driver/_scan_batch.py`) is the group's service loop. A backend
chooses its own group: `start_group` asks the first member to prepare it,
`members[0].prepare_group(members, problems, requests)`, and starts the returned
`PreparedGroup`, naming no backend. Two groups serve it:

- **The native group session** (`solprop_native.group.NativeGroupSession`) serves
  every native group, on a CPU or a device build. It owns
  one native execution session per member (`solprop_native.session.NativeSession`,
  adopting the member backend the scan built), so each member's march, events,
  attempts, statistics rows, tickets and boundary monitor are its own session's, and the group only orders the members'
  advances. Within a window the members take turns in rounds of at most 256 common
  steps (`ROUND_STEPS`), each turn native from end to end, and each member's noise
  realisation and the noise phase its planes store are its own session's. The running members stand
  at one common step between rounds, and a group-wide stop is polled there. A member's
  trajectory is its lone native session's, bit for bit at one thread. The members share
  the schedule and nothing mutable: no fused kernel, because the native marches own
  each whole step. Its refusals (`solprop_native.group.refusal`) are named and made
  before allocation: a run the native session refuses, an adaptive member, step planes,
  or members that do not share the schedule, save planes, method, field shape and
  statistics request. A batched scan forms none of these: it requires a fixed schedule,
  refuses step planes and groups only points of one execution layout
  (`solprop.driver.batch.compatible_groups`).
- **The reference's group** (`solprop.backends.host.group_session.HostGroupSession`,
  prepared by `HostPreparedGroup`) serves every `numpy_ref` group.
  It runs over the reference's group runner (`HostGroup`, or `NumpyBatchBackend`'s
  member-axis stage arithmetic, which `select_batch` builds), which adopts the loaded member backends, each with its own tables,
  state and noise, and advances them through each common step in turn. It keeps that
  runner rather than composing one `HostSession` per member, because the runner's
  arithmetic and failure positions are what a reference group means:
  the module docstring gives the reasons.

Admission uses memory ledgers and a free-device-memory query with headroom.

Every backend has exact member-versus-singleton identity checks under the same
schedule. That does not imply a throughput improvement on every workload: whether a
batch beats independent runs depends on the problem size and the device. Adaptive
lockstep is not implemented. A boundary stop (the monitor's error, or its warning
raised as an error) fails its member alone. A group-level failure (a non-finite norm,
a step floor, a trial exception, an in-flight failure) fails every member of that
group, each reporting its own committed position and checkpoint eligibility. Members
take their steps in turn, so those ahead of the failing one have already taken the
failed step: the reference group's members are then in flight and poisoned, having
stepped without recording; a native group's have committed and recorded their whole
turn and stay eligible. Recovering the surviving members is an open question
([#171](https://github.com/LupoLab/solprop/issues/171)). A native member commits its
launch plane before its monitor judges the launch, as a lone run does (SEP 47 section
5.3), so a refused launch ends it in the window after the launch's; the reference
group attaches the monitors first.

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
the CPU it is the inner loop of the fused pass. The device path ties no team to a node, so a
one-node fibre fills the GPU: rates and the pointwise terms are ranges over (node, time),
and the trapezoid's exposure and free current are two-level deterministic scans over tiles
of 1,024 samples (one team per row and tile sums the tile's increments, then each tile is
rescanned from its row's earlier totals). Each column's norm reductions are split over up
to 256 teams of at least 1,024 bins, reduced on the host. Response teams are 256 below 8,192 time samples and 512
at and above it; deterministic reduction blocks are 512. Fixed batches are
implemented; their measured crossover depends on workload
([backends.md](backends.md)). Public selection prefers the installed native backend,
otherwise, with a warning, `numpy_ref` (ADR-029, ADR-064); the installed native build determines its execution space.

## 13. Memory discipline

- Every buffer is named, preallocated at backend construction and listed in `memory_ledger()`.
- The Runge–Kutta march holds every stage row its tableau and dense output read, and
  running accumulators for the solution and the error estimate; the low-storage march
  folds each slope into its registers as it is produced (ADR-009 as amended by ADR-050).
- Space-domain work runs over node chunks; the chunk size is chosen from a memory budget
  (default: the whole node set when it fits a per-problem budget, else the largest chunk that
  does). The backend plans it when it is prepared, not the translation (SEP 47 section 6.5,
  `solprop.backends.node_chunk`), and an explicit `node_budget` is authoritative.
- `HDF5Output` never holds more than the current plane, in datasets preallocated for the
  planned planes with an `n_saved` count; `MemoryOutput` is allocated for the planned
  planes and grows by doubling past them.
- A kept plane's field is exported once, in the stored form, and shared by every consumer
  of that plane (the store, a plane observer, a terminal checkpoint): into
  `MemoryOutput`'s block row with no host volume of its own, or into one new array for
  `HDF5Output` ([backends.md §1](backends.md#1-the-seam), "Export into a destination").
- A run keeps its fields on every plane, the named planes or none
  (`FieldRetention`,
  [SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)): a plane without its field is not copied, converted or
  stored, the stores allocate field storage for the kept planes only, and the admission
  charges those alone (its `output` group). The backend is asked for the field only on
  the kept planes: `HostDriver.observe(field=False)` transfers nothing from a device,
  scans nothing for finiteness, skips the collar diagnostic (counted in the execution
  record) and forms no noise-phase volume, so a field-free plane costs its statistics
  alone. Under the low-storage march a step abandoned in flight after such a plane leaves
  no checkpoint there (ADR-067).
- The Cartesian and planar collar diagnostic reconstructs the physical field by block of
  frequency, at most a node chunk's worth of samples at a time, never the whole volume.
- A `SimulationRunner` keeps one backend's buffers and plans across many inputs, and is
  admitted once; each session's load restores everything a march reads (the native
  backend's `_load`, the reference's `HostDriver.set_initial`), so a reused backend
  repeats a fresh one's run. An input array of more than 2²⁰ elements is
  recorded by its SHA-256 digest, not inline, so the run record stays small; the runner
  reads such an array in place and marches on a problem without it, so no input outlives
  its load. A caller that identifies its input states that instead (`StatedInput`), and
  nothing digests it.
- Φ tables are shared where possible; the constant-D fast path stores one array.
- Fine-grid buffers use the smallest `ffac` the declared responses need.

## 14. Logging and progress

`logging.getLogger(__name__)` per module; the package installs a `NullHandler`;
`solprop.configure_logging(level, file)` for scripts. Debug logs are plentiful in P, T and D
(translation summaries, table sizes, schedule statistics, memory ledger) and guarded with
`logger.isEnabledFor(logging.DEBUG)` when formatting costs anything. Nothing logs inside
compiled kernels or per-stage code. Progress is reported by the driver through one
`ProgressReporter` protocol (`solprop/log.py`): `started` once, `step` from inside the step
loop on every attempted step, `plane` per save plane and `finished` at the end. Before the
first step `run_problem` builds the run's summary from the kernel problem, the run's
metadata and the backend's execution record (`driver/summary.py`) and hands it to a
reporter that implements `summary` (`SummarisingReporter`); a plane store keeps it as a
post-run-style annotation, outside the record and its hash. The default
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
the loop raises, before the error propagates, so a bar can give the terminal back. One
that implements `observe_plane` (`PlaneObserver`, SEP 41) is shown each committed plane
as a lazy view, and is detached if it fails; a scan reports to a `ScanReporter`, whose
`point_reporter` may watch each point's run, relayed from a pool's workers. A bar
writes the summary and the closing line to its stream unless a console handler would
print them, because without a handler a record reaches nobody.

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

The [issues](https://github.com/LupoLab/solprop/issues) record selected work and optional extensions.

## 16. Coupled linear flows, and non-goals

**The linear operator need not be diagonal in the modes.** Where channels couple
linearly — a capillary's coherent wall, the taper coupling of a moving multimode guide
(the connection of a basis that moves along z), a declared unitary basis of a split
block on the general modal route — the problem carries the optional
`linear.coupled_flow` leaf (`translate/coupled.py`) beside the diagonal Φ table. Its
generator does not commute with itself at different z, so the flow is built per
section of the operator table's breakpoints, in one of two ways:

- `"constant_schur"`: a constant block per section, whose exponential is formed from its
  Schur decomposition and is exact over any step inside the section;
- `"frozen_residual"`: the block frozen per section inside the exponential, and the rest of
  the generator — its z variation and the connection — as an explicit residual that the
  Runge–Kutta source carries, so the embedded pair sees its error.

The coupled flow runs on `numpy_ref` and the native CPU build, which forms
the per-bin flow actions in its own kernels and agrees with the reference to a derived
rounding bound; a certificate records each flow's conditioning. A multimode capillary
taper carries the taper coupling by default; `taper_coupling=False` is the explicit
opt-out to the adiabatic route
([the taper coupling](../guide/waveguides/taper-coupling.md#the-taper-coupling-without-a-wall)). The general
modal route and its moving guides are described in
[general modal propagation](../guide/waveguides/general-modal.md).

**Non-goals.** Extended precision; adaptive spatial quadrature; Julia interoperability.
Queue and Slurm executors are implemented; more general distributed task graphs are
outside the scan model.
