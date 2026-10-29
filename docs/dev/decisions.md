# Decision log

Architecture decision records for solprop. One entry per decision, numbered and never
renumbered: a reversed decision gets a new entry that supersedes the old one, and a
withdrawn record keeps its number as a one-line stub. Code comments cite these numbers,
so they never change. Status is `accepted`, `proposed`, `superseded by ADR-NNN` or
withdrawn.

Each record states the context, the decision and its consequences in a few lines. The design
itself lives in [architecture.md](architecture.md), [stepper.md](stepper.md),
[backends.md](backends.md) and [conventions.md](conventions.md); this file is the *why*.

**Reading the records.** Each record states the decision as it stands, with the
measurements that justified it. Where a later record changes part of an earlier one, both
say so. For open work see [roadmap.md](roadmap.md); for what each backend executes and how
it is qualified, [backends.md](backends.md); for the current seam,
[kernel-contract.md](kernel-contract.md).

## ADR-001 — Three-layer design: physics, translation, backends

- **Status:** accepted
- **Context:** Luna mixes physics, buffers, FFT plans and threading in the same structs
  (`NonlinearRHS.jl`). That made every optimisation touch the physics and every physics change
  touch the hot loop.
- **Decision:** solprop separates (P) a physics/convenience layer in plain NumPy/SciPy with no
  performance tricks, (T) a translation layer that turns physics objects into validated,
  kernel-ready arrays, (B) backends behind a narrow seam that own state, plans and kernels,
  (D) a driver layer (`Simulation`, run loop, batch, interface) and (X) processing/plotting.
  Imports flow P → T → B; B never imports P; X never imports B.
- **Consequences:** The seam contract ([kernel-contract.md](kernel-contract.md)) is a versioned
  frozen dataclass and the only object that crosses layers. Physics code stays readable; hot
  code stays hand-optimised; a new backend needs no physics knowledge.

## ADR-002 — NumPy oracle, numba CPU, and a native Kokkos backend for CPUs and devices

- **Status:** accepted; the default selection is ADR-029's
- **Context:** CPU performance is the stated priority; GPU is wanted but secondary. An
  earlier prototype study found that an array compiler (JAX/XLA) reaches within ~8 % of
  hand-written kernels on a bandwidth-bound step but loses 11× on multi-accumulator
  reductions and cannot execute FFTs in place; at solprop's array sizes per-call overhead
  and pass count dominate, which favours fused compiled kernels. JAX would give GPU and
  `vmap` batching for free.
- **Decision:** `numpy_ref` is the readable oracle and `numba_cpu` the always-installed
  optimised CPU backend. The accelerated backend, `native`, is built on Kokkos with
  Kokkos-FFT behind nanobind, in the `solprop-native` distribution built from `native/`:
  one source for OpenMP on the CPU, CUDA on NVIDIA and HIP on AMD GPUs. JAX is
  evaluated, not adopted. A Numba CUDA backend was measured against the native CUDA
  build and did not justify a second device implementation beside one that also covers
  AMD hardware; it is not part of the package.
- **Consequences:** numba and pyfftw are core dependencies (ADR-023). Two hand-written
  backends are kept honest by a validation ladder against the reference. The two
  repositories have independent gates; the coupling is the seam version, which covers the
  driver's protected surface and the stage bundles as well as the problem's fields. The
  GPU crossover, below which a GPU loses to the CPU, is measured and documented, not
  assumed: on a data-centre NVIDIA GPU a small multimode problem runs at about a laptop
  CPU's latency (1.11 s against 1.09 s on the reduced multimode RDW case), while the
  reduced hydrogen comb-to-continuum case is 4.1 times faster (2.16 s against 8.84 s). GPU
  selection is explicit or follows the installation (ADR-029), and scan throughput on a
  device is not claimed from single-problem timings.
- **The JAX evaluation.** On one trial step of each target problem, with the translation
  layer's Chebyshev passes hoisted out of every execution, agreement with the reference at
  3.3–4.5e-16 and the best thread count for each, JAX takes 2.8 to 3.0 times numba's time
  on the RDW step and 0.88 to 0.95 times on the comb-to-continuum case with the hydrogen
  Raman convolution, on a laptop and on a 36-core server alike. In the pattern production
  runs — concurrent single-thread runs filling the server — numba delivers 2.1 times JAX's
  throughput. JAX therefore wins on neither problem in practice, and `numba_cpu` is the
  CPU backend either way; its per-element bodies are shared with a device family as this
  record intends. See [backends.md](backends.md) §5.

## ADR-003 — Kernel state layout `(n_batch, n_modes, n_pol, n_omega)`

- **Status:** accepted
- **Context:** Luna has different array shapes per geometry and special code paths for scalar
  vs vector polarisation and for mode-averaged vs modal propagation. Polarisation support must
  not slow single-polarisation runs; batch mode must be a first-class leading axis.
- **Decision:** One canonical complex128 layout with the frequency axis contiguous. Trivial
  axes (`n_batch = 1`, `n_modes = 1`, `n_pol = 1`) are ordinary cases; kernels are specialised
  on static shape flags, not duplicated. Space-domain buffers are
  `(chunk_of_nodes, n_pol, n_to)` and responses run over node chunks bounded by a memory
  budget.
- **Consequences:** No geometry-specific transform structs. User-facing accessors squeeze
  trivial axes for convenience; storage and the seam are always canonical.

## ADR-004 — Interaction-picture DOPRI5 with an embedded pair and a lagged PI controller

- **Status:** accepted; the interval cap is superseded on the diagonal-flow routes by
  ADR-038, and the default integrator is ADR-045's
- **Context:** An earlier prototype study measured IP-DOPRI5/IP-Tsit5 as best or second
  best on every rung (Kerr, shock, plasma) at matched error, with a free embedded error
  estimate; splitting schemes lose their closed-form advantage as soon as shock or plasma
  appear and become 4–7× dearer. Rejections are 0–0.4 % so a reject-free, lagged
  controller costs nothing and removes the per-step device synchronisation.
- **Decision:** IP-DOPRI5, named by `RungeKutta()`, with IP-Tsit5 available; PI
  controller with `lag` (default 1) and `reject_free` options (growth cap 1.1, safety
  0.8); weak global error norm over the in-band mask with a weighted option; step capped
  by the containing Chebyshev interval (where ADR-038 keeps that cap) and the next hard
  kink; fixed schedules (ADR-019) are a first-class mode.
- **Consequences:** Six nonlinear evaluations per step; `exp(ΔΦ)` per distinct stage fraction.
  Luna's `RK45.jl` is not ported; only the tableau constants carry over, with tests for the two
  DOPRI5 traps fixed upstream (FSAL reuse before dense output was consumed; the 4th/5th-order
  weight vectors mislabelled).

## ADR-005 — Cumulative linear operator as piecewise Chebyshev in z

- **Status:** accepted
- **Context:** Luna evaluates the z-dependent operator by a right-endpoint rule, which is
  globally first order in h and invisible to the error estimate. Because D(ω,z) is diagonal in
  ω it commutes with itself, so `exp(Φ(z₂) − Φ(z₁))` with `Φ = ∫D dz` is the *exact* linear
  flow over any step.
- **Decision:** Store Φ(ω, z) as a piecewise Chebyshev polynomial with array-valued
  coefficients, built by adaptive splitting of the scalar parameter paths (density, core
  radius) with stage boundaries as mandatory breakpoints, then resolved on D itself; integrate
  in the Chebyshev basis; evaluate by Clenshaw; track hard kinks separately from split edges;
  special-case constant and piecewise-constant D (exact, one interval, degree 1). Where the
  operator depends on z only through a few scalar parameters, a parameter expansion replaces
  the z table; the z table is the general fallback. When the interval budget is exhausted
  the builder relaxes its tolerance within declared limits and warns if intervals remain
  unresolved.
- **Consequences:** The linear part no longer constrains the step size. Memory is
  `n_intervals × (degree+1) × n_omega × n_modes` complex per problem (tens of MB at most);
  batch members sharing a fibre share one table. See [stepper.md §4](stepper.md).

## ADR-006 — Co-moving frame with analytic β₁

- **Status:** accepted
- **Context:** A fixed frame walks the pulse out of the time window on any gradient fibre
  (94 % of the peak on the window edge in an earlier prototype study); a finite-difference
  β₁ has a roughness floor of ~5e-8 that caps the cumulative operator's accuracy at 1e-8.
- **Decision:** β₀(z) and β₁(z) are recomputed along z. The modes protocol provides
  `dneff_domega(omega, z)` analytically (Sellmeier and Marcatili expressions differentiated
  symbolically; splines differentiated exactly); a validated finite-difference fallback exists
  for user-supplied modes and logs the accuracy it can reach.
- **Consequences:** Every refractive-index model ships with its derivative and a test against
  finite differences at three step sizes.

## ADR-007 — Absorbing boundaries as rates

- **Status:** accepted
- **Context:** A per-step multiplicative window gives total absorption `Wⁿ` for `n` steps,
  so the answer depends on the tolerance (Luna PR #438).
- **Decision:** Spectral collar → rate α(ω) = −2 log(W)/ℓ folded into D before Φ is
  integrated (free, exact). Temporal collar → the bounded rate profile
  `α(t) = 12.6·(1 − W(t))/ℓ` (`boundaries.temporal_collar_rate`), applied as ADR-025
  describes. The hard band limit outside the **active** mask (Luna's `sidx`,
  `ω_min/2 < ω < 1.1 ω_max`) stays a per-step 0/1 mask (idempotent); the in-band mask is
  narrower and would erase the collar the rate acts on. The collar geometry is validated to
  be non-degenerate (Luna's power-of-two `twin` bug).
- **Why the temporal profile is bounded:** `12.6·(1 − W)` is calibrated to the capped
  logarithm's integrated attenuation, so the walk-off rule and `boundary_N` keep their
  meaning, while its largest rate is a fifth of the cap and every derivative is
  continuous. Under ADR-025's in-stage term the ceiling rate is an eigenvalue of the
  linearised problem; as a capped logarithm it would set the stability limit and, through
  the error estimate, the step count.
- **Consequences:** Results converge with `rtol`; the absorber strength is a physical length.

## ADR-008 — Modified shot noise, drawn once and advanced by the unitary linear flow

- **Status:** accepted; the input model is the default and this one an option (ADR-055)
- **Context:** Luna's `:modified` noise held the noise field constant in z, which is wrong in
  a dispersive medium (1.4–6.6× per frequency for spontaneous MI; 50–80× in conservative
  propagation). Luna PR #440 advances it by the unitary part of the linear operator:
  dispersion yes, loss no, gain no.
- **Decision:** Draw one realisation `N₀(ω)` per mode; at every stage use
  `N(z) = N₀ · exp(i · Im Φ(ω, z))`, obtained from the same Chebyshev table; add it in the
  frequency domain inside the fused `exp(ΔΦ)·y + N(z)` pass before the inverse FFT (no extra
  transform, unlike PR #440's time-domain addition). The traditional input-noise model remains
  available. The saved plane's `noise_phase` restores the background in processing. The realisation and
  seed are stored in the output so runs are reproducible and resumable.
- **Consequences:** One extra complex multiply-add per element per stage. Ensembles use fixed
  schedules (ADR-019) so the numerical error is a common systematic.

## ADR-009 — Steps land on save planes by default

- **Status:** accepted; amended by ADR-039 (opt-in interpolated save planes for both
  steppers) and, for the stage storage, by ADR-050
- **Context:** Dense output needs the stage vectors kept until the interpolant is consumed
  (Luna's FSAL/dense-output ordering bug came from exactly this) and two extra field-sized
  buffers on a device. Save planes are few (10²) against steps (10⁴–10⁶).
- **Decision:** The controller clips the trial step to land exactly on the next save plane
  (Luna's `step_on` mechanism, generalised to hard kinks and stage boundaries). The
  solution and error estimates are running accumulators. The Runge–Kutta march keeps the
  stage rows its tableau reads: DOPRI5's Butcher matrix is full lower triangular and its
  dense output reads every stage, so `numba_cpu` and `native` hold every stage row, both
  accumulators, the accept buffer and the recycled stage, and `numpy_ref`'s ledger counts
  them. Only the low-storage march (ADR-050) consumes each slope as it is produced, in
  three registers. Stepping across planes and interpolating them is ADR-039's opt-in.
- **Consequences:** Saved fields are exact step endpoints. A run with many save planes takes
  slightly shorter steps near them (negligible at the usual ratios).

## ADR-010 — Fixed-quadrature multimode transform only

- **Status:** accepted
- **Context:** Luna's adaptive-cubature `TransModal` is serial, host-only, has a
  data-dependent cost per step and an error estimate that is 10–1000× optimistic with a
  kinked ionisation rate. The fixed Gauss/Kronrod rule (`TransModalFixed`) is exact for Kerr
  and Raman (polynomial in the mode fields) and spectrally convergent for plasma once the
  ionisation rate is smooth.
- **Decision:** Only the fixed-quadrature transform is ported: batched FFT → GEMM synthesis →
  fused responses over node chunks → GEMM projection → batched FFT, with the embedded
  coarse-rule error estimate as an *alarm*, never a controller. Mode-averaged propagation is
  the `n_nodes = 1` case of the same pipeline with the effective-area normalisation.
- **Consequences:** The ionisation rates, PPT among them, come from optmat
  ([SEP 3](seps/archive/sep-0003-optmat-materials.md)); the quadrature order is a validated
  default with a convergence test in the manual.

## ADR-011 — Response definitions with declared traits, compiled into one fused pass

- **Status:** accepted
- **Context:** Luna has columnwise, pointwise and batched variants of every response and a
  serial fallback that cannot run on a device; some responses live in the time domain,
  Luna PR #412 adds spectral-domain ones, PR #417 adds z dependence.
- **Decision:** A response is a frozen dataclass declaring its domain (pointwise; with memory
  such as plasma; convolution such as Raman; spectral such as TPA), supported grids
  (field/envelope), polarisation handling (scalar, vector-isotropic, tensor) and z dependence
  (scalar paths). The translation layer compiles the declared set into one fused time-domain
  traversal per column (|E|² → plasma scan → current → Kerr → Raman drive) plus any spectral
  responses. Batched variants do not exist as types: batching is the kernel layout.
- **Consequences:** Adding a response means adding a definition, a scalar body and a test; the
  kernel composition is generated, not hand-written per combination.

## ADR-012 — Two output handlers, one metadata schema; statistics at saved planes

- **Status:** accepted
- **Context:** High-speed small problems need a memory-only output and large ones a
  streaming HDF5 output. Luna computes statistics on every accepted step, which cost ~20 %
  of a step and dominated device runs.
- **Decision:** `MemoryOutput` and `HDF5Output` share one schema (grid, modes, simulation type,
  configuration, versions, noise seed and field, step schedule, step history, statistics) and
  one save protocol; statistics are computed at saved planes by backend kernels in one
  deterministic pass, with an optional periodic mode for step-level diagnostics; HDF5 output
  supports cache/resume and compression.
- **Consequences:** Processing and plotting read either handler through the same accessors.

## ADR-013 — Batch mode: lockstep with fixed schedules, or independent adaptive runs

- **Status:** accepted; batch agreement is held to ADR-049's rounding bound
- **Context:** Many small problems must run in a highly parallel way; several large ones in
  separate processes. Adaptive stepping in a vectorised batch forces every member onto the
  stiffest member's step; ensembles must use identical steps so the numerical error is a
  common systematic.
- **Decision:** *Lockstep* execution (leading batch axis; the GPU and vectorised-CPU path) uses
  fixed schedules only — discovered on a representative member, calibrated, replayed —
  as a hard rule. *Independent* execution runs each problem adaptively in its own spawned
  process, because thread pools, FFTW plans and the OpenMP runtime are process-global:
  `Local` is the scan API's default executor and `Pool` the CPU parallel one. Φ tables are
  shared across members that share a fibre and memory-budgeted otherwise. Fixed batches
  adopt independently loaded member backends and keep per-member observations.
- **Consequences:** A batch member agrees with its sequential run under the same schedule
  to a rounding bound (ADR-049); this is a test. Recovery and failure semantics are in
  [scans.md](scans.md).

## ADR-014 — Stateful `Simulation` object

- **Status:** accepted
- **Context:** Luna's free-function `run` cannot be re-run, reset or vectorised; the `LunaSim`
  branch showed a callable, re-runnable object is a better fit.
- **Decision:** `Simulation` holds the physics objects, owns translation and backend selection,
  and exposes `run()`, `reset()`, `propagate_to(z)`; `prop_capillary` and `prop_gnlse` are
  thin conveniences that build one.
- **Consequences:** Batch mode vectorises `Simulation`s; the kernel problem is built once and
  reused across runs.

## ADR-015 — Logging with the standard library, nothing in kernels

- **Status:** accepted
- **Decision:** `logging.getLogger(__name__)` per module, `NullHandler` on the package,
  `isEnabledFor` guards around any formatting that costs more than a string, no logging call
  inside compiled kernels or per-stage code; progress is reported by the driver through one
  pluggable reporter -- `started`, `step` per attempted step, `plane` per save plane,
  `finished` -- whose default emits a periodic line on a **wall-clock period** rather than
  every N steps, so that the log volume of a run does not depend on its step rate;
  `solprop.configure_logging()` for scripts.

## ADR-016 — Documentation stack with sphinx-gallery for executed examples

- **Status:** accepted
- **Decision:** Sphinx + myst-parser + pydata-sphinx-theme + numpydoc (validation wired into
  pre-commit) + recursive autosummary + mathjax + intersphinx + matplotlib plot directive +
  sphinx-gallery. Tutorials are Python scripts under `docs/tutorials/`, rendered with outputs
  and figures; there are no notebooks.

## ADR-017 — Optics sign convention at the interface, FFT convention inside

- **Status:** accepted
- **Context:** Optics writes a plane wave as `e^{i(k·r − ωt)}`, so the time-domain
  transform is `Ẽ(ω) = ∫ E(t) e^{+iωt} dt` and positive group-velocity dispersion is a
  positive parabola. FFT libraries use the opposite sign, and a real-to-complex transform
  cannot be conjugated for free. Luna keeps the optics convention for physics and the FFT
  convention for stored fields, and asks the user to flip signs by hand when adding
  dispersion to a field.
- **Decision:** All physical expressions, user-facing complex spectra and phases, the
  offline propagators, processing outputs and HDF5 files use the **optics** convention; the
  kernel state, the linear operator, the noise field, the translation layer and everything
  behind the backend seam use the **FFT** convention, Luna's validated internal signs. The
  boundary is a conjugation (with the half-window origin shift for spectra) at exactly
  three places, through named helpers in `solprop.fourier`. The conventions, the three
  places and the helpers are stated once, in [conventions.md §2](conventions.md).
- **Consequences:** No cost in the hot loop; no sign flips for users; no bare `np.conj`
  outside the three boundary places; a round-trip test guards the sign. Free-space
  transverse transforms follow the same rule so that all signs flip together.

## ADR-018 — Typing with jaxtyping on NumPy arrays, checked by beartype in tests

- **Status:** accepted
- **Decision:** Public signatures use `jaxtyping` annotations on `np.ndarray` with the standard
  axis names from [conventions.md](conventions.md); the beartype import hook in
  `tests/conftest.py` makes them load-bearing. No JAX dependency is implied.

## ADR-019 — Fixed schedules: discover, calibrate, replay

- **Status:** accepted
- **Context:** `rtol` is a local tolerance; the global error at the output is problem dependent
  and misses the request by one to three orders either way. Ensembles and GPU runs want
  identical, deterministic steps.
- **Decision:** One adaptive run discovers `h(z)`; a three-level convergence check with an
  order gate calibrates it; production runs (every ensemble member, every scan point where
  the schedule transfers) replay the scaled schedule. Dithered schedules are an option to
  decorrelate round-off.
- **Consequences:** Step schedules are first-class, saved with the output and re-loadable.

## ADR-020 — Priorities: accuracy and reliability, then CPU performance, then GPU

- **Status:** accepted
- **Decision:** No optimisation is accepted that moves a validated result beyond its stated
  tolerance; CPU is the default and best-supported target; the GPU path is designed in from
  the start (layout, seam, batch-first execution) and is implemented only for what the CPU
  backends have already validated and benchmarked.

## ADR-021 — Python 3.14 and current dependencies; no backwards compatibility

- **Status:** accepted; ADR-058 says what the 0.x releases promise
- **Decision:** `requires-python = ">=3.14"`, numpy ≥ 2.5, scipy ≥ 1.18, matplotlib ≥
  3.10.7 (the floors and their reasons are in [dependencies.md](dependencies.md)); modern
  language features are used freely; nothing is promised about API stability until a
  first release.

## ADR-022 — Testing policy: near-complete coverage in lanes

- **Status:** accepted; its repeatability and thread-count tests are rounding-bound tests
  (ADR-049)
- **Decision:** Target coverage is 100 % of non-kernel code measured in the default lane and
  of kernel bodies measured in a `NUMBA_DISABLE_JIT=1` lane through the identity-decorated body
  family; compiled kernels are tested for agreement with the reference, for repeatability
  and across thread counts, each to a stated bound. Markers: `medium`, `slow`, `gpu`,
  `benchmark`. Reference problems: linear-only (P0), N-soliton with phase (P1), analytic
  z-varying loss and dispersion (P2), cross-scheme Richardson with order gate (P3), and stored
  Luna results for the two target simulations. See [testing.md](testing.md).

## ADR-023 — numba and pyfftw are core dependencies

- **Status:** accepted
- **Decision:** The optimised numba CPU backend is always installed, with pyfftw for its
  transforms. The native backend is an optional second distribution
  (`solprop-native`, built from `native/`), installed through the `native` extra, which
  pins it to solprop's own version; solprop declares no extra for any GPU runtime.
  See [dependencies.md](dependencies.md).

## ADR-024 — withdrawn

Withdrawn (development-process record, not a design decision).

## ADR-025 — Temporal absorber placement

- **Status:** accepted; amended by ADR-057 (the spatial routes default to the split scheme)
- **Context:** See ADR-007. The temporal rate cannot be folded into Φ (not diagonal in ω) and
  cannot share the nonlinear polarisation's FFT because the UPPE prefactor `−iω/(2ε₀c n_eff)`
  is applied after that transform. Two exact-as-a-rate schemes exist: (a) a per-accepted-step
  split factor `exp(−α(t)·h/2)` on the modal time-domain field (one batched FFT pair on
  `n_modes` columns per accepted step); (b) the term `−α(t)E(t)/2` added inside every
  nonlinear evaluation with its own transform (one extra modal FFT per stage). The choice
  was made first on accuracy and convergence — the stepper's observed order unchanged with
  the absorber active, the absorbed energy independent of the step count, the in-window
  field unaffected to the tolerance — and only then on cost.
- **Decision:** (b), `absorber_scheme="nonlinear"`, is the default. Both schemes stay
  implemented behind `absorber_scheme` on every backend, because a scheme nobody can run
  is a scheme nobody can re-measure, and because (a)'s unconditional stability is the one
  thing it is better at.
- **The measurement.** The GNLSE N = 3 soliton launched into its own temporal collar,
  against the same problem at 1600 fixed steps, with the bounded temporal profile of
  ADR-007 (`tests/backends/test_absorber_schemes.py`):

  | scheme | relative error at 100 / 200 / 400 steps | observed orders |
  |---|---|---|
  | (a) `split_step` | 1.713e-3 · 8.564e-4 · 4.282e-4 | 1.00 · 1.00 |
  | (b) `nonlinear` | 3.543e-9 · 2.191e-10 · 7.077e-12 | 4.02 · 4.95 |

  From 100 steps to 400 the energy the collar removes moves by 5.9e-5 of itself under (a)
  and by 2.3e-13 under (b). All three accuracy criteria fail for (a).
- **Why (a) is first order.** It is exact for the absorber alone — the factors telescope,
  which is what "exact as a rate" means — but applying it once per accepted step is a
  **Lie splitting** of a term that does not commute with the linear flow. The splitting
  error is O(h²) local and O(h) global, and it caps the whole scheme at order one wherever
  the collar removes anything. A control with a centred pulse and the default collar shows
  the same order, at an absolute error small enough to go unnoticed.
- **What (b) costs.** One extra modal forward transform per stage, partly repaid: (b)
  makes FSAL available, which (a) denies by changing the state after the last stage, and
  it no longer pays the split factor's own transform pair per accepted step. FSAL needs
  the in-stage term **masked to the active band**, as the polarisation beside it already
  is through a prefactor that is exactly zero there; without the mask the stage writes
  outside the band the equation is solved on, the accept pass's hard mask removes it, and
  the first-same-as-last identity fails at 7.4e-7 instead of 1.8e-16. (b) is also
  **conditionally stable**: the collar's ceiling rate is an eigenvalue of the linearised
  problem, so an explicit method needs `h·α/2` inside its stability region (DOPRI5: 3.3
  on the negative real axis). `boundaries.stable_step` returns the limit and
  `KernelProblem.validate` warns for the two configurations that cannot find it for
  themselves — a replayed schedule and a step floor already past it. With the bounded
  profile the limit binds only on runs shorter than about `2·boundary_N` steps, and the
  step count matches the split factor's (456 against 459 on the reduced multimode RDW
  case). See [stepper.md §6](stepper.md).

## ADR-026 — The field is primary; the envelope is its analytic signal; the GNLSE is an interface mapping

- **Status:** accepted
- **Context:** Luna propagates the real field as the primary object; its envelope mode is the
  analytic signal of that field with the carrier removed, in the same √W units
  (`Processing.envelope`, `Fields.It`), so power, energy and every nonlinear formula follow
  from the field's normalisation. The GNLSE in Luna is an add-on that maps γ onto this
  envelope (`prop_gnlse`: `n₂ = γ A_eff / k₀`, `χ⁽³⁾ = (4/3)(1 − f_R) n₂ ε₀ c`, `n_eff = 1`,
  `A_eff = 1`) and is rarely used.
- **Decision:** solprop keeps the field as king. The envelope grid's state is
  `A(t) = E_a(t) e^{−iω₀t}` and inherits all field normalisations; `|A|² = P` is a consequence,
  not a convention. The GNLSE is a thin interface mapping onto the mode-averaged envelope
  problem — a dispersion-specified `Mode` plus the coefficient mapping above, with the shock
  term supplied by the `ω/ω₀` dependence of the prefactor — and never a separate solver path or
  normalisation. It is kept because the N-soliton problems are the cheapest exact tests of
  the solver core.
- **Consequences:** conventions.md §4 states the envelope as derived from the field; the
  field/envelope parity test is exact by construction; `nonlinear/gnlse.py` contains only the
  mapping and a mode; a user who wants "textbook GNLSE units" gets them because `|A|² = P`
  already holds.

## ADR-027 — A medium is a mixture of species; each species is a channel of the contract

- **Status:** accepted
- **Context:** A single-gas design assumes one species in many places: a closed gas
  registry, one density path per mode and per problem, one response of each kind, one
  plasma rate and potential, plasma statistics with no species axis, and an interface
  taking one gas and one pressure. Luna supports mixtures through a tuple-of-tuples
  convention that nothing enforces. Atmospheric and molecular work needs, for example,
  nitrogen and oxygen to ionise separately under one composition that also feeds the
  dispersion, the absorption, the Kerr and the Raman terms, and needs each species'
  ionised fraction at each point.
- **Decision:** a **species** is layer-P data (`physdata.species.Species`: a per-particle
  complex susceptibility whose imaginary part is absorption, an equation of state, and
  optional γ₃, ionisation and Raman data; the registries are one way to make one and a user
  may make another), and a **mixture** is an ordered set of labelled components with their
  own density paths (`physdata.mixtures.Mixture`). The linear operator forms
  `χ = Σ_s ρ_s γ_s` and `n` through the cancellation-free form; a fill (one shared
  pressure path) keeps the two-axis parameter expansion with the total pressure as the
  parameter; independent profiles use the z table on the CPU backends and a total-degree
  expansion where measured worth it. In the kernel contract each ionising species is a
  **channel** (its rate table, potential, pre-ionisation, density and electrons per event),
  each Raman-active species a **term** with its own amplitude path and its linewidth at the
  total density, and the Kerr coefficient one composite path; the fused column runs one
  exposure scan per channel and shares the current and loss integrals. The single-gas
  path is the one-channel specialisation and keeps the single-gas operation order, so it
  runs bitwise as a pure-gas implementation would.
- **Consequences:** every backend carries the channel loop and the per-species statistics
  (`species_ionisation` at every node); `prop_capillary` takes a gas, a mapping of mole
  fractions or a `Mixture`; `Simulation.with_mixture` replaces the composition without
  rebuilding what does not depend on it; there is no mixture equation of state (Dalton
  with real-gas partials, declared); no Raman constant or collision law is invented where
  the data are missing; dissociation channels are a later value of `electrons_per_event`,
  not a schema change.

## ADR-028 — withdrawn

Withdrawn (development-process record, not a design decision).

## ADR-029 — Prefer the installed native backend when no backend is specified

- **Status:** accepted; sets the default selection that ADR-002 leaves open
- **Context:** the optional native backend has measured CPU and GPU improvements, and
  installing its target-specific build should make ordinary propagation use it. Numba CPU
  remains a core dependency and the portable fallback. This is an installation policy,
  not a claim that one backend wins every workload.
- **Decision:** public propagation constructors and `SimulationSettings` default to
  `backend="auto"`. Resolve it without import or caching to `native` when its package can
  be found, otherwise Numba CPU. Store the concrete name in settings before execution
  and transfer it to workers. Explicit names retain control; `select` accepts `"auto"`
  for direct callers. Log the actual backend at INFO after successful construction.
- **Consequences:** the installed native build chooses OpenMP, CUDA or HIP. An
  installed backend's import, schema, device or unsupported-problem error is visible and
  never triggers a retry on another backend: a problem `native` refuses names the CPU
  backends in its message. A builder running on another host resolves against that
  host's installation unless its backend is explicit. The seam, physical model,
  precision, stepper, scan semantics and resume key do not depend on the resolution. The
  NumPy reference remains the oracle. See the [manual](../guide/setup/running.md).

## ADR-030 — Amplitudes are true-power amplitudes on every route

- **Status:** accepted
- **Context:** The modal field is built from the vacuum-admittance norm
  `N_m = ½√(ε₀/μ₀)∫|e_m|² dA`, inherited from Luna and from the hollow-core UPPE literature
  where `n_eff ≈ 1`, so the power carried by amplitude `a` is `Re n_eff |a|²`. The dynamics
  are exact for the true launched power, but a requested launch lands `n_eff` high, every
  energy, power and intensity output is low by `n_eff`, and along a pressure gradient the
  linear operator lacks `−½∂_z ln Re n_eff`. The offset is 0.56 % at 20 bar argon or 40 bar
  hydrogen and would be the core index for a solid or liquid core. Other modal propagation
  work (Snyder and Love; Kolesik, Wright and Moloney, Appl. Phys. B **79**, 293 (2004))
  normalises by the Poynting flux, so that the amplitude squared is the power.
- **Decision:** The user-facing amplitude on every route is the true-power amplitude
  `b = √(Re n_eff) a`: `P = |b|²` for a monochromatic field, and the Parseval sum of `|b|²`
  is the true energy, on both grids, at every position. The capillary route keeps its
  internal vacuum-normalised state and its hot loop; the conversion lives at the
  boundaries, as the sign convention does (ADR-017), and the linear operator gains the
  exact cumulative term `−½ ln(Re n_eff(ω, z)/Re n_eff(ω, 0))`. Temporal statistics use the
  analytic Poynting flux. The GNLSE route's factor is one (ADR-026's `index=1.0`), and the
  general modal route is Poynting-normalised by construction. The amplitudes, energy,
  flux, intensity, varying-index term and noise draw are stated in
  [conventions.md §4](conventions.md).
- **Consequences:** A nominal launch energy means true energy. Comparisons with Luna are
  made on internal amplitudes, so a like-for-like comparison scales the launch to Luna's
  internal amplitude (0.49 % less for the 40 bar hydrogen comb-to-continuum case, 0.004 %
  for the helium multimode case). Two approximations remain: the spatial functionals keep
  the vacuum factor `ε₀c/2`, and the modified noise keeps ADR-008's unitary transport, so
  the noise floor drifts along a gradient by at most 0.3 % of the noise amplitude over a
  20 bar gradient. The rejected alternative, carrying the true-power amplitude as the
  internal state, would have changed the first stage pass and the on-axis synthesis in
  every backend for the same result. The first approximation was removed by
  [SEP 34](seps/archive/sep-0034-medium-flux.md) (2026-10-23): the spatial functionals, launches
  and noise now use the medium's axial admittance, and a radial gas gradient carries the
  matching endpoint term.

## ADR-031 — The default capillary polarisation is x, and parity is defined once

- **Status:** accepted
- **Context:** Every mode carries a channel label (family, orders, parity), and saved runs
  persist it. The families disagreed about "even": the exact step-index `HE_1m` even member
  is x-polarised, while the capillary `MarcatiliMode` at `phi = 0` was y-polarised,
  inherited from Luna's `ϕ = 0`. "Even" is defined as the member whose transverse pattern
  is `cos((n − 1)θ)` about the x axis — the x-polarised member for `n = 1` — and "odd" as
  its quarter-turn partner; under Luna's orientation the default capillary mode would be
  labelled `HE1,1:odd`.
- **Decision:** the definition stands, and `phi = 0` is the x-polarised member for every
  `n`: the default `MarcatiliMode` (and `VincettiMode`) is `HE1,1:even`, its
  `orthogonal_polarisation` `HE1,1:odd`, and `principal_component` of a default capillary
  run is `"x"`. Long-term consistency across families wins over agreement with Luna. There
  is no alias and no compatibility path; a single-polarisation run is unchanged by
  rotational symmetry, and in a two-component run the components swap roles relative to
  Luna.
- **Consequences:** solprop's capillary convention differs from Luna's by a quarter turn
  (listed with the other [differences from Luna](../guide/workflows/luna.md)); a script that named `"y"` for
  a default capillary run in Luna names `"x"`; the persisted `parity` of a default mode
  reads `even`.

## ADR-032 — A steady-flow fill follows the bore and chokes its vacuum end

- **Status:** accepted
- **Context:** Luna's `Capillary.gradient` makes `p²` linear in `z` on every stage — the
  uniform-bore Poiseuille law — whatever the core radius, and a declared vacuum end is
  taken at its declared value, which numerically needs a floor. Neither is the steady flow
  a real fibre carries. In a taper the pressure drop concentrates where the bore is narrow:
  constant mass flow of an isothermal ideal gas in Poiseuille flow makes `p²` linear in
  `∫ a⁻⁴ dz` (Pennetta et al., APL Photonics 4, 056105 (2019), eq. 6), and with a
  temperature path in `∫ η(T) T a⁻⁴ dz`. A fibre exhausting into a pumped chamber cannot
  pass more than the sonic throughput at its exit, so its end sits at the pressure where
  the laminar and sonic throughputs are equal — tens to hundreds of millibar for common
  fibres, far above any floor.
- **Decision:** A gradient built by a simulation constructor for a circular hollow guide is
  the steady flow of `solprop.gasflow.steady_fill`: it follows the bore (and the temperature
  path), and **by default** each outer end whose declared pressure is below its choked
  pressure is raised to it — any declared value, not only zero — with the laminar–sonic
  equality solved exactly and the adiabatic sound speed at the exit.
  `vacuum_end="floor"` keeps the declared ends, floored at `1e-4` of the peak pressure; it
  concerns the ends only, so a taper's fill still follows the bore, and a uniform bore
  with `"floor"` is Luna's uniform-bore law. The flow is laminar Poiseuille flow
  throughout: a Reynolds number above 2000 (Gebhardt et al., Opt. Express 29, 6957 (2021))
  or a rarefied stage warns that the fill is not valid there rather than replacing it.
- **Consequences:** Every gradient-to-vacuum result departs from Luna's. Comparisons with
  Luna witnesses, and committed records of such runs, pin `vacuum_end="floor"` so that they
  stay like for like. A floor matters only where a fill is built without the gas or the
  bore (a `PressureGradient` constructed directly, a `Mixture.fill` without a radius) or
  under `"floor"`. Rectangular guides keep the uniform law, and an antiresonant fibre's
  flow uses its core radius only.

## ADR-033 — Conventional branch-and-pull-request development

- **Status:** accepted; amended by ADR-035 (major changes start as enhancement proposals)
- **Decision:** Changes arrive as pull requests from topic branches and merge into `main`
  when the full gate (`poe check`, see [tooling.md](tooling.md)) and continuous
  integration are green. Design documents under `docs/dev/` describe the code as it is,
  and a change that alters the design updates them in the same pull request: a new or
  reversed decision as a new record here, open work on the [roadmap](roadmap.md), and a
  deliberate difference from Luna.jl's behaviour on the
  [differences from Luna](../guide/workflows/luna.md) page. Luna functionality is added
  when a use case needs it. Numerical claims need an analytic or independent reference and
  a stated tolerance (ADR-020).

## ADR-034 — A distribution native build: a CPU baseline and the platform's BLAS

- **Status:** accepted; no build carries numerical pins (ADR-049); x86-64 baseline raised
  to `x86-64-v3`, with an automatic fallback below it (SEP 29, phase 3)
- **Context:** A portable `solprop-native` build — no vendor BLAS, generic x86-64 — lost
  to `numba_cpu` on every measured case at one thread: Numba compiles for the processor it
  runs on and uses NumPy's OpenBLAS, which selects its kernels at run time. Builds compared
  on an 18-core-per-socket AVX-512 server ([benchmarks.md](benchmarks.md#backends-and-builds))
  separated the local build's gain: the vendor BLAS is all of it on capillary problems
  (2.6–3.8× on straight capillaries), and the instruction set most of it on the general
  route. Against `x86-64-v3`, an `x86-64-v2` build's median cost on the benchmark core is
  2–4 %, because FFTW and OpenBLAS choose wider kernels at run time either way; the
  general modal route is the exception ([CPU performance](cpu-performance.md)). A
  pthreads OpenBLAS bundled beside Kokkos ran a second thread pool and made the modal and
  wall cases 10 to 75 times slower at full width.
- **Decision:** `solprop-native` has three profiles, `SOLPROP_NATIVE_PROFILE=LOCAL`
  (default), `DISTRIBUTION` and `REFERENCE`. The distribution build, for wheels, compiles
  for a CPU baseline — `x86-64-v3` on x86-64 (`x86-64-v2` is a source-build option),
  `neoverse-n1` on Linux aarch64, `apple-m1` on Apple silicon — and links Accelerate on
  macOS and, on Linux, an OpenBLAS built with OpenMP on the extension's own runtime,
  copied into the wheel with its licence. The Windows build links the `scipy-openblas32`
  OpenBLAS, held to one thread beside the Kokkos pool, until its single-runtime recipe
  exists (roadmap). The package reads the baseline from a file beside the extension and
  refuses to import, naming the missing features, on a processor below it (x86-64 and
  Linux aarch64; every Apple silicon Mac meets `apple-m1`); `backend="auto"` then
  chooses `numba_cpu` with a warning, the one exception to selection never falling back,
  and an explicit `backend="native"` raises the refusal. The baseline was first
  `x86-64-v2`; it became `x86-64-v3` to recover the general modal route's cost below. There
  is no multi-architecture build and no run-time BLAS switch (libblastrampoline): the
  measurements leave neither much to gain.
- **Consequences:** A native wheel can be published without giving up most of the local
  build's CPU speed; it still needs FFTW and the OpenMP runtime bundled (`auditwheel`,
  `delocate`, `delvewheel`), and a wheel that bundles FFTW is distributed under the GPL's
  terms, which are solprop-native's own (ADR-048). Processors below the baseline build
  from source. The reference build is the gate's tidy and sanitizer build; the gate
  builds and smoke-tests the distribution wheel. MKL, 1.4–1.8× faster than OpenBLAS on
  TE/TM walls, remains a local-build choice.

## ADR-035 — Enhancement proposals for major changes

- **Status:** accepted; amends ADR-033
- **Context:** Under ADR-033 every change is an ordinary branch and pull request. That
  suits fixes and small additions, but not a new model, route or numerical scheme: such
  work needs its theory, its implementation options and its verification agreed before
  code is written, is implemented in several pull requests across three backends and two
  repositories, and leaves decisions — tolerances, refused cases, measured costs — that the
  finished code does not explain.
- **Decision:** A major change starts as a solprop Enhancement Proposal (SEP), modelled on
  Python's PEPs and NumPy's NEPs, under [`docs/dev/seps/`](seps/index.md);
  [SEP 0](seps/sep-0000-process.md) is the process and `docs/dev/seps/template.md` the
  form. A SEP is standalone — it cites solprop's own documentation and code, the
  literature and public reports, never another code's source (optmat excepted, SEP 0 §7)
  or private notes — and carries the motivation, the theory with its references, the
  implementation options and the verification plan. Accepted SEPs are listed on the
  [roadmap](roadmap.md). They are implemented in phases on branches, `numpy_ref` first,
  then `numba_cpu`, then the native CPU build, until the backends agree; each phase
  appends a report of its deviations, decisions, verification and open questions to the
  SEP. A finished SEP is archived under `docs/dev/seps/archive/` and linked from the manual
  pages it produced. Fixes, documentation, tests, refactoring, tooling and optimisations
  that leave results unchanged do not need one. No SEP is written for existing
  functionality.
- **Consequences:** SEPs are public design records in the developer documentation. ADRs
  still record durable decisions and are what code cites; the manual and design pages
  describe the code as it is, and a final SEP is not updated to follow them.
  `tests/test_seps.py` keeps the SEP files, their index and the roadmap consistent with
  each SEP's status.

## ADR-036 — Dense observation for the low-call Adams march

- **Status:** superseded by ADR-039
- **Summary:** The Adams march steps across save planes and observes each by integrating
  the accepted step's own polynomial in the interaction picture, then applying the exact
  linear flow to the plane; hard kinks and the end of the run are still landed on, and an
  interpolated plane never enters the history or costs a source call. ADR-039 keeps this
  interpolant behind the one switch both steppers share,
  `SimulationSettings(interpolate_planes=True)`, in place of `Multistep(dense=...)`.

## ADR-037 — The low-call Adams method: one-call PEC, variable order

- **Status:** accepted; made the default integrator by ADR-045; amended by ADR-057 (the spatial routes' default collar scheme, and an opt-in quiet collar)
- **Context:** SEP 13 compared interaction-picture Adams methods for responses that
  dominate a run's cost: explicit Adams-Bashforth, PEC and PECE, matched and
  one-order-higher correctors, fixed and variable order. It ran them at full length on a
  gas-flow capillary at 1 and 2 mJ, and on RDW, CtC, GNLSE and soliton cases
  ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md)).
- **Decision:** `solprop.stepper.multistep.Multistep` is one-call PEC: an
  Adams-Bashforth predictor of order *q*, one source call at the predicted field, and an
  Adams-Moulton corrector one order higher through it. The step is controlled on the
  predictor-corrector difference. The method defaults to variable order up to five and
  has no interval cap.
  - Its error norm is scaled by the accepted state before the step, not by the larger
    of the states before and after, so that an unstable step cannot enlarge its own
    tolerance.
  - The in-stage temporal collar stays the default. The per-step collar is safe only
    when the collar holds no field.
  - PECE, explicit Adams-Bashforth and the matched pairs remain only as
    `ComparisonMultistep` for benchmarks.
- **Why:**
  - At constant step the predicted states advance by Adams-Bashforth of order *q* + 1,
    with an Adams-Moulton output correction, at one call a step.
  - Measured in source calls at matched accuracy, it was best or equal everywhere:
    4.5–7× fewer than IP-DOPRI5 on the 1 mJ capillary, and 2.4× on RDW.
  - Matched pairs underestimated their error far from the asymptotic regime, and one
    diverged undetected under the RK norm.
  - PECE never repaid its second call.
- **Consequences:** `numpy_ref`, `numba_cpu` and `native` implement it, in both
  formulations of ADR-041 and, beside a coupled flow, as ADR-044 describes; a backend that
  does not refuses `Multistep()` by name. `RungeKutta()` names IP-DOPRI5 (ADR-045).

## ADR-038 — Step limits by route: no interval cap on diagonal flows, a start-up rule

- **Status:** accepted; supersedes the interval cap of ADR-004 on the diagonal-flow routes
- **Context:** On gas-flow capillaries
  ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md)), steps were limited by things
  other than the error estimate. ADR-004 capped every step at the width of the Chebyshev
  interval containing it, and a choked end's table bisects to its width floor, so the
  runs took hundreds of µm-scale steps there. From `dz_init = 1e-4 L` the growth cap of
  1.1 took 45 steps to reach the natural step. On the canonical 4.2 m helium case these
  limits set most of the steps at `rtol ≥ 1e-3`.
- **Decision:**
  - **Interval cap.** It is off on the diagonal-flow routes. It stays on for a coupled flow
    and a reject-free controller. `StepperSettings.interval_cap` records the choice, and
    `SimulationSettings(interval_cap=...)` overrides it.
  - **Start-up rule.** When no `dz_init` is given, the step after the first is the
    deadbeat step the first one's scaled norm implies, at most 100 times as long
    (`PIController.start_factor`). It is off for an explicit `dz_init`, a coupled flow and
    a reject-free controller (`StepperSettings.auto_init`).
  - **Growth cap.** It stays at 1.1.
  - **Records.** Both settings are recorded in `StepLimits`.
- **Why:**
  - **The flow is exact across intervals.** The cumulative table (ADR-005) makes the
    diagonal flow exact for any step. What a step still integrates numerically is the
    source's z-dependence, which the embedded estimate sees. A coupled flow's own error it
    does not see, so its cap stays.
  - **No kernel needs the cap.** None needs a stage inside one interval; each stage's
    scalars are computed per stage.
  - **The canonical case.** At `rtol` 1e-2 to 1e-3 the two rules take 2.3 and 1.8
    times fewer calls than the interval cap and the fixed start, with the same errors.
  - **At `rtol` 1e-5 and 1e-6.** On the canonical case the exit plane's error roughly
    doubles under the 1.25 criterion. At 1 mJ, a tighter `rtol` reaches the capped
    errors in fewer calls; at 2 mJ and `rtol` 1e-5 it does not.
  - **RDW and the CtC slice.** RDW's calls are within 2%. On the CtC slice, a tighter
    `rtol` reaches the capped errors at 8% fewer calls at 1e-5 and at equal calls at
    1e-6.
  - **A looser growth cap.** A cap of 1.5 or 2 saved a further 15–20%, but cost up to 11
    times in field error at `rtol=1e-7` and 3–5 times in the weak far-UV band, which the
    weak norm does not see.
- **Consequences:**
  - The landing stretch stays 1.1, independent of the growth cap.
  - **GPU runs take the same defaults.** They are host-side limits of the shared march,
    and no kernel depends on them; a reject-free run keeps the interval cap and the fixed
    start on every device.
  - A looser growth cap waits for a norm that sees weak bands.

## ADR-039 — Opt-in dense output and natural-step saving

- **Status:** accepted; amends ADR-009 for `interpolate_planes` and `step_planes`;
  supersedes ADR-036
- **Context:** ADR-009 lands every step on a save plane. With many planes that caps the
  steps at the plane spacing: on SEP 13's canonical capillary at `rtol=1e-3`, 501 planes
  take 504 attempts against 91 for the natural steps.
- **Decision:** `SimulationSettings(interpolate_planes=True)`
  (`StepperSettings.interpolate_planes`) steps across the save planes and observes each
  inside its accepted step. One switch serves both steppers: the interpolant follows
  from the stepper, DOPRI5's extension or the Adams march's polynomial (ADR-036's), and
  the Hermite reference is `ComparisonMultistep(interpolant="hermite")`, for the
  benchmarks.
  - **The interpolant** is DOPRI5's fourth-order continuous extension (Hairer, Nørsett
    and Wanner II.6), contracted from the accepted step's stages into five coefficients
    in the anchor's interaction picture, then carried to the plane by the exact linear
    flow. A Hermite interpolant was rejected as one order below the step's error
    control. No other tableau has an extension here; Tsit5 is refused.
  - **Stiff bins.** A bin the linear flow attenuates by more than an e-fold over the step
    (the spectral collar) takes the anchor's value carried by the flow: its derivative
    grows exponentially across the step and no polynomial follows it. The same guard
    applies to the Adams interpolant.
  - **Landed still:** the end of the run and every hard kink.
  - **Refused** ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md) §5): the per-step
    temporal collar, a coupled flow, the `modified` noise model and a fixed schedule.
  - **Resume.** A file whose last plane was interpolated is refused for `resume=True`,
    for both steppers; `resume="interpolated"` continues afresh from it and records the
    position in the annotation `interpolated_resumes`.
  - **Every CPU backend.** `numpy_ref`, `numba_cpu` and `native` share the march; each
    supplies the accepted step's stage derivatives to the host (`_dense_stages`).
  - **Recorded** as `StepLimits.interpolate_planes`, in the resume key.
- **Why:** the extension costs no call and its error is the step's own order less one;
  on the canonical case at 50 planes the interpolated planes are as accurate as the
  landed ones, and at 501 planes the run keeps its natural 91 attempts.
- **Consequences:**
  - Landing stays the default: the step's norm is set by the strong components, so a
    weak band (the UV) at an interpolated plane is not controlled as at a landed one.
  - A plane is interpolated when no accepted step of the run ends on it.
  - Per-step statistics come from the march's accepted steps; what is computed from a
    plane's field is computed from the interpolated one.
- **Natural-step saving.** `step_planes=k` (`StepperSettings.step_planes`, recorded as
  `StepLimits.step_planes`) also saves the end of every `k`-th accepted step, beside the
  requested planes, with or without dense output and with the Adams march.
  - **No extra step and no interpolation:** `propagate_to` returns at the step's end,
    short of its target, and the driver asks for the target again. The steps are the
    run's own, so the planes follow the problem's scales, and a run stopped on one
    resumes as from any landed plane.
  - **Consequences, opt-in and documented:** the file grows with the step count, so `k`
    thins it; runs do not share planes, so comparing two runs needs interpolation;
    material processing that reads the requested planes' positions refuses the extra
    ones; scans refuse it, because a scan saves the same planes at every point.
- **The recycled stage is kept across planes.** Dropping the first-same-as-last stage at
  every save plane would cost one evaluation per plane, about one in six with a plane at
  every step. `numba_cpu` and `native` keep it, and a resumed replay agrees with the
  uninterrupted one to rounding (ADR-049).
  - The host keeps five field-sized coefficient buffers while a step that crosses a plane
    is held; `native` copies the seven stages to the host once per such step.

## ADR-040 — Tables accept a tail at the sampling-noise floor of their positions

- **Status:** accepted
- **Context:** SEP 14 found the canonical capillary's Φ table spending 99 of its 123
  intervals at the 4 µm width floor in the last half millimetre before its choked exit.
  The steady-flow pressure goes as the square root of the distance to the exit. Sampled
  at floating-point z near 4.2 m, it carries a relative noise of about
  `spacing(z) / (2 d)` at a distance `d`, from the positions alone, and above the 1e-12
  tolerance the coefficient tail cannot converge. The entrance's identical cusp, near
  z = 0, resolved in eight intervals.
- **Decision:** the adaptive Chebyshev builder also accepts an interval when twice its
  coefficient tail is within `SAMPLING_NOISE_FACTOR = 4` of `spacing(z) max|f'| / |f|`
  on that interval, with `|f'|` bounded from the coefficients
  (`translate.chebyshev._sampling_noise`). Changing variables at the exit, the SEP's
  other candidate, was not needed.
- **Why:** below that floor the samples carry no information about the function's
  variation, so no refinement can reduce the tail; the table is evaluated at the same
  floating-point z. On smooth functions the floor is orders below the tolerance and
  changes nothing. The exit is a real choked steady flow (3.9e-4 bar, 0.2 % of the
  peak), not a vacuum. There the gas adds little to the waveguide's dispersion and its
  nonlinear response is negligible, so an absolute tolerance would also stop the
  refinement. It was not added: the acceptance test is relative per element (with a
  comparison floor of 1e-10 of the table's largest element) so that weak bins keep their
  accuracy, while the noise floor needs no parameter and covers any function sampled
  where floating-point z is coarse, not only gas density. An absolute tolerance on the
  density paths remains open as a complement.
- **Consequences:** the canonical capillary's table has 30 intervals instead of 123 and
  translates in 0.3 s instead of 0.8 s; Φ is unchanged to 3.6e-13 rad of 850. Routes that keep the interval cap (a coupled flow, a reject-free
  controller) take longer steps near a choked end.

## ADR-041 — One Adams family, two formulations: interaction and exponential
- **Status:** accepted ([SEP 16](seps/archive/sep-0016-unified-multistep.md)); the default
  formulation is ADR-045's
- **Context:** SEP 15 found that interpolating the interaction picture's transported
  source limits the linear phase an Adams step can take in a driven bin to about one,
  and that interpolating the untransported source and integrating the polynomial
  exactly against the linear flow (exponential Adams) removes that limit on constant
  operators. Both are the same predict–evaluate–correct recurrence with one call a
  step, the same history lifecycle, controller, budget and observations.
- **Decision:** `Multistep.formulation` selects `"interaction"` (SEP 13's method) or
  `"exponential"`. The march is shared (`solprop.backends.multistep.AdamsMarch`); the
  exponential formulation stores untransported samples in the same history store,
  evaluates its source through the protected hook `StepDriver._adams_lab_source`, forms
  the moments $k!\varphi_{k+1}(h D)$ once per trial (`solprop.backends.exponential`) and
  applies every formula — predictor, corrector, the neighbouring orders' defects, a dense
  observation — in the monomial basis of its interpolant, so no per-node weight volume is
  formed. The method's representation names its formulation, and so does its name
  (`exponential-adams-pec5/6-variable`). Varying operators take ADR-042's compensation and
  coupled flows ADR-044's treatment; CF4 sections and the comparison variants are refused.
- **Why:** one family keeps one set of settings, validation, budgets and lifecycle
  tests for both (SEP 16 §10 rejects a class per formulation). A separate hook rather
  than an identity anchor through `_adams_source` avoids the identity propagator and the
  repeated anchor work SEP 15's study paid, and leaves the transported hook, which the
  TDSE adapter and `native` implement, untouched.
- **Consequences:** a backend without `_adams_lab_source` refuses the exponential
  formulation by name. Switching formulation mid-run is not defined; a resume
  restarts the history under the run's own formulation.

## ADR-042 — The compensated reference for varying operators: the transported treatment
- **Status:** accepted ([SEP 16](seps/archive/sep-0016-unified-multistep.md))
- **Context:** the exponential formulation integrates its source exactly against a
  constant operator's flow. Along a varying operator the exact flow
  $U = e^{\Phi(z_2)-\Phi(z_1)}$ is available, but the source's polynomial is integrated
  against a constant exponential only. SEP 16 §4.2 rearranges exactly,
  $u' = B_nu + [F + (D-B_n)u]$ with the chord $B_n$ of each trial, and compares three
  treatments of the linear compensation $C_j = (D(z_j)-B_n)u_j$: interpolated with the
  source (untransported), integrated in the reference flow's frame (transported), or
  omitted (the chord). Scalar checks and full RDW and canonical capillary runs measured
  them.
- **Decision:** the transported treatment,
  $h\sum_j a_j e^{(1-\theta_j)hB_n}C_j$ with the ordinary Adams weights, in the
  predictor, the corrector, every scored defect and dense observation. The rate is the
  derivative of the representation that defines the flow
  (`translate.linop.phi_rate`), read from the side of an edge a step came from at its
  end. Each sample keeps the state it was evaluated at (the predicted one for PEC) and
  its rate beside its source. A structurally absent source takes the chord alone. The
  untransported treatment remains a test witness (`AdamsMarch(compensation=...)`); the
  chord is never offered as a mode.
- **Why:** the compensation is linear in the state, so it carries each free bin's own
  phase: interpolated untransported it loses accuracy once $|\operatorname{Im}D|h$
  approaches one and its error is under-reported to the controller where the
  compensation dominates; the chord's error is invisible to the estimate altogether.
  On the full physical runs the two compensations were equal, the transported one
  costing 6–14 % more time in the study's fused form.
- **Consequences:** two more state-sized rows per history slot (states and rates). In
  a driven bin whose state follows a smooth laboratory-frame drive the transported
  compensation is the less smooth one. The exponential-free form of the transport
  (SEP 16 §4.2) remains a cost target.

## ADR-043 — The Adams march hands its state-sized work to an engine
- **Status:** accepted ([SEP 16](seps/archive/sep-0016-unified-multistep.md)); amended by
  ADR-060, under which `native` executes the exponential march's controller itself
- **Context:** a march that does its field arithmetic on host arrays and asks a backend
  only for a source and an adoption makes `native` copy the state, the predicted field and
  every source between its engine and the host each step, keep the history in host
  memory and run the step arithmetic in NumPy: on a device each step crosses the bus
  several times, and on its CPU build the exponential formulation's per-element work
  runs uncompiled.
- **Decision:** `AdamsMarch` keeps the controller -- the stencil, orders, rejection,
  restarts, the call budget, the ledger, schedules and lockstep -- and hands every
  state-sized operation of its fused and exponential paths to an engine
  (`backends.multistep.AdamsEngine`): the startup sample; a trial's predictor, source,
  corrector and scored defect norms, held until the decision; the adoption and
  transport of an accepted step; holding and restoring the accepted state around a
  dense observation; the observation itself. Only scalars, the coefficient tables and
  the norms cross. `HostHistory` is the host engine, over the existing hooks and
  kernels; a backend returns its own from the protected hook `StepDriver._adams_engine`.
  The general path (PECE and the comparison variants) keeps host arrays.
- **Why:** one controller for every backend keeps the order, rejection and lifecycle
  rules in one place, tested once; a resident engine is then only arithmetic, checked
  against the host engine on the same trials. Splitting at a finer grain (a hook per
  pass) would leave the state crossing between passes.
- **Consequences:** an engine must reproduce `HostHistory` within the parity bounds of
  the compiled backends. A varying operator's per-sample rate is formed on the host
  (`translate.linop.phi_rate`) and handed to the engine, one row per accepted step.

## ADR-044 — The exponential formulation beside a coupled flow
- **Status:** accepted ([SEP 16](seps/archive/sep-0016-unified-multistep.md))
- **Context:** a coupled flow (a coherent capillary wall, a split block, a moving guide)
  is not a diagonal exponential. The interaction formulation would transport its history
  through the flow's inverse actions, whose conditioning grows with loss. Beside a
  coupled flow the cumulative diagonal is a multiple of the identity in the channel axis
  (the kernel contract's `coupled_flow` row), so a step's reference
  `B = (ΔΦ/h) I + G_s`, with `G_s` the frozen block of the step's section, commutes.
- **Decision:** the exponential formulation steps constant-Schur and frozen-residual
  sections. A trial's flow and moments are matrix functions of `hB` on the bins that
  carry a block (`backends.multistep.coupled_moments`): in the reference, from the
  exponential of an augmented matrix carried back by the block's Schur vectors. A
  frozen-residual section's residual is linear in the state and joins the transported
  compensation of ADR-042: each sample keeps its flow weights, a trial forms the residual
  against its own section's block, and every factor is a forward flow. The lab-frame
  sample is the material source alone. A backend supplies the block through the
  protected hook `StepDriver._adams_block`. The interaction formulation and CF4 sections
  stay refused; the untransported witness stays for diagonal operators.
- **Why:** only forward matrix functions of a frozen block are needed, which the flow's
  Schur factors already serve; a section change rebuilds the residual from the stored
  weights as a step-size change rebuilds the compensation, with no inverse action.
- **Consequences:** `numba_cpu` compiles the matrix passes from the flow's eigenbasis;
  `native` keeps them resident when every bin of the block's sections is on the eigenbasis
  route and the block has at most 16 channels, and otherwise runs them on the host. A
  section edge is not a kink unless declared; a declared kink restarts the history.

## ADR-045 — `Multistep()` is the default integrator; `RungeKutta()` names the march
- **Status:** accepted ([SEP 16](seps/archive/sep-0016-unified-multistep.md)); amended by
  ADR-055, which chooses each interface's default: `RungeKutta()` on `prop_gnlse`,
  `prop_waveguide` and `prop_free`, `Multistep()` on `prop_capillary` (an override of
  its rule), `prop_radial` and `prop_planar`
- **Context:** the exponential formulation runs on every route and backend, devices
  included, and took the fewest calls on every case SEP 16 measured. With the
  error norm over the whole active band it was, at matched error on one CPU thread,
  5.5 to 5.8 times faster than IP-DOPRI5 on the full RDW capillary, 2 to 3 times on the
  CtC launch, 1.2 to 2.4 times on the focusing beams and filaments and level to 1.3
  times on the canonical capillary, and on the native CPU build as fast or faster on every case;
  IP-DOPRI5 stayed ahead on the single-mode GNLSE window on `numba_cpu` (1.2 to 1.4
  times) and, on a device, on the canonical capillary and the coupled wall. The Adams
  history costs memory: 7 to about 60 further copies of the state, by backend and
  formulation.
- **Decision:** `SimulationSettings.integrator` defaults to `Multistep()`, whose own
  formulation defaults to `"exponential"`; `RungeKutta(tableau=DOPRI5)` names the
  interaction-picture march. The kernel contract is unchanged: translation passes
  `RungeKutta`'s tableau with no Adams integrator, and `Multistep` with DOPRI5 as the
  unused tableau. The method's recorded representation always names its formulation. A
  run that fits the machine's memory, or an accelerator's free memory, only without the
  Adams history is refused before it allocates, naming `RungeKutta()`; an adaptive Adams
  run at `rtol` of 5e-2 or more is warned on.
- **Why:** the expensive-source regimes, where the exponential formulation's call
  saving is time, are where most runs spend their time; the cheap ones lose little, and
  the memory refusal turns the one regime where the history does not fit into advice
  instead of a failed allocation.
- **Consequences:** a resume without a kept history (ADR-046) restarts it, so it
  continues within its tolerance of the uninterrupted run; replayed schedules meet the
  Adams method's smaller negative-real stability interval under the in-stage collar; a
  few fixed steps start at order one. Tests whose subject is the interaction formulation
  or the Runge–Kutta march name it.

## ADR-046 — A resume continues the Adams history
- **Status:** accepted; agreement to a rounding bound (ADR-049)
- **Context:** an Adams resume that restarts the history at order one charges a startup
  evaluation again and climbs back through the orders at short steps. With a source that
  takes minutes per evaluation -- the TDSE -- and runs that stop on a call budget and
  continue, that is the cost of every resume, and the resumed run is a different
  trajectory from the uninterrupted one.
- **Decision:** the march's state at an accepted position is a `MarchSnapshot` (the
  accepted state as the backend held it, the engine's rows, the entries, the order, the
  rejection count, the proposed step). The run loop keeps one beside each save plane and
  terminal checkpoint in an output that can hold it (`HDF5Output.save_march`, a group
  rewritten in place, its position written last); `SimulationSettings.resume_history`
  (default: while it is at most 256 MiB) decides. A resume at the snapshot's position,
  with the same method, installs it after loading the plane. The snapshot is outside the
  record and the configuration hash; a file without one resumes afresh.
- **Why:** the configuration hash already fixes the method, so the snapshot is pure
  execution state, as a terminal checkpoint is; continuing it makes a resume the
  uninterrupted run, to rounding, and spends no evaluation on it.
- **Consequences:** on `numpy_ref` and `numba_cpu` a resumed Adams run continues the
  uninterrupted one; `native` keeps its rows resident, does not export them, and resumes
  afresh (roadmap). Superseded in part by SEP 29 §6.6: `native` exports and installs
  its resident rows, and continues too. A file holds one snapshot, of the history's size (7 to about 60
  copies of the state), rewritten at every save.

## ADR-047 — Signals reach a run between its commits
- **Status:** accepted; "bit for bit on the host engines" is restated by ADR-049 as
  agreement to a rounding bound; the low-storage march, which accepts in place, guards
  its step in flight instead (ADR-050)
- **Context:** Python raises a signal's exception at whatever line the interpreter has
  reached. A run's commits -- a step's state and position, a plane and the march's
  snapshot, a terminal checkpoint -- span several statements, and an exception inside
  one left the state and its position apart (an Adams step was installed before its
  position moved) or a file half written. A scheduler's `SIGTERM` killed the process
  outright, losing every step since the last plane: for a TDSE run, many expensive
  evaluations.
- **Decision:** every commit is a `committing()` section, and the Adams accept moves
  the position (and restarts at a hard kink) inside the same section as the state.
  During a run (`handled()`, installed by `run_problem` from the main thread over
  Python's defaults only), `SIGTERM` raises `RunTerminated` -- a `SystemExit` with
  status 143 -- and `SIGINT` `KeyboardInterrupt`, at once outside a commit and as the
  commit ends inside one. The run loop treats either as a failure and writes its
  terminal checkpoint and the march's snapshot; a second signal while it does so is
  dropped.
- **Why:** a trial in flight holds nothing that is committed, so abandoning it at once
  is safe and keeps the stop within a scheduler's grace period even when a step takes
  minutes; only the commits, a few statements each, must not be cut.
- **Consequences:** a terminated run resumes as an interrupted one does, bit for bit on
  the host engines, losing the step in flight. Native code is not interrupted: a signal
  waits for a native call to return. An application that installs its own `SIGTERM`
  handler keeps it, and a run started outside the main thread gets none.

## ADR-048 — GPL-3.0-or-later for the code, CC-BY-4.0 for the documentation

- **Status:** accepted
- **Context:** Pyroraptor, the optional TDSE engine, is GPL-3.0-or-later, and the
  solprop-native distribution wheel bundles FFTW, which is GPL-2.0-or-later. optmat,
  the materials data package, is meant to be shared by several projects.
- **Decision:** solprop and solprop-native are licensed GPL-3.0-or-later, and their
  prose documentation and figures CC-BY-4.0. optmat stays MIT. The repository follows
  the REUSE specification: code files carry SPDX headers, `REUSE.toml` covers
  documentation, figures and data, and `reuse lint` is part of the gate and the
  pre-commit hooks. Luna.jl's chirped-mirror
  data keep their MIT licence, and the vendored Sphinx inventories their projects'
  licences. GPU (CUDA, HIP) builds of solprop-native are built from source, never
  published as wheels.
- **Why:** copyleft keeps derived propagation codes open. Every runtime dependency is
  GPLv3-compatible (BSD, MIT, PSF, MPL-2.0, Apache-2.0, Apache-2.0 with the LLVM
  exception, GPL-2.0-or-later FFTW), and the three solprop packages share one licence.
  CC-BY-4.0 lets the text and figures be reused with attribution, and is one-way
  compatible with GPLv3, so code shown in the documentation can be copied into the program.
- **Consequences:** a program that distributes solprop, modified or combined with other
  code, is distributed under the GPL; running it and publishing its results are not
  affected. Every new file needs an SPDX header or a `REUSE.toml` entry. A published
  solprop-native wheel carries the licence texts of what it bundles and points to the
  tagged source. The CUDA and ROCm libraries are not redistributed with GPL binaries.

## ADR-049 — Reproducibility to the error bound

- **Status:** accepted
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)); supersedes the bitwise
  clauses of ADR-013 (batches bit-identical to sequential runs), ADR-022 (bitwise
  repeatability and thread-count invariance tests), ADR-034 (pins for the reference
  build) and ADR-039 (`numpy_ref` resumes bitwise), and restates the "bit for bit"
  consequences of ADR-046 and ADR-047; amended before the first release for device
  builds (the last bullet)
- **Context:** solprop required bitwise agreement in several places: two runs of one
  backend, one thread against many, a batch member against its solo run, a scan point
  against its run, a few kernels against the reference, and a `numpy_ref` resume. Backend
  agreement with `numpy_ref` had long been a rounding bound. What remained constrained the
  kernels (a lagged ring-buffer band plasma column, a serial single-node column, order-fixed
  reductions and GEMMs, no contraction, small FFTs on one thread partly for invariance),
  kept platform pins valid on one platform and one FFTW build and skipped elsewhere, and
  set bit identities as success criteria in open proposals. A propagation's answer is only
  as good as its discretisation error, many orders above the difference two summation
  orders make, and no use of solprop has needed a bitwise repeat.
- **Decision:** independent floating-point computations are compared by numerical error,
  not bits. No bitwise agreement is required across backends, builds, machines, thread
  counts, batch membership, scan execution or repeated propagation runs. Matched discrete
  computations (the same method, orders, startup and restart history, schedule and inputs)
  are held to rounding bounds fixed in advance from operation counts, magnitudes,
  conditioning, transforms and propagation sensitivity; adaptive runs, ordinary replay and
  resume are held to per-case accuracy envelopes against a tighter or stored converged
  reference, since `rtol` and `atol` alone do not bound the global error. A replay meets
  the run's accuracy target and need not reproduce its trajectory. Kernels may reorder,
  vectorise, thread and contract: contraction is on by default, the protected helpers
  (stable increments, small-argument functions, defects, compensated sums) opt out and stay
  strict, reassociation is enabled per kernel on the loops that need it, and umbrella
  fast-math, `nnan`/`ninf`, `nsz`, approximate functions or reciprocals and flush-to-zero
  are forbidden. Exact structural contracts stay exact: lossless copies and storage,
  integer and identity metadata, hard-mask zeros and explicitly exact no-ops; accounting
  for actual work is exact while adaptive work counts may differ.
- **Why:** bitwise identity establishes neither accuracy nor correctness, and keeping it
  cost kernel speed, test infrastructure and test coverage off one platform. ADR-020's rule
  that no optimisation may move a validated result beyond its stated tolerance is kept
  word for word; bitwise identity was never part of it.
- **Consequences:** tests fall into five classes, structural (exact), rounding (R),
  accuracy (T), recorded state and noise ([testing.md](testing.md) section 3), with the
  shared bounds in `tests/bounds.py`; the platform pins, the FFTW witness, the native
  bitwise branch and the need for a particular pyFFTW build are removed. A rounding
  difference may flip an accept/reject decision, so adaptive step counts are diagnostics
  held to bands. `REFERENCE` keeps its name for its tidy and sanitizer role. There is no
  deterministic mode. Kernel changes that bitwise agreement would block are each
  qualified to their rounding budget and to SEP 16's work–precision cases.
- **Amendment, device builds:** the protected helpers are strict on CPU builds only.
  CUDA and HIP builds leave the strict blocks empty (`SOLPROP_NATIVE_STRICT_FUNCTION`
  and `SOLPROP_NATIVE_STRICT_BLOCK` in `native/include/solprop_native/fp.hpp`, under
  `__CUDACC__` or `__HIPCC__`, which includes nvcc's host pass) and follow their
  compilers' contraction defaults; `build_info().optimization_policy` records
  "compiler-default contraction". On a device the φ series, the phase sums, the
  compensated defects and the ionised fraction may therefore contract; lossless copies
  and sums are unaffected. Device accuracy is bounded by the measured device envelopes
  of the GPU tests (`pytest -m gpu`) against the oracle, not by the strict-helper
  argument. Round-to-nearest intrinsics in the protected helpers on devices (CUDA's
  `__dmul_rn` and `__dadd_rn`, Clang's contraction pragma on HIP), with a device test
  that a protected block does not fuse, are on the [roadmap](roadmap.md), after the
  first release.

## ADR-050 — A replay-only low-storage march, accepted in place, with no estimate

- **Status:** accepted
  ([SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md));
  amends ADR-009's stage clause and, for this march only, ADR-047's premise that a step
  in flight holds nothing committed
- **Context:** a Cartesian free-space state can take gigabytes, and the Runge–Kutta march
  holds 15 or 16 copies of it, the Adams march 32 to 44. A fixed-step run calibrated on its
  observable from outside (ADR-019) needs neither an embedded estimate nor the stage rows
  an adaptive pair keeps. RK4 in the midpoint interaction picture (Hult's RK4IP) needs
  three registers beside the evaluation's scratch.
- **Decision:** `LowStorageRungeKutta()` is a third integrator, executed by
  `solprop.backends.low_storage.LowStorageMarch` through an engine that holds the
  registers (`LowStorageEngine`; the host one is `numpy_ref`'s oracle, a backend may keep
  its own through `StepDriver._low_storage_engine`). It **replays** a prescribed schedule
  and nothing else: it is refused without a schedule, with interpolated planes, beside a
  coupled flow or the TDSE response, past RK4's negative-real interval 2.785, and on a
  backend that does not execute it. It **accepts each step in place**: the state is
  overwritten by its interaction-picture image at the first pass and by the accepted
  step at the last, and the position moves with `_finish` in one commit. It **records no
  estimate**: the step history's norm is 0.0, meaning "not estimated", so
  `met_tolerance` is vacuously true and the log lines that print a tolerance ratio print
  "not estimated (replayed schedule)"; divergence is detected on the accepted state's
  norm. Its source evaluations are counted as `source_evaluations`, never as logical
  `response_calls`. The step **in flight** is guarded: from the step's start to its
  commit the backend refuses every entry that reads or advances its state --
  `observe`, `propagate_to`, `march_snapshot`, each backend's `stats`, a lockstep runner's
  replay, statistics and observations -- with a `BackendError` naming the step; only
  `set_initial` clears it, and the run loop writes no terminal checkpoint for such a
  failure and says why.
- **Why:** dropping the estimate frees the storage an embedded pair costs, and an
  observable calibrated by three-level Richardson certifies what a whole-field norm
  cannot (a weak band under a strong pump). The 0.0 keeps every record's form and
  passes every finiteness check; NaN would conflate absence with divergence, and an
  increment proxy would score `O(h)`, not the local error. The guard costs nothing and
  covers an exception as well as a signal; holding signals for a step defers a
  scheduler's stop by up to a step and covers no exception, and a fourth register costs
  one state.
- **Consequences:** an exception between a step's first pass and its commit loses the
  terminal checkpoint: the file ends at its last plane, from which a resume continues to
  rounding, and `step_planes` bounds the loss. The same holds for a divergence, which a
  replay of the Runge–Kutta march catches before it writes the state. A resume rebuilds
  the half-step flow table from its own first step and agrees with the uninterrupted run
  to rounding. The kernel contract and the output schema carry the kind.

## ADR-051 — No compatibility with earlier configurations, and one TDSE source form

- **Status:** accepted; withdraws the source filter of
  [SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md) and
  [SEP 26](seps/archive/sep-0026-edge-free-tdse-source.md); one exception added on
  2026-10-28 (`collar_placement="inside"`)
- **Context:** before a first release there are no external output files to keep. An
  option or reader whose only purpose is to reproduce an earlier form of a configuration
  or record is a second code path to test and document, with no scientific use.
- **Decision:** solprop keeps no such options. Every `tdse_native` source is gated and
  edge-free, by its atomic window's gate or the grid collar, and admitted by the
  steady-current certificate alone; the cubic replacements are never gated. A TDSE
  descriptor has one layout, `solprop.tdse-descriptor.v5`, and any other is refused. A
  file of another output schema is never resumed, through its resume key. A run record
  holds every field of every dataclass it serialises, at its default too: no field is
  left out to keep an earlier record's hash. Options with a scientific use stay:
  `ComparisonMultistep`, both Raman spellings, the Luna witnesses (`LEGACY_CONVENTIONS`,
  the trapezoid plasma rule) and `vacuum_end="floor"`.
- **The exception: `collar_placement="inside"`** (maintainer, 2026-10-28). It rebuilds
  the time-window geometry of before [SEP 21](seps/archive/sep-0021-absorbing-boundaries.md),
  the collar inside the request, bin for bin. It stays public because the Luna
  witnesses and the frozen cross-backend comparisons were measured on that geometry, and
  re-measuring them is work of its own. Tests pin it, through `tests/legacy_geometry.py`,
  only where a stored measurement depends on it ([testing](testing.md)); the others move
  to the default geometry after the first release.
- **Why:** one obvious way to do each thing. The ungated source failed every window
  screen it was tried on (17.8–19.4% of the flat-band source between 60 and 120 fs), and
  no scientific use of it was found.
- **Consequences:** an unwindowed native SAE run's lattice must pass the certificate. A
  change that moves an identity bumps the version it moves and adds no compatibility
  layer; before the first public release a resume-compatibility policy is decided
  (roadmap, *Release preparation*).

## ADR-052 — One thread budget per process, which every pool stays within

- **Status:** accepted
- **Context:** each piece sized its threads its own way. `numba_cpu` read the affinity
  mask and the root control group's `cpu.max`, rounded to nearest, but not
  `NUMBA_NUM_THREADS`, so a large problem failed at construction when that was below the
  CPUs (a 64-CPU Slurm allocation with `NUMBA_NUM_THREADS=8`). Its `SOLPROP_*_THREADS`
  variables forced a pool's size rather than capping it, so the scan executors held a
  worker to its share only by forcing every pool, small problems included. The native
  backend took `os.cpu_count()`, the machine rather than the allocation, and honoured
  any `SOLPROP_NATIVE_THREADS`. `Pool` had a third reader. Under `poe test`, 370 of
  1175 sampled pool-building tests started an 18-thread kernel pool beside 17 other
  workers.
- **Decision:** a process has one thread budget, the smallest of its affinity mask (read
  at import, before an OpenMP runtime can bind the main thread), the tightest CPU quota
  on its own control-group path (versions 1 and 2, rounded **down**, at least one) and
  `SOLPROP_THREADS`, read alike by both backends; `numba_cpu`'s kernels are also held to
  `NUMBA_NUM_THREADS`, Numba's own ceiling. Every pool's automatic size stays within it,
  and so does every explicit count (`threads=`, the per-pool variables,
  `SOLPROP_NATIVE_THREADS` for the Kokkos pool), clamped rather than refused. `SOLPROP_OVERSUBSCRIBE=1` lets an explicit
  count exceed the CPUs and the quota, never `SOLPROP_THREADS`. The pools take turns, so
  each may use the whole budget. The budget and the limit that set it are recorded with
  the pools in a run's metadata, with the same meaning on every backend. The executors
  set `SOLPROP_THREADS` per worker (a Slurm instance's controller keeps its whole CPU
  share, and its own instance slice is taken before its budget), and the test lanes set
  it to one per xdist worker.
- **Why:** a quota limits CPU time, not CPUs: a team that exceeds it is throttled for the
  rest of each period, and one that synchronises at every step stalls whole, so 1.5 CPUs
  is one thread (Go and Java round up for throughput workloads that tolerate it). A
  limit that only narrows cannot oversubscribe a machine from a shell profile. The
  oversubscription opt-in exists because thread-count invariance tests need more
  threads than a CI runner has CPUs; production never does.
- **Consequences:** the native backend cannot import solprop and sizes Kokkos at import,
  so it carries a copy of the reader (`solprop_native._budget`); `tests/test_process.py`
  runs both copies on the same control groups and environment. A process reads its own
  quota once. A test that compares thread counts lifts its budget explicitly.
  `SLURM_CPUS_PER_TASK` stays a `Pool` setting rather than part of the budget, since
  Slurm narrows the mask itself where it binds tasks. A quota is shared by every
  process in its control group, and no process can tell how many siblings it has: two
  solprop processes in one CPU-limited container (Kubernetes CPU limits, Docker
  `--cpus`), whether independent runs or scan instances sliced by
  `SOLPROP_SCAN_CPU_OFFSET`/`COUNT`, each take the whole quota. Each should set
  `SOLPROP_THREADS` to its share; `Pool` already shares its budget among its workers.
  Virtual machines (cloud instances included) and Slurm jobs set no quota by default;
  their limit is the mask.

## ADR-053 — Runtime type checks in one CI lane; the CI lanes sharded

- **Status:** accepted
- **Context:** the jaxtyping and beartype import hook (ADR-018) wraps every annotated
  function in the test session: each call binds its arguments and matches every array's
  shape. The reference backend's inner loops call small annotated functions millions of
  times. One medium test took 257 s with the hook and 22 s without; the whole default
  lane took 15 min with it and 7 min without on 18 workers, with 2.3 times less CPU.
  Every CI lane ran the default lane with the hook, on two or three xdist workers, so a
  pull request waited two to two and a half hours for its longest jobs. The coverage
  lane was cancelled at 99 % after two and a half hours.
- **Decision:**
  - `SOLPROP_TEST_TYPECHECK=0` leaves the hook out of a session (`tests/runtime_types.py`).
    The default stays on, so a developer's `poe test` checks as before.
  - In CI, the coverage lane runs with the hook, so every annotated call that the default
    lane makes is still checked once per run, on Linux, beside the measure of 100 %. The
    macOS, Windows, lowest-dependency and slow lanes run without it.
  - A test that asserts what the hook raises skips without it (`requires_typecheck`).
  - A test that reaches past the hook uses `unchecked()`.
  - Each long lane runs as parallel jobs, the shards of whole modules that
    `SOLPROP_TEST_SHARD=i/n` selects (`tests/shards.py`). They are balanced by measured
    seconds per module.
  - The coverage shards' data files are combined in one job, which requires 100 %.
- **Why:** the checks find annotation and shape errors in solprop's own Python code.
  Those errors do not depend on the platform or on the dependency floor, and one lane
  that runs every test sees every call. Repeating the checks on four platforms bought
  nothing and cost most of CI's time. Shards trade more, shorter jobs for wall time,
  which costs nothing on public runners up to the account's concurrency.
- **Consequences:** a shape error reached only by a platform-specific path, or only by a
  slow test, is no longer checked in CI. Run `poe test` locally, which still checks, to
  find one. The shard weights (`tests/data/shard_weights.json`) need refreshing when the
  suite's balance shifts: `tests/shards.py` rewrites them from the JUnit reports that
  each shard uploads. A stale table costs time, never coverage.

## ADR-054 — Single precision as an opt-in arithmetic, validated against its oracle

- **Status:** accepted; amends [architecture.md §1](architecture.md) ("Double precision
  throughout") and records an exception to ADR-020 for the `mlx` route
  ([SEP 33](seps/archive/sep-0033-single-precision-mlx.md))
- **Context:** Apple-silicon GPUs have several times the CPU's single-precision
  throughput and no binary64 arithmetic, and Kokkos has no Metal backend. Whether a
  problem survives binary32 depends on the problem, so it has to be measured, against
  criteria fixed before the measurement.
- **Decision:** binary64 stays the default and the only arithmetic of `numba_cpu` and
  `native`. Binary32 is an explicit per-run choice, `precision="single"`, under SEP 33's
  contract: the state, stages, transforms and responses in binary32; the controller,
  scalars and schedules in binary64 on the host; the linear propagator formed to
  binary64 accuracy and then rounded; every narrowed buffer with its own power-of-two
  exponent and the state at an exact scale. `numpy_ref`'s single-precision path is the
  oracle. The `mlx` backend is validated against that oracle, not against a compiled
  CPU backend in the same precision, because none computes in binary32 and Apple GPUs
  cannot compute in binary64.
- **Why:** an oracle in the same precision separates rounding from method error, and
  gives a later single or mixed mode of `native` or `numba_cpu` a target. ADR-020's
  first clause stands: no validated binary64 result moves.
- **Consequences:** the precision is a field of every run's hashed configuration. Routes
  and responses that have not been qualified in binary32, and noise models, are refused
  under single precision by name. The survival record and the tolerance floor are
  published with the user guide.

## ADR-055 — Noise on by default, and defaults chosen for each interface

- **Status:** accepted ([SEP 37](seps/archive/sep-0037-noise-and-stepper-defaults.md)); amends
  ADR-045 and ADR-008; the capillary's override of the selection rule requalified on
  2026-10-28
- **Context:** every run was noiseless unless it asked, every interface stepped with
  `Multistep()`, and the default tolerances were partly unexplained. A one-photon
  seed is part of the physics a run models; the integrator that is fastest at matched
  accuracy depends on the interface, measured with noise on and off.
- **Decision:**
  - `noise="input"` is the default wherever the run can carry noise, and the resolution
    is recorded (`RunMetadata.noise_resolution`, output schema 26). Single precision
    and the planar grid fall back to none, recorded and printed; an interior launch
    takes its spectrum as given. The modified model is an option: it equals the input
    model where the linear operator is unitary and is an unqualified approximation
    elsewhere. Whether it stays is decided with SEP 7.
  - Each interface takes its own default integrator and tolerances by a fixed rule
    (the fastest at matched accuracy among the methods every route of the interface
    admits) and a convergence protocol, with one override. `prop_gnlse` takes
    `RungeKutta()` and `rtol` 1e-7; `prop_waveguide` takes `RungeKutta()` and 1e-5;
    `prop_capillary` keeps `Multistep()` and 1e-6 for every representation, against
    the rule (below); `prop_radial` and `prop_planar` keep `Multistep()` and take 1e-5
    with no `atol` where the caller gives no controller; `prop_free` takes
    `RungeKutta()` and 1e-4 with no `atol`. A default never depends on the physics
    switched on, except the TDSE's, by the maintainer's decision, and the free-space
    noise.
  - **The capillary overrides the rule** (SEP 37 D11, the maintainer's decision). On
    the capillary cases the rule prefers `RungeKutta()` at its passing 1e-5: each
    candidate's time over the fastest, as a geometric mean, was 1.15 and 1.09 at one
    and sixteen threads against 2.15 and 2.26 for `Multistep()` when SEP 37 chose,
    and is 1.08 and 1.05 against 2.51 and 2.52 on the release's tracking cases. The
    Adams method stays the default because it wins the runs the capillary is mostly
    used for, and the TDSE response runs only on it. Requalified on 2026-10-28 on the
    tracking set's seven capillary cases and four worked examples (native CPU
    backend, 16 cores of an AMD EPYC 7B13, every run's field against a reference a
    hundred times tighter):
    - *Several resolved modes.* The source beats at the modes' phase mismatches,
      which the interaction picture must resolve and the exponential Adams method
      carries in its propagator: the Runge–Kutta march takes 3.6 times its one-mode
      steps on the four-mode RDW capillary, where it is 2.3 to 2.9 times slower than
      the Adams method at 1e-6 and 1.4 to 1.7 times at 1e-5. On the three-mode
      comb-to-continuum case it is 1.1 to 1.35 times slower at 1e-6; at 1e-5 it is
      faster, with five times the field error.
    - *Kerr-only single-mode runs.* The Adams method is 1.6 to 3.3 times faster on
      self-phase modulation in argon and 1.3 to 1.9 times on soliton
      self-compression in helium, against either Runge–Kutta tolerance.
    - *The cost.* On one mode or a mode-averaged one with plasma or Raman, the
      Runge–Kutta march is faster: 1.25 to 4 times at 1e-6 and 2 to 6 times at 1e-5
      on the tracking cases, and 1.3 to 2.9 times on the ionisation and Raman worked
      examples. Neither collar scheme changes the Adams steps there; the Adams method
      needs about as many source calls as the Runge–Kutta march, and on these cheap
      sources each of its steps costs up to four times a Runge–Kutta stage. The
      stepper guide names `RungeKutta()` for those runs.
    - *Accuracy.* Every run stayed within 0.15 dB of its reference spectrum, against
      the protocol's 0.5 dB. At equal `rtol` the Runge–Kutta march was the more
      accurate, two to ten times, on every case but the RDW capillaries (equal on
      four modes, five times less accurate mode-averaged); at 1e-5 it was 2 to 12
      times less accurate than the Adams method at 1e-6, except on the envelope grid.
    - *The collar.* With the in-stage collar back in every stage (ADR-057's quiet
      collar is opt-in), the capillary default costs 0 to 4 % more than with the
      quiet collar on by default, measured with the boundary monitor a run attaches;
      the GNLSE default costs 9 to 23 % more. Neither changes which method is faster.
- **Why:** the input model is the least bad noise model solprop implements, used by
  the literature and the group for years; the defaults follow measurement, not habit,
  and the one override follows the workloads the capillary serves.
- **Consequences:**
  - Every default run is stochastic unless it fixes `seed`. Tests and worked examples
    fix theirs, or pass `noise=None` where their lesson is noiseless.
  - The realisation is in the resume key, so a run that drew its noise without a seed
    is resumed by drawing it again from the seed its file recorded, and a scan draws an
    unseeded point's noise from the point's seed or one derived from the scan's name
    (the maintainers' ruling of 2026-10-28; `docs/guide/results/resuming.md`).
  - Low-energy GNLSE runs take more steps with noise (the seed's dynamics are
    resolved). Replayed schedules for noisy ensembles must be calibrated with noise on.

## ADR-056 — One thread per core, and idle native threads sleep at once

- **Status:** accepted; amends ADR-052
- **Context:** the thread budget counted CPUs, so on a machine with two hardware threads
  per core a default run took two threads per core. The native backend left OpenMP's
  waiting to the runtime, whose idle threads spin between parallel regions, and it runs
  several thread teams in turn (Kokkos', the threaded FFTs', the BLAS's). Measured with
  the tracking set (SEP 38): on 16 cores of an AMD EPYC 7763 the runtime's own waiting
  ran the four-mode helium capillary in 334 s against 13.5 s with idle threads
  sleeping, the argon blueshift in 432 s against 37.3 s and the radial filament in 72 s
  against 5.9 s; even 10,000 spins cost them 1.4 to 3.3 times, while the single-team
  supercontinuum gained 20 % from spinning. On 36 cores of two Intel Xeon Gold 6240 (72
  hardware threads), with the machine to itself, one thread per core was 6 to 31 %
  faster than two on every case, and at one thread per core the runtime's own waiting
  was faster than sleeping on every case (the supercontinuum 49 s against 94 s). On an
  Apple M5 Max the policy made no difference.
- **Decision:** the budget also counts the physical cores that hold the allowed CPUs
  (Linux topology, Windows' processor table), unless `SOLPROP_THREADS` is set, which may
  then ask for up to the CPUs. The native import sets `OMP_WAIT_POLICY=passive` before
  its runtime loads, unless the user set `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` or
  `KMP_BLOCKTIME`. The choice stays in the environment and is recorded in the run's
  metadata (`openmp_wait_policy`). Threads stay unpinned.
- **Why:** sleeping is never catastrophic; spinning is, whenever the threads that spin
  take cores another team needs. Spinning only from spare hardware threads was
  considered and rejected: the runtime reads the policy once, at start-up, and work
  that arrives on the machine later takes the spare threads. Pinning gained 6 to 30 %
  on a dedicated allocation but would put several processes' threads on the same first
  cores, so it stays the user's choice too.
- **Consequences:** a hyperthreaded machine runs half the threads it did, faster. A run
  on a machine it has to itself, with spare hardware threads, is up to 1.9 times faster
  with `OMP_WAIT_POLICY=active`, and single-team problems such as the GNLSE gain from it
  even without spare threads; the tuning guide says so. `numba_cpu` keeps its own
  passive settings for pyFFTW's and Numba's runtimes.

## ADR-057 — Temporal boundaries: the collar outside the window, a watched edge, a quiet collar on request

- **Status:** accepted; amends ADR-025 and ADR-037
- **Context:** [SEP 21](seps/archive/sep-0021-absorbing-boundaries.md) changed the
  temporal boundaries' defaults:
  - the collar lies outside the requested window (the grid widens to
    `trange / (1 - 2 collar_fraction)`, `collar_fraction` 0.1);
  - transform sizes default to `fft_size="auto"`;
  - GNLSE and spatial runs take the default collar;
  - the spatial routes integrate it by the split scheme (D14);
  - a boundary monitor watches the edges by default (D7, D13).

  D15 also left a quiet in-stage collar out of the stages. That collar was judged above
  the shot-noise floor, so the term was dropped while light about three times the floor
  sat in it. Under the default input noise, a 1 W, 200 fs pulse walking into the collar
  ended 1.6e-2 from the always-on collar's field inside the requested window, and up to
  4 dB from its spectrum above -80 dB.

  The monitor checks only committed states. A noise-free Multistep run taking 0.1 m
  steps carried light from a quiet collar (1.6e-13 of the energy) to 3.6e-8 within one
  step, unabsorbed, and ended 2.7e-5 from the always-on result, against 2.5e-8 with the
  collar in.

  The switch also contradicted SEP 21's own "no automatic change of scheme" and the
  rule that a speed-up valid only under conditions the run decides is opt-in.
- **Decision:**
  - **The defaults above stand.** The split scheme is the default on spatial routes;
    modal routes keep the in-stage scheme of ADR-025.
  - **The quiet collar is opt-in:** `BoundaryCheck(quiet_collar=True)`; the default is
    `False` (maintainer, 2026-10-28). When asked for:
    - it rests the collar after two consecutive checks find the collar's share of the
      whole field, floor included, below `QUIET` (1e-12);
    - it puts the term back at the first check that finds light;
    - while the term is out, no step is longer than the monitor's check interval
      (`length / 256`), and the monitor adds no removal;
    - the exponential Adams order choice ignores the collar's stability limit while
      the term is out, and holds the first step after it returns to that limit;
    - one rule, decided by the step driver for every backend, says where it applies: a
      binary64 modal stage with the in-stage scheme and a temporal collar alone. The
      low-storage march, coupled flows, the general modal route, a radial real-space
      collar and single precision keep the collar.
  - **The monitor's rules** (maintainer, 2026-10-28):
    - one rule from the launch on: the square root of a guard occupancy above the
      budget warns, and above `stop_factor` budgets stops the run, or refuses it at the
      launch before its first step. D5's separate launch threshold (`1e-4` of the energy
      in a guard or the collar) is withdrawn; light the launch puts in the collar alone
      is absorbed and reported with the removed fraction;
    - under the split scheme the removal is what each accepted step's split takes of
      the power it finds, `P(1 − e^{−αh})`, summed by the backend's own step finish at
      every accepted step, checked or not, on every backend and march (the native
      engine's included), and read at the next check; not read back from the state
      after the split, whose deep-collar residue `e^{αh}` multiplied into removals above
      the launch energy;
    - the monitor's reading is the backend's: at a check each backend reduces the
      state's energy and its four weighted energies where the state lives, and only
      those five numbers and the split's removal reach the host; `numpy_ref`'s host
      reduction is the reference, and the input-noise floor is read on the host once,
      at the launch (2026-10-28);
    - the record continues across a resume: no launch check, the totals, launch
      occupancies and rows restored against the original launch energy. The
      `BoundaryCheck` is recorded per execution and stays outside the resume key, so a
      run stopped at an edge can be resumed under a relaxed check; the quiet-collar
      opt-in is inside it;
    - the per-plane rows are written in the closing record only, not in the record
      rewritten at every plane;
    - `absorber=False` zeroes the collars' rates and keeps their room: one grid with or
      without an absorber, so the Raman tail taper and the held-plasma gate keep their
      samples.
- **Why:**
  - A run cannot know in advance whether its light stays clear, and an omission judged
    too early is an error nothing reports. The robust default is the collar in every
    stage.
  - Judged on the whole field, the omitted term acts on less than 1e-12 of the energy,
    so it moves the field by less than 1e-6 of its norm.
  - The step cap keeps D15's premise: light must walk about twenty windows over the run
    to cross the collar between two checks.
- **Consequences:**
  - Default modal runs pay the in-stage collar's transform in every stage. D15 measured
    4.25 against 3.4 ms per step on the native GNLSE Raman/shock case.
  - The option acts only on noise-free runs: under input noise the floor alone fills
    the collar above the threshold. On a 2 cm noise-free GNLSE Raman/shock run whose
    pulse stays clear, it saved 6 to 9 % per step on `numba_cpu` and `native`.
  - On runs whose steps exceed the check interval, such as a short capillary under the
    exponential Adams march, the cap removes the gain. A per-step check that would lift
    the cap is on the roadmap.
  - The monitor's record counts `collar_left_out_checks`.

## ADR-058 — The 0.x releases are unstable; the first output schema is 1

- **Status:** accepted (maintainer, 2026-10-28)
- **Context:** ADR-021 promised nothing about stability until a first release, and
  nothing said what the first release promises. The exported surface is large
  (`solprop.translate` alone exports 61 names), and some internal seams are known to be
  provisional: the state's vestigial batch axis, the two step loops that one march
  protocol would replace, and the integrator's tie to `PIController`. Development output
  files carry schemas up to 27, and solprop keeps readers for them.
- **Decision:**
  - The first releases are 0.x. A minor release may break the API, the output schema or
    the resume key; every break is listed under "Changed (breaking)" in the changelog.
  - Before the first tag the output schema restarts at 1 and the readers of the
    development schemas go; the kernel's `SCHEMA_VERSION` is renamed
    `KERNEL_SCHEMA_VERSION`, so that the two numbers cannot be confused.
  - `solprop.translate`, `solprop.backends` and the internals of `solprop.output` are
    unstable within 0.x, and three seams are reserved for change: the state's batch
    axis, the step-loop protocol, and the split of the integrator from the controller.
- **Why:** breaking costs nothing before anyone depends on the package, and a promise
  made now would freeze seams already known to need work.
- **Consequences:** a released version cannot read a development output file; keep an
  older install for those. The resume-compatibility policy for 1.0 stays on the roadmap
  (*Release preparation*).

## ADR-059 — Host vector maths on SLEEF for the native CPU kernels on x86-64

- **Status:** proposed with the change that makes it; the maintainers asked for SLEEF to
  be evaluated and adopted if it paid (2026-10-28)
- **Context:** a profile of the native exponential Adams trial found glibc's scalar
  `exp`, `cos` and `sin` about two thirds of its predict pass on a pressure-gradient
  helium case on an AMD EPYC 7763. A survey of the native CPU backend with Valgrind's
  callgrind (instruction counts, one thread, the same machine) put the library's share
  at 36 to 47 % of the Runge–Kutta stage entry, 60 to 74 % of the Adams phi-functions
  and compensation, 3 to 12 % of the band-rule plasma column and up to 69 % of the
  low-storage flow table; the stage entry alone is 12 to 36 % of a Runge–Kutta run on the
  single-mode capillaries. A microbenchmark of the complex exponential
  `e^a (cos b, sin b)` over 111,928 arguments taken from the tracking cases, against an
  80-bit reference, one core:
  - AMD EPYC 7763: glibc 15.7 to 30.3 ns; SLEEF 3.9.0's AVX2 functions 3.3 to 3.6 times
    faster at the same worst error (0.8 to 1.1 ulp); glibc's `libmvec` 2.7 to 3.7 times
    at 2 ulp, reachable from GCC only through fast-math.
  - Intel Xeon Gold 6240: glibc 9.8 ns; SLEEF's AVX-512 functions 2.9 times and MKL's
    HA pair 4.2 times, both at scalar accuracy.
  - Apple M5 Max: the system libm 3.25 ns; SLEEF's AdvSIMD functions gain nothing, and
    Accelerate's vForce 1.3 times at 1.7 ulp.
- **Decision:**
  - One private layer, `native/include/solprop_native/vmath.hpp`, gives the host kernels
    batch `exp`, `log`, `sincos` and the complex exponential over contiguous arrays. A
    kernel forms a block's arguments into a buffer, calls the layer once over it, and
    finishes each element from the results. Converted: the Runge–Kutta stage entry, the
    band-rule plasma column's rates and fractions, the exponential Adams predict pass
    (flows, phi-function exponentials, compensation) and the low-storage flow table.
  - On x86-64 CPU builds the layer calls SLEEF 3.9.0's u10 functions (within 1.0 ulp of
    the correctly rounded value over each function's whole domain): AVX2 where the
    target has it (the wheels' x86-64-v3 included), SSE2 otherwise. Its AVX-512F
    functions are opt-in (`SOLPROP_NATIVE_VECTOR_MATH_AVX512`): faster than AVX2 on an
    AMD EPYC 9B45 (Zen 5) and slower on an Intel Xeon of the Ice Lake generation. On Arm (Apple silicon, measured, and Linux aarch64, not
    yet) and in device builds it is the system libm, and each converted kernel keeps
    its per-element form there (`if constexpr (vmath::vectorised)`), which measured
    faster with the libm; device kernels keep their own functions.
    `SOLPROP_NATIVE_VECTOR_MATH=AUTO|SLEEF|SCALAR` overrides the choice and
    `build_info().vector_math` records it.
  - SLEEF is fetched at its release tag like Kokkos (`FETCHCONTENT_SOURCE_DIR_SLEEF` for
    an offline build) and built static at build time as its own CMake project: only
    `libsleef`, with the SSE2, AVX2 and AVX-512F (or AdvSIMD) variants, its own
    `-ffp-contract=off`, no IPO, and the C driver beside the build's C++ compiler. The
    wheel carries its licence (BSL-1.0) as `LICENSE-sleef.txt`.
  - The absorber split's factor `exp(-rate h / 2)`, which depends on the sample alone,
    is formed once a tile of samples for a block of rows, on every host route.
- **Why:**
  - The functions keep the error bound the rounding budgets already assumed of a library
    function (one ulp; [accuracy](accuracy.md) §7), so results change only by
    rounding, within the derived budgets, and no SEP is needed: SEP 25 keeps
    *approximate* functions off (its Q3), and SLEEF's u10 functions are not approximate
    in that sense; their accuracy over the kernels' argument ranges is tested against a
    50-digit reference. Steps taken were identical on every measured case.
  - SLEEF is one small, permissively licensed library with the fastest accurate vector
    functions on AMD and within 1.5 times of MKL on Intel, without MKL's size or a
    compiler-specific route. `libmvec` is less accurate and reachable from GCC only with
    fast-math; Accelerate's vForce gains a fifth on Apple silicon at 1.7 ulp.
  - Building it as a separate project keeps its contraction and target names its own:
    added as a subdirectory, its `common` target collides with Kokkos-FFT's, and this
    project's `-ffp-contract=fast` would follow its `-ffp-contract=off` and override
    it.
- **Consequences:**
  - A native CPU build on x86-64 and one on Arm differ in their transcendentals by
    rounding, as builds on different machines already could.
  - Every x86-64 build fetches and compiles SLEEF (about 20 s) and needs a C compiler of
    the C++ compiler's version.
  - Measured end to end (interleaved fresh processes, the tracking cases): see the
    [benchmark entry](benchmarks.md#host-vector-maths).
  - Open, on the roadmap: SLEEF on Linux aarch64, AVX-512 against AVX2 end to end, and
    the kernels still per element (the trapezoid and envelope plasma columns, the Adams
    corrector and dense observation, the separable stage propagators).

## ADR-060 — The exponential Adams march executes natively; solprop's is the reference
- **Status:** accepted ([SEP 39](seps/sep-0039-native-adams-march.md)); amends ADR-043
- **Context:** ADR-043 keeps the Adams controller in solprop and hands the state-sized
  work to an engine, so each trial of a native run crosses the seam several times and
  runs Python on the host: 150 to 210 µs per trial on an AMD EPYC 7763 before SEP 39,
  and 10 to 120 µs on an Apple M5 Max after its phase 1. On the single-mode capillaries
  a trial's kernels take 100 to 300 µs, so the host is a few per cent to a tenth of a
  CPU run, and most of a device run.
- **Decision:**
  - Every number the march's algorithm reads is a record solprop forms
    (`stepper.multistep.MarchPolicy`); its decisions are pure functions of the policy,
    the march's state and the trial's norms (`backends.adams_core`), and the kernel
    contract's section 11 specifies them in full.
  - `native` executes the exponential `Multistep()` on its resident engine for a run
    of attempts per call (`native/src/adams_march.cpp`, compiled without contraction
    or fast-math), returning only at a host event: a reached target, an accepted step
    on which the driver has work, the monitor, the call's bound (256 attempts or
    0.1 s), the budget, the floor or a native error. solprop books the call's attempts
    through `StepDriver._record_attempt` and `_skip_step`.
  - solprop's `AdamsMarch` stays the reference and steps every configuration the
    native march does not cover, with a `HostMarchWarning` that names the reason.
- **Why:** ADR-043's "one controller ... tested once" becomes one specification of the
  controller (the kernel contract), one reference implementation (solprop's) and one
  native execution, held to the reference by exact decision traces on prescribed norms
  and by accuracy envelopes on whole runs. Every choice a user sees or sets -- the
  method, tolerances, defaults, landings, admission, warnings and messages -- is still
  solprop's (`native/AGENTS.md`).
- **Consequences:**
  - The controller exists twice; a change to it changes the contract, the policy record
    (`MARCH_POLICY_FIELDS`, checked exactly) and both implementations together.
  - Within one build the two marches read the same tables and norms and take the same
    steps; across builds they differ by rounding, as any two builds do.
  - A native run's metadata records which march stepped it (`adams_march`).
