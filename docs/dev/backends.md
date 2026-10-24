# Backends: executing the propagation

Everything that touches the propagated state during a run is reached through one seam, and
only the backends know how the state is laid out, transformed and stepped. This document
fixes the seam, the reference backend, the numba CPU backend, the native backend (built from `native/`), the
threading model, the memory ledger and admission, and which backend runs what. Which
backend to choose for a run is the user guide's
[Choosing a backend](../guide/performance/backends.md). Decisions: ADR-002, ADR-003,
ADR-013, ADR-020, ADR-023.

## 1. The seam

Public simulation settings default to `backend="auto"`. `resolve_name` probes the
native package without importing it, resolves to `"native"` if present and otherwise
`"numba_cpu"`, and stores that name before execution or worker transfer. Explicit names
are preserved. `select` also accepts `"auto"` and logs the actual backend at INFO after
successful construction. Import, schema, device and unsupported-problem errors never
cause fallback. This selection policy changes no kernel-contract member (ADR-029).

The protocol is drawn at the level of operations on the state, because an `xp`-style abstraction over
`numpy`/`jax.numpy`/`cupy` would force every backend to keep the array compiler's structure
(out-of-place transforms, separate scaling passes, materialised stages, one reduction per
statistic), and that structure is precisely what a fast backend exists to change.

```python
class Backend(Protocol):
    name: str
    max_batch: int  # 1 for backends that step one problem

    def set_initial(
        self, field: Field, noise: NoiseRealisation | None, *, z: float = 0.0
    ) -> None: ...
    def set_progress(self, reporter: ProgressReporter) -> None: ...  # observer only
    def propagate_to(self, z: float) -> StepSummary: ...
    def observe(
        self, *, field: bool = True
    ) -> Observation: ...  # host arrays: field at z, step history
    def restore_call_ledgers(self, logical, atomic=None) -> None: ...
    def stats(self, spec: StatsSpec) -> dict[str, NDArray]: ...
    def metadata(self) -> dict[str, object]: ...  # what the execution was
    def memory_ledger(self) -> dict[str, int]: ...
    def available_device_memory(self) -> int | None: ...
    def close(self) -> None: ...  # idempotent
```

The listing is abbreviated; `solprop/backends/protocol.py` holds the exact signatures and
docstrings, in the order of [architecture.md §9](architecture.md#9-backends-in-one-page).

Rules: a backend is constructed from a validated `KernelProblem`
([kernel-contract.md](kernel-contract.md)) and host arrays and returns host arrays; its
buffers, layout and plans are private; the state is never a parameter of `propagate_to` or
`observe`; the numbers are the contract — a backend is correct when it reproduces the
reference to the stated bound. Selection is by name (`backends.select(name, problem)`);
availability is detected with `importlib.util.find_spec` (importing an accelerator package
can initialise a device runtime), never cached; an unknown name raises `ValueError` listing
what is usable, a known name whose dependencies are missing raises `ImportError` naming the
extra to install. `metadata()` says what the execution *was* — the name, the
resolved propagator mode, and whatever pools, plans and digests the backend has — and the
driver writes it into the run's `RunMetadata.backend`, outside the resume key.

**Periodic-statistics storage hooks.** The seam carries three storage hooks for the
periodic statistics — `_reset_step_stats`, `_record_step_stats`, `_drain_step_stats` — so
that a device backend can keep its rows resident and copy them once per plane; the shared
march still decides which accepted steps produce rows and passes their history indices.
The CPU backends inherit the host defaults. The hooks are part of the frozen surface
([kernel-contract.md §8](kernel-contract.md)).

**Batches.** `BatchBackend` adds independent member observations and metadata through
`select_batch(owners)`. `numpy_ref`, `numba_cpu` and `native` implement this
fixed-schedule interface. Singleton constructors keep their single-member shape; the
batch owns a sequence of independently loaded workspaces, fuses the accepted update over
members and leaves stage, response and FFT/BLAS work member-owned. A batch member agrees
with the same member run singly under the same schedule within a matched-discrete rounding
bound, on every backend
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)).
`available_device_memory()` returns free device bytes or `None` on hosts, enabling common
admission without importing backend-specific array libraries in the driver. On a device
the per-member launch cost is not amortised, so a batch's gain over independent runs is
small and a batch ratio for any shape needs its own measurement
([benchmarks.md](benchmarks.md#indicative-results)).

A general modal guide is a batchable member under a **narrower** admission rule than the
fibre one: the members of a general-guide group share the guide, the table, the gauge, the
materials and one explicit fixed schedule, and differ in their launch and their noise seed
alone. `modal_tables.identity`, the metric's kind and gauge and the compiled responses'
model and policy enter `driver.batch._signature`, because every other entry there
describes an execution *layout* and two different guides share all of it; and a group
assembled outside that grouping is refused by name, at `driver.batch`'s
`refuse_mixed_guides` and again in `backends/batch.py`'s `_validate_members`. The lockstep
runner hands each member its own singleton row, so a batched general run is what its
members are alone, to the same rounding bound. A **fused** batch axis carrying modal tables is refused by
`numpy_ref`, and a member carrying a coupled flow is refused by every backend's batch.

**Three things every backend owes, beyond the ten methods.**

**1. Reductions to the error bound.** The error norm and every statistic a backend reduces
are compared by numerical error, not bits
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): a reduction may be ordered,
chunked and threaded however measures fastest, provided its rounding bound, `γ_n` of the
sum of the magnitudes for a reduction of length `n`, is the one its tests assume. The norm
is *control flow*: it decides whether a step is accepted and what the next proposal is, so
a norm that changes in its last bit with the thread count can fork an adaptive trajectory.
That is accepted. Two adaptive runs whose trajectories may fork are compared by their
accuracy envelopes, and comparisons at rounding level run both sides on one prescribed
schedule ([testing.md](testing.md) section 3). `numba_cpu` computes in **two stages** —
one partial per column, `partials[3, n_columns]`, then a sum over each row in index order,
the columns threaded above a size threshold (§3.2) — and the device form is a fixed-order
tree reduction (a fixed block count, each block reducing its own contiguous range, then
one serial pass over the block results). Any other order (`parallel_reduce`, a threaded
tree) is admissible with the bound of its own tree. The tests in `test_numba_cpu.py` compare two runs, and one thread against all of
them, on a fixed schedule within the step's rounding bound;
`test_numba_cpu_kernels.py` holds the accept kernel and the weighted residual to the
reference's within theirs.

**2. The device form of the stage.** A stage's z-dependent quantities cross the seam as a
small bundle of host scalars (`translate.stage.scalars`) against a fixed set of arrays
uploaded once (`translate.stage.terms`); `ΔΦ` is one linear combination of the terms over
the bundle's floats and nothing about the representation reaches the backend
([kernel-contract.md §3](kernel-contract.md)). The contract's z-table fallback has no such
split, so `KernelProblem.device_ready` is false for it. NumPy evaluates the general
representation directly; Numba's host adapter evaluates it with preallocated Clenshaw stage
arrays. Numba's constant and parameter-expansion paths keep their own arithmetic. The
native CPU build gives the table the split itself (SEP 29 section 6.6): its coefficients
are uploaded once (`table_upload`, `include/solprop_native/z_table.hpp`), and before each
native call the adapter (`_ZTables`) has the table evaluated at that call's distinct
positions into term slots of the state's size (`table_evaluate`, a strict Clenshaw pass
per position), least recently used first so that a step's end is not evaluated again as
the next step's anchor. The stage weights select the slots: `ΔΦ` is `e_1 - e_0`, a value
`e_1`, so every consumer of the terms -- the Runge-Kutta stages, the low-storage flows,
the resident Adams phase and the noise -- runs unchanged, and an increment is one
rounded subtraction of two values, as on the host. The phase sums skip a zero weight,
which is exact for finite terms. A compensated Adams march reads `D` from the derivative
table into one more slot (`table_rate`). `Phi` and the prefactor are independent: either
may be a table beside the other's expansion. A device build refuses the table by name
through `translate.stage.require_device_ready`, which gives its size.

**Who evaluates the stage scalars and the Adams basis.** Two small host computations sit
in shared code on every backend's path: the scalar tables' Clenshaw rows at a step's
positions (`translate.stage.anchor`, `scalars` and `scalars_at`, through
`PiecewiseChebyshev.at_positions`, and an expansion's rate rows, `translate.linop.phi_rate`
and `rate_weights`) and the Lagrange basis of an Adams formula on its actual nodes
(`stepper.multistep.integrated_weights` and `lagrange_coefficients`). The shared code
takes the implementation from the backend that runs the stage, never from a
process-wide switch or from what is installed (SEP 29 section 6.6, R16): each
`StepDriver` subclass sets the class attribute `helpers` (kernel schema 34), a
`solprop._helpers.Helpers` record of the two operations, and the driver, the Adams march,
its engines and the native backend's resident engine pass it down. `numpy_ref` takes
`PLAIN`, the uncompiled Python forms, so the oracle never runs the compiled code it
checks; `numba_cpu` takes `NUMBA`, the same functions compiled; `native` takes its own
`NATIVE`, C++ transcriptions (`native/include/solprop_native/chebyshev.hpp` and
`lagrange_basis.hpp`). All three perform the same operations in the same order under
strict arithmetic, so the rows a table keeps for its last positions are shared whichever
filled them. Translation runs before any backend exists and keeps the shared reference:
`clenshaw` and `PiecewiseChebyshev.__call__` without a record, and the shared functions'
defaults, are Numba's. So is a z table's state-sized evaluation on the hosts
(`delta_phi`, and the rate of `phi_rate` on a z table), which is not a stage scalar:
`numpy_ref` takes it there; `numba_cpu` has its own host table, and the native CPU build
its own kernel (point 2). `tests/backends/test_helper_ownership.py` runs each march on
both backends in one process, with every other implementation made to raise after
translation, and a native z-table run of each march with Numba and the plain forms
forbidden.

**3. One step loop.** The march, the replay, the landing, the controller bookkeeping, the
history, the noise models, **the progress reporter and the statistics channel between
planes** are `backends/driver.py` and are not reimplemented. A backend supplies
`_trial_step`, `_apply_mask` and `stats`, overrides `_finish` if it has boundary treatments,
and hooks `_begin_plane` and `_on_accepted_step` if it has something to score.

`set_progress` attaches the observer the loop calls (the driver, not the backend, also
tells an `InterruptibleReporter` when its loop raised). `StepDriver` calls
`ProgressReporter.step` from `_record`, which every attempted step of every backend already
passes through, so a host backend gets the per-step channel for nothing; a backend whose
loop runs on a **device** -- where a per-step host callback would mean a synchronisation per
step -- is allowed to call `step` once per plane instead, with the plane's own totals. That
is the only place a backend may differ in what it reports. What it may never do is let the
reporter change anything: the calls are observers made after the step is decided, and a run
with a reporter is the run without one, to the rounding bound of any repeat of the same run
(`tests/backends/test_driver.py`); the reporter's calls are exact no-ops on the state.

**The statistics channel between planes** is the second thing the loop owns and a backend
does not. After every `period`-th accepted step the march calls the backend's own `stats`
with the periodic subset promoted to `names`, and appends one row — the position, the index
of the accepted step in this plane's history, and one small array per statistic — to a
buffer that `observe` drains into `Observation.step_stats`. Two rules make it sound:

- **The call site is the loop body, after `_finish` and after `z` advances**, not
  `_on_accepted_step`. The two marches are not symmetric there: `_march_to` records an
  attempt *before* it finishes the state and advances `z`, `_replay_to` after both, so a
  hook would read the previous accepted step on an adaptive run and the current one on a
  replayed one — and a replayed schedule is exactly what a cost measurement uses, so the
  wrong version measures clean and records a curve shifted by one step.
- **One implementation of every statistic.** The row goes through the same `stats` a save
  plane does, so a row landing on a plane is what the plane records, to the statistic's
rounding bound (the same implementation on the same state; a threaded reduction need not
repeat its last bit). Luna carries a
  host and a device implementation of nearly every statistic and lets them drift.

**The buffer** grows by doubling from `INITIAL_STEP_STATS_CAPACITY` and is reset per plane
**without releasing what it holds**, so the capacity a plane needed is the capacity the next
plane starts with — a measured estimate of a plane's step count rather than a predicted one,
and after the first plane a run allocates nothing per step. **The device form** is the same
arithmetic with the allocation on the device: rows accumulate in a device buffer of shape
`(capacity, n_scalars)`, the capacity seeded from the plane's expected step count times a
margin and doubled when it is reached, and copied to the host once per plane rather than once
per row. The per-row reductions are the fixed-order tree of rule 1 above. A backend that
cannot evaluate one of the subset's statistics between planes refuses the spec by name, as it
refuses a plane statistic it cannot form.

**What a backend may do to make it cheap** is read what the accepted step already computed
rather than computing it again — but only where the state it describes is provably the state
the run adopted. `numba_cpu` does this for the plasma pair: §3.1a has the measurement and the
predicate.

## 2. `numpy_ref`: the oracle

Plain NumPy and `scipy.fft`, one function per pass, no fusion, no in-place tricks, no
threading beyond what `scipy.fft` does with `workers=1`. It exists to be read and to be right:
every step of the anatomy in [architecture.md §8](architecture.md) is a named function with a
docstring stating the formula. It is slow (tens of separate array operations per stage) and
that is acceptable; it is the reference against which every other backend is validated, and
the backend that runs on any machine.

**The fused-pass route.** `backends/numpy_ref/passes.py` holds one pure
function per numbered line of [architecture.md §8](architecture.md) — `stage_input`,
`to_time`, `synthesise`, `nonlinear_sources`, `project`, `to_frequency`,
`stage_output`, `accumulate` — with the response stages `kerr_polarisation`,
`raman_polarisation`, `ionisation_rate` and `plasma_scan` beneath the fourth,
`raman_spectrum` beside them. Field plasma returns ungated current separately;
`held_plasma_spectrum` assembles its two products with the analytic collar derivative
([SEP 27](seps/archive/sep-0027-held-plasma-current.md)), while the physical-polarisation
helpers retain their independent oracle role.
The backend also owns the once-per-step `error_norm`, `accept_step`, `apply_hard_mask` and `apply_temporal_rate`;
`backends/numpy_ref/backend.py` holds the step loop and the seam. It propagates a
**single-member** problem on either grid — mode-averaged or multimode on the fixed
transverse quadrature, scalar or vector-isotropic, or a spatial geometry (radial,
Cartesian, planar) — with the Kerr term, the delayed Raman term and the plasma,
using IP-DOPRI5 with the cumulative operator, the lagged PI controller in both its weak
and weighted norms, landing on save planes and hard kinks, the modified shot-noise field
added in ω, the hard band mask and both of ADR-025's temporal-absorber schemes. Every stage
is evaluated: the tableau's first-same-as-last property would save one evaluation in seven,
at the cost of rebasing the recycled stage by `exp(ΔΦ)` and of invalidating it whenever the
absorber changes the state, and the oracle takes correctness over the saving.

**The general modal route.** A problem carrying `transform.modal_tables` does not go
through the fused pass at all: it goes through `backends/numpy_ref/general_modal.py`,
which synthesises a three-component analytic field at the guide's own quadrature nodes in
the *frequency* domain — a general guide's profile is complex and frequency dependent, so
a real static matrix in the time domain cannot carry it — evaluates the material's own
response there and projects the real polarisation back with the record's reciprocal map,
chunked over nodes by a ledger of its own. It is admitted in exactly one combination — a
field grid, a certified identity `linear.modal_metric` or the dense metric of a declared
unitary change of basis, the tables' own execution maps (at one interval of degree zero
for a constant guide; any sections and degree for a moving one, below), `n_pol = 1`, the
constant unit `power_weight` and each region's declared responses — the instantaneous
cubic in either form, and a delayed silica tensor Raman term beside it — over the
reference's three components — and refuses every other by name, before it allocates. The
real cubic is evaluated by its **symmetric contraction**, `symmetric_cubic`: the tensor
summed once per node and evaluation over the orderings of each monomial
`(E_j E_k) E_l`, `j <= k <= l`, and ten monomials a sample in place of the dense 81
products — exact for any tensor, and the order the compiled backends reproduce
(algorithms.md §7). A
guide whose regions are all linear short-circuits the pass exactly as the fused path
short-circuits a problem whose `response_order` is the accumulation alone. Everything
after the projection is shared with the fused path: the prefactor, the spectral collar,
the in-stage absorber and the return to the interaction picture.

**The coupled flow.** `numpy_ref` executes `linear.coupled_flow` from three producers:
the coherent wall block of a capillary (scalar, one `HE_1m` family, or vector, the two
radial families of one azimuthal order), the declared unitary basis of a *split* block on
the general modal route, and the connection of a moving solid guide. All take the same path
through this backend. Each section's method is one of three:

- **`"constant_schur"`.** The backend decomposes the leaf's frozen generator once per
  section, in its own constructor, and applies a Schur matrix-function action over the
  channel axis inside `_trial_step`, taking the inverse the interaction picture needs as
  the matrix function of the negated exponent rather than as a solve and measuring
  `norm(U) norm(U^-1)` of the whole step against the leaf's declared bound. The Schur
  factors never cross the seam. The vector wall block is the same section with a
  different dense generator; it carries no factored wall pair, which the backend admits
  as optional.
- **`"frozen_residual"`**, for a guide that varies along z. Each section carries the
  generator as Chebyshev coefficients, the block frozen inside the exponential is that
  expansion at the section's midpoint, and everything else — the generator's variation
  inside the section and the taper connection — is an **explicit residual** added to the
  same lab-frame source the material response is evaluated in, before the interaction
  picture's own factors carry it back. The residual is evaluated from the stage bundle's
  `flow_section` and `flow_weights` rows, and applied only on the bins the error norm
  scores (`grids.inband_mask`); the frozen flow alone carries the spectral collar.
- **`"cf4"`**. A step anchored in a CF4 section applies, for each stage fraction `c`, the
  commutator-free product `exp(B₂) exp(B₁)` of the whole generator `G − C` at the two
  Gauss nodes of `[z, z + c h]` — the node weights built from the leaf by
  `translate.coupled.flow_weights`, not from the bundle, because a node is not a stage —
  and its exact inverse `exp(−B₁) exp(−B₂)`, both by the same scaled Taylor series the
  Schur path uses; the source carries no residual there. The four factors' spectral norms
  are measured at every stage and refused above the section's declared pair bound. A flow
  whose declared fallback left some sections `"frozen_residual"` is executed step by step
  by the method of the anchor's section. The held actions are keyed on the step's start as
  well as its section and length, so the resident peak `memory_ledger` reports does not
  grow.

The route needs the cumulative `Phi` beside the flow to be a multiple of the identity,
because the two factors are applied as a product, and it checks that; a frozen-residual
or CF4 flow needs `Phi` only to be a **weighted sum of fixed terms**, and a problem whose
`Phi` took the z table is refused by name. FSAL is absent there as it is everywhere in this
backend.

**Moving maps.** A moving solid guide's synthesis and projection are one Chebyshev
expansion per section of the guide's path, and `numpy_ref` evaluates `S(z1)` and `Q(z1)`
at every stage as the weighted sum of their coefficients over the stage bundle's
`transform_weights`, in the seam's own evaluation form, before the source pass runs on
them (`GeneralMaps.at`). It admits them only beside a `"frozen_residual"` flow carrying the
connection on every section; a CF4 flow or none at all beside moving maps is refused by
name.

The anatomy of one stage — the node chunks, mode-averaged propagation as the one-node
case (ADR-010), the transverse components, the dual synthesis–projection pair, the fused
response pass and the quadrature alarm — is [stepper.md §1](stepper.md#one-step-pass-by-pass).
What this backend adds to it: the Raman convolution keeps one real history per scalar or
isotropic-only vector node, or three histories for a rank-two vector response, and forms
the spectra with `raman_spectrum` or `tensor_raman_spectrum`; it memoises them on the
**decay rates** on the one-component path. Multiple Raman components
share the selected histories: their spectra are weighted by their own amplitude
densities before summation, and the cache key contains
every amplitude density and every decay rate. The default mixture linewidth uses total
density, an explicit approximation. `eps0` is read here, by that pass alone: the
response writes it outside the convolution integral, so folding it into a contract row would
change the row's stated units.

It consumes the contract only through `linop.delta_phi`, `linop.unitary_phase`,
`responses.prefactor_at` and the scalar tables `responses.density`,
`transform.taper_scale`, `transform.aeff`, the plasma group's `preionisation` and the Raman
decay rates, so a change of the operator's representation cannot reach it. It refuses, by
name and with `BackendError`: an unknown `schema_version`; a spectral response other than
the compiled envelope-plasma assembly; scalar-only Raman on a two-component
problem; general z-dependent response data; a transform and a state that
would both carry the transverse components; and a batch axis above one.

Measured on the verification ladder: P0 exact to 4.8e-16 for a constant operator and
6.4e-16 for a z-varying one and independent of the step count; P1 order 5.10 and 5.82 over
three levels with the N = 3 recurrence at 1.7e-10 and its phase −π/4; P2 (independent
z-varying loss and dispersion, with and without a step in α) to 1.0e-13. A single step
matches a DOP853 reference at observed local orders 6.8 and 6.4, and self-phase modulation
is reproduced on the field and envelope grids to 7e-13 of the peak nonlinear phase. On the
multimode pipeline: a one-mode multimode run reproduces the mode-averaged one to 1.3e-16
(and to 1.1e-11 on a rule of nine nodes, which is the rule's own error), a chunk of one
agrees with all the nodes to 1.8e-15, a lossless two-mode run through the whole pipeline
conserves energy to 4.4e-16, the vector parity holds to 5.1e-16 at four angles, and the
plasma passes reproduce
`nonlinear.responses.PlasmaResponse.evaluate` **to rounding** — on its three analytic cases
and on one stage of an ionising gradient capillary composed by hand.

On the Raman convolution: `raman_polarisation` reproduces
the scalar `nonlinear.raman.RamanResponse.evaluate` **to rounding** given the same kernel, on both grids
for N₂, H₂ and silica with `thg` on and off, and a chunk of nodes agrees with the nodes one
at a time; `raman_spectrum` agrees with `translate.raman.RamanKernels.spectrum` to rounding at
three densities including zero, where H₂'s Dicke term diverges and the cap binds; one stage
of a hydrogen gradient capillary matches the two responses composed by hand to 3.4e-16, the
slack being the Kerr coefficient's own grouping; the small-signal gain of nitrogen's
vibrational line matches `-Im H(Ω)` to 0.4 % on a 140 ps window and 1.6 % on 70 ps, and
that kernel matches the line's Lorentzian to 1e-5; the soliton self-frequency shift matches
Gordon's rate to 2.4 % at T₀ = 800 fs; and the photon number of a lossless GNLSE run with
the shock term and Raman on is conserved exactly while the energy falls by 7.2e-6.

**Selection.** `backends/registry.py`: `available()` detects with
`importlib.util.find_spec` and never caches; `select(name, problem, *, grid, modes,
profiles)` builds one. The three keyword arguments are what `stats()` needs and the kernel
contract does not carry.

**Spatial statistics measure the medium's flux** ([SEP 34](seps/archive/sep-0034-medium-flux.md)).
The shared spatial statistics, `backends.spatial.collect_spatial`, read the plane's
coefficients through `problem.flux.at(z)` (kernel schema 35): energy and the far-field
spectra weight each column and bin by the real admittance `c Re k_z/omega` of that root,
and fluence, intensities and the on-axis spectrum synthesise the closure `G` from the
complex admittance before forming `Re(E conj(G))/2`, signed and not clipped. The widths
refuse a fluence negative beyond its rounding budget. The collar diagnostic,
`backends.spatial.collar_field_fraction`, stays an electric-field-norm share and is named
as one. A backend whose own statistics do not apply this measure refuses the schema.

## 3. `numba_cpu`: the compiled in-tree backend

### 3.1 Why numba, and what the measurements say

At solprop's sizes a stage is tens of microseconds to a few milliseconds. An earlier
prototype study measured that per-call overhead (numba dispatch 1–5 µs, `np.fft` allocation
and plan lookup, non-jitted JAX dispatch 50–100 µs) and the *number of passes* over the
data dominate, not FFT placement (§6). Luna's own step is instructive: of an isolated RDW
step, about 40 % was stage updates, error norm, windows and output and about 20 %
statistics — the non-RHS work that fusion removes.

So the wins are: fused passes (one traversal of t per column for the whole nonlinear term;
one pass for stage combination + propagator + noise + padding; one pass for crop + prefactor +
windows + back-transform + accumulation + norm), preallocated buffers and plans, no
materialised stages, two-stage reductions, and — measured before adopted — moving the
whole inner loop into compiled code. The whole nonlinear term at one node — the rate
lookup, the trapezoid plasma column's two running sums, the loss-current guard, the Kerr
cubic and the Raman drive — is **one traversal of `t`**, the plasma's sums carried in
scalars across the loop (`make_response_column`); the band column (§3.12) uses complete
row passes instead. The solution and the embedded estimate are running accumulators, so
nothing is kept beyond what the next stage needs (ADR-009), which is also why there is no
dense output by default. An isolated step timing is not comparable with a whole-run
average, which carries the once-per-run FFTW planning and the output's planes.

### 3.1a What exists

`src/solprop/backends/numba_cpu/` — `kernels.py`, `fft.py`, `backend.py` — over the shared
per-element bodies of `backends/bodies.py` and the shared step loop of `backends/driver.py`.
The public default is `"auto"`: the installed native backend first, otherwise `numba_cpu` (ADR-029).
`numpy_ref` stays selectable by name.

**On the full M2 case** (RDW VUV, 4 modes, 1761 accepted steps at Luna's tolerance) the
two backends reach *identical* achieved error: both return an in-band distance of
2.5941e-1 from Luna's field and 0.99534 of its in-band content, to five figures. Timings
against `numpy_ref` and Luna are in [benchmarks.md](benchmarks.md#indicative-results).

**Three response families**, which is all `kerr_coefficients` compiles to, each its own
compiled column:

| family | grid | `thg` | the column works on | plasma |
|---|---|---|---|---|
| `field_real` | field | yes | the real field | yes |
| `field_analytic` | field | no | the analytic signal | yes |
| `envelope` | envelope | either | the envelope, with the carrier when the harmonic is radiated | refused at compile time |

`field_analytic` is where the fusion pays twice. The reference forms the analytic signal
with a whole complex transform pair inside `kerr_polarisation`, and again inside
`raman_drive`, **once per node chunk**; it is the complex inverse of the same one-sided
spectrum the pad already holds, with the positive bins doubled, so the backend gets it from
the inverse plan it was running anyway. The reference keeps the explicit pair, because it is
written to be read.

**Agreement is 1e-15, not bitwise, and the reason is measured.** numpy's complex multiply
contracts into a fused multiply-add on some hosts (an Apple-silicon one, for example) —
`fma(ar, br, -(ai·bi))` reproduces `a*b` exactly and the naive form does not — where a
compiled kernel's need not. Writing the product out by hand does not close it; only
emitting the same FMA would, and that would reproduce an accident of one host's SIMD loop
rather than the formula. So one complex product costs 1.42 ε, `np.abs(z)**2` against a sum
of squares costs 1.4 ε, and a transform costs `log2(n)·ε`. Every comparison — the plasma
column's running sums against the independent NumPy scans under either running-integral
rule (§3.12), the statistics column's fraction and peak rate, two runs of `numba_cpu`, one
thread against many, a same-backend resume — is a rounding bound fixed from its operation
counts (the running sums by the prefix bound of their signed increments;
[kernel-contract.md](kernel-contract.md)), and a resume of an adaptive run is held to its
accuracy envelope ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)). With
the kernels contracting (§3.2), numba's products round once where the reference's may
round twice, and that is within the bound.

**Held field current** ([SEP 27](seps/archive/sep-0027-held-plasma-current.md)). The trapezoid column writes non-plasma polarisation
and total physical current separately. The band column writes current after its second
running integral and needs five per-node scratch rows. `field_plasma.HeldWorkspace`
owns the current, its two collared products, projected rows, spectra and all forward
plans. Modal routes project before transforming; radial routes transform and combine
at each node, crop, then make one frequency-space projection. Cartesian/line sources
combine before their transverse FFT, and planar sources before the y transform.
Partial chunks clear every row the fixed plans read. The memory estimate replaces the
reference's held-source allowance with the compiled workspace's actual array sizes;
provider-owned FFT plan heap growth remains a measured ledger row.

**Mixtures.** The compiled column takes a channel loop around the
plasma recurrence: one exposure scan per labelled channel with its own rate table,
potential, density and preionisation in a `(n_channels, n_nodes_chunk, 4)` scratch, the
free current and the ionisation-energy loss summed over channels, and the one-channel
case specialised to the single-species column, which it matches within the column's
rounding bound.
Several Raman components fold their density-weighted kernels into one convolution; one
component keeps the density outside. `tables.py` evaluates the general z table on the
host in the reference's Clenshaw order into persistent work arrays (agreeing with the
translation layer's to rounding), `populations.py` carries the per-channel populations the
five `species_*` statistics read (`stats.py`, one serial index-ordered reduction per
statistic), and the accepted stash has `1 + 2 n_channels` columns.

**The plasma statistics between planes are read off the step, not found again.** The
response column already forms the ionisation rate and the ionised fraction at every node
and every fine sample; specialised by one more `ColumnSpec` flag it also writes each node's
**final fraction and largest rate** for every plasma channel as it goes. The stash has
shape `(n_nodes, 1 + 2*n_channels)`, including its validity column, and the
statistics row is then a two-stage reduction over the nodes. The flag is a closure constant,
so a run without the channel compiles the two stores away and pays nothing, and the numbers
agree with the ones the per-plane path produces to rounding: the same compiled bodies over
the same nodes.

It is worth it because the per-plane path is not merely slower but *disproportionately* so
inside a step loop: `_nodes_at` re-synthesises every node into a buffer of megabytes, which
evicts the step's own working set. Measured on the reduced M2 case at `period = 1`, the
electron density alone cost **14 %** of a step through that path and **1.3 %** through the
stash, and the whole default subset went from 15.8 % to **2.5 %** — the difference between
missing the ten per cent statistics budget and meeting it with room. The
predicate for reading it is `first_same_as_last` — the seventh stage's *input* is the accept
pass's own result, so the last column evaluation of an accepted step scanned exactly the
state the run adopted — which is also false for the `split_step` absorber, where `_finish`
changes the state after the stages have run. Under the modified noise model the stash is
not compiled at all: the last stage's input is the adopted state *plus* the advanced noise
draw, while the statistics are defined on the committed field the reference scans, and
reading the stash moved the field run's peak rate by 2.3e-5 relative (`native` makes the
same choice). Anywhere else, and at a save plane whose field may not be the one the last
stage saw, the nodes are synthesised and scanned in full.

**Cross-backend resume.** A run continued on the other backend agrees with
the uninterrupted run on *either* to 2.2–2.7e-16, which is smaller than the 4.1e-16 between
the two uninterrupted runs — it shares half its history with each. Swapping backends
mid-run therefore costs nothing beyond running on either, and the digest does not need the
backend name for agreement: the name is out of the digest and the execution is recorded
beside it instead, in `RunMetadata.backend`, so a run started on one backend and finished
on the other is the same run and the file still says which produced it.

**FSAL** is **taken**, with the recycled stage kept across save planes (ADR-039); the
`split_step` scheme, whose `_finish` changes the state, gives it up. The identity, why the
hard band mask and the in-stage absorber preserve it, and the measured saving are
[stepper.md §2](stepper.md).

### 3.2 Kernel rules

- `@njit(cache=True, nogil=True)`. The kernels are specialised
  by closure constant and so cannot carry a written signature until the problem is known;
  they are compiled eagerly at construction by being run once on the buffers they will be
  handed, which is stronger than a hand-written signature because it cannot drift from
  them. numba's on-disk cache does key on the closure's contents — verified in-process and
  across processes, because a cache keyed on the source location alone would silently hand
  the second specialisation the first one's machine code.
- **Every kernel compiles through `solprop._numba_cache.njit`**, numba's `njit` with
  its own disk cache. numba's keeps one index per function and updates it without a
  lock: two processes saving different keys of one function at once (two closure
  variants, or two signatures, as an `xdist` run or a batch of jobs on one source
  tree does) can both take data file `.1.nbc`, leaving the index naming one key and the
  file holding the other's code, which numba then loads without checking its
  signature — a `TypeError` at unboxing when the arguments differ, the other variant's
  body silently when they agree. Twelve `make_raman_finish` variants compiled at once
  into an empty cache cross-wired an entry in three trials of four. The keyed cache has
  no index: each data file is named by a digest of the key's signature, target and
  closure cells and opens with a header of the numba version, the source stamp and the
  whole key, and a load that reads any other header recompiles. The header also
  carries a digest of what the compilation took from outside the function, which
  numba's own invalidation (the defining file's stamp) cannot see: the bodies a kernel
  inlines from `backends/bodies.py` and the constants it imports (`EPS0`, `PHI_JOIN`),
  both frozen into the machine code. The digest follows the names the bytecode references:
  each dispatcher reached contributes its function's bytecode, constants and names,
  each number, string or array its value, read from the code in memory, so a process
  that loaded an old body cannot file it as current. The same probe gave
  no wrong entry and every entry read back in eight trials;
  `tests/test_numba_cache.py` builds the cross-wired state and races real processes,
  and `tests/backends/test_numba_cache_policy.py` keeps numba's own `njit` out of the
  package, its tests and its benchmarks.
- **Not every kernel is disk-cached.** A parallel closure is
  compiled with `cache=not parallel`: numba names a parfor's gufunc from an object address
  and a per-process counter, never from the closure's constants, so two variants compiled
  in different processes can carry one symbol, and loaded into one process the second
  calls the first's body (a probe on `combine_and_pad` found 8 wrong bodies in 800
  processes). The response column, the statistics column and the envelope
  source column are not cached at all: they capture dispatchers, whose pickled form
  differs per process, so their key never matches and every process would append an entry.
  The factories that build them memoise in-process (`functools.cache`), so each variant
  compiles once per process rather than once per backend.
- Every compiled body and kernel carries `@no_type_check` beneath its compiler decorator.
  The test session's jaxtyping hook appends its checker *innermost*, so `njit` would
  otherwise compile the checker's wrapper and fail with `UnsupportedBytecodeError`
  (`DICT_MERGE`); `@no_type_check` is what the hook itself looks for and skips, so the
  annotations stay for readers and for basedpyright.
- Explicit eager signatures for the stage kernels (fixed
  dtypes and layouts so signatures do not multiply); `parallel=True` only on kernels that
  `prange` over an outer extent, with a serial variant selected at construction. Why the
  serial variant matters, and why the count is chosen per problem: the RDW step is fastest
  at **four** threads of eighteen and the CtC step with Raman at all eighteen; a `prange`
  over the four or six columns of the statistics pass was 14x slower than plain NumPy until
  it was written serially. The specialisation must also be at compile time: a Raman drive
  write passed as a kernel argument rather than a closure constant puts a runtime branch in
  the innermost loop, which cost the RDW step -- which has no Raman term -- 38 per cent.
- **The floating-point licence** is ADR-049's (contraction on by default, strict
  protected helpers, reassociation per kernel, umbrella fast-math, `nnan`, `ninf`, `nsz`,
  `afn`, `arcp` and flush-to-zero forbidden). On `numba_cpu`:
  - **Contraction.** The `solprop._numba_cache.njit` wrapper applies
    `fastmath={"contract"}` to every kernel. Its one hazard is exact cancellation
    (`x*y - y*x` need not be zero).
  - **Strict helpers.** A helper that relies on an exact cancellation or forms a small
    difference takes `strict=True`. numba's flags apply to a whole compiled function
    including what it inlines, so a strict helper is compiled separately and called, not
    inlined.
  - **Reassociation** licenses rewrites such as `(a + b) - a → b` that silently remove a
    compensation term, so a kernel asks for it explicitly
    (`fastmath={"contract", "reassoc"}`), on the loops that need it. The wrapper rejects
    any flag outside the licence, so the finite-value guards that detect divergence stay
    trustworthy.

  The evidence that a kernel is safe under its flags is its rounding-bound comparison
  against `numpy_ref` on stressing inputs (weak plasma signals, oscillatory currents, tiny
  phase increments, non-finite guards), not an audit. A rounding difference may change
  an accepted step and fork an adaptive trajectory; adaptive comparisons are accuracy
  envelopes for that reason ([testing.md](testing.md) section 3).
  - **Where it is enforced.** `solprop._numba_cache.licence` refuses anything else,
    and a strict helper asked to be inlined. The licence, and every callee's options,
    are part of the disk-cache key, so a kernel compiled under two licences keeps two
    entries and a callee that becomes strict invalidates its callers. The protected
    helpers are `fraction_at`, `envelope_population`, `envelope_axes` (whose
    determinant `xr yi - xi yr` must vanish for a linear field) and
    `envelope_moments` in `bodies.py`, passed as `define_bodies`'s `protect`; the
    integrated weights, phi-functions, correctors with their defects and the varying
    operator's compensation in `numba_cpu/multistep.py`; the Adams basis rows and the
    Chebyshev tables' Clenshaw sums. The phase increment itself is formed on the host
    in NumPy and is untouched, except on a separable free-space route, where the
    Adams trial's exponent comes from the propagator kernel's own element formula
    (`spatial.write_exponent`). `tests/backends/test_numba_licence.py` checks the
    refusals, the cache key, that every protected helper is strict and called, and,
    in the disassembly, a fused multiply-add in an ordinary body and none in a strict
    one. numba compiles its own library functions (the complex `exp`, for one) once
    per process under the first caller's flags; their contraction is benign.
  - **Measured.** Reassociation on the band column bought nothing measurable, so no
    kernel asks for it.
- Per-element mathematics lives in `backends/bodies.py` as a `define_bodies(decorate)`
  factory: bodies are defined inside a function parameterised by the compiler decorator and
  bind to each other through the closure, so a compiled family can never reach a host
  dispatcher. `HOST = define_bodies(njit(inline="always"))` in
  `numba_cpu/kernels.py`; `PLAIN = define_bodies(lambda f: f)` serves the coverage lane
  and tests. The native package in `native/` expresses the same operations in C++20
  device-callable bodies.
- Real `exp` × `sincos` for propagator factors, never a complex `exp`; purely imaginary
  prefactors applied as a swap; every rational Butcher coefficient precomputed as a reciprocal
  or product; the transforms' `1/N` folded into coefficients (each transform pair has its own).
- Reductions (error norm, statistics) are two-stage: per-column partials, then a sum over
  the columns. Their order is free
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)), and repeats and thread
  counts are tested against the reduction's rounding bound, not for identical bits. On
  `numba_cpu` the accept pass and the weighted residual thread their columns (a
  `prange`, each column's sum serial) for a state of at least `ACCEPT_ELEMENTS`
  (65 536) elements in at least `ACCEPT_COLUMNS` (8) columns: 2048 × 1024 elements ran
  3.6 times faster on six threads, while four columns of 8192 were slower threaded.
  Reassociating a column's masked sum was measured and gained nothing (the mask's select
  keeps the loop scalar; multiplying by the mask would turn a non-finite out-of-band
  value into a NaN norm), so it is not enabled.
- No allocation inside kernels; every buffer is a parameter; the `memory_ledger()` names them.
- No logging, no Python objects, no exceptions inside kernels; failure is reported through
  returned status codes checked by the driver.

### 3.3 Transforms

**The plan flag is size-dependent, and the default is earned on the grid that matters.** On
the modal pair `FFTW_MEASURE` executes within noise of `FFTW_ESTIMATE`. On the **doubled**
grid the Raman convolution runs on, `ESTIMATE` is **5.6x worse**, which more than triples
the whole CtC step. `MEASURE` costs 3-6x the planning time (hundredths of a second) and
must be paid wherever a Raman term runs.

FFTW's threading does not scale to the machine, and the best count is problem-dependent: the
RDW modal pair is 4.6 times slower at eighteen threads than at one, while the CtC step with
Raman is fastest at eighteen. So "the outer extent above a size threshold" below is too
simple, and the regime has to be chosen per grid from measurement.
Three buffer traps, each of which gives a plausible wrong answer rather than an error: a
complex-to-real plan **destroys its input**, so a zero-padded tail must be rewritten every
stage; the forward transform needs its own output buffer or it overwrites the pad; and
`FFTW_MEASURE` writes over its input while planning, so plans must be built before any
buffer is filled.

pyfftw plans bound to `pyfftw.empty_aligned` buffers and executed in place with `.execute()`:
a batched c2c pair on the modal fine grid (envelope) or r2c/c2r pair (field), a batched
r2c/c2r pair on the doubled node-chunk grid for Raman, a batched pair on the modal coarse
grid for the split-step temporal absorber, and a batched forward plan on the modal fine grid
for ADR-025's in-stage absorber term (`−alpha/2` times the modal field; on a radial run the
composed collar matrix is applied first, as one real GEMM through the interleaved real view
of a complex row). That term has input and output buffers of its own, because the stage's
finish still reads the polarisation's spectrum when the term is formed, and it is cropped
and masked into a coarse buffer held for the run: a stage allocates nothing. Every plan of
the modal family takes FFTW's modal pool and the run's plan flag. Plans are created once per
backend instance; `FFTW_MEASURE`
is the default (plans execute millions of times and small power-of-two sizes gain from
codelet selection), with `FFTW_ESTIMATE` selectable for tests and short runs, and a warning
that planning must happen on a quiet machine (the search is timed against the load: a
`MEASURE` plan made on a busy host has been measured 2.4× *slower* than `ESTIMATE`). Wisdom
import/export is explicit (`fft.py`), never at import time, and `fft.wisdom_digest` records
which wisdom a run's plans came from. Transforms run in place, because the buffers are
owned anyway and it helps cache residency (§6).
FFTW threads default to 1 for transforms below a measured size and to the outer extent above
it; numba's pool and FFTW's pool are set separately and both held within the process's
thread budget (ADR-052: the affinity mask, the cgroup CPU quota on the process's own path
rounded down, and `SOLPROP_THREADS`; never `os.cpu_count()`).

### 3.4 Threading regimes

| regime | when | inside a step | across problems |
|---|---|---|---|
| small single problem | resolved policy selects serial work | serial kernels, FFT threads 1 | independent scans: Local, or spawned processes through Pool |
| large single problem | above that | `prange` over node chunks in the fused response pass; FFTW threads on the batched plans; BLAS threads on the GEMMs | one process per problem |
| fixed-schedule fibre batch | compatible independently loaded members | fused accept update over members; remaining work stays member-owned | one batch runner owns the group |

Three measurements motivate the policies. They say the regime cannot be inherited from
the environment, and that three numbers matter more than the choice of backend:

1. **The backend owns all five thread pools, and they are not equal.** (The fifth is
   the transverse FFTW pool of a Cartesian run, sized from its own operand and
   overridden by `SOLPROP_TRANSVERSE_FFTW_THREADS`; a plan built for a different
   thread count is a different plan, so that override can move the last bits, as the
   temporal pool's can.) Three pools of N on N cores is 3N runnable threads, and the idle
   ones spin. On the RDW step at four cores of a two-socket, 36-core server, every pool at
   four with OpenMP spinning — which is what an unconfigured process does — took nine
   times as long as the sweep's best; `OMP_WAIT_POLICY=passive` alone recovered more than
   a factor of three, and dropping the BLAS to one thread most of the rest. **A factor of
   nine, from configuration alone.** Set `OMP_WAIT_POLICY=passive` always; give the cores
   to numba; give the BLAS one thread below the size at which its own threads pay. The
   GEMM is pathologically skinny in the mode count, so it wants a few threads and not the
   machine; `fft.choose_pools` sets the kernels' pool, FFTW's, the BLAS's and the process
   cap separately, each from its own threshold.
2. **That size is between the two cases.** Dropping the BLAS to one thread wins on RDW
   below eight threads and loses above it; on CtC + Raman it loses everywhere (by a factor
   of two at 36 threads), the GEMMs being large enough for the BLAS's threads to pay. RDW's
   synthesis GEMM is `33 x 4 @ 4 x 8192`, CtC's is `41 x 6 @ 6 x 65536`.
3. **The best thread count is not the machine's.** RDW peaks at about half the cores on both
   a 36-core server and an 18-core laptop; CtC + Raman uses all of them. And for the scan
   pattern, k independent single-thread runs beat one k-thread run by 8.5x, so a batch
   runner should prefer processes to threads; the scan executors do this through `Pool`
   ([scans.md](scans.md)).

The regime is chosen from element counts with explicit overrides and decides **the
kernels' pool only** — whether the node loop is a `prange`. The transforms and the GEMMs
are governed by their own measured thresholds, because a transform of 393 216 elements
repays its threads whatever the rest of the problem looks like. The choice and the thread
counts are logged and stored in the output metadata; the user-facing settings are
[Performance tuning](../guide/performance/tuning.md).

### 3.4a Measured fibre policies

Six construction-time policies refine the general rules for a **fibre** problem (no
spatial geometry: a radial or Cartesian run keeps the general rules). Each is recorded in
the backend metadata, and each was accepted on a predeclared cohort of ten-accepted-step
windows with zero changed field bytes and exact step histories:

| policy | domain | metadata | measured gain |
|---|---|---|---|
| the parallel kernel regime above 131 072 elements of `n_to · n_modes · n_pol · n_nodes` (the single-node and spatial cutoff is 1 000 000) | more than one node | `kernel_regime_elements` | −52 % reduced M2, −35 % reduced CtC |
| stacked fine/coarse real projection, packed per chunk at construction | field grid, embedded rule, 2–4 modes, ≥ 256 nodes, > 1 chunk | `modal_projection`, `stacked_projection_*` | −7.6 % over eight azimuthal shapes; one row +5.9 %, accepted under the aggregate rule |
| the Raman plans' FFTW threads sized on their own doubled-grid batch | Raman | `raman_fftw_threads` | −46.6 % reduced CtC |
| a persistent actual-width Raman tail plan | Raman, a chunk remainder | `raman_tail_nodes`, `raman_tail_fftw_threads` | −13.9 % full CtC |
| the ionised fraction reused while the rounded exposure is unchanged | one channel; a serial column, or one resolved component | `plasma_fraction_reuse` | −16 % / −29 % on its domains, production / one worker |
| the real-field magnitude dropped from two-component columns without plasma | Kerr only, `n_pol == 2` | `column_plasma_magnitude` | −11 % / −22 % on four HE shapes |

How these policies compare with the native CPU backend is measured, not assumed:
see [benchmarks.md](benchmarks.md#indicative-results).

Rejected by measurement: the OpenMP and TBB threading layers, a blocked temporal plasma
scan (42× slower on a single node), an eight-node time-outer traversal, `expm1`, a
reciprocal rate lookup, and every smaller kernel width on a 36-core server. The two rejected
traversals are kept as prototypes under `tests/prototypes/`, not in the package.
`OMP_WAIT_POLICY=PASSIVE` buys 14–39 % on a 36-core server's CtC cases at 18 and 36
threads. For pyFFTW's own OpenMP runtime it is the default on Linux
(`backends/numba_cpu/openmp.py`). pyFFTW's Linux wheels thread FFTW through a private
GNU OpenMP, whose idle threads spin after every transform and then hold the cores the
Numba kernels' pool needs. On eight cores that made every planned transform 35 to 60
times slower whenever FFTW had more than one thread, and the RDW step five times slower;
at the default pools the full CtC window, threaded at four by FFTW's size rule, took 1.4
times as long as with the passive policy, while RDW, whose transforms get one thread, did
not move. The import sets the variable for that
runtime only and restores the environment, and a user's own `OMP_WAIT_POLICY` or
`GOMP_SPINCOUNT` wins. For the Kokkos runtime and Numba's OpenMP layer it remains a
documented launch line.

### 3.5 The step driver: Python first, compiled when measured

**One loop, for every backend.** `backends/driver.py` is the march, the replay, the
landing, the controller bookkeeping, the history and the noise models, and every backend
inherits it. Per stage a backend asks the translation layer for one bundle of host
scalars and combines the terms it holds over them (§1); the driver applies the boundary
treatments of `_finish` to a step it keeps, so a rejected step does not pay the temporal
absorber's transform pair for a state that is thrown away.

The driver is Python calling ~5 fused kernels and ~4 plan executes per stage (≈ 50 calls
per step). **Measured, the per-call overhead is below the noise floor**: eight steps against
eight times one step differ by between -4.1 and +3.0 per cent on eleven of twelve
configurations, under the ten per cent gate; on the GNLSE soliton a step makes 39
Python-level dispatches into compiled kernels and plans, which is **1.69 % of a step at
1024 bins, 0.62 % at 4096 and 0.30 % at 8192**. A compiled
step driver — the whole `propagate_to(z)` loop in one `@njit` function calling FFTW through
a plan pointer (ctypes function pointers are callable from nopython code; `rocket-fft` is
the alternative) so a step is one Python call — is therefore not built; the backend
interface would not change if it were.

### 3.6 Propagator factors

`ΔΦ_i` for each distinct stage fraction is a Clenshaw pass; its exponential is one real `exp`
and one `sincos` per element. Tabulating the six factors per step costs up to 6 F of memory
and one pass each; recomputing them costs the transcendental twice per stage. An earlier
prototype study found the table wins on a transcendental-bound host and loses on a
bandwidth-bound GPU; at cache-resident sizes the answer may reverse again. Both paths are
implemented behind a flag. In a prototype with separate passes, tabulating won — 18 % of a
whole RDW step and 4 % on CtC without Raman, a wash with the Raman convolution.

**In the fused backend the answer is a property of the fusion.** Both signs are consumed by
the stage that formed them — the forward in pass 1 and the inverse in pass 7 — so
`recompute` evaluates `ΔΦ` exactly as often as `tabulate` does and the table buys residency
and nothing else: a step within 2 %, and a sixth of the propagator memory (128 KiB
against 768 on the soliton case). The flag's
default is therefore `"auto"` and the choice is the backend's: `numpy_ref` resolves it to
`tabulate`, because it forms two complex exponentials per stage and the table saves five of
them, and `numba_cpu` to `recompute`, because both signs come from one `exp` and a `sincos`
in the stage that consumes them. The resolved value is written into the run's `backend`
metadata group. The ledger test asserts the consequence directly: `numba_cpu`'s
`propagators` row is the reference's divided by the number of distinct stage fractions.

### 3.7 Coverage and testing lanes

- Default lane: compiled kernels; tests assert agreement with `numpy_ref` to tolerances
  derived from summation bounds (Higham: `n · ε` on the reduction length, then the measured
  margin), repeat runs and thread counts on a fixed schedule within the step's rounding
  bound, use-after-close errors,
  the memory ledger, and end-to-end file equivalence through both output handlers.
- Coverage lane: `NUMBA_DISABLE_JIT=1` with the `PLAIN` body family, so every line of every
  kernel body is executed by the interpreter and counted.
- Validation ladder per backend: one linear application, one nonlinear evaluation, one step,
  eight steps, each statistic, a full save interval, a full run.

### 3.8 The general modal route

`numba_cpu` executes the general modal route — `transform.modal_tables`,
`linear.modal_metric` and `responses.regional_responses` — in exactly the combinations
`numpy_ref` admits, and refuses the rest with the reference's own words: its admission
runs the shared list `solprop.backends.numpy_ref.admission.refuse_general_route` under
its own name, so a new refusal there is a refusal here. `linear.coupled_flow` is §3.9. The
route is a **second execution path**, `numba_cpu/general_modal.py`, beside the fused
pipeline and not an extension of it, for the reason §3.1a's three families make plain:
those are one real transverse matrix in the time domain, and this route's synthesis is a
complex frequency-dependent map of three Cartesian components.

- **Maps.** Derived by the reference's own `_general_maps`, so the two backends cannot
  read a record differently: `S`, `Q`, the per-node `chi3 T`, the form from `model_id`,
  and the delayed response's owners, tensors and weighted kernel spectra.
- **A fourth plan family**, `fft.GeneralPlans`: per node chunk, a complex inverse of the
  doubled one-sided spectrum (the analytic signal, rows `3q + c`) and a real forward of
  the polarisation; with a delayed response, a real pair on the doubled axis over the six
  stored pair histories (rows `6q + e`). A partial last chunk has its own plans on prefix
  views of the same buffers. Built once; nothing is planned or allocated inside a stage.
- **Kernels** over the per-element bodies of `define_bodies` (`general_kerr`,
  `general_kerr_component`, `general_symmetric_row`, `general_symmetric_sum`,
  `general_real_kerr`, `general_pair_drive`, `general_delayed`), with every reduction
  in this backend's own order, held to their rounding bounds
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): the synthesis over
  channels, the real cubic's symmetric contraction (its coefficients formed once per
  node, then ten monomials a sample), the analytic form's contraction over `k, l` then
  `j`, the delayed contraction over `j, k, l`, the projection over nodes and components
  into the source. Each is diagonal in the bin or sample (a matrix-vector product per
  bin), not a GEMM, so the loops run the bin innermost over contiguous rows; on a field
  grid the crop is the leading bins and the synthesis and projection index the fine
  spectrum directly (at one thread, a quarter to a third off a stage on the benchmark
  case). The `1/n_to` of the inverse transform
  and the `1/(2 n_to)` of the doubled axis are folded into a scale and the contraction,
  exact on a power-of-two grid. The node loops are a `prange` in the large regime; the
  projection reduces per output bin, so the result does not depend on the thread count
  (tested against the stage's rounding bound).
- **The stage** combines the state onto the coarse grid once, evaluates the source, and
  finishes through the same crop, prefactor, window, frame and accumulation kernel as every
  other route; a guide with no responding region skips the source, as the reference does,
  and ADR-025's in-stage absorber takes the legacy combine and inverse transform, which is
  what the reference's absorber reads. FSAL is taken, as everywhere on this backend.
- **Moving maps.** One compiled kernel, `combine`, forms each map's stage value from its
  coefficients into a stage buffer allocated once: the coefficients are flattened to one
  axis cut into 4 096-element blocks shared by the threads, each element summing its
  section's `n_deg` terms in order and skipping the zero-weight terms of every other
  section (thread-invariant; within the rounding of `n_deg` terms of the reference's BLAS
  product). The combination adds `2 u n_deg (3 kappa_S + kappa_Q)` to the source bound. A
  constant guide never combines and is the constant route exactly. Ten steps cost 1.75
  times the constant entrance guide's, about 30 % of that the combination's memory
  traffic.
- **Parity is a rounding statement** against `numpy_ref`, with the bound stated first
  (`tests/general_parity.py`): measured 1.5–3.5e-16 on one stage's source and 1.9–4.5e-16
  after ten fixed steps across six cases, against bounds of 2e-13 to 2.5e-12; with moving
  maps, ten steps agree to 6.8e-16 to 2.3e-15 against bounds of 2.0e-12 to 4.6e-11. Not
  bitwise: FFTW is not pocketfft, and numpy's complex product is an FMA on some hosts
  (§3.1a).
- **Statistics** consult the accessor whitelist with the flow's withheld names and hand
  the metric-weighted statistics the plane's metric, exactly as the reference does, so on
  one state the two agree to the statistic's rounding bound. The **lockstep batch**
  (`NumbaCpuBatchBackend`) takes general members unchanged, within the matched-discrete
  rounding bound of what they run alone.
- **Ledger**: the `GENERAL_LEDGER_ROWS` of `numba_cpu/general_modal.py`, named and zero on
  every other route; the maps and the record's resident E/H reference are counted as the
  reference counts them, the rest are this backend's persistent buffers and the plan heap
  growth. Moving maps add the coefficients (`general_synthesis`, `general_projection`) and
  the two maps a stage evaluates (`general_stage_maps`).

### 3.9 The coupled flow

`numba_cpu` executes `linear.coupled_flow` in the combinations `numpy_ref` admits but
CF4: the single constant-Schur section (the scalar wall, the TE/TM wall, the split block
on the general route) and the varying `"frozen_residual"` flow with its explicit residual
and the taper connection. A `"cf4"` section is refused by name at the backend and at the
capillary entry point — its Gauss-node positions are not stage positions, and no compiled
evaluation of them is defined — and so is a flow beside a Cartesian free-space problem,
which no translation builds. Its admission is the shared list under its own name.

- **Host-side, shared with the reference.** The Schur factors of every section's frozen
  block (`_coupled_flow_maps`, once at construction), the reference's own function on
  the same record, and the eigenbasis of each triangular factor
  (`solprop.translate.coupled.schur_eigenbasis`, once at construction, shared with
  `native`).
- **The matrix functions, compiled.** `form_flow_actions` forms every stage
  fraction's forward and inverse action of the step's section into the held buffers
  whenever the section or the length changes (the reference's memo key). A bin whose
  eigenbasis `W = Z V`, `Y = V^-1 Z^H` of `T = V Lambda V^-1` is well conditioned takes
  `W exp(+-l Lambda) Y`: `n` real exponentials, cosines and sines shared by the pair,
  and one `n³` product per action, where the reference sums a sixteen-term series of
  `n x n` products and back-transforms. A bin whose basis is ill-conditioned — a nearly
  degenerate pair with a nonnormal coupling, where `kappa = || |W| || || |Y| ||` grows
  as the inverse of the separation, or an exactly degenerate one, where `V` does not
  exist — takes the reference's scaling, series and squaring compiled, with its own
  squaring count. The gate is `kappa <= eigenbasis_limit(n)`, the `kappa` at which the
  eigenbasis route's formation bound `kappa (n + 11 + 2a) u` meets the reference
  algorithm's smallest, `16 u + 2 sqrt(2) n (n + 2) u` (2.04 at one channel, 4.17 at
  three), the point where the two routes' vanishing-step rounding terms meet. It is a
  heuristic gate, not a bound comparison at every step length; measured, the eigenbasis
  route is at least as accurate as the reference's on every adversarial case tried
  (nearly defective, strongly dissipative and nonnormal blocks). Why not Parlett's
  recurrence or divided differences: they are exact for separated eigenvalues too, but
  re-run the `O(n³)` recurrence for every length and direction, and near a coincidence
  need the blocked Schur–Parlett's reordering; the eigenbasis moves all `O(n³)` work to
  construction and shares one basis between the six fractions and both directions, and
  the gate sends exactly the bins a plain recurrence would lose to the
  scaling-and-squaring fallback. `flow_conditioning` measures each spectral norm of the
  whole step's pair by cyclic Jacobi rotations on the Gram matrix, plus the Frobenius
  norm of what is left off its diagonal (Weyl), an upper bound within a few `n u` of the
  singular value; the refusal is the reference's (`_refuse_flow_conditioning`), in its
  words. The squaring limit is checked on the section's largest row-sum norm times the
  length before either route runs. Nothing is allocated. On an adaptive varying run the
  actions are re-formed every trial, and this formation takes 7–11 % of a step (a serial
  kernel), where forming them on the host with the reference's functions took 26–59 %.
- **Compiled** (`numba_cpu/coupled_flow.py`, serial kernels): the stage combination
  mixed by the forward action (applied to the state and the combination separately, as
  the reference does), the pad onto the transform grid, the residual's combination over
  the stage weights minus the frozen block, the finish (the crop, prefactor, window,
  frame and in-stage absorber of the diagonal route plus `exp(−dPhi) R lab`), the return
  by the inverse action with both accumulations, and the accept pass through the whole
  step's action with the error norm's partials. The order of each sum is this
  backend's own ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)); the
  combination, the accumulation and the accept pass thread over bins (the accept over
  columns) from `FLOW_ELEMENTS` (32 768) elements, four channels of 8192 bins running the
  combination six times faster on six threads; off the block and at the zero fraction each pass agrees with the diagonal
  route's compiled arithmetic within its rounding bound (asserted against those kernels). A warm trial at a held
  length allocates nothing field-sized the diagonal route does not.
- **Parity** is a rounding statement with the bound of `tests/flow_parity.py` stated
  first: the actions are within 2.4 to 6.1 `u` of the reference's (bounds 83 to 174 `u`)
  and ten fixed steps within 1.0 to 2.0e-15 on five cases, bounds 2.2e-13 to 2.0e-12.
- **FSAL** stays off beside a flow, as on the reference ([stepper.md §2](stepper.md));
  the lockstep batch refuses a member carrying the leaf, as the reference's does.
- **Ledger**: `FLOW_LEDGER_ROWS`, named and zero on every other route; the factors, the
  residual's terms and the action pairs with the reference's sizes, then the stage's
  residual block, the lab-frame input, the source before the inverse action, the mixed
  error and the bin index.

### 3.10 The hybrid planar kind

`numba_cpu` executes the `"slab"` spatial kind — slab modes in x on the
transform's own real synthesis and projection, an FFT in y, columns `(p, k_y)` with the
channel fastest — in every combination `numpy_ref` does: field (real and analytic
families) and envelope, Kerr, Raman, field and envelope plasma, the in-stage and
split-step absorbers with the y collar's in-stage rate, a modified noise draw, and every
statistic. It is separable, so the backend's `_free` flag holds, and it branches off the
Cartesian path wherever its columns are not its nodes (`_slab`).

- **The stage** (`NumbaCpuBackend._stage_slab`) is the reference's `_nonlinear_slab` on
  persistent buffers. The stage input is combined onto the columns with the scale `1/n_y`
  and taken to its `(p, y)` rows by one unnormalised in-place FFTW transform over the y
  axis of the `(n_y, n_p n_pol, n_omega)` volume. The rows go to time a **block of whole
  y rows** at a time, as many as the node chunk holds (the reference's block), so
  `n_p n_y` rows are transformed each way, never `n_x n_y` nodes. From the one inverse
  temporal transform of a block both terms come back: the material source (the x
  synthesis as a batched real GEMM per y row on the complex rows' interleaved real view,
  the shared response column node by node, the x projection likewise, and the envelope
  plasma's two current streams projected beside it) and ADR-025's in-stage absorber
  (`−alpha/2` times the row, the y collar's rate per row plus the temporal one, through
  the Cartesian `absorber_columns` kernel). The absorber's rows go into a second half of
  the same column volume, so one forward y transform returns both, and the finish
  (`planar.finish_stage`) forms `(prefactor window material + mask absorbed) exp(−dPhi)`
  per column in the reference's order. The synthesis carries the pad scale, so both
  passes pack with `1/n_to` alone.
- **The operator** is evaluated per element and never tabulated (the contract refuses
  `propagator_mode = "tabulate"` for the kind). The shared body `slab_element` subtracts
  the column's channel's `kappa_sq` **last**, `q = ((k² − k_y²) + (k²_low − k²_y,low)) −
  kappa_p²`, the order of `separable_values`, and shares the root, evanescent clamp,
  frame and loss with the free-space element through `longitudinal_rate`, so free
  space's compiled arithmetic is untouched by the kind. The propagator, the noise phase
  and the stage's prefactor all use it.
- **The split-step absorber** acts on the rows with the coarse pair, chunk by chunk,
  then the forward y transform and the hard mask. **The statistics** of the field
  (energy, peak and on-axis intensity, the widths in y) are the shared `collect_spatial`,
  which synthesises the node volume itself, as on the reference, so they are the
  reference's to rounding on the same state. The plasma statistics take the stage's
  route (rows to time a block at a time, the synthesis into the node buffer, the compiled
  scan) and read the stash between planes. No quadrature alarm is scored, as on the
  reference.
- **Parity** is a rounding statement with the bounds of
  `tests/backends/test_numba_planar.py` stated first. What the ladder adds to the
  Cartesian one is the operator: numpy's and numba's complex square roots are each within
  one ulp of the principal root, so `Re k_z` differs by at most `2 eps k_max` and the
  phase of a run of length `L` by `2 eps k_max L`. Measured, the roots differ by one ulp
  in some bins; at `k ≈ 1e7 m⁻¹` over 10 cm that is 1e-11 to 1e-10 of the field, and it
  dominates every run's departure (ten fixed steps on eight cases: 1.5e-11 to 9.4e-11
  against bounds of 4.8e-10 to 1.5e-9; one evaluation within 2e-17 to 5e-16). Repeats,
  and one against four threads, agree within the same bounds.
- **Ledger**: `transverse_scratch` is the column volume, both halves beside an in-stage
  term; `slab_row_rate` is the y collar's rate per row (zero bytes on every other route);
  the temporal plans hold a block's `(p, y)` rows and the node buffers its `(x, y)` nodes.
  A warm stage allocates nothing field-sized.

### 3.11 The dispersive electronic response

`numba_cpu` executes [SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)'s
`responses.dispersive_kerr` on every route `numpy_ref` runs it -- the fused node path
(mode-averaged, multimode and radial), Cartesian free space and the line, the hybrid
planar route and the general modal route -- on field and envelope grids, with both
evaluators, both sectors, one, two or three components, mixtures, beside Raman, plasma
and envelope plasma. The operator is the reference's `G[w (G E)^3]`: the temporal collar
on the products, before `Π₊` and the output multipliers. Everywhere the multipliers move
off the nodes, to wherever the route already holds a spectrum:

- **The fused node path** (`backends/numba_cpu/dispersive.py`, `DispersiveWorkspace`).
  The legacy transform synthesises and projects with real matrices that do not depend
  on frequency, and the collar is the same at every node, so a spectral multiplier and
  the collar commute with both: `P G[w (G E)^3] = G[w P (G E)^3]`. The input multipliers
  act on the padded modal spectrum the stage combination already formed (no forward
  transform of the node field), one batched inverse transform gives every filtered modal
  field, the real synthesis GEMM puts them at a chunk's nodes, a kernel forms the
  products in the leaf's sector (closed forms for one component, the isotropic
  contraction for two, in `bodies.py`), and the projection GEMM takes each product to
  the columns of both rules. The collar multiplies the projected products, one batched
  forward transform takes them to frequency, and their output multipliers and the
  channels' `ε₀ γ₀ ρ`, summed per product (`weigh_terms`), land them in the stage's own
  polarisation spectrum after the stage's forward transform (`add_spectra`): the
  response has no inverse transform of its own.
- **Cartesian free space and the line** (`backends/numba_cpu/dispersive_spatial.py`,
  `SpatialDispersive`). Each node chunk's columns are its nodes and go to time and back
  on their own, so the response runs per chunk: the input multipliers on the chunk's
  packed spectrum, the inverse transform writing each filtered field straight into the
  block its product takes, the products in place, the collar, one forward transform and
  the spectra added to the chunk's.
- **The hybrid planar route**, likewise per block of `(p, y)` rows: the filtered rows are
  synthesised at the block's nodes by the slab rule in x, multiplied, and projected
  back to rows, as the bare field is.
- **The general modal route** (`backends/numba_cpu/dispersive_general.py`,
  `GeneralDispersive`). Its synthesis depends on frequency, so the input multipliers act
  *in synthesis*: on each node chunk's synthesised spectrum, one batched inverse
  transform giving the filtered analytic signals. The products are formed on the real
  field (the full sector, `kerr_form="real"`) or on the analytic signals (the
  conserving one) through the three-component isotropic contraction, at the nodes of
  each channel's region and zero elsewhere, and their spectra are added to the chunk's
  polarisation spectrum before the projection. The route has no collar. A guide whose
  only nonlinear region is a gas carries no regional tensor; its identically zero
  contraction and transform are skipped and the dispersive spectrum is written.
- **Alone** (no constant Kerr term, Raman, plasma or envelope plasma beside it) the
  response column and the polarisation's projection do not run, and the stage's forward
  transform is replaced by the response writing the polarisation spectrum. A symmetric
  response of one species whose product has the polarisation's dtype goes further: its
  product is the stage's own projected polarisation, the stage's forward transform is
  the product's and `g` finishes it in the stage's spectrum (`weigh_spectrum`) -- the
  constant cubic's transforms exactly -- and where nothing reads the bare field in time
  afterwards (no in-stage absorber on the fused and planar paths; always in free space,
  whose absorber is a pass of its own) it filters the stage's spectrum in place and
  holds no field-sized buffer. A symmetric mixture, one product under its mixture
  filter, takes the same path. Beside other responses the bare field stays theirs and
  the response holds its own buffers.
- **Mixtures.** The symmetric evaluator forms one product per mixture -- per region on
  the general route -- under the filter of the mixture's own Kerr diagonal,
  `g_mix = [Σ_s f_s g_s⁴ / Σ_s f_s]^(1/4)` with `f_s = ε₀ γ₀,s ρ_s`
  ([SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)). `StageWeights` holds the stage's own input and output rows and
  rebuilds a mixture's (`mixture_rows`, `O(n_omega × n_species)` and two square roots
  per bin) only when the densities differ from the last stage's: a fill of constant
  composition builds it once, a composition varying along z at every stage. The kernel
  is `mixture_filter`'s arithmetic, so the two backends' filters agree to rounding; on the fused
  path the stage weighs the channels in `prepare`, before
  the input multipliers filter the spectrum. The polynomial groups species with equal
  leg bounds, which share their derivative fields `D E` and `D² E` and add their
  coefficient tables density-weighted at every stage (`DispersiveTerms`): an exact
  rewriting of the per-species sum, so its transforms grow with the number of distinct
  leg bounds (krypton and xenon one; air's three species three). On the general route
  a group is also one region's, and each region has its own cube.
- **The polynomial reads the bare analytic signal**, which keeps the zero-frequency bin
  once where `Π₊` excludes it; the state carries some (the split-step absorber's product
  in time puts it there), so its constant is subtracted where the products read it.
- **The embedded rule's products** are projected at every stage and windowed,
  transformed and combined only when a step is accepted, from the last stage's buffers,
  before the alarm uses the embedded spectrum -- as the envelope currents are.
- **Node budget.** The translation's node chunk counts the response's buffers: one node
  buffer for a symmetric response beside others, none alone, two for the polynomial; a
  Cartesian or planar column adds its products and their spectra
  (`translate/problem.py`, `_dispersive_chunk`). The general modal route's chunk, the
  backends' own (`_general_maps`), counts two buffers per node and component for each
  filtered field and each product of its workspace
  (`numpy_ref.general_modal.general_dispersive_buffers`, `general_node_chunk`).
- **Parity** (`tests/backends/test_numba_dispersive_kerr.py`,
  `test_numba_dispersive_routes.py`): one evaluation 2–5 ε of the reference's on every
  route, ten replayed steps 0.5–8 ε; bounds from the transforms and node reductions both
  backends perform, 560 to 3500 ε. Beside the plasma column the bound adds what the
  column already disagrees by. Repeats, and one against four threads (the parallel regime
  forced), agree within the same bounds on the fused, free-space, planar and general routes.
- **Ledger**: the `dispersive_*` rows, named at zero on every problem; borrowed buffers
  are the backend's own rows.
- **Cost** against the constant cubic on the same problem, same-process medians:
  [benchmarks.md](benchmarks.md#dispersive-electronic-response).

### 3.12 The plasma column's band rule

[SEP 18](seps/archive/sep-0018-band-exact-plasma-integrals.md)'s band rule filters each of the
column's population and free-current integrands with the problem's $2K+1$ taps before its running sum, reading
$K$ samples ahead, so the fused one-sample traversal cannot carry it.

- **`make_band_plasma_column`.** The plasma runs in whole-row passes per node: the rate;
  the filtered rate, then the exposure's scan fused with the fraction, the loss guard and
  the weight $f\,q/m$; per component the free-current integrand, its filter, then
  its scan fused with the current. The held source consumes that current directly through
  its two collared spectra. Each filter is a pass over a finished row, its interior branch-free,
  and sums in the reference's order. The $K$ samples at each end close the window by
  point reflection (`filtered_at`).
- **Scratch.** Five rows of $n_{to}$ per node of a chunk, `(n_nodes_chunk, 5, n_to)`,
  the `plasma_work` ledger row: the raw rate (kept for the loss current and the
  stash), the weight, the guard, the integrand and the filtered row a
  scan reads. Several channels sum their weights and work into the same rows.
- **The fused column after it.** The physical current goes into its own complete
  output rows. The fused column is compiled without plasma or a preloaded source;
  it forms and collars only the Kerr/Raman polarisation. The trapezoid column
  writes the same two outputs. `HeldWorkspace` then forms `gJ` and `g'J`.
- **Shortcuts.** A stencil of exact zeros -- before an integrand first leaves zero
  and after it last does -- filters to exactly +0.0, the reference's value, so it is
  not read. The fraction is reused while the rounded exposure does not change.
- **Statistics.** The statistics column has a band twin, `make_band_stats_column`. It
  runs the first stage alone, in the same passes over two rows per node, and writes the six per-node parts of the run diagnostics
  after the channels' fractions and peak rates. Given the trapezoid's one tap, it also
  answers a trapezoid run that asks for the diagnostics.

**Verification.**
- **Against the reference.** On one chunk the band column agrees with `plasma_scan`
  and `plasma_current` within an absolute budget carried through the whole chain
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): the stencil's
  $\gamma_{2K+2}$ bound, the prefix bound of each scan's signed increments, and the
  fraction, guard, currents and density between them
  (`tests/backends/test_numba_band_bounds.py`). The stress set is pre-ionised
  oscillation, zero crossings, tiny exposure, near threshold, a long cancelling prefix,
  pulses at either edge, two channels, scalar and vector, $K = 4$ and $K = 0$. The budget
  carries the current's first loss sample too, and bounds the rate by the log-spline
  operand bound of [SEP 27](seps/archive/sep-0027-held-plasma-current.md). Relative
  checks (`tests/backends/test_numba_band_kernels.py`) sit beside it.
- **Routes.** Mode-averaged, analytic, multimode, mixture, radial, Cartesian, line and
  planar runs compare one, eight and ten fixed steps under both rules, with zero
  and ten-percent pre-ionisation. Their fields, plasma statistics and diagnostics
  use the equivalence ladder's bound with the sums widened by $2K+1$
  (`tests/backends/test_band_plasma_parity.py`).
- **Threads.** The parallel column agrees with the serial one within the same bound.

**Serial columns.** A segmented, time-parallel variant (local prefixes, the totals' scan,
offsets) was built and measured: numba's parallel regions ran the passes 1.8 to 6 times
slower per thread, and six threads only matched the serial column at 131 072 samples, so
single-node columns stay serial. The band rule's cost against the trapezoid's is in
[benchmarks.md](benchmarks.md#band-rule-plasma-integration).

**native CPU** runs the same design in its OpenMP traversal: current and non-plasma
polarisation have separate complete rows, with persistent transform plans for the
collared current products, and the statistics scan and diagnostics keep their physical
meaning.

**native device builds** run a different column, written down in
`native/include/solprop_native/plasma.hpp`: parallel filters and scans form the exposure and the free current, with the
pointwise population and loss-current steps between, on both the team and the tiled
device paths. The stencil keeps the reference's summation order, so it is deterministic
and independent of the team size; the scan associates differently from the serial sum, so
the device column is held to the device envelope of the parity bounds. Its filter reads
the halo directly from device memory; a build option stages each row's tile in team
scratch instead and gives the same bits. Which is faster depends on the vendor, and each
build defaults to its own: direct reads on CUDA, staging on HIP
([benchmarks.md](benchmarks.md#band-rule-plasma-integration)).

### 3.13 The far-field statistics' banded pass

[SEP 23](seps/archive/sep-0023-far-field-aperture-spectra.md)'s six far-field statistics and
`on_axis_spectrum` are one compiled pass over the stored state on the Cartesian and line
kinds (`numba_cpu/spatial.py`, `FarFieldPass`), replacing the shared host pass
`spatial.aperture_statistics` that `numpy_ref` keeps as the oracle. Stage one
(`make_far_field`) runs a `prange` over contiguous **bands** of columns -- whole rows on
a plane, blocks of `LINE_BLOCK = 64` columns on a line -- and each band writes its row of
partials, shape `(n_bands, n_slots, n_omega)`; columns run in ascending order with
frequency innermost. The band count is
`min(n_units, FAR_FIELD_BAND_BUDGET // (8 n_slots n_omega))` with a 128 MiB budget
(`backends.spatial.far_field_bands`): fixed by the shapes, **never by the thread count**,
so the order of every sum is the same on any pool. Stage two (`far_field_sum`) adds the
bands serially in ascending order, and the host finishes with `epsilon_0 c/2`, the
spectral-density factor, the energy weights of `absorbed_power` and each amplitude's
pairing `Re(A conj(G))` with its magnetic closure.

Every quantity is measured by the medium's axial flux (SEP 34): each squared modulus
is weighted by the real admittance `c Re k_z / omega` of the operator's own root, and
each amplitude is accumulated twice, as `sum alpha W E` and as its closure
`sum alpha W conj(eta) E` (the internal spectrum takes the conjugate admittance). The
admittance is formed per element, from the root the operator uses, never stored as a
column-by-frequency table. The slots (`backends.spatial.far_field_layout`) are the
present ones of `[beam, absorbed x 4, regions..., per aperture (collected, Re and Im of
the on-axis amplitude per polarisation, then of its closure), on-axis amplitude and
closure]`. The last is the `W = 1` slot:
`on_axis_spectrum` is formed there, which holds SEP 23's identity between the two to
rounding by construction, and `on_axis_intensity` reads the same amplitude and closure
(`collect_spatial(..., axis=(E, G))`), so neither forms the contraction again. The bodies are
`define_bodies`' `mask_plane_radius` (protected: compiled strict and called, so a hard
edge decides as the reference's `spatial.mask_plane_radius` does, to the bit),
`aperture_weight` (the profile codes, the raised cosine's early-out, and the exact zeros
at `omega <= 0` and beyond the paraxial disc) and `region_member` (the half-open test on
the negated stored axes); `Re k_z`, the propagating test and the evanescent clamp come
from `longitudinal_root`, the body `longitudinal_rate` calls, recomputed per element
and never stored. When every aperture shares its centre and distance the radius is
evaluated once per element (the shared-mask path): the same strict expression on the same
inputs, so its decisions are the general path's exactly. The partials and their band sum
are sized from the problem's own request at construction (`statistics_partials` in the
ledger, `backends.memory.far_field_memory` in the estimate) and reused by the periodic
channel; a spec handed to the run loop with larger tables uses buffers of its own. Stage
one is a parallel closure, compiled on the first statistics call (the entrance plane) and
never disk-cached; its serial form and stage two are cached. Measured cost: about the
`energy` statistic's per plane, 30 to 37 times below the host pass
([benchmarks.md](benchmarks.md#far-field-statistics-pass)).

**On `native`** the same pass is native: `GeneralState::aperture_statistics`
in `native/` (`far_field.hpp`, `FarFieldPass`), with its own flag mask beside
`spatial_statistics` (values from 32: the six names and `on_axis_spectrum`, the `W = 1`
slot, which on the Cartesian and line kinds replaces `spatial_statistics`' serial
`axis_spectrum` loop; the radial and planar kinds keep it). It mirrors the `numba_cpu`
pass: the slot layout of `far_field_layout` and the band edges of `far_field_bands`, with
the same units and budget; a team per band whose threads share its frequencies, each
(band, bin) owned by one thread that walks the band's columns in ascending order (on the
host a team is one thread, so frequency is innermost, as on `numba_cpu`); then the bands
added in ascending order and the constant factors applied natively, so only the finished
statistics cross to the host, `n_omega` values per row. Every sum is therefore the
`numba_cpu` pass's in the same association and the two agree within the rounding bound of
`tests/backends/test_native_far_field.py`. The radius is a protected function (strict on
the host; round-to-nearest intrinsics in a device build), so a hard edge decides as the
reference decides. Two exact shortcuts keep it within a few per cent of a step: a `tanh`
weight beyond `|t| = 22` is exactly 0 or 1 (`tanh` lies within 1.6e-19 of `±1` there, so
its correctly rounded value is `±1`), so the transcendental is skipped, and apertures that share a distance share the image's
distance from the origin (and the radius too when they share a centre). The native `Problem` gains the stored axes `spatial_k_x` and
`spatial_k_y` and the slot count of the problem's own request
(`spatial_far_field_slots`), which sizes the band partials at construction
(`far_field_partials` in the ledger, to the byte the estimate's row). `native` forms
`on_axis_intensity` in `spatial_statistics`, not from the `W = 1` slot, so its slot count
leaves that name out (`far_field_partials(problem, axis_folded=AXIS_SPECTRUM_ONLY)`). The
aperture loop and the shared radius are gated on the layout's aperture slot:
apertures declared in the spec but no aperture statistic asked for (a periodic
`beam_spectrum` sample) evaluate no weight. The call carries the
tables, `(n_apertures, 6 + n_omega)` rows of the host-formed constants and scale row and
`(n_regions, 4)` boxes, which the native `far_field_tables` row holds with the axes and the
packed output. A build declares the names it forms in `NativeBackend.far_field_statistics`;
solprop's registry admits them by it and refuses them by name on an older build, before it
is built (`registry.statistics_refusal`). The device path compiles but is not
qualified: its partials live in global memory, not in registers
([benchmarks.md](benchmarks.md#far-field-statistics-pass)).

**The plane measure on the compiled backends** (SEP 34, kernel schema 35). Both compiled
backends measure every spatial field statistic by the medium's axial flux, as
`collect_spatial` defines it, and read the admittance `c k_z / omega` from the root the
operator uses, per element, with no column-by-frequency table. A Cartesian, line or
planar run's plane measure is its own separable operator (the adapter refuses another);
a radial run packs the entrance's material and transverse squares
(`flux_k_squared`, `flux_kperp_sq` and their residuals) and, for a medium that varies
along z, uploads the material square as a `"flux"` z table of `n_omega` values, which
`spatial_statistics(flags, factor, z)` evaluates at the statistic's plane. On `native`
energy weighs each column and bin by `Re eta`; the on-axis spectrum pairs the coherent
axis amplitude with its closure; the on-axis intensity, peak intensity and fluence
synthesise the closure `G` beside `E` -- on the radial kind through the dense synthesis
of a closure time volume (`statistics_weighted_time`), on the separable kinds through a
second transverse volume (`flux_closure_transverse`, one state's bytes, the estimate's
`flux_closure_memory`) and the chunk's closure time rows -- and pack each node's signed
fluence with its magnitude `dt sum |E| |G|`, whose largest value fixes the width
refusal's normwise budget (`backends.spatial.fluence_budget`). The closure buffers are
allocated at construction when the problem's own request has a local statistic and,
outside any step, when a later request first needs one. `numba_cpu` forms the far-field
pass's closure slots in its compiled pass and leaves the node closure to
`collect_spatial`.

## 4. `native`: the compiled CPU and GPU backend, a second distribution

`solprop-native` is selected lazily as `backend="native"`. It lives in `native/`, outside
`src/solprop`, and is the repository's second distribution: one checkout, one uv workspace
and lockfile, one version from the repository's tags, and solprop's `native` extra pins it
to that version. `uv sync --extra native` builds it editable into the development
environment; `uv run poe check-native` is its gate. There is no other GPU backend: GPU
runs use the native backend's CUDA or HIP build, and the build determines whether the
backend runs on the CPU (OpenMP) or a device. How to build it is
[Building the native backend](../guide/performance/building-native.md); the native rules are in
`native/AGENTS.md`.

The backend depends on solprop's versioned seam, `SCHEMA_VERSION` in
`solprop.translate.problem`: the `KernelProblem`, the protected surface of `StepDriver` and the stage bundles, all frozen under
`SCHEMA_VERSION` by `tests/backends/test_driver.py`. Its `check_build` refuses another
schema and looks the driver members, bundle fields and contract tuples it relies on up by
name. It records the repository version it was built at without comparing it, so an
editable build stays valid across solprop commits until the schema bumps. Solprop knows only the registry
row and imports the package only when it is selected.

**Every backend refuses what it cannot execute.** A schema check is not enough on the
general modal route: its optional leaves (the coupled flow, the moving guide and the
others) are part of the schema whether or not a backend executes them, and a backend that
accepted a problem carrying one it ignores would silently run a different model. A backend
therefore calls
{py:func}`~solprop.backends.driver.refuse_general_leaves` before it runs, naming in
`admits` the leaves it executes, as `numpy_ref` and `numba_cpu` do, or makes the same
refusals itself, as `native` does in its own adapter. A third-party backend owes the same. A change that
touches `native/` is complete only when both gates (`poe check-all`) and solprop's lanes
are green with the current build installed.

**The fused route.** The backend runs the three field families, the Kerr variants, both
noise and absorber schemes, adaptive/replay/resume, FSAL and hard masking, multimode scalar
and two-effective-component transforms, exact node chunks, taper-scaled matrices,
fine/coarse projection, plasma and its accepted-state stash, weighted controllers and
device-owned statistics, and the radial, Cartesian and planar spatial kinds. Persistent
Kokkos `LayoutRight` views and Kokkos-FFT plans stay resident for a run; one nanobind call
executes all seven stages of a trial and returns fixed-order error partials, with one
additional native pass only for a frequency-weighted controller. Observation remains the
explicit fence and owning host copy. Period-one statistics instead reduce on the execution
space and return one packed scalar transfer. The modal power weight is consumed in the
statistics kernels alone.

General-modal CPU sources share real cubic monomials across output components and
use separate FFT plans for the final partial chunk, aliased onto existing storage.
Their serial Kokkos-FFT plans consume measured wisdom in the linked FFTW library.
Threaded modal transforms default to estimate planning because measured wisdom can
select slower execution plans. `SOLPROP_FFTW_PLAN=estimate` retains cheap planning
at every width; explicit `measure` and experimental `patient` enable timed planning
at any width. Estimate can still consume matching wisdom already present in the
process; the backend does not clear shared FFTW wisdom. Planning is destructive and
precedes state initialization.
Contiguous Raman transforms use the same planning policy, including long doubled
rows and partial chunks. On macOS, NEON FFTW handles serial and long Raman rows;
short parallel power-of-two rows (at most 16384 samples) retain Accelerate. Selection
uses the linked FFTW's codelet family and the plan's dimensions and thread width at
construction. A local FFTW without NEON retains the previous short-serial FFTW rule.
The full-chunk Raman provider and task counts, planning seconds, and plan heap growth remain
in execution metadata. General-modal tail views add no field buffers. These choices preserve
explicit transform scaling and the device paths. See the multimode CPU measurements
in [benchmarks.md](benchmarks.md#multimode-cpu-step-costs).

For ordinary CPU DOPRI5 modal trials with an embedded transverse rule, only the
final source projects the coarse rule. Earlier stages use fine-only products,
including the fine rows of existing stacked projection scratch. The final coarse
time buffers remain live until accepted-step scoring performs their FFTs and, for
held field-plasma streams, their spectral assembly. Private readiness flags are
invalidated by a new trial, a standalone source evaluation, reset and closure;
accepted diagnostic values and peaks have separate storage and survive rejection.
The validated seven-stage FSAL method data determine eligibility. Adams, low-storage,
general-modal, dispersive and envelope-plasma paths keep their existing scheduling.
No seam fields, persistent field buffers or arithmetic protections change.

Scalar and declared tensor Raman are native too. Persistent doubled-grid batched
r2c/c2r plans transform one or three real histories per node and a separate
r2c plan transforms one or two effective kernels. The kernel
spectrum is memoised by the exact ordered tuple of stage decay rates and, for multiple
components, their amplitude densities. Tail chunks use persistent plans for
their actual width. The arithmetic is explicitly none-normalised,
`(drive_spectrum * kernel_spectrum) * (1/(2*n_to))`, with `dt_fine` folded into the kernel
spectrum; the kernel has a plan of its own because a Kokkos-FFT plan is tied to its batch
extent. Real/analytic/envelope drives keep the shared contract's `E^2`,
`|E_a|^2/2`, and `|A|^2/2` forms, including real-field THG. The temporal window is applied
after the combined Kerr/plasma/Raman response.

**Envelope plasma** uses two persistent envelope current streams with independent
projected, embedded and spectral buffers. Each channel/chunk evaluates cycle moments and a
fresh population scan; a planned scalar status readback rejects unresolved angular
quadrature before its transforms. On a device the envelope status reduction uses an exact
integer minimum, preserving the first failing sample without changing floating-point
arithmetic. Saved and periodic envelope population statistics recompute the observed
field, independently of the field-plasma accepted-state stash. The
[kernel contract](kernel-contract.md#5-responses-the-compiled-nonlinear-term) specifies
the ordering and memory ownership.

KokkosKernels 5.2.0 supplies ordinary GEMM—not Batched GEMM—for the skinny-by-wide node
transforms. The backend keeps custom, TPL-free KokkosBlas, vendor-TPL and measured hybrid
build variants. Hybrid builds use Accelerate for eligible real transforms on macOS;
OpenBLAS builds additionally use persistent preconverted complex matrices for eligible
envelope transforms. Vendor and Kokkos/FFTW pools are recorded separately and are never
changed inside the extension.

On CPU builds, a scalar radial analytic field uses real GEMM for synthesis even
when the node budget splits the Hankel matrix into chunks. Each chunk aliases its
contiguous matrix rows and the complex samples as interleaved real/imaginary
columns; the final short chunk uses its actual row count. This changes neither
precision nor the discrete transform and adds no packing buffer. In particular,
chunking must not send this field family through the scalar custom synthesis loop
while reporting an interleaved provider. Device dispatch retains its own policy.

**On a device.** Non-plasma responses distribute time samples within a node team; plasma
uses the same team shape with a deterministic inclusive scan. Error, quadrature,
weighted-control and statistics partials use persistent fixed-width device reductions. The
geometry selected on a data-centre NVIDIA GPU is response team 256 for `n_to < 8192`,
response team 512 otherwise, and reduction block 512. Persistent packed chunk and tail
matrices let KokkosBlas dispatch exact node chunks through cuBLAS without trial-time
conversion or allocation. cuFFT plans are persistent and explicitly none-normalised. The
response column's `if constexpr` blocks carry a capture anchor, because CUDA 13's extended
lambdas otherwise refuse the capture.

The singleton constructor requires a one-member problem; fixed-schedule fibre batches go
through `select_batch` over independently loaded member workspaces (§1). The backend
refuses scalar-only declarations on vector Raman problems, spectral responses other than
the compiled envelope-plasma assembly, and z-tabulated operators or prefactors before
field-sized allocation. Supported mixture density, pre-ionisation and Raman decay paths
cross as stage scalars; z dependence alone is not a refusal.

**The general modal route** is a second execution path beside the node pipeline
(`src/general_modal.cpp`, `GeneralModalSource`), exactly as it is one in `numpy_ref`
(§2). Per node chunk: the coarse-grid synthesis of the three Cartesian components from
the complex map `S`, placed into a zeroed fine spectrum; one complex inverse transform to
the analytic signal; the rank-four contraction on the real field or on the analytic
signal, as the record's `model_id` declares, with the component count a template
parameter -- on the real field the reference's symmetric contraction (each node's
coefficients formed once per team), on the analytic signal the dense one; the six symmetric pair histories on the doubled axis through the engine's Raman
FFT provider (vDSP on macOS) and their two delayed tensors; one real forward transform of
the real polarisation; and the projection with `Q`, accumulated over chunks, times
`-i omega`. The synthesis, the instantaneous contraction and the projection are team
kernels over contiguous samples and bins, every sum in the reference's order. On a device
the projection is its own kernel — whole teams, the rows split across threads and their
partial sums added in thread order — so its summation order is fixed per build but is not
the host's, and a device answer agrees with the CPU build's to rounding, not to the bit.
The lab-frame input, the in-stage absorber, the prefactor and the interaction picture are
the engine's existing code; the coarse channel spectra, times the pad scale and the
analytic doubling, are written by the existing stage-combination kernel. Views, host-map
release and plans happen at construction; eighteen `general_*` rows join
`memory_ledger()`. The adapter derives the per-node tensor, the form, the chunk and the
delayed maps with `solprop.backends.numpy_ref.general_modal`'s public functions, so the
arrays are the reference's own, and refuses every combination the reference refuses by
name before allocating. A general modal run's statistics are solprop's `collect` on the
observed channel field with the basis metric.

**Moving maps.** The adapter admits moving maps through the shared list
(`moving_maps=True`) and packs each stage's `transform_weights`; the engine uploads every
Chebyshev coefficient of both maps once, finds each stage's section on the host when a
trial's weights are loaded, and before each stage's source combines that section's `n_deg`
terms in a named team kernel (one team per map row, the row's bins its vector lanes, the
terms summed in order from zero, `GeneralModalSource::combine`) into the two maps the
synthesis and projection read; nothing is allocated in a trial, and the ledger adds the
weights (`general_map_weights`).

**The coupled flow** (`src/coupled_flow.cpp`, `CoupledFlowAction`) executes the
constant-Schur and frozen-residual sections; CF4 stays `numpy_ref`'s and is refused by
name. The adapter uploads each section's eigenbasis
(`solprop.translate.coupled.schur_eigenbasis`: `W`, `Y`, the eigenvalues and the route per
bin, with the Schur factors of the bins routed to scaling and squaring) once at
construction, and `CoupledFlowAction::form_actions` forms every stage's and the whole
step's pair in one launch parallel over the active bins — `W exp(+-l Lambda) Y`, or the
reference's scaling, series and squaring on a routed bin, a repeated fraction copied — and
measures the whole step's `norm(U) norm(U^-1)` per bin by Jacobi rotations on the Gram
matrix; the adapter checks the squaring limit before and the certificate after, with the
reference's allowance and words. The algorithm and its gate are `numba_cpu`'s (§3.9). Per
stage the engine carries the state and the stage sum forward (`U u + h U sum a_ij k_j`, as
the reference's `stage_input`), writes the source into the flow's drive, adds
`exp(-dPhi) R_s(z_i) lab` for a frozen-residual flow (the residual combined from the
stage's nonzero `flow_weights` over the held terms, minus the step's frozen block), carries
the drive back by the inverse action and accumulates; the accepted update takes `U(h)` of
the state, the solution and the error sum, the old norm on the unmixed state. A zero stage
fraction is flagged and not applied; FSAL is off on a flow. Fifteen `flow_*` rows join
`memory_ledger()`, zero on every other route. Admission is solprop's shared list under the
name `native` plus one native limit (the delayed response's three components); the batch
backend refuses a flow member by name.

**The hybrid planar kind** runs on the same engine: columns `(p, k_y)` go to `(p, y)` rows
by a one-axis y FFT, a node chunk holds whole y rows, and the slab rule synthesises and
projects them in team kernels with each sum in increasing order; each column's operator
subtracts its channel's `kappa_sq` last. Ten fixed steps agree with `numpy_ref` to
1.2e-14 to 6.6e-14 against bounds of 2.5e-11 and more, and thread counts agree within
those bounds.

**The dispersive electronic response** (`src/dispersive.cpp`, `DispersiveResponse`;
[SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)) runs on every route
`numba_cpu` runs it, with `numba_cpu`'s structure route by route (§3.11): the fused node
path (one or two components), `SeparableDispersive` on the Cartesian free-space, line and
planar routes, and `GeneralDispersive`, held by `GeneralModalSource`, on the general modal
route (one to three Cartesian components). What is the native backend's own:

- **Layout.** The adapter lays the leaf out as `dispersive_terms_layout` does (the leaf's
  groups, each group's filtered fields and products, each output term's multiplier and
  factor, the sector's `Π₊` and factor in the multipliers), and the engine runs it inside
  the stage.
- **Weights.** Every route weighs its channels by one rule (`DispersiveWeights`): `ε₀ γ₀ ρ`
  per channel, a symmetric mixture's filter rebuilt by `mixture_filter`'s arithmetic
  (channel order, two square roots, one at a vacuum), the polynomial group's pair weights,
  all only when the stage's densities differ from the last rebuild's.
- **The fused path.** A functor forms the products in the leaf's sector at each chunk's
  nodes and multiplies them by the collar; the projection GEMMs take them to both rules'
  columns, each chunk's columns of the scaled projections packed once a stage; one batched
  forward transform and a landing kernel add the weighted spectra after the stage's own
  forward transform. Alone, the response takes over the stage's transforms exactly as on
  `numba_cpu`. A coupled wall flow beside it carries each stage's input, the source being
  the fused path's own.
- **The separable routes** hold the filtered fields and one node slot (symmetric) or two
  (polynomial) on the planar route; free space's inverse transform writes the filtered
  fields straight into the product blocks. A lone symmetric response always filters the
  chunk's spectrum in place: the in-stage absorber is a pass of its own there.
- **The general route** keeps its inverse transform's `1 / n_to`, which the response
  applies to what it reads (a power of two).
- **Order and memory.** Every kernel is a fixed-order loop inside one output element;
  views and plans are made at construction, and seven `dispersive_*` rows join
  `memory_ledger()`, zero on every other problem;
  `solprop.backends.memory.dispersive_memory` predicts the rows on each route to the byte.

**CPU-build arithmetic choices.** Each is a speed choice that moves an answer by rounding
at most:

- **GEMM without a vendor BLAS.** A `HYBRID` build with TPL BLAS off (the reference build)
  would send eligible real synthesis and projection to KokkosKernels' generic
  `KokkosBlas::gemm`, whose team kernel copies 24 x 16 and 16 x 64 blocks through
  scratch and dominated a small capillary trial. `native_order_gemm` (`gemm.hpp`)
  keeps its per-element arithmetic exactly — each 16-deep inner block an FMA chain from
  zero, the blocks added in order to a total that starts at zero, then `alpha * total`
  or `beta * C + alpha * total` — in 8 x 128 (or 4 x 128) register-blocked tiles; a
  native test compares it with `KokkosBlas::gemm` within the GEMM rounding bound over inner
  lengths below, at and past one block, partial tiles, both `beta` and row subviews, so
  the kernel's blocking and order are free. Vendor-BLAS and device builds use
  `KokkosBlas::gemm`.
- **Small transforms on one FFTW thread.** A modal or spatial plan whose real side holds
  fewer than **32 768** samples per execution (`default_threaded_fft_min_samples`;
  `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES` overrides it) is made on a one-thread
  execution instance, because below that a threaded FFTW plan only costs time: the
  threshold is a speed threshold. FFTW's planner may choose a transform's factorisation by
  the thread count (a two-column 256-point inverse took radix 4 with 64-point leaves at
  three and four threads and radix 8 with 32-point leaves at others), which moves ten
  realistic steps by at most 8 u: within the transform's rounding bound, so it constrains
  nothing.
- **One stage phase.** `sum_t dPhi_t Phi_t` is needed three times per stage (the
  combination, the finish and a flow's residual) over up to 306 terms on a varying guide;
  a vectorised kernel forms it once into `stage_phase` (a ledger row) in the same
  per-bin order, and the whole step's at the last stage.
- **The residual's zero weights.** A frozen-residual stage combines only its non-zero
  `flow_weights`, listed per stage in `set_trial`: a zero weight adds an exact zero to a
  sum that starts at `+0`, so skipping it moves nothing (tested with padded zero terms).

**Build profiles.** A plain build of `solprop-native` is the **local** build: it links a
vendor BLAS behind KokkosKernels, found at configure time — Accelerate on macOS; on Linux
MKL when oneAPI's `MKLROOT` (or `SOLPROP_NATIVE_MKL_ROOT`) names it, else OpenBLAS; cuBLAS
or rocBLAS in a device build — compiles for the build machine (`Kokkos_ARCH_NATIVE`) and
keeps checked IPO, so `HYBRID` sends the eligible real transforms to vendor DGEMM.
`SOLPROP_NATIVE_BLAS_VENDOR` or `SOLPROP_NATIVE_BLAS_LIBRARIES` overrides the order and
`SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF` keeps the native-order GEMM; with no BLAS found the
build falls back to it with a warning rather than failing. `SOLPROP_NATIVE_PROFILE` names
the other two builds. `DISTRIBUTION` is the build for wheels that run on other machines
(ADR-034): it compiles for a CPU baseline, `x86-64-v3` on x86-64, `apple-m1` on Apple
silicon and `neoverse-n1` on Linux aarch64, which the package checks against the
processor before it loads the extension (`solprop_native/_cpu.py`: `/proc/cpuinfo` on
Linux, `IsProcessorFeaturePresent` on Windows). On a processor below it,
`resolve_name("auto")` chooses `numba_cpu` with a warning, once per process; it loads
that pure-Python check by path and runs it, so it neither imports the package nor
starts its runtime, and the refusal and the fallback are one check. An explicit
`"native"` raises the refusal. It links Accelerate on macOS; on Linux an
OpenBLAS 0.3.34 that the build compiles with OpenMP and the extension's own compiler, so
that BLAS and Kokkos share one OpenMP runtime and one thread pool (selecting its kernels at
run time; `solprop_`-prefixed symbols, through KokkosKernels' `F77_BLAS_MANGLE`); in the
Windows wheel the same OpenBLAS built static on LLVM's `libomp`, the runtime of its Kokkos
and FFTW ([Releasing and the native wheels](releasing.md)), and in a Windows build
without it the `scipy-openblas32` OpenBLAS (`scipy_` prefix). Each is carried in the
wheel. At import the package fits a host OpenBLAS to the Kokkos pool, as
[CPU performance](cpu-performance.md) records (a pthreads OpenBLAS held to one thread, an
OpenMP one capped at the Kokkos thread count, unless the user sets the BLAS's own thread
variables); `build_info().blas_thread_policy` records which. `REFERENCE` (no vendor BLAS, no
machine-specific code) is the gate's tidy and sanitizer build. `build_info()` records
`build_profile`, `cpu_baseline` and `blas_selection`. A vendor BLAS sums in its own order
and a CPU target may contract multiply-adds, so the builds' answers agree to rounding, not
bitwise, and every build is held to the same parity bounds against `numpy_ref` and the same
rounding bounds between repeats and thread counts. No build carries numerical pins
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): `REFERENCE` keeps its name for
its tidy and sanitizer role only. The native adapter's contract tuples `MODAL_TABLES_FIELDS`, `REGIONAL_RESPONSE_FIELDS`
and `COUPLED_FLOW_FIELDS` name the general leaves' members its adapter reads, so a rename
is refused by `check_build`.

## 5. JAX: evaluated, not adopted

A JAX backend was measured against `numba_cpu` at the step level on the RDW and CtC
problems: its out-of-place transforms, unrequestable multi-accumulator reductions and
materialised dtype promotions lose the small RDW step by about three times while it is
slightly ahead on the Raman-carrying CtC step, and under the production pattern of many
concurrent single-thread runs `numba_cpu` has twice its throughput. So JAX stays
evaluated-only (ADR-002), `numba_cpu` is the in-tree CPU backend under every reading, and
the GPU path is the native package.

## 6. In-place transforms

Transforms are in place because the buffers are owned anyway and it helps residency; nothing
is designed around it. An out-of-place FFT reads A and writes B and an in-place one reads
and writes A — the same byte count — so the saving appears only if out-of-place forces a
copy back, which pointer swapping avoids. At solprop's sizes what matters is **cache
residency**: a field of 64 KiB–2 MB with its plan tables and the propagator factors stays in
L2 across a stage only if the working set is small.

## 7. Benchmarks

`benchmarks/` holds the standard suite (see [benchmarks.md](benchmarks.md)): bounded windows
of full-resolution problems, run in fresh processes, whose results stay on the machine that
ran them. Every performance claim in this document must point at a measurement recorded
under [benchmarks.md](benchmarks.md#reporting-a-performance-claim)'s rules or be marked as
an estimate.

## 8. Memory ledger

Every backend reports `memory_ledger()`: buffer name → bytes, including transform work areas.
A test asserts which entries are field-sized, that everything is named even when empty, and
that the total is within the documented budget for the standard cases.

**Resident Adams storage.** `native` allocates dense-output storage only when
`interpolate_planes=True`. For maximum predictor order `q`, its ordinary resident
history owns `2 (q + 2) + 4` complex field volumes, plus the norm's small scalar arrays.
Dense output adds `q + 2` slope volumes and four saved fields; a varying operator's
compensation adds another `q + 2` dense terms only with interpolation enabled.
All required buffers are allocated before stepping. This changes storage, not the
method or its maximum order. The [Cartesian memory investigation](benchmarks.md#resident-adams-storage)
distinguishes these allocations from process peak RSS, host copies and FFT workspaces.

**Admission.** Before a backend allocates, the driver estimates every run from the same
formulas (`solprop.backends.memory.estimate_memory`, equal to each backend's route ledger
rows to the byte, the Python-owned reference counted on the native backend too) and compares the
estimate with a budget: the `memory_budget=` keyword, else `SOLPROP_MEMORY_BUDGET`, else
four fifths of the machine's memory (the smallest of physical memory, a cgroup limit and a
Slurm allocation). A declared budget refuses; the undeclared default warns
(`solprop.driver.memory`). A general modal guide is admitted at construction too, from its
declared shape and before the route builds its reference, maps and chunk
(`estimate_guide_memory`, `admit_guide_memory`), so a guide over a refusing budget
allocates none of them. An in-memory output is charged as the `output` group for the
planes whose fields the run keeps ([SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)'s
`fields`), and a field-free run under the low-storage march adds one `held_plane`
volume; a `SimulationRunner` is admitted once for its backend and each run for its own
output beside it. On the general modal route memory is dominated by the dense maps
and the resident E/H reference, `16 Q C M n_omega` bytes per map and up to 14 such maps on
a constant guide on every backend (`native` holds two natively; the other twelve are the
Python-owned record).

**The estimate's groups** (`solprop.backends.memory.MEMORY_GROUPS`) are grouped as the
backends' own ledgers group their buffers, so a refusal can name the largest and what to
change: `base` (the state, the stages, the propagators and the transform buffers), on the
general modal route `general_maps`, `general_reference` (the dense E/H reference at every
frequency bin, the largest group on a guide with many channels), `general_chunk` and
`general_delayed`, `coupled_flow` for a coherent wall or a varying guide, and
`adams_history` for the Adams integrator's history rows (on the native backend the resident
engine's); the driver adds `output` and `held_plane`. The route groups equal each
backend's ledger to the byte. `base` is the reference's count of its buffers on every
backend, not the compiled backend's own, plus each compiled backend's own rows where the
two differ (the far-field partials, the dispersive response's buffers, the held
field-current source's). What is not counted is named on every estimate: FFTW's planning
heap (tens of MiB on `numba_cpu`'s delayed route), per-stage transients and the problem's
own tables. A run over its budget is refused by name before anything is written — the
estimate, the budget, the largest groups and what to change — rather than paging or being
killed hours in; nothing is ever approximated to fit. The user-facing account is
[Execution resources](../guide/performance/resources.md#memory).

**The low-storage march** ([SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md))
holds no stage, accumulator or propagator-pair row: `numpy_ref`'s ledger
(`reference_base_memory`, method-aware) names the state, `register_accumulator` and
`register_stage`, the transverse scratch on the free kinds and, tabulating, the flow
table in `propagators` (one volume of `exp(h D / 2)` on a constant operator, the step's
`P` and `Q` on a varying one, nothing when recomputing). On a Cartesian grid that is five
field-sized volumes tabulated and four recomputed, one more with the `modified` noise,
against 13 to 16 for the Runge–Kutta march and 20 to 44 with the Adams history;
`estimate_memory` equals the ledger to the byte (`tests/backends/test_low_storage.py`).
A run `_admit_history` refuses under Adams but that fits under the march is admitted
with it. `numba_cpu` and the native backend run it on register kernels
holding none of the Runge–Kutta march's buffers -- on the native backend, `GeneralWorkspace` then
allocates no `stages_`, `solution_`, `error_`, `residual_`, `recycled_`, `advanced_`,
`whole_propagator_` or `stage_inverse_` -- and both elide the stage register on the
Cartesian and line routes without an in-stage collar (`low_storage_stage_register`):
four volumes tabulated, three recomputed. Their ledgers' march rows (the state, the
registers, the flows counted by `low_storage_flows`, the transverse scratch) equal the
estimate to the byte. Recomputing on a modal or general route, `numba_cpu` forms the
step's `P`, then `Q` over it before the third pass, in one held volume
(`low_storage_flow_scratch`, in `propagators`), exponentiated per element from the terms
of `Phi` or the z table's held difference, so no flow or exponent is a transient outside
the ledger; `native` forms them per element from the phase terms and holds none. The
native state refuses what its folds cannot read: an elided stage register beside an
in-stage absorber, two flow volumes on a separable geometry, and a step reusing a
constant operator's table it has not built since the field was loaded (solprop's rules
never ask for any of them). `native` holds a field-free plane's state in its memory space
(`held_plane`), so nothing crosses to the host at such a plane.

**The far-field pass** (§3.13) holds `statistics_partials` on `numba_cpu`, its band
partials and their sum, `8 (n_bands + 1) n_slots n_omega` bytes, and
`far_field_partials` on the native backend, the partials alone, `8 n_bands n_slots n_omega` bytes;
the estimate counts each to the byte in `base`. `numpy_ref`'s reference pass allocates
per call.

**Statistics.** The modal power weight enters the statistics alone: `numpy_ref` and
`numba_cpu` share the analytic Poynting functional, and `native` consumes the weights in its
statistics kernels. The extra weighted inverse transform is confined to diagnostics; the
propagation stage and response bodies keep their operation order.

## 9. What runs where

Every backend refuses what it does not run by name, before allocating; nothing falls back.
"Qualified" means the parity and repeatability tests pass within their stated bounds on
that execution space.

| capability | `numpy_ref` | `numba_cpu` | native CPU | native CUDA / HIP |
|---|---|---|---|---|
| fibre: mode-averaged and multimode, field and envelope grids, Kerr, scalar and tensor Raman, field and envelope plasma, noise, both absorber schemes | yes | yes | yes | qualified |
| an operator or prefactor that took the z table (not `device_ready`) | yes | yes | yes: the coefficients resident, evaluated natively into term slots | refused, naming the table's size |
| radial and Cartesian free space (including a line, `n_y = 1`) | yes | yes | yes | qualified (CUDA) |
| the hybrid planar kind (`"slab"`) | yes | yes | yes | held field-plasma source qualified on HIP; other combinations not qualified |
| general modal route (modal tables, metric, regional responses) | yes | yes | yes | qualified |
| coupled flow: constant-Schur and frozen-residual sections | yes | yes | yes | qualified |
| coupled flow: `"cf4"` sections | yes | refused | refused | refused |
| the dispersive electronic response (`responses.dispersive_kerr`) | every route | every route | every route | qualified on one AMD Instinct device (HIP) and one NVIDIA Hopper device (CUDA) |
| the low-call Adams integrator (`StepperSettings.integrator`), both formulations | yes | every route; beside a coupled flow the exponential formulation alone, on constant-Schur and frozen-residual sections | as `numba_cpu`, the history resident, beside a coupled flow too when every bin takes the flow's eigenbasis and it has at most 16 channels | qualified |
| the low-storage RK4IP march (`LowStorageRungeKutta`), replay-only | every diagonal-flow route | every diagonal-flow route, register kernels | every diagonal-flow route, a resident engine whose step is one native call | flows recomputed per element; qualified under HIP, not under CUDA |
| moving solid guide (moving maps beside a frozen-residual flow) | yes | yes | yes | qualified |
| fixed-schedule lockstep batches (no coupled flow) | fibre and general guides | fibre and general guides | fibre and general guides | fibre and general guides |
| periodic statistics between planes | yes | yes | yes | yes, device-resident rows |
| Adams snapshots and a resume that continues the history (`MarchSnapshot`) | yes | yes | yes: the resident engine copies its rows (`store`, and `states`, `rates`, `applied` where kept) to the host at a kept save and back on a resume | compiles; not qualified |
| field-free observation (`observe(field=False)`), the runner, a transverse collar's rate and a set evanescent rate on the separable element | yes | yes | yes (field-free observation needs no native change; the rate is one view in `SpatialCoefficients`) | compiles; not qualified |
| materials from optmat: gas Raman leaves of many rows on gradients, rate tables, optmat walls, the silica bundles | yes | yes | yes | not qualified |
| coupling between runs (`field_plane`, `regrid`, `couple`, `Outcouple`, `transport`, `with_input`; SEP 31) | host-side: any source and receiving backend | host-side | host-side | receiving runs not qualified |

**Materials from optmat** ([SEP 3](seps/archive/sep-0003-optmat-materials.md), a run under
`MaterialSelection`) reach the backends as the arrays any other run carries: index and
wall tables, carrier Kerr coefficients, rate tables, and Raman rows with their decay
paths. No backend reads optmat. All three CPU backends are qualified on them against
`numpy_ref` (`tests/backends/test_optmat_*_parity.py`): argon and nitrogen plasma on
field and envelope grids, the rate table's hold above a model's maximum field, a
helium/argon Kerr mixture, silver and silica walls, the GNLSE with the silica bundle,
and gas Raman leaves of 4, 72 and 126 rows — dry air's composition leaf, driven by the
summed density of its labels — on gradients to vacuum, scalar and vector, field and
envelope, and in lockstep batches. Each agrees with the reference to a few times
`1e-16`, far inside its operation-count bound. Neither compiled backend assumes a term
count: `native` holds a leaf's `n_terms × 2 n_to` rows and its per-stage decay rates as
views sized from the problem. A fill whose populations follow independent density
paths takes the z table, which every CPU backend runs (on `native` the CPU build only);
a shared pressure path does not. Device builds are not qualified on these runs.

**Coupling between runs** ([SEP 31](seps/archive/sep-0031-simulation-coupling.md)) never crosses
the seam: a field plane is read from a run's stored output and its record, coupled and
transported in NumPy and SciPy on the host, and installed in the receiving run as its
initial state through `with_input`. A source and a receiver may therefore run on
different backends; the integrated compositions (`tests/coupling/test_workflows.py`)
pair `numpy_ref` and `numba_cpu` sources with native CPU receivers, and the relaunch
tests cover every pair of the three CPU backends. Device receivers are not qualified.

`numpy_ref` is the oracle: every other backend is qualified against it, and it is the one
to run when two backends disagree. `numba_cpu` is always installed. The native backend's device builds
are qualified on a data-centre NVIDIA GPU (CUDA 13) and an AMD Instinct GPU (a ROCm 7
prerelease stack): the native tests, the native pytest suite, solprop's device lane and its
slow lane on `native` pass, repeats and batch members agree within
[SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)'s rounding bounds, and the
CUDA sanitizers (memcheck, racecheck, initcheck, synccheck) are clean. Thread-count probes
are the only tests skipped on a device, which has no thread count to vary.

**Field plasma on a device.** The held field-plasma source
([SEP 27](seps/archive/sep-0027-held-plasma-current.md)) is qualified under HIP — native
execution, the native and solprop GPU lanes, the held-current ladders and bounded timings on
modal, radial, Cartesian, line and planar routes under both integration rules. Under CUDA
it is built but not executed in qualification, so the table's CUDA entries for field
plasma do not cover it. The far-field statistics' device pass is not qualified (§3.13).

## 10. What to use

Which backend and build to choose for a run is the user guide's
[Choosing a backend](../guide/performance/backends.md#which-backend-to-use).

## 11. Current limits

- `"cf4"` coupled-flow sections run only on `numpy_ref`.
- The dispersive electronic response runs on `numpy_ref`, `numba_cpu` and `native` on
  every route; on a device it is qualified on one AMD Instinct generation (HIP) and one
  NVIDIA Hopper device (CUDA); a polynomial mixture costs one group per distinct leg
  bound (§3.11).
- The held field-plasma source is qualified on a device under HIP only (§9). On the
  hybrid planar kind other response combinations are unqualified on a device build.
- `native`'s device builds refuse any problem whose operator or prefactor took the z
  table; its CPU build runs it.
- The default periodic-statistics subset costs `numba_cpu` about 2.5 % of a Runge–Kutta
  step at `period = 1`, because the plasma rows read the accepted step's stash (§3.1a);
  under the default Adams integrator the source is evaluated at the predicted state, so
  the plasma rows scan the accepted field afresh and cost more.
- HIP is qualified on one AMD Instinct generation with a prerelease ROCm stack; stable ROCm
  and other devices are unqualified, and ROCm ships no device sanitizer.
- A device answer on the general route agrees with the CPU build's to rounding (the device
  projection's summation order), and the local and distribution native builds' answers agree
  with the reference build's to rounding: which is all any comparison requires
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)).
- Batches are fixed-schedule only; adaptive and spatial batches, and batches carrying a
  coupled flow, are refused.
