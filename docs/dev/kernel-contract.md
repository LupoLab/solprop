# The kernel contract

`solprop.translate.problem.KernelProblem` is the only object that crosses from the physics and
translation layers into a backend. It is a frozen dataclass of plain NumPy arrays and scalars,
validated, versioned, and documented here field by field. A backend may assume everything
stated here and nothing else; the translation layer promises everything stated here and
nothing else. Changing this file and the dataclass together is the only way to extend the
contract, and a test checks that the fields listed here and the fields of the dataclass agree.

Axis names follow [conventions.md](conventions.md): `n_batch`, `n_modes`, `n_pol`, `n_omega`
(coarse spectral), `n_omega_fine`, `n_t`, `n_to` (fine time), `n_nodes`, `n_z`, `n_intervals`,
`n_deg` (Chebyshev degree + 1). The algorithms behind the fields — the cumulative
operator, the quadrature, the compiled responses — are in [algorithms.md](algorithms.md).

**Sign convention.** Every complex array in this contract and in `Observation` is in the
**FFT convention** of [conventions.md §2](conventions.md) (raw forward DFTs, propagation
`e^{i(ωt − βz)}`, linear operator `−iβ`). The translation layer supplies `field0` and `noise0`
already converted from the optics-convention user inputs; the output layer converts
observations back. Backends never conjugate.

**Current kernel schema: 35; saved-output schema: 24.** These versions are independent.
Internal modal amplitudes are vacuum normalised; the launch and observation boundaries
carry the true-power conversion, and every backend reads the modal power weight (§3) in its
statistics alone. Radial columns are Hankel
frequencies and Cartesian columns are transverse FFT frequencies, the latter flattened as
`ix + n_x*iy`. Every scalar transverse transform is applied independently to each state
polarisation component.

**Agreement is a rounding bound.** The contract fixes quantities, not operation order. A
backend reproduces the reference's results within a rounding bound fixed from its
operation and reduction counts, never bit for bit
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)). That includes the plasma
column's running sums against `plasma_scan` and `plasma_current`, under either
running-integral rule (§5); the Raman pass against `RamanResponse.evaluate` and
`raman_spectrum` against `RamanKernels.spectrum`; and the statistics evaluated on one
state. What the contract states exactly stays exact: integer and identity fields,
hard-mask zeros, and explicitly exact zeros and no-ops.

## 1. Identity and validation

| field | type | meaning |
|---|---|---|
| `schema_version` | `int` | incremented on every change to this contract, to the protected surface of `StepDriver` or to the stage bundles of `translate/stage.py` (both frozen in `tests/backends/test_driver.py`) |
| `grid_kind` | `"field" \| "envelope"` | selects r2c/c2r vs c2c transforms and the response formulas |
| `shape` | `(n_batch, n_modes, n_pol, n_omega)` | canonical state shape |
| `flags` | frozen dataclass of static flags and counts | `single_mode`, `scalar_pol`, `single_batch`, `has_plasma`, `has_raman`, `has_spectral_responses`, `has_noise`, `has_temporal_absorber`, `constant_linop`, `mode_averaged`, `n_channels`, `n_raman` — kernels specialise on these at compile time. Derived from the content, never passed in |
| `grids` | frozen dataclass | section 2 (`translate.masks.Grids`) |
| `linear` | frozen dataclass | section 3 (`translate.linop.LinearOperator`) |
| `transform` | frozen dataclass | section 4 (`translate.modal.Transform`) |
| `responses` | frozen dataclass | section 5 (`translate.responses.CompiledResponses`) |
| `initial` | frozen dataclass | section 6 (`translate.problem.Initial`) |
| `stepper` | frozen dataclass | section 7 (`translate.problem.StepperSettings`) |
| `tdse` | `TDSEDescriptor` or `None` | immutable atomic specification, resolved response lattice and scientific identity; section 5 |
| `current_numerator` | `(n_modes,n_pol,n_omega) float64` or `None` | independent current coefficient numerator, `-crop_scale/(2 epsilon_0 c)` on active bins; divided by the stage real effective index |
| `flux` | `FluxPlan` or `None` | a spatial run's plane measure ([SEP 34](seps/archive/sep-0034-medium-flux.md), `translate.spatial.FluxPlan`): `base`, the separable coefficients with the entrance material square, whose principal root gives the dimensionless axial admittance `c k_z/omega` of every column and bin (zero on nonpropagating, inactive and nonpositive-frequency bins); `material`, the z table of the complex material square `(omega n/c)²` for a varying radial medium, `None` for a uniform one; and `admission`, the varying path's adiabatic certificate. `at(z)` gives a plane's coefficients. Every spatial statistic weights its columns by the real admittance at its plane (energy, spectra, absorbed powers) or synthesises the closure `G` with the complex one (fluence, intensities, on-axis spectrum). Required with spatial geometry and `None` otherwise; its active mask equals `grids.active_mask`. Added in version 35 |
`KernelProblem` also carries one **property**, which is not a field and so not a row
above: `device_ready` is true when Φ (§3) and the prefactor (§5) are each a weighted sum
of a fixed set of arrays, and false when either took the z table.

**How a stage is evaluated, and why `device_ready` exists.** Three of the quantities a
stage needs are *arrays* that depend on z — the propagator's exponent ΔΦ, the response
prefactor, and the noise's unitary phase — and a handful are *scalars*: the number
density, the pre-ionisation, the mode set's taper, the Raman decay rates. The seam splits
them in two, and `translate.stage` is the split:

- **terms** (`stage.terms(problem)`), the fixed arrays, uploaded once per run. A constant
  quantity is one term; an expansion is `n_terms` of them. `StageTerms.separable`
  carries the shared spatial coefficients instead when both array term groups are
  absent. Its stage weights are the distance, position and unit prefactor weight.
- **scalars** (`stage.scalars(problem, z0, z1)`), a small frozen record of host floats per
  stage: the weights that multiply the terms, and every scalar path value the responses
  read. `stage.anchor(problem, z0)` hoists the part every stage of one step shares.

A backend evaluates `ΔΦ = Σ_j w_j · terms_j` (`stage.combine`) and knows nothing else
about the representation. It is the arithmetic of `delta_phi` and
`prefactor_at`, rearranged, so a backend that goes through the bundle reproduces them to
rounding.

**The z table has no such split** — its evaluation is a Clenshaw pass over an array of the
state's size times the degree. NumPy evaluates it directly;
Numba uses preallocated host stage arrays and the same Clenshaw order. The native CPU
build holds the coefficients resident and evaluates the table at each native call's
distinct positions into term slots, which the stage weights then select (`+-1` for an
increment, one-hot for a value), so the bundle's arithmetic is unchanged and no field of
the contract changes (SEP 29 section 6.6). A device build calls
`stage.require_device_ready(problem, name)` and refuses a table, naming its size and,
for a mixture, the labels whose independent profiles exceed the qualified tensor
expansion. A
total-degree expansion is not part of the contract: it failed the joint memory and
build-cost criterion on the short fixture.

The current version is **36** (`solprop.translate.problem.SCHEMA_VERSION`), and the
tables of this document describe it. A version bump that adds an optional field
leaves every problem that does not use it unchanged, `schema_version` itself being the
one thing that moves. Removing an admitted value needs no bump, because no consumer can
misread a problem that no longer carries it.

**Batches and device memory.** `BatchBackend` batches independently translated and loaded
member backends, which carry their own parameter tables and noise across construction;
successful batch construction adopts their lifetime. `observe_members()` returns one
ordinary host `Observation` per member, preserving independent error histories and
periodic-statistics rows. `member_metadata()` returns each member's resolved policies plus
batch settings. `Backend.available_device_memory()` returns current free device bytes,
excluding live allocations and allocator reserves, or `None` for host execution;
`StepDriver` provides the host default and a device backend overrides it. Common batch
admission uses that probe before allocating a complete group, with 20% headroom and a
conservative per-member upper bound from the complete memory ledger. Only compatible
layouts under the same explicit fixed schedule can share a batch; mixed backend names are
refused. The batch's `max_batch` is its constructed member count. Calling its
single-observation API is refused because it would discard the independent diagnostic
histories. The `StepDriver` signatures, stage bundles and singleton output schema are
those of a singleton run.

**The general-modal leaves.** `linear.coupled_flow` and `linear.modal_metric` (§3),
`transform.modal_tables` (§4) and `responses.regional_responses` (§5) are optional. Each
defaults to `None`, and a problem with all four `None` carries no field-sized or dense
buffer for them. The leaves carry `int32` region and support indices beside float64,
complex128, bool and intp. **A backend refuses by name each leaf it does not execute**,
through `solprop.backends.driver.refuse_general_leaves(..., admits=...)`, or its own
equivalent check in a compiled backend's adapter (as `native`'s does), because the version check alone cannot fire
for a backend compiled against the same constant. The combinations a backend executes
are one shared list, `solprop.backends.numpy_ref.admission.refuse_general_route`, run
under the caller's name with the section methods it executes and whether it evaluates
moving maps (`moving_maps=False` by default), so every backend refuses the same problems
in the same words. On the general route `response_order` is `(accumulate,)` alone — the
response is the regional leaf and not a fused-pass stage — and the prefactor row is the
real crop scale. What a backend reads across the seam on that route is the leaves' arrays,
`grids.crop_index`, `grids.pad_scale`, `n_to` and the prefactor row.

**Stage-bundle rows of a coupled flow.** `StageScalars.flow_section` is which section of
`linear.coupled_flow` the step's **anchor** lies in — every stage of one step carries the
same value, because the exponential the interaction picture is built on is frozen for the
whole step — and `flow_weights`, float64 of shape `(n_intervals * n_deg,)`, the weights
the flow's own `(n_intervals, n_deg)` coefficient block is combined over at the stage's
position, flattened in that order and zero outside the one section the stage is evaluated
on. A problem carrying no coupled flow reports `flow_section = -1` and
`flow_weights = None`. The rows are the seam's own evaluation form applied to the flow: a
backend evaluates `G(z₁) − C(z₁) = Σ_j w_j · terms_j` exactly as it evaluates `ΔΦ`, and
the explicit residual is that combination minus the frozen block `flow_section` names.
**Which section a stage on a section edge belongs to is normative**: it is the step's own
whenever the position lies in that section's closed interval, matched within 1e-10 of the
flow's span, and the section the stage *closes* in otherwise. A section edge is a kink —
the declared taper corner the sections were placed on — so a stage arriving at one must
read the side it came from; reading the opening side instead drops a whole run from fifth
order to **first** (measured 2.1e-3 at ten steps of a qualification taper, against
3.45e-9 once the tie is broken this way).

A `"cf4"` section stores exactly what a `"frozen_residual"` one does — the same Chebyshev
coefficients of `G` and `C` along z — and differs in how a backend integrates it and in
what its rows mean: no explicit residual (`residual_norm_bound` is zero) and a pair bound
for `inverse_condition_bound`. A CF4 step of length `l` from `z` exponentiates the whole
generator `L = G − C` at the two Gauss nodes `z + c_j l`, which are **not** stage
positions, so the stage bundle's `flow_weights` cannot supply them; `numpy_ref` builds them
from the leaf itself — `solprop.translate.coupled.flow_weights` of the node against the
step's `flow_section`, the same Chebyshev evaluation the bundle row is, from `z_edges` and
the `n_deg` axis the leaf already carries. A compiled backend would need up to twelve such
positions per step (two nodes at each of six distinct stage fractions); no bundle row
carries them, which is why CF4 runs on `numpy_ref` alone.

The Schur factors of a section's frozen block and the actions formed from them are
backend-local and never cross the seam. The compiled backends form the actions in their
own kernels, from the eigenbasis of each section's triangular factor
(`solprop.translate.coupled.schur_eigenbasis`, host-side at construction and shared) or by
the reference's scaling and squaring on the bins where that basis is ill-conditioned, so
their actions are within a derived formation bound of the reference's rather than bitwise
its (`tests/flow_parity.py`); the certificate is checked per step, on a compiled
measurement, by the reference's rule. FSAL is **absent** on a flow rather than disabled by
a flag: a coupled flow would need the analogous certified forward action at the old
endpoint before any backend could reuse a stage ([stepper.md §2](stepper.md)).

The vector wall block is the same `"constant_schur"` (or, on a varying guide,
`"frozen_residual"`) leaf with a different dense generator — one azimuthal order of a
capillary rather than one `HE_1m` family — and stores **no factored wall pair**, because
its wall term is rank two and the pair's `w (v x v)` is rank one. Its `gauge_id` names the
order it was built for (`"capillary azimuthal order nu=1"`) rather than
`"capillary HE_1m"`, which is how a resume tells the two blocks apart.

**Stage-bundle row of moving modal maps.** `StageScalars.transform_weights`, float64 of
shape `(n_intervals * n_deg,)`, holds the weights `transform.modal_tables`' `synthesis`
and `projection` are combined over at the stage's position `z1`, their
`(n_intervals, n_deg)` coefficient axes flattened in that order and zero outside the one
section `z1` lies in (`solprop.translate.modal_path.transform_weights`). A backend
evaluates `S(z1) = Σ_j w_j S_j` and `Q(z1) = Σ_j w_j Q_j` in the seam's own evaluation
form, exactly as it evaluates `ΔΦ`. The row is **separate from `flow_weights`**, so that
the transform's sectioning is its own: the moving route puts the two on the same edges,
which is natural and not required. **No one-sided rule** is needed for it, unlike
`flow_weights`: the maps are continuous across a section edge even at a kink of the
geometry, where only their slope jumps, so a position on an edge takes the section
starting there (within 1e-10 of the span) and the value differs from the other side's by
the representation error alone. A problem whose maps do not move — no modal tables, or
maps of one coefficient — reports `None`. A backend that does not evaluate moving maps
refuses such a problem by name through the shared admission list.

**The frozen literals and the contract tuples.** `tests/backends/test_driver.py` freezes
`DRIVER_SURFACE`, `LINEAR_OPERATOR_FIELDS` (`dataclasses.fields(LinearOperator)`),
`STAGE_TERMS_FIELDS`, `STAGE_SCALARS_FIELDS` and `SEPARABLE_OPERATOR_FIELDS` under
`SCHEMA_VERSION`; a bump rewrites the literal it changes. `solprop-native` exports its own
tuples of the names it relies on — `LINEAR_OPERATOR_FIELDS`, `TRANSFORM_FIELDS`,
`RESPONSE_FIELDS`, `STAGE_SCALARS_FIELDS`, and `MODAL_TABLES_FIELDS` (`electric`,
`omega_fine`, `projection`, `region_index`, `source_frequency_factor`, `synthesis`),
`REGIONAL_RESPONSE_FIELDS` (`n_terms`) and `COUPLED_FLOW_FIELDS` (`connection`,
`generator`, `inverse_condition_bound`, `n_deg`, `n_modes`, `section_methods`) for the
members of `ModalTransformTables`, `RegionalResponseTables` and `CoupledFlowTables` its
adapter reads itself — and its `check_build()` refuses a solprop without one (a dataclass
field, or a property for the derived counts), so a renamed leaf or member fails there with
a message naming it rather than as a missing attribute inside the adapter. The two sets are
separate objects that this document keeps in step. Members read only inside solprop's own
derivations the adapter calls (`node_kerr_tensor`, `node_delayed_maps`, `metric_at`,
`frozen_generator`, `schur_blocks`) are solprop's to keep consistent and are not listed;
for that reason `ModalMetricTables` has no tuple, the adapter handing the leaf to
`metric_at` whole.

The dataclass tree is exactly `KernelProblem` and the six array groups `grids`,
`linear`, `transform`, `responses`, `initial` and `stepper`; every other value a field
holds — a `Flags`, a `StatsSpec`, a `Tableau`, a `PIController`, a `RateTable`, a
`ZProfile` table — is a leaf whose own fields are not rows of this document. Fields whose
name begins with an underscore are implementation detail and are outside the contract
too. The test that compares this document with the dataclass walks the tree by that rule,
in both directions; a second test checks the flags and counts named in the `flags` row of this section
against the fields of `Flags`.

**Source reservations in the shared Adams march.** The protected hooks
`StepDriver._reserve_source_calls(count)` and `StepDriver._release_source_calls()`
bound the source evaluations of a trial. Before starting a trial, the shared march
reserves its required source evaluations through these hooks; on every exit it
releases unused capacity. The ordinary backend implementation needs no atomic
reservation. NumPy's TDSE implementation translates the reservation into the
required persistent-plan chunks, while provider attempts remain charged before
invocation, including failures. Logical source-call and atomic-attempt ledgers
remain separate. The base driver owns `_adams`, initially `None`, and exports and
restores its logical ledger on every Adams backend. Loading a checkpoint resets
numerical history but preserves charged calls; startup is charged again. Atomic
ledger restoration remains exclusive to the bound NumPy TDSE path.

**The lab-frame source of the exponential formulation.**
`StepDriver._adams_lab_source(lab, position, *, purpose)` is the complete source
assembly at a lab-frame field and position -- the `modified` noise draw advanced to
that position's absolute phase, the in-stage absorbers and, on the reference backend,
the TDSE adapter with its labels -- returned in the lab frame, with no propagator
applied. The exponential Adams formulation
([SEP 16](seps/archive/sep-0016-unified-multistep.md)) stores these untransported samples and
integrates their polynomial against the linear flow; the base driver's hook refuses,
naming the backend, so a backend that has not implemented it refuses the formulation
rather than stepping it with another method. `Multistep.formulation` is
`"interaction"` or `"exponential"`.

**The Adams engine.** With `StepDriver._adams_engine(method, compensation)`,
`AdamsMarch` keeps the controller -- orders, slots, rejection,
restarts, the call budget and the ledger -- and hands every state-sized operation of
its fused and exponential paths to an engine
(`solprop.backends.multistep.AdamsEngine`): the startup sample, a trial's predictor,
source, corrector and scored defect norms, the adoption and transport of an accepted
step, holding and restoring the accepted state around a dense observation, and the
observation itself. Only scalars, the coefficient tables and the norms cross the
interface. The default returns `None`, and the march uses `HostHistory`, the host
engine, over the other Adams hooks; `native` returns one resident in its native state.
The general path (PECE and the comparison variants) keeps host arrays.

`StepDriver._adams_source(lab, anchor, position, *, purpose)` takes a
`TDSEPurpose` label so startup, predictor, corrected-field and diagnostic
observations retain their actual provenance. This is a host seam operation:
the shared march neither accesses NumPy's private binding nor adds atomic
physics to a compiled backend. Numba and native continue to refuse TDSE by name.
A corrected-field diagnostic still needs an identical cached response or a
separately charged solve; a predictor label never implies accepted-field data.
If the corrected-field solve fails after candidate adoption, `_adams_restore`
restores the previous accepted field and position before a checkpoint is written.

**A `ZProfile` table** is how anything that varies along z crosses the seam: a
scalar-valued `translate.chebyshev.PiecewiseChebyshev` over the same z range as `Phi`,
built by `translate.linop.tabulate`. A backend evaluates it with one Clenshaw pass on the
host and never sees a Python `ZProfile`. Where a path is exactly zero at an interval edge
(a declared vacuum), `translate.linop.profile_table` makes its table exactly zero there
too, within the table's tolerance, so the non-negativity check on density tables never
meets the fit's rounding.

`validate()` checks: all arrays finite, C-contiguous, read-only, and of the stated dtype
(float64, complex128, bool and intp everywhere, and `int32` at exactly three declared
index leaves — `transform.modal_tables.region_index`,
`responses.regional_responses.response_region` and
`responses.regional_responses.kernel_valid_samples`, named in
`translate.problem._INT32_LEAVES`; the dtype rule applies to every leaf the tree walk
reaches, so `int32` is admitted by name and not globally)
and shape; masks Boolean; breakpoints strictly increasing and covering `[0, z_max]`; hard
kinks are breakpoints; every `ZProfile` table covers the same z range as `Phi`; every
general-modal leaf tabulated along z covers `[0, z_max]` and carries `grids.omega`
itself; a replayed
schedule lands on every save plane and hard kink; `schema_version` equals the backend's
expected version. A failed check raises `TranslationError` naming the field. Quadrature
weights and mode matrices are checked in `modes.quadrature`; physical modal weights
are also carried as `transform.node_weights` for species inventories.

## 2. Grids and transforms

| field | shape / dtype | meaning |
|---|---|---|
| `omega` | `(n_omega,) float64` | coarse angular frequencies (field: non-negative half spectrum; envelope: FFT order about `omega0`) |
| `omega0`, `dt`, `dt_fine` | float | reference frequency, coarse and fine time steps |
| `n_t`, `n_to`, `ffac` | int | coarse and fine time lengths, oversampling factor |
| `crop_index` | `(n_omega,) intp` or slice descriptor | where coarse bins sit in the fine spectrum (field: leading bins; envelope: leading and trailing halves) |
| `pad_scale`, `crop_scale` | float | the scaling factors of the zero-pad/crop operations |
| `inband_mask` | `(n_omega,) bool` | physical frequencies inside the wavelength limits, the requested band |
| `active_mask` | `(n_omega,) bool` | Luna's `sidx`, `ω_min/2 < ω < 1.1 ω_max`: the hard band mask (0/1 per accepted step); `D` and the prefactor are zero outside it |
| `omega_window` | `(n_omega,) float64` | the taper applied to the nonlinear polarisation in the spectral collar: the grid's collar profile (1 in band) when `assemble(..., polarisation_taper=True)`, all ones by default. The rate absorber on the field is inside `Phi` either way; the taper is Luna's `ωwin`, kept as an option because it sets a floor of the size of the field's content in the collar |
| `t_window_fine` | `(n_to,) float64` | temporal window applied inside the response pass |
| `t_window_fine_derivative` | `(n_to,) float64` | analytic derivative of the temporal collar, s⁻¹; the held field-plasma source uses this for its second current stream |
| `carrier_2w0` | `(n_to,) complex128` or `None` | `exp(2 i ω₀ t)` on the fine time axis, present exactly when a third-harmonic Kerr term is compiled on an envelope grid: the constant term with `thg`, or the dispersive term's full cubic (§5), whose one-into-three group takes its conjugate. That term sits an offset `2ω₀` from the carrier the envelope is written against, so the response pass needs the factor; built from `nonlinear.responses.thg_carrier` and the grid's own `t_fine`, because a backend deriving it from `omega0`, `n_to` and `dt_fine` would be carrying a piece of physics |
| `norm_mask` | `(n_omega,) bool` | the bins the error norm scores: `active_mask` by default, `inband_mask` when the controller's `region` is `"band"`. Every backend's norm, the Adams engines' state magnitude and a frozen-residual flow's residual bins read it |

## 3. Linear operator

The scalar parameters an expansion runs over are named by the mode set's
`parameter_paths`: a capillary filled with one gas declares `core_radius` and `density`;
a fill of several species at one total pressure declares `core_radius` and `pressure`,
so the two-parameter expansion still applies and a one-component fill stays exactly the
single-gas case; independent per-species density profiles declare one
`density:<label>` path per component, which exceeds the two-parameter tensor expansion
and falls to the z table below. The names are the vocabulary of
`solprop.modes.protocol.ParameterisedMode` and of the reuse check in
`Simulation.with_mixture`; a backend never sees them, only the representation flag.

| field | shape / dtype | meaning |
|---|---|---|
| `phi_representation` | `"constant" \| "expansion" \| "table" \| "separable"` | the representation consumed by `delta_phi` and `unitary_phase`; every kind except the z table has a device form |
| `phi_breakpoints` | `(n_intervals + 1,) float64` | interval edges in z — of the table itself, or of the expansion's scalar tables |
| `phi_degrees` | `(n_intervals,) intp` | degree used per interval (diagnostic) |
| `phi_coefficients` | `(n_intervals, n_deg, n_batch_or_1, n_modes, n_pol, n_omega) complex128` or `None` | Chebyshev coefficients of Φ(ω, z) = ∫D dz − ½ ln[w(z)/w(0)] along z (the batch axis is 1 when members share a fibre); present for `"table"` and for `"constant"`, whose degree-1 intervals are exact — one, or one per stretch between the hard kinks a response path declares — and absent for `"expansion"` |
| `phi_expansion` | frozen dataclass or `None` | Φ as an expansion in the mode set's scalar paths (`translate.expansion.ParameterExpansion`; a leaf, see below); present exactly when `phi_representation` is `"expansion"` |
| `hard_kinks` | `(n_kinks,) float64` | z positions where D, or a scalar path a response consumes (a plasma channel's preionisation, an explicit density), jumps or kinks; steps land on them |
| `phi_constant` | `(n_batch_or_1, n_modes, n_pol, n_omega) complex128` or `None` | `D` itself, present exactly when `phi_representation` is `"constant"`, so that `delta_phi` returns `D (z₂ − z₁)` to the last bit rather than the difference of two Clenshaw sums. As a bundle it is the one-term case: one array, and the distance as its weight |
| `power_weight_representation` | `"constant" \| "expansion" \| "table"` or `None` | modal power weight representation, present exactly for modal transforms and absent for spatial geometry |
| `power_weight` | `PiecewiseChebyshev` or `None` | positive dimensionless `Re n_eff`, or the prefactor's index override, per `(n_modes, n_pol, n_omega)`; table and constant representations only; unit weight outside the active band |
| `power_weight_expansion` | `ParameterExpansion` or `None` | the weight with plain basis-product weights; expansion representation only |
| `power_weight_constant` | `(n_modes, n_pol, n_omega) float64` or `None` | exact constant weight, with a matching degree-zero table; constant representation only. It carries no batch axis, unlike `phi_constant`, because the members of one problem share the fibre and therefore share `Re n_eff` |
| `beta0`, `beta1` | `ZProfile` tables | frame quantities along z (diagnostic; already inside Φ) |
| `separable` | `SeparableOperator` or `None` | shared constant spatial coefficients, present exactly for `"separable"`; described below and shared by identity with `responses.separable` |
| `coupled_flow` | `CoupledFlowTables` or `None` | the generator of a guide whose channels couple, dense or factored, with its residual, conditioning and error certificate and one method per section; a leaf in `translate.coupled`, described in [the coupled-flow leaf](#the-coupled-flow-leaf) below |
| `modal_metric` | `ModalMetricTables` or `None` | the explicit Hermitian flux metric W, a documented leaf here rather than a record nested inside the flow, shared by identity with the controller's norm, the linear flow and the output layer so that no second W is allocated. `kind` is `"identity"`, `"diagonal"` or `"dense"`, a dense W being Hermitian at every interval, degree and frequency. `"diagonal"` is admitted by the record and refused by **every** backend, because nothing in the package builds one. Entries are **dimensionless**: the diagonal weights float64 and positive, a dense W complex128, the error and eigenvalue rows float64; `omega` is `grids.omega` itself and `z_edges` covers the run. `None` for the orthogonal routes, whose weight is `power_weight` above. Every backend executes it as part of the general modal route, in two kinds: `"identity"`, while its `error` row is inside `translate.modal_tables.DEFAULT_GRAM_ATOL` — an identity metric that is not the identity is a silent error of `\|1/W₀₀ − 1\|` — and `"dense"`, the metric of a declared unitary change of basis of such a reference, whose `gauge_id` must equal the tables' and whose stored block the backend measures itself against `translate.general.MAX_DENSE_METRIC_DEPARTURE` rather than trusting the declared row. Both gates are the same statement: the backends propagate through the **diagonal** cumulative flow and measure the controller's norm on the coordinate amplitudes, neither of which is the basis's own where that basis is not orthonormal in true power. A `"diagonal"` metric is refused by name, and so is a nonunitary transform, at three layers: the declaration's own unitarity budget, the per-knot invariance of the spectral norm, and this one |

The separable representation carries no `phi_coefficients`, `phi_constant` or
`phi_expansion`. It has one interval, degree 1, its declared `z_span` as the two
breakpoints and no hard kinks. Free propagation requires this representation and
refuses varying media and `propagator_mode="tabulate"` explicitly.

`SeparableOperator` is a frozen leaf in `translate.spatial`. Its frequency vectors
are `k_squared` and `k_squared_low` (complex128, material wave-number square
and its residual in m^-2), `frame`
(float64, frame subtraction in m^-1), `spectral_rate` (float64, nonnegative power
absorption in m^-1), `prefactor_numerator` and `prefactor_clamped` (complex128,
purely imaginary, with temporal crop scaling), and `active` (bool). Each has
`n_omega` entries. `kperp_sq` is a nonnegative float64 vector of `n_modes`
transverse wave-number squares in m^-2; `kperp_sq_low` has the same shape and
dtype and stores their signed residuals. Both low parts are rounded from exact
rational squares of the unsquared binary inputs. The scalars are `evanescent_rate`
(positive finite field attenuation cap in m^-1), `n_pol` (1 or 2), and `z_span`
(two finite increasing positions in m). There are no arrays proportional to
`n_modes*n_omega` in this leaf. `kappa_sq` is `None` except on the
hybrid planar kind (§4): complex128 `(n_channels, n_omega)` in m^-2, each slab
channel's transverse square `κ_p(ω)² = k²ε_co − β_p(ω)²` from the mode's own complex
propagation constant (its wall term under its loss policy), zero at inactive
frequencies; column `c` belongs to channel `c % n_channels`, and `kperp_sq` is then
`k_y²` per column with the channel fastest. `kperp_rate` is float64,
`n_modes` entries, finite and nonnegative, a transverse collar's power rate per column
in m^-1 (`solprop.boundaries.TransverseCollar`); `None` is stored
as zeros, and a nonzero one is refused beside `kappa_sq`. `evanescent_rate` is
`solprop.spatial.EVANESCENT_RATE` unless the run sets `SimulationSettings.evanescent_rate`.

For each column and frequency form
`q=(k_squared-kperp_sq)+(k_squared_low-kperp_sq_low)` in that order, then, when
`kappa_sq` is present, `q = q - kappa_sq[c % n_channels]`, and its principal
complex square root `s`. Where `Re(q)>0`,
`D=-Im(s)-i*(Re(s)-frame)` and the prefactor is
`prefactor_numerator/Re(s)`. Else the attenuation is
`min(abs(Im(s)), evanescent_rate)` and the prefactor is `prefactor_clamped`.
Subtract `kperp_rate[c]/2` from the attenuation term, `D = -(a + kperp_rate/2) - i*(...)`,
formed as `-a - kperp_rate/2` so that a zero rate leaves `D` exactly as it was. Multiply both
physical terms by `active` (the transverse rate therefore acts on the active band
alone), then subtract `spectral_rate/2` from `D`. Propagating material gain retains its sign. Evanescent components decay.
The exact cumulative increment is `D*(z2-z1)`; modified noise takes only its
imaginary part. Compiled backends evaluate these formulas per element.

`KernelProblem.validate` holds a step the run cannot shrink -- a replayed schedule's
longest step, or the floor `dz_min` -- to the propagator's exponent budget
([SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md) §4.5): with `λ₋` and `λ₊` bounds on `max(-Re D)` and `max(Re D)` from
`translate.spatial.rate_bounds` (a sorted search over the columns, `O(n_omega log
n_modes)`), a step `h` with `h max(λ₋, σλ₊) > ln(DBL_MAX)` is refused naming the
binding term and the admissible step and rate, and one past nine tenths of it warns
(`ExponentBudgetWarning`); `σ` is 1 for the Runge-Kutta and low-storage marches and
`q + 1` for an Adams method of largest order `q`. Adaptive runs are never refused.
The radial kind carries a dense operator rather than a separable one, and every backend
forms the same inverse factor `exp(-ΔΦ)` from it, so it is held to the same budget with
bounds read off the operator (`translate.radial.radial_rate_bounds`): a
constant table's exact extremes of `Re D`, and, for a varying medium, a sufficient bound
on each interval of the rate's Chebyshev series (`|T_j| ≤ 1`) for a z table or for an
expansion's scalar weights against its terms. The refusal names the radial operator's
attenuation and the admissible step. The modal kinds are not checked.

The imaginary part of Φ is also the unitary phase for the noise field. Modal Φ
includes the real endpoint term `-1/2 [ln w(z) - ln w(0)]`. It vanishes identically
for a constant fill and for the GNLSE unit-index override. Statistics evaluate
weights only through `translate.linop.power_weight_at(problem, z)`. Radial Φ in a varying
gas carries the spatial counterpart, `-1/2 ln[Y_z(ω,k_r,z)/Y_z(ω,k_r,0)]` with
`Y_z ∝ Re k_z/ω`, evaluated from endpoint admittances in both the table and the
parameter-expansion representations ([SEP 34](seps/archive/sep-0034-medium-flux.md) §4.4); it
is exactly zero in a uniform medium, and a radial statistic reads the admittance at its
plane through `flux.at(z)`.

An expansion may append an endpoint block with `endpoint_degrees` on the same
parameter box. Its terms follow the integral block; its scalar weights are plain
basis products minus their entrance values. `n_terms` is then the sum of both
blocks' tensor sizes. Reuse reweights each block with its original weight kind.

**`phi_expansion` is a leaf**, like a `PlasmaChannel`, so its fields are not rows of
this document; they are `parameters` (the path names, in the order of the term
multi-index), `ranges` (`(n_params, 2) float64`, the values each path reaches over the
tabulated range), `degrees` (`(n_params,) intp`, the Chebyshev degree per parameter),
`terms` (`(n_terms, n_batch_or_1, n_modes, n_pol, n_omega) complex128`, with
`n_terms = Π(degrees + 1) + Π(endpoint_degrees + 1)` when the endpoint block
is present, otherwise just the first product, and each multi-index flattened in C
order) and `weights`, the concatenated integral and endpoint tables along z as a `(n_terms,)`-valued `PiecewiseChebyshev`. Its
breakpoints, degrees and hard kinks **are** the three rows above, which `validate`
checks, so the local step cap and the schedule read one field whichever representation a
problem got.

**Why there are two representations.** A mode whose whole z dependence runs through
declared scalar paths — a Marcatili mode's core radius `a(z)` and number density `ρ(z)` —
makes `D` a function of those parameters, `D(ω, z) = D(ω; ρ(z), a(z))`, and the two
dependences are nothing alike. `D` is analytic in the parameters; `ρ(z)` is what carries
the square-root cusp at a near-vacuum entrance, the stage boundaries and the taper
corners. So `D` is expanded once over the range each path spans and Φ is that expansion
against scalar integrals `I_j(z) = ∫₀ᶻ Π T(p̂(z')) dz'`, which cost 8 bytes per term per
coefficient instead of 16 bytes per state element. The dispersion contribution is
Σ_j d_j(ω) I_j(z), exact to the truncation in the parameters. The appended log-weight
block adds its plain basis-product endpoint differences to form modal Φ. On a gradient
capillary the expansion is a few terms and a fraction of a megabyte where the z table is
tens of megabytes, and it agrees with the z table well inside the tolerance both are
built to ([accuracy.md §1](accuracy.md#1-the-integrators-and-the-tables)).

The z table stays as the general representation, and the translation layer chooses it
from the mode set's traits: for a mode that declares no parameters, for a set that
declares different ones or disagrees about their values, for a mode whose declared
parameters are **not the whole** of its z dependence, and for a quantity that kinks in
a parameter rather than being analytic in it — a loss clamp crossing its limit inside the
range, the evanescent index floor binding partway along it — where the degree search
reaches its ceiling. Which representation was built, and why, is in the debug log.

The third of those is the optional `parameters_own_z` trait, read by name and
defaulting to true, which `translate.linop.parameters_own_z` collects and the two
expansion-reuse gates read as well. The case is a
`solprop.nonlinear.gnlse.SpecifiedMode` whose dispersion follows a declared path along
z beside its effective area: the area is still its kinks, and is still what
`declared_paths` resolves first, but freezing the area at a value would leave the
dispersion moving, so there is no parameter space to expand in and the operator is
tabulated along z. The contract needs nothing for it — no field, no representation, no change to
what a backend sees beyond the flag it already reads. Every CPU backend runs it, the
native one evaluating the table in its own kernels; a device build refuses it through
`translate.stage.require_device_ready`, as it refuses any z table. The same fibre with a
**constant** dispersion and a varying area keeps the one-parameter expansion, and the
device routes with it.

The tolerance is not the number asked for but the larger of it and a hundred times
the **measured** round-off floor of `D`: the frame subtracts `β₀ + β₁(ω − ω₀)` from a `β`
that reaches 8·10⁷ m⁻¹ at the top of a wide field grid's active band, so `D` is only known
to about 1e-11 relative and no interpolant can resolve it below its own noise. The floor
is a property of one evaluation of `D` and serves both representations. See
[`docs/guide/numerics/linear-operator.md`](../guide/numerics/linear-operator.md), "The table".

### The coupled-flow leaf

`linear.coupled_flow` (`CoupledFlowTables` or `None`) is the generator of a guide whose
channels couple, dense or factored, with its residual, conditioning and error certificate
and one method per section. The generator and every term of it are **complex128 in 1/m**,
the scalar weights dimensionless complex128, the three certificate rows float64; `omega`
is `grids.omega` itself and `z_edges` covers the run. A leaf in `translate.coupled`, so
its own fields are not rows here.

It is `None` except on three routes. The **coherent wall** of a capillary, which
`solprop.translate.coupled.constant_schur_flow` (constant geometry) or
`frozen_residual_flow` (a guide that varies along z) builds from a condensed Robin
boundary operator when `prop_capillary` is given a `wall=`. A **general modal guide that
declares a unitary basis of a split block**, where `U^H diag(beta) U` is not diagonal and
`solprop.translate.general.coupled_block` reads the transformed generator relative to one
reference channel for `constant_schur_flow` to certify. A **moving solid guide**, whose
route carries every channel's generator less the reference channel's **entrance** diagonal
and the measured connection `Γ = W⁻¹S` (`solprop.modes.connection.band_connection`), both
expanded on the maps' sections, for `frozen_residual_flow` (`certificate_id`
`"general-moving-frozen-residual"`); `Φ` beside it is the reference channel's entrance
dispersion, a one-term expansion whose declared path puts the section edges into
`hard_kinks`. `connection` is `Γ` as measured and **not** its conjugate — the provider's
fields are internal-convention and the synthesis stores them unconjugated, so the kernel's
amplitude is their coefficient — and needs no power conversion, the Poynting amplitude
being the true-power one on this route (`W^(-1/2) C W^(1/2)` is the identity there, and
`Re Γ_nn = W'_nn/2` is each channel's own normalisation). The producers share the leaf,
the executor and nothing else: the split block carries no wall pair, no connection and no
validity ratio, and its `gauge_id` is the tables' composed one rather than `"capillary
HE_1m"`. The leaf itself is provenance-blind and neither the stage bundle nor the backend
asks which built it.

**The `n_deg` axis is Chebyshev coefficients along z within each section**, in the basis
`numpy.polynomial.chebyshev.chebval` evaluates on `x = (2z − lo − hi)/(hi − lo)`, and
`connection` shares those axes exactly; the **frozen** block of a section is that
expansion at its midpoint (`solprop.translate.coupled.frozen_generator`), read out of the
stored coefficients rather than carried beside them, so the certificate and the executor's
Schur factors are the same matrix to the last bit. `z_edges` are the sections, placed on
the operator table's own breakpoints — which land on every declared kink of the radius and
the density — subdivided `CoherentWall.sections` times. In the rank-one wall pair the
**vectors are dimensionless** and the **weight carries the whole 1/m**, so that `w (v x
v)` is the generator's own 1/m, and the constructor checks the pair reconstructs the dense
off-diagonal.

The three certificate rows are per section: `generator_error` is the frozen block's Schur
diagnostic **plus** the relative Chebyshev tail of the z expansion, `residual_norm_bound`
is the enclosure `Σ_k ‖r_k‖₂` of the explicit residual over the **whole** section (`|T_k|
≤ 1`, so it bounds the extrema no sample sees), and `inverse_condition_bound` is
`exp(dz_max (μ_max − μ_min))` of the frozen block. On a **`"cf4"`** section the same three
rows mean: the Chebyshev tail alone, **zero** (the whole generator is in the exponential),
and the **pair bound** `exp((2/√3) dz_max (m_hi − m_lo))` on the product of the four
factor norms of any forward-and-inverse CF4 pair a step anchored there can form, `[m_lo,
m_hi]` enclosing the Hermitian-part spectrum of `G − C` over the step's reach by Weyl's
inequality on the stored coefficients; its growth bound `exp((2√3 − 3)/6 · dz_max (m_hi −
m_lo))` is that row to the power `(2 − √3)/4` and needs no row of its own
(`solprop.translate.coupled.cf4_growth_bound`). `0.9/(K · cond)` from the last two is the
advisory step `solprop.translate.problem.validate_coupled_flow` warns `dz_init`, `dz_max`
and a fixed schedule against; no row carries it.

**What executes.** Two families of combinations, either of which may carry the vector
block (which differs only in the generator it stores and in carrying no factored pair; on
a varying guide its `connection` is block diagonal over the order's two radial families,
each family's own and exactly zero between them, stored like the scalar one in the
true-power amplitude): `section_methods == ("constant_schur",)`, the dense generator at
one interval of degree zero, no connection and an identically zero `residual_norm_bound`,
beside a `Phi` that is the exact constant representation; and `section_methods` that is
`"frozen_residual"` or `"cf4"` on **every** section, the dense generator at any degree
along z, an optional connection and, on a frozen-residual section, a residual the
interaction-picture source carries, beside a `Phi` of any device-ready representation. A
`"cf4"` section is integrated by the commutator-free fourth-order exponential `exp(l(a A₁
+ b A₂)) exp(l(b A₁ + a A₂))` of the whole generator at the Gauss nodes (`a, b = (3 ∓
2√3)/12`, the early-weighted factor acting first; `solprop.translate.coupled.CF4_NODES`,
`CF4_WEIGHTS`) with no residual, its inverse the product of the negated factors in reverse
order. A step is integrated by the method of the section its anchor lies in, so a flow
whose declared fallback mixed the two is the same equation throughout. For a CF4 section
the backend measures the four factor norms of every stage's pair and refuses a product
above the declared pair bound; the flow's own `O(l⁵)` defect is invisible to the embedded
pair, so a run through CF4 sections is at most **fourth** order.

Both need no noise draw and a scalar state (`n_pol = 1`, the action contracting the
channel axis alone; the synthesis may resolve two components, which the flow never reads),
and a `Phi` that is a **multiple of the identity** in the channel axis -- the flow action
and the cumulative diagonal are applied as a product, which is the same equation only when
they commute. Because `Phi` is then one reference channel's, its normalisation term `−½
ln[w(z)/w(0)]` is that channel's for all of them, and the capillary wall's varying
generator carries the rest of each channel's own on its diagonal, `−½ d ln(w_n/w_1)/dz`
(real, zero on the reference channel and on a constant guide), so `exp(ΔΦ)` times the flow
renormalises every channel by its own power weight; the term is part of the stored
generator and so of the frozen block, the residual and the CF4 exponent alike. The backend
measures `norm(U) norm(U^-1)` of the whole step itself and refuses above the declared
`inverse_condition_bound` rather than trusting it.

**Which section a stage that lands on a section edge is evaluated on is normative**: it is
the **step's own** section wherever the position lies in that section's closed interval,
matched within 1e-10 of the flow's span, so that a schedule arriving a few ulp short of a
declared kink and one arriving a few ulp past name the same section. A position genuinely
outside the step's section — which only a step spanning an edge produces — takes the
section **containing** it, resolved on the side that **closes** there when the position is
itself on an edge: a stage that ran past its step's section arrived from below, and the
residual it is measured against is the one-sided limit on the side it came from. An edge
is a kink, so reading the opening side instead drops a whole run from fifth order to
**first** (§1).

A `linear.modal_metric` beside it is refused on the **mode-set** route, whose flow is a
condensed boundary block with no bound reciprocal basis and therefore no metric this
package has certified, and admitted on the **general modal** route, where
`transform.modal_tables` is present and the metric is the guide's own sampled basis Gram,
which a declared unitary rotation leaves where it was. A flow that mixes a
`"constant_schur"` section with any other, a factored generator, a `"diagonal"` section, a
varying degree or a connection beside a constant section, and a `Phi` that took the z
table beside a varying one, are refused by name. `numpy_ref` executes every admitted
combination; `numba_cpu` and the native backend execute the constant-Schur and frozen-residual
sections and refuse a `"cf4"` section by name

## 4. Multimode transform

| field | shape / dtype | meaning |
|---|---|---|
| `synthesis` | `(n_nodes·n_pol, n_modes) float64 or complex128` | mode fields at nodes (element type of the time-domain field) |
| `projection` | `(n_modes, n_nodes·n_pol)` | weights folded in, and the basis normalisation `ε₀c/2` with them, so that `projection @ synthesis = pad_scale · I` on an orthonormal set (`modes.quadrature.BASIS_NORMALISATION`, folded in by `mode_matrix`) |
| `projection_coarse` | same | the embedded coarse-rule projection for the error estimate |
| `taper_scale` | `ZProfile` table or `None` | the scalar rescaling of synthesis/projection along z for scale-invariant modes |
| `node_chunk` | int | nodes per chunk for the response pass |
| `aeff` | `ZProfile` table | effective area for mode-averaged normalisation (`kind == "mode_averaged"`), a diagnostic otherwise |
| `geometry` | `SpatialGeometry` or `None` | the spatial leaf below; absent for modal transforms |
| `node_weights` | `float64 (n_nodes,)` or `None` | physical transverse weights in m² at z=0 for resolved modal populations; divide by taper scale squared at later planes; mode-averaged runs use `aeff` and spatial runs use geometry weights |
| `kind` | `"mode_averaged"`, `"modal"`, `"radial"`, `"free"` or `"slab"` | what the transform represents, checked on construction against the geometry, the tables, the node count and the node weights; `mode_averaged` is `kind == "mode_averaged"`, so a one-node quadrature is `"modal"` and gets no area conversion |
| `modal_tables` | `ModalTransformTables` or `None` | the full E/H reference of a general modal guide — separate synthesis S and projection Q, both on `omega`, the dense complex E and H per √W, the reciprocal partners and their Gram, the coefficient dual, the node rule with its Cartesian coordinates, area weights and per-node `int32` region index, the `−iω` source factor, the active-support mask and the declared profile error, with optional `factors`. A leaf in `translate.modal_tables`. S and Q are Chebyshev coefficients along z, `(n_intervals, n_deg, ...)`. A snapshot guide's are one coefficient over one interval, exact for a map that does not move; a **moving** solid guide's are one expansion per section of its path (`solprop.translate.modal_path.path_transform_tables`), sampled at the section's Chebyshev-Lobatto points on the provider's own rule at each position — a rule whose panel edges move with the material interfaces, so node `q` is one point `x_q(z)` moving smoothly with the geometry and keeps one material region, which is why `region_index` stays one row per node — and combined per stage over `StageScalars.transform_weights` (§1). The dense reference, its partners, the coordinate dual, `node_xy`, `weights` and `valid` are then stored at the section edges, which are `z_edges`. `omega` is `grids.omega` itself, `omega_fine` the problem's fine spectrum — which sizes the support mask `valid` and nothing else, the state being a coarse spectrum and both maps therefore coarse, `(..., n_omega)` — and `z_edges` covers the run. When it is present the general route reads these tables and the real static `synthesis`, `projection` and `projection_coarse` above are **declared placeholders**: the single zero column `(1, n_modes)` and its two `(n_modes, 1)` transposes, with `node_chunk = 1` and no taper, node weights or geometry, all checked by `Transform.__post_init__`, so that a route that does not use them cannot carry a second, silently wrong transform. The route's own chunking is its backend's ledger and not `node_chunk`. It is never present beside spatial `geometry`; a general guide has `n_pol = 1` and three physical components per channel (`MAX_N_COMPONENTS`), which is not the same thing as three state polarisations (`MAX_N_POL` stays 2). Every backend executes it in one combination — a field grid, a certified identity `linear.modal_metric` or the dense metric of a declared unitary change of basis, the tables' own execution maps at one interval of degree zero (or a moving guide's maps of any sections and degree beside the `"frozen_residual"` coupled flow that carries their connection; CF4 and no flow at all are refused by name beside them), `n_pol = 1`, the constant unit `power_weight` and each region's declared responses, instantaneous and delayed — and refuses every other by name, before it allocates. A declared change of basis also composes its identity into `gauge_id`, `<tracking>+circular(i,j)`, which the metric must match |

For radial geometry, `synthesis` is the inverse raw QDHT times the temporal pad
scale, `projection` is the forward QDHT, and `projection_coarse` equals
`projection`. Their departure from an exact inverse is the independently derived
truncation-plus-roundoff contract in [the Hankel manual](../guide/free-space/hankel.md).
For free geometry all three matrices have shape `(0, 0)`; the transform uses the
geometry's FFT axes and scalings instead. Spatial transforms have `_n_pol=1`,
no `taper_scale`, and share the scalar map between the state's polarisation
components. The node-buffer budget applies (1 GiB by default on spatial grids, 64 MiB on
modal routes).

`SpatialGeometry` is a frozen leaf in `translate.spatial`, with the following fields:

- `kind`: `"radial"`, `"free"` or `"slab"`: the hybrid planar
  transform (`solprop.modes.slab.PlanarGrid`), slab modes in x on
  the transform's own `synthesis` `(n_x, n_p)`, `projection` and `projection_coarse`
  `(n_p, n_x)` (the slab rule, pad scale in the synthesis), times a raw DFT in y.
  Its state columns are `(p, k_y)`, flattened `p + n_p*iy` with the mode index
  fastest, amplitudes in sqrt(W/m) per y frequency; its nodes `(x, y)`,
  `ix + n_x*iy`. `n_modes = n_p*n_y` and `n_nodes = n_x*n_y`, so its column
  vectors (`spectral_weights`, `on_axis`) and its node vectors (the rest) have
  different lengths. Its `spectral_axes` is `(k_y,)` alone; its `spectral_weights`
  are `2 dy/(ε₀ c n_y)` in m·Ω, the raw DFT's `dy/n_y` divided by `ε₀c/2`, so the
  common `ε₀c/2` energy measure gives joules; its `window` and `absorber_rate` depend
  on y alone (checked), so a backend may apply them per `(p, y)` row; its scalings
  are the y FFT's, `1` and `1/n_y`. A constant gap and constant media only; the
  operator is §3's with `kappa_sq`, and the prefactor is §3's per column.
  `numpy_ref`, `numba_cpu` and native CPU execute it.
- `shape`: `(n_r,)` or `(n_x, n_y)`, each dimension at least 2, except `n_y = 1` for a
  Cartesian line uniform in y (`grid.LineGrid`), whose one y sample is `y = 0` with
  `k_y = 0` and unit measure: its weights are then in m and its integrals per metre of
  y.
- `axes`: tuple of physical `r` or `(x, y)` float64 axes in m.
- `spectral_axes`: tuple of `k_r` or `(k_x, k_y)` float64 axes in rad/m;
  Cartesian frequencies have FFT order.
- `spatial_weights` and `spectral_weights`: positive float64 vectors in flattened
  column order. Area weights have units m^2; spectral weights have units m^-2
  for Hankel spectra and m^2 for raw Cartesian DFT spectra.
- `coordinates`: float64 `(n_nodes, 2)` pairs `(r, 0)` or `(x, y)` in m.
- `window`: float64 `(n_nodes,)` collar profile in [0, 1].
- `absorber_rate`: nonnegative float64 `(n_nodes,)` spatial power loss in m^-1,
  applied as `-absorber_rate*E/2` inside each nonlinear evaluation, independently
  of the nonlinear material prefactor.
- `on_axis`: complex128 `(n_modes,)` linear weights reconstructing the physical
  axis from the transverse spectrum. Temporal oversampling is not included.
- `forward_scale`, `inverse_scale`: positive finite common transverse scalings;
  m^2 and m^-2 for the radial pair, dimensionless for Cartesian transforms.

The spectral weights define spatial energy and noise normalisation. The physical
weights, coordinates and on-axis reconstruction support fluence-based spatial
statistics. Physical spatial fields are in V/m; radial spectra are in V m, and
Cartesian raw DFT spectra are in V/m.

`synthesis` carries the **pad** scale `n_to/n_t` of the oversampling round trip, so that
the field handed to the response pass is the physical one in V/m, and the crop scale goes
into `prefactor` (§5); why each sits where it does is in
[the kernel problem](kernel-problem.md#where-the-normalisations-live).

For the mode-averaged case the synthesis is `pad_scale / sqrt(ε₀ c A_eff(0) / 2)` and the
projection is 1, the reciprocal being already inside `prefactor`; `taper_scale(z) =
sqrt(A_eff(0)/A_eff(z))`, which for a scale-invariant mode is the `a(0)/a(z)` of
`modes.quadrature.taper_scale`, so one mechanism serves both geometries.

## 5. Responses (the compiled nonlinear term)

| field | shape / dtype | meaning |
|---|---|---|
| `response_order` | tuple of enum | the fused-pass sequence (e.g. `intensity, ionisation_rate, plasma_scan, plasma_current, kerr, raman_drive, accumulate`) |
| `kerr_coefficients` | frozen dataclass | `chi3`-derived coefficients with the transform normalisation folded in; THG flag; vector cross-term factors |
| `dispersive_kerr` | `DispersiveKerrTerms` or `None` | the frequency-dependent electronic response, a leaf described below: one channel per population with its own fine-grid multipliers and density table, `None` without one |
| `ionisation_table` | tuple of `RateTable` | rate tables in plasma-channel order, identical to each channel's `rate`; empty without a plasma response |
| `plasma_channels` | tuple of `PlasmaChannel` | labelled independent exposure channels, each with its own density, initial fraction, rate, potential and free-current coefficient; empty without plasma |
| `plasma_integral` | `RunningIntegral` | the running-integral rule of the plasma column, a leaf described below, one per problem because the channels share the current integrals; the trapezoid without plasma |
| `raman_terms` | tuple of `RamanCoefficients` | Raman-active components in species order, each with its amplitude density, label, undamped rows, decay tables, fraction and drive form |
| `density` | scalar Chebyshev table | total number density along z in m^-3, or the dimensionless normalisation for a solid/GNLSE |
| `labels` | tuple of str | unique stable component labels, in species order |
| `species_names` | tuple of str | chemical species names in the same order; names may repeat |
| `species_densities` | tuple of scalar Chebyshev tables | one number density per label, including zero and non-ionising populations |
| `spectral_responses` | tuple of `EnvelopePlasmaAssembly` | explicit post-transform source assembly, empty or the same object as `envelope_plasma` |
| `envelope_plasma` | `EnvelopePlasmaAssembly` or None | labelled envelope channels with validated instantaneous rate/cycle rules and coarse-axis Drude/loss multipliers; ordinary polarisation remains a separate stream until spectral assembly |
| `prefactor_representation` | `"constant" \| "expansion" \| "table" \| "separable"` | which representation carries the prefactor, following §3 |
| `prefactor` | `(n_intervals, n_deg, n_modes, n_pol, n_omega) complex128` Chebyshev table or `None` | the UPPE/envelope prefactor `−iω/(2ε₀c Re n_eff)` (or its envelope form) times the crop scale — on the general modal route the **real crop scale alone**, the `−iω` being `transform.modal_tables.source_frequency_factor` and the reciprocal normalisation being inside Q — as a table along z over the same span and the same declared kinks as Φ (a `"constant"` prefactor stays one interval of degree 0 and does not carry a response path's kinks); the real part of the index only, the loss stays in `D`. Exactly zero outside the active mask at every position. Present for `"table"` and for `"constant"`, whose one interval of degree 0 is exact, and absent for `"expansion"` |
| `prefactor_expansion` | frozen dataclass or `None` | the same quantity as §3's `ParameterExpansion`, with the plain products of the basis polynomials as its weights rather than their integrals, since the prefactor is a value at z and not a cumulative one. Present exactly when `prefactor_representation` is `"expansion"` |
| `prefactor_constant` | `(n_modes, n_pol, n_omega) complex128` or `None` | the prefactor itself, present exactly when `prefactor_representation` is `"constant"` — a mode set that does not vary along z, so the representation beside it is degenerate (one interval of degree zero, or one term) — and a backend takes the array instead of a Clenshaw pass; `validate` checks that it is that degenerate and agrees with the array, and nothing else, so a synthetic z-varying operator on a fixed fibre still validates. The mirror of §3's `phi_constant` |
| `temporal_absorber` | `(n_t,) float64` or `None` | rate α(t) on the coarse grid, for ADR-025's split-step scheme |
| `temporal_absorber_fine` | `(n_to,) float64` or `None` | the same rate on the fine grid, for ADR-025's in-the-nonlinear-term scheme; both are carried so that switching schemes is not a retranslation |
| `absorber_scheme` | `"split_step" \| "nonlinear"` | the temporal-absorber application: the default in-stage term (`"nonlinear"`), or the retained first-order split alternative. ADR-025 records the completed comparison and stability limits |
| `separable` | `SeparableOperator` or `None` | present exactly for `"separable"`, shared by identity with `linear.separable`; the prefactor table, constant and expansion are then absent |
| `regional_responses` | `RegionalResponseTables` or `None` | which material responds at each node of `transform.modal_tables`, and with what rank-four tensor and delayed kernel. `response_region` is an `int32` node-region index and never a species or plasma channel index; `chi3` is in m²/V² as Chebyshev coefficients along z, the tensors are dimensionless over 1, 2 or 3 Cartesian components, the kernels are in 1/s on the doubled fine axis and the temporal weight in s. The component count is the one `transform.modal_tables` resolves. The existing response owners remain the source of every law. The record carries the **instantaneous Kerr term** and, when its term axis is non-empty, **delayed terms**: the silica tensor Raman model's two channels. Every backend executes it as part of the general modal route — `chi3` at one interval of degree zero, the component count of `transform.modal_tables`, and each region's declared delayed response as well as its instantaneous one — and refuses every other by name, before it allocates. **`model_id` is normative.** It is a `MODEL_ID_SEPARATOR`-joined (`"+"`) list of model tags whose **first** tag names the instantaneous cubic's form, one of `KERR_FORM_MODEL_IDS` (`"isotropic-kerr"` for the analytic-signal form, `"isotropic-kerr-real"` for the real-field one), and whose remaining tags name the delayed models in compilation order, `DELAYED_MODEL_ID` (`"silica-tensor-raman"`) being the only one so far; `instantaneous_model_id` composes it and a backend reads the form back from the first tag. A backend **must refuse an identity outside the vocabulary by name rather than executing it as a neighbour**, and must refuse a record whose tags and term axis disagree: the backends genuinely branch their physics on this string — the real and analytic cubics contract different tensors against different fields — so the identity is part of the contract even though no schema field carries it |

The envelope source leaves are frozen in
`tests/translate/test_envelope_plasma_compilation.py`. The full field route keeps
its own `plasma_channels`; `population_channels` selects those or the envelope
assembly's channels for position-dependent density and initial-fraction tuples.
No population history crosses the seam or survives an RHS evaluation.

- `EnvelopePlasmaAssembly(channels, factors)` has ordered, uniquely labelled
  populations and one coarse-axis multiplier record.
- `EnvelopePlasmaChannel(label, species_identity, ionisation_potential, density,
  preionisation, cycle, charge_mass_ratio, electrons_per_event)` carries compiled
  scalar paths and SI constants. Single ionisation fixes `electrons_per_event=1`.
- `CycleIntegralData(rate, rules, validation, identity, moment_rtol,
  negligible_rate)` retains the explicitly instantaneous rate convention and
  species binding. `AngularRule(nodes, weights)` contains the 64, 128, 256 and
  512 point split rules. `CycleValidation(probes, residuals, cutoff_rate,
  format_version)` records independent Cartesian-cycle checks of the averaged
  rate and both directional moments, with the model cutoff audited separately.
- `EnvelopeSourceFactors(active, drude, loss, carrier_policy, omega0)` owns
  read-only contiguous arrays on the coarse absolute-frequency axis. The
  multipliers are `-1/omega**2` and `-1j/omega` inside the positive active mask,
  zero elsewhere. `KernelProblem.validate()` binds these values to its grid.
  `shock=False` freezes their denominators and the explicit UPPE numerator at
  `omega0`, retaining mode-index dispersion and masks.

Execution order is cycle moments, a fresh trapezoidal population scan per channel
and physical node, formation of Drude drive and loss current, material collar,
spatial projection where applicable, two forward temporal transforms, coarse
cropping, spectral addition to ordinary polarisation, and the UPPE
prefactor. Cartesian propagation transforms the currents within each physical
node chunk before the transverse forward transform. The embedded quadrature
alarm projects and transforms both streams too. There is no inverse transform to
manufacture a time-domain plasma polarisation.

The CPU workspace owns `(2, chunk*n_pol, n_to)` complex sources, five real
`(chunk, n_to)` histories, one integer status per node, two projected complex
streams and their spectra, projection scratch, and optional embedded streams.
Its plans keep the full chunk width; unused rows are cleared for a final partial
chunk. Rules are packed once, and all owned buffers and measured FFT plan allocator
growth enter the memory ledger. `native` owns the corresponding resident
streams, rules, histories and status arrays. A planned scalar status readback
per channel/chunk rejects failed angular refinement before any current transform.
NumPy, Numba CPU and native scan the observed envelope field afresh for population
statistics. [backends.md §9](backends.md) records qualification separately for each
execution space.

**Each `PlasmaChannel` is a leaf**, like `kerr_coefficients`, so its fields are not rows
of this document; they are `charge_mass_ratio` (`e²/mₑ` in C²/kg, the constant of the
free-electron current `(e²/mₑ)∫f E dt`), `ionisation_potential` (in J, the constant of the
loss current `I_p W (1−f) E/|E|²`), `preionisation` (a `ZProfile` table of `f₀`, a constant
table for the plain number a response carries, so a backend reads one value per plane), and
`rate`, the ionisation rate table. That table **is** the `ionisation_table` row above — the
same object, so a run carries one rate and `validate` checks the identity. The ionised
fraction is `f = 1 − (1 − f₀)e^{−∫W dt}`, with the initial neutral fraction
multiplying the survival exponential. The
group is present only on a field grid, which `validate` also checks.

**`plasma_integral` is a leaf** (`nonlinear.ionisation.RunningIntegral`), designed by
`translate.plasma_integration.design_running_integral`. Its fields are `policy`
(`"band"` or `"trapezoid"`), `taps` (float64 `(K+1,)`, centre first, `(1.0,)` for the
trapezoid), `dt` and `band_edge` (the fine spacing in s and the band edge
$\omega^{\rm win}_{\max}$ in rad/s the taps were designed for, `None` for the
trapezoid), `certified_error` (the certified $\max|H-1|$ in band), `boundary` and
`exposure`. Each running integral is the cumulative trapezoid of the filtered
integrand $\tilde y_n = c_0 y_n + \sum_{k=1}^K c_k (y_{n-k} + y_{n+k})$. The reference
sums it centre first, then each pair added before it is weighted; a backend may sum the
stencil and the running integrals in any order, held to the stencil's
$\gamma_{2K+2} A_n$ bound and the absolute prefix bounds of the scans' signed increments
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md) §4.6). The $K$ samples past
each end are $y_{-m} = 2y_0 - y_m$ and $y_{N-1+m} = 2y_{N-1} - y_{N-1-m}$. The exposure
is floored at zero pointwise, $X_n = \max(S_n, 0)$, before the fraction is formed; the
loss current keeps the raw rate. So `plasma_scan` finalises the fraction at sample $n$
from rates up to $n+K$, the free current from samples up to $n+2K$, and
the production column stops at the total current. The physical oracle
`PlasmaResponse.evaluate` also integrates that current to polarisation, from samples up
to $n+3K$. The `"trapezoid"` rule has no filter, no boundary and no floor.

**The held field-plasma source** ([SEP 27](seps/archive/sep-0027-held-plasma-current.md)).
The field-plasma column writes three complete streams: the collared non-plasma
polarisation `P`, the collared current `X = gJ` and its collar-derivative companion
`Z = g'J`, with `g'` the analytic `grids.t_window_fine_derivative` (§2). On active bins
their spectra combine as `P + X/(iω) − Z/(iω)²`; inactive bins are exactly zero and no
zero-frequency division executes. Modal routes project before transforming; radial
routes transform on nodes, combine and crop, then project once; Cartesian, line and
planar routes combine before their transverse transform. The derivative and every
additional buffer are charged to memory admission. The steady-current certificate uses
the complete fine transform (DC included) in its denominator, must not exceed 1e-10 and
is recorded under `plasma_source`.

`response_order` reaches `intensity, ionisation_rate, plasma_scan, plasma_current, kerr,
accumulate` for a problem declaring both a Kerr and a plasma response; `dispersive_kerr`
follows `kerr` when a dispersive electronic response is compiled; `raman_drive` joins
it after them when a Raman response is compiled, so a gas-filled capillary with all three
reaches `intensity, ionisation_rate, plasma_scan, plasma_current, kerr, raman_drive,
accumulate`.

**Each member of `raman_terms` is a leaf**, like a plasma channel, so its fields are not
rows of this document; they are `undamped` (`(n_terms, n_to_doubled) float64`: each
independently damped term's undamped response, causal from `t = 0`, tapered by the grid's
temporal collar, zero-padded to the doubled grid, sampled once per grid by
`translate.raman.undamped_kernels`; `n_terms` is 1 for a glass or a vibration-only gas
and 2 for a gas with a rotation and a vibration), `decay_rates` (one `ZProfile` table per
row: the decay rate `1/τ₂` along z, composed by the translation layer from the term's
dephasing and the density path, `translate.raman.decay_rate_profiles`, finite everywhere
and capped at `MAX_DECAY_EXPONENT / dt_fine`), `fraction` (`f_R` multiplying the summed
kernel, in `(0, 1]`; 1 for an absolute molecular response) and `thg`, the drive form. The
kernel damps row `k` by `exp(−γ_k(z) t)`, sums, scales by `fraction` and transforms
(`rfft · dt`, one real transform of `n_to_doubled`) whenever a rate changes, memoising the
spectrum; the convolution is `irfft(spectrum · rfft(drive padded))` on the doubled grid.
Each leaf also carries `label` and `density`, the component identity and its amplitude
density, separate from the density the linewidth reads.

The leaf splits the kernel into the expensive part, which is fixed, and the cheap part,
which is not, and hands a backend arrays and scalar paths rather than physics objects.
The decay rates are paths rather than dephasing objects so that the contract stays a
description of arrays and scalar paths along z, as it is for the density itself, and the
physics that turns a linewidth into a rate stays in the physics layer; a coherence that
collapses at zero density carries the cap `MAX_DECAY_EXPONENT / dt_fine`
(`708/dt_fine`), which damps it within one sample. In the scalar case a backend evaluates

$$
P_R(t) = \varepsilon_0\,\rho\,E(t)\;
\mathrm{irfft}\!\left[\tilde H \cdot \mathrm{rfft}(D)\right](t)
$$

per chunk of quadrature nodes (`solprop.backends.numpy_ref.passes.raman_polarisation`),
with $D$ the real drive of the grid written into the first half of an array of twice the
length. The damped kernel spectrum $\tilde H$ does not depend on the nodes, so a stage
forms it once and shares it between the chunks
(`solprop.backends.numpy_ref.passes.raman_spectrum`: each row damped at the rate its
table gives at this plane, summed, scaled by `fraction` and transformed). A backend
memoises it on **the rates**, not on the density, because the rates are what it depends
on: a glass's are zero at every density, so its kernel is transformed once for a whole
run, and a gas on a pressure gradient refreshes it when its stage positions present new
rates. Only the most recent set of rates is remembered: a gradient presents a new density
at every stage, so a growing cache would hold thousands of spectra and hit none of them.
A change of rate costs one exponential per term over `n_to` samples and one or two real
transforms of length `2 n_to`, against `n_modes · n_pol` transforms of length `n_to` per
stage for the field itself.

Each Raman leaf also carries `tensor_weights`, a read-only contiguous float64
`(n_terms, 2)` array, and `tensor_capable`, a boolean. Isotropic rows carry
`(1, 0)` and rank-two rows `(1/4, 3/4)`. A scalar-only row carries `(1, 0)` with
capability false; a response is vector-capable only if every term is qualified.
The weights form trace/traceless effective kernels after each physical row is
damped; amplitude densities and fractions apply before combining components.
Scalar execution keeps the ordered unweighted kernel sum. Vector execution
uses `S=Qxx+Qyy`, `D=Qxx-Qyy`, `C=2Qxy` and three convolutions, or just `S`
when all traceless weights vanish. These are local Cartesian components after
synthesis. Unsupported scalar-only vector responses are refused before execution.

**The drive form is carried, and the two refusals that make it true stand.** Which real
quadratic form drives the coherence on a *field* grid — `E²`, or the cycle-averaged
`|E_a|²/2` — is a property of the problem and not of each response: a drive that keeps the
`2ω` beat radiates a third harmonic through the later multiplication by the field, and a
grid built without one does not oversample enough to hold it. So the translation layer
checks that the declared Kerr and Raman responses agree about `thg` on a field grid
and refuses a field-grid Raman response with no Kerr response beside it; the answer
reaches a backend as each Raman leaf's `thg` rather than being inferred from
`kerr_coefficients`' `analytic_signal` flag. An envelope
grid's drive is `|A|²/2` whatever the flag says, so neither check applies there and a Raman
response may stand alone.

The `undamped` rows are `h` itself and carry **no** `ε₀`: the response writes it outside
the convolution integral (`docs/guide/media/raman.md`), so the reference backend is the one place
a backend reads a physical constant. Folding it into the rows would change their stated
units and loosen the rounding-level agreement with `RamanResponse.evaluate` that pins the
pass.

**`dispersive_kerr` is a leaf** (`translate.responses.DispersiveKerrTerms`), so its
fields are not rows of this document. It is the frequency-dependent electronic
response of [SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md) reduced to arrays, sampled by the translation layer on the problem's own fine
spectral layout at **absolute** angular frequency: the one-sided `(n_to // 2 + 1,)`
layout on a field grid, the full FFT-order `(n_to,)` layout on an envelope grid. Its
fields are `evaluator` (`"symmetric"` or `"polynomial"`), `sector` (`"full"`, every
quartet, for `thg=True`; `"conserving"`, the quartets with two positive and two negative
signed legs, for `thg=False`), `carrier` (true exactly for the full sector on an
envelope grid, which needs `grids.carrier_2w0`), `labels` and `coefficients` (one per
channel: the population and `ε₀ γ₀`, the static coefficient per unit density),
`positive` (bool, `Π₊`: the bins of strictly positive absolute frequency, the field
grid's DC and Nyquist bins excluded), `filters` (float64 `(n_channels, n_omega_fine)`:
the symmetric factor `g`, or the polynomial's bounded leg weight `d̃ = min((ω/2πc)², ν_c²)`
in m⁻²), `outputs` (float64 `(n_channels, 2, n_omega_fine)`, the polynomial's output
multipliers `1 + a d̃ + (b + b′) d̃²` and `3a + 6b d̃`, `None` for the symmetric
evaluator), `products` (float64 `(n_channels, 2)`, the polynomial's constant weights
`3(b + b′)` and `6b` in m⁴, `None` for the symmetric evaluator) and `densities` (one
scalar Chebyshev table per channel, m⁻³, read through the stage bundle's
`dispersive_kerr_densities`) and `regions` (int32 `(n_channels,)`, present exactly on
the general modal route: the node region of `transform.modal_tables.region_index` each
channel acts in, a region's gas fill giving one channel per species; `None` on every
other route, where a channel is a population and acts at every node). The channels
combine by evaluator, in the groups of `DispersiveKerrTerms.groups()`. For the
**polynomial** a backend evaluates, per channel and node,
`P += coefficient · ρ · cubic(E)`; for the **symmetric** evaluator, per group of the
channels acting at the same nodes (every channel, or one general modal region's), one
`P += F · cubic(E)` under the mixture filter `g_mix = [Σ_s f_s g_s⁴ / F]^(1/4)`, with
`f_s = coefficient_s · ρ_s` at the stage and `F = Σ_s f_s`, rebuilt whenever the
densities change (`solprop.nonlinear.responses.mixture_filter`: the species summed in
channel order, the fourth root as two square roots, a unit filter where `F = 0`; one
channel is its own `g` and `coefficient · ρ` unchanged). `cubic` is
`solprop.nonlinear.responses.dispersive_cubic` on the multipliers: the
filtered fields, their isotropic-Kleinman trilinear products in the sector (on an
envelope grid the three-photon group carries `carrier_2w0` and the one-into-three group
its conjugate, and `Π₊` acts on absolute frequency), the temporal collar
`grids.t_window_fine` on each product, each product's spectrum times its output
multiplier, and one inverse transform: `G[w (G E)³]`, the collar before the output
multipliers, and the response is **not** windowed again with the other responses'
polarisation. It reads the bare field and changes no other response's source. The
polynomial's channels are the exact per-species sum, which a backend may rewrite
exactly -- species with equal leg bounds share their derivative fields and add their
coefficient tables, as `numba_cpu` does -- but not approximate. The symmetric
evaluator's mixture filter is SEP 6 §6's. `numpy_ref` and `numba_cpu` execute it on every route: the fused node path
(single-mode, mode-averaged multimode and radial), the Cartesian free-space, line and
hybrid planar pipelines, and the general modal route at the nodes of each channel's
region, on the real three-component field, beside the region tensors of
`regional_responses` (the route has no collar). `numba_cpu` moves the multipliers off
the nodes -- onto the modal columns, onto each chunk's columns or rows, or into the
general route's synthesis ([backends.md
§3.11](backends.md#311-the-dispersive-electronic-response)). `native` executes it on every
route with the same structure ([backends.md §4](backends.md#4-native-the-compiled-cpu-and-gpu-backend-a-second-distribution)).

Vector polarisation: `n_pol == 2` selects the vector-isotropic Kerr form; tensor responses
(χ⁽²⁾) carry their own coefficient arrays when they exist. The transverse components are
carried by §4's `synthesis` rows **or** by the state's polarisation axis and never by both,
so the product of `transform`'s component count and `n_pol` is at most two; `validate`
checks it.

**`prefactor` follows z, and the two halves of it are carried in different places.**
`A_eff(z)` is carried by §4's `taper_scale`, which rescales the synthesis and divides the
projection; the table therefore holds the area at the reference position and lets only
`Re n_eff(ω, z)` vary. The identity `prefactor(z) = mode_average_prefactor(z) ·
taper_scale(z) · crop_scale` holds at every position and is a test; on a fill, where the
bore is constant, the table simply *is* `mode_average_prefactor(z) · crop_scale`.

The prefactor is built by the machinery of §3 minus the parts it does not need. On the
expansion it is the same construction with the same parameters and the plain products as
weights; on the z table the mode set's scalar paths are resolved first, so each interval
holds a lower degree. No round-off floor is measured and no tolerance relaxed, unlike Φ:
those exist to recover from the eight digits the co-moving frame cancels out of `β`, and
`−iω√(ε₀cA_eff/2)/(2ε₀c·Re n_eff)` has no such cancellation, so the build converges at
the tolerance it is asked for. Why the prefactor follows z, and what carrying it costs, is
in [the kernel problem](kernel-problem.md#where-the-normalisations-live).

A backend evaluates the prefactor only through `translate.responses.prefactor_at`, as it
evaluates the linear flow only through `translate.linop.delta_phi` and `unitary_phase`,
never by reading the coefficient arrays. That is what lets the representation change —
a z table, an expansion in the scalar parameters, a constant — without a backend
noticing.

Each plasma channel also carries `label`, `density` and `electrons_per_event=1.0`. Dissociation and multiple electrons per event are unsupported. Channel
labels select species metadata by name, and rates are identity-checked per channel.
Kerr coefficients carry their own scalar `density` table: for several components
it contains `sum rho_s gamma3_s`, with `coefficient=epsilon_0`; one component keeps
the single-species coefficient and density multiplication order. Raman components carry
`label` and `density` beside their other fields. Their linewidths use total
number density, an explicit approximation; their amplitudes use their own populations.
`CompiledResponses.n_species` is a property, the number of labels. Compatibility
properties `plasma_coefficients` and `raman_coefficients` return the sole leaf only
when that specialised path applies; channel-aware code consumes the tuples.

The `StageScalars` fields, in order, are `z0`, `z1`, `phi_delta`, `phi_at`,
`prefactor`, `density`, `taper_scale`, `raman_rates`, `kerr_density`,
`plasma_densities`, `plasma_preionisation`, `raman_densities`, `flow_section`,
`flow_weights`, `transform_weights` (these three described in §1) and
`dispersive_kerr_densities`, each channel of `dispersive_kerr` its number density in
m^-3 at the stage, in the leaf's order, empty without the leaf. The plasma tuples have
one entry per channel; Raman densities one per component; flattened Raman rates
visit every kernel row in component order. The `preionisation` property serves the
one-channel specialisation. All changing Raman amplitudes and rates participate in
kernel-cache validity; the one-component kernel keeps the single-species amplitude order.

## 6. Initial state, noise and batch parameters

### Optional TDSE source binding

`tdse` describes the atomic basis, prepared ground state, discrete shadow order,
output selection, calibration and response lattice as canonical JSON and a SHA-256
identity. The scalar fields are checked against this record. It contains no
native handle. The driver passes a `translate.tdse.TDSEBinding` explicitly to
`registry.select(..., tdse_binding=...)`, constructs and closes it outside the
backend, and refuses an unsupported route before expensive setup.

The binding accepts chronological, real SI fields `(n_nodes, n_to)` and returns
owning read-only `TDSEResult` arrays. Each consumed sample is valid and finite.
Its atomic counter is charged immediately before every provider invocation,
including failed calls and requested diagnostics. Stable chunks are reserved for
the entire Adams trial before startup. Unused reservations are released on every
exit path. `restore_call_ledgers(logical, atomic)` restores cumulative counters
after loading a field; it restores no Adams history or atomic state.

The host binding protocol also labels its evaluations for an optional raw archive. Each reservation opens one optical trial, `evaluate` takes
an optional `density` label (m⁻³), and `record_trial(accepted=...)` reports the
stepper's decision on the latest trial. The reference backend calls it from its
step record (`_record`), which every Adams attempt reaches, rejected or accepted;
it changes no step, charge or reservation. None of this is part of the protected
`StepDriver` surface or the kernel problem, which compiled backends see, so it is
outside the kernel schema: Numba and the native backend refuse TDSE by name and never hold a
binding.

The backend independently projects macroscopic P and J with the existing modal
weights and crop scale. It subtracts unscaled discrete shadows, then applies
density and the declared scales once. The ordinary optmat cubic keeps its
existing temporal source window. Raw atomic dipole/current histories are never
modified or mean-subtracted. A native descriptor's source is always gated and
edge-free (SEP 26; `TDSEDescriptor.gated`); a cubic replacement's never is.

Every descriptor has one layout, `solprop.tdse-descriptor.v5`; any other schema tag is
refused. Its response record holds every
response field but the consistency policy: `response.model`, a haCC basis tagged
`kind="hacc_helium"` and `response.atomic_window` (`null` for the full lattice); a
recorded `source_filter` is refused. `lattice.time_collar` records the grid collar's
exact Planck breakpoints in seconds and `lattice.tail_policy="unchanged"` the
untouched raw history; the seam checks the collar against the actual fine-grid gate,
so a collar change changes the scientific identity. A windowed response's
`lattice.atomic_window` records the resolved duration, taper, nominal bounds, retained
sample range, `t_s`, `t_e` and the gate's Planck breakpoints; a haCC `setup` is the
pinned record `{"initial": "ground_root", "restart": "every_evaluation", "growing":
"refuse_below_cut", "norm_growth_limit": 2.0}`. The binding drives the atom with the
retained samples multiplied by the gate and pads the returned currents with zeros;
`TDSEResult` keeps its full-lattice shape. The consistency policy of a haCC response
never enters the descriptor.

For a gated source the gate is the atomic window's when one is set, else
`lattice.time_collar`. The backend forms `S = j_total - j1`, the gated current `g S`
and the edge array `g' S`, with `g'` the analytic gate derivative of
`TDSEDescriptor.source_gate_derivative()` (`solprop.grid.planck_taper_derivative`),
applies density and scale once to both, projects, transforms and crops both, and on
the active bins forms `X - E/(i omega)` before the current factor. It changes the
macroscopic current source; the atomic driving field and separate in-stage field
collar remain unchanged. Operand-derived roundoff indicators include the gate's
multiplication and the edge term. Translation refuses a steady-current certificate
above 1e-10, relative to the full transform of the gate
(`solprop.translate.tdse_source.steady_current_certificate`); the certificate is run
metadata (`tdse_source`) and part of the last-evaluation `edge` diagnostic, never part
of the descriptor or an identity. The gate is an explicit approximation: its 1 % full-source width robustness is not
established and the atomic-cut uncertainty remains
([TDSE accuracy and limitations](../guide/media/tdse.md#accuracy-and-limitations)).

Same-row embedded quadrature diagnostics identify the actual evaluated field
(`startup`, `predictor`, `corrected` or `diagnostic`) and carry absolute cancellation
floors. They are not accepted-field statistics. An exact accepted-field diagnostic
needs an identical cached input or another charged solve.

The initial-state fields below are independent of that optional binding.

| field | shape / dtype | meaning |
|---|---|---|
| `field0` | `(n_batch, n_modes, n_pol, n_omega) complex128` | initial state in the lab frame, FFT convention |
| `noise0` | same or `None` | modified shot-noise realisation |
| `noise_model` | `"none" \| "input" \| "modified"` | |
| `batch_params` | dict of `(n_batch,) float64` | per-member scalars recorded in the output; per-member table indices where tables differ |
| `z` | `float` | absolute initial position in m, non-negative and before the first requested save; true-power input is normalised at this position |

## 7. Stepper settings and schedule

| field | type | meaning |
|---|---|---|
| `tableau` | frozen dataclass | `solprop.stepper.tableau.Tableau`: nodes, matrix, solution weights, error weights `b − b̂`, orders, FSAL flag |
| `controller` | frozen dataclass | `solprop.stepper.controller.PIController`: `rtol`, `atol`, `safety`, `max_factor`, `min_factor`, `beta1`, `beta2`, `lag`, `reject_free`, `estimator_order`, and `weights` (`None` for the weak norm, a `(n_omega,)` array for the weighted one). The tolerances and the norm choice live here and nowhere else. |
| `dz_min`, `dz_max`, `dz_init` | float | caps on the controller's proposal; `dz_min` defaults to `solprop.stepper.controller.default_dz_min` of the run length and never applies to a landing step |
| `stretch` | float | landing stretch factor (`StepTargets.stretch`), default 1.1, equal to the controller's default growth cap but independent of it; 1 switches it off |
| `schedule` | `(n_steps,) float64` or `None` | fixed schedule to replay (lockstep batches require it); its edges must contain every save plane and hard kink, which `FixedSchedule.uniform` guarantees for a plain fixed-step run |
| `save_planes` | `(n_z,) float64` | z positions the driver will ask for; steps land on them unless `interpolate_planes` is set |
| `propagator_mode` | `"auto" \| "tabulate" \| "recompute"` | ADR-004 and [backends.md §3.6](backends.md). `"auto"` is the default and is the backend's own choice, resolved at construction and written into the run's `backend` metadata group: `tabulate` for `numpy_ref` and `recompute` for `numba_cpu` on a modal problem; `recompute` on every backend for a separable free-space operator, whose increment `D (z₂ − z₁)` is exact. Under the low-storage march `tabulate` is one volume of `exp(h D / 2)` reused while the step holds (a varying operator's `P` and `Q` for the step), admitted on the free and planar kinds too, and `"auto"` resolves to `tabulate` for a constant operator and `recompute` for a varying one (SEP 24 §4.3), and to `recompute` on a native device build, where an extra field read costs more than the exponential |
| `integrator` | `Multistep \| LowStorageRungeKutta \| None` | The method stepping the problem instead of the tableau, or `None` for the Runge-Kutta march. `Multistep` ([SEP 13](seps/archive/sep-0013-expensive-response-steppers.md), [SEP 16](seps/archive/sep-0016-unified-multistep.md)) is the Adams march in its exponential or interaction formulation, with an optional carrier frame (`carrier_frame`, [SEP 35](seps/archive/sep-0035-carrier-frame-exponential-adams.md), added in version 36: `"off"` or `"scalar"`, the latter on `numpy_ref`, `numba_cpu` and native CPU builds and refused by name elsewhere); it runs on `numpy_ref`, `numba_cpu` and Kokkos on every diagonal-flow route, and beside the constant-Schur and frozen-residual coupled flows in the exponential formulation; with a `schedule` it replays the schedule's steps (`AdamsMarch._replay_to`); TDSE remains NumPy-only. `LowStorageRungeKutta` ([SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md)) is RK4 in the midpoint interaction picture in three registers: it replays a `schedule` and nothing else, records the norm 0.0 ("not estimated"), refuses a coupled flow, the TDSE response, interpolated planes and a step past RK4's negative-real interval 2.785, and runs on `numpy_ref`, `numba_cpu` and solprop-native (a resident engine whose step is one native call) |
| `interval_cap` | bool | whether a step is capped at the width of the operator-table interval it starts in. `assemble` turns it off on the diagonal-flow routes and on for a coupled flow or a reject-free controller, unless set ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md) §4.1) |
| `auto_init` | bool | whether the start-up rule takes the step after the first from the first step's own scaled norm (`PIController.start_factor`, at most 100 times `dz_init`). Set by the convenience interfaces when no `dz_init` is given; off for a coupled flow or a reject-free controller ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md) §4.2) |
| `interpolate_planes` | bool | steps across the save planes and observes each inside its accepted step, through DOPRI5's continuous extension or, with an `integrator`, the Adams march's own polynomial (ADR-039); the end of the run and the hard kinks are still landed on. Refused with a schedule, a Runge-Kutta tableau without an extension, the per-step collar, a coupled flow and the `modified` noise |
| `step_planes` | int | a positive `k` also saves the end of every `k`-th accepted step since the load: the shared march returns from `propagate_to` there, short of its target, and the driver saves the plane and asks for the target again (ADR-039). `0` never stops short |

## 8. Statistics requests

| field | type | meaning |
|---|---|---|
| `stats_spec` | frozen dataclass | `solprop.stats.StatsSpec`: which statistics, energy windows (`(n_windows, 2)` in ω), periodic subset and period, validated against the catalogue at construction; and the far-field tables of [SEP 23](seps/archive/sep-0023-far-field-aperture-spectra.md), `apertures` (an `ApertureSet`: centres, diameters, distances, edges, profile, weighting, labels) and `spectral_regions` (`(n_regions, 4)` boxes of physical transverse wavevector in rad/m, the one leaf whose entries may be infinite), read only when a name that uses them is requested, so a backend that does not know a name refuses it rather than misreading a table. Fields of the leaf, not rows of the contract |

The `periodic` subset and its `period` are read **when the backend is built**, not when a
run is started: the shared step loop arms the between-planes channel from
`problem.stats_spec`, so a spec handed to `run_problem` that disagrees with the problem's
own about either is refused. Every name in the subset must carry the catalogue's
`between_planes` trait, which `StatsSpec` enforces at construction and by name; a backend
that cannot form one of them at all refuses the run the same way its plane statistics do.

The driver exposes three storage hooks; the shared march decides which accepted
steps produce rows and passes their exact history indices:

| hook | contract |
|---|---|
| `_reset_step_stats(*, release: bool)` | Discard rows on a fresh load (`True`, release storage) or at a new plane (`False`, retain capacity). Period counting remains in the shared march. |
| `_record_step_stats(spec: StatsSpec, z: float, step: int)` | Evaluate a row after boundary treatments, state commitment and position advancement. `step` includes rejected attempts. Device implementations retain device values and share their evaluator with `stats()`. |
| `_drain_step_stats() -> StepStats` | Return owning host rows at observation, then reset the row count while retaining capacity. A device implementation copies packed values once here. An empty drain transfers nothing. |

The defaults keep a host buffer and call `stats()`. No backend needs to
override the march or read its private row buffer. Device arrays remain behind the
seam. The native backend inherits the host defaults. A backend must implement the current
schema (§1),
including the batch admission interface of §1, explicit handling or refusal of envelope
source assemblies, and an explicit refusal of each general-modal leaf, section method and
spatial kind it does not execute.

## 9. What comes back

`Observation` (host arrays only, FFT convention): `z`, `field` in the lab frame `(n_batch,
n_modes, n_pol, n_omega)`, `noise_phase` (if a noise model is active, so processing can restore the
background), both `None` for an observation made with `observe(field=False)`: nothing then crosses to the host and nothing scans the field, so a
diverged field-free run is caught by the step loop's finite-norm rule and the
statistics; the spatial collar diagnostic is not evaluated there, and the backend
counts such planes in `metadata()` (`collar_diagnostic_skipped`) and logs the first, `step_history` (arrays of z, h, error norm, accepted flag, wall time since the
last observation; recorded whether the run was adaptive or replayed a schedule, since the
embedded estimate is computed either way), `n_steps`, `n_rejected`, and `step_stats`. `stats()` returns a dict of small host arrays
keyed by statistic name with a leading batch axis.

`step_stats` is a `StepStats`: `z` `(n_rows,)` and `step` `(n_rows,)` — the index of the
accepted step each row was taken after, into *this* observation's `step_history`, which
counts rejected attempts — and `values[name]` shaped `(n_rows, n_batch, *shape)`, the row
axis first and then the batch axis every statistic crossing this seam carries. Empty
unless the spec named a periodic subset, which is the default; the field has a default so
an `Observation` built without it is unchanged. It is flushed at every observation, like
the step history and over the same interval, and a row may not index past it.

## 10. Extension procedure

1. Add the field to the dataclass with a default under which a problem that does not use
   it is unchanged.
2. Add the row here and bump `schema_version`.
3. Extend `validate()`.
4. Implement in `numpy_ref` first, with a test; then in the other backends, with the
   equivalence test.
5. Update the memory budget in [architecture.md §8](architecture.md) if a buffer is added.
