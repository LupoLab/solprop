# Backends: executing the propagation

Everything that touches the propagated state during a run is reached through one seam, and
only the backends know how the state is laid out, transformed and stepped. This document
fixes the seam, the reference backend, compiled execution and the native backend (built
from `native/`), the threading model, the memory ledger and admission, which backend runs
what, and what each backend is for and where new features land (§10). Which
backend to choose for a run is the user guide's
[Choosing a backend](../guide/performance/backends.md). Decisions: ADR-002, ADR-003,
ADR-013, ADR-020, ADR-023, ADR-061, ADR-064.

## 1. The seam

Public simulation settings default to `backend="auto"`. `resolve_name` probes the
native package without importing it, resolves to `"native"` if it is installed and the
processor is at or above a build's baseline, and otherwise raises `BackendError`, saying what
to do; it stores that name before execution or worker transfer. `auto` has no fallback.
Native being installed is a packaging policy ([SEP 43](seps/archive/sep-0043-retire-numba-cpu.md) §5),
so a run does not have to prove that native could execute here, and nothing loads a runtime to
enforce it. Two requests resolve before the registry and never consult native: a TDSE
response's `auto` is `numpy_ref`, and single precision's is refused: name `numpy_ref`. Explicit names are preserved, so `backend="numpy_ref"` runs where native is
absent and `backend="native"` on a processor below the baseline raises the package's refusal.
`select` also accepts `"auto"`, checks `SOLPROP_REQUIRE_DEVICE` against the actual build before
anything is allocated, and logs the actual backend at INFO after
successful construction. Import, build, device and unsupported-problem errors stay visible and
stay distinct, and the registry finds the processor baseline out
by running the package's own pure-Python check by path, without importing the package. This
selection policy changes no kernel-contract member (ADR-029).

The protocol is drawn at the level of operations on the state, because an `xp`-style abstraction over
`numpy`/`jax.numpy`/`cupy` would force every backend to keep the array compiler's structure
(out-of-place transforms, separate scaling passes, materialised stages, one reduction per
statistic), and that structure is precisely what a fast backend exists to change.

```python
class Backend(Protocol):
    name: str
    max_batch: int  # 1 for backends that step one problem
    reads_transverse_edge: bool  # whether a requested real-space collar can run on it

    def set_initial(
        self,
        field: Field,
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
        resume: bool = False,
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

    # What the run layer asks beside those; StepDriver gives each its default.
    def set_stop(self, flag: StopFlag | None) -> None: ...  # polled between trials
    def set_held_plane(self, held: bool) -> None: ...  # the low-storage march's hold
    def attach_monitor(self, monitor: BoundaryMonitor) -> None: ...
    @property
    def in_flight(self) -> tuple[float, float] | None: ...  # a step abandoned there
    def held_observation(self) -> Observation | None: ...
    def march_snapshot(self) -> MarchSnapshot | None: ...  # the Adams history
    def march_snapshot_nbytes(self, z: float) -> int | None: ...  # copying nothing
    def march_export_refusal(self) -> str | None: ...
    def restore_march(self, snapshot: MarchSnapshot) -> str | None: ...
    def timing_probe(self, start: Start) -> TimingProbe: ...  # commits nothing
    @property
    def closed(self) -> bool: ...  # a failure can close a backend
```

The listing is abbreviated; `solprop/backends/protocol.py` holds the exact signatures and
docstrings, in the order of [architecture.md §9](architecture.md#9-backends-in-one-page).

Rules: a backend is constructed from a validated `PropagationProblem`
([kernel-contract.md](kernel-contract.md)), the `RunRequest` that carries the run's
policy ([SEP 47](seps/sep-0047-run-ownership.md) §5.2; it checks the two with
`validate_run` before it allocates) and host arrays, and returns host arrays; its
buffers, layout and plans are private; the state is never a parameter of `propagate_to` or
`observe`; the numbers are the contract — a backend is correct when it reproduces the
reference to the stated bound. Selection is by name (`backends.select(name, problem, request)`);
availability is detected with `importlib.util.find_spec` (importing an accelerator package
can initialise a device runtime), never cached; an unknown name raises `ValueError` listing
what is usable, a known name whose dependencies are missing raises `ImportError` saying
what to install, and a removed name keeps a tombstone (`REMOVED_BACKENDS`):
`backend="numba_cpu"` raises `BackendError` naming `native` and `numpy_ref`, wherever a
requested name is resolved. A recorded name is not a request, so old output and snapshots
that record it still load. `metadata()` says what the execution *was* — the name, the
resolved propagator mode, and whatever pools, plans and digests the backend has — and the
driver writes it into the run's `RunMetadata.backend`, outside the resume key.

**Registration.** A backend is one row of `registry.BACKENDS`, a `BackendEntry` that
names by import path (`"module:attribute"`) everything selection reads of it, so
`select`, `select_batch` and the policy checks read the row and never branch on the
name: the modules it `requires` and the `extras` or `install` text that supply them;
its `factory`, called as `factory(problem, request, *, grid, modes, profiles, options)`;
the transitional lockstep runner `batch(owners)` (`select_batch`, deleted in SEP 47
phase 5); and the native policies, each `None` on a row without it: `first_use`, called
before each construction (SEP 46's first-use calibration), `device`, the build
description that `SOLPROP_REQUIRE_DEVICE` reads (a row without one is refused a required
device by name), and `far_field`, the far-field statistics the installed build declares
(`statistics_refusal`). `auto` is `registry.AUTO`, `native`, with no fallback. Nothing
is imported until selection needs it, and only after `requires` is found. A new
backend (Tier 3, ADR-061) in another distribution registers with one row, and runs and groups through its
own `prepare` and `prepare_group` (**Selection** below); there is no entry-point
discovery, so the row is added to `BACKENDS` in this repository.

**Periodic-statistics storage hooks.** The seam carries three storage hooks for the
periodic statistics — `_reset_step_stats`, `_record_step_stats`, `_drain_step_stats` — so
that a device backend can keep its rows resident and copy them once per plane; the shared
march still decides which accepted steps produce rows and passes their history indices.
The CPU backends inherit the host defaults. The hooks are part of the frozen surface
([kernel-contract.md §8](kernel-contract.md)).

**Batches.** `BatchBackend` adds independent member observations and metadata through
`select_batch(owners)`. `numpy_ref` and `native` implement this
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
runner hands each member its own problem, so a batched general run is what its members
are alone, to the same rounding bound; a problem is one trajectory and carries no batch
axis. A member carrying a coupled flow is refused by every backend's batch.

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
schedule ([testing.md](testing.md) section 3). The native CPU build computes in **two
stages** — the partials of each column, `partials_(3, slots)`, in a team kernel, then the
host's sum over each row in index order (`collect_trial`) — and the device form is a
fixed-order tree reduction (a fixed block count, each block reducing its own contiguous
range, then one serial pass over the block results). Any other order (`parallel_reduce`, a
threaded tree) is admissible with the bound of its own tree. The native tests
(`native/tests/test_api_contract.py`, driving `thread_invariance_probe.py`) compare one
thread against many on a fixed schedule within the step's rounding bound.

**2. The device form of the stage.** A stage's z-dependent quantities cross the seam as a
small bundle of host scalars (`translate.stage.scalars`) against a fixed set of arrays
uploaded once (`translate.stage.terms`); `ΔΦ` is one linear combination of the terms over
the bundle's floats and nothing about the representation reaches the backend
([kernel-contract.md §3](kernel-contract.md)). The contract's z-table fallback has no such
split, so `PropagationProblem.device_ready` is false for it. NumPy evaluates the general
representation directly. The native CPU build gives the table the split itself (SEP 29 section 6.6): its coefficients
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
checks; `native` takes its own
`NATIVE`, C++ transcriptions (`native/include/solprop_native/chebyshev.hpp` and
`lagrange_basis.hpp`). Both perform the same operations in the same order under
strict arithmetic, so the rows a table keeps for its last positions are shared whichever
filled them. A record may add a third operation, an exponential Adams trial's
coefficient tables in one call (`Helpers.tables`, the composition of
`backends.multistep.trial_tables` over its own basis): `NATIVE` has it, in C++ beside
the basis (`trial_tables` in `lagrange_basis.hpp`), so its resident engine forms every
table of a trial in one call rather than a dozen; `PLAIN` composes them. Translation runs
before any backend exists and needs nothing compiled:
`clenshaw` and `PiecewiseChebyshev.__call__` without a record are NumPy's, and the shared
functions' defaults are `PLAIN`. A z table's state-sized evaluation
(`delta_phi`, and the rate of `phi_rate` on a z table) is not a stage scalar and stays on
that shared pass on `numpy_ref`; the native CPU build has
its own kernel (point 2). `tests/backends/test_helper_ownership.py` runs each march on
both backends in one process, with every other implementation made to raise after
translation, and a native z-table run of each march with the plain forms
forbidden.

**3. One driver, seven marches.** `StepDriver.propagate_to` is the one way in, and it hands
the steps to one of these loops:

| Loop | Steps | Where its state-sized work runs |
|---|---|---|
| `StepDriver._march_to`, in `backends/driver.py` | adaptive Runge-Kutta | the backend's `_trial_step` |
| `StepDriver._replay_to` | Runge-Kutta on a fixed schedule | the backend's `_trial_step` |
| `AdamsMarch.march_to`, in `backends/multistep.py` | adaptive Adams | an `AdamsEngine`: `HostHistory`, or `native`'s resident `_ResidentAdams` |
| `AdamsMarch._replay_to` | Adams on a fixed schedule | the same engines |
| `LowStorageMarch.replay_to`, in `backends/low_storage.py` | low-storage, always on a schedule | a `LowStorageEngine`: `HostLowStorage`, or `native`'s `_ResidentLowStorage`, whose step is one native call |
| `LockstepRunner._replay_to`, in `backends/batch.py` | a batch's members on one schedule | `_trial_step` under Runge-Kutta; each member's own march otherwise |
| `_NativeAdamsMarch.march_to`, with `native/src/adams_march.cpp` | Adams, adaptive and replayed, compiled (ADR-060) | the resident engine; it returns to Python only at a host event |
| `_NativeRungeKuttaMarch.march_to`, with `native/src/rk_march.cpp` | Runge-Kutta, adaptive and replayed, compiled (ADR-069, [SEP 45](seps/archive/sep-0045-native-rk-march.md)) | the staged trial and native norm; it returns to Python only at a host event |

Only the last two loops are compiled. The driver's Runge-Kutta march is the reference for the
compiled Runge-Kutta march, which holds to it through kernel contract §12 (ADR-069), and for the
march's bookkeeping. `AdamsMarch` is the reference for the compiled Adams march, which holds to it
through kernel contract §11 (ADR-060), and steps every configuration the compiled march doesn't
cover. A new compiled loop needs a contract that specifies it in full and a reference to compare
it with.

What stays common, and is never reimplemented, is what the loops do around a step:
`propagate_to` and `observe`, the history (`_record`, or `_record_attempt` and `_skip_step`
for the attempts a compiled march books together), the committed step's work
(`_take_step_stats`: **the statistics channel between planes** and the boundary monitor), the
natural stops, the noise models and **the progress reporter**. A backend supplies
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
and after the first plane a run allocates nothing per step. **On a device** the rows are the
same host buffer: the reductions run on the device, in the fixed-order tree of rule 1 above, and
each row returns one packed scalar transfer. The hooks (`_reset_step_stats`,
`_record_step_stats`, `_drain_step_stats`) would let a backend keep rows on the device and copy
them once per plane, but none does. A backend that
cannot evaluate one of the subset's statistics between planes refuses the spec by name, as it
refuses a plane statistic it cannot form.

**What a backend may do to make it cheap** is read what the accepted step already computed
rather than computing it again — but only where the state it describes is provably the state
the run adopted. `native` does this for the plasma pair: §3.1a has the measurement and the
predicate.

**The boundary monitor's reading** is the third thing the loop calls at the same point, and
the backend reduces it (SEP 21, ADR-057). At a check the driver asks
`_boundary_energies(state)` for five numbers, the state's energy and its four weighted
energies (`solprop.monitor.BoundaryEnergies`), and the backend forms them where its state
lives: `native` in its memory space through its statistics buffers (CPU and device builds
alike). Only the five cross to the host; `numpy_ref`'s host reduction
(`BoundaryMonitor.energies`) is the reference. `_bind_monitor` hands a backend the run's
monitor when it is attached, and `None` when a state is loaded. Under the `split_step`
scheme the backend's `_finish` — and the native engine's own finish, on every march —
adds what each accepted step's split takes of the power it finds, `P (1 − e^{−αh})`, to an
accumulator the next reading drains, so the removal is exact on unchecked steps too, and
keeps the energy and the guards' and collar's energies of that power halfway through the
factor, `P e^{−αh/2}`, which the reading returns beside the five (`BoundaryEnergies.found`,
kernel schema 43) and the monitor judges the occupancies on. The split collar's crossing limit
reads the state's energy per walk-off bucket at each check, in the same reduction style: one
histogram per thread and no atomics, which is why its bucket count has a compiled limit. The
input-noise floor is read on the host, once, at the launch.

**Export into a destination** ([SEP 47](seps/sep-0047-run-ownership.md) §5.4) is how a
session's field reaches the output. A session's `export(ticket, recipe, out=None)` fills a
host destination with the pending observation's field converted by an `ExportRecipe`
(`backends/export.py`): the true-power weight's square root on a modal route (a spatial
route's field measure is already the stored quantity), then `(-1)^k conj(.)` into the optics
convention. The output layer owns that definition (ADR-017); the sessions only evaluate it,
the reference by `apply_recipe` and `NativeSession` in one fused native pass whose complex
products keep NumPy's zero terms, so every route's stored bits are those of the two-step
`power_amplitude` then `spectrum_to_optics` path, signed zeros included. `apply_recipe`
writes each step straight into the destination (no copy first) and converts a large field's
rows in threads, as `stored_spectrum` does. A destination is validated by one rule,
`destination_refusal`, which names what is wrong: the shape, the dtype, the layout (one whose
elements overlap one another is not contiguous) or a read-only array.

The service loop (`driver/service.py`, `_Exports`) asks for one export per pending ticket
and shares it among the plane's consumers. A plane store that `takes_exports` gives the
export its own room with `field_destination`: `MemoryOutput` the block row the plane will
occupy, allocated with the reserved planes, so a kept field is written once, where it is
kept; `HDF5Output` none, so the session allocates a new array, which h5py writes into the
plane's chunk as it is. `save_exported` then stores the plane, which carries no field of
its own (its noise phase rides beside the field in a `StoredField`; a session's
`noise_phase(ticket)` forms it at the pending plane, from the problem's scalar tables on the
host, or, when `Phi` took the z table, from the table's slot at the plane, which
`NativeSession` reads through the native session under its lock, as the backend reads it
from its state, and a group's member from its own session). A plane observer reads
the stored field with no fetch, and a terminal checkpoint at the same plane takes the
internal convention from it by the exact inverse (`spectrum_to_internal`, the store's own
reader), with no second export or amplitude conversion. An output that is not a plane
store, or a store subclass that overrides `save`, is handed the consumer field
(`ExportRecipe(power=True, optics=False)`) on the plane, as before. A field-free plane
exports nothing. The room is never recycled: no other plane is given it and nothing writes
to it once the plane is stored, and an array a consumer retains is never written again.

A device session fills the same host destination: the recipe runs where the state is, into
the device's scratch, and one copy crosses into `out`; whether the conversion runs before or
after the crossing is the device's to measure. The CPU-only session prototype refuses device
builds, so this is phase 4's; the interface needs no change for it, and no device array
crosses it.

## 2. `numpy_ref`: the oracle

Plain NumPy and `scipy.fft`, one function per pass, no fusion, no in-place tricks, no
threading beyond what `scipy.fft` does with `workers=1`. It exists to be read and to be right:
every step of the anatomy in [architecture.md §8](architecture.md) is a named function with a
docstring stating the formula. It is slow (tens of separate array operations per stage) and
that is acceptable; it is the reference against which every other backend is validated, and
the supported executor of the short list of features `native` does not run yet: the TDSE
response, CF4 coupled-flow sections and single precision away from Apple silicon
([the list](../guide/performance/backends.md#features-that-run-on-the-reference-backend)).
A new feature joins the list only as an accepted exception (§10).

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
products — exact for any tensor, and the order `native` reproduces
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
name and with `BackendError`: a spectral response other than
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

**Single precision** ([SEP 33](seps/archive/sep-0033-single-precision-mlx.md), ADR-054). A
problem whose `precision` is single is executed by `NumpySingleBackend`
(`backends/numpy_ref/single.py`), which the registry constructs for `numpy_ref`: the
reference's passes and step loop in binary32, through hooks of the reference that leave
its binary64 arithmetic unchanged (`SourceTables`, `_propagator`, `_rounded`,
`_error_norm`, `_envelope_sources`). It narrows the source tables once at construction,
each with its block exponent (applied at upload where that is exact, refused otherwise
unless the pass carries it, as the Raman kernel's does); holds the state at the launch's
power-of-two scale `s`; forms every propagator, prefactor and z-table value in binary64
and rounds it once; evaluates Kerr, Raman (scalar, tensor, mixtures) and the field-grid
plasma in binary32 with each term's homogeneity factor in one host coefficient, the
stable population forms and blocked running integrals
(`translate.precision.prefix_sum`); reduces the controller's norms as scaled sums of
squares against `atol / s`; applies the split-step collar as the correction
`a + F(F^-1(a)(f - 1))` in binary32, `f - 1` formed in binary64 and rounded once (a
binary64 state keeps `F(F^-1(a) f)`); and checks after every trial that the
state and stages stayed complex64, and after every step that the finished state is
complex64 and finite. Its fixed-schedule batches are the plain `LockstepRunner`, each
member through its own binary32 trial: `NumpyBatchBackend`'s stage arithmetic over the
member axis is binary64, so it refuses a single-precision member by name, and every
lockstep batch refuses members of mixed precisions (`_validate_members`). The
ionisation rate's table
lookup, the envelope-grid plasma's cycle moments and the statistics are evaluated in
binary64 from the binary32 field, which is what the oracle defines. Its constructor applies the settings' own admission (`translate.precision.admit_precision`):
a problem that reaches `registry.select` without passing `SimulationSettings` is still
refused an adaptive `rtol` below the floor. The Adams integrator is admitted (#342):
binary32 Adams agrees with binary64 Adams to the method's own truncation on the survival
cases, and a single-precision history resumes in binary32 (`HostHistory.load_rows`,
`NumpySingleBackend._adams_restore`). `native` refuses a single-precision
problem by name.

**Selection.** `backends/registry.py`: `available()` detects with
`importlib.util.find_spec` and never caches; `select(name, problem, request, *, grid,
modes, profiles)` builds one, for the problem and the request it is run with: the
request's method, error control, itinerary and planes are lowered once, here (the native
session lowers its statistics as it starts, from profile tables cached per problem), and
a run's session serves that request adapted to one run (`RunRequest.for_run`); a
preparation refuses any other request by name (`driver_session.foreign_request`). A run
starts through the backend's own preparation shape (SEP 47 §5.2):
`backend.prepare(problem, request)` returns a `PreparedExecution` whose
`start(start, *, listener, started, stop, owns)` opens the backend's session, and
`backends.sessions.start_session` does nothing else, so a backend chooses its session
without the run layer naming it. The reference's opens `HostSession`; the native
backend's (`solprop_native.session.NativePrepared`) opens `NativeSession` where it admits
the run and the transitional `DriverSession` otherwise, logging the reason at debug level
(`session_refusal`, which a test replaces to force the bridge); any other driver's
(`backends.driver_session.DriverPrepared`) opens `DriverSession`. Statistics the problem does not
carry are refused by name before any session starts, on every executor, rather than left to the
first plane (`backends.session.statistics_unavailable`, read by every preparation, by
`check_group` and by the run layer): a modal statistic of a spatial route, a spatial one of a modal
route, the plasma integration diagnostics without the field-grid plasma column and the electron
density or peak ionisation rate without a compiled plasma response (a `BackendError` at every door).
`zdw` is read natively from a table of each mode's root along z, lowered once at preparation
(`_statistics._zdw_tables`): a z-independent mode's table holds its one root, the reference's at
every plane to the bit; any other mode's root is tabulated over the propagation's span with the
problem's table edges as knots, within a capped fit (`_ZDW_FIT`: each sample is a root search of
about 0.2 s), and held to `ZDW_TOLERANCE` (1e-6 of the wavelength, a picometre at a micrometre,
measured on a check set against the root search itself and recorded under `statistics_tables`), or
refused by name, as is a mode whose band holds a root over part of the run only. Not rounding level,
because the reference is not: its root scatters by a few times 1e-8 between neighbouring positions
on a gas capillary (`beta_2` is a finite difference), and a pressure gradient over a decade is
resolved to about 5e-7 within the fit's budget. Where the session's request fixes every position the statistic is taken at -- its start and save planes, with no step or interpolated plane -- and there are no more than `_ZDW_PLANE_LIMIT` (34), a z-varying mode's root is instead searched once at exactly those positions before the run and held in a table constant about each (`source: "save planes"`, error zero): the reference's own roots, at planes times modes searches. Native refuses by name, before anything is allocated, the metric
statistics of a general modal basis whose metric varies inside an interval
(`_statistics.metric_refusal`): the plan's metric root is exact only on a metric constant on each
interval. The native session's own statistics refusal then names only the on-axis plasma
statistics (`solprop_native._statistics.PLANE_ONLY`), which the transitional session still finishes
on the host. A group starts through the same shape (SEP 47 §6.7):
`start_group` checks the group (`check_group`) and asks its first member,
`members[0].prepare_group(members, problems, requests)`, for a `PreparedGroup` whose
`start(starts, *, stop)` opens the backend's `GroupSession`, so it names no backend
either. The native backend's (`solprop_native.group.NativePreparedGroup`) opens
`NativeGroupSession` where `solprop_native.group.refusal` admits the group and the
transitional `DriverGroupSession` otherwise, logging the reason at debug level; every
other driver's (`driver_session.DriverPreparedGroup`), the reference's among them,
opens `DriverGroupSession` over the lockstep runner its registry row names. A backend
that runs no group raises `BackendError` from `prepare_group`. The
three keyword arguments are what `stats()` needs and the kernel contract does not carry.

**Spatial statistics measure the medium's flux** ([SEP 34](seps/archive/sep-0034-medium-flux.md)).
The shared spatial statistics, `backends.spatial.collect_spatial`, read the plane's
coefficients through `problem.route.flux.at(z)` (the spatial route's plane measure) (kernel schema 35): energy and the far-field
spectra weight each column and bin by the real admittance `c Re k_z/omega` of that root,
and fluence, intensities and the on-axis spectrum synthesise the closure `G` from the
complex admittance before forming `Re(E conj(G))/2`, signed and not clipped. The widths
refuse a fluence negative beyond its rounding budget. The helper
`backends.spatial.collar_field_fraction` stays an electric-field-norm share and is named
as one; its end-of-plane log is retired, and the boundary monitor's transverse edge
(SEP 40) judges the boundary instead. A backend whose own statistics do not apply this measure refuses the problem.

**The transverse edge is read where the state lives** ([SEP 40](seps/archive/sep-0040-transverse-absorbing-boundaries.md), [SEP 44](seps/archive/sep-0044-native-edge-read.md)).
A backend whose `StepDriver.reads_transverse_edge` is set hands the monitor's
`TransverseEdge` an `_EdgeReading` from `StepDriver._edge_reading` at the edge's reads, at saved planes, and with a
requested collar at every committed step. The reading is numbers and never the field: region sums, each
column's energy and any collar's strip and current. The default forms them with `_HostEdgeReduction` on a host copy,
the reference, which `numpy_ref` uses. `native` forms them on its own state against
the edge's `_EdgeTables`, four regions' weights on the nodes, so it holds no face, mask or judgement, and a
field-free run crosses no state (`tests/backends/test_native_edge_reading.py`). A requested collar's lines are described
once, as `_edge_lines`, for both reductions (`tests/backends/test_native_edge_collar.py`): native takes each strip's padded
transform in batches of planes under `_STRIP_BATCH_BYTES`, on buffers listed in the `transverse_edge` ledger row, and the
current from the onset's three nodes, rectified per row, all on one load of the transverse scratch. Under the driver
the current is still accumulated in Python (under the native session, below, nothing is). Everything else, the weights, the spacing, the ledgers and the judgement, is one
code on every backend. The read spacing is a ceiling on the step, and
a step is not made to land on a read: the edge is read at the last committed position
from which the next step, a replay's scheduled one or else the controller's proposal, would
carry past the next read (`TransverseEdge.due`), so
reads stay at most a spacing apart without shortening the steps between them (landing on
each read cost 10-35 % more steps on the suite's line and planar cases). Every solprop
backend sets it; `numpy_ref` keeps its state on the host and steps from there, under the
driver's ceiling. The native Adams march
takes the next read and the spacing as the call's `read_at` and `read_spacing` bounds:
an adaptive step is capped at the spacing, and the call returns at the committed
position from which the next step would pass the read, so the host reads the state
there. With a collar the host step is one, so the march returns at every committed step.
A saved plane reads the edge unless the step that reached it already did (`TransverseEdge.wants_read`), as a second read at one position records nothing.
A backend that does not set it reports `unwatched` and refuses a requested collar by name.

**Under the native execution session the judgement is native too** ([SEP 47](seps/sep-0047-run-ownership.md) §6.4).
`NativeSession` hands no reading to a Python edge: `transverse::Edge`
(`native/include/solprop_native/transverse.hpp`, a strict translation unit) owns the edge's every decision and record
-- the launch's judgements, the spacing and its ceiling, each read's ledgers and verdict, a collar's current and return,
the resume and the record -- and the session reads the resident state for it (`Resident::edge_reading`, the same
reduction as `_edge_reading`). The march's `read_at`/`read_spacing` bounds return it to the session, not to Python, where
a read falls due, and with a collar at every committed step, so a collar's current costs no return to Python. The edge
follows the temporal monitor at each point of SEP 47 §5.3: launched after the temporal launch, read after a committed
step's check and before its statistics row, and at a plane before its row. A point's several warnings (a launch can
raise five) are each a `ServiceRequired` barrier, the edge holding its place between them (`Edge::proceed`), so a
warning the caller raises leaves the rest of that point undone, as solprop's inline warning does. The scalar decisions
are solprop's exactly on the same readings, the worst face's ties broken on its name as `max` breaks a tuple's and the
reach's tail summed in order over the columns as solprop sorted them; the launch's share in the ramp's band and the
ramp's removal per interval are the native reading, summed in order with the C library's `expm1` where NumPy sums
pairwise with its own, within `(n - 1 + ceil(log2 n))u` of it, plus the two `expm1`s' error for the removal
(`tests/backends/test_transverse_owner.py`). The removal's error accumulates across reads, at most that bound of each
read's term plus a rounding per read. So the two decisions taken on these readings -- the ramp's band at the launch
and the ramp's removal at each read, warning or stop -- may differ from solprop's where a share lies within that bound
of a threshold: a native share a rounding below the stop warns where solprop stops (SEP 47 §4.3, the maintainer's
decision 4). Every other decision is solprop's on the same readings. With the input noise, the launch carries the
draw's readings (`transverse::Noise`): each column's energy, the weighted node energy the noise's allowance takes, the
weighted columns' energy in the ramp's band, a collar's strip of the draw and the launch's columns without it, formed
from the draw once, on the host, by solprop's own reductions (`_transverse.noise_readings`), before the session leases
the state; the noise's share of the ramp's removal per interval is the native reading, as the raw removal is. A noisy
split run's temporal record holds the floor's share of the split's removal, which the native monitor also sums in
order (`watch.hpp`). `solprop_native._transverse` formats the
record and the verdicts with the edge's own texts, which `solprop.transverse_monitor` keeps as module functions.

## 3. Compiled execution: the design `native` runs

This section records how a compiled CPU backend executes the seam, and the algorithms,
gates and bounds of its routes. `native` is the one compiled backend, and everything here
is true of its CPU build unless it says otherwise; §4 records the package around it, its
engine's own choices, device builds and build profiles. `numpy_ref` (§2) stays the
reference for all of it.

### 3.1 Why fused passes, and what the measurements say

At solprop's sizes a stage is tens of microseconds to a few milliseconds. An earlier
prototype study measured that per-call overhead (`np.fft` allocation and plan lookup,
non-jitted JAX dispatch 50–100 µs) and the *number of passes* over the data dominate, not
FFT placement (§6). Luna's own step is instructive: of an isolated RDW step, about 40 % was
stage updates, error norm, windows and output and about 20 % statistics — the non-RHS work
that fusion removes.

So the wins are: fused passes (one traversal of t per column for the whole nonlinear term;
one pass for stage combination + propagator + noise + padding; one pass for crop + prefactor +
windows + back-transform + accumulation + norm), preallocated buffers and plans,
two-stage reductions, and moving the whole inner loop into compiled code: one native call
executes a Runge–Kutta trial (§3.5). On a CPU build's trapezoid column the whole nonlinear
term at one node — the rate lookup, the plasma's two running sums, the loss-current guard,
the Kerr cubic and the Raman drive — is **one traversal of `t`**, the plasma's sums carried
in scalars across the loop (`response_plasma`, `native/src/general.cpp`); the band column
(§3.12) uses complete row passes instead. The solution and the embedded estimate are
running accumulators (ADR-009), which is also why there is no dense output by default. An
isolated step timing is not comparable with a whole-run average, which carries the
once-per-run FFTW planning and the output's planes.

### 3.1a What exists

The adapter `native/src/solprop_native/backend.py` and the engine `native/src/general.cpp`
over the shared driver of `backends/driver.py`. The public default is `"auto"`: the
native backend, with no fallback (§1, ADR-029).
`numpy_ref` stays selectable by name.

**Three response families** (`FieldFamily`, `include/solprop_native/core.hpp`), which is
all `kerr_coefficients` compiles to, each its own compiled column:

| family | grid | `thg` | the column works on | plasma |
|---|---|---|---|---|
| `field_real` | field | yes | the real field | field plasma |
| `field_analytic` | field | no | the analytic signal | field plasma |
| `envelope` | envelope | either | the envelope, with the carrier when the harmonic is radiated | envelope plasma, with streams of its own (§4) |

`field_analytic` is where the fusion pays twice. The reference forms the analytic signal
with a whole complex transform pair inside `kerr_polarisation`, and again inside
`raman_drive`, **once per node chunk**; it is the complex inverse of the same one-sided
spectrum the pad already holds, with the positive bins doubled, so the backend doubles them
in the stage combination and gets the signal from the inverse transform it was running
anyway. The reference keeps the explicit pair, because it is written to be read.

**Agreement is 1e-15, not bitwise, and the reason is measured.** numpy's complex multiply
contracts into a fused multiply-add on some hosts (an Apple-silicon one, for example) —
`fma(ar, br, -(ai·bi))` reproduces `a*b` exactly and the naive form does not — where a
compiled kernel's need not. Writing the product out by hand does not close it; only
emitting the same FMA would, and that would reproduce an accident of one host's SIMD loop
rather than the formula. So one complex product costs 1.42 ε, `np.abs(z)**2` against a sum
of squares costs 1.4 ε, and a transform costs `log2(n)·ε`. Every comparison — the plasma
column's running sums against the independent NumPy scans under either running-integral
rule (§3.12), the statistics column's fraction and peak rate, two runs of `native`, one
thread against many, a same-backend resume — is a rounding bound fixed from its operation
counts (the running sums by the prefix bound of their signed increments;
[kernel-contract.md](kernel-contract.md)), and a resume of an adaptive run is held to its
accuracy envelope ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)). With
the kernels contracting (§3.2), native's products round once where the reference's may
round twice, and that is within the bound.

**Held field current** ([SEP 27](seps/archive/sep-0027-held-plasma-current.md)). The
trapezoid column writes non-plasma polarisation and total physical current separately. The
band column writes current after its second running integral. The engine owns the current,
its two collared products, the projected rows, the spectra and all forward plans at
construction (the `held_current*` ledger rows). Modal routes project before transforming;
radial routes transform and combine at each node, crop, then make one frequency-space
projection. Cartesian and line sources combine before their transverse FFT, and planar
sources before the y transform.

**Mixtures.** The fused column takes a channel loop around the plasma recurrence: one
exposure scan per labelled channel with its own rate table, potential, density and
preionisation in a per-channel, per-node scratch, the free current and the
ionisation-energy loss summed over channels, and the one-channel case specialised at
compile time, which the mixture path matches within the column's rounding bound. Several
Raman components share the node's convolution, their kernels weighted by their amplitude
densities (§4). `backends/populations.py` carries the per-channel populations the five
`species_*` statistics read, and the accepted stash has `1 + 2 n_channels` columns.

**The plasma statistics between planes are read off the step, not found again.** The
response column already forms the ionisation rate and the ionised fraction at every node
and every fine sample; compiled with one more template flag it also writes each node's
**final fraction and largest rate** for every plasma channel as it goes. The stash has
shape `(n_nodes, 1 + 2*n_channels)`, including its validity column, and the statistics row
is then a reduction over the nodes. A run without the channel compiles the two stores away
and pays nothing, and the numbers agree with the ones the per-plane path produces to
rounding.

It is worth it because the per-plane path is not merely slower but *disproportionately* so
inside a step loop: re-synthesising every node into a buffer of megabytes evicts the step's
own working set. Measured on the retired `numba_cpu` backend, on the reduced M2 case at
`period = 1`, the electron density alone cost 14 % of a step through that path and 1.3 %
through the stash, and the whole default subset went from 15.8 % to 2.5 %. The predicate for
reading it (`stash_plasma` in the adapter) is `first_same_as_last` — the seventh stage's
*input* is the accept pass's own result, so the last column evaluation of an accepted step
scanned exactly the state the run adopted — which is also false for the `split_step`
absorber, where the finish changes the state after the stages have run, and beside a
coupled flow. Under the modified noise model the stash is not compiled at all: the last
stage's input is the adopted state *plus* the advanced noise draw, while the statistics are
defined on the committed field the reference scans (reading the stash moved a field run's
peak rate by 2.3e-5 relative, measured on the same retired backend). Anywhere else, and at a
save plane whose field may not be the one the last stage saw, the nodes are synthesised and
scanned in full.

**Cross-backend resume.** A run continued on the other backend is the same discrete march
as the uninterrupted run on either, on a replayed schedule, and agrees with it within the
matched march's rounding budget, which bounds two uninterrupted runs on the two backends as
well (`tests/driver/test_resume.py`). The backend's name is therefore out of the digest and
the execution is recorded beside it instead, in `RunMetadata.backend`, so a run started on
one backend and finished on the other is the same run and the file still says which
started it.

**FSAL** is **taken** on `native`, with the recycled stage kept across save planes
(ADR-039); the `split_step` scheme, whose finish changes the state, gives it up, as does a
coupled flow (§3.9). `numpy_ref` evaluates every stage (§2). The identity, why the hard band
mask and the in-stage absorber preserve it, and the measured saving are
[stepper.md §2](stepper.md).

### 3.2 Kernel rules

- **The floating-point licence** is SEP 25's and ADR-049's (contraction on by default,
  strict protected blocks, reassociation per kernel; umbrella fast-math, `nnan`, `ninf`,
  `nsz`, approximate functions and reciprocals, and flush-to-zero forbidden). On `native`,
  as `native/AGENTS.md` and `native/CMakeLists.txt` set it:
  - **Contraction.** A CPU build compiles with `-ffp-contract=fast` under GCC and
    `-ffp-contract=fast-honor-pragmas` under Clang and `icpx` (plain `fast` makes Clang
    ignore the pragmas), so the compiler may fuse any multiply and add into one rounding.
    Its one hazard is exact cancellation (`x*y - y*x` need not be zero).
    `build_info().optimization_policy` records the flag.
  - **Strict blocks.** A protected function or lambda carries
    `SOLPROP_NATIVE_STRICT_FUNCTION` and starts its body with `SOLPROP_NATIVE_STRICT_BLOCK`
    (`include/solprop_native/fp.hpp`): under Clang and `icpx` a `#pragma clang fp
    contract(off)`, which survives inlining; under GCC, which ignores the pragma,
    `optimize("fp-contract=off")` on the function, so a protected helper is a call there.
    Both settings are lexical, and `Kokkos::complex`'s operators keep their header's
    contraction even when inlined, so a protected block spells its complex arithmetic with
    the `strict_*` helpers. Library functions keep their own error bounds; a protected block
    that uses the vector-maths layer takes its result as an input.
  - **Reassociation** licenses rewrites such as `(a + b) - a → b` that silently remove a
    compensation term, so it is scoped to the loops that need it. No kernel enables it
    today. The finite-value guards that detect divergence are comparisons, which
    contraction cannot change; what keeps them trustworthy is the absence of `nnan`, `ninf`
    and fast-math from every build, and the `__FAST_MATH__` refusal stays.
  - **What is protected.** The stable phase increments, the small-argument phi-functions,
    the ionised fraction (every column's `1 - (1 - p) e^{-X}` and the envelope's
    `p + (1 - p) (-expm1(-X))`, in the kernels and in the statistics), the multistep defects
    and their compensation, the Adams basis rows and the Chebyshev and z-table Clenshaw sums
    (`lagrange_basis.hpp`, `chebyshev.hpp`, `z_table.hpp`), the far-field radius
    (`far_field.hpp`), and in their own translation unit the Adams march's decisions, so
    that they round as solprop's do.
  - **Device builds.** CUDA and HIP builds leave the strict blocks empty and follow their
    compilers' contraction defaults, held to the measured device envelopes (ADR-049,
    amended) until device round-to-nearest helpers land. The far-field radius already has them.

  The evidence that a kernel is safe is its rounding-bound comparison against `numpy_ref`
  on stressing inputs (weak plasma signals, oscillatory currents, tiny phase increments,
  non-finite guards), not an audit. `native/tests/native/test_fp.cpp` checks per build that
  a protected block, and a strict helper inlined into one, rounds a product and a sum
  separately, and that an ordinary one contracts where the build optimises for a target
  with fused multiply-add. A rounding difference may change an accepted step and fork an
  adaptive trajectory; adaptive comparisons are accuracy envelopes for that reason
  ([testing.md](testing.md) section 3).
- Standard C++20 with device-callable per-element bodies and `Kokkos::complex`; explicit
  `LayoutRight` and execution-space instances; named kernels and allocations
  (`solprop_native.*`); top-level kernels dispatched through `launch::parallel_for`,
  `parallel_reduce` and `parallel_scan`. The specialisation is at compile time, by
  template (family, component count, plasma, stash), so a branch a problem cannot take is
  not in the innermost loop.
- Real `exp` × `sincos` for propagator factors, never a complex `exp`: the angle's parts
  are even and odd, so `exp(-phase)` is `e^{-a} (cos b, -sin b)`, never a reciprocal of the
  forward value. On the host these go through the vector-maths layer (§3.6). Transforms are
  none-normalised, and the scaling is explicit and folded into the coefficients the kernels
  already apply.
- Reductions (error norm, statistics) are two-stage: per-column partials, then a sum over
  the columns. Their order is free
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)), and repeats and thread
  counts are tested against the reduction's rounding bound, not for identical bits.
- No allocation, planning, logging, Python call or unplanned fence in a trial step; every
  persistent buffer is allocated at construction and named by `memory_ledger()`. No
  exceptions or I/O in device code; a failure inside a kernel returns as a status the host
  reads (the envelope plasma's quadrature status, §4), and the adapter raises.

### 3.3 Transforms

Every transform on a CPU build is FFTW through Kokkos-FFT, except macOS's short threaded Raman
rows (§4), none-normalised, on a persistent
plan made once per state at construction: the modal pair (complex on the envelope grid, real
on the field grid); the doubled-grid Raman pairs, with their kernel's plan and the plans of
the last, partial chunk at its actual width; the coarse pair of the split-step temporal
absorber (on a free-space envelope, an in-place pair on the state's own rows, with no buffer
of its own) and the forward plan of ADR-025's in-stage absorber term, whose buffers are the
run's own because the stage's finish still reads the polarisation's spectrum when the term
is formed; the held and envelope currents' streams; the general route's family (§3.8); and
the transverse plans of the free and planar kinds. A stage allocates nothing.

**The plan's effort depends on the transform's shape and width, and the rule is one
function** (`fftw_seed_effort`, §4). The measurements behind seeding a row transform at all
were made on the retired `numba_cpu` backend's plans: on the modal pair `FFTW_MEASURE`
executed within noise of `FFTW_ESTIMATE`; on the **doubled** grid the Raman convolution runs
on, `ESTIMATE` was 5.6 times worse, which more than tripled the whole CtC step; and a
`MEASURE` plan made on a busy host was 2.4 times *slower* than `ESTIMATE`, because the search
is timed against the load, so planning belongs on a quiet machine. Why threaded row
transforms stay at `ESTIMATE` is
[benchmarks.md](benchmarks.md#threaded-fft-plan-selection).

Three buffer traps, each of which gives a plausible wrong answer rather than an error: a
complex-to-real plan **destroys its input**, so a zero-padded tail must be rewritten every
stage; the forward transform needs its own output buffer or it overwrites the pad; and
measured planning may write over both its buffers, so plans must be built before any buffer
is filled.

Threads are a property of the plan, chosen at construction: a plan of fewer real-side samples
than `default_threaded_fft_min_samples` runs on one thread (§4), and the others take their
width from the size rules of §3.4, within the process's thread budget (ADR-052: the affinity
mask, the cgroup CPU quota on the process's own path rounded down, and `SOLPROP_THREADS`;
never `os.cpu_count()`). The plans runs measure are kept by default, per machine and
build (ADR-076, §4).

### 3.4 Threading regimes

| regime | when | inside a step | across problems |
|---|---|---|---|
| compact problem | one column on one node, fewer than 65 536 fine samples | kernels on a team of at most eight threads, transforms on that team, products on one BLAS thread | independent scans: Local, or spawned processes through Pool |
| sized problem | the rest | each operation's width chosen from its own element count at construction: stage, response, dense product, FFT, Raman, copy, flow, modal and Adams | one process per problem |
| fixed-schedule batch | compatible independently loaded members | fused accept update over members; remaining work stays member-owned | one batch runner owns the group |

The widths are constants of `include/solprop_native/core.hpp` (`operation_threads`, with
their calibration in
[benchmarks.md](benchmarks.md#kokkos-cpu-operation-size-threading)), and `SOLPROP_NATIVE_*_THREADS`
overrides a pool ([Performance tuning](../guide/performance/tuning.md)). The stage boundary
and the threaded-FFT boundary differ on macOS (§4). Three things motivate the policy. They
say the regime cannot be inherited from the environment, and that a few settings matter more
than the choice of backend:

1. **Idle threads spin briefly within native work** (ADR-071, superseding ADR-056's
   sleeping at once). A step opens tens of short parallel regions, and waking sleeping
   threads cost 16–72 µs a region. Unless the user chose, the import sets
   `GOMP_SPINCOUNT=10000` (GNU's runtime) and `KMP_BLOCKTIME=0` (LLVM's), and the core keeps
   LLVM's threads spinning for 1 ms inside each native call (macOS and Windows); teams wider
   than a chip's fastest cores sleep at once (`solprop_native/_wait.py`, recorded as
   `openmp_wait_policy`). The pools that once competed with spinning threads (Kokkos's
   teams, the threaded FFTs, the BLAS) now share one runtime.
2. **The BLAS is fitted to the pool.** A pthreads OpenBLAS runs a second pool beside Kokkos's
   and is held to one thread; an OpenMP one is capped at the Kokkos thread count, unless the
   user sets the BLAS's own thread variables. NumPy's own BLAS is held to one thread while a
   native backend lives, or to `SOLPROP_BLAS_THREADS` within the budget
   (`solprop_native/_host_blas.py`). A dense product stays serial below about a million
   elements of work, since the GEMM is skinny in the mode count: RDW's synthesis is
   `33 x 4 @ 4 x 8192`, CtC's `41 x 6 @ 6 x 65536`.
3. **The best thread count is not the machine's.** It differs by operation and by problem,
   which is why each width is sized separately. For the scan pattern, k independent
   single-thread runs beat one k-thread run by 8.5x (measured on the retired `numba_cpu`
   backend), so a batch runner should prefer processes to threads; the scan executors do this
   through `Pool` ([scans.md](scans.md)).

The pools, the widths and the wait policy are stored in the output metadata
(`thread_budget`, `execution_sizing`). The settings of the retired backend
(`SOLPROP_NUMBA_THREADS`, `SOLPROP_NUMBA_REGIME`, `SOLPROP_FFTW_THREADS`,
`SOLPROP_TRANSVERSE_FFTW_THREADS`, `SOLPROP_PASS_TIMERS`, `NUMBA_DISABLE_JIT`,
`NUMBA_NUM_THREADS`) are read by no backend; `SOLPROP_THREADS`, `SOLPROP_BLAS_THREADS`,
`SOLPROP_FFTW_PLAN` and `SOLPROP_FFTW_WISDOM` remain. Pass timings are the native engine's
Kokkos regions.

### 3.5 The step driver: Python first, compiled when measured

**One driver, for every backend.** `backends/driver.py` is the Runge-Kutta march and replay,
the controller bookkeeping, the history and the noise models, and every backend inherits it
(§1, point 3, lists the other marches). Per stage a backend asks the translation layer for one bundle of host
scalars and combines the terms it holds over them (§1); the driver applies the boundary
treatments of `_finish` to a step it keeps, so a rejected step does not pay the temporal
absorber's transform pair for a state that is thrown away.

Under Runge–Kutta the driver's loop is the reference, and on `native` one call executes all
seven stages of a trial (§4), its rows formed natively from the uploaded stage tables and its
norm reduced natively ([SEP 45](seps/archive/sep-0045-native-rk-march.md) phase 2). An earlier
measurement on the retired `numba_cpu` backend found the loop's Python cheap (about 50 calls a
step cost 1.69 % of a step at 1024 bins on the GNLSE soliton), but the host's whole per-step
work, the stage scalars and the packing with the loop, came to about 200 µs a step on an Intel
Xeon Gold 6240: a fifth of a small-grid fibre run (SEP 45 §3.4). The march is compiled where
the kernel contract specifies it in full, with solprop's implementation as the reference:
`native` steps `Multistep()` itself (ADR-060, §8), the Runge–Kutta march (ADR-069),
and makes one native call a step under the low-storage march (§9).

### 3.6 Propagator factors

`ΔΦ_i` for each distinct stage fraction is a weighted sum of fixed terms (a Clenshaw pass on
the z table); its exponential is one real `exp` and one `sincos` per element. Tabulating the
six factors per step costs up to 6 F of memory and one pass each; recomputing them costs the
transcendental twice per stage. An earlier prototype study found the table wins on a
transcendental-bound host and loses on a bandwidth-bound GPU, and in a prototype with
separate passes tabulating won — 18 % of a whole RDW step and 4 % on CtC without Raman, a
wash with the Raman convolution.

The native backend's host kernels form these exponentials, the exponential Adams
method's flows, phi-function exponentials and compensation factors, and the plasma's
rates and fractions in passes over blocks of elements through one vector-maths layer
(`native/include/solprop_native/vmath.hpp`): SLEEF's 1.0 ulp vector functions on x86-64
builds, the system libm, with each element's own calls, on Arm and on devices
([ADR-059](decisions.md)).

**In a fused backend the answer is a property of the fusion.** Both signs are consumed by
the stage that formed them — the forward in pass 1 and the inverse in pass 7 — so
recomputing evaluates `ΔΦ` exactly as often as tabulating does and the table buys residency
and nothing else (a step within 2 %, measured on the retired `numba_cpu` backend). The flag's
default is therefore `"auto"` and the choice is the backend's (`StepDriver.propagator_default`):
`numpy_ref` resolves it to `tabulate`, because it forms two complex exponentials per stage and
the table saves five of them, and implements both paths; `native` resolves it to `recompute`,
because both signs come from one `exp` and a `sincos` in the stage that consumes them
(`stage_phase` holds the stage's phase once, and a lossless stage whose phase weights repeat
the last one's reuses its inverse). The resolved value is written into the run's `backend`
metadata group. `native` has no Runge–Kutta table: its `propagators` ledger row is the whole
step's propagator, and an explicit `"tabulate"` under Runge–Kutta changes nothing, so the
`backend` group records `recompute` while the run's configuration keeps what was asked. Only
the low-storage march holds flows (§8).

### 3.7 Coverage and testing lanes

- Default lane: on the wheel `native-wheel` builds and caches, on macOS, Windows x64 and Windows
  on Arm, and on Linux through the coverage lane (`tests/selected_backend.py` takes what
  `backend="auto"` selects unless `SOLPROP_TEST_BACKEND` names another); tests assert agreement with `numpy_ref` to tolerances derived from summation bounds (Higham:
  `n · ε` on the reduction length, then the measured margin), repeat runs and thread counts on
  a fixed schedule within the step's rounding bound, use-after-close errors, the memory ledger,
  and end-to-end file equivalence through both output handlers. `native/scripts/check.sh`
  (`poe check-native`) is the package's own gate.
- Coverage lane: the native wheel with `SOLPROP_TEST_BACKEND=native`, measuring solprop's
  Python.
- `no-native` lane: no native package, a few default-lane modules on `numpy_ref`, which checks
  `auto`'s refusal and a real run on `numpy_ref` by name.
- The `docs`, `slow` and `oldest` lanes install the CI-built wheel.
- Validation ladder per backend: one linear application, one nonlinear evaluation, one step,
  eight steps, each statistic, a full save interval, a full run.

### 3.8 The general modal route

`native` executes the general modal route — `transform.modal_tables`, `linear.modal_metric`
and `responses.regional_responses` — in the combinations `numpy_ref` admits, and refuses the
rest with the reference's own words: its admission runs the shared list
`solprop.backends.numpy_ref.admission.refuse_general_route` under its own name, so a new
refusal there is a refusal here. It adds one limit of its own: a delayed response must be the
six channels of the silica tensor Raman model on three components (§4,
`_refuse_native_route`). `linear.coupled_flow` is §3.9. The route is a **second execution
path** (`src/general_modal.cpp`, `GeneralModalSource`) beside the fused pipeline and not an
extension of it, for the reason §3.1a's three families make plain: those are one real
transverse matrix in the time domain, and this route's synthesis is a complex
frequency-dependent map of three Cartesian components.

- **Maps.** Derived by the reference's own functions in
  `solprop.backends.numpy_ref.general_modal`, so the two backends cannot read a record
  differently: the per-node `chi3 T`, the form from `model_id`, and the delayed response's
  owners, tensors and weighted kernel spectra.
- **Plans.** Per node chunk, a complex inverse of the doubled one-sided spectrum (the analytic
  signal) and a real forward of the polarisation; on the host the real form reads its field
  from a complex-to-real inverse with the doubled bins halved, half the inverse transform's
  work. With a delayed response, a real pair on the doubled axis over the six stored pair
  histories. A partial last chunk has its own plans on views of the same buffers. Built once;
  nothing is planned or allocated inside a stage.
- **Kernels.** Team kernels with every sum in a fixed order, held to their rounding bounds
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): the synthesis over
  channels, the real cubic's symmetric contraction (its coefficients formed once per node,
  then ten monomials a sample), the analytic form's dense contraction, the delayed contraction,
  and the projection over nodes and components into the source. Each is diagonal in the bin or
  sample (a matrix-vector product per bin), not a GEMM, so the loops run the bin innermost over
  contiguous rows, and the projection reduces per output bin, so the result does not depend on
  the thread count on the host (tested against the stage's rounding bound; a device's summation
  order is §4's). The pad scale and the analytic doubling go on the channel spectra, and the
  inverse transform's `1/n_to` on the transform.
- **The stage** combines the state onto the coarse grid once, evaluates the source, and
  finishes through the same crop, prefactor, window, frame and accumulation kernel as every
  other route; a guide with no responding region skips the contraction and its transform, as
  the reference does, and ADR-025's in-stage absorber takes the shared inverse transform.
- **Moving maps.** One kernel, `GeneralModalSource::combine`, forms each map's stage value from
  its coefficients into a stage buffer allocated once, each element summing its section's
  `n_deg` terms in order and skipping the zero-weight terms of every other section
  (thread-invariant; within the rounding of `n_deg` terms of the reference's BLAS product). The
  combination adds `2 u n_deg (3 kappa_S + kappa_Q)` to the source bound. A constant guide never
  combines and is the constant route exactly.
- **Parity is a rounding statement** against `numpy_ref`, with the bound stated first
  (`tests/general_parity.py`, `native/tests/test_general_modal.py`,
  `tests/driver/test_native_moving_guide.py`). Not bitwise: FFTW is not pocketfft, and numpy's
  complex product is an FMA on some hosts (§3.1a).
- **Statistics** are solprop's `collect` on the observed channel field with the basis metric, as
  on the reference, so on one state the two agree to the statistic's rounding bound. The
  **lockstep batch** (`NativeBatchBackend`) takes general members unchanged, within the
  matched-discrete rounding bound of what they run alone.
- **Ledger**: the `general_*` rows, named and zero on every other route; the maps and the
  record's resident E/H reference are counted as the reference counts them. Moving maps add the
  coefficients (`general_synthesis`, `general_projection`), the two maps a stage evaluates
  (`general_stage_maps`) and the weights (`general_map_weights`).

### 3.9 The coupled flow

`native` executes `linear.coupled_flow` in the combinations `numpy_ref` admits but CF4: the
single constant-Schur section (the scalar wall, the TE/TM wall, the split block on the general
route) and the varying `"frozen_residual"` flow with its explicit residual and the taper
connection. A `"cf4"` section is refused by name at the backend and at the capillary entry
point, and so is a flow beside a Cartesian free-space problem, which no translation builds
(`require_coupled_flow`). Its admission is the shared list under its own name.

- **Host-side, shared with the reference.** The Schur factors of every section's frozen block,
  once at construction, as the reference forms them, and the eigenbasis of each triangular
  factor (`solprop.translate.coupled.schur_eigenbasis`). Both cross to the engine once.
- **The matrix functions, native.** `CoupledFlowAction::form_actions` forms every stage
  fraction's forward and inverse action of the step's section into the held buffers whenever
  the section or the length changes (the reference's memo key). A bin whose eigenbasis
  `W = Z V`, `Y = V^-1 Z^H` of `T = V Lambda V^-1` is well conditioned takes
  `W exp(+-l Lambda) Y`: `n` real exponentials, cosines and sines shared by the pair, and one
  `n³` product per action, where the reference sums a sixteen-term series of `n x n` products
  and back-transforms. A bin whose basis is ill-conditioned — a nearly degenerate pair with a
  nonnormal coupling, where `kappa = || |W| || || |Y| ||` grows as the inverse of the
  separation, or an exactly degenerate one, where `V` does not exist — takes the reference's
  scaling, series and squaring, with its own squaring count. The gate is
  `kappa <= eigenbasis_limit(n)`, the `kappa` at which the eigenbasis route's formation bound
  `kappa (n + 11 + 2a) u` meets the reference algorithm's smallest,
  `16 u + 2 sqrt(2) n (n + 2) u` (2.04 at one channel, 4.17 at three), the point where the two
  routes' vanishing-step rounding terms meet. It is a heuristic gate, not a bound comparison at
  every step length; measured, the eigenbasis route is at least as accurate as the reference's
  on every adversarial case tried (nearly defective, strongly dissipative and nonnormal blocks).
  Why not Parlett's recurrence or divided differences: they are exact for separated eigenvalues
  too, but re-run the `O(n³)` recurrence for every length and direction, and near a coincidence
  need the blocked Schur–Parlett's reordering; the eigenbasis moves all `O(n³)` work to
  construction and shares one basis between the six fractions and both directions, and the gate
  sends exactly the bins a plain recurrence would lose to the scaling-and-squaring fallback.
  Each spectral norm of the whole step's pair is measured by cyclic Jacobi rotations on the Gram
  matrix, plus the Frobenius norm of what is left off its diagonal (Weyl), an upper bound within
  a few `n u` of the singular value; the adapter's refusal is the reference's, in its words. The
  squaring limit is checked on the section's largest row-sum norm times the length before either
  route runs. Nothing is allocated.
- **The passes**: the stage combination mixed by the forward action (applied to the state and
  the combination separately, as the reference does), the pad onto the transform grid, the
  residual's combination over the stage weights minus the frozen block, the finish (the crop,
  prefactor, window, frame and in-stage absorber of the diagonal route plus
  `exp(−dPhi) R lab`), the return by the inverse action with both accumulations, and the accept
  pass through the whole step's action with the error norm's partials. Each product is a
  fixed-order loop over the channel axis inside one output element, so the result does not
  depend on the thread count (`tests/driver/test_native_coupled_flow.py`). The flow's width is
  sized from `flow_thread_min_elements` (32 768, active bins times the cube of the channels). A
  warm trial at a held length allocates nothing field-sized the diagonal route does not.
- **The exponential Adams formulation** beside the flow is resident on a CPU build at any width
  and on either route (SEP 16 phase 6, SEP 47 phase 4): a bin on the eigenbasis route takes its
  flow and moments `k! phi_(k+1)` as `W f(Delta Phi + h lambda) Y`, and a bin off it reads them,
  as the reference does, off the first block row of the exponential of the augmented matrix of
  `X = Delta Phi I + h T` (`include/solprop_native/flow_functions.hpp`): the reference's
  scaling, sixteen-term series and squaring, where the reference takes SciPy's Padé `expm`, a
  rounding-level difference (SEP 25), with the shift's exponential kept apart in the carry of a
  history sample. Every block row past the first is a scaled shift, so the passes form the first
  row and one scalar per diagonal of the rest, upper triangular, which is the full augmented
  algorithm's arithmetic to the sign of a zero, in one scratch row per active bin (a trial reads
  one section). A trial whose moments' exponent, or a history sample's carry, would need more
  than 30 squarings, or whose exponent there is not finite, is refused by name before it moves
  the state. Past 16 channels the per-bin host
  passes hold a bin's channel vectors in a row of their own (`adams_channels_`) rather than on
  the stack. A device build keeps the 16 channels and the eigenbasis route on every bin, and
  steps anything else on solprop's host engine, as before. The bounds are
  `flow_parity.adams_formation_bound` (a logarithmic-norm growth and a weighted augmented basis,
  which cancels most of the moments' factorial) and `tests/driver/test_native_flow_session.py`,
  which also holds each trial's assembled flow and moments to `multistep.coupled_moments` at
  1e-12 (`adams_block_to_host`, a test observation).
- **Parity** is a rounding statement with the bound stated first: `tests/flow_parity.py` for the
  actions and the steps, and `parity_bound` in `tests/driver/test_native_coupled_flow.py` for ten
  fixed steps on each case.
- **FSAL** stays off beside a flow, as on the reference ([stepper.md §2](stepper.md));
  the lockstep batch refuses a member carrying the leaf, as the reference's does.
- **Ledger**: the `flow_*` rows, named and zero on every other route (§4).

### 3.10 The hybrid planar kind

`native` executes the `"slab"` spatial kind — slab modes in x on the transform's own real
synthesis and projection, an FFT in y, columns `(p, k_y)` with the channel fastest — in every
combination `numpy_ref` does: field (real and analytic families) and envelope, Kerr, Raman, field
and envelope plasma, the in-stage and split-step absorbers with the y collar's in-stage rate, a
modified noise draw, and every statistic.

- **The stage** is the reference's `_nonlinear_slab` on persistent buffers. The stage input is
  combined onto the columns with the scale `1/n_y` and taken to its `(p, y)` rows by one
  unnormalised in-place transform over the y axis. The rows go to time a **block of whole y rows**
  at a time, as many as the node chunk holds (the reference's block), so `n_p n_y` rows are
  transformed each way, never `n_x n_y` nodes. The x synthesis and projection are team kernels
  (`slab_contract`, `include/solprop_native/slab.hpp`) with each sum in increasing order and the
  vector loop over contiguous samples, and SEP 6's dispersive response shares them.
- **The operator** is evaluated per element and never tabulated under Runge–Kutta (the contract
  refuses `propagator_mode = "tabulate"` for the kind outside the low-storage march). It subtracts
  the column's channel's `kappa_sq` **last**, `q = ((k² − k_y²) + (k²_low − k²_y,low)) − kappa_p²`,
  the order of `separable_values`, and shares the root, evanescent clamp, frame and loss with the
  free-space element, so free space's arithmetic is untouched by the kind. The propagator, the
  noise phase and the stage's prefactor all use it.
- **The statistics** of the field (energy, peak and on-axis intensity, the widths in y) are formed
  natively from the flux measure (§3.13). The plasma statistics take the stage's route and read
  the stash between planes. No quadrature alarm is scored, as on the reference.
- **Parity** is a rounding statement with the bounds of `tests/backends/test_planar_native.py` and
  `tests/backends/planar_cases.py` stated first. What the ladder adds to the Cartesian one is the
  operator: each library's complex square root is within one ulp of the principal root, so
  `Re k_z` differs by at most `2 eps k_max` and the phase of a run of length `L` by
  `2 eps k_max L`. Repeats, and one against four threads, agree within the same bounds. §4 has the
  measured agreement.
- **Ledger**: `transverse_scratch` is the column volume; the temporal plans hold a block's
  `(p, y)` rows and the node buffers its `(x, y)` nodes. A warm stage allocates nothing field-sized.

### 3.11 The dispersive electronic response

`native` executes [SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)'s
`responses.dispersive_kerr` on every route `numpy_ref` runs it -- the fused node path
(mode-averaged, multimode and radial), Cartesian free space and the line, the hybrid
planar route and the general modal route -- on field and envelope grids, with both
evaluators, both sectors, one, two or three components, mixtures, beside Raman, plasma
and envelope plasma (`src/dispersive.cpp`). The operator is the reference's
`G[w (G E)^3]`: the temporal collar on the products, before `Π₊` and the output multipliers.
Everywhere the multipliers move off the nodes, to wherever the route already holds a spectrum:

- **The fused node path** (`DispersiveResponse`). The legacy transform synthesises and projects
  with real matrices that do not depend on frequency, and the collar is the same at every node,
  so a spectral multiplier and the collar commute with both: `P G[w (G E)^3] = G[w P (G E)^3]`.
  The input multipliers act on the padded modal spectrum the stage combination already formed (no
  forward transform of the node field), one batched inverse transform gives every filtered modal
  field, the real synthesis GEMM puts them at a chunk's nodes, a functor forms the products in
  the leaf's sector and multiplies them by the collar, and the projection GEMMs take each product
  to the columns of both rules. One batched forward transform takes them to frequency, and their
  output multipliers and the channels' `ε₀ γ₀ ρ` land them in the stage's own polarisation spectrum
  after the stage's forward transform: the response has no inverse transform of its own.
- **Cartesian free space and the line** (`SeparableDispersive`). Each node chunk's columns are its
  nodes and go to time and back on their own, so the response runs per chunk: the input multipliers
  on the chunk's packed spectrum, the inverse transform writing each filtered field straight into
  the block its product takes, the products in place, the collar, one forward transform and the
  spectra added to the chunk's.
- **The hybrid planar route** (`SeparableDispersive`), likewise per block of `(p, y)` rows: the
  filtered rows are synthesised at the block's nodes by the slab rule in x, multiplied, and
  projected back to rows, as the bare field is.
- **The general modal route** (`GeneralDispersive`, held by `GeneralModalSource`). Its synthesis
  depends on frequency, so the input multipliers act *in synthesis*: on each node chunk's
  synthesised spectrum, one batched inverse transform giving the filtered analytic signals. The
  products are formed on the real field (the full sector, `kerr_form="real"`) or on the analytic
  signals (the conserving one) through the three-component isotropic contraction, at the nodes of
  each channel's region and zero elsewhere, and their spectra are added to the chunk's polarisation
  spectrum before the projection. The route has no collar. A guide whose only nonlinear region is a
  gas carries no regional tensor; its identically zero contraction and transform are skipped and
  the dispersive spectrum is written.
- **Alone** (no constant Kerr term, Raman, plasma or envelope plasma beside it) the response column
  and the polarisation's projection do not run, and the stage's forward transform is replaced by
  the response writing the polarisation spectrum. A symmetric response of one species whose product
  has the polarisation's dtype goes further: its product is the stage's own projected polarisation,
  the stage's forward transform is the product's and `g` finishes it in the stage's spectrum -- the
  constant cubic's transforms exactly -- and where nothing reads the bare field in time afterwards
  (no in-stage absorber on the fused path; always on the Cartesian, line and planar routes, whose
  absorber is a pass of its own) it filters the stage's spectrum in place and holds no field-sized
  buffer of its own, only its per-bin multiplier tables. In free space the engine's stage buffers
  then stop aliasing, so the run still allocates them. A
  symmetric mixture, one product under its mixture filter, takes the same path. Beside other
  responses the bare field stays theirs and the response holds its own buffers.
- **Mixtures.** The symmetric evaluator forms one product per mixture -- per region on the general
  route -- under the filter of the mixture's own Kerr diagonal,
  `g_mix = [Σ_s f_s g_s⁴ / Σ_s f_s]^(1/4)` with `f_s = ε₀ γ₀,s ρ_s`
  ([SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)). `DispersiveWeights` rebuilds a
  mixture's filter (`O(n_omega × n_species)` and two square roots per bin) only when the densities
  differ from the last rebuild's: a fill of constant composition builds it once, a composition
  varying along z at every stage. Its arithmetic is `mixture_filter`'s, so the two backends' filters
  agree to rounding. The polynomial groups species with equal leg bounds, which share their
  derivative fields `D E` and `D² E` and add their coefficient tables density-weighted at every
  stage: an exact rewriting of the per-species sum, so its transforms grow with the number of
  distinct leg bounds (krypton and xenon one; air's three species three). On the general route a
  group is also one region's, and each region has its own cube.
- **The polynomial reads the bare analytic signal**, which keeps the zero-frequency bin once where
  `Π₊` excludes it; the state carries some (the split-step absorber's product in time puts it
  there), so its constant is subtracted where the products read it.
- **The embedded rule's products** are projected at every stage and windowed, transformed and
  combined only when a step is accepted, from the last stage's buffers, before the alarm uses the
  embedded spectrum (`finish_coarse`) -- as the envelope currents are.
- **Node budget.** The backend's node chunk counts the response's buffers: one node buffer for
  a symmetric response beside others, none alone, two for the polynomial; a Cartesian or planar
  column adds its products and their spectra (`backends/node_chunk.py`, `_dispersive_chunk`). The
  general modal route's chunk counts two buffers per node and component for each filtered field and
  each product of its workspace (`numpy_ref.general_modal.general_dispersive_buffers`,
  `general_node_chunk`).
- **Parity** (`tests/backends/test_native_dispersive_kerr.py`,
  `tests/backends/test_native_dispersive_routes.py`): one evaluation, one step and ten replayed
  steps against `numpy_ref`, under bounds derived in `tests/backends/dispersive_cases.py` from the
  transforms and node reductions both backends perform. Beside the plasma column the bound adds
  what the column already disagrees by. Repeats, and one against four threads, agree within the
  same bounds.
- **Ledger**: the `dispersive_*` rows, named at zero on every problem; borrowed buffers are the
  backend's own rows.
- **Cost** against the constant cubic on the same problem, same-process medians:
  [benchmarks.md](benchmarks.md#dispersive-electronic-response).

### 3.12 The plasma column's band rule

[SEP 18](seps/archive/sep-0018-band-exact-plasma-integrals.md)'s band rule filters each of the
column's population and free-current integrands with the problem's $2K+1$ taps before its running
sum, reading $K$ samples ahead, so the fused one-sample traversal cannot carry it.

- **The team column** (`GeneralWorkspace::team_band_plasma_column`, described in
  `native/include/solprop_native/plasma.hpp`). The plasma runs in whole-row passes per node: the
  rate; the filtered rate (`team_band_filter`), then the exposure's scan (`team_trapezoid_window`),
  then pointwise the fraction, the loss guard and the weight $f\,q/m$; per component the
  free-current integrand, its filter, then its scan with the current formed from the prefix. The
  held source consumes that current directly through its two collared spectra. Each filter sums in
  the reference's order, each pair added before it is weighted. The $K$ samples at each end close
  the window by point reflection (`band_filter_at`). The loss current keeps the raw rate.
- **Teams and scratch.** The host's teams split the threads between the chunk's nodes
  (`band_team_policy`): a chunk of fewer nodes than threads gives each node a team of several threads
  over chunks of t, and a chunk of many nodes one thread per node, whose scans are then serial. Each
  team keeps its own rows, so they are sized by the concurrent teams, not by the chunk. Several
  channels add their weights in channel order. The scan associates as the team splits the row, and
  the parity tests carry its prefix bound.
- **After it.** The physical current goes into its own complete output rows. Kerr, the Raman drive
  and their collar follow in a separate pointwise pass, and the trapezoid column writes the same two
  outputs. Persistent current streams then form `gJ` and `g'J`.
- **Shortcuts.** A stencil of exact zeros -- before an integrand first leaves zero and after it last
  does -- filters to exactly +0.0, the reference's value, so a host team skips the work outside each
  row's live window (`band_support`): the prefix keeps its value and an unchanged exposure its
  fraction. The fraction is reused the same way on the trapezoid column.
- **Statistics.** `team_band_statistics` runs the first stage alone, per channel, and writes the
  maxima and the paid work's sum, the run diagnostics, over the same rows. The trapezoid's run
  diagnostics come from it too.

**Verification.**
- **Against the reference.** The column agrees with `plasma_scan` and `plasma_current` within an
  absolute budget carried through the whole chain
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)): the stencil's $\gamma_{2K+2}$
  bound, the prefix bound of each scan's signed increments, and the fraction, guard, currents and
  density between them (`native/tests/plasma_bounds.py`). The stress set
  (`native/tests/test_band_stress.py`) is a preionised field whose rate is identically zero, a field
  just above threshold, pulses cut by either end of the window, a mixture there, and a two-component
  field on the radial route.
- **Routes.** Mode-averaged, analytic, multimode, mixture, radial, Cartesian, line and planar runs
  compare one, eight and ten fixed steps under both rules. Their fields, plasma statistics and
  diagnostics use the equivalence ladder's bound with the sums widened by $2K+1$
  (`tests/backends/test_band_plasma_parity.py`).
- **Threads.** One and many threads agree within the same bound
  (`native/tests/test_band_plasma.py`).

The band rule's cost against the trapezoid's is in
[benchmarks.md](benchmarks.md#band-rule-plasma-integration).

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
`on_axis_spectrum` are one native pass over the stored state on the Cartesian and line kinds
(`GeneralState::aperture_statistics`, `native/include/solprop_native/far_field.hpp`,
`FarFieldPass`), in place of the shared host pass `spatial.aperture_statistics` that `numpy_ref`
keeps as the oracle. The pass has its own flag mask beside `spatial_statistics` (values from 32:
the six names and `on_axis_spectrum`, the `W = 1` slot, which on the Cartesian and line kinds
replaces `spatial_statistics`' serial `axis_spectrum` loop; the radial and planar kinds keep it).

It runs over contiguous **bands** of columns -- whole rows on a plane, blocks of `LINE_BLOCK = 64`
columns on a line -- each writing its row of partials, shape `(n_bands, n_slots, n_omega)`. The
band count is `min(n_units, FAR_FIELD_BAND_BUDGET // (8 n_slots n_omega))` with a 128 MiB budget
(`backends.spatial.far_field_bands`): fixed by the shapes, **never by the thread count**, so the
order of every sum is the same on any pool. A team per band has threads that share its frequencies,
each (band, bin) owned by one thread that walks the band's columns in ascending order (on the host
a team is one thread, so frequency is innermost); then the bands are added in ascending order and
the constant factors applied natively (`epsilon_0 c/2`, the spectral-density factor, the energy
weights of `absorbed_power` and each amplitude's pairing `Re(A conj(G))` with its magnetic closure),
so only the finished statistics cross to the host, `n_omega` values per row. The two backends agree
within the rounding bound of `tests/backends/test_native_far_field.py`.

Every quantity is measured by the medium's axial flux (SEP 34): each squared modulus
is weighted by the real admittance `c Re k_z / omega` of the operator's own root, and
each amplitude is accumulated twice, as `sum alpha W E` and as its closure
`sum alpha W conj(eta) E` (the internal spectrum takes the conjugate admittance). The
admittance is formed per element, from the root the operator uses, never stored as a
column-by-frequency table. The slots (`backends.spatial.far_field_layout`) are the
present ones of `[beam, absorbed x 5, regions..., per aperture (collected, Re and Im of
the on-axis amplitude per polarisation, then of its closure), on-axis amplitude and
closure]`. The last is the `W = 1` slot: `on_axis_spectrum` is formed there, which holds SEP 23's
identity between the two to rounding by construction. `spatial_statistics` reads `on_axis_intensity`'s
spectra from that slot too (`FarFieldPass::axis_spectrum`) and transforms them, on the Cartesian and
line kinds: a sum per bin over every column would otherwise be a thread's serial walk across a whole
state on a device.

The per-element parts mirror the reference's: the radius of `mask_plane_radius` as a protected
function (strict on the host; round-to-nearest intrinsics in a device build), so a hard edge
decides as the reference decides; the weight of `_aperture_weight` (the profile codes, the raised
cosine's early-out, and the exact zeros at `omega <= 0` and beyond the paraxial disc); the half-open
test on the negated stored axes in `aperture_statistics`; and `Re k_z`, the propagating test and the
evanescent clamp from the operator's own `longitudinal_root` evaluation, recomputed per element and
never stored. Two exact shortcuts keep it within a few per cent of a step: a `tanh` weight beyond
`|t| = 22` is exactly 0 or 1 (`tanh` lies within 1.6e-19 of `±1` there, so its correctly rounded
value is `±1`), so the transcendental is skipped, and apertures that share a distance share the
image's distance from the origin (and the radius too when they share a centre). The aperture loop
and the shared radius are gated on the layout's aperture slot: apertures declared in the spec but no
aperture statistic asked for (a periodic `beam_spectrum` sample) evaluate no weight.

The native `Problem` carries the stored axes `spatial_k_x` and `spatial_k_y` and the slot count of
the problem's own request (`spatial_far_field_slots`), which sizes the band partials at construction
(`far_field_partials` in the ledger). The call carries the tables, `(n_apertures, 11 + n_omega)` rows
of the host-formed constants and scale row and `(n_regions, 4)` boxes, which the native
`far_field_tables` row holds with the axes and the packed output. A build declares the names it
forms in `NativeBackend.far_field_statistics`; solprop's registry admits them by it and refuses them
by name on an older build, before it is built (`registry.statistics_refusal`). The device path
compiles but is not qualified: its partials live in global memory, not in registers. Measured cost:
about 2.3–2.6 % of one Runge–Kutta step at the sizes of
[benchmarks.md](benchmarks.md#far-field-statistics-pass), paid at save planes and periodic rows only.

**The plane measure on `native`** (SEP 34, kernel schema 35). `native` measures every spatial field
statistic by the medium's axial flux, as `collect_spatial` defines it, and reads the admittance
`c k_z / omega` from the root the operator uses, per element, with no column-by-frequency table. A
Cartesian, line or planar run's plane measure is its own separable operator (the adapter refuses
another); a radial run packs the entrance's material and transverse squares (`flux_k_squared`,
`flux_kperp_sq` and their residuals) and, for a medium that varies along z, uploads the material
square as a `"flux"` z table of `n_omega` values, which `spatial_statistics(flags, factor, z)`
evaluates at the statistic's plane. Energy weighs each column and bin by `Re eta`; the on-axis
spectrum pairs the coherent axis amplitude with its closure; the on-axis intensity, peak intensity
and fluence synthesise the closure `G` beside `E` -- on the radial kind through the dense synthesis
of a closure time volume (`statistics_weighted_time`), on the separable kinds through a second
transverse volume (`flux_closure_transverse`, one state's bytes, the estimate's
`flux_closure_memory`) and, for the intensities, the chunk's closure time rows. The largest magnitude
over the nodes fixes the width refusal's normwise budget (`backends.spatial.fluence_budget`). The
closure buffers are allocated at construction when the problem's own request has a local statistic
and, outside any step, when a later request first needs one.

**A separable fluence takes no time transform.** The reference sums `Re(E conj(G))` over `n_t`
samples; by Parseval's theorem that is a sum over the bins the transverse volumes already hold,
weighted one or four as the energy is (`separable_fluence`). Its magnitude, the reference's
`dt sum_t |E| |G|`, would need the samples, so the budget's scale is the Cauchy-Schwarz bound on it,
`dt sqrt(sum_t |E|^2 sum_t |G|^2)`, formed from the same bins. That is never below the reference's,
and equal where `G` is proportional to `E` at the node, as it is across a beam whose admittance varies
little over its band. Only a negative value between the two budgets would be refused by the reference
and read as rounding here.
The axis's fluence, which only a radial width reads, isn't formed.

**A Cartesian plane's widths reduce where the field lives** (`plane_fluence_moments`,
`plane_fluence_trace`). The reference forms them on the host from every node's fluence
(`solprop.spatial.cartesian_fluence_widths`): a centroid, a second moment, and an azimuthal average
of the bilinear interpolant on a few thousand angles per ring, about 3.7 million samples at
768 x 768. Native forms each as a kernel over the fluence it holds, so that the moments and one value
per ring cross, and solprop keeps the rule: `solprop.spatial._fluence_rings` fixes the rings,
and its angles, which depend on the axes alone, are formed once at lowering
(`solprop.spatial._ring_angles`).

**The widths are finished natively** ([SEP 47](seps/sep-0047-run-ownership.md) §6.4).
`beam_radius` and `fwhm_r` are statistics of the native plan (`Stat::beam_radius`,
`Stat::fwhm_r`), so the native session and `NativeBackend.stats` take them with no Python finishing,
at planes (neither is periodic-capable). The plan runs the same fluence passes the host finishing
read -- flag 16's block on a radial, line or planar profile, flag 32's moments and the trace on a
Cartesian plane -- and `session_widths.cpp`, a strict translation unit, finishes them as
`backends.spatial.signed_fluence_widths` and `plane_fluence_widths` do, through
`spatial.radial_fluence_widths`, `cartesian_fluence_widths` and `_line_fluence_widths`: the
refusal and its budget, the clamp, the slab's sum over x, the centroid, the rings and the threshold
crossings. NumPy's pairwise summation and `numpy.interp` are reproduced, so a radial, line or planar
width is the host finishing's to the bit on the same passes; a plane's extent is the platform's
`std::hypot`, which may differ from Python's `math.hypot` in its last bit (the session tests hold
it to 16 units of rounding). Refusals are the reference's exceptions and messages
(`_statistics.translated`): a negative lobe the `BackendError`, an invalid profile the `GridError`.

## 4. `native`: the compiled CPU and GPU backend, a second distribution

`solprop-native` is selected lazily as `backend="native"`. It lives in `native/`, outside
`src/solprop`, and is the repository's second distribution: one checkout, one uv workspace
and lockfile, one version from the repository's tags, and a required dependency of solprop,
pinned to that version in the built metadata. A plain `uv sync` builds it editable into the
development environment; `uv run poe check-native` is its gate. No other backend runs on a
GPU: those runs use the native backend's CUDA or HIP build, and the build determines whether the
backend runs on the CPU (OpenMP) or a device. How to build it is
[Building the native backend](../guide/performance/building-native.md); the native rules are in
`native/AGENTS.md`.

The backend depends on solprop's seam: the `PropagationProblem`, the protected surface of `StepDriver`
and the stage bundles, all frozen by `tests/backends/test_driver.py`. A change to any of them
rewrites those tables in the same commit and adds a release-note fragment. The two packages are
released together, and solprop's metadata pins this one to its own version. `check_build` looks
the driver members, bundle fields and contract tuples the adapter relies on up by name. In an editable
checkout it also recomputes the digest of `native/CMakeLists.txt` and the C++ under `native/src`
and `native/include`, once per process, and refuses an extension built from anything else
(`BuildInfo.source_digest`, ADR-068), which catches a C++-only change too. It records the repository
version it was built at without comparing it, so an editable build stays valid across solprop
commits that leave `native/` alone. Solprop knows only the registry
row and imports the package only when it is selected.

**Every backend refuses what it cannot execute.** The general modal route's optional
leaves (the coupled flow, the moving guide and the others) are part of the contract whether
or not a backend executes them, and a backend that
accepted a problem carrying one it ignores would silently run a different model. A backend
therefore calls
{py:func}`~solprop.backends.driver.refuse_general_leaves` before it runs, naming in
`admits` the leaves it executes, as `numpy_ref` does, or makes the same
refusals itself, as `native` does in its own adapter. A third-party backend owes the same. A change that
touches `native/` is complete only when the gates (`poe check-all`) and solprop's lanes
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
Kokkos-FFT plans at `FFTW_ESTIMATE` and consumes measured wisdom in the linked FFTW
library, which the workspace seeds before it makes its first plan: a plan made before
its wisdom stays at estimate for the run. One policy (`fftw_seed_effort`,
`include/solprop_native/fft_plan.hpp`) decides every seed by the transform's shape and
width. Unset, row transforms (temporal, modal, Raman and plasma) are measured on one
thread and left at estimate when threaded, because measured wisdom can select slower
threaded plans; free space's transverse planes are measured at any width. An explicit
`SOLPROP_FFTW_PLAN` applies to every transform at every width: `estimate` keeps cheap
planning, `measure` and experimental `patient` time it. Estimate can still consume
matching wisdom already present in the process; the backend does not clear shared FFTW
wisdom. Planning is destructive and precedes state initialization.
The plans runs measure outlive the process, by default (ADR-076). The adapter
(`solprop_native._wisdom`) keeps them by plan key (`fft_plan_key`: kind, sizes,
strides, placement, alignment and threads), each with the wisdom of that transform
alone, in `wisdom.json` in the calibration store's directory of the machine and build
(`<store>/machines/<fingerprint>/`, ADR-074's fingerprint), or as
`<name>.native-<fingerprint>` beside the file `SOLPROP_FFTW_WISDOM` names;
`SOLPROP_CALIBRATION=off` keeps no default file, and the process keeps what it measured
in memory for its later workspaces either way. A run's own planning (the policy's, or
the profile's) is offered the stored plans (`FftPlanning::stored`) and asked to keep
what it measures (`keep`). In `begin_seed` a key the profile holds is planned from the
profile's wisdom, as before; any other seeded above estimate plans *alone*: the
workspace's accumulated wisdom is exported and forgotten, only that key's known wisdom
imported (stored, or measured earlier in this workspace, so a key seeded twice is
measured once; a key to measure starts from what the workspace measured so far, whose
sub-plans it shares), and after the seed (`release_seed`) the transform's own wisdom is exported
into `measured` when the seed added an entry the stored text lacked, or the key is
listed in `reused`; the workspace's wisdom is then imported back over the transform's,
so that a profile's entries win where both hold one. A key left at estimate that the
store holds is not seeded: its wisdom is imported beneath the workspace's for
Kokkos-FFT's estimate plan to take, and it counts as reused. Seeds and their Kokkos-FFT
plans interleave (a workspace seeds several transforms before planning them), so the
wisdom accumulates within a workspace rather than being forgotten between keys; SEP 46's
transaction still forgets it before and after a profiled workspace. After construction
the adapter stores the measured plans not held by the profile, so a profile's winners
are never overridden. Readers take no lock, since the file only ever changes by an
atomic rename; writers serialise on `<file>.lock` (`flock` on POSIX), merge the file's
plans with their own and replace it only when that adds or changes one. A calibration's
own workspaces (given `fft_planning`) and its children (`off`) neither read nor write
the store. The C++ side only converts wisdom to and from text (`export_fftw_wisdom`,
`import_fftw_wisdom` in `fft_plan.hpp`), registering FFTW's threaded solvers first,
since they are part of the planner signature an import must match. Metadata records
`wisdom_loaded` (the file held plans) and `wisdom_digest` (of the plans the workspace
built from wisdom or measured).
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

On a square map (as many nodes as modes) that product moves to the retained spectrum
(provider `<blas>-retained-interleaved-real`). A real static synthesis commutes with
the temporal transform, so each evaluation gathers once the coarse bins the stage
entry wrote, and each chunk multiplies them as `2 n_omega` interleaved real columns
instead of the padded analytic signal's `2 n_to`, writes each node's retained bins into
the free leading rows of the modal time buffer (whose other bins stay zero: an
out-of-place complex transform leaves its input as it was) and transforms the nodes
(persistent full- and tail-chunk plans). The node
transforms replace the modal ones one for one; the held radial plasma already projects
after cropping, so both products cover only retained frequencies. Runs with an
in-stage absorber or a Raman drive (which read the modal time field), dispersive
responses and non-square maps keep the time-domain product. The two paths
agree to rounding (a radial filament stage from the same state to about 1e-15 of its
peak).

**Raman on the node pipeline (host builds).** A scalar node field is
`x_j = sum_m S_jm a_m` with a real static synthesis, so its Raman drive (`x_j^2`, or
the cycle average `|x_j|^2 / 2`) is a quadratic form in the modal fields and the causal
convolution is linear. When the `M (M + 1) / 2` unordered modal products are fewer than
the nodes and fit the node drive's leading rows, each product (cross terms doubled) is
convolved once on the doubled grid and every node's convolved drive is reconstructed
from them with `S_jm S_jn`, in place of one drive transform pair per node
(`raman_fft_tasks["products"]` is then nonzero). Three capillary modes at 33 nodes
transform 6 rows instead of 33. Vector states, tensor drives, the general and
dispersive routes and devices keep the node drives.

**On a device.** Non-plasma responses distribute time samples within a node team; plasma
uses the same team shape with a deterministic inclusive scan. Error, quadrature,
weighted-control and statistics partials use persistent fixed-width device reductions. The
geometry selected on a data-centre NVIDIA GPU is response team 256 for `n_to < 8192`,
response team 512 otherwise, and reduction block 512. Persistent packed chunk and tail
matrices let KokkosBlas dispatch exact node chunks through cuBLAS without trial-time
conversion or allocation. cuFFT plans are persistent and explicitly none-normalised. The
response column's `if constexpr` blocks carry a capture anchor, because CUDA 13's extended
lambdas otherwise refuse the capture.

Every top-level kernel dispatches through `launch::parallel_for`, `parallel_reduce` and
`parallel_scan` (`native/include/solprop_native/launch.hpp`). On a device they hint a
functor of at most 3840 B light-weight, which Kokkos passes as the kernel's argument;
Kokkos's HIP default copied any functor of 512 B or more into constant memory, a blit
kernel behind a wait on the previous such launch, which kept the host from queuing a
trial's kernels ahead. A plain range reduction on HIP takes launch bounds of 256
threads with the hint, since Kokkos gives a light-weight HIP reduction its largest block.
CUDA builds pass functors as grid constants and are unchanged; host builds dispatch as
Kokkos does. The device Adams passes hold a trial's phi-functions and history in registers,
dispatched on its order and width (one to eight samples): indexed per-thread arrays live
in scratch memory. An accepted Adams step ends without a fence on a device, whose
execution instances are its one stream. A device holds the two held plasma currents'
fine and coarse rows in one buffer that one plan transforms.

On a device an elementwise Adams trial's source evaluation runs in a graph region
(`graph::Region`, `native/include/solprop_native/graph.hpp`). The region's launches, its
device-to-device copies and its transforms go through `launch::parallel_for`,
`graph::deep_copy` and `graph::execute`, which key each operation by its kind and the
value bytes of its policy and functor, or of its views, plan and scalars. The first trial
captures them into a CUDA or HIP graph; a trial whose keys all repeat in order issues
nothing and launches the instantiated graph, which saves the device about a microsecond a
kernel; a trial that departs re-captures from the first difference. A key leaves out a
type's padding, which the region zeroes on the stack, and the host pointer that nvcc's
wrapper of an extended `__host__ __device__` lambda appends to its captures. Reductions,
scans, vendor GEMM, team launches with level-1 scratch, functors Kokkos would stage in
constant or global memory, work on another stream, copies to or from the host and loaded
Kokkos tools disable the region for the workspace's life, as does device work that
bypassed the wrappers (found in the capture before the next operation is skipped). The
header states why a matching key replays exactly what the trial's own launches would do.
Host builds call Kokkos's and Kokkos-FFT's own functions through those names and compile
none of it; `SOLPROP_NATIVE_DEVICE_GRAPH=OFF` does the same on a device.

A trial's per-stage scalars (with a Raman run's stage rates and densities and a compensated
Adams trial's rate weights) live in one packed buffer of two slots and cross in one copy a
trial from a ring of eight pinned host buffers (`Kokkos::SharedHostPinnedSpace`), the ring's
reuse waiting for the space to drain; an Adams trial's step scalars fill slot 0 and its
sample's slot 1, and the stage views turn to slot 1 for the source
([SEP 39](seps/archive/sep-0039-native-adams-march.md) §6.7). A radial analytic field's node
chunks take the interleaved real GEMM on a device as on the host, and a square radial
analytic map synthesises its retained spectrum there too (the gather and the pad a thread
per element, the chunk's padded rows zeroed first): the product covers `n_omega` complex
columns rather than `n_to`, which halves the binary64 work that bounds a radial trial. On free space the split-step collar acts on the stored
transverse-spectrum columns (every build), and a device forms its factor once a step per
sample. A free-space envelope's collar transforms those columns in place, a block of rows at
a time, through no coarse buffer; the transform's normalisation goes with the factor. A
free-space envelope whose material response is the Kerr term alone, with one
polarisation, on a time grid the spectrum fills (`n_t = n_to = n_omega`, the crop the
identity) takes its stage core on a device through one rank-3 transform of the scratch
volume each way, forming the polarisation in place between them, instead of the chunks'
transposing pack and crop and their temporal transforms. On a device the stage holds that
scratch as `(y, x, omega)`, the state's own order, so the transform keeps frequency
contiguous and the stage's input and fold stream beside the registers. Every other pass
over the scratch forms its own input in `(omega, y, x)`, and an in-stage absorber keeps
that order for the stage too. The state crosses the seam once
at a load and once at an observation, straight between the memory space and the NumPy
buffer.

The singleton constructor requires a one-member problem; fixed-schedule batches go
through `select_batch` over independently loaded member workspaces (§1). The backend
refuses scalar-only declarations on vector Raman problems, spectral responses other than
the compiled envelope-plasma assembly, and, on a device build, z-tabulated operators or
prefactors, before field-sized allocation (the CPU build runs them, §1). Supported mixture density, pre-ionisation and Raman decay paths
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
release and plans happen at construction; the `general_*` rows join
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
reference's allowance and words. §3.9 has the algorithm and the reason for its gate. Per
stage the engine carries the state and the stage sum forward (`U u + h U sum a_ij k_j`, as
the reference's `stage_input`), writes the source into the flow's drive, adds
`exp(-dPhi) R_s(z_i) lab` for a frozen-residual flow (the residual combined from the
stage's nonzero `flow_weights` over the held terms, minus the step's frozen block), carries
the drive back by the inverse action and accumulates; the accepted update takes `U(h)` of
the state, the solution and the error sum, the old norm on the unmixed state. A zero stage
fraction is flagged and not applied; FSAL is off on a flow. The `flow_*` rows join
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
`numpy_ref` runs it, with the structure of §3.11 route by route: the fused node
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
  forward transform. Alone, the response takes over the stage's transforms (§3.11). A
  coupled wall flow beside it carries each stage's input, the source being
  the fused path's own.
- **The separable routes** hold the filtered fields and one node slot (symmetric) or two
  (polynomial) on the planar route; free space's inverse transform writes the filtered
  fields straight into the product blocks. A lone symmetric response always filters the
  chunk's spectrum in place: the in-stage absorber is a pass of its own there.
- **The general route** keeps its inverse transform's `1 / n_to`, which the response
  applies to what it reads (a power of two).
- **Order and memory.** Every kernel is a fixed-order loop inside one output element;
  views and plans are made at construction, and seven `dispersive_*` rows join
  `memory_ledger()`, zero on every other problem.

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
- **Small transforms on one FFTW thread; larger ones by shape.** A modal or spatial plan
  whose real side holds fewer than **8 192** samples per execution (32 768 on macOS; the
  execution record's `fft_threshold`; `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`
  overrides it) is made on a one-thread execution instance, because below that a
  threaded FFTW plan only costs time: the threshold is a speed threshold. Above it a
  batch takes a thread per `fft_thread_elements` samples, at most one a row unless its
  rows are `fft_long_row` long, when it may take the medium width
  ([transforms by shape](benchmarks.md#transforms-by-shape-and-one-forward-pack)). FFTW's planner may choose a transform's factorisation by
  the thread count (a two-column 256-point inverse took radix 4 with 64-point leaves at
  three and four threads and radix 8 with 32-point leaves at others), which moves ten
  realistic steps by at most 8 u: within the transform's rounding bound, so it constrains
  nothing. This threshold and the stage threshold switch only on `__APPLE__`
  (`include/solprop_native/core.hpp`): Windows and linux-aarch64 builds take the values
  measured on x86-64 Linux (Cascade Lake and Zen 3), which have not been measured on
  those platforms ([#328](https://github.com/LupoLab/solprop/issues/328)).
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
Linux, then NumPy's `__cpu_features__` on x86-64, and a refusal if neither answers;
`IsProcessorFeaturePresent` on Windows). The x86-64 wheels hold a second extension
at `x86-64-v2`, built by `native/ci/build_variant.py`: `select_variant` takes the first
build the processor meets and registers it as `solprop_native._core`, and
`SOLPROP_NATIVE_CPU_VARIANT` names a lower one, for tests and comparisons. On a processor
below a wheel's floor,
`resolve_name("auto")` raises the package's refusal as a `BackendError`; it loads
that pure-Python check by path and runs it, so it neither imports the package nor
starts its runtime, and the two refusals are one check. An explicit
`"native"` raises the refusal at import. It links Accelerate on macOS; on Linux an
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

## 4a. `mlx`: retired

[SEP 33](seps/archive/sep-0033-single-precision-mlx.md)'s experimental single-precision
backend for Apple GPUs was retired, unused, by
[SEP 47](seps/sep-0047-run-ownership.md) §6.6, before the driver hierarchy it inherited
was restructured. `backend="mlx"` raises `BackendError` (`registry.REMOVED_BACKENDS`),
and output that records it still opens. Single precision runs on `numpy_ref`, its
oracle. SEP 33 and the benchmark ledger keep what it measured.

## 5. JAX: evaluated, not adopted

A JAX backend was measured against the since-retired `numba_cpu` backend at the step level
on the RDW and CtC problems: its out-of-place transforms, unrequestable multi-accumulator
reductions and materialised dtype promotions lost the small RDW step by about three times
while it was slightly ahead on the Raman-carrying CtC step, and under the production pattern
of many concurrent single-thread runs `numba_cpu` had twice its throughput. So JAX stays
evaluated-only (ADR-002), and the compiled path, CPU and GPU, is the native package.

## 6. Transform placement

`native`'s temporal and modal row transforms run out of place between persistent views, and
its transverse and slab transforms in place on the transverse scratch; nothing is designed
around either. An out-of-place FFT reads A and writes B and an in-place one reads
and writes A — the same byte count — so the saving appears only if out-of-place forces a
copy back. At solprop's sizes what matters is **cache
residency**: a field of 64 KiB–2 MB with its plan tables and the propagator factors stays in
L2 across a stage only if the working set is small.

## 7. Benchmarks

`benchmarks/` holds the performance tracking suite (see [benchmarks.md](benchmarks.md)):
complete runs of representative problems, each in its own process, recorded per hardware
set in `benchmarks/history/`. Every performance claim in this document must point at a measurement recorded
under [benchmarks.md](benchmarks.md#reporting-a-performance-claim)'s rules or be marked as
an estimate.

## 8. Memory ledger

Every backend reports `memory_ledger()`: buffer name → bytes, including transform work areas.
A test asserts which entries are field-sized, that everything is named even when empty, and
that the total is within the documented budget for the standard cases.

**The native Adams march** (ADR-060, [SEP 39](seps/archive/sep-0039-native-adams-march.md)).
`native` steps `Multistep()` itself, in either formulation, the one-order-higher PEC
comparisons and replayed schedules: `_NativeAdamsMarch` hands the native state
solprop's `MarchPolicy` (and a replay's schedule) once, then each call runs attempts on
the resident engine -- stage scalars and tables evaluated natively from the run's
uploaded expansions, the trial, the decision, the accept, and the boundary monitor's
checks ([SEP 45](seps/archive/sep-0045-native-rk-march.md) §4.8) -- until a host event, and
returns the
march's state, one row per attempt (its end, step, norm, verdict, order and wall
time) and the readings its checks took, which the driver replays through solprop's
monitor and holds the march's predictions to (`BackendError` on a difference). The decisions are compiled without contraction in their own translation unit,
so that they round as solprop's do; within one build the two marches take the same
steps. A configuration it does not cover runs solprop's march on the same engine with a
`HostMarchWarning`, and the metadata's `adams_march` says which stepped a run. The
host marches are no longer production modes ([SEP 47](seps/sep-0047-run-ownership.md)
§6.3): the environment variables `SOLPROP_ADAMS_MARCH`, `SOLPROP_RK_MARCH` and
`SOLPROP_RK_ROWS` and the backend's `adams_march` keyword are gone, and only a test or
a diagnostic benchmark arm (`benchmarks.adams_host --host-march`) selects solprop's
march, through the module's test interface (`_ADAMS_MARCH`, `_RK_MARCH`, `_RK_ROWS`),
which sends the run to the transitional bridge. Every representation the translation
forms is read by the native stage tables (a z table on CPU builds only, which device
builds refuse), so no admitted run has stage rows the host forms; a problem stripped of
its representation's leaves is refused by name at admission (`host_rows_refusal`).

**Resident Adams storage.** `native` allocates dense-output storage only when
`interpolate_planes=True`. For maximum predictor order `q`, its ordinary resident
history owns `2 (q + 2) + 4` complex field volumes, plus the norm's small scalar arrays.
Dense output adds `q + 2` slope volumes and four saved fields; a varying operator's
compensation adds `3 (q + 2) + 3` volumes, and `q + 2` more with interpolation.
All required buffers are allocated before stepping. This changes storage, not the
method or its maximum order. The [Cartesian memory investigation](benchmarks.md#resident-adams-storage)
distinguishes these allocations from process peak RSS, host copies and FFT workspaces.

**No admission check.** Nothing compares a run's memory with the machine's before the
backend allocates ([ADR-065](decisions.md)). A backend allocates and first-touches its
buffers at construction, so a run that doesn't fit fails there, with `MemoryError` or
killed by the kernel or the scheduler; `Simulation.estimate` reports the constructed
ledger's total. Scan workers and batch groups are still bounded, from measured RSS and
the ledger ([scans](../guide/workflows/scans.md)), and so is TDSE's atomic storage, by
`TDSEExecution.memory_limit`. On the general modal route memory is dominated by the dense
maps and the resident E/H reference, `16 Q C M n_omega` bytes per map and up to 14 such
maps on a constant guide on every backend (`native` holds two natively; the other twelve
are the Python-owned record). The user-facing account is
[Execution resources](../guide/performance/resources.md#memory).

**The low-storage march** ([SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md))
holds no stage, accumulator or propagator-pair row: `numpy_ref`'s ledger
(`reference_base_memory`, method-aware) names the state, `register_accumulator` and
`register_stage`, the transverse scratch on the free kinds and, tabulating, the flow
table in `propagators` (one volume of `exp(h D / 2)` on a constant operator, the step's
`P` and `Q` on a varying one, nothing when recomputing). On a Cartesian grid that is five
field-sized volumes tabulated and four recomputed, one more with the `modified` noise,
against 13 to 16 for the Runge–Kutta march and 20 to 44 with the Adams history.
The native backend runs it on register kernels
holding none of the Runge–Kutta march's buffers -- `GeneralWorkspace` then
allocates no `stages_`, `solution_`, `error_`, `residual_`, `recycled_`, `advanced_`,
`whole_propagator_` or `stage_inverse_` -- and elides the stage register on the
Cartesian and line routes without an in-stage collar (`low_storage_stage_register`):
four volumes tabulated, three recomputed. Its ledger's march rows (the state, the
registers, the flows counted by `low_storage_flows`, the transverse scratch) are the
reference's. Recomputing on a modal or general route, `native` forms the flows
per element from the phase terms and holds none, so no flow or exponent is a transient
outside the ledger. The
native state refuses what its folds cannot read: an elided stage register beside an
in-stage absorber, two flow volumes on a separable geometry, and a step reusing a
constant operator's table it has not built since the field was loaded (solprop's rules
never ask for any of them).

**The far-field pass** (§3.13) holds `far_field_partials` on the native backend, the band
partials, `8 n_bands n_slots n_omega` bytes; the estimate counts them to the byte in `base`.
`numpy_ref`'s reference pass allocates per call.

**Statistics.** The modal power weight enters the statistics alone: `numpy_ref` evaluates
the analytic Poynting functional, and `native` consumes the weights in its
statistics kernels. The extra weighted inverse transform is confined to diagnostics; the
propagation stage and response bodies keep their operation order.

## 9. What runs where

Every backend refuses what it does not run by name, before allocating; nothing falls back.
"Qualified" means the parity and repeatability tests pass within their stated bounds on
that execution space.

| capability | `numpy_ref` | native CPU | native CUDA / HIP |
|---|---|---|---|
| fibre: mode-averaged and multimode, field and envelope grids, Kerr, scalar and tensor Raman, field and envelope plasma, noise, both absorber schemes | yes | yes | qualified |
| an operator or prefactor that took the z table (not `device_ready`) | yes | yes: the coefficients resident, evaluated natively into term slots | refused, naming the table's size |
| radial and Cartesian free space (including a line, `n_y = 1`) | yes | yes | qualified (CUDA) |
| the hybrid planar kind (`"slab"`) | yes | yes | held field-plasma source qualified on HIP; other combinations not qualified |
| general modal route (modal tables, metric, regional responses) | yes | yes; a delayed response only as the six silica tensor channels on three components | qualified |
| coupled flow: constant-Schur and frozen-residual sections | yes | yes | qualified |
| coupled flow: `"cf4"` sections | yes | refused | refused |
| the dispersive electronic response (`responses.dispersive_kerr`) | every route | every route | qualified on one AMD Instinct device (HIP) and one NVIDIA Hopper device (CUDA) |
| the low-call Adams integrator (`StepperSettings.integrator`), both formulations | yes | every route, the history resident; beside a coupled flow the exponential formulation alone, on constant-Schur and frozen-residual sections, the history resident at any width and on either route of each bin (the eigenbasis or the augmented exponential), in the native session | beside a coupled flow the history resident when every bin takes the flow's eigenbasis and it has at most 16 channels, and otherwise solprop's march with a `HostMarchWarning`; qualified |
| the Adams carrier frame (`Multistep(carrier_frame="scalar")`, SEP 35) | every guided route; free space refused | CPU builds, resident, on every representation of the operator (a z table's chord carried as its own scalar table, #255); device builds refused | refused by name |
| the low-storage RK4IP march (`LowStorageRungeKutta`), replay-only | every diagonal-flow route | every diagonal-flow route, a resident engine whose step is one native call | flows recomputed per element; qualified under HIP, not under CUDA |
| moving solid guide (moving maps beside a frozen-residual flow) | yes | yes | qualified |
| fixed-schedule lockstep batches (no coupled flow) | fibre and general guides | fibre and general guides | fibre and general guides |
| periodic statistics between planes | yes | yes | yes, host rows, one packed transfer per row |
| Adams snapshots and a resume that continues the history (`MarchSnapshot`) | yes | yes: the resident engine copies its rows (`store`, and `states`, `rates`, `applied` where kept) to the host at a kept save and back on a resume | compiles; not qualified |
| field-free observation (`observe(field=False)`), the runner, a transverse collar's rate and a set evanescent rate on the separable element | yes | yes (field-free observation needs no native change; the rate is one view in `SpatialCoefficients`) | compiles; not qualified |
| materials from optmat: gas Raman leaves of many rows on gradients, rate tables, optmat walls, the silica bundles | yes | yes | not qualified |
| coupling between runs (`field_plane`, `regrid`, `couple`, `Outcouple`, `transport`, `with_input`; SEP 31) | host-side: any source and receiving backend | host-side | receiving runs not qualified |
| single precision (`precision="single"`, SEP 33): fibre, radial and Cartesian routes, Kerr, Raman, field and envelope plasma, both absorber schemes, Runge–Kutta and the low-storage march, fixed-schedule batches | the oracle | refused | refused |

**Materials from optmat** ([SEP 3](seps/archive/sep-0003-optmat-materials.md), a run under
`MaterialSelection`) reach the backends as the arrays any other run carries: index and
wall tables, carrier Kerr coefficients, rate tables, and Raman rows with their decay
paths. No backend reads optmat. The native CPU build is qualified on them against
`numpy_ref` (`tests/backends/test_optmat_*_parity.py`): argon and nitrogen plasma on
field and envelope grids, the rate table's hold above a model's maximum field, a
helium/argon Kerr mixture, silver and silica walls, the GNLSE with the silica bundle,
and gas Raman leaves of 4, 72 and 126 rows — dry air's composition leaf, driven by the
summed density of its labels — on gradients to vacuum, scalar and vector, field and
envelope, and in lockstep batches. Each agrees with the reference to a few times
`1e-16`, far inside its operation-count bound. `native` assumes no term
count: it holds a leaf's `n_terms × 2 n_to` rows and its per-stage decay rates as
views sized from the problem. A fill whose populations follow independent density
paths takes the z table, which `numpy_ref` and native's CPU build run;
a shared pressure path does not. Device builds are not qualified on these runs.

**Coupling between runs** ([SEP 31](seps/archive/sep-0031-simulation-coupling.md)) never crosses
the seam: a field plane is read from a run's stored output and its record, coupled and
transported in NumPy and SciPy on the host, and installed in the receiving run as its
initial state through `with_input`. A source and a receiver may therefore run on
different backends; the integrated compositions (`tests/coupling/test_workflows.py`)
pair `numpy_ref` sources with native CPU receivers, and the relaunch
tests (`tests/driver/test_coupling_relaunch.py`) cover every pair of the two CPU backends.
Device receivers are not qualified.

`numpy_ref` is the oracle: every other backend is qualified against it, and it is the one
to run when two backends disagree, and the one that needs nothing beyond the package's own
dependencies. The native backend's device builds
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

## 10. What each backend is for

These rules set where development goes, not what a backend may run (ADR-061; ADR-064 retired
`numba_cpu` and superseded the statement that the documentation build installs no native
backend; ADR-068 makes `native` the primary backend). A feature lands on `native` and
`numpy_ref` in either order. Every backend keeps what it runs, correct and in parity with the
reference, and refuses the rest by name.

| backend | role | new features |
|---|---|---|
| `native` | production at speed, on the CPU and on a device; what `auto` selects | every one: the CPU build first, device builds in one qualification pass later; first for testing and development |
| `numpy_ref` | the numerical reference, for correctness and for developing an algorithm, and the executor of [the listed features](../guide/performance/backends.md#features-that-run-on-the-reference-backend) | every one; first for testing and development |

A SEP is final when its feature runs on both `numpy_ref` and `native` with parity tests
([SEP 0](seps/sep-0000-process.md)), or an accepted exception adds it to the list. A backend's
role changes by a new decision record, and a new backend starts as experimental. `backend="auto"`
is `native`, and refuses where the native package is missing or the processor is below its
wheel's baseline (§1). The CI `docs` lane installs the native wheel, so a lesson or tutorial may
use anything `native` runs; Read the Docs executes no gallery script, because all gallery
output is committed, and installs no native backend.

Which backend and build to choose for a run is the user guide's
[Choosing a backend](../guide/performance/backends.md#which-backend-to-use).

## 11. Current limits

- `"cf4"` coupled-flow sections run only on `numpy_ref`.
- Single precision runs on `numpy_ref`, its oracle, alone; it
  refuses the general modal route, the hybrid planar kind, coupled flows, TDSE responses,
  the dispersive electronic response, the far-field statistics, noise models, the Adams
  integrator and an adaptive `rtol` below `1e-6`.
- The dispersive electronic response runs on `numpy_ref` and `native` on
  every route; on a device it is qualified on one AMD Instinct generation (HIP) and one
  NVIDIA Hopper device (CUDA); a polynomial mixture costs one group per distinct leg
  bound (§3.11).
- The held field-plasma source is qualified on a device under HIP only (§9). On the
  hybrid planar kind other response combinations are unqualified on a device build.
- `native`'s device builds refuse any problem whose operator or prefactor took the z
  table; its CPU build runs it.
- The default periodic-statistics subset is cheap under Runge–Kutta at `period = 1`, because
  the plasma rows read the accepted step's stash (§3.1a; the slow lane gates its cost, and
  it was about 2.5 % of a step on the retired `numba_cpu` backend); under the default Adams
  integrator the source is evaluated at the predicted state, so the plasma rows scan the
  accepted field afresh and cost more.
- HIP is qualified on one AMD Instinct generation with a prerelease ROCm stack; stable ROCm
  and other devices are unqualified, and ROCm ships no device sanitizer.
- A device answer on the general route agrees with the CPU build's to rounding (the device
  projection's summation order), and the local and distribution native builds' answers agree
  with the reference build's to rounding: which is all any comparison requires
  ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)).
- Batches are fixed-schedule only; adaptive and spatial batches, and batches carrying a
  coupled flow, are refused.
- No native build runs on Intel macOS, ppc64le, Arm below Neoverse N1 or Windows before 10
  version 2004, nor on x86-64 below x86-64-v2, and solprop doesn't support them. With no
  wheel the installer refuses; where a wheel installs, `auto` and `native` refuse on the
  processor. NumPy's own baseline may refuse x86-64 below v2.
- A GPU build of `native` runs nothing on the CPU in process. A CPU run beside one needs a
  second environment with the CPU wheel, or `numpy_ref` in process.
