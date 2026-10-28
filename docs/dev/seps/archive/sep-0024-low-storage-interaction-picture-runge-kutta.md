# SEP 24 — A low-storage interaction-picture Runge–Kutta march for memory-bound fixed-step runs

| | |
|---|---|
| **SEP** | 24 |
| **Title** | A low-storage interaction-picture Runge–Kutta march for memory-bound fixed-step runs |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-16) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-16 |
| **Resolution** | Accepted 2026-10-18 by John Travers after his final review, on the revisions of 2026-10-18 recorded in Part III; Part I is frozen from here |
| **Requires** | [SEP 16](sep-0016-unified-multistep.md) (the integrator family and the `integrator=` setting); [SEP 14](sep-0014-stepper-efficiency.md) (fixed-schedule replay beside natural-step and interpolated planes); related, not required: [SEP 22](sep-0022-memory-lean-spatial-execution.md), drafted alongside |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | phase 1: Opus 5.5 (AI), branch `feature/sep-0024-phase-1` (solprop and solprop-kokkos); phase 2: Opus 5.5 (AI), branch `feature/sep-0024-phase-2`; phase 3: Opus 5.5 (AI), branch `feature/sep-0024-phase-3` (solprop and solprop-kokkos); closing: Opus 5.5 (AI), branch `feature/sep-0024-closing` (solprop and solprop-kokkos) |
| **Manual pages** | [Choosing a stepper](../../../guide/numerics/choosing-a-stepper.md#the-low-storage-march), [fixed schedules](../../../guide/numerics/fixed-schedules.md#the-low-storage-march), [the solver](../../../guide/numerics/solver.md#three-integrators) |

## Part I — Proposal

### 1. Abstract

Add a third integrator, `LowStorageRungeKutta`, beside the default `Multistep()` and
`RungeKutta()`. It is the classical fourth-order Runge–Kutta method in the interaction
picture anchored at the step's midpoint: Lawson's transformation [1] in Hult's
arrangement [2], known as RK4IP. It runs as a replay-only march of a prescribed
`FixedSchedule`. Beside the source pipeline's scratch it holds three field-sized
registers: the state (overwritten in place by its interaction-picture image, then by the
accepted step), one accumulator and one stage register. Each slope is consumed as it is
produced, and each step is accepted in place. The method
has no embedded error estimate. Its accuracy is controlled from outside, by solprop's
existing three-level Richardson calibration (`solprop.stepper.schedule.calibrate`)
applied to the measured observable.

On a Cartesian free-space grid, where one state takes gigabytes, the march's field-sized
volumes fall to four, against 15 (Kokkos) or 16 (`numba_cpu`) for the Runge–Kutta march
and 44 for the default Adams march. A tabulated half-step flow adds one volume, and the
`modified` noise another. The design is also the fast one. In a drafting measurement on a
Cartesian Kerr case, given as motivation only, a standalone prototype of this register
form cost a fifth of what the current IP-DOPRI5 replay costs per source evaluation.

Success is judged by:

- an observed fourth order on the soliton, an analytic varying operator and a
  free-space Kerr case;
- a linear limit exact to rounding;
- agreement with the in-tree RK4IP to a round-off bound;
- parity across the three CPU backends;
- ledgers within the target;
- ten-step benchmarks against IP-DOPRI5, at equal step and at equal Richardson error, with
  an in-tree target of at least 3 times less time per evaluation and 4 times less per
  step.

An embedded pair for adaptive stepping is an optional fourth phase.

### 2. Motivation and scope

**One state takes gigabytes.** A Cartesian free-space run stores its state on
`(n_batch, n_modes, n_pol, n_omega)` with `n_modes = n_x n_y`
([conventions](../../conventions.md) §6). The field-sized unit of
[architecture](../../architecture.md) §8 is therefore
$F = 16\,n_\mathrm{batch}\,n_x n_y\,n_\mathrm{pol}\,n_\omega$ bytes. At $768\times768$
columns and 600 bins, with one polarisation and one member, $F$ is 5.66 GB (5.27 GiB).
Doubling $n_x$ and $n_y$ for a transverse convergence check quadruples it.

**What the marches hold.** The backends' ledgers count these field-sized volumes on the
free kind:

| backend | Runge–Kutta march (DOPRI5) | Adams history beside it (exponential, order 5) |
|---|---|---|
| Kokkos (`GeneralWorkspace::base_memory_ledger`, `adams_history`) | 15 F: state, accept buffer, seven stage rows, two accumulators, the weighted-norm residual, the recycled first-same-as-last (FSAL) stage, the whole-step propagator, the transverse scratch | 29 F, so 44 F in all |
| `numba_cpu` (`NumbaCpuBackend.memory_ledger`) | 16 F: state, seven stage rows, two accumulators, accept buffer, exponent, recycled stage, a propagator pair, the transverse scratch; plus one accept-buffer copy per trial (`_trial_step`) | 16 F (`adams_store` 7, `adams_moments` 8, `adams_flow` 1), so 32 F |
| `numpy_ref` (`solprop.backends.memory.reference_base_memory`) | 13 F counted: state, seven stages, two accumulators, a propagator pair, the transverse scratch | 7 F (`adams_store`), so 20 F |

The history counts come from `solprop.backends.memory.adams_memory` ($q+2=7$ slots at
order five). At the shape above the Runge–Kutta march takes 79 GiB on Kokkos and 84 GiB
(plus a 5.3 GiB transient) on `numba_cpu`. Adams takes 232 GiB on Kokkos, which
`solprop.driver.memory._admit_history` refuses, naming `RungeKutta()`. Even with a single
stage row, the free-kind pipeline on Kokkos would hold about seven
volumes: state, transverse scratch, the row, two accumulators, the propagator and the
accept buffer. No tableau gets below that floor.

**What a fourth-order step needs.** Hult's midpoint arrangement [2] applies only forward
half-step flows. In the accumulator organisation each slope is folded into a running sum
and the next stage's input as soon as it is produced. The step then needs three registers
beside the evaluation's scratch (§4.3). Families built for this exist [3–6]: two-register
(2N), 2R and 3R schemes with embedded estimates, and minimum-storage classes that can
restart a step or estimate its error. solprop has none; its marches were built for
adaptive control.

**What the registers cost in time.** These are drafting measurements, given as
motivation only. They were taken on a contended machine against an out-of-tree prototype,
so they are not acceptance evidence; the acceptance target is set in-tree in §7.4. They
also predate [SEP 25](sep-0025-rounding-level-reproducibility.md)'s phases 3
and 4, which fused the compiled passes and the Kokkos stage kernels and batched the
stage scalars, so the IP-DOPRI5 replay of the current tree is faster than the figures
below; §7.4 is measured against the current tree. The
case was a Cartesian envelope grid of $192\times192\times300$ ($F=177$ MB) with Kerr
only, no transverse collar and 0.5 µm fixed steps, timed over ten-step windows on one
thread of an Apple-silicon laptop.

- **The IP-DOPRI5 replay** on `numba_cpu` took about 7.0 s a step. It ran a spectral-only
  absorber under `absorber_scheme="split_step"`, which forgoes the first-same-as-last
  stage, so it made seven source evaluations a step, about 1.0 s each (an earlier
  reading divided by six and quoted 1.15–1.19 s); Kokkos CPU was within 2 % of that.
- **A standalone prototype of §4.3's register form** (three registers, the half-step flow
  tabulated once, one fused elementwise pass per stage, transforms in place) took about
  0.21 s per evaluation: about 4.8 times less, and about eight times less per step.
- **With 18 threads** the ratio is 3.5–6: 22–30 ms per evaluation against 106–131 ms,
  with the same seven-evaluation accounting.

A profile of the `numba_cpu` march splits one DOPRI5 stage (about 470 ms in the profiled
run) as follows:

| share | where |
|---|---|
| about 60 % | `_stage_free`: the Butcher combination over earlier stage rows, the 2-D transverse transform pair, pack and crop, accumulation |
| about 21 % | `_write_propagator`: the per-element complex square root and exponential, recomputed every stage because the free kinds refuse `propagator_mode="tabulate"` (`KernelProblem.validate`) |
| about 17 % | the chunked temporal transform pairs (`_transform_free_sources`, `_chunk_to_frequency`) |
| about 3 % | the Kerr response |

The accepted step's `_finish` adds about 170 ms a step. Chunking is not the explanation:
a node budget of 1 GiB instead of 64 MiB gains 1.3 times, and no more once one chunk
holds every node. The response is a few per cent of a stage, and the rest is register
traffic and the propagator. Fewer volumes, each read once, mean fewer passes, so the
memory target and the speed target are the same design.

**When the embedded estimate buys little.** The target is a fixed-step run whose accuracy
is calibrated from outside, by Richardson comparison of the observable at $h$, $h/2$ and
$h/4$ (`calibrate`, ADR-019). That suits a weak signal under a strong pump, such as a band
orders below the peak or a small difference of strong fields, where a whole-field norm is
blind. The norm is scaled by the whole state over the active band
([stepper.md](../../stepper.md) §3), so a band holding a fraction $\delta$ of the energy
contributes about $\sqrt\delta$ of it; how to score such observables is open
([stepper.md](../../stepper.md) §9). SEP 14's band-norm study (phase 6) found weighting a
weak band equivalent to tightening
`rtol`: the step is shared, and the weak band is made where the strong field evolves.
Calibrating on the observable certifies what a per-step estimate cannot, and dropping the
estimate frees the storage the embedded pair costs.

**Scope.** It serves large Cartesian, line and hybrid planar runs, ensembles and scans on
one schedule (ADR-013), and devices holding a few states but not fifteen. In scope are
the integrator and its record; a replay-only, accept-in-place march and its seam; the
three CPU backends with ledgers and admission; the semantics of a march without an
estimate; refusals by name; documentation, a tutorial and benchmarks; and, optionally,
an embedded pair (phase 4). Out of scope are changed defaults or changes to the existing
marches' arithmetic; adaptive stepping before phase 4; dense output; coupled flows; the
TDSE response; a weak-observable norm (the roadmap item stays open); the pipeline's own
buffers and observation copies (SEP 22); reduced precision; exponential Runge–Kutta
methods; and device qualification (Q7).

### 3. Context

**The integrators.** `SimulationSettings.integrator` (`solprop.driver.simulation`) is
`Multistep()` by default (ADR-045), or `RungeKutta(tableau=DOPRI5)` (ADR-004).
Translation passes the tableau with no integrator, or `Multistep` with DOPRI5 unused.
`StepperSettings.integrator` is typed `Multistep | None` (kernel schema 28).

The Runge–Kutta march is the driver's own (`StepDriver._march_to`, `_replay_to`). Its
abstract `_trial_step(z, dz)` returns a norm and a *raw* end state. `_finish` treats only
a kept step, and the accepted state changes only inside the commit (`committing()`,
ADR-047).

The Adams march (`solprop.backends.multistep.AdamsMarch`, ADR-037, ADR-041 to ADR-046) is
the precedent for a march that owns its buffers through an engine: `AdamsEngine`,
`HostHistory`, or a backend's own from `_adams_engine` (ADR-043). Its source comes from
two hooks every CPU backend implements: `_adams_lab_source` (the complete source at a
lab-frame field and position) and `_adams_propagator` (the flow $U$ between two
positions). It replays schedules itself (`AdamsMarch._replay_to`, `replay_step`), as a lockstep
member does (`LockstepRunner._replay_to`).

**Fixed schedules.** `FixedSchedule.uniform` lands on every plane and kink.
`discover`, `calibrate` and `replay` are ADR-019's workflow
([fixed schedules](../../../guide/numerics/fixed-schedules.md), [stepper.md](../../stepper.md)
§8). `calibrate` refuses an observed order more than `ORDER_MARGIN = 3` above the
scheme's, and calls a ladder inconclusive within `CONCLUSIVE_FACTOR` of a declared
round-off floor.

A replay records each step's norm and warns when `ControlRecord.met_tolerance` is false.
`StepDriver._replay_to`, `AdamsMarch.replay_step` and `StepHistory`
(`solprop.backends.protocol`) all refuse a non-finite norm; 0.0 passes all three.
`StepperSettings` refuses `interpolate_planes` beside a schedule, `step_planes` works
under replay (ADR-039), and a replay resumes at any edge (`StepDriver._schedule_index_at`).

**What the backends allocate, whatever the integrator.** `NumbaCpuBackend._allocate`
takes every tableau stage row, both accumulators, the accept buffer, the recycled stage
and a propagator pair. Kokkos's `GeneralWorkspace::allocate` takes `stages_`,
`solution_`, `error_`, `residual_`, `recycled_`, `advanced_` and `whole_propagator_`.
`numpy_ref` forms stages per step (none for Adams), but its ledger
(`reference_base_memory`) counts them for every problem. ADR-009's "never materialised beyond what the next stage needs" is therefore not what
the backends do. DOPRI5's Butcher matrix is full lower triangular, and dense output
(ADR-039) reads every stage.

**A third kind is a seam change.** Every site that reads the integrator assumes that a
non-`None` value is a `Multistep`:

- **Construction.** `NumpyRefBackend.__init__`, `NumbaCpuBackend._adams_march` and
  solprop-kokkos's `KokkosBackend.__init__` each build an `AdamsMarch` from it.
  `NumbaCpuBackend._fsal_ready` tests it against `None`.
- **Admission.** Each backend's `_refuse_adams` (`numpy_ref`, `numba_cpu` and
  solprop-kokkos) reads `integrator.formulation`.
- **Validation.** `KernelProblem._check_tdse` refuses only `method is None`, and
  `KernelProblem._runge_kutta_advice` advises `RungeKutta()` for any non-`None` integrator.
- **Translation.** The call in `solprop.driver.simulation` maps a non-`RungeKutta` to
  DOPRI5 and a non-`Multistep` to no integrator. Only the `SimulationSettings` type check
  stops a third kind from running silently as DOPRI5.

`Tableau` checks that the error weights sum to zero and that `0 < embedded_order <
order`, so all-zero error weights pass, with a norm that is always zero.

**An independent reference.** `solprop.driver.scaffold.propagate_fixed_rk4` is a
fixed-step RK4IP in Hult's form [2]. It builds the mode-averaged GNLSE's Kerr term from
the physics layer and shares nothing with the reference passes.
`tests/backends/test_numpy_ref_soliton.py::test_the_reference_backend_and_the_scaffold_agree`
uses it to check IP-DOPRI5 at the level of truncation error.

### 4. Theory

#### 4.1 Equation, flows and conventions

SI units and the internal FFT convention of [conventions](../../conventions.md) §2 apply. The
state $\tilde A(z)$ is the stored spectrum: square root of watts on the modal routes,
volts per metre on the free kinds. It obeys

$$\partial_z\tilde A = D(z)\,\tilde A + F\big(z,\tilde A\big),\qquad
D = -i\big(\beta-\beta_0-\beta_1(\omega-\omega_0)\big)-\alpha/2-\alpha_\mathrm{abs}/2,$$

with $D$ complex and diagonal per bin. On the free kinds $\beta$ is each column's
longitudinal wavenumber, assembled per element from one-dimensional arrays (the
`separable` representation, [stepper.md](../../stepper.md) §4).

$F$ is the complete source assembly (SEP 16 §4.1): the response, the in-stage absorbers,
and the `modified` noise draw $\eta(z)$ added to the stage input at its position. $z$ and
$h$ are in metres, $D$ in m⁻¹ and $F$ in state units per metre. With
$\Phi(z)=\int_0^zD\,dz'$ and $U(z_2,z_1)=\exp(\Phi(z_2)-\Phi(z_1))$, diagonal values
commute. $U$ is therefore the exact linear flow over any interval, and
$U(z_3,z_2)\,U(z_2,z_1)=U(z_3,z_1)$ ([stepper.md](../../stepper.md) §1).

#### 4.2 The midpoint interaction picture and RK4

For a step $[z_n,z_n+h]$ let $z_m=z_n+h/2$, $P=U(z_m,z_n)$ and $Q=U(z_n+h,z_m)$. On a
constant operator, which includes every free kind, $P=Q=e^{hD/2}$. Lawson's change of
variable [1], anchored at the midpoint, is

$$u(z)=U(z_m,z)\,\tilde A(z),\qquad u'=g(z,u)=U(z_m,z)\,F\big(z,U(z,z_m)\,u\big).$$

Classical RK4 [8, §II.1] from $u_n=P\tilde A_n$, with $U(z_m,z_m)=1$ at the midpoint
stages, gives

$$g_1=P\,F_1,\quad F_1=F(z_n,\tilde A_n),\qquad
F_2=F\big(z_m,u_n+\tfrac h2 g_1\big),\qquad F_3=F\big(z_m,u_n+\tfrac h2F_2\big),$$

$$F_4=F\big(z_n+h,\;Q(u_n+hF_3)\big),\qquad
\tilde A_{n+1}=Q\Big(u_n+\tfrac h6g_1+\tfrac h3F_2+\tfrac h3F_3\Big)+\tfrac h6F_4 .$$

The fourth slope in the picture is $g_4=Q^{-1}F_4$, and the final $Q$ cancels that
inverse. The step applies four forward flows per element and no inverse: $P$ to the state
and to $F_1$, $Q$ to $u_n+hF_3$ and to the sum. This is Hult's RK4IP [2] and the
scaffold's stencil.

Anchored at $z_n$, as solprop's Runge–Kutta march is, the three later stages each take a
forward flow on their input and an inverse on their output, and the final flow makes
seven. The final flow cancels $k_4$'s inverse, leaving six applications, two of them
inverse flows. An inverse flow multiplies a damped bin by $e^{+\alpha c_ih/2}$; the
midpoint form never does.
The source is evaluated four times a step. No slope carries over, because $F_4$ is taken
at a predicted state, not at $\tilde A_{n+1}$.

#### 4.3 The register form

$A$, $B$ and $C$ are field-sized registers. $S$ is the pipeline's scratch: the transverse
scratch on the free kinds, or the fine spectral buffer on the modal routes. The
evaluation reads its input from $S$ and leaves the source there, in frequency. A step is
four passes:

| pass | input formed in $S$ | evaluation | finish, per element |
|---|---|---|---|
| 1 | $A+\eta(z_n)$ | $F_1$ at $z_n$ | $A\leftarrow PA$; $g\leftarrow PS$; $B\leftarrow A+\tfrac h6g$; $C\leftarrow A+\tfrac h2g$ |
| 2 | $C+\eta(z_m)$ | $F_2$ at $z_m$ | $B\leftarrow B+\tfrac h3S$; $C\leftarrow A+\tfrac h2S$ |
| 3 | $C+\eta(z_m)$ | $F_3$ at $z_m$ | $B\leftarrow B+\tfrac h3S$; $C\leftarrow Q(A+hS)$ |
| 4 | $C+\eta(z_n+h)$ | $F_4$ at $z_n+h$ | $A\leftarrow QB+\tfrac h6S$ |

After pass 1, $A$ holds $u_n$ and $B$ accumulates the bracket of $\tilde A_{n+1}$. $C$
holds the next stage's lab-frame input, which at the midpoint equals the
interaction-picture value. Pass 4 leaves $\tilde A_{n+1}$ in $A$. The march then applies
`_finish` (the hard band mask, and the split-step collar where chosen) and moves the
position in one commit. The noise draw's phase $\exp(i\operatorname{Im}\Phi(z))$ is formed
in the same pass, so no rotated copy is stored.

**The flows: tabulated or recomputed.** On a constant operator $P=Q=e^{hD/2}$ depends only
on $h$, and on a fixed schedule it repeats from step to step; Hult's scheme uses this one
factor [2]. Recomputing it costs a complex square root and an exponential per element, a
fifth of a stage in §2's profile. The march offers both treatments.

- **Tabulated.** One volume holds $e^{h_\mathrm{t}D/2}$, built from the first step that
  needs it after a load. It is reused while
  $|h-h_\mathrm{t}|\le$ `LANDING_ULPS`$\cdot\mathrm{ulp}(h_\mathrm{t})$ and rebuilt
  otherwise.
  - **The slack.** `FixedSchedule.uniform` takes a segment's steps from
    `np.diff(np.linspace(...))`, which differ in their last bits, so an exact key would
    miss. Within the slack the phase error is below $8\varepsilon|D|h/2$ per element and
    step.
  - **Other steps.** A discovered schedule whose steps all differ rebuilds the table every
    step, which costs more than recomputing.
  - **Resume.** A resume may build its table from a different step of the segment than
    the uninterrupted run did. It therefore agrees with that run to rounding (§4.5), not
    bit for bit, as an Adams resume without a snapshot does.
- **Recomputed.** The flow is formed per element in each finish from the operator's 1-D
  arrays, as the Kokkos free-kind stage input forms its propagator. No volume is held.

The table pays on a CPU. On a device an extra field read costs more than the exponential.
A varying operator ($P\ne Q$, both changing per step) forms one or two flow volumes per
step, on routes whose $F$ is small.

**The count** is three registers and $S$, plus the table when used. Each input reads one
register, where an IP-DOPRI5 stage reads up to five earlier rows (the nonzero entries of
its Butcher row), and no accumulator or accept buffer is needed, since the march never
rejects.

**Two evaluations per pass.** Under an in-stage absorber the free kinds' compiled stages
evaluate twice per stage: the material response, then the absorber term, each from the
same input re-formed in $S$. Today they keep the material output in a stage row
meanwhile (`NumbaCpuBackend._stage_free`, the Kokkos free-kind stage). The march needs no
such row. Between the two evaluations one per-element exchange takes the material output
$t$ out of $S$, re-forms the input, and folds $t$ into the registers. In pass 2:

$$t\leftarrow S;\quad S\leftarrow C+\eta(z_m);\quad C\leftarrow A+\tfrac h2t;\quad
B\leftarrow B+\tfrac h3t .$$

After the absorber term lands in $S$, the finish is $C\leftarrow C+\tfrac h2S$ and
$B\leftarrow B+\tfrac h3S$. The other passes adapt it:

- **Pass 3** is the same, with $C\leftarrow A+ht$ in the exchange and $C\leftarrow
  Q(C+hS)$ in the finish.
- **Pass 1** re-forms $A+\eta(z_n)$ first. It then sets $A\leftarrow PA$ in the exchange
  and folds $Pt$, then $PS$.
- **Pass 4** sets $A\leftarrow QB+\tfrac h6t$, then $A\leftarrow A+\tfrac h6S$.

$t$ is a per-element temporary, so the collar case keeps three registers and $S$ and
needs no fifth volume. A collar whose temporal rates are all zero crosses the seam as
none once [SEP 22](sep-0022-memory-lean-spatial-execution.md)'s phase 1 lands, so a
spectral-only run makes one evaluation per pass; SEP 22 §3 measured the second
evaluation at about a factor of two per step on the Runge–Kutta march.
A pipeline that reads its input once can instead write the next
input straight into $S$ (for example $S\leftarrow A+\tfrac h2S+\eta(z_m)$), and then
needs no $C$ (Q4).
Ketcheson's counts [6] assume a right-hand side added into its own register. solprop's
evaluation transforms through a field-sized scratch, so every published count gains $S$
here.

#### 4.4 Order, and its range of validity

Because $D$ is diagonal, the change of variable is exact for any step, $QP=U(z_n+h,z_n)$,
and $g$ is as smooth as $F$ and $\Phi$. RK4 therefore keeps its classical order four
[1; 8, §II.2]: the global error is $O(h^4)$ as $h\to0$ on a fixed grid, with a constant
set by the derivatives of $g$. Under a coupled flow the half-step flows are not exact
exponentials of commuting values, and their error lies outside the Runge–Kutta order;
refused.

The constant is the interaction picture's known weakness. Each $z$-derivative of
$U(z_m,z)$ brings a factor $D$. In a *driven* bin, whose source does not follow the bin's
free phase, each derivative of $g$ grows like $|\operatorname{Im}D|$. The asymptotic range
therefore starts only where $|\operatorname{Im}D|\,h/2\lesssim1$ in populated driven
bins (the primer's chapter 12; SEP 15 §4). Beyond it the method converges, but its
observed order can fall, as for Lawson-type methods on stiff problems [9, §2.7].

SEP 14 (phase 6) measured IP-DOPRI5 at order four, not five, on a soliton under fixed
steps. Empty high-$k_\perp$ bins are not driven and do not limit the order. The schedule
lands on every hard kink (`FixedSchedule.uniform`, checked by `KernelProblem.validate`),
and $\Phi$ is smooth inside a table interval.

#### 4.5 The linear limit and invariants

With no source the step is $\tilde A_{n+1}=QP\tilde A_n$: the exact flow, with two
rounded complex products per element instead of one. Over $N$ steps the products
contribute about $4N\varepsilon$ per element, relative. The arguments are rounded too.
Each $e^{hD/2}$, and the reference $\exp(\Phi(z_N)-\Phi(z_0))$, carries about
$\varepsilon|\Delta\Phi|$, and differencing the cumulative table adds about
$\varepsilon|\Phi(z)|$. The test's tolerance is therefore about
$\varepsilon\,(4N+c\,(N|D|h+|\Phi(z_N)|))$ per element, with $c$ a small constant. The
argument term dominates in strongly dispersive bins: with 200 steps at
$|\operatorname{Im}D|h/2=300$ the measured difference was 655 times $4N\varepsilon$. The hard band mask is a no-op, because the prefactor is exactly zero
outside the active band, and the propagating field never contains the noise draw
(ADR-008).

#### 4.6 Stability of explicitly integrated damping

Damping inside $F$ — the in-stage temporal and spatial collars, plasma loss — is
explicit. RK4's stability function is $R(x)=1+x+x^2/2+x^3/6+x^4/24$. On the negative real
axis $|R|\le1$ for $x\ge-2.785$, the root of $|R|=1$; the imaginary-axis interval is
$2\sqrt2$. A replayed step must keep $h\alpha_\mathrm{max}/2\le2.785$, against 3.3 for
DOPRI5 (`solprop.boundaries.STABILITY_RADIUS`) and 0.0877 for PEC5/6
(`PEC_REAL_INTERVAL`). `KernelProblem._stability_radius` and
`_check_spatial_collar_stability` take the new interval. With the default temporal
collar's strength ($AN = 252$, [SEP 21](sep-0021-absorbing-boundaries.md) §4.3) the
interval sets a floor of about 45 steps over a run, against 38 for DOPRI5. SEP 21's
implicit collar, if built, is for the multistep march; this march refuses
`absorber_scheme="implicit"` as the Runge–Kutta march does.

#### 4.7 No local estimate: control from outside

The march's accuracy is its schedule's, measured on the observable $Q_w[\tilde A]$. In
the asymptotic range $Q_w(h)=Q_w^\star+c_wh^p+O(h^{p+1})$, with $c_w$ unrelated to any
whole-field constant. Nested replays at $h$, $h/2$ and $h/4$ give the differences $d_1$
and $d_2$. From them come $p_\mathrm{obs}=\log_2(d_1/d_2)$ and the base error
$e(h)\approx d_1/(1-2^{-p_\mathrm{obs}})$. This is `calibrate` with refinement two, whose
gate refuses $p_\mathrm{obs}>7$; `replay` then rescales to a target.

For a weak observable the round-off floor matters. The strong field's rounding,
$O(\varepsilon\|\tilde A\|)$ per operation, leaks into the band and sets the floor that
`calibrate(floor=...)` needs. The parity difference between two backends on one schedule
is pure rounding and measures that floor.

#### 4.8 Low-storage families for adaptive use

Williamson's 2N schemes [3] update two registers, $\Delta\leftarrow a_i\Delta+hg_i$ and
$u\leftarrow u+b_i\Delta$; Carpenter and Kennedy's fourth-order 2N schemes have five
stages [4]. Kennedy, Carpenter and Lewis [5] give 2R and 3R schemes with embedded
estimators, among them five-stage fourth-order pairs labelled RK4(3)5[2R+]. Ketcheson's
2S\* methods [6] keep the previous step in one of two registers, so a step can be
restarted, and his 3S\* methods add an embedded estimate in a third; such pairs now have
step-size controllers for method-of-lines problems [10]. Balac and Mahé's ERK4(3)-IP [7]
is an embedded 4(3) pair that keeps RK4IP's nodes and midpoint structure. In the
interaction picture a stage at node $c_i$ needs the flow to $c_i$ forward on its
input and back on its output. Only nodes at 0, ½ and 1 are inverse-free. An adaptive
march also keeps the accepted state until the decision: one register beyond any
estimate's.

### 5. Proposed interface

`solprop.stepper.low_storage.LowStorageRungeKutta`, exported from `solprop.stepper`, is a
frozen dataclass with one field, `scheme: Literal["rk4ip"] = "rk4ip"`; phase 4 may add a
scheme. Its `name` is `"low-storage-rk4ip"`, its representation
`LowStorageRungeKutta(scheme='rk4ip')` and its `real_stability_interval` 2.785. The name
parallels `RungeKutta`. `InteractionRK4` was rejected because it names one scheme, and
`LowStorage` because it says nothing of what is stored.

```text
from solprop.driver import SimulationSettings, prop_free
from solprop.stepper import (FixedSchedule, LowStorageRungeKutta, PIController,
                             calibrate, replay)

planes = [0.01, 0.02, 0.03]
schedule = FixedSchedule.uniform(0.0, 0.03, 2e-4, save_planes=planes)
settings = SimulationSettings(
    controller=PIController(rtol=1e-6, atol=1e-12),  # recorded; see below
    save_planes=planes,
    dz_max=2e-4,                         # at least the schedule's longest step
    dz_init=2e-4,                        # informational under a replay
    integrator=LowStorageRungeKutta(),   # RK4 in the interaction picture, 3 registers
    schedule=schedule,                   # required: the march replays only
    backend="numba_cpu",
)
# Calibrate on the measured quantity, not on rtol:
calibration = calibrate(schedule, weak_band_energy, order=4,
                        requested=1e-3, floor=parity_floor)
production = replay(schedule, calibration, target_error)
```

**Admitted:** a schedule, which is required; every route whose diagonal-flow pipeline the
backend executes (modal guides, the radial, Cartesian, line and hybrid planar kinds,
constant general guides); field and envelope grids; the in-stage collars within §4.6's
interval, the split-step collar (first order where it acts, ADR-025) and the spectral
collar; the `modified` noise (§6.3); `step_planes`, statistics between planes and scans;
and lockstep batches where admitted, each member owning its march. A resume at a schedule
edge needs no history and no recycled stage. Because the flow table may be rebuilt from
a different step (§4.3), the resumed run agrees with the uninterrupted one to rounding,
within §4.5's bound including its argument term, not bit for bit. SEP 14 gave up a
bitwise resume, and the Adams march does not provide one either. `backend="auto"` resolves
to `numpy_ref` until phase 2, then per problem.

**The required settings keep their meaning.** `SimulationSettings` still requires
`controller`, `save_planes`, `dz_max` and `dz_init`.

- **`dz_max` and `dz_init`** are informational under a replay. They are recorded in
  `StepLimits`, and the translation refuses a schedule whose longest step exceeds
  `dz_max`, so the recorded limit is true of the run.
- **The controller's `rtol`** feeds only `ControlRecord`'s requested tolerance and the
  spatial collar diagnostic's threshold (`StepDriver.observe`). No step is controlled.

**The propagator mode.** `SimulationSettings.propagator_mode` picks §4.3's treatment.
Under this march the free and planar kinds admit `"tabulate"`, meaning the single volume
of $e^{hD/2}$. They keep refusing it under the Runge–Kutta march, whose six distinct
fractions would need twelve volumes. `"auto"` resolves to `"tabulate"` on the CPU
backends for a constant operator, and to `"recompute"` on a device or a varying operator.
The choice is recorded in the `backend` metadata. The modes agree to rounding, not
bitwise.

**Refused by name**, before allocation, by a new `_admit_low_storage` beside
`_admit_integrator` and by `StepperSettings.__post_init__`: no schedule ("the low-storage
march replays a prescribed schedule; pass `schedule=FixedSchedule.uniform(...)`, or step
adaptively with `RungeKutta()`"); `interpolate_planes`, which a schedule already refuses;
any coupled flow; the TDSE response; a step past §4.6's interval; SEP 21's `"implicit"`
collar scheme should it land; and a backend that does not yet execute the march, naming
one that does.

**Records.** `StepperRecord.method` is the representation above (the next free output
schema at landing, 19 at this writing), and
`tableau` keeps the unused DOPRI5, as for `Multistep`. The step history's `error_norm` is
0.0 on every step, meaning "not estimated" (§6.3). `ControlRecord.met_tolerance` and
`StepSummary.met_tolerance` are then vacuously true, and the "too coarse" warning cannot
fire. The log lines that print a tolerance ratio say "not estimated (replayed schedule)"
instead (§6.3).

**Source evaluations** (four a step) go in the backend metadata under a physical key of
their own, `source_evaluations`, never under `response_calls`. `driver.run._record_execution`
saves `response_calls` as logical calls. On resume, `Simulation` hands those to
`StepDriver.restore_call_ledgers`, which raises "logical response calls require an Adams
run" on any backend without an Adams march. Keeping the count out of `response_calls`
leaves the logical ledger empty, so the restore path passes.

**Tableaux.** `Tableau` refuses an all-zero error-weight vector. Under the adaptive march
such a tableau scores every step zero and grows the step at its cap without end
([stepper.md](../../stepper.md) §10, trap 11). A scheme without an estimate belongs to this
march, which takes no tableau. No shipped tableau is affected.

### 6. Implementation strategies

#### 6.1 Layers

The specification goes in the stepper layer (`solprop/stepper/low_storage.py`). The kind
and its refusals enter translation (`StepperSettings`, `KernelProblem.validate`) and the
driver (settings, admission, records). The march and host engine go in
`solprop/backends/low_storage.py`, with compiled register passes in each backend. No
physics changes ([architecture](../../architecture.md) §3).

#### 6.2 The march

`LowStorageMarch` follows `AdamsMarch`: it owns its registers through an engine, and it
replays only. Its `replay_to` and `replay_step` mirror `AdamsMarch._replay_to` and
`replay_step`. Each step calls the engine's `step(z, dz)`, applies `_finish`, moves the
position inside `committing()`, and calls the driver's `_record`, `_take_step_stats` and
`_natural_stop`, and after the commit the accepted-endpoint hook SEP 21's phase 2 adds
for its accumulators, if that lands first. `StepDriver.propagate_to` gains one branch: a backend holding a march in
the protected attribute `_low_storage`, set at construction as `_adams` is, replays
through it. `snapshot()` returns `None`, as under `RungeKutta()`.

Since the march never rejects, it writes the accepted state before the step completes.
For a replay only, this relaxes the adaptive rule that a trial leaves the accepted state
alone until the commit. The relaxation needs one guard, and it moves one check:

- **A step abandoned in flight.** An exception or signal between pass 1 and the commit
  leaves the state holding neither $\tilde A_n$ nor $\tilde A_{n+1}$. The march marks the
  backend *in flight* from its first write to the commit (a protected flag beside
  `_low_storage`).
- **Where the guard reaches.** Every entry that reads or advances the state refuses an
  in-flight backend with a `BackendError` naming the step. That covers the driver's
  `observe`, `propagate_to` and `march_snapshot`, and each backend's `stats`, through a
  driver check every override calls. A second `propagate_to` after a caught exception
  must not march on from a corrupt state. `LockstepRunner` checks each member in its
  replay loop, its `stats` and `observe_members`. Only `set_initial` clears the flag.
- **What a failure writes.** The run loop's failure path (`solprop.driver.run`) observes
  the backend to write its terminal checkpoint, so it writes none and says why. The file
  ends at its last plane, from which a resume continues to rounding, and `step_planes=k`
  bounds the loss to $k$ steps.
- **The alternatives (Q6).** Holding signals for the whole step keeps the checkpoint, but
  defers a scheduler's stop by up to a step (tens of seconds at the target size on a CPU)
  and does nothing for an exception. A fourth register keeping the state costs 1 F.
- **Divergence** is detected on the state. Pass 4 reduces $\|\tilde A_{n+1}\|^2$ over the
  norm mask, and a non-finite value raises `ConvergenceError` with the replay's message.
  That error is raised in flight, so it too loses the terminal checkpoint. Today's
  replay keeps it, because `StepDriver._replay_to` checks the norm before it writes the
  state. This is a deliberate cost of accepting in place, and the manual says so.

#### 6.3 The error norm on a replay, and the noise

| option | cost | what the norm column then means | verdict |
|---|---|---|---|
| 0.0 | none | "not estimated"; passes `StepHistory`, `StepSummary` and the replay's finiteness test; `met_tolerance` vacuous | **recommended**, with §6.2's state check |
| increment proxy, such as $\|\tfrac h6F_4\|/\|\tilde A\|$ | one reduction | the nonlinear change per step, $O(h)$, not the local error $O(h^5)$; no scaling makes "≤ 1" mean "met the tolerance" | rejected |
| stiffness indicator $h\rho=2\|F_3-F_2\|/\|F_2-g_1\|$ | two reductions; $g_1$, $F_2$ recovered from $C$ and $A$ at no memory, with cancellation where the increment is small | an estimate of $\lvert h\lambda\rvert$ for the dominant linearised source eigenvalue, which §4.6 bounds | not an error norm; logged if it tracks the collar's and plasma's margins (Q1) |
| NaN as "absent" | relaxing `StepHistory`, `StepSummary` and every consumer | conflates absence with divergence | rejected |

The history's five columns, `ControlRecord` and `StepSummary` keep their forms, and the
output schema bump documents the 0.0. Two log lines print the ratio as a number today:
the closing information line of `solprop.log`'s reporter ("achieved/requested tolerance
ratio") and the driver's debug line after each `propagate_to`. Under this march both
print "not estimated (replayed schedule)" instead of 0.0.

**The `modified` noise is admitted.** The draw is one counted volume, as for every march.
Its phase is formed per element in the stage-input pass. Kokkos's free-kind stage input
already does this. `numba_cpu` currently rotates the draw into its `exponent` volume and
fuses the rotation in phase 2. A backend that would need a stored rotated copy on a free
kind refuses the noise by name (Q3).

#### 6.4 The seam

Phase 1 bumps `SCHEMA_VERSION` to the next free version (29 at this writing; SEP 19 took
27 and SEP 25 took 28, which added `_stage_bundles` to `DRIVER_SURFACE`):

- **`StepperSettings.integrator`** becomes `Multistep | LowStorageRungeKutta | None`.
  Widening the field, rather than adding one, makes a backend that misses the kind fail
  loudly where it builds its `AdamsMarch`, instead of stepping the tableau silently.
  Every site of §3's inventory then tests for `Multistep` explicitly.
  - The three `_refuse_adams` functions are reached only for a `Multistep`.
  - `_check_tdse` refuses anything but a `Multistep`, and `_runge_kutta_advice` speaks
    only for one.
  - Once the sites test for `Multistep`, a `numba_cpu` (and a solprop-kokkos) backend
    that has no march of its own must refuse the new kind by name in phases 1 and 2. A
    non-`Multistep` integrator would otherwise read as `None` and run DOPRI5 silently.
- **Two protected attributes**, `_low_storage` and the in-flight flag, join
  `DRIVER_ATTRIBUTES` in `tests/backends/test_driver.py`.
- **`StepDriver._low_storage_engine(method) -> object`** joins `DRIVER_SURFACE`
  (`tests/backends/test_driver.py::test_the_surface_a_backend_subclasses_is_frozen`).
  `None` takes the host engine.
- **The engine protocol, `LowStorageEngine`**, is part of the seam: `step(z, dz)` returns
  the accepted state's squared norm, and `reset()` clears the engine. A later change to it
  bumps the version again.

The host engine reuses `_adams_lab_source` and `_adams_propagator`, already generic in
meaning; renaming them would touch solprop-kokkos and SEP 12's adapter for no behaviour
(Q2). solprop-kokkos follows the version in phase 1, refusing the kind, and implements it
behind the hook in phase 3.

#### 6.5 `numpy_ref` (phase 1)

The host engine is the oracle. $A$ is the state array, updated in place, and $B$ and $C$
are host arrays. Each pass calls `_adams_lab_source` once and takes $P$ and $Q$ from
`_adams_propagator` or the table, allocating transients as the reference always does.
The step's three distinct stage positions are known before its first pass, so the engine
takes their host scalars together through `StepDriver._stage_bundles` (SEP 25 §6.3), and
the hooks' own `_stage_scalars` calls then return the kept bundles.
`reference_base_memory` becomes method-aware: it counts the registers and the table beside
the pipeline rows, and no stage or accumulator rows. `LockstepRunner._replay_to` dispatches
to each member's `replay_step`, as for Adams.

#### 6.6 `numba_cpu` (phase 2)

With the march selected, `_allocate` creates no stage rows, accumulators, accept buffer,
recycled stage, propagator pair or exponent. Each route's stage splits into three parts:
its input formation (the combine kernel, or `_stage_free`'s input), the shared core
(transforms, responses, chunk loop) and its finish. The march adds register versions of
the input, which fuses the noise rotation, and of §4.3's finish, which takes $P$ and $Q$
from the table or from the separable arrays. FFTW runs in place on the transverse scratch.

§2's profile says where the time is to be saved:

- one register read per input, instead of up to five rows combined;
- the propagator formed once while the step length holds, or fused into the finish,
  instead of once per stage;
- no accumulators and no per-trial copy.

What remains is the transform pairs and the response. The acceptance target is §7.4's
in-tree ratio to IP-DOPRI5. The drafting prototype is not a target. Fused kernels agree
across thread counts within a rounding bound fixed before measurement
([SEP 25](sep-0025-rounding-level-reproducibility.md)), with two-stage reductions of
fixed shape. The ledger names the registers, and `estimate_memory` reproduces them to the
byte. Admission then uses this backend's own count rather than the reference's, whose
`base` rows are 0.975–1.046 times `numba_cpu`'s (module docstring of
`solprop.backends.memory`). `_admit_history`'s refusal also names the march when its
estimate fits.

#### 6.7 Kokkos CPU (phase 3)

`KokkosBackend._low_storage_engine` returns a resident engine whose `step` is one native
call. It makes four register passes and accepts in place. Each pass forms the input in
`transverse_` (or the modal fine buffer), transforms in place, runs the existing chunked
core, transforms back, and updates $A$ (`current_`), $B$ and $C$.

With the march selected, `GeneralWorkspace::allocate` creates no `stages_`, `solution_`,
`error_`, `residual_`, `recycled_`, `advanced_` or `whole_propagator_`. The CPU build
tabulates by default. A device forms the flow per element, as `separable_stage_input`
forms its propagator. Kokkos CPU times within 2 % of `numba_cpu` on §2's case, so the same
targets apply.

A signal does not interrupt the native call, but its handler runs at the first bytecode
after the call returns (`solprop._interrupts`), which is before the commit. So the
native step, `_finish` and the position move sit in one `committing()` block. That costs
nothing on Kokkos, since the signal could not be delivered during the call anyway, and it
leaves only a native exception able to abandon a step in flight. The ledger and estimate
agree to the byte, and `scripts/check.sh` passes. A device build
compiles the same passes, one call per step with no host synchronisation inside
([stepper.md](../../stepper.md) §8); its qualification is separate (Q7).

#### 6.8 Memory, time and the flow table

| backend | registers | scratch | flow table | noise | total on the free kinds |
|---|---|---|---|---|---|
| Kokkos CPU, `numba_cpu` | $A$, $B$, $C$ | transverse | +1 by default | +1 | 5 F tabulated, 4 F recomputed; +1 with noise, −1 with Q4's elision |
| Kokkos device | $A$, $B$, $C$ | `transverse_` | recomputed | +1 | 4 F |
| `numpy_ref` | $A$, $B$, $C$ | transverse | +1 | +1 | 5 F counted, plus per-call transients |

Node-chunk buffers stay bounded by the node budget (64 MiB by default,
[free-space model](../../../guide/free-space/model.md)). At $768\times768\times600$ the march
holds 26.4 GiB tabulated or 21.1 GiB recomputed, plus 5.3 GiB with noise. That compares
with 79 or 84 GiB for the Runge–Kutta march and 232 GiB for Adams on Kokkos. At
$1536\times1536\times600$ the march holds 84–105 GiB against 316 GiB.

A step makes four source evaluations against IP-DOPRI5's six with FSAL (seven on
`numpy_ref`), applies four flows from one table against six formed per stage, and reads
one register per input against combinations of up to five rows. §2's drafting
measurement suggests factors of five or more per evaluation. The acceptance target is
§7.4's in-tree ratio: at least 3 times less time per evaluation and at least 4 times less
per step. Fourth order needs more steps than fifth asymptotically, but DOPRI5 measures
about fourth order at fixed step on stiff problems (§4.4). The ranking at equal
Richardson error is therefore measured, not assumed (§7.4). No new dependency is
proposed.

### 7. Verification and validation

#### 7.1 Numerical contract (phase 1 on `numpy_ref`, repeated per backend)

- **Linear limit.** With no response, the march must equal $U(z_N,z_0)\tilde A_0$ within
  §4.5's bound, $\varepsilon\,(4N+c\,(N|D|h+|\Phi(z_N)|))$ per element. Cases: a constant
  modal operator, P2's analytic
  varying operator ([stepper.md](../../stepper.md) §12, where $P\ne Q$) and the Cartesian
  operator. The Cartesian case reproduces the width and Gouy phase of
  `tests/backends/test_free_diffraction.py::test_cartesian_gaussian_width_gouy_exact_diffraction`.
- **Order four.** Three-level `calibrate` ladders on the third-order soliton with its
  phase (P1), on P2, and on a 10–40-step Cartesian diffraction-and-Kerr case on the grids
  of `tests/backends/test_free_ladders.py::test_cartesian_nonlinear_ladder`.
  - **Steps.** The ladder's steps are chosen inside the asymptotic range of §4.4,
    $|\operatorname{Im}D|\,h/2\lesssim1$ in the populated driven bins. The test states
    that bound for its steps, computed from the grid and the populated band.
  - **Comparison.** The same ladder is run with IP-DOPRI5. The march must read four within
    the spread IP-DOPRI5's observed order shows about its own value on that ladder (at
    most 0.3), above the round-off floor.
  - **The soliton** is also compared with its analytic error.
- **The independent RK4IP.** `numpy_ref` multiplies the source by `grids.omega_window`
  (`passes.stage_output`); the scaffold does not. The existing test also agrees only to
  1.7e-3 of the scaffold's truncation error. "Rounding only" therefore has to be
  established first:
  - **One evaluation.** With the window all ones (the default), one source evaluation on
    `numpy_ref` and one in the scaffold must agree within the pass's summation-length
    bound.
  - **Whole runs.** The bound on whole runs is derived from that measured per-evaluation
    difference times the step count, and must sit orders below the truncation error the
    same test reports. The two are not bitwise equal: accumulation and source composition
    differ.
- **Lifecycle and refusals.** `step_planes` planes and a lockstep batch must match the
  uninterrupted run, or the batch's members run alone, on the same edges (exact) and
  within a rounding bound fixed before measurement (ADR-013's batch identity, as SEP 25
  restates it). A resume at a schedule
  edge, including a mid-segment edge under the table, must agree with the uninterrupted
  run within §4.5's rounding bound, as SEP 16 qualified the Adams resume. Each
  refusal of §5 must fire before allocation and name its remedy. An exception injected
  between passes must leave `observe`, `propagate_to`, `stats` and `march_snapshot`
  refusing, the file ending at its last plane, and the resume within the rounding bound.
  A run with an
  in-stage collar must hold no row beyond three registers, $S$ and the table (§4.3's
  exchange).
- **Noise.** With the draw and no response the state must be the pure flow; with Kerr the
  stage inputs must carry the draw at $z_n$, $z_m$ and $z_n+h$, against a hand-formed one.

#### 7.2 Parity (phases 2 and 3)

`numba_cpu` and Kokkos CPU against `numpy_ref`, on one pass, one step and ten steps. Cases:
the soliton, a varying modal case with Raman and plasma, and Cartesian and line Kerr cases
with the in-stage collar and noise. Deviations are peak-normalised and bounded by the
summation length ([testing](../../testing.md) §3). Repeats and runs at different thread
counts must agree within a rounding bound fixed before measurement.

#### 7.3 Memory

Each ledger must name the registers and the table. On a Cartesian case its field-sized
rows must total at most 5 F tabulated or 4 F recomputed, one more with noise. The stage,
accumulator, accept and recycled rows must be zero, and the propagator row too when
recomputing. `estimate_memory` must equal the ledger to the byte, as
`tests/backends/test_kokkos_dispersive_routes.py::test_the_estimate_is_the_ledger_to_the_byte`
checks for its own rows. A run that `_admit_history` refuses under Adams, but that fits
under the march, must be admitted with it.

#### 7.4 Benchmarks

Ten-step windows on `numba_cpu` and Kokkos CPU, against IP-DOPRI5 replaying the same
schedule, under the ledger's rules ([benchmarks](../../benchmarks.md) §2: machine record,
quiet machine, same-process A/B comparisons labelled):

- **At equal step.** On §2's case ($192\times192\times300$, Kerr, no collar, 0.5 µm) and on
  a larger Cartesian Kerr beam. Each run records time per source evaluation and per step,
  for both flow treatments, with the peak ledger, on one thread and on all threads.
- **The acceptance target** is in-tree. On one thread of a quiet machine, at equal step,
  the march takes **at least 3 times less time per source evaluation, and at least 4 times
  less per step**, than IP-DOPRI5 on the same backend. §2's drafting measurement motivates
  the target and does not count towards it.
- **At equal Richardson error.** `calibrate` runs both methods on one window. `replay`
  takes each to a common target on the weak observable and on the field. The report gives
  steps per metre times time per step.
- **A stage profile** (input, transforms, response, finish, flow), beside §2's DOPRI5
  breakdown.

Points below the reference floor are not ranked. Where IP-DOPRI5 wins in time, the result
is documented, and the memory saving still stands.

### 8. Documentation

- **[Choosing a stepper](../../../guide/numerics/choosing-a-stepper.md).** When to take
  `LowStorageRungeKutta()`, what it costs and what it refuses. Its memory table gains the
  march's row, and its free-kind figures are corrected: 14, 30 and 43 copies against the
  ledgers' 15 or 16, 32 and 44.
- **[Fixed schedules](../../../guide/numerics/fixed-schedules.md).** Replaying with the
  march; calibrating on the observable rather than `rtol`; a weak observable's floor from
  backend parity; `calibrate`'s `requested` for a uniform schedule; the order gate at four.
- **Elsewhere:** [the solver](../../../guide/numerics/solver.md) (three integrators),
  [stepper.md](../../stepper.md) §2 and §8, the [kernel contract](../../kernel-contract.md) §7
  and schema history, [backends](../../backends.md) §8–§9, [architecture](../../architecture.md)
  §8; an ADR in phase 1 (replay-only, accept in place, no estimate, the 0.0 norm, the
  in-flight guard); docstrings; and a `CHANGELOG.md` line citing this SEP at Final.
- **A tutorial in `docs/tutorials/free_space/`.** A line or small Cartesian run with a weak
  band under a strong pump: a uniform schedule, a three-level `calibrate` with a parity
  floor, `replay` to a target, and the whole-field norm's blindness beside it, in seconds.

### 9. Compatibility and impact

No default changes, and the existing marches are untouched; existing runs take the same
code path as before (an exact no-op contract), since backends allocate their usual buffers
unless the new kind is selected.
In phase 1 the kernel schema moves to its next free version (28 → 29 at this writing),
with `DRIVER_SURFACE` and its literal test, and solprop-kokkos follows on the same
revision. The output schema moves to its next free version (18 → 19 at this writing):
`StepperRecord.method` may name the march, whose runs record 0.0 norms meaning "not
estimated", which older records never do. `Tableau` refuses all-zero error weights (no
shipped tableau has them), and the free and planar kinds' refusal of `"tabulate"` is
lifted for this march only. A resumed run of the march agrees with the uninterrupted one
to rounding, not bit for bit (§5), the same resume semantics SEP 14 and the Adams march
already have. Statements found stale while drafting are corrected in phase 1:

- **ADR-009** gets an amendment note on "never materialised beyond what the next stage
  needs" (§3).
- **The kernel contract's §7 `integrator` row** still describes SEP 13's
  interaction-picture method on mode sets and says it "cannot replay a schedule".
  `AdamsMarch._replay_to` and `replay_step` have replayed since SEP 16.
- **[backends](../../backends.md) §9's Adams row** says "every route but a coupled flow" on
  the compiled backends and calls devices "not qualified". The
  [Adams limitations](../../../guide/numerics/adams.md#limitations) say otherwise: the
  exponential formulation runs beside constant-Schur and frozen-residual flows, and SEP 16
  qualified devices.
- **solprop-kokkos's `README.md`** names kernel schema 25, where `build_info()` reports 28,
  and says the free-space routes refuse Adams.

### 10. Alternatives considered

- **RK4 as a `Tableau` under the existing march.** It takes Kokkos from 15 F to 12 F,
  still three times the target, and its zero error vector fails silently under adaptive
  control. Rejected and guarded against (§5).
- **A low-storage Runge–Kutta march.** DOPRI5's matrix, embedded pair and dense output need
  the stage rows. Reorganising the default's arithmetic would change every recorded state
  and require its recapture. Rejected.
- **Carpenter and Kennedy's five-stage 2N scheme [4].** It needs three volumes ($u$,
  $\Delta$, $S$) whatever the passes, against RK4's four under a two-pass pipeline. It
  costs 25 % more evaluations, and inverse flows at its inner nodes. Kept as a fallback
  (Q4).
- **The host march on every backend.** On `numba_cpu` the existing hooks write stage row 0,
  use the propagator volumes and return copies. On Kokkos every pass would cross the
  host–device boundary twice. Used on `numpy_ref` only.
- **An embedded pair at once.** It needs rollback and an estimate register (5–6 F), and a
  qualification the replay use does not need. Deferred to phase 4.
- **Symmetric split-step.** It takes two volumes but is second order, and it needs a
  closed-form nonlinear flow that shock and plasma terms lack
  ([stepper.md](../../stepper.md) §2). Rejected.
- **Lower-order 2N schemes [3].** They save a volume but need several times the steps for a
  weak observable. Rejected.
- **Exponential Runge–Kutta methods [9].** They suit driven bins, but need
  $\varphi$-function weights and more registers. Out of scope.

### 11. Phases

Each phase merges independently with its tests, documentation and report. The full gate
runs once per phase, and the end-of-SEP lanes once before Final.

1. **Specification, seam and `numpy_ref`.** Delivers `LowStorageRungeKutta`;
   `solprop.backends.low_storage` (march, engine protocol, host engine); translation,
   admission, refusals and the in-flight guard; both flow treatments; the kernel schema bump,
   followed by solprop-kokkos with a refusal by name; the output schema bump; lockstep
   members;
   the `Tableau` guard; `numpy_ref`'s ledger and estimate; the ADR, manual sections and
   tutorial; the §9 corrections. Verifies §7.1 and measures Q1's indicator.
2. **`numba_cpu`.** Delivers register kernels on every admitted route with no Runge–Kutta
   buffers; the table and the per-element flow; the fused noise rotation; the ledger and
   an exact estimate; `backend="auto"` per problem; the admission message. Verifies §7.2
   (thread counts within a rounding bound), §7.3 and §7.4. Decides Q4.
3. **Kokkos CPU.** Delivers §6.7's resident engine with no Runge–Kutta views, its ledger
   and estimate, and a passing `scripts/check.sh`. Verifies §7.2–§7.4 on the CPU build.
   Decides Q5.
4. **Optional: an embedded pair.** Starts if phases 2 and 3 show the march competitive and
   an adaptive use is wanted. It chooses the pair (Q8), adds adaptive control on a fifth or
   sixth volume, and qualifies it against the adaptive Runge–Kutta march, as SEP 16 §7.3
   qualified Adams.

### 12. Open questions

- **Q1 — phase 1: the norm column.** Record 0.0 (recommended); log or warn on §6.3's
  stiffness indicator if it tracks the collar's and plasma's margins, never in the column.
- **Q2 — deferred: hook names.** Rename `_adams_lab_source` and `_adams_propagator` with
  the next seam change touching solprop-kokkos, not here. Destination: a roadmap item
  (*Performance and backends*), taken up by the next seam-changing SEP.
- **Q3 — decided here, for review: the noise.** Admitted, draw counted, phase formed per
  element; a backend that would store a rotated copy refuses it by name.
- **Q4 — phases 2–3: the stage register.** Drop $C$ when the pipeline reads its input once,
  at the cost of a second finish variant per route. If instead the fourth register binds
  under a collar, add Carpenter and Kennedy's 2N scheme as a second `scheme`.
- **Q5 — phase 3: Kokkos's stage.** Can the separable stage, which writes `stages_`,
  `solution_` and `error_`, be split into shared input, core and finish without moving the
  Runge–Kutta arithmetic? If not, the march takes a parallel path sharing the chunked core.
- **Q6 — phase 1: in flight.** Choose between the invalid-state guard (recommended),
  holding signals per step, and a fourth register (`in_place=False`, 1 F).
- **Q7 — after phase 3: devices.** Qualification follows [backends](../../backends.md)'
  standing policy, is not required for Final, and until done a device refuses the kind.
  Destination: the roadmap's device-qualification items (*Performance and backends*),
  where a line for this march is added at Final.
- **Q8 — phase 4: the pair.** Start with ERK4(3)-IP [7], which keeps the phase 1 stencil,
  kernels and checks, once its coefficients, stages and FSAL structure are confirmed
  from [7]. Take 2R [5] or 3S\* [6, 10] if one register decides; their inner nodes need
  inverse flows and fresh qualification.
- **Q9 — beyond this SEP: transfer.** Does a schedule calibrated on a coarser transverse
  grid transfer to the production grid? Part of [stepper.md](../../stepper.md) §9's question.

### 13. References

1. J. D. Lawson, "Generalized Runge–Kutta processes for stable systems with large
   Lipschitz constants," *SIAM Journal on Numerical Analysis* **4**, 372–380 (1967).
   [doi:10.1137/0704033](https://doi.org/10.1137/0704033)
2. J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for simulating
   supercontinuum generation in optical fibers," *Journal of Lightwave Technology*
   **25**, 3770–3775 (2007).
   [doi:10.1109/JLT.2007.909373](https://doi.org/10.1109/JLT.2007.909373)
3. J. H. Williamson, "Low-storage Runge–Kutta schemes," *Journal of Computational
   Physics* **35**, 48–56 (1980).
   [doi:10.1016/0021-9991(80)90033-9](https://doi.org/10.1016/0021-9991(80)90033-9)
4. M. H. Carpenter and C. A. Kennedy, "Fourth-order 2N-storage Runge–Kutta schemes,"
   NASA Technical Memorandum 109112, NASA Langley Research Center (1994). No DOI;
   [NASA Technical Reports Server 19940028444](https://ntrs.nasa.gov/citations/19940028444).
5. C. A. Kennedy, M. H. Carpenter and R. M. Lewis, "Low-storage, explicit Runge–Kutta
   schemes for the compressible Navier–Stokes equations," *Applied Numerical Mathematics*
   **35**, 177–219 (2000).
   [doi:10.1016/S0168-9274(99)00141-5](https://doi.org/10.1016/S0168-9274(99)00141-5)
6. D. I. Ketcheson, "Runge–Kutta methods with minimum storage implementations,"
   *Journal of Computational Physics* **229**, 1763–1773 (2010).
   [doi:10.1016/j.jcp.2009.11.006](https://doi.org/10.1016/j.jcp.2009.11.006)
7. S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for step-size control in the
   interaction picture method," *Computer Physics Communications* **184**, 1211–1219
   (2013). [doi:10.1016/j.cpc.2012.12.020](https://doi.org/10.1016/j.cpc.2012.12.020)
8. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd ed., Springer (1993), §§II.1–II.2.
   [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
9. M. Hochbruck and A. Ostermann, "Exponential integrators," *Acta Numerica* **19**,
   209–286 (2010). [doi:10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048)
10. H. Ranocha, L. Dalcin, M. Parsani and D. I. Ketcheson, "Optimized Runge–Kutta methods
    with automatic step size control for compressible computational fluid dynamics,"
    *Communications on Applied Mathematics and Computation* **4**, 1191–1228 (2022).
    [doi:10.1007/s42967-021-00159-w](https://doi.org/10.1007/s42967-021-00159-w)

[1, 2] give the transformation and stencil; [3–6, 10] the low-storage families (counts
assume an in-place right-hand side, §4.3); [7] the IP pair; [8] RK4's order; [9] the
stiff-order caveat. None certifies the march on a given problem: §4.7's calibration does.

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

### Phase 1 report — specification, seam and `numpy_ref`

- **Scope:** phase 1 of §11, 2026-10-18, Opus 5.5 (AI). Branch
  `feature/sep-0024-phase-1` in solprop (from `main` 62b0250) and in solprop-kokkos
  (from its `main` 57f688c); branch heads at the merge: the commit carrying this report
  (solprop) and 24f6a4e (solprop-kokkos), merged into both `main`s on 2026-10-18.
- **Delivered:**
  - The specification: `src/solprop/stepper/low_storage.py` (`LowStorageRungeKutta`,
    `LOW_STORAGE_REAL_INTERVAL` 2.785), exported from `solprop.stepper`; the `Tableau`
    guard against an all-zero error-weight vector (`src/solprop/stepper/tableau.py`).
  - The march and host engine: `src/solprop/backends/low_storage.py`
    (`LowStorageMarch`, `LowStorageEngine`, `HostLowStorage`).
  - The seam: `src/solprop/backends/driver.py` (`_low_storage`, `_in_flight`,
    `_low_storage_engine`, the `propagate_to` branch, the guard in `_require_state` and
    `march_snapshot`, `_stage_bundles` keyed by anchor and position, the "auto"
    propagator rule, `source_evaluations`, the "not estimated" debug line);
    `src/solprop/translate/problem.py` (kernel `SCHEMA_VERSION` 29, the widened
    `StepperSettings.integrator`, `_check_low_storage`, `_check_integrator`, the
    tabulate refusal lifted, `_check_tdse` and `_runge_kutta_advice` testing for
    `Multistep`); `src/solprop/translate/tdse_source.py`,
    `src/solprop/backends/exponential.py`, `src/solprop/backends/memory.py`
    (method-aware `reference_base_memory`, `adams_memory` and the estimate's propagator
    rule), `src/solprop/backends/numpy_ref/backend.py` (construction, `_refuse_adams`),
    `src/solprop/backends/numba_cpu/backend.py` (refusal by name),
    `src/solprop/backends/batch.py` (lockstep dispatch); solprop-kokkos
    `src/bindings.cpp` (schema 29), `src/solprop_kokkos/backend.py` (refusal by name),
    `src/solprop_kokkos/__init__.py`, `README.md`, `CHANGELOG.md`.
  - Driver and records: `src/solprop/driver/simulation.py` (`_admit_low_storage`,
    `LOW_STORAGE_BACKENDS`, the integrator mapping, widened annotations),
    `src/solprop/driver/interface.py`, `src/solprop/driver/run.py` (no checkpoint from
    a step in flight, the reporter's `estimated`), `src/solprop/log.py`
    (`NOT_ESTIMATED`, `estimated=`), `src/solprop/output/schema.py` (output
    `SCHEMA_VERSION` 19), `src/solprop/driver/scaffold.py` (`gnlse_source`).
  - Tests: `tests/backends/test_low_storage.py` (§7.1 and §7.3 on `numpy_ref`),
    `tests/stepper/test_low_storage.py`, `tests/driver/test_low_storage_settings.py`;
    solprop-kokkos `tests/test_backend.py::test_the_low_storage_march_is_refused_by_name`;
    the frozen literals of `tests/backends/test_driver.py`; schema pins rewritten in
    `tests/translate/test_problem.py`, `tests/translate/test_dispersive_kerr_terms.py`,
    `tests/backends/test_mixture_channels.py`,
    `tests/backends/test_numpy_ref_backend.py`, `tests/test_display.py` and sixteen
    output, driver and processing modules (output schema 18 → 19); `tests/bounds.py`
    (`step_evaluations` method-aware); solprop-kokkos `tests/test_api_contract.py`.
  - Documentation: `docs/guide/numerics/choosing-a-stepper.md` (the march's section,
    the memory table, the free-kind figures corrected to 15 or 16, 32 and 44),
    `docs/guide/numerics/fixed-schedules.md` (the march's section and a running
    example), `docs/guide/numerics/solver.md` (three integrators), `docs/dev/stepper.md`
    §2 and §8, `docs/dev/kernel-contract.md` (§7, the schema history, the header's
    stale "27"), `docs/dev/backends.md` §8 and §9, `docs/dev/architecture.md` §8 and
    §13, `docs/dev/decisions.md` (ADR-050; the ADR-009 and ADR-047 notes), the
    tutorial `docs/tutorials/free_space/plot_05_low_storage.py`, `CHANGELOG.md`.
- **Deviations from the proposal:**
  1. **The in-stage exchange is not separate on `numpy_ref`.** §6.5 has each pass call
     `_adams_lab_source` once; that hook returns the material response and the
     absorber term summed, through the reference's own evaluation and its transverse
     scratch, so the host engine folds the sum. §4.3's two evaluations per pass belong
     to the compiled stages (phases 2–3), which hold the material output in a stage row
     today. The collar case's ledger holds three registers, `S` and the table (tested).
  2. **`_stage_bundles` keeps several anchors.** The hooks read each position as its
     own anchor (the lab source at `(p, p)`, the flows at `(z_n, z_m)` and
     `(z_m, z_n + h)`), so the kept bundles are keyed by anchor and position until the
     next hoist; the engine takes its three anchors' bundles, one call each. Recorded
     in the kernel contract's row 29.
  3. **The engine protocol carries `stiffness`** (Q1, below), beside `step` and `reset`.
  4. **The flag is set from the step's start**, by the march before it calls the
     engine, not at the engine's first write: engine-agnostic and conservative (an
     exception in pass 1's evaluation also leaves the backend refusing).
  5. **The failure path records the outcome without a checkpoint.** It writes
     `failed` at the last saved plane with a message naming the abandoned step, adds
     the note to the exception and logs it.
  6. **The reporter's closing line** needed to know the run has no estimate:
     `LogReporter`, `BarReporter`, `default_reporter` and `resolve_reporter` take
     `estimated` (default `True`), which `run_problem` sets for this march.
  7. **A varying operator under `"tabulate"`** holds the step's `P` and `Q` (two
     volumes in the ledger), §4.3's "one or two flow volumes per step"; the table reused
     across steps is a constant operator's alone.
  8. **The table is built from the step alone**, as `U(o + h/2, o)` from the
     operator's origin `o` (where a constant operator's increment is the half step
     itself), not as `U(z_n + h/2, z_n)`: `fl(z_n + h/2) - z_n` is off by up to half an
     ulp of the position, an error that the reused table carries into every step and
     that accumulates, where the recomputed flows' differences telescope. Built this
     way, a reused flow is off by at most `|D| |h - h_t| / 2 <= 4 eps |D| h` within the
     slack, `8 eps |D| h` a step: twice §4.3's "below `8 eps |D| h / 2`", which counts
     one flow; the linear-limit test's `c = 16` covers it (the worst element fell from
     0.20 to 0.11 of the bound with the change).
  9. **The `dz_max` refusal** is applied under this march only, where §5 states it,
     with a slack of four ulps of the run's positions (a schedule's steps are
     differences of rounded positions).
  10. **`gnlse_source`** is a new public function of `solprop.driver.scaffold`, so that
     one evaluation can be compared with the scaffold's (§7.1).
  11. **The order comparison rule** of §7.1 is read as: the march must read four within
      `min(0.3, |p_DOPRI5 - 5|)`. The Cartesian ladder uses `test_free_ladders`' grids
      with a 0.2 µJ beam over 1 cm (0.4 Rayleigh lengths): that test's 0.1 µJ over
      100 µm has a Kerr phase of about 1e-5 rad, whose truncation error sits below
      rounding.
  12. **What the reference holds.** The ledger is the reference's peak per step, from
      `reference_base_memory`, and the test also checks the arrays the backend and its
      engine actually hold after a run: the state, the two registers and the table, and
      nothing else of a state's size (`HostLowStorage.held` names the engine's). To make
      that so, the lab-frame source's unit propagator became a one-element view broadcast
      to the state's shape (it was a full array of ones, held by exponential Adams runs
      too), `numpy_ref` keeps its last propagator only for an Adams march, and `close()`
      releases the engine's registers and table (`LowStorageEngine.close`, added to the
      protocol). The transverse scratch `S` is a per-call allocation on `numpy_ref`,
      counted at its peak.
  13. **`source_evaluations` counts committed steps:** four a step, added in the commit,
      so a step abandoned in flight or diverged adds none.
  14. **A lockstep runner refuses before stepping any member** when one is in flight, so
      the members ahead of it are not advanced past the itinerary; its closing debug line
      keys "not estimated" on the run's integrator, since the runner holds no march.
- **Decisions:**
  - **Q1 (the norm column):** 0.0, "not estimated"; the two log lines print "not
    estimated (replayed schedule)". The indicator `h rho = 2 ||F3 - F2|| / ||F2 - g1||`
    is formed by the host engine from the step's slopes (no recovery from the
    registers: the oracle holds them as transients) and kept as
    `LowStorageEngine.stiffness`, never in the column. Measured: it is `|h lambda|`
    exactly on `u' = lambda u`; on an ionising argon capillary (10 fs, 75 µm, 1 bar, 60
    steps over 5 cm, split-step collar) it read 2.07 on the first step at 110 µJ, and
    the run survived, and 3.28 at 120 µJ, the step before the replay diverged; beside
    an in-stage collar it read 0.17 (pulse at the centre) to 0.65 (pulse in the collar)
    of `h alpha_max / 2`, while a power iteration on the source Jacobian found the
    collar's `h alpha_max / 2 = 2.09` exactly. It tracks the margin of the content the
    step integrates, and so the plasma's, not the worst case the translation's collar
    refusal bounds. Decision: the march **logs a warning**, once per replay, when it
    passes 2.785.
  - **Q3 (the noise):** admitted; the draw is one counted volume; on `numpy_ref` its
    phase is formed per pass inside `_adams_lab_source` and no rotated copy is held.
    The compiled backends refuse the whole march in this phase, so no noise-specific
    refusal is needed yet.
  - **Q6 (in flight):** the invalid-state guard, as recommended: `BackendError` naming
    the step from `observe`, `propagate_to`, `march_snapshot` and every backend's
    `stats` (through `_require_state`), and from a lockstep runner's replay,
    statistics and observations through its members; only `set_initial` clears it.
  - **The "auto" flow** resolves to `"tabulate"` for a constant operator and
    `"recompute"` for a varying one on every backend that runs the march (only
    `numpy_ref`); the device rule waits for phase 3.
  - **ADR-050** records replay-only, accept in place, no estimate, the 0.0 norm and
    the guard; ADR-009's stage clause and ADR-047's premise carry notes.
- **Verification** (on the branch tree; the full gate, the slow lane and the coverage
  lane were not run; they run once, at the end of the SEP):
  - `tests/backends/test_low_storage.py` (45 fast and medium tests and the slow P1
    ladder, run by path with the marker expression lifted), the two other new modules
    and a targeted set of 63 modules touching the changed code under xdist: 3095 tests,
    10 failures on two pinned version strings, fixed and rerun (430 passed).
  - **Linear limit** (§4.5's bound with `c = 16`, fixed before measuring): worst
    element at 0.11 of its bound tabulated and 0.12 recomputed on the constant modal
    operator (200 steps, `|Im D| h / 2` up to 31.5), 0.013 on P2's varying operator;
    the Cartesian beam within `test_free_diffraction`'s own width, Gouy-phase and
    amplitude bounds.
  - **Order four**, three-level `calibrate` with a floor (two flow treatments' difference,
    at least `4 n eps`): P1 3.96 at 200/400/800 steps, IP-DOPRI5 5.08,
    `|Im D| h / 2 <= 0.80` over the bins holding 1 % of the run's peak amplitude, and the
    Richardson estimate the analytic error to 0.03 %; P2 3.96 at 40/80/160,
    IP-DOPRI5 6.24, `<= 0.53`; the Cartesian beam 3.98 at 10/20/40, IP-DOPRI5 4.55,
    `0.14`. At a base of 100 steps P1 reads 3.57 with `|Im D| h / 2 = 1.6`: §4.4's range,
    observed.
  - **Independent RK4IP:** one evaluation equal to the scaffold's (asserted within
    `fft_bound`); 400 steps of the N = 3 soliton 7.0e-12 apart against a derived bound of
    9.8e-7 and a truncation error of 4.93e-4.
  - **Lifecycle:** `step_planes` stops on the schedule's 7th, 14th, ... edges; resumes
    at a plane and at a mid-segment edge differ by 0 (the same table, by coincidence of
    the steps) and 2.3e-16 against a bound of 1.5e-12, asserted as a bound; a lockstep
    batch matches its members, and a member left in flight stops the next step before
    any member moves; every refusal fires through its entry point (the translation, or
    the settings), naming its remedy -- the TDSE one through `validate`, the spatial
    collar's 2.785 through `assemble`, the first admitted schedule against the one
    before it, which DOPRI5 admits; an injected exception leaves the four readers
    refusing, the file ending at its last plane with no checkpoint and a failed outcome,
    and the resume within bound.
  - **Memory:** a Cartesian ledger of five field volumes tabulated and four recomputed,
    six with the noise, matched by the arrays actually held (the state, the registers,
    the table) and released by `close()`; no stage, accumulator or propagator-pair row;
    the estimate
    equal to the ledger to the byte (the noise, loaded after construction, is named
    among what it does not count); a run `_admit_history` refuses under Adams admitted.
  - Lint: `ruff check`, `ruff format`, `basedpyright` (no errors) and `numpydoc lint` on
    the changed modules; `codespell`, `git diff --check`, `tests/test_seps.py`.
  - solprop-kokkos `scripts/check.sh` (with `SOLPROP_KOKKOS_SOLPROP_SOURCE` at this
    branch): format, lint, REUSE, pyright, clang-format, the tidy and sanitizer native
    builds and tests, 314 Python tests passed and 5 skipped, and the distribution wheel's
    smoke test, all green. The wheel built from the kokkos branch is installed in the
    solprop worktree's environment and reports schema 29.
  - `poe docs` (warnings as errors): build succeeded; the tutorial runs in about 1.1 s in the
    gallery.
  - Not run: the full gate, the slow lane beyond the P1 ladder, the coverage lane, GPU
    tests, benchmarks (§7.4 is phases 2–3); no backend but `numpy_ref` executes the
    march, so §7.2 does not apply yet.
- **Gate (at the merge, 2026-10-18):** after an independent numerics review (no
  blockers; six should-fix items resolved: the flow table built from the step alone, the
  lockstep pre-check, the held arrays named and released, the batch debug line, the
  evaluation count on commit, the refusals tested through their entry points), the lane
  set ran on a Linux cluster node (32 cores, GCC 16 toolchain, the Kokkos wheel built
  from the solprop-kokkos branch installed): the full check sequence (lint, format,
  types, docstrings, spelling, licences, then 15 558 tests passed), the coverage lane at
  100 % (15 558 passed) and the slow lane (128 and 1 passed); the Kokkos-selected lanes
  on the pre-review head (the default suite and the slow lane, 146 passed), which no
  later commit's Kokkos path touches; the documentation build on the final head. The
  lanes found, and the merge fixed: the kernel schema in two translation doctests, the
  SEP 12 harness admitting output schemas up to 18, the committed structure figures
  without the two new modules, a step-limits test reading an Adams field off the widened
  integrator type, and the formatter's rewrite of the fixed-schedules snippet. One
  pre-existing margin, unrelated to this phase and failing identically on the unchanged
  `main` on that node, was widened with its measurement recorded:
  `test_every_statistic_matches_the_reference`'s sampled response bound.
- **Sources checked:** the interval 2.785 against RK4's stability function (root
  −2.7853; the imaginary interval `2 sqrt 2`), and the scheme against `R(h lambda)` on
  a linear source. Lawson [1] and Hult [2] are cited as the SEP cites them; the papers
  were not re-read in this phase.
- **Limitations:** `numpy_ref` only; the device rule for `"auto"` is not implemented;
  the table's slack (Q10) makes the table rebuild often far from the origin; an
  exception in a step's first evaluation also loses the checkpoint; no benchmark.
- **Open questions:**
  - **Q10 — phase 2: the table's slack.** A uniform schedule's steps differ by an ulp of
    the *position*, which far from the origin is more than `LANDING_ULPS` ulps of the
    step: on the lifecycle case (`z / h` up to 40) the table was rebuilt 5 times in a
    20-step segment, and at `z / h` in the thousands it would be rebuilt nearly every
    step. A slack in ulps of the position keeps one table but lets the phase drift by
    up to `eps |D| z` a step; keying on the segment's mean step would not drift.
    Phase 2, which times the table, to choose.
  - **Q11 — phases 2–3: the indicator on a resident engine.** Form `h rho` natively
    (two reductions, `g1` and `F2` recovered from `C` and `A` with the cancellation §6.3
    names) or report NaN, which silences the warning.
  - **Q12 — the merge: resolved.** This phase merged first, so it took output schema 19
    (kernel schema 29) and SEP 22's phase 1 takes 20; ADR-050's number stands.
  - **Q13 — phase 2: `_fsal_ready`.** `numba_cpu`'s first-same-as-last switch still
    tests the integrator against `None`, which is right while the Runge–Kutta march is
    the only one that recycles a stage; the low-storage march must keep it off when
    phase 2 runs it on that backend.
- **Handoff:** phase 2 replaces `numba_cpu`'s refusal (`_refuse_adams`) with
  `_low_storage_engine` returning a resident engine, adds it to
  `LOW_STORAGE_BACKENDS` with `backend="auto"` per problem, names the registers and
  the table in its ledger (the rows of `reference_base_memory`'s `_stepping_rows`) with
  an exact estimate, splits `_stage_free` for §4.3's exchange, and names the march in
  `_admit_history`'s refusal. `tests/backends/test_low_storage.py` runs on
  `numpy_ref`; its cases are the parity cases to parametrise. The output schema
  version enters the configuration hash, so any digest of a configuration recorded at
  schema 18 or 19 differs from this branch's.

### Phase 2 report — `numba_cpu`

- **Scope:** phase 2 of §11, 2026-10-19, Opus 5.5 (AI). Branch
  `feature/sep-0024-phase-2` in solprop from `main` 8fc5e80 (kernel schema 31, output
  schema 20, with SEP 22 phase 2); no solprop-kokkos change and no schema change.
  Merged into `main` on 2026-10-19 by the merge commit that follows the commit
  carrying this report.
- **Delivered:**
  - The resident engine and register kernels:
    `src/solprop/backends/numba_cpu/low_storage.py` (`NumbaLowStorage`; the modal,
    Cartesian and planar inputs, finishes and flow tables; `exponentiate_forward`);
    the fold of §4.3's table as one body, `low_storage_fold`, with `FOLD_COMPLETE`,
    `FOLD_EXCHANGE` and `FOLD_SECOND` (`src/solprop/backends/bodies.py`).
  - `src/solprop/backends/numba_cpu/backend.py`: the march built before the buffers,
    `_low_storage_engine`, no Runge–Kutta buffer under it (`_allocate_runge_kutta`),
    the register rows in `memory_ledger`, the Cartesian and planar cores factored out
    of the Runge–Kutta stages and shared (`_free_pass`, `_slab_pass`), the noise
    phase of an observation in a transient, `_refuse_adams` no longer refusing the
    march, `_fsal` kept off for it (Q13, commented).
  - The table key (Q10) on both engines: `table_steps` and `schedule_key` in
    `src/solprop/backends/low_storage.py`, `HostLowStorage` keyed on them.
  - Memory: `low_storage_stage_register` and `low_storage_memory` and the estimate's
    elided register (`src/solprop/backends/memory.py`); the admission names the march
    (`src/solprop/driver/memory.py`); `LOW_STORAGE_BACKENDS = ("numba_cpu",
    "numpy_ref")`, `"auto"` resolving to `numba_cpu` (`src/solprop/driver/simulation.py`).
  - Tests: `tests/backends/test_numba_low_storage.py` (parity on every route, threads,
    ledger, noise, held plane and resume, the guard, the estimate),
    `tests/backends/test_numba_low_storage_kernels.py` (every kernel against the table,
    added to `poe cov-kernels`), updates to `tests/backends/test_low_storage.py` and
    `tests/driver/test_low_storage_settings.py`.
  - Benchmarks: `benchmarks/sep24/phase2_costs.py`; the ledger entry in
    `docs/dev/benchmarks.md`.
  - Documentation: `docs/guide/numerics/choosing-a-stepper.md`,
    `docs/guide/numerics/fixed-schedules.md`, `docs/dev/stepper.md`,
    `docs/dev/backends.md` (§9's table), the structure figure (`layers.svg`,
    `docs/dev/figures/build.py`), `CHANGELOG.md`.
- **Deviations from the proposal:**
  1. **The in-tree speed target is not met.** §7.4 asks for at least 3 times less time
     per evaluation and 4 times less per step than the IP-DOPRI5 replay on one thread;
     measured 1.22 and 1.83 (shape A), 1.18 and 1.77 on all threads, and 1.15 and 1.73
     at shape B. The target was set against a replay that SEP 25 has since made about
     six times faster per evaluation (§2's 1.0 s against 0.176 s now); 69 % of a march
     step is the evaluation's core, which both marches share. The memory saving stands:
     4 volumes against 16.
  2. **The register input of passes 2 to 4 is fused into the finish before it** where
     the stage register is elided (Q4, below), so those passes have no input traversal.
  3. **`backend="auto"` is not chosen per problem.** The settings resolve the backend
     before any problem exists, as for every integrator; `numba_cpu` runs every route the
     translation admits the march on, so `"auto"` resolves to it.
  4. **The exact estimate covers the march's rows.** The registers, the table, the state
     and the transverse scratch agree with `numba_cpu`'s ledger to the byte (tested on
     every flow treatment, with and without the collar); the chunk-sized rest of `base`
     stays the reference's count, as for every march (module docstring of
     `solprop.backends.memory`).
  5. **The modal routes' recomputed flows are transients**, the step's `P` and `Q` formed
     from `_delta_phi`, a modal state's size; the separable kinds form them per element.
  6. **The stiffness indicator is NaN on this backend** (Q11, below), so the march's
     warning is `numpy_ref`'s alone.
- **Decisions:**
  - **Q4 (the stage register):** elided on the Cartesian and line routes where a pass
    reads its input once (a response and no in-stage absorber): the finish of each of
    the first three passes writes the next input, noise included, into the transverse
    scratch. It saves one volume (4 tabulated, 3 recomputed, against 5 and 4) and the
    step's input traversals (the input fell from 13 % to 3.6 % of a step, 0.591 to
    0.546 s at shape A, one thread). Held under an in-stage absorber, where the exchange
    re-forms the input from `C`, and on the modal and planar routes, whose inputs are
    a padded fine spectrum or a column volume. The 2N scheme is not needed: `C` does
    not bind.
  - **Q10 (the table's slack):** keyed on the mean step of each run of equal steps,
    `table_steps`, on both engines; a run of one step keys on its own value. Ten steps
    at 1 m from the origin, where a uniform schedule's steps take two values 2.1 × 10⁶
    ulps of the step apart: one build against seven for a per-step key, 0.159 against
    0.161 s a step (within the spread). Chosen because the phase the table applies over
    a run telescopes to the run's own, where a slack in position ulps drifts.
  - **Q11 (the indicator on a resident engine):** NaN. Recovered from the registers,
    `2 (C - A) / h`, it agreed with the exact slopes' to 1e-7 relative at 0.98 and 98 W
    on a soliton and departed only on nearly linear steps (3.7 × 10⁻³ against
    2.6 × 10⁻¹⁰ at 1 µW, far below 2.785); but on the Cartesian and line routes `C` is
    elided or the slope is split across the exchange's two evaluations, and recovering
    it there would take a volume.
  - **Q13 (`_fsal`):** off for the march, as the `None` test already made it; tested.
  - No new ADR: the decisions refine ADR-050's engine, which stands.
- **Verification** (on the branch; the full gate, the slow lane and the coverage lane
  not run):
  - `tests/backends/test_numba_low_storage.py` (47) and
    `tests/backends/test_numba_low_storage_kernels.py` (21, compiled and with
    `NUMBA_DISABLE_JIT=1`), `tests/backends/test_low_storage.py`,
    `tests/driver/test_low_storage_settings.py`, `tests/stepper/test_low_storage.py`,
    `tests/test_seps.py`, `tests/backends/test_driver.py` (44, alone); 725 tests of 17
    neighbouring modules (the numba Cartesian, line, planar, radial, general,
    dispersive, memory and multistep ones), all passing.
  - **Parity** within `parity_bound`, fixed before measuring (four evaluations a step of
    32 products and 8 roundings per transform level, the response's addition depth,
    the flows' argument term of §4.5 and, on the separable kinds, two ulps of `k` over
    the run, with a stated growth of 2): measured 2.2e-17 against 1.1e-11 (Cartesian),
    1.1e-15 against 2.3e-11 (Cartesian, in-stage collar and exchange), 7.8e-16 against
    1.9e-11 (nitrogen gradient capillary, Raman and plasma, a varying operator),
    1.1e-11 against 9.5e-10 (planar Kerr); every route and flow treatment passes.
  - **Threads:** the parallel kernels against the serial ones at 1, 2 and 4 threads
    within 4 eps per element and `n_columns eps` on the norm; whole runs at 1, 2 and 4
    threads (large regime) within the parity bound.
  - **Memory:** the ledger's field rows exact (4/3 volumes without a collar, 5/4 with
    one, one more each for the noise and a held field-free plane) and released by
    `close()`; the estimate exact for them.
  - Coverage of `numba_cpu/low_storage.py` and `backends/low_storage.py` at 100 % (the
    kernels interpreted, then the engine compiled, `coverage-cpu.ini`); every added line
    of the other changed modules covered but moved Runge–Kutta stage lines, which the
    lane's existing tests run.
  - Doctests of the changed modules; `ruff`, `basedpyright`, `numpydoc` on them;
    `poe docs`; the public-tree scan.
  - At review, on Linux with the schema-31 Kokkos wheel: `poe check` (16 060 passed,
    29 skipped), `poe cov` at 100 %, `poe test-slow` (147 passed, 1 skipped) and the
    Kokkos-selected default (16 060 passed) and slow (148 passed) lanes. The first check
    stopped at nine type errors in the new numba test module, fixed at review; the
    import-time budget missed once on a loaded node (0.28 s of CPU against a local
    0.037 s on both this branch and `main`) and passed on the re-run.
  - Not run: GPU; equal Richardson error and the larger-beam one-thread timing of §7.4
    (shape B on all threads only).
- **Sources checked:** none beyond phase 1's; the fold is §4.3's table, checked
  element by element against it.
- **Limitations:** the speed target (deviation 1); the indicator off on `numba_cpu`;
  the planar and modal routes hold `C`; the benchmarks are same-process ratios on a
  loaded workstation (load 3.7 to 8.1). Found at review, for phase 3 or the roadmap:
  on the modal route with a varying operator under `"recompute"`, two state-sized
  transients a step (and the phase-difference exponent with its contiguous copy, even
  when tabulating) are outside the ledger and the estimate, so a large multimode run's
  peak exceeds the estimate by a few volumes; and the engine's held-table count and the
  estimate's use two predicates that agree only because a separable operator is always
  constant, which one shared predicate should replace.
- **Open questions:**
  - **Q14 — phase 3: the speed target.** Kokkos CPU's fused stage may give a different
    ratio; whether §7.4's target is revised to the current tree's replay is the
    maintainer's (destination: phase 3's report).
  - **Q15 — later: the evaluation's core.** Two copies per node chunk (`_free_material`)
    and a zeroed projected buffer per chunk are common to both marches; trimming them
    speeds both (destination: roadmap, *Performance and backends*).
  - **Q16 — later: the planar route's `C`.** Its finish could write the next input into
    the column volume as the Cartesian one does (one volume).
- **Handoff:** phase 3 implements §6.7 on Kokkos behind `_low_storage_engine`; the
  fold body, the exchange, the elision rule (`low_storage_stage_register`, extended to
  `"kokkos"` when its engine elides) and the table key (`table_steps`) are the
  reference forms; `tests/backends/test_numba_low_storage.py`'s cases are the parity
  cases to parametrise over the Kokkos backend.

### Phase 3 report — Kokkos CPU

- **Scope:** phase 3 of §11, 2026-10-19, Opus 5.5 (AI). Branch
  `feature/sep-0024-phase-3` in solprop (from `main` 9eef542, kernel schema 31, output
  schema 20) and in solprop-kokkos (from its `main` 9ec03d5); kernel schema 32, output
  schema unchanged. Merged with `main` at e9dd4b8 (SEP 27 drafted, documentation only)
  and merged into `main` on 2026-10-19 by the merge commits that follow the commit
  carrying this report, solprop-kokkos alongside.
- **Delivered:**
  - solprop-kokkos, native: `GeneralState::low_storage_step` (one call a step: the four
    register passes, `low_storage_separable_pass` and `low_storage_modal_pass`, their
    inputs and folds, the fold body `low_storage_fold`, `register_traversal` with the
    final pass's per-column norm), the flows (`low_storage_flows_formed`: a constant
    operator's table from scalar row 2, a varying operator's step pair, or none),
    `hold_plane` and `held_to_host`; `Problem::low_storage`,
    `low_storage_stage_register`, `low_storage_flows`; `GeneralWorkspace::allocate`
    under the march creating none of `stages_`, `solution_`, `error_`, `residual_`,
    `recycled_`, `advanced_`, `whole_propagator_` or `stage_inverse_`, but the
    accumulator, the stage register (unless elided) and the flows; `finish` acting on
    the state in place; the ledger rows `register_accumulator`, `register_stage`,
    `held_plane` and the Runge–Kutta rows read off the allocated views
    (`include/solprop_kokkos/core.hpp`, `src/general.cpp`, `src/bindings.cpp`,
    schema 32).
  - solprop-kokkos, Python: `KokkosBackend._low_storage_engine` returning
    `_ResidentLowStorage` (`atomic`, NaN `stiffness`, the table keyed by
    `table_steps`), `_hold` and `_held_field` on the native state, the device rule of
    `_propagator_mode`, the refusal removed (`src/solprop_kokkos/backend.py`,
    `__init__.py`'s `DRIVER_SURFACE`, `_core.pyi`); README, CHANGELOG.
  - solprop: kernel schema 32 (`translate/problem.py`); `LowStorageEngine.atomic` and
    the march's single commit for an atomic engine (`backends/low_storage.py`);
    `StepDriver._held_field` and the surface's `_hold`/`_held_field`
    (`backends/driver.py`, `tests/backends/test_driver.py`); one shared flow
    predicate `low_storage_flows` used by both engines and the estimate, and the
    elision predicate extended to `"kokkos"` (`backends/memory.py`,
    `backends/numba_cpu/low_storage.py`); `LOW_STORAGE_BACKENDS` with Kokkos and
    `"auto"` resolved as for any integrator (`driver/simulation.py`).
  - Tests: `tests/backends/test_kokkos_low_storage.py` and its thread probe
    `kokkos_low_storage_thread_probe.py`; `test_numba_low_storage.py`'s parity cases
    run on every compiled backend (`compiled_backends`, `assert_compiled`); the atomic
    commit on the host engine (`test_low_storage.py`); solprop-kokkos
    `tests/native/test_core.cpp` (the passes against a host RK4IP, the refusals) and
    `tests/test_backend.py` (parity with the reference, both flow treatments).
  - Benchmarks: `benchmarks/sep24/phase3_costs.py`, the ledger entry in
    `docs/dev/benchmarks.md`. Documentation: `docs/guide/numerics/choosing-a-stepper.md`,
    `fixed-schedules.md`, `docs/dev/backends.md` (§8, §9's row),
    `docs/dev/kernel-contract.md` (row 32, §7, the header's stale schema), `CHANGELOG.md`.
- **Deviations from the proposal:**
  1. **The in-tree speed target is not met on Kokkos either.** One thread, shape A:
     2.02 times less per step and 1.35 per evaluation than the Kokkos IP-DOPRI5 replay
     (target 4 and 3); all threads 1.81 and 1.21 (shape A), 1.85 and 1.24 (shape B).
     Against `numba_cpu`'s march: 7 % faster on one thread, 1.6 (A) and 2.1 (B) times
     on all threads.
  2. **Schema 32 adds two surface members and one protocol attribute** where §6.7
     names none: `atomic` (so the shared march, not the backend, commits the native
     step with `_finish` and the move) and `_hold`/`_held_field` (so the held plane
     lives in the memory space, the decided Q15 treatment).
  3. **The held plane is allocated at the first field-free plane**, not at
     construction (the native state does not know the run's retention); one volume,
     kept, in the ledger and the admission as before.
  4. **`"auto"` on a device build recomputes** and the native state holds no table,
     but `low_storage_flows` (a CPU count) still counts it: the estimate is
     conservative by one volume there.
  5. **No stage profile on Kokkos**: the step is one native call.
  6. **The march's quadrature score is not taken** on Kokkos (the embedded
     projection's alarm is the Runge–Kutta and Adams paths'), as on `numba_cpu`.
- **Decisions:**
  - **Q5 (Kokkos's stage):** split, without moving the Runge–Kutta arithmetic. The
    separable stage became input (`separable_stage_input`), core (`separable_core`:
    the transverse pair, the chunk loop, the response or the collar term) and finish
    (`separable_finish_transform`, accumulation); the modal stage became entry, core
    (`modal_core`, `general_core`) and finish. The march forms its own inputs and folds
    around the same cores; the Runge–Kutta path is code movement only, and every
    existing Kokkos parity test of it ran unchanged.
  - **Q11 (the indicator):** NaN, as on `numba_cpu`: `C` is elided on the routes the
    march is for, and recovering `g1` and `F2` would cost a volume and two reductions.
  - **Q4 on Kokkos:** the same elision rule (`low_storage_stage_register`): free space
    without an in-stage collar; the modal and planar routes hold `C`.
  - **The review's follow-ups:** the shared predicate (`low_storage_flows`) replaces
    the two that agreed by accident, in both engines and the estimate; Kokkos forms a
    varying operator's recomputed flows per element from the phase terms, so it holds
    no transient outside the ledger. `numba_cpu`'s own modal transients are untouched
    (not this engine).
  - **Schema:** 32 (no problem field moves; an older wheel refuses the march).
- **Verification** (the full gate, the slow lane and the coverage lane not run):
  - Parity on every case of `test_numba_low_storage.py` for Kokkos and `numba_cpu`
    against `numpy_ref` within the phase 2 `parity_bound` fixed before measuring (47
    passed, medium included): soliton with and without noise, both flow modes; linear
    and collar-only modal; varying capillary with Raman and plasma; radial with its
    collar; general modal (pair, Raman, collar, linear); Cartesian with collar and
    noise; line (Kerr, Raman, plasma, collar); planar (seven cases); the dispersive
    response. Cartesian Kokkos departures 1.9e-17 to 1.2e-15 against bounds of 1.1e-11
    to 2.3e-11. Two artificial linear cases now also drop the Kerr coefficients,
    which Kokkos reads from the problem rather than the response order.
  - `test_kokkos_low_storage.py` (15): field rows exactly 4/3 volumes, 5/4 with a
    collar, no Runge–Kutta view, the estimate equal to the ledger for the march's rows;
    a varying operator's pair or none; the held plane natively, its noise phase and a
    resume within the parity bound (Cartesian, and the soliton against the
    reference); the in-flight guard after a raising step; a Ctrl-C during the native
    step landing after the commit; `"auto"` running the march on Kokkos; one thread
    twice and four threads in fresh processes within the parity bound.
  - Native gtests: the passes against a host RK4IP over three steps (both flow modes,
    4096 eps per element, fixed beforehand), the ledger, the held plane, the refusals.
  - solprop: `test_low_storage.py`, `test_driver.py`, `test_memory.py`,
    `test_low_storage_settings.py`; coverage of the changed lines (low_storage.py
    100 %); doctests of the changed modules; `basedpyright src tests` 0 errors;
    `numpydoc lint`, `codespell`, `ruff`; `poe docs`; the public-tree scan.
  - solprop-kokkos `scripts/check.sh` with `SOLPROP_KOKKOS_SOLPROP_SOURCE` at this branch: format, lint, REUSE, pyright, clang-format, the tidy and sanitizer native builds and tests, 325 Python tests passed and 5 skipped, the distribution wheel and its smoke test, all green (Homebrew Clang 23, macOS arm64, OpenMP; Python tests at 4 threads). The wheel built from the branch is installed in the solprop worktree and reports schema 32.
  - At review, on Linux with the schema-32 wheel built from the solprop-kokkos branch:
    `poe check` (16 079 passed, 29 skipped), `poe cov` at 100 %, `poe test-slow` (147
    passed, 1 skipped) and the Kokkos-selected slow lane (148 passed). The first round
    found the kernel contract's current-version line still at 31, the one-run figure
    not redrawn, and two branches untested (the held buffer's reuse, a read-only
    accepted state); all four were fixed at review and the check and coverage lanes
    re-run green. The Kokkos-selected default lane failed only on the first two.
  - Not run: GPU builds (not compiled here either); equal-Richardson-error timings;
    the one-thread shape B timing.
- **Sources checked:** none beyond phases 1 and 2; the fold is phase 2's body.
- **Limitations:** the speed target; no indicator on the compiled backends; the device
  path compiled nowhere in this phase; Kokkos benchmarks on a loaded workstation (load
  3.2 to 12.8). Found at review, native guards that solprop cannot reach as wired today
  (its elision rule excludes an in-stage absorber, a separable operator is constant, and
  the table key starts at NaN), for a follow-up: `allocate` should refuse an elided stage
  register beside an in-stage absorber (the exchange fold would read the null register)
  and a separable geometry with two flow volumes (its fold reads only the first flow), and
  a `table_ready_` flag should refuse a step on a table never built. The native test
  covers the modal route with one constant operator and equal flow rows; the separable
  fold, the exchange and the elision are covered by the Python parity tests only.
- **Open questions:**
  - **Q17 — the maintainer: the speed target.** Both compiled backends miss §7.4's
    ratios against SEP 25's fused replay (Q14); whether to restate the target is the
    maintainer's (destination: the SEP's Final review).
  - **Q18 — device qualification (Q7).** Compile the passes for CUDA and HIP and
    qualify them (destination: the roadmap's device items).
- **Handoff:** the three CPU backends run the march; phase 4 (the embedded pair) may
  reuse `low_storage_fold`'s organisation and Kokkos's `register_traversal`.

### Closing report — the native guards, the follow-ups and phase 4 not started

- **Scope:** the close of the SEP after phase 3, 2026-10-19, Opus 5.5 (AI). Branch
  `feature/sep-0024-closing` in solprop (from `main` 7affe12, kernel schema 32, output
  schema 20) and in solprop-kokkos (from its `main` 391ca7d); no schema change. The
  follow-ups phase 3's review named, the decision not to start phase 4, the open
  questions' destinations, and the device checks under HIP and CUDA.
- **Phase 4 is not started** (the maintainer, 2026-10-19). Phases 2 and 3 show the
  march competitive in memory (four volumes against fifteen or sixteen) and faster per
  step than the IP-DOPRI5 replay on both compiled backends, but no adaptive memory-bound
  use is waiting for it, and the fixed-step use is served by the replay with an
  external calibration. The embedded pair (Q8) is a roadmap item under *Numerics*,
  with the SEP's recommendation (ERK4(3)-IP [7] first) carried over.
- **Delivered:**
  - solprop-kokkos, native (`src/general.cpp`): `GeneralWorkspace::allocate` refuses,
    with `std::invalid_argument` and before any view, an elided stage register beside
    an in-stage absorber (the exchange fold re-forms its input from `C`) and more than
    one flow volume on a separable geometry (its fold applies one, the table); a
    `table_ready_` flag, set when a step rebuilds the table and cleared by
    `set_initial` and `close`, makes `low_storage_step` refuse (`std::logic_error`) a
    step that would reuse a constant operator's table never built. Tests:
    `tests/native/test_core.cpp` (`LowStorageRefusesRegistersItsFoldsCannotRead`,
    `LowStorageRefusesATableNeverBuilt`, each with the accepted counterpart beside the
    refusal) and, through Python with a forged stale key since solprop's engine keys
    the table from NaN after every load, `tests/test_backend.py`
    (`test_a_low_storage_table_never_built_is_refused`). CHANGELOG.
  - solprop, `numba_cpu`'s modal follow-up (phase 2's review): recomputing on a modal or
    general route the engine holds one flow scratch volume, forms `P` in it for pass 1
    and `Q` over it before pass 3 (pass 2 reads no flow, passes 3 and 4 only `Q`), and
    exponentiates a modal flow per element from the terms of `Phi`
    (`exponentiate_terms`, `src/solprop/backends/numba_cpu/low_storage.py`) or from the
    z table's held difference, so neither the two state-sized flows nor the exponent
    array (with its contiguous copy, which tabulating also made) is allocated in a step.
    The scratch is counted in the ledger's `propagators` row and by the estimate
    (`low_storage_flow_scratch` in `src/solprop/backends/memory.py`, used by
    `estimate_memory` and `low_storage_memory`), to the byte.
  - Tests (`tests/backends/test_numba_low_storage.py`): the varying capillary's flows
    held (two volumes tabulated, one recomputed), the reference's count plus the
    scratch, the estimate's base equal to the reference's plus the scratch, the
    admission's figure equal to the estimate from the march or the default stepper, and
    no state-sized allocation from the engine module and no `_delta_phi` call in a step;
    a z-table operator (a dispersion declared along z) against the reference, both flow
    treatments; `exponentiate_terms` against `exp(tensordot)` and, for one term of unit
    weight, the forward exponential exactly (`test_numba_low_storage_kernels.py`).
  - Documentation: `docs/dev/backends.md` (§9's paragraph and the device column),
    `docs/dev/roadmap.md` (the SEP's entry; the pair under *Numerics*; Q9 in the
    stepper's open questions; Q2, Q16 and the prefactor under *Performance and
    backends*; the device item), `docs/dev/stepper.md` §9, `CHANGELOG.md`.
- **Deviations from the proposal:** none beyond phases 2 and 3's. A recomputing modal
  route on `numba_cpu` now holds one volume where §6.8's table says none for the
  recomputed flows; it was a transient of two volumes before, and Kokkos still holds
  none.
- **Decisions:**
  - **The modal follow-up:** one reusable scratch, in preference to
    forming the flows inside the fold (Kokkos's way), which would need a second fold
    variant per route and an exponential per element in three passes.
  - **Q14 and Q17 (the speed target):** withdrawn. The maintainer set no fixed target:
    the march is to be as fast as possible, ranked by measurement against the other
    marches as §7.4 does, the memory saving standing in its own right. The measured
    ratios stay in phases 2 and 3's reports and the benchmark ledger.
  - **Q15 (the evaluation's core):** covered by
    [SEP 28](sep-0028-free-space-evaluation.md) (Draft), which takes the copies and
    clears inside each node chunk on every stepper; no separate item.
  - **Q11 (the indicator on the resident engines):** stays NaN, as decided in phases 2
    and 3; not reopened.
  - No new ADR.
- **Device qualification (Q7, Q18):**

  | build | device | date | result |
  |---|---|---|---|
  | HIP | AMD MI325X (gfx942, ROCm 7.14) | 2026-10-19 | compiled; kokkos `-m gpu` (31), solprop `test_kokkos_gpu.py` (2) and the march's device tests (15) pass |
  | CUDA | NVIDIA H200 (CUDA 13.1, HOPPER90, GCC 13.4 host) | 2026-10-19 | compiled; the same three sets pass (31, 2, 15) |

  The march at 768x768x600 with statistics-only planes takes 610 ms a step on the MI325X
  and 674 ms on the H200 (high-water 46 and 34 GB, transform workspace included), against
  1077 and 1092 ms for the IP-DOPRI5 replay; the gap to a single-transform engine on the
  same card is [SEP 28](sep-0028-free-space-evaluation.md)'s subject. These checks build
  `main` (solprop 29b3cbc, solprop-kokkos 95b7502), not this branch; the native guards
  here add refusals only.

- **Closing lanes**, on Linux (an AWS `c8a.24xlarge`, the S3 build cache) on the closing
  branches merged with `main` (solprop 95fa3ee, solprop-kokkos with SEP 23 phase 3): `poe
  check` (16 106 passed, 29 skipped), `poe cov` at 100 %, `poe test-slow` (147 passed, 1
  skipped) and the Kokkos-selected default (16 106 passed) and slow (148 passed) lanes,
  all green.
- **Verification** (targeted, before the closing lanes):
  - solprop: `tests/backends/test_numba_low_storage.py` (50), `..._kernels.py` (22,
    compiled and interpreted), `tests/backends/test_low_storage.py`,
    `tests/backends/test_memory.py`, `tests/driver/test_memory.py`,
    `tests/driver/test_low_storage_settings.py`, `tests/backends/test_kokkos_low_storage.py`
    and `tests/backends/test_driver.py`: 332 passed, with the closing branch's
    distribution wheel installed. The z-table parity departure 7.8e-17 against a bound of 9.9e-13
    fixed before measuring (the phase 2 `parity_bound`). Coverage of the changed lines
    of `numba_cpu/low_storage.py` and `backends/memory.py` at 100 % (kernels
    interpreted, then the engine compiled, `coverage-cpu.ini`). Doctests of both
    modules; `ruff`, `basedpyright`, `numpydoc lint`; `poe docs`; the public-tree scan.
  - solprop-kokkos: the native low-storage gtests (4) on a debug build; the
    low-storage Python tests; `scripts/check.sh` with `SOLPROP_KOKKOS_SOLPROP_SOURCE` at the
    closing branch, green: format, lint, REUSE, pyright (its first run found the new
    test's untyped engine, fixed), clang-format, the tidy and sanitizer native builds and
    tests, 326 Python tests passed and 5 skipped, the distribution wheel and its smoke
    test (Homebrew Clang 23, macOS arm64, OpenMP). That wheel is installed in the solprop
    worktree and reports schema 32.
  - Not run: the full gate and the end-of-SEP lanes (run before Final); device timings.
- **Sources checked:** none new.
- **Limitations:** the native guards are not reachable through solprop's rules (they
  guard the native state against a caller that breaks them); the Kokkos device estimate
  stays conservative by one volume (phase 3, deviation 4); `numba_cpu`'s varying modal
  prefactor is still a state-sized array per pass on both marches (found here, a
  roadmap item).
- **Open questions** (every one now decided or with a destination):
  Q1, Q3–Q6, Q10–Q13 decided in phases 1 to 3; Q2 (hook names), Q16 (the planar
  route's `C`) and Q7/Q18 (devices) to the roadmap's *Performance and backends*;
  Q8 (the pair) to *Numerics*; Q9 (transfer across transverse grids) to the
  stepper's open numerical questions on the roadmap and `stepper.md` §9; Q14/Q17
  withdrawn (no fixed target); Q15 to SEP 28.
- **Handoff:** the closing lanes, Final and the archive move are the maintainer's.

## Part III — History

- 2026-10-16: Drafted.
- 2026-10-16: Reviewed by Opus 5.5 (AI). Resolved in the text:
  - the flow table, reused within `LANDING_ULPS` of the step that built it, with a resume
    that agrees to rounding;
  - the in-stage exchange;
  - the dispatch inventory;
  - the guard's reach;
  - the commit block;
  - the records;
  - the in-tree benchmark target.
- 2026-10-17: Restated for [SEP 25](sep-0025-rounding-level-reproducibility.md).
  Thread-count invariance of the fused kernels (§6.6, §7.2, phase 2 in §11), repeats
  (§7.2) and the `step_planes` and lockstep-batch identities with the uninterrupted run
  (§7.1) become agreement within a rounding bound fixed before measurement, the last
  on the same edges, which stay exact. The resume, flow-mode and independent-RK4IP
  statements already stated agreement to rounding, not bit for bit (§4.3, §5, §7.1,
  §9), and stay. Kept exact as structural: existing runs taking the same code path
  (§9) and the ledger and `estimate_memory` to the byte (§6.6). §10's rejection of a low-storage Runge–Kutta march now cites recapture of
  recorded states rather than moved digests.
- 2026-10-18: Revised against SEP 25 Final and SEP 21 as reviewed. The schema numbers
  follow SEP 20's 27 and SEP 25's 28 (§3, §5, §6.4, §9, §11: the next free versions are
  29 and 19 at this writing); the host engine takes its stage scalars through SEP 25's
  `_stage_bundles` (§6.5); the drafting timings of §2 predate SEP 25's kernel work; the
  temporal collar's floor under this march and the refusal of SEP 21's implicit scheme
  are stated (§4.6, §5), and the march calls SEP 21's accepted-endpoint hook (§6.2). The
  ledgers of §2 were re-measured on the current tree on `numpy_ref` and `numba_cpu` and
  stand.
- 2026-10-18 (later): §2's drafting timing re-read: it ran under `split_step` without
  FSAL, seven evaluations a step, so the per-evaluation gap is about 4.8, not 5.5, and
  the per-step factor stands; §4.3 notes SEP 22's zero-rate elision.
- 2026-10-18: Accepted (John Travers) after his final review; Part I frozen.
- 2026-10-18: Phase 1 implemented (specification, seam and `numpy_ref`); status In
  progress.
- 2026-10-19: Phase 2 implemented (`numba_cpu`).
- 2026-10-19: Phase 3 implemented (Kokkos CPU).
- 2026-10-19: Phase 4 not started (the maintainer); the native guards and `numba_cpu`'s
  modal flow scratch closed the SEP's follow-ups; the march compiled and passed its
  device tests under HIP (MI325X) and CUDA (H200).
- 2026-10-20: Final (John Travers). Phases 1–3 and the closing work merged with green closing lanes; phase 4 not started; device checks under HIP and CUDA passed; moved to the archive.
