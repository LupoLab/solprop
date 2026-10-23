# The stepper

This document describes the implemented solver: its design, the evidence behind it, the
numerical questions still open and the known numerical traps. Some of the evidence comes
from an earlier prototype study that compared schemes, controllers and representations of
the linear operator before solprop was written; numbers attributed to it below were
measured there, and none of its code is in solprop. Decisions are ADR-004 to ADR-009 and
ADR-019 in [decisions.md](decisions.md). What each algorithm *is* — its method, its
theoretical order, its cost and its traps — is in [algorithms.md](algorithms.md), and this
page does not restate it.

## 1. The equation and the interaction picture

The equation, the cumulative operator `Φ` and the interaction picture are
[algorithms.md §1](algorithms.md#1-the-equation-and-the-interaction-picture): `D` is
diagonal in ω, so `exp(Φ(z₂) − Φ(z₁))` is the exact linear flow over any step, and each
stage costs one evaluation of `N` plus two applications of `exp(±ΔΦ_i)`. This section is
what one stage does, in the order a backend executes it.

### One step, pass by pass

`N` is the expensive half, and what it costs is passes over the data rather than
arithmetic. The table is one Runge–Kutta stage `i` of a step anchored at `z_n`. It mirrors
[architecture.md §8](architecture.md), where the fusion, the buffer reuse and the FFT
counts live, and names the function of `solprop.backends.numpy_ref.passes` that performs
each pass: the reference backend writes every one of them out separately, and every other
backend is validated against what it produces.

| # | pass | formula |
|---|---|---|
| 1 | `stage_input` | `u = exp(ΔΦ_i) (y + h Σ_j a_ij k_j) + N₀ exp(i Im Φ(z_i))` |
| 2 | `to_time` | zero-pad onto the fine grid and inverse transform every modal column |
| 3 | `synthesise` | `E_ip(t) = τ(z) Σ_m S_ip,m a_m(t)` on a chunk of quadrature nodes, in V/m |
| 4 | `nonlinear_polarisation` | `P(t)` from the compiled responses, times the temporal collar |
| 5 | `project` | `P_m(t) = Σ_ip W_m,ip P_ip(t) / τ(z)`, accumulated over the chunks |
| 6 | `to_frequency` | forward transform every modal column and crop to the coarse grid |
| 7 | `stage_output` | `k_i = exp(−ΔΦ_i) p(ω, z) W(ω) P̃` |
| 8 | `accumulate` | `acc += b_i k_i`, and the embedded difference into its own accumulator |

`a_ij`, `b_i` and `h` are the Runge–Kutta coefficients and the step; `N₀` is the shot-noise
realisation of the `"modified"` noise model (§7; zero without one); `S` and `W_m,ip` are the
synthesis and projection matrices between modal amplitudes and the field at the quadrature
nodes; `τ(z)` is the taper scale that carries a varying effective area; `p(ω, z)` is the
UPPE prefactor with the crop scale folded in; and `W(ω)` is the spectral collar profile
([kernel-problem.md](kernel-problem.md)). With the default in-stage absorber scheme, pass 7
also adds the temporal absorber's own term, `−α(t) E(t)/2` transformed to the modes, beside
the normalised polarisation (§6).

**Passes 3 to 5 are a loop over node chunks.** The transforms of passes 2 and 6 run once
over all `n_modes n_pol` columns; the space-domain work runs over `transform.node_chunk`
nodes at a time and the projections accumulate, so the buffers `E` and `P` are sized by the
chunk and not by the rule ([architecture.md §13](architecture.md)). Mode-averaged
propagation is the one-node, one-column case of exactly this code and not a second path
(ADR-010): the "quadrature" is then the mode itself, the synthesis is
`1/sqrt(ε₀ c A_eff / 2)` and the projection is one. A one-mode multimode run reproduces the
mode-averaged one to 1.3e-16, and to 1.1e-11 on a rule of nine nodes, which is the rule's
own error.

**Eight passes is how the equation is written, not how it is executed.** The reference
backend performs each as its own traversal, which is what makes it the oracle: every
intermediate is an array you can look at. `numba_cpu` does the same arithmetic in the same
order with those arrays never formed — passes 1 and 2's scatter in one traversal, the whole
of 4 in one traversal of `t` per node, and 6 and 7 with both accumulations in one more. It
agrees with the reference to about 1e-15 rather than to the last bit, for an arithmetic
reason and not an algebraic one: NumPy's complex multiply can contract into a fused
multiply-add where a compiled kernel's need not. Which passes are fused, and what each
costs, is [backends.md §3](backends.md).

The transverse components are carried by the rows of `S` **or** by the state's polarisation
axis, never by both: a row of the synthesis is one Cartesian component at one node, and so
is one entry of the state's polarisation axis. A vector run is therefore either a
polarisation-resolved mode set with both components resolved at each node, or a spatial set
whose amplitudes carry the two components; the assembly refuses the combination that would
count them twice.

The synthesis and the projection are a **dual pair**: the propagated basis is
`e_m / sqrt(N_m)`, whose overlaps are `2/(ε₀ c)` rather than one, so the projection carries
`ε₀ c / 2` beside the quadrature weights and `W S = (n_to / n_t) I` on an orthonormal set
(`solprop.modes.quadrature.BASIS_NORMALISATION`, folded in by `mode_matrix`).

The interaction picture is re-anchored at the start of every step, so
`ΔΦ_i = Φ(z_n + c_i h) − Φ(z_n)` and the anchor is hoisted once per step. Each exponent is
evaluated once and **negated** for the inverse (§4), and `propagator_mode` decides whether
the six distinct exponents are kept for the whole step or recomputed per stage; its default,
`"auto"`, is the backend's choice ([backends.md §3.6](backends.md)). The noise of pass 1 is
added in the frequency domain and costs no transform (§7).

Two scale factors are already inside the problem rather than in these passes: the
oversampling **pad** scale is folded into `transform.synthesis`, so that every response sees
the physical field in volts per metre whatever its degree in the field, and the **crop**
scale is folded into `responses.prefactor`, which is applied in the same pass as the crop
([kernel-problem.md](kernel-problem.md)). The prefactor is exactly zero outside the active
band, so no unnormalised polarisation survives there for the stages and the error estimate
to see.

When the stages are done the step is completed and scored:

```
y ← exp(ΔΦ) (y + h Σ_i b_i k_i),     e = exp(ΔΦ) h Σ_i (b_i − b̂_i) k_i,
```

so the local error is that of the *propagated* state and not of the interaction variable
(`accept_step`). The error is reduced to one scaled number over the norm mask (`error_norm`,
the norm of §3), and an accepted step ends with the boundary treatments applied once per
step rather than once per stage: the hard band mask (`apply_hard_mask`), idempotent because
the operator and the prefactor are already zero outside the active band, and, only under
`absorber_scheme="split_step"`, the temporal absorber as an exact rate over the step
(`apply_temporal_rate`). The default scheme has already applied the absorber inside every
stage (pass 7), which keeps the scheme fifth order where the split factor would make it
first order in the collar (§6).

### The fused response pass

Pass 4 walks `responses.response_order`, which the translation layer compiled, and calls one
named function per stage, accumulating `P(t)` and applying the temporal collar once at the
end — except to the dispersive response, which applies it to its own products before its
output multipliers and is added after:

| stage | pass | formula |
|---|---|---|
| `intensity`, `ionisation_rate` | `ionisation_rate` | `W(abs(E))` from the table; `hypot(E_x, E_y)` for two components |
| `plasma_scan` | `plasma_scan` | `f = 1 − (1 − f₀) exp(−∫W dt)` and `J_free = (e²/m_e) ∫ f E dt` |
| `plasma_current` | `plasma_current` | `J_loss = I_p W (1 − f) E / abs(E)²`, then `P = ρ ∫ (J_free + J_loss) dt` |
| `kerr` | `kerr_polarisation` | the two cubic forms of `solprop.translate.responses.KerrCoefficients` |
| `dispersive_kerr` | `dispersive_kerr_polarisation` | per polynomial channel, or per symmetric group under its mixture filter, `P = ε₀ γ₀ ρ` times the filtered cubic of `solprop.nonlinear.responses.dispersive_cubic` on the multipliers of `solprop.translate.responses.DispersiveKerrTerms`, `G[w (G E)³]` with the collar `w` inside |
| `raman_drive` | `raman_polarisation` | `P = ε₀ ρ E(t) irfft[H̃ · rfft(D)]` on the doubled grid, with `D` the real drive of the grid |

The Raman pass is a convolution and so has memory: the drive of every column is zero-padded
to twice the fine grid, transformed once, multiplied by the damped kernel spectrum and
transformed back, and only the first `n_to` samples are kept
([raman.md](../guide/media/raman.md), "The doubled grid"). The drive is real on both grids,
so one real transform pair suffices. The kernel spectrum does not depend on the nodes, so a
stage forms it once — `raman_spectrum`, damping each row of `raman_coefficients.undamped` at
the rate its table gives at this plane — and shares it between the chunks; the backend
memoises it on those rates, so a glass transforms its kernel once for a whole run and a gas
on a pressure gradient once per plane. Given the oracle's own kernel the pass reproduces
`solprop.nonlinear.raman.RamanResponse.evaluate` to rounding on both grids, for a glass and
for a molecular gas, with and without the third harmonic.

The plasma passes integrate in the argument order of
`solprop.nonlinear.responses.PlasmaResponse.evaluate`, so the reference backend reproduces
the oracle to rounding. They are kept as separate passes deliberately, because here the
multipass form *is* the reference ([nonlinear.md](../guide/media/nonlinear.md)): fusing the
running sums into one sequential sweep belongs to the compiled backends, which are validated
against these numbers.

### The quadrature alarm

The Kronrod rule's embedded Gauss subset shares every node it has, so projecting the same
node polarisation a second time with `transform.projection_coarse` costs one GEMM and one
transform per accepted step. The relative difference of the two, `quadrature_error`, is **a
statistic and a logged warning above 1e-6, once per plane, and never a controller input**: a
rule too coarse for the integrand is a modelling error no step size can fix, and a fixed
rule is used rather than an adaptive cubature precisely so that the cost of a step does not
depend on the field. A rule with no embedded pair reports `nan` rather than the zero its own
difference with itself would give. Measured on a three-mode set: 1.6e-1, 2.1e-2, 4.1e-5,
7.2e-9 and 1.2e-15 at `n_r = 8, 16, 24, 32` and 48.

## 2. Scheme: IP-DOPRI5 (IP-Tsit5 available)

The pairs, their orders and their tableau theory are
[algorithms.md §2](algorithms.md#2-the-rungekutta-pairs). The evidence for choosing them:
the prototype study measured (matched error, RDW parameters, co-moving frame): Kerr rung —
splitting (SRKNb6) leads by 11 %; shock rung — IP-DOPRI5 first, splitting 4.4× dearer; plasma rung — IP-Tsit5 first, IP-DOPRI5 within 20 %,
splitting 6.7× dearer. Splitting needs the nonlinear flow in closed form; with shock or
plasma it needs an inner solver whose order caps the composition (`min(p, q)` measured to two
decimals). In a multimode code, where `N` is an order of magnitude heavier, the 6-versus-28
evaluations-per-step gap is the whole argument.

Tableau constants (DOPRI5 and Tsit5) are the only things carried from Luna's `dopri.jl` and
`tsit.jl`, and the two upstream bugs of Luna PR #436 (algorithms.md §2, traps) are tests: the
order conditions are checked for both weight vectors and the propagated solution is the
fifth-order one, and the FSAL stage is not recycled before anything that depends on the
completed step (irrelevant without dense output, but tested).

**FSAL, the evidence.** The rebasing identity `k₁ⁿ⁺¹ = exp(ΔΦⁿ)·k₇ⁿ` and the condition it
needs (nothing changes the state after the last stage) are algorithms.md §2. **The hard
band mask satisfies it:** the prefactor is exactly zero outside the active mask, so no stage
writes there, the state never acquires content there and the mask is a no-op (measured at
9.9e-17). **The temporal absorber's per-step split factor does not** (ADR-025), which is
why the default absorber is applied inside the stage (§6): that is what makes the saving
available at all. The same argument requires the in-stage absorber's own term to be masked
to the active band: unmasked it writes outside the band, the accept pass's mask removes it,
and the identity fails at 7.4e-7 rather than 1.8e-16.

`numba_cpu` takes the saving — 9.2 to 9.5 % of a step, four modes and one, with the
answers agreeing to 3e-16 and 6e-16 over 400 fixed steps — and keeps the recycled stage
across save planes (ADR-039), so a resumed replay, which has no stage to recycle,
agrees with the uninterrupted one to round-off. The reference backend evaluates all seven
stages and takes correctness over the one evaluation in seven, so that no stage it uses
depends on the bookkeeping of the step before (trap 10 below).

**The FSAL invariant.** The cache holds the final interaction-picture stage k₇, not an
independently stored physical derivative. When re-anchoring the same equation at the
accepted endpoint, the first stage is exp(ΔΦ) times that cached stage. Including explicit
linear coupling inside the source preserves this identity. A changed state, gauge,
operator section, noise identity or incompatible event invalidates the cache; a save
plane does not. A coupled exponential flow needs the analogous
certified forward action at the old endpoint, and the same lifecycle test, before a stage
may be reused through it.

**A coupled flow recycles no stage, on any backend.** A frozen-residual flow adds
`R_s(z)·u` to the same lab-frame source the material response is evaluated in, so its
stages are the same objects a diagonal flow's are. The flow actions a step holds are keyed
on `(section, dz)`, so a step that begins in a different section rebuilds them; a `"cf4"`
section's actions are keyed on `(section, dz, z)` as well, because they exponentiate the
generator at Gauss nodes inside each stage's own interval, and nothing is reused between
steps. An adaptive run, whose `dz` changes every step, re-forms the actions every trial;
the compiled backends form them in their own kernels, at 3 to 11 % of a step.

The identity itself is known. On a frozen-residual or constant section the last stage's
input is the accepted state `y` and its stage is
`k₇ = U_h⁻¹ e^{−ΔΦ} (N(y) + R_s(z + h) y)`; the next step's first stage, at the zero
fraction where the action is the identity, is `N(y) + R_{s'}(z + h) y`. So
`k₁ⁿ⁺¹ = e^{ΔΦ} U_h k₇ⁿ` holds **exactly when the next step is anchored in the same
section**, `s' = s`: the certified forward action it needs is the whole step's `U_h`,
which the step already holds and whose conditioning it has measured. It fails at a
section edge, where the next anchor's frozen block and one-sided weights differ — the
same invalidation as the actions' own key. A CF4 section has no such identity (its
actions depend on `z`). In rounding the rebased stage differs from a fresh one by the
round trip `U_h U_h⁻¹`, a few `C u` with `C` the declared conditioning.

It is not taken. One stage evaluation on the coherent-wall cases (scalar wall, TE/TM
wall, taper; one thread) costs 0.09 to 0.57 ms on `numba_cpu` and 0.18 to 1.39 ms on
native CPU; ten fixed steps could recycle nine stages on a constant section and six on
a taper whose steps cross three section edges, **4 to 13 % of a ten-step window**. But
the reference recycles nothing on any route, so giving it FSAL beside a flow would move
its recorded coherent-wall states; and on the compiled backends alone
it would turn their flow parity from a per-stage into a per-step rounding statement,
which can fork an adaptive trajectory at that level. Whether to take it is an open
question (§9).

**A section edge is a one-sided position.** The sections of a coupled flow are placed on
the operator table's breakpoints, which land on every declared kink of the geometry, so
the generator is genuinely discontinuous across an interior edge. A stage that lands
exactly on one must read the side its own step came from; reading the other drops a run
from fifth order to first. The rule is in the kernel contract's `coupled_flow` row and is
normative: the step's section wins wherever the position lies inside it, matched within
1e-10 of the flow's span so that a schedule arriving a few ulp short of a kink and one
arriving a few ulp past name the same section, and a stage that ran past its step's
section takes the section it **closes** in.

**A CF4 section's flow error is outside the embedded pair.** On a `"cf4"` section the
interaction picture is built on the commutator-free fourth-order exponential of the whole
generator rather than on a frozen block, and no explicit linear term enters the source.
The embedded difference compares two combinations of the same stages, each carried by the
same approximate flow, so the flow's own local defect — `O(h⁵)`, which makes the run at
most **fourth** order — never appears in the error norm, and the controller can neither
see nor bound it; the flow's error enters the source only through the stage actions'
perturbation, one order of `h` higher. What governs it is the declared step:
`validate_coupled_flow` compares the step-doubling estimate `(16/15) ‖V(h) − V(h/2)²‖` at
the largest declared step with the controller's `rtol` and warns, advisory like the frozen
split's advisory step. That estimate is asymptotic, not a bound; it has been checked
against independent solves and the run's measured order. CF4 is admitted on `numpy_ref`
only and is never selected by default.

### The low-storage march

The design record is [SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md).

`stepper/low_storage.py` and `backends/low_storage.py` hold a third integrator,
`LowStorageRungeKutta()` (ADR-050): classical RK4 in the interaction picture anchored at
the step's **midpoint**, Hult's RK4IP. With `P = U(z_m, z_n)` and `Q = U(z_n + h, z_m)`
only forward flows are applied, four per element, and on a constant operator
`P = Q = exp(hD/2)`. The step is four passes over three registers -- the state `A`,
overwritten in place by `P A` and then by the accepted step, the accumulator `B` and the
stage input `C` -- beside the pipeline's scratch, each pass one source evaluation
(`_adams_lab_source`, the complete lab-frame source with the noise draw at the pass's
position) and one fold:

| pass | evaluation | fold |
|---|---|---|
| 1 | `F1` at `z_n` | `A <- P A`; `g <- P S`; `B <- A + h/6 g`; `C <- A + h/2 g` |
| 2 | `F2` at `z_m` | `B <- B + h/3 S`; `C <- A + h/2 S` |
| 3 | `F3` at `z_m` | `B <- B + h/3 S`; `C <- Q (A + h S)` |
| 4 | `F4` at `z_n + h` | `A <- Q B + h/6 S` |

`LowStorageMarch` replays a schedule and nothing else, mirroring `AdamsMarch`'s replay:
each step calls the engine (`LowStorageEngine.step`, the host one `HostLowStorage`, or a
backend's own from `_low_storage_engine`), applies `_finish` and moves the position in
one commit. There is no estimate: the history's norm is 0.0, "not estimated". The step
is **in flight** from its start to its commit (`StepDriver._in_flight`), and every entry
that reads or advances the state refuses it until `set_initial`; divergence is detected
on the accepted state's squared norm, which pass 4 reduces over the norm mask. The
stiffness indicator `h rho = 2 ||F3 - F2|| / ||F2 - g1||` (exactly `|h lambda|` for
`u' = lambda u`) is kept by the host engine and logged when it passes RK4's negative-real
interval 2.785: measured, it passed it on the step before an ionising replay diverged,
and read 0.17 to 0.65 of an in-stage collar's `h alpha_max / 2` as the collar held less
or more of the field; `numba_cpu`'s resident engine reports NaN. The **flow
table** (`"tabulate"`, the default for a constant operator) holds `exp(h_t D / 2)`, built
from its key alone (from the operator's origin, so no position's rounding enters it):
the key is the mean step of the step's run of equal steps (`table_steps`), so
a uniform segment builds one table and the phase it applies over the run telescopes to
the run's own; a varying operator's `P` and `Q` are formed per step. **On `numba_cpu`**
(`backends/numba_cpu/low_storage.py`) the engine is resident: each pass is a
register input kernel (the draw's phase formed per element on the separable kinds), the
route's own core shared with the Runge–Kutta march (`_free_pass`, `_slab_pass`, the
modal and general pipelines), and one fused finish that forms the source and folds it
(`low_storage_fold` in `backends/bodies.py`); a Cartesian or line pass under an in-stage
absorber folds the material source and re-forms the input in the transverse scratch in
one traversal (the exchange of SEP 24's register design) before the absorber's evaluation. The
step's three positions are each read as their own anchor by the hooks, so the engine
takes their bundles together (`_stage_bundles` keeps several anchors until the next
hoist). The march refuses a coupled flow, the TDSE response, interpolated planes and a
step past 2.785 beside an in-stage collar. Its order is four in the asymptotic range,
`|Im D| h / 2 <= 1` in the populated bins (measured 3.96 to 3.98 on the soliton, P2 and
a Cartesian Kerr beam, `tests/backends/test_low_storage.py`).

### Low-call Adams methods

`stepper/multistep.py` and `backends/multistep.py` hold the default integrator
(ADR-037, ADR-045; `stepper/runge_kutta.py`'s `RungeKutta` names the tableau's march).
`Multistep` is an Adams march with a derivative history, in the exponential formulation
by default ([SEP 16](seps/archive/sep-0016-unified-multistep.md)) and in the interaction
picture, transporting the history, on request
([SEP 13](seps/archive/sep-0013-expensive-response-steppers.md)):

- one source call per accepted step (PEC);
- a corrector one order higher than its predictor;
- variable order up to five by default.

`ComparisonMultistep` keeps the variants SEP 13 compared and did not select (PECE,
explicit Adams–Bashforth, matched pairs) for the benchmarks.

The march (`backends/multistep.py`) is shared: a backend supplies the step driver's two
hooks `_adams_source`, one source evaluation at a lab-frame field returned in the step's
anchor frame, and `_adams_propagator`, and takes the march as its `_march_to` when the
problem carries an integrator. `numpy_ref` evaluates through its reference passes;
`numba_cpu` through its stage kernels -- the modal route's fused source for the
interaction formulation, and for every other route and the exponential formulation a
zero-length first stage of the route's own pipeline, whose propagator is exactly one --
refusing only a coupled flow. `native` keeps
its state natively, so it also supplies `_adams_state` and `_adams_adopt`, a host copy of
the accepted state and the native finish; its source is stage zero of a trial whose
every stage row is the evaluation position's (`State.adams_source`), on every route
but a coupled flow, and its lab-frame source the same with the anchor at the
position, and after a dense
observation, which loads the interpolated field natively, `_adams_restore` reloads the
march's own state. The selected family, one-order-higher PEC, takes a **fused**
path: the history is a store of rows, and a step is three passes of `AdamsKernels` --
the predictor's lab-frame field, the corrected field with the defects of the step's own
and the neighbouring orders, and the transport -- which `numba_cpu` compiles
(`backends/numba_cpu/multistep.py`), with the rows of a large problem's state in a
`prange` (the exponential formulation's moments, predictor, corrector and compensation
likewise). Dense observation of save planes is ADR-036's; on the
fused path the step's slopes are copied before the in-place transport.
**The exponential formulation** (ADR-041) shares the march. Its samples are
untransported: the source comes from the hook `_adams_lab_source`, the lab-frame
evaluation, and lands in the same store of rows, which an accepted step does not
re-anchor. A trial takes `x = ΔΦ` over the step from the backend's own `_delta_phi`,
so that its flow `exp(x)` is the propagator the interaction formulation would apply,
forms the moments `k! φ_{k+1}(x)` once (`backends/exponential.py`), and applies the
predictor, the corrector and the neighbouring orders' defects in the monomial basis of
their interpolants (`ExponentialKernels`): the defects are already at the step's end,
so no propagator multiplies them. The defect, the norm's scale and the lifecycle are
the transported formulation's; so is the order choice, but for the in-stage collar's
stability, which it weighs (`AdamsMarch._order_weight`): each order's
step factor is capped at `COLLAR_STABILITY` times PEC's negative-real interval over
`h max(α) / 2`, the step itself left to the chosen order's defect. **Beside a coupled flow** (ADR-044) the backend's hook `_adams_block` returns the frozen block of the step's
section (`CoupledBlock`); `coupled_moments` forms the trial's flow and moments as
matrix functions of `(ΔΦ/h) I + G_s` on the block's bins, `CoupledExponentialKernels`
applies them -- on `numba_cpu` `NumbaCoupledExponentialKernels`, whose compiled passes
take the matrix functions from the flow actions' eigenbasis (`CoupledBlock.basis`;
`numpy_ref` keeps the augmented exponential as the reference), reached through
`ExponentialKernels.coupled()` -- and a frozen-residual section's residual joins the
compensation below,
formed from each sample's flow weights against the step's own section. The lab-frame
source runs without the flow's pipeline on every backend. The interaction formulation
and CF4 sections are refused by name (`backends.exponential.refuse_formulation`).

**Varying operators** (ADR-042): a trial takes the chord
`B_n = ΔΦ / h` as its reference; the march keeps two more rows per history slot, the
state each sample was evaluated at and its rate `D(z_j)` (`translate.linop.phi_rate`,
the derivative of the flow's own representation), and forms each trial's
compensation `(D(z_j) − B_n) u_j`, carried by `exp((1 − θ_j) ΔΦ)` and integrated with
the ordinary Adams weights (`_Compensation`), as an additive term of the predictor,
the corrector and each defect; the compiled corrector pass takes those terms. A
structurally absent source (`source_absent`) takes the chord alone.

**The engine** (ADR-043): the march keeps the controller -- the
stencil, the orders, rejection, restarts, the budget and the ledger -- and hands every
state-sized operation of the fused and exponential paths to an `AdamsEngine`: the
startup sample, a trial's predictor, source, corrector and scored norms (held until the
decision), the adoption and transport of an accepted step, holding the accepted state
across a dense observation, and the observation. `HostHistory` is the host engine,
whose passes are the kernels above; a backend may return its own from the hook
`_adams_engine`, as `native` does to keep the history resident in its
native state -- beside a coupled flow too, whose block functions, coupled predictor,
corrector, carry and defect norms are native passes over its resident eigenbasis, a
frozen-residual flow's generator applied once per sample. The general
path (PECE and the comparison variants) keeps host arrays.

**Resume** (ADR-046): an engine that can export its rows (`snapshot_rows`, and
`load_rows` to install them) lets `AdamsMarch.snapshot` return a `MarchSnapshot` -- the
accepted state, the rows, each history entry's position, kind and row, the order, the
rejection count and the proposed step -- which the run loop keeps beside every save
plane and terminal checkpoint through an output's `save_march`, and which
`AdamsMarch.load` installs at the same position on a resume. `HostHistory` exports
them, and so does the native backend's resident engine (SEP 29 §6.6): it copies its
persistent views to the host, losslessly, under the host engine's names and layout
(`store`, and `states` and `rates` for a compensated history), and a frozen-residual
flow's `G(z_j) u_j` per sample as `applied` where the host engine keeps the sample's
weights (`weights_<row>`). Each engine refuses rows it does not keep, so those two
histories do not cross; every other history does. The engines' trial and dense-step
bookkeeping is not exported: a load forgets it (`reset`), as the next step is a new
trial. Both methods are part of the `AdamsEngine` protocol. A march that cannot carry
its history (`AdamsMarch.export_refusal`: the general path), or one whose saved rows the engine
refuses (another engine's rows, another build's shapes), resumes afresh, and the driver
warns.
Nothing in the trial lifecycle reads a held trial across the snapshot: it is taken
only at an accepted position with nothing ahead of it.

**Prescribed schedules**: with a schedule the march replays its steps
(`AdamsMarch._replay_to`), accepting each whatever its estimate, which is recorded as
the Runge–Kutta replay records it; a hard kink on an edge restarts the history.
`AdamsMarch.replay_step` is also the unit a lockstep batch advances each member by
(`LockstepRunner._replay_to`), each member keeping its own history and ledger.

The derivation, the stability limits and the measured comparison are in
[SEP 13](seps/archive/sep-0013-expensive-response-steppers.md) and on the manual page
`docs/guide/numerics/adams.md`. Two differences from the Runge–Kutta march are
deliberate:

- **The norm is scaled by the state before the step.** RK uses the larger of the states
  before and after. An Adams step outside its stability region can multiply the state
  while its predictor and corrector still agree to a fraction of the new state.
- **The step is not capped at Chebyshev interval widths by default.** On the canonical
  capillary the cap cost 20–40 % of the calls for no accuracy.

## 3. Step control

The norm, the PI controller, landing and the equivalent norm are defined in
[algorithms.md §3](algorithms.md#3-step-control); this section is the design choices and
the measurements behind them.

- **Error estimate**: the embedded pair, order 4, observed order 4.03 (DOPRI5) / 3.99 (Tsit5)
  against the analytic local error in the prototype study, and 4.02 / 4.03 on solprop's own
  global-error ladder (`y' = −y` to 64 steps, both falling to four from above:
  `tests/data/convergence/tableau_order.json`). Judge an estimator on the exponent it implies, not on its
  ratio to the truth.
- **Norm**: the weak norm, one global scale, over the **norm mask** (`grids.norm_mask`).
  The mask is the controller's `region`: by default every **active** bin, the requested
  band and the margin around it where generated light propagates and the spectral collar
  absorbs it; `region="band"` scores the requested band alone. The margin's light was integrated without
  control: on a filament at three times the critical power the exponential formulation's
  order 5 got it wrong by as much as it held (8.7e-5 of the field's norm) while the band
  met `rtol=1e-6`, and zeroing the source there made the band 30 times worse, so the
  margin feeds the band. Scored, the order choice avoids it (216 calls against 231, the
  margin's error 1.0e-6). It costs the exponential formulation −7 to +5 % calls on every
  campaign case and IP-DOPRI5 +11 % on the canonical capillary and +89 % on the full RDW
  one, whose margin it integrated without control. Per-mode scaling is catastrophic: the
  in-band dynamic range is ~10¹⁷ and a third of in-band bins carry no light, so a 5e-15 error
  in an empty bin scores 7660 and the step collapses (measured in the prototype study). A *weighted* option (per-band or per-mode weights,
  rolling per-bin scales) is provided for observables far below the pump; how to score
  such observables in general is open (§9).
- **Controller**: PI.4.2 with the constants of algorithms.md §3, growth cap 1.1 (a cap of
  1.5 or 2 was measured to save 15–20 % of a gas-flow run's steps once the
  start-up rule below is on, and to cost up to 11× in field error at `rtol=1e-7` and 3–5×
  in the weak far-UV band, which the norm does not see, at `rtol=1e-4`: kept at 1.1),
  `lag` (default 1) so the step-size update is applied one step late and the norm can return
  asynchronously from a device, and `reject_free` (never reject; the lagged update absorbs
  overshoot). Measured: lag 1 costs nothing (error ×1.00–1.05, cost ×0.99–1.00); rejections
  happen 0.0–0.4 % of the time; reject-free with the cap and safety takes ~50 % more steps and
  buys 2.4–22.7× better error, so it is neutral to favourable on efficiency. On the CPU the
  default is lag 1 with rejection allowed. The same controller is used on devices;
  `reject_free=True` is an explicit option, not a backend-selected default.
- **Start-up rule** (`auto_init`, [SEP 14](seps/archive/sep-0014-stepper-efficiency.md)): the first step is `dz_init`, a guess (the
  convenience interfaces' `1e-4` of the length); the step after it is the deadbeat step its
  own scaled norm implies, `safety · norm^(−1/(q+1))`, at most 100 times the first
  (`PIController.start_factor`), and the history is given the norm the first step would
  have scored at that length so the lagged controller does not apply the margin twice.
  At the growth cap of 1.1 a thousandfold recovery from the guess otherwise takes 72
  steps. It costs nothing, since the first step is a real step, and it is taken again
  after a resume. On by default when no `dz_init` is given; off for an explicit
  `dz_init`, a coupled flow and a reject-free controller.
- **Caps**: `h ≤ min(max_dz, distance to the next hard kink, distance to the next save
  plane)`, and, for a coupled flow or a reject-free controller (`interval_cap`), the width
  of the Chebyshev interval containing z. The cumulative table (ADR-005) makes the
  diagonal flow exact across intervals, and what a step still integrates numerically, the
  source's z-dependence, its embedded estimate sees; a coupled flow's own error it does
  not, so there the cap stays (ADR-038). On a gas-flow capillary the cap
  forced µm steps at a choked end, where the table bisects to its width floor; without it
  the run takes a third fewer calls at loose and medium tolerances, and at `rtol ≤ 1e-5`
  the last plane's error can rise about twofold while the run stays within a few times
  `rtol`. Do **not** cap by the smallest interval anywhere (the prototype study measured a
  12 000× ratio between the finest local scale and the median). Neither cap
  resolves a smooth source feature narrower than the step: the table, which represents
  the cumulative Φ, does not refine around it, and both embedded formulas can step over
  it unseen (a 10 µm bump in a fibre's effective area under 87 µm steps lost its whole
  contribution to the phase, with the cap or without). Declare such a feature as a kink,
  or bound `dz_max`.
  Landing on genuine kinks matters for splitting schemes (order 2.8 vs 6.3) and is harmless
  for IP-RK; landing on save planes replaces dense output (ADR-009) unless
  `interpolate_planes=True` asks for DOPRI5's continuous extension (ADR-039): then the save planes leave the landing targets, the accepted step that
  crosses a plane is contracted into its five extension coefficients (driver hook
  `_dense_stages`), and the plane is evaluated and carried there by the exact flow while
  the march's own state is held ahead.
- **Floor**: a proposal below `dz_min` raises `ConvergenceError` at once (algorithms.md §3,
  traps). The default is `default_dz_min`, 1e-9 of the run length: a run that wants steps
  that small needs over a billion of them and would never finish, and the finest legitimate
  scale on a real fibre sits five orders above it.
- **Stretch**: a landing target within `stretch` × the capped proposal is landed on directly
  instead of taking the step and then a sliver (`StepTargets.stretch`; 1 switches it off). A
  stretch never exceeds `dz_max`.
- **The proposal is the controller's state, not the executed step.** After a clipped or
  stretched step the driver bases the next proposal on the previous *proposal*, and feeds
  the controller the norm the proposal would have scored,
  `norm · (h_proposed / h_taken)^(q+1)` (`PIController.equivalent_norm`; the same rescaling
  feeds `rejection_factor` when a clipped step is rejected). Otherwise every save plane
  restarts the controller from a sliver and it regrows at 1.1 per step: a thousandfold
  recovery takes 72 steps, more per plane than the run needs between planes.
- **The growth factor is applied once per accepted step.** `propose` updates the
  proposal after an accepted step, from the history that step completed; a rejected
  step is retried at the proposal `rejection_factor` left, with no growth on top.
  Applying the growth again before the retry would undo the shrink, and with a growth
  cap above about 1.27 a step whose norm sits just above one is refused unchanged forever.
- **Acceptance is decided on the executed step's own norm**, not on the equivalent norm: a
  landing step that was short and accurate is kept even when the proposal it was clipped
  from would have been over tolerance. The equivalent norm is what the controller's state
  sees; the history, and so the closing tolerance ratio and the "too coarse" warnings,
  record the norm each step actually scored; recording the equivalent instead would make
  a run whose last steps were forced short close as a failure.
- **Achieved vs requested tolerance** is recorded on every run: the map from `rtol` to global
  error is problem dependent and misses by one to three orders either way; calibrate, do not
  assume.

## 4. The cumulative linear operator (`translate/chebyshev.py`, `translate/linop.py`)

The three representations — constant, the z table and the parameter expansion — with
their tail test, Clenshaw evaluation, order and cost are [algorithms.md §4](algorithms.md).
This section is how they are built and what was learnt building them. The expansion is
taken when a mode's whole z dependence runs through scalar paths it declares (a Marcatili
mode's core radius a(z) and number density ρ(z)); its array-valued object is `n_terms`
arrays of the state's size, and the hard structure lives in scalar tables of 8 bytes per
term per coefficient. Anything else takes the z table. Both carry the subset of
breakpoints that are **hard kinks** (where D itself jumps or kinks: pressure-stage
boundaries, taper corners — or where a scalar path a response consumes kinks and the modes
do not declare it: a plasma channel's preionisation, an explicit density), tracked
separately from the far more numerous edges the adaptive split introduces to resolve
smooth variation.

Construction (Chebfun-style splitting, in both variables):

1. Resolve the scalar parameter paths — density ρ(z), core radius a(z), any other `ZProfile`
   — adaptively: sample at Chebyshev points, accept when the coefficient tail is below
   `tol · max|coef|`, otherwise raise the degree up to `n_max`, otherwise bisect; declared
   breakpoints are mandatory edges. This locates jumps and kinks automatically. A tail at
   the interval's **sampling-noise floor** is accepted too: a node is a floating-point z,
   uncertain by `spacing(z)`, so the samples are uncertain by `spacing(z)·|f′|`, and
   relative to the value scale that bounds what any refinement can reach
   (`SAMPLING_NOISE_FACTOR`, 4, times `spacing(z)·max|f′|/|f|`, with `|f′|` bounded from
   the coefficients). Near a choked exit, where the pressure goes as `√(z_e − z)`, the
   floor rises above `tol = 1e-12` within half a millimetre of a 4.2 m exit; without it
   the split tiled that region at its width floor, 99 intervals of 4 µm, where the
   entrance's identical cusp, near z = 0 where z is fine-grained, took eight. With it the exit refines geometrically like the entrance: the canonical
   capillary's table has 30 intervals instead of 123, Φ unchanged to 3.6e-13 rad of
   850.

   The split walks its intervals depth-first, but samples them in batches: when the walk reaches an interval not yet resolved, the intervals it is
   certain to visit (that one and every one pending) are resolved together, and below
   them the halves of every unresolved interval for four more levels, each round of
   each level in one call of the sampler and one stacked transform and tail test. The
   walk then replays its own decisions, budget included, on those outcomes, so the
   pieces are the interval-by-interval walk's, bitwise wherever the sampler is
   pointwise in z (every built-in one is). Within the budget nothing extra is sampled;
   a budget that runs out inside the looked-ahead levels leaves some resolutions
   unused. Batches are capped at 32 MiB of samples, which turns the look-ahead off for
   samples so large that one interval at the highest degree fills a batch.
2. **For the expansion**: bracket each path's range, sample `D` on the tensor grid of
   Chebyshev–Lobatto points of the parameters, and resolve each parameter axis by the same
   tail test — one global box per parameter, no splitting, because a quantity that needs
   one is not analytic there. Then build the scalar `Iⱼ(z)` with the same adaptive machinery
   from the paths, on the paths' own edges, and integrate them exactly in the Chebyshev
   basis. If an axis is unresolved at `n_max`, fall back to step 3.
3. **For the z table**: take the union of breakpoints; on each interval sample `D(ω, z)` at
   the Chebyshev nodes (cheap for Marcatili modes: three scalars per z); resolve D itself;
   integrate exactly in the Chebyshev basis; fix the constant for continuity across intervals.
4. Validate: sample D at extra z per interval and check the Chebyshev tail at a few ω near
   the band edges; check `exp(Φ(z₂) − Φ(z₁))` against `scipy.integrate.quad_vec` at random
   pairs. This is the acceptance test of either representation.
5. Budget exhaustion: the z-table builder relaxes its tolerance as described below and
   logs unresolved intervals if its budget is still exhausted. An unresolved table carries
   no stated accuracy.

Evaluation: for the expansion, one Clenshaw pass over the `(n_terms,)`-valued scalar table
and one linear combination over the state — fewer flops than the z table's *two* Clenshaw
passes at degree 64, and one pass over the state instead of two. For the z table, bisect
for the interval (scalar, host), then one Clenshaw pass over the coefficient arrays (`K`
fused multiply-adds per element). Constant and piecewise-constant D are special-cased in
either: one interval, degree 1, exact and cheapest.

Implementation requirements, each found the hard way:

- Hoist the anchor: `Φ(z_n)` once per step, reused for every stage.
- Make the round trip exact: memoise each `ΔΦ_i` and use its negation for the inverse; never
  evaluate forward and backward independently.
- Inset endpoint samples by one ulp, so a jump on a breakpoint is read from the right side
  (algorithms.md §4, traps).
- Compute `n_eff − 1` directly (the round trip through unity caps the tail at 1e-12 instead of
  2e-16).
- Apply any clamping of α to D **before** integration.
- Build to a tolerance that is measured, not assumed, **and measure it in the acceptance
  test's own norm**. The frame subtracts `β₀ + β₁(ω − ω₀)` from a β that reaches ~1e8 m⁻¹
  at the top of a wide field grid's active band, leaving a residual of a few thousand:
  about eight digits of cancellation, so one evaluation of `D` carries a relative
  round-off of ~1e-11 at the envelope whether β₁ is analytic or differenced. The analytic
  β₁ of ADR-006 still matters, for a frame that is consistent at kinks, but it does not
  lower this floor. No interpolant resolves a function below the noise of its own
  evaluation, and an adaptive search asked to do so bisects until the budget is gone.
  `build_phi` therefore measures the floor by a second difference before building and
  raises the requested tolerance to `ROUNDOFF_SAFETY` (100) times it; if the budget is
  exhausted anyway, it relaxes by decades to 1e-7 and logs each step. The floor is
  measured **per element against that element's own value**, as the tail test measures the
  tail it must bound: read against the global envelope instead, a problem whose elements
  span orders — a spectral collar beside a residual β, a dispersion-specified mode beside a
  capillary one — reports a floor set by the largest, exhausts the budget and pays the
  whole search twice.

Memory: the expansion is `n_terms × n_modes × n_pol × n_omega × 16 B` plus scalar tables of
`n_intervals × (degree+1) × n_terms × 8 B`; the z table is
`n_intervals × (degree+1) × n_modes × n_pol × n_omega × 16 B`. On the comb-to-continuum
size — six modes, 16 385 bins, a 1.5 m hydrogen gradient — that is 49.7 MB against 792 MB
for Φ and 15.1 MB against 759 MB for the prefactor, so the pair falls from 1.55 GB to
64.8 MB; `ΔΦ` falls from 17.8 ms to 1.2 ms with it. Either is shared
across batch members with the same fibre.

## 5. The co-moving frame and analytic β₁

The frame and why β₁ must be analytic are algorithms.md §4 ("The co-moving frame") and
ADR-006. The evidence: a fixed frame walks the pulse out of the window on any gradient
fibre (the prototype study measured 94 % of the peak on the window edge; co-moving puts it
at 7e-28), and a finite-difference β₁ has a relative roughness of 4.75e-8 independent of
the differencing step, which caps the whole table at 1e-8 and defeats the adaptive builder.
So the modes protocol supplies `dneff_domega` analytically (Sellmeier and Marcatili
expressions differentiated symbolically; spline-based indices differentiated exactly); the
fallback differences with Richardson extrapolation and reports its floor.

## 6. Absorbing boundaries

The collars as rates, the two temporal applications, their orders and the in-stage term's
stability limit are [algorithms.md §8](algorithms.md#8-absorbing-boundaries). The design
and the measurements behind it:

- **Default (ADR-025): the in-stage term**, `−α(t)E(t)/2` added inside every stage with its
  own modal transform, masked to the active band so that no stage writes outside the band
  the equation is solved on (which is also what keeps FSAL, §2).
- **The per-step split factor**, kept behind the same switch, is followed by the hard band
  mask, as the in-stage term is: the product in time writes into every bin, and on a field
  grid with a pulse driven into a strong collar the DC it left reached a percent of the
  spectral peak. Its first order is measured: observed orders 0.99 and 1.01 against the
  in-stage term's 4.52 and 6.16, and an absorbed energy moving by 6.5e-5 against 9.0e-10
  over a quadrupling of the steps.
- **The in-stage term's stability limit** (`boundaries.stable_step`, `h·α/2` inside 3.3 for
  DOPRI5) is enforced by the kernel problem's validation, which refuses a replayed schedule
  or a step floor already past it. With the bounded temporal profile it binds only on runs
  shorter than about `2·boundary_N` steps and the step count is the split factor's (456
  against 459 on the reduced M2 case); as the capped logarithm the temporal rate bound at
  `9·boundary_N`.
- **Geometry**: the temporal collar is a fraction of the window with a guaranteed minimum
  width (Luna's power-of-two grids had no temporal absorber at all); the run warns when the
  absorber is too weak for the group-delay walk-off the grid can produce.

## 7. Noise inside the stepper

The noise models and their cost are [algorithms.md §9](algorithms.md#9-noise). In the
stepper, the modified model's realisation is added in the fused pass that applies
`exp(ΔΦ_i)` (pass 1 of §1). The propagating field never contains the noise: each saved
plane records the noise's accumulated phase (`noise_phase`) so that the background can be
restored, and the realisation and its seed are saved.

## 8. Fixed schedules: discover, calibrate, replay

Adaptive stepping does not remove the convergence study (`rtol` is local); fixed stepping is
not less accurate, and it wins outright for ensembles (identical steps make the numerical error
a common systematic), for repeat calculations and debugging (a replay meets the run's
accuracy target) and on GPUs (no synchronisation, graph-capturable). So both are first
class:

1. **Discover**: one adaptive run records `h(z)`.
2. **Calibrate**: replay at `h`, `h/2`, `h/4`; check the observed order with a gate that
   rejects `order > p + 3` (a two-level extrapolant cannot fire; the three-level gate did fire
   in the prototype study and refused a poisoned reference); this maps `rtol` to global error once.
3. **Replay**: production runs, ensemble members and scan points (where the schedule
   transfers; see §9) replay the scaled schedule. Schedules are saved with the output and
   re-loadable.

A plain fixed-step run needs no discovery: `FixedSchedule.uniform(z_start, z_end, dz,
save_planes=..., hard_kinks=...)` lays near-equal steps of at most `dz` within every segment
between consecutive planes and kinks, so each is an edge of the schedule (there is no dense
output, so a fixed schedule must contain the planes). Replay is not blind either: the
embedded estimate costs nothing extra, so a fixed run still computes the scaled norm every
step, records it in the step history and in `ControlRecord`, and the driver logs a warning
when `met_tolerance` is false: a step was too coarse for the tolerance the norm is scaled
by. That is the fixed-mode counterpart of the `dz_min` error. The low-storage march is
the exception: it has no estimate, records 0.0, "not estimated", and is
calibrated on the observable with `calibrate(..., order=4)`; its log lines print "not
estimated (replayed schedule)" where a ratio would stand.

**What a replay reproduces.** A replay is required to meet the overall accuracy target
of the run it replays, not to reproduce its bits or every adaptive decision
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)). `FixedSchedule` stores
step lengths and a start position, nothing more. The Runge–Kutta replay takes those steps
with the same fixed-order method; the Adams replay does not repeat the original
computation: `AdamsMarch._replay_to` chooses its orders afresh, by the adaptive rule from
the replay's own estimates, and does not replay rejected trials, so where the adaptive run
rejected a step the replay may choose a different order; and a resumed march can rebuild
its history from a restart rather than carry the original one. A replay of an adaptive
run is therefore an accuracy comparison (an envelope against a tighter or stored
reference), and so is a resumed run against the uninterrupted one. Rounding-level
comparisons — backend against backend, one thread against many, a batch member against its
solo run — run **both** sides on one prescribed schedule with a fixed-order method and
matched startup, restart history and save planes ([testing.md](testing.md) section 3).
Recording and replaying every adaptive decision is not provided.

Batch lockstep execution uses fixed schedules only (ADR-013).

## 9. Open numerical questions

The solver's open questions — error norms for weak observables, reject-free efficiency at
matched error, schedule transfer, round-off growth with varying operators, save-plane
density and FSAL through a coupled flow — are listed with the others under
[numerical methods](open-questions.md#numerical-methods) in the open questions. None
blocks the current solver.

## 10. Traps (each silent, each measured)

1. Per-mode error scaling collapses the step to zero.
2. The minimum interval width is not a global step cap.
3. Stop at kinks, but only genuine kinks; track them separately from split edges.
4. Never trust a two-level extrapolant; use three levels and an order gate.
5. A vacuum-terminated fill caps every scheme at low order — the prototype study read ~3, solprop's M2
   study reads 1.4 to 1.6 ([accuracy.md §5](accuracy.md)); start at a small non-zero
   pressure.
6. Know where the asymptotic regime starts before ranking anything.
7. A complexified Kerr flow is not a group (irrelevant for IP-RK; relevant if splitting is ever
   revisited).
8. Complex splittings need a co-moving frame on physical grounds (as above).
9. Clip the ionisation rate: `(I/I_ref)^16` overflows on a diverged coarse step and a NaN
   poisons the ladder.
10. Luna's DOPRI5 weight labelling and FSAL/dense-output ordering (PR #436).
11. Luna's right-endpoint linear operator is first order in h and invisible to the error
    estimate; a zero-error estimate is not a converged answer.
12. Check the soliton recurrence **with its phase** (a global phase `π/4` is asserted, not
    fitted); amplitude-only metrics understate the error by 10–140×.

## 11. Verification ladder for solprop's solver tests

| rung | problem | truth | asserts |
|---|---|---|---|
| P0 | linear only, nasty D(ω, z) (pressure step, taper, absorbing edge) | `quad_vec` per ω to machine precision | Φ table 1e-14; right-endpoint rule reproduces `(h/2)ΔD` as a regression witness |
| P1 | N-soliton recurrence, constant D, N = 3…8 | analytic, phase-resolved | observed order 5 above the floor; phase `π/4`; error at ×4 steps ~1e-10 |
| P2 | N-soliton with arbitrary z-varying d(z) and α(z), γ = d·e^Γ | analytic | dissipative, z-varying, exact; step-function α (P2b); out-of-band absorber (P2a) |
| P3 | full UPPE: Kerr → shock → plasma → Raman; RDW parameters | three-level Richardson with order gate; cross-scheme reference | efficiency at matched error; weak-bin metric |
| targets | RDW VUV 4-mode gradient; CtC H₂ 6-mode | stored Luna results and solprop's own converged references | regression within stated tolerance; convergence ladder |
