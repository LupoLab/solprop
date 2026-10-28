# SEP 39 — A native Adams march: the Multistep step loop in the compiled backend

| | |
|---|---|
| **SEP** | 39 |
| **Title** | A native Adams march: the Multistep step loop in the compiled backend |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-28) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-28 |
| **Resolution** | — |
| **Requires** | [SEP 16](archive/sep-0016-unified-multistep.md) (the march, its engine, ADR-043); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); [SEP 29](archive/sep-0029-compiled-backend-distribution.md) (native z tables and the history export) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Move the step loop of `Multistep()` from solprop's Python `AdamsMarch` into
`solprop-native`. That loop is the step-size controller, the variable-order choice, the
proposal caps and landings, the coefficient tables and the stage scalars of every trial.
On the native backend a run of accepted steps would then cross the nanobind seam once
per **host event**: a save plane, a boundary-monitor check, a statistics row, the end of
a march. Today it crosses two to four times per trial and spends 0.1–0.5 ms of Python
on each one.

The recommended design (option B, §6.2) keeps every policy in solprop. solprop hands
the backend one record holding every controller and landing parameter, plus the events
at which it must return. Native then executes an algorithm that this SEP (§4) and the
kernel contract specify in full. The Python march stays: it is the reference, and it
still runs on `numpy_ref` and `numba_cpu`.

The case for it, from a profiling study of 2026-10-28:

- The Python side of a trial costs 110–140 µs on an Apple M5 Max and 390–510 µs on an
  AMD EPYC 7763. On the EPYC that is 1.5 to 1.7 source evaluations for the cheap
  mode-averaged capillaries.
- The native march would remove about nine tenths of it.

Success means:

- on the tracking set's capillaries, the native march takes the same steps as the
  Python march on the same engine (a diagnostic, never a promise; SEP 25);
- it meets each case's accuracy envelope;
- with a quiet boundary monitor, its host cost per trial is at most 15 µs on the M5 and
  40 µs on the EPYC;
- it is nowhere slower than the Python march beyond the A/B spread, including on
  expensive sources and runs that return every step (§7.6).

Together with the native-side work that needs no SEP (§6.9), the estimates are:

- the single-mode helium capillary comes level with `RungeKutta()` on Apple silicon, and
  within 1.2–1.3 times of it at one thread on the EPYC;
- the argon blue-shift capillary stays about twice as slow as Runge–Kutta, because it
  takes more source calls than Runge–Kutta does.

This SEP changes no default, and it does not decide the capillary default.

### 2. Motivation and scope

**The use case** is the default integrator of `prop_capillary`, which is `Multistep()`
at `rtol` 1e-6 (ADR-055). [SEP 37](archive/sep-0037-noise-and-stepper-defaults.md)'s
phase 0 found `RungeKutta()` faster on 8 of 14 capillary rows at one thread and on 11 of
14 at sixteen. Multistep won the four-mode `rdw` and the noiseless comb-to-continuum.
The maintainers kept the capillary default and asked where Multistep's per-step time
goes.

The 2026-10-28 profiling study of the native CPU backend measured that time
(§3.3, with hardware). Its findings:

- **Host cost per trial.** A Multistep trial spends 110–140 µs in Python on the M5
  and 390–510 µs on the EPYC. The EPYC's host side alone is 0.4–0.5 ms of a 1.1–2.1 ms
  trial.
- **Runge–Kutta's host cost** is 10 µs per call on the M5 and 40–50 µs on the EPYC,
  spread over six calls a step.
- **Native passes.** Beyond its one source call, a trial's native passes cost 0.9 to
  3.9 source calls by machine and case. Part of that is being removed by ordinary
  optimisations (§6.9).
- **The Python pieces** on the EPYC, per step:
  - the stage-scalar packing, twice per trial: 130–156 µs;
  - the coefficient tables: 126–134 µs;
  - the compensation tables: 34 µs;
  - the controller's decision: 39–47 µs;
  - the accept: about 20 µs;
  - the rate weights: 22 µs.
- **Moving the step loop** to native removes the largest remaining host item on every
  single-mode case. `native/AGENTS.md` keeps the controller in solprop, which is why
  this is a SEP.

**Who needs it.** Every native-backend run with `Multistep()` on a source cheap enough
for the host to matter. The method is the default of `prop_capillary`, `prop_radial` and
`prop_planar`. That covers:

- the mode-averaged and single-mode capillaries;
- scans of them, where the same small run repeats many times;
- device builds, where SEP 16 found that "about 26 launches and 0.2 ms of host work"
  per trial is what a cheap source cannot hide.

The maintainer wants every bit of performance available here. This SEP is the largest
host-side lever, and §6.9 places it among the native levers.

**In scope:**

- a pure, data-driven controller core in solprop, which the Python march and the native
  march's tests share (phase 0);
- native coefficient tables and native stage scalars (phase 1). These are useful on
  their own and are the first half of the march;
- the native march for the exponential formulation (the default) on the resident engine
  (phase 2), over:
  - diagonal operators, whether constant, an expansion or a z table, with the
    transported compensation;
  - the carrier frame;
  - the in-stage collar and the quiet collar;
  - dense observation and hard kinks;
  - `step_planes`, `max_calls` and statistics rows;
  - the boundary monitor;
  - snapshots and resume;
- the interaction formulation's fused path, and replayed schedules of single runs
  (phase 3);
- the resident coupled-flow sections and moving maps (phase 4). The aim is to cover
  every configuration that has a resident engine (Q2);
- device builds (phase 5), qualified before `Final` (Q6);
- the seam: a march-policy record, the scalar tables, the host events, one metadata
  field;
- the amendment of `native/AGENTS.md`'s policy rule and of ADR-043 (§6.3).

**Not in scope:**

- **Any change to the method, the controller's algorithm, its constants or any
  default.** The native march executes the Python march's algorithm, and
  `native/AGENTS.md` forbids tuning the controller for speed.
- **The capillary default, and every other default.** Outside this SEP's scope
  (decided, Q10).
- **Fewer steps on the argon blue-shift cases.** Multistep takes 11 169 calls there
  against Runge–Kutta's 10 105, so no overhead work wins that case (§4.1). That needs
  a method change, not an execution change.
- **Runge–Kutta's host loop.** Its cost per call is small. A native Runge–Kutta march
  is not planned, and can be taken up another time if it is needed (decided, Q8).
- **The fixed 90–190 ms per run** that goes to metadata, provenance, output and plane
  observation. It is the same for both methods. This draft adds it to the roadmap as
  its own item.
- **The general Adams path** (`ComparisonMultistep`: PECE, explicit Adams–Bashforth,
  matched pairs), lockstep batches, and a compiled march for `numba_cpu` (§6.6).
- **A controller resident on a GPU.** On device builds the controller runs on the host
  thread (§6.7).
- **The native optimisations of §6.9.** They need no SEP and proceed independently.

### 3. Context

#### 3.1 The march today

`solprop.backends.multistep.AdamsMarch` is the step loop of every backend
([stepper.md](../stepper.md), "Low-call Adams methods"). ADR-043
([decisions.md](../decisions.md)) split it from its arithmetic:

> `AdamsMarch` keeps the controller -- the stencil, orders, rejection, restarts, the
> call budget, the ledger, schedules and lockstep -- and hands every state-sized
> operation of its fused and exponential paths to an engine.

The reason it gave: "one controller for every backend keeps the order, rejection and
lifecycle rules in one place, tested once."

On `native` the engine is `_ResidentAdams` (`native/src/solprop_native/backend.py`). The
history, the predictor, the source, the corrector and the defect norms are resident.
`GeneralWorkspace::adams_exponential_trial` (`native/src/general.cpp`) evaluates the
source in C++ inside the same call: it runs the stage pipeline with the GIL released and
returns at most three norms. What Python still does per exponential trial:

1. hoists the anchor (a Clenshaw pass per expansion);
2. forms `exponential_tables`, plus `compensation_tables` and the endpoint's rate
   weights on a varying operator;
3. forms two stage-scalar bundles, one for the step and one for the lab-frame sample,
   packed into fresh arrays (`_resident_arrays`, `_trial_arrays`, `_weight_rows`);
4. forms the carrier shift from the reference element's chord;
5. makes the one native trial call, whose binding copies about 28 arguments;
6. runs the controller (`_stencil`, `_decide`, `_choose_order`, `_reject`, `_factor`,
   `limit_step`), the proposal caps (`_capped`, `_quiet_capped`, `_collar_reentry`) and
   the landing rule (`stepper.schedule.landing`, `_approach`);
7. makes one native accept call when the trial is accepted;
8. adds one or two `table_evaluate` calls on a z-table run;
9. runs the driver's per-step bookkeeping:
   - `_record` on every attempt (history row, progress);
   - `_take_step_stats` on every accepted step (the monitor's `due`/`skip`, statistics
     rows);
   - `_natural_stop`;
   - `committing()` around the accept.

The low-storage march of
[SEP 24](archive/sep-0024-low-storage-interaction-picture-runge-kutta.md) already makes
a step one native call (`_ResidentLowStorage.step`). It only replays a prescribed
schedule and keeps no estimate (ADR-050), so it has no controller to move. It is the
precedent for the commit pattern (§6.5.7), not for policy.

#### 3.2 What the earlier proposals measured and left

- **[SEP 14](archive/sep-0014-stepper-efficiency.md), phase 2.** A Kokkos Adams attempt
  on the canonical capillary took 0.67 ms, of which 0.29 ms (43 %) was the host march.
  The native step was "left for a later change".
- **[SEP 16](archive/sep-0016-unified-multistep.md).**
  - On a device, compiled Clenshaw and Lagrange helpers took the canonical trial's host
    work from 0.50 ms to 0.13 ms.
  - On the coupled wall, a quarter of a step (0.37 ms) was "the controller's and the
    tables' Python, which a device's faster kernels will not shrink".
- **SEP 25 §6.3** notes that Kokkos "still pays host table evaluation, controller work
  and records" per trial.
- **[backends.md](../backends.md) §3.5** rejected a compiled step driver because the
  per-call overhead was below the noise floor. That was measured on Runge–Kutta, where
  a step makes six calls and spreads the cost. An Adams step makes one call and pays
  all of it.

#### 3.3 The measurements this SEP starts from

**Method.** Measured on main at 6216178, as complete runs of three tracking cases,
timed by the benchmark worker after its warm-up, with the suite's thread pools and the
`passive` OpenMP policy (ADR-056). The cases:

- `he_rdw_averaged`: helium on a pressure gradient, shape (1, 1, 1, 2561). Its
  expansion operator gives the exponential trial the transported compensation.
- `argon_blueshift_averaged`: argon at constant pressure, shape (1, 1, 1, 1601).
- `rdw`: four modes on the helium gradient, shape (1, 4, 1, 2561).

How the costs were taken:

- **Native time per labelled kernel:** a host-only Kokkos Tools timer.
- **Host time:** timers wrapped around the march's Python pieces. The host gap is wall
  time less Kokkos time.
- **Hardware:**
  - Apple M5 Max: macOS, Homebrew Clang 23.1.3, `-mcpu=native`, a shared workstation,
    so these numbers are indicative;
  - AMD EPYC 7763: Linux, 16 cores of one NUMA domain, GCC 16.2, glibc 2.34, OpenBLAS
    0.3.34, FFTW 3.3.11, `-march=native`.
- **A/B comparisons:** fresh processes, interleaved over three rounds, medians.
- **The run's fixed host cost** is excluded from the per-trial figures below. It is
  90–190 ms per run (122 ms on the EPYC) for metadata, provenance, output and plane
  observation, and it is the same for both methods.

**Per trial (µs), one thread.** "Passes" are the exponential trial's native work beyond
its source: phase, moments and predictor (with the compensation), corrector, defect
norms, history, mask and copies. RK's figure is per source call, with its host share in
brackets.

| Case | Machine | Source | Passes | Host | Trial | RK per call |
|---|---|---:|---:|---:|---:|---:|
| `he_rdw_averaged` | Apple M5 Max | 174 | 306 | 140 | 623 | 219 (10) |
| `argon_blueshift_averaged` | Apple M5 Max | 121 | 110 | 108 | 338 | 138 (9) |
| `rdw` | Apple M5 Max | 2631 | 1352 | 195 | 4123 | 2505 (34) |
| `rdw`, 16 threads | Apple M5 Max | 1382 | 971 | 228 | 2508 | 1083 (34) |
| `he_rdw_averaged` | AMD EPYC 7763 | 323 | 1274 | ≈500 | ≈2100 | 441 (49) |
| `argon_blueshift_averaged` | AMD EPYC 7763 | 239 | 475 | ≈405 | ≈1120 | 291 (39) |

Notes on the table:

- The pieces come from separate profiles, so a row's sum differs from its trial by up to
  3 %.
- The single-column cases do not thread at sixteen: they lie below the stage pool's size
  threshold (8192 elements on macOS, 4096 on Linux). Their sixteen-thread host costs
  were 134 µs (helium) and 111 µs (argon) on the M5.
- On the EPYC the host side is a quarter to a third of a trial.
- The coefficient tables alone cost 126–134 µs a step on the EPYC and 15–34 µs on the
  M5.

**Complete runs, baseline (s).**

| Case | Machine, threads | `Multistep()` | `RungeKutta()` |
|---|---|---:|---:|
| `he_rdw_averaged` | Apple M5 Max, 1 | 0.967 | 0.705 |
| `he_rdw_averaged` | Apple M5 Max, 16 | 0.973 | 0.692 |
| `argon_blueshift_averaged` | Apple M5 Max, 1 | 3.863 | 1.451 |
| `argon_blueshift_averaged` | Apple M5 Max, 16 | 3.840 | 1.465 |
| `rdw` | Apple M5 Max, 16 | 4.627 | 11.60 |
| `he_rdw_averaged` | AMD EPYC 7763, 1 | 3.150 | 1.389 |
| `argon_blueshift_averaged` | AMD EPYC 7763, 1 | 12.61 | 3.012 |
| `rdw` | AMD EPYC 7763, 1 | 22.03 | 59.06 |
| `rdw` | AMD EPYC 7763, 16 | 21.32 | 47.43 |
| `he_rdw_averaged` | AMD EPYC 7763, 16, shared node | 3.392 | 1.654 |
| `argon_blueshift_averaged` | AMD EPYC 7763, 16, shared node | 12.60 | 2.969 |

The `rdw` RK figure on the M5 at sixteen threads comes from a profile run and was not
interleaved with the others.

**Steps and calls.**

- `argon_blueshift_averaged`:
  - Multistep: 11 168 trials, 11 169 source calls.
  - Runge–Kutta: 1 684 steps, 10 105 calls.
- `he_rdw_averaged`:
  - Multistep: about 1 400 trials.
  - Runge–Kutta: about 2 700–2 900 calls.
- `rdw`:
  - Multistep: about 1 800 trials.
  - Runge–Kutta: about 10 500 calls.

The helium and `rdw` counts are inferred from each run's time and its per-trial or
per-call cost, so read them as about ±5 %. Phase 0 replaces them with counted values,
in a public benchmark entry (§7.6).

**Prototypes.** Every prototype took the baseline's accepted and rejected steps on every
case, with final fields within 4e-10 of the baseline. They are listed with their results
in §6.9.

#### 3.4 The rules this SEP meets

- **`native/AGENTS.md`, Product and architecture:**

  > The layers are solprop -> Python backend adapter -> nanobind seam -> a Python-free
  > C++20 core. Native code never imports Python or solprop concepts and never owns
  > user-facing policy.

- **The same file, Numerical and performance discipline:** "never tune the step-size
  controller to gain speed"; and a trial "performs no allocation, planning, logging,
  Python call or unplanned fence".
- **ADR-043:** one controller for every backend (§3.1).
- **ADR-047:** "Native code is not interrupted: a signal waits for a native call to
  return."
- **ADR-054:** the controller, scalars and schedules are binary64 on the host. A native
  controller keeps both properties: binary64 arithmetic, run on the host thread.
- **SEP 0 §7:** backends in order (`numpy_ref`, then `numba_cpu`, then native), and
  parity across all three at `Final` or an exception written into the accepted SEP.

### 4. Theory

This section is the algorithm the native march executes, written so that it can be
implemented and checked without reading `backends/multistep.py`. The Python march is its
reference implementation, and a disagreement between the two is a defect in one of them.
Notation follows SEP 13 and SEP 16:

- $z$ is the position in m and $h$ a step in m.
- $\theta = (z' - z)/h$ is a normalised position inside the step anchored at $z$.
- $x = \Phi(z + h) - \Phi(z)$ is the step's exponent, in the internal convention
  ([conventions.md](../conventions.md)).

#### 4.1 Why the host matters: the cost of a run

A Multistep trial costs

$$
T_{\mathrm{MS}} = C_N + C_A + C_H,
$$

where:

- $C_N$ is the source evaluation;
- $C_A$ is the Adams passes;
- $C_H$ is the host side.

A Runge–Kutta call costs $T_{\mathrm{RK}} = C_N' + c_H$, where $C_N'$ includes the
stage's entry propagator and $c_H$ is the host's share of one call. Multistep is faster
when

$$
\frac{n_{\mathrm{MS}}}{n_{\mathrm{RK}}} < \frac{T_{\mathrm{RK}}}{T_{\mathrm{MS}}},
$$

with $n$ the source calls of each method. The measurements of §3.3 give these ratios:

| Case | $n_{\mathrm{MS}}/n_{\mathrm{RK}}$ | $T_{\mathrm{RK}}/T_{\mathrm{MS}}$, M5 | $T_{\mathrm{RK}}/T_{\mathrm{MS}}$, EPYC | The same with $C_A = C_H = 0$, M5 / EPYC |
|---|---:|---:|---:|---:|
| `he_rdw_averaged` | ≈0.50 | 0.35 | 0.21 | 1.26 / 1.37 |
| `argon_blueshift_averaged` | 1.105 | 0.41 | 0.26 | 1.14 / 1.22 |
| `rdw` (M5 at 16 threads) | ≈0.17 | 0.43 | — | 0.78 / — |

Three conclusions follow.

- **Helium:** Multistep makes half the calls and loses only through $C_A + C_H$. The
  host is $C_H/T_{\mathrm{MS}}$ = 22 % of a trial on the M5 and 24 % on the EPYC.
- **Argon:** even at zero overhead Multistep would be at most 3–10 % faster (1.105
  against 1.14–1.22), and any real $C_A$ loses that. That case needs fewer steps.
- **`rdw`:** Multistep already wins, and cutting $C_A$ or $C_H$ only widens the margin.

#### 4.2 The march's policy and state

**The policy** $\mathcal{P}$ is everything the algorithm reads but never changes. solprop
forms it once per run, from the method (`Multistep`), the run's stepper settings
([kernel-contract.md](../kernel-contract.md) §7) and the march's module constants:

| Symbol | Meaning | Source today |
|---|---|---|
| $q_{\max}$ | highest predictor order, 1–6 | `Multistep.order` (5) |
| — | variable order | `Multistep.variable_order` (true) |
| $s$, $g$, $f_{\min}$ | safety, growth cap, smallest factor | `Multistep.safety` (0.8), `max_growth` (2), `min_factor` (0.2) |
| $g_{\mathrm{rej}}$ | growth cap of a rejection's factor | the literal 0.9 in `AdamsMarch._reject` |
| $H$ | order hysteresis | `ORDER_HYSTERESIS` (1.2) |
| $R$ | consecutive rejections before a history restart | `RESTART_AFTER` (3) |
| $c$, $P_k$ | collar stability share; PEC negative-real interval of order $k$ | `COLLAR_STABILITY` (1); `PEC_REAL_INTERVAL` |
| $\rho$ | the in-stage collar's largest rate, $\max\alpha/2$ in m⁻¹; 0 without one, outside the exponential formulation, or under the split scheme | `AdamsMarch._collar_rate` |
| $C_\kappa$ | carrier phase cap, rad, when a carrier frame is active | `CARRIER_PHASE_CAP` (1) |
| $B$ | the call budget, or none | `Multistep.max_calls` |
| — | equal approach; interval cap | `Multistep.equal_approach`; `interval_cap` |
| $h_0$, $h_{\min}$, $h_{\max}$, $\sigma$ | first proposal, floor, cap, stretch | `dz_init`, `dz_min`, `dz_max`, `stretch` |
| $z_{\mathrm{end}}$ | end of the run | `z_end` |
| $\mathcal{Z}_s$, $\mathcal{Z}_k$, $\mathcal{Z}_b$ | save planes, hard kinks, interval edges | `save_planes` (empty when planes are interpolated), `linear.hard_kinks`, `linear.phi_breakpoints` (or its first and last entry without the interval cap) |
| $u$ | the landing tolerance in units in the last place | `LANDING_ULPS` (8) |
| $r_{\mathrm{tol}}$, $a_{\mathrm{tol}}$, $w$ | the norm's scale | `PIController.scale` (the PI gains are Runge–Kutta's and unused here) |
| $I_Q$ | the quiet collar's step cap | `BoundaryMonitor.interval` |

**The state** $\mathcal{S}$ is everything the algorithm changes:

- the position $z$ and the proposal $h_p$;
- the order $q$;
- the history entries $(z_j, \iota_j, \kappa_j)$, oldest first, with $\iota_j$ the
  identity (accepted or predicted) and $\kappa_j$ the engine row;
- the consecutive rejections $r$;
- whether the last attempt was rejected, $\lambda$;
- whether the collar was quiet at the last attempt, $w_Q$;
- the call ledger: `startup`, `predictor`, `rejected`, `accepted_steps`,
  `rejected_steps`, `restarts`, `failure_restarts`, `dense_observations`, and the order
  of every accepted step.

Two inputs come from the host at every event: the collar's quiet flag $Q$, set by the
boundary monitor, and the event bounds of §4.8.

**Comparison semantics.** Wherever this section writes $\min$ or $\max$ of two values,
it means exactly what the reference does. $\min(a, b)$ is $b$ if $b < a$, else $a$.
$\max(a, b)$ is $b$ if $b > a$, else $a$. $\arg\max$ over a list keeps the first element
unless a later one compares strictly greater. These forms decide what a NaN does, and
they equal C++'s `std::min` and `std::max` of two arguments. A comparison with a NaN is
false.

#### 4.3 One attempt

Given $\mathcal{S}$ and a march target $z_T$:

1. **Budget.** The attempt costs $c = [\text{history empty}] + 1$ calls. If $B$ is set
   and the calls charged so far plus $c$ exceed $B$, stop with the event *budget*. The
   host raises `ResponseBudgetError` with the last accepted position and state.
2. **Start.** If the history is empty:
   - add one to `startup`, before the evaluation, as the reference does: a call that
     fails has still been charged (R4);
   - evaluate the source once at the accepted state;
   - set the history to $[(z, \text{accepted}, 0)]$;
   - set $q = 1$ under variable order, otherwise $q = q_{\max}$.
3. **Proposal.** Apply three caps in turn:
   - **The carrier cap.** With a carrier frame active, let
     $e = \min(z + h_p, z_{\mathrm{end}})$ and let $\phi$ be the chord phase of the
     reference element over $[z, e]$. If $\phi > C_\kappa$, set
     $h_1 = (e - z)\,C_\kappa/\phi$; otherwise $h_1 = h_p$.
   - **The quiet cap.** $h_2 = \min(h_1, I_Q)$ while $Q$; otherwise $h_2 = h_1$.
   - **Collar re-entry.** If the collar was quiet at the last attempt and is not now, and
     $\rho > 0$, set $h_3 = \min(h_2, c P_q/\rho)$; otherwise $h_3 = h_2$. Then record
     the current $Q$ as $w_Q$.
4. **Landing** (§4.6). This gives the step $h$, its end $z_e$, and whether $z_e$ is a
   hard kink.
5. **Stencil.**
   - Nodes $\theta_j = (z_j - z)/h$ and rows $\kappa_j$, newest first; $n$ entries held.
   - The step's order is $q' = \min(q, n)$.
   - It may rise when the last attempt was not rejected and more entries are held than
     it uses: $\mathrm{rising} = \neg\lambda \wedge n > q'$.
   - The scored orders are $O = [q']$, followed, under variable order, by $q' - 1$ and
     then $q' + 1$, each kept when $1 \le k \le q_{\max}$, $k \le n$ and
     ($k < q'$ or rising).
6. **Trial.**
   - Add one to `predictor`, before anything is evaluated, as the reference does.
   - Form the coefficient tables of §4.7 from the nodes.
   - Evaluate the source once.
   - Reduce the scaled defect norms $\varepsilon_k$ for $k \in O$ (SEP 16 §4.2;
     [stepper.md](../stepper.md)). The scale is $r_{\mathrm{tol}}\,w\,\lVert u_n \rVert
     + a_{\mathrm{tol}}$, with $u_n$ the accepted state before the step, over the norm
     mask.
7. **Decide** (§4.4), and record the attempt (§4.8).

Steps 6 and 7 are the engine's and the controller's; everything else is the march's.

**A prescribed schedule** (phase 3) is `AdamsMarch._replay_to`, with its own stopping,
landing and kink rules (R6). The state adds the schedule's cursor $m$, which is
`StepDriver._schedule_index`; a load rebuilds it from the position
(`_schedule_index_at`), as today. The march uses:

- the schedule's precomputed edges $E$ (`FixedSchedule.edges`, the running sum of its
  steps);
- the slack $\delta$ = `replay_tolerance(schedule, z_T)`. This is
  $\max(u, n_{\mathrm{steps}})$ units in the last place of the larger of $z_T$ and
  the schedule's length, because a sum of $n$ rounded steps drifts by about $n$ ulps.

While $m < n_{\mathrm{steps}}$ and $z < z_T - \delta$, each attempt:

1. takes $h$ = `steps[m]` and $z_e = E_{m+1}$, with no caps and no landing rule;
2. treats $z_e$ as a kink when some hard kink lies within $\delta$ of it;
3. runs steps 1, 2, 5 and 6, accepts every finite norm, and appends it to the accepted
   norms as `_replay_to` does;
4. advances $m$ in the same commit as the position.

A non-finite norm is the event *diverged*, which the host raises as `ConvergenceError`
with the reference's message. If $z < z_T - \delta$ when the schedule is used up, the
event is *exhausted*, which the host raises as the reference's `BackendError`. The
adaptive test `reached` is not used on a replay: the schedule ending 12 ulps short of
`1e-3` in `replay_tolerance`'s own example must still count as arriving.

#### 4.4 Acceptance, rejection and the step-size factor

The factor an order-$k$ defect allows is SEP 13's elementary controller: the exponent is
the defect's own, $1/(k+1)$ [1, 5].

$$
F(\varepsilon, k, \gamma) =
\begin{cases}
\gamma & \varepsilon \le 0, \\
\min\!\big(\gamma,\ \max(f_{\min},\ s\,\varepsilon^{-1/(k+1)})\big) & \text{otherwise.}
\end{cases}
$$

The next proposal is `limit_step`:

$$
h' = \min(h\,f,\ h_{\max}).
$$

If $h' < h_{\min}$, stop with the event *floor*. The host raises `ConvergenceError` with
the reference's message.

Let $\varepsilon = \varepsilon_{q'}$. Reject when $\neg(\varepsilon \le 1)$; a NaN
rejects. On a rejection:

1. add one to `rejected`, `rejected_steps` and $r$, and set $\lambda$;
2. if $r \ge R$ and more than one entry is held, empty the history, set $r = 0$ and add
   one to `failure_restarts`;
3. take the factor $f = f_{\min}$ if $\varepsilon$ is not finite, otherwise
   $F(\varepsilon, q', g_{\mathrm{rej}})$;
4. set $h_p = $ `limit_step`$(h, f)$.

On an acceptance, in one commit (§6.5.7):

1. keep the $q_{\max}$ newest entries;
2. give the new sample the lowest engine row the kept entries do not use;
3. accept in the engine: the step's finish, boundary treatments included, and the store
   of the new sample with its state and rate on a compensated history;
4. append $(z + h, \text{predicted}, \text{row})$;
5. add one to `accepted_steps`, append $q'$ to the ledger's orders, and set $r = 0$;
6. choose the stored order $q_s$, which the next stencil starts from, and the control
   order $q_c$ with its norm $\varepsilon_c$, which set the next proposal's exponent
   (§4.5);
7. set $h_p = $ `limit_step`$(h, F(\varepsilon_c, q_c, g))$, set $q = q_s$, and clear
   $\lambda$;
8. move to $z = z_e$. At a hard kink, empty the history and add one to `restarts`.

**A defect in the reference's commit order.** `AdamsMarch._decide` tests the floor in
step 7 before it moves the position in step 8. A floor raised there leaves the adopted
state at $z + h$ with the position still at $z$.

It can happen. A landing may be shorter than $h_{\min}$, since `landing` never consults
the floor, and the next proposal is formed from the step taken. So an accepted landing
shorter than $h_{\min}/g$ always raises the floor.

The Runge–Kutta march does neither: it commits the state and position first, and grows
its next proposal from the proposal rather than from the landed step.

Phase 0 moves the position into the commit before the floor is tested, so that the state
and position always agree, and the native march follows that order. Before the floor is
raised, the accepted attempt's row, the kink restart and the ledger are published to the
host like any other accepted step (§6.5.7, R3). The rule itself stays for now. If the
implementation finds short landings in practice, it may change the rule, recording the
change and its evidence in the phase report (decided, Q13).

#### 4.5 Order choice

**Under a fixed order** the two orders differ: $q_s = q_{\max}$, but $q_c = q'$ and
$\varepsilon_c = \varepsilon_{q'}$. This is `AdamsMarch._choose_order`, which stores
`method.order` and returns the step's own order.

At startup and after a restart, $q'$ is below $q_{\max}$ until the history fills. The
exponent is then the step's own, not $q_{\max}$'s. With `Multistep(order=5,
variable_order=False)`, one sample held and $\varepsilon = 0.25$, the factor is 1.6;
taking $q_{\max}$ would give 1.0079 (the AI review's probe, R1).

**Under variable order** $q_s = q_c$ is the chosen order $q_n$ below, and
$\varepsilon_c = \varepsilon_{q_n}$. Every scored order $k$ has the weight

$$
W_k = \min\!\big(s\,\varepsilon_k^{-1/(k+1)},\ \Sigma_k\big),
\qquad
\Sigma_k =
\begin{cases}
\dfrac{c\,P_k}{\rho\,h} & \rho > 0 \text{ and } \neg Q, \\[4pt]
\infty & \text{otherwise.}
\end{cases}
$$

- $\varepsilon_k^{-1/(k+1)}$ is taken as $\infty$ when $\varepsilon_k \le 0$.
- $\Sigma_k$ is the step the in-stage collar leaves order $k$ stable at (SEP 16 phase 6;
  ADR-057 for the quiet collar).
- $b = \arg\max_{k \in O} W_k$, taken in $O$'s order.

The march adopts $q_n = b$ when $b \ne q'$ and $W_b > H\,W_{q'}$; otherwise
$q_n = q'$. The cap $\Sigma_k$ weighs only the choice, and the step is the chosen order's
defect's (SEP 16 phase 6). Without the cap, near the collar's limit the choice was
decided by rounding. That is why §4.9 treats order changes as knife edges.

#### 4.6 Landings, caps and equal approach

The landing rule is `stepper.schedule.landing`
([stepper.md](../stepper.md) §3). Its tolerance is
$\tau = u\,\mathrm{ulp}(\max(|z|, |z_{\mathrm{end},T}|))$, where:

- $\mathrm{ulp}(y)$ is the spacing of binary64 at $y$ (`numpy.spacing`);
- $z_{\mathrm{end},T}$ is the run's end when planes are interpolated, and the march
  target $z_T$ otherwise.

The rule:

- **Caps.** $(h_3, \text{none})$, $(h_{\max}, \text{max step})$ and (the width of the
  interval of $\mathcal{Z}_b$ containing $z$, interval width).
- **Landings.** The first hard kink, the first save plane, and $z_{\mathrm{end},T}$, each
  taken strictly beyond $z + \tau$.
- **Choice.** Both sets choose the shortest, with ties going to the later of
  none < max step < interval width < hard kink < save plane < end.
- If the shortest landing gap is within the shortest cap, land on it.
- Otherwise, if it is within $\min(\sigma \cdot \text{cap}, h_{\max})$, stretch the step
  to land on it.
- Otherwise take the cap and land nowhere.

**Equal approach.** If the step lands nowhere and equal approach is on, find the next
forced landing $a$: the nearest of $z_T$, the next save plane and the next kink beyond
$z + \tau'$, with $\tau' = u\,\mathrm{ulp}(\max(|z|, |z_T|))$. If $h < a - z < 2h$, take
$h = (a - z)/2$, end at $z + h$, with no kink.

**Kinks.** Otherwise the end is the landing's target, or $z + h$. It is a kink when:

- it is interior, meaning it lies below $z_{\mathrm{end}} - \tau''$; and
- either the landing was a hard kink, or some kink lies within $\tau''$ of it.

Here $\tau'' = u\,\mathrm{ulp}(\max(|z_e|, 1))$.

#### 4.7 Coefficient tables

Every table is a small dense computation on at most $q_{\max} + 2 \le 8$ nodes.

**Lagrange coefficients.** For distinct nodes $\vartheta_0, \dots, \vartheta_{m-1}$ they
are the matrix $L$ with $\ell_i(\theta) = \sum_p L_{ip}\theta^p$ and
$\ell_i(\vartheta_j) = \delta_{ij}$. That is, $L = V^{-\mathsf{T}}$ for the Vandermonde
matrix $V_{jp} = \vartheta_j^p$. The integrated weights are

$$
w_i = \int_0^1 \ell_i(\theta)\,\mathrm{d}\theta = \sum_p \frac{L_{ip}}{p+1}.
$$

They are exact for polynomials of degree below $m$ on any distinct nodes [1, 2].

**The exponential trial** (SEP 16 §4.2) integrates the interpolant against the flow. The
moments are

$$
\mu_p(x) = \int_0^1 e^{(1-\theta)x}\,\theta^p\,\mathrm{d}\theta = p!\,\varphi_{p+1}(x)
$$

[3, 4]. The engine forms them per element. The host supplies:

- the predictor's $L$ over $(\theta_0, \dots, \theta_{q'-1})$;
- the corrector's $L$ over $(1, \theta_0, \dots, \theta_{q'-1})$;
- for every scored order $k$, the factored defect: the divided-difference coefficients
  $\gamma_t = 1/\prod_{s\ne t}(p_t - p_s)$ over $p = (1, \theta_0, \dots,
  \theta_{k-1})$, and the ascending monomial coefficients of
  $\pi_k(\theta) = \prod_{j<k}(\theta - \theta_j)$.

**The transported compensation** (ADR-042) adds:

- the spans $1 - \theta_j$;
- the integrated weights over the predictor's nodes;
- the integrated weights over the corrector's nodes;
- each scored order's corrector weights less its predictor weights.

**The interaction formulation's fused path** (phase 3) needs the predictor's integrated
weights, the corrector's, and the same defect rows (`fused_tables`).

**Accuracy.** No useful conditioning bound holds over the histories the march admits
(R7):

- a landing may shorten a step by any ratio while an ordinary save plane keeps the
  history;
- rejections compound $f_{\min}$;
- `max_growth`, `min_factor` and `equal_approach` are settings.

With four unit steps followed by a trial shortened to $10^{-10}$, the corrector's nodes
are $(1, 0, -10^{10}, \dots, -4\times10^{10})$. Their monomial Vandermonde matrix has
an $\infty$-norm condition estimate of about $2\times10^{53}$. That bound says nothing
useful about the error of the algorithms actually used (products of roots and divided
differences), or of the weights a trial applies.

The native tables use the algorithm of `stepper.multistep` (`_basis_rows`, already
native and strict as `_core.basis_rows`). They are therefore not held to a
condition-number budget. Instead, both the native tables and the reference's are
compared with an exact oracle: the same tables computed in rational arithmetic from the
binary64 nodes, with Python's `fractions.Fraction` and no new dependency (§7.2).

This SEP proposes no new restart or admission rule for extreme node ratios. If the
corpus shows the reference losing accuracy on histories the march admits, that is
reported to the author as a question of numerical policy.

#### 4.8 Stage scalars along z, and the host events

**Stage scalars.** A trial evaluates its source at $z + h$ and reads the stage scalars
there ([kernel-contract.md](../kernel-contract.md); `translate.stage`): the weights of
$\Phi$ and of the prefactor, the density, the taper scale, the Raman decay rates and
densities, the Kerr density, the plasma channels' densities and pre-ionised fractions,
and the dispersive Kerr densities. The flow section and weights, and a moving guide's
transform weights, are phase 4.

Each scalar is a piecewise Chebyshev table $s(z) = \sum_n a^{(i)}_n T_n(\xi)$ on interval
$i$, evaluated by Clenshaw's recurrence. The weights of an expansion operator are
integrated Chebyshev tables, and the increment's weights are $W(z_1) - W(z_0)$. A
varying operator's rate $D(z)$ is the derivative of the same representation, taken on
the closing side at a kink (`translate.linop.phi_rate`, `before=True`). A z table is
already evaluated natively into term slots (SEP 29 §6.6).

The native march evaluates every table in C++ with the strict Clenshaw helper that the
native backend's `Helpers` already use (`_core.clenshaw_points`). Its bundles therefore
equal the ones the backend forms today through `translate.stage.scalars_at`, to the bit.
That is a structural property, tested exactly (§7.1). The tables are uploaded once per
run.

**Host events.** The march returns to the host only when the host has work, and every
return is drained completely (R5).

- **At entry,** the host does what `march_to` does today before its loop:
  - it restores a march held ahead of a dense observation;
  - it returns at once if a natural stop is due (`_natural_stop_due`) or the target is
    reached;
  - otherwise it passes the bounds below.

  No call is made when no attempt is due.
- **Before each attempt,** native tests the budget (§4.3, step 1).
- **After each attempt,** accepted or rejected, it tests the call bound, and returns any
  error.
- **After each accepted attempt,** it tests the accepted-step events, with the host's
  own expressions:

| Event | Native test after an accepted attempt |
|---|---|
| target reached | `reached(z, z_T)`: $z \ge z_T - \tau$ |
| crossed (dense observation) | $z > z_T$ with planes interpolated |
| host step | the accepted-step count since entry reaches $n_e$, the smallest of: the steps to the next statistics row (`row_period`), to the next natural stop (`step_planes`), and to the boundary monitor's next cadence check (1 while an edge is busy, otherwise `CADENCE` less the steps since its last check) |
| monitor distance | $z - z_{\mathrm{last}} \ge I_{\mathrm{check}}$, the monitor's own expression, or its first check |

The other returns have no event step:

| Return | When it is tested |
|---|---|
| call bound | after any attempt: the attempts or the wall time since entry reach the call's bounds (§6.5.7) |
| budget | before an attempt (§4.3, step 1) |
| floor | when a proposal is formed, after a rejection or an acceptance (§4.4) |
| diverged, exhausted | on a replay (§4.3) |
| error | wherever a native failure occurs. The steps committed before it stand, and the error is returned rather than thrown (§6.5.7) |

Every attempt writes one row into a host buffer: end, step, norm (with a non-finite norm
written as $\infty$, as `_record` does), whether it was accepted, its order, and its wall
time since entry. A call can return:

- with no rows (the budget, or an error, at its first attempt);
- with rows but no acceptance (rejections up to the call bound);
- with accepted steps followed by rejections.

**Draining.** Inside the commit of §6.5.7, the host ingests every row exactly once, in
order. It appends the step-history row, and for an accepted row applies that step's
accounting exactly once: the monitor's skip, the statistics and natural-stop counters,
and `_accepted_norms` on a replay.

Only one accepted row can have host work: the **event step**, the accepted attempt
after which native returned with an accepted-step event. By construction no earlier
accepted row had any. For the event step the host runs the reference's own sequence,
`_take_step_stats` then `_natural_stop`, unchanged. So coincident events are serviced
in the reference's order and exactly once: for example, a plane that is also a
statistics row and a monitor check, or a natural stop due on a dense crossing.

A return for any other reason has no event step, and its accepted rows take the skip
path only. The other reasons are the call bound, the budget, the floor, divergence,
exhaustion and an error.

**After the commit,** the host:

1. makes the progress callbacks, one per attempt in order, so a reporter's display
   updates once per call (Q12);
2. writes the debug lines;
3. does the event step's work;
4. raises the error last, if there is one.

So every statistics row, monitor check and natural stop falls on the accepted step it
falls on today.

**The backend's accepted-step hook is not replayed.** On native, `_on_accepted_step`
calls `promote_fsal`, which scores the quadrature alarm from the step's buffers. The
native march does that scoring itself after each accepted step, where the buffers are
current, and the host's ingestion does not call the hook.

#### 4.9 What "the same steps" means

Both marches read norms from the same native kernels. They differ only in:

- the coefficient tables, and so the norms formed with them;
- the order of the scalar operations that combine them.

How far the norms differ is measured per case, not assumed (R8). Phase 1 records, on
every qualification case, the distribution of $|\Delta\varepsilon|/\varepsilon$
between trials run with the host's tables and with the native ones.

The controller's own arithmetic is the same IEEE operations on the same inputs: strict,
with the platform's `pow` (§6.5.1) [7, 8]. Its outputs are discontinuous functions of
the norms: acceptance at $\varepsilon = 1$, adoption at $W_b = H W_{q'}$, and the
landing and kink tolerances.

A decision flips only when the two marches' inputs straddle a threshold, that is, when
its margin is below their difference. Such flips are rare, but they are not impossible.
After one flip the two trajectories differ by more than rounding, and they agree only
within the case's accuracy envelope (R10).

So, as SEP 25 §4.4 rules for adaptive runs, "the same steps" is:

- **expected:** the same accepted and rejected sequence, orders and landings on every
  qualification case;
- **recorded:** with the smallest decision margin of each run;
- **never asserted:** the assertions are an accuracy envelope on the fields and exact
  structural contracts (§7).

A prescribed schedule with a fixed order has no decisions, so it is a matched
computation that agrees to a rounding budget (§7.2).

### 5. Proposed interface

No call or setting changes. A user's script, configuration and output read as before. On
the native backend, `Multistep()` runs through the native march whenever the
configuration is admitted (§6.6). Otherwise it runs through the Python march as today
and warns once per run with a new `SolpropWarning` subclass in `solprop.exceptions`,
the one public addition, naming the reason (Q2).

The run's metadata records which march ran, so that a timing or a rounding difference
can be traced:

```text
RunMetadata.adams_march: "native" | "host" | None   # None for Runge–Kutta runs
```

A private keyword on the native backend's constructor (`adams_march="host"`) forces the
Python march. Its users:

- the parity tests;
- the benchmark harness's A/B comparisons;
- anyone developing the controller, who can change it in Python and run it on native
  without a rebuild.

It is not part of the documented interface.

### 6. Implementation strategies

#### 6.1 Three options, costed

**(A) The controller and the march fully in C++, with the Python `AdamsMarch` kept as
the reference.** Native carries the algorithm and its constants. Its defaults (safety,
hysteresis, restart count, stability intervals, the landing rule's tolerances) are
written in C++ beside the Python ones, and the host passes only the method's fields and
the run's settings.

**(B) Policy as data, mechanism native.** solprop forms one record $\mathcal{P}$ per run
(§4.2) holding every parameter and constant the algorithm reads. That includes those
that are module constants today: the hysteresis, the restart count, the stability
intervals, the carrier cap, the rejection's growth cap and the landing tolerance. At
every entry solprop also passes the event bounds and the quiet flag (§4.8).

Native holds no default and no constant of its own. It refuses a record that lacks a
field or holds one out of range, by name. It advances under the algorithm of §4 to the
next host event. The algorithm is written in the kernel contract and versioned by
`SCHEMA_VERSION` like every other part of the seam, and the Python march stays its
reference.

**(C) No controller move.** Python keeps the decision. Native does two things:

- **Forms the coefficient tables** from the nodes inside the trial call.
- **Evaluates the stage scalars** from tables uploaded once. Python passes only the
  trial's two positions and its phase increment.

The trial and the accept stay separate calls. Fusing the accept into the next trial call
was considered and dropped (R2). The accepted state would wait for a next trial that may
never come: a save plane, a natural stop, a spent budget, the end of the run or an
interruption can intervene, and a monitor verdict can change the next trial's collar
setting after the state must already have been adopted. Phase 2 removes those crossings
anyway, because it decides and commits inside the native call.

**What each option takes off a trial**, per trial, from today's host cost (§3.3).

**Option C.** Its parts:

- the tables become a native helper (−15 to −34 µs on the M5, −126 to −168 µs on the
  EPYC with the compensation's tables);
- the stage scalars are evaluated natively (−130 to −156 µs on the EPYC);
- the rate weights go native (−22 µs).

The decision (39–47 µs on the EPYC), the march loop, `_record`, the step bookkeeping
and two crossings per accepted trial stay.

**Options A and B** remove all of it but a residual:

- the controller's scalar work in C++, a few µs;
- one crossing per host event. Events come every eighth accepted step or more often by
  the monitor's distance on a quiet window, and at every step while an edge is busy.
- the bulk records, about 1–2 µs an attempt.

Per-trial savings:

| | C | A | B |
|---|---:|---:|---:|
| Saving per trial, Apple M5 Max (helium / argon), µs | 50–80 / 40–70 | 125–135 / 93–103 | 125–135 / 93–103 |
| Saving per trial, AMD EPYC 7763 (helium / argon), µs | 265–335 / 240–280 | 460–485 / 365–390 | 460–485 / 365–390 |

Host time left per trial, and the run-level effect:

| | C | A | B |
|---|---|---|---|
| Host time left per trial, M5 / EPYC, µs | 40–90 / 125–235 | 5–15 / 15–40 | 5–15 / 15–40 |
| Seam crossings | two per accepted trial | one per host event | one per host event |
| Run-level saving alone, helium (EPYC, of 3.15 s) | 0.38–0.48 s | 0.65–0.69 s | 0.65–0.69 s |
| Run-level saving alone, argon (EPYC, of 12.6 s) | 2.7–3.1 s | 4.1–4.4 s | 4.1–4.4 s |

Ownership, rules and risk:

| | C | A | B |
|---|---|---|---|
| Who owns the parameters | solprop | native, with solprop mirroring it | solprop; native has none |
| `native/AGENTS.md` | kept | broken: native owns policy | kept in intent; its wording amended (§6.3) |
| ADR-043 | kept | replaced | amended: one algorithm, two executions, Python the reference |
| Risk | leaves a third of the EPYC's host cost | two sets of constants that drift apart | a specification and a trace test to maintain |

The M5 savings for C take the M5's measured table cost (15–34 µs) and scale the other
EPYC pieces by the measured 3.5–4 times. The helium figures include the compensation's
tables and rate weights; argon has a constant operator and needs neither. The residual
for A and B assumes events every eighth accepted step or so. Busy monitors,
`step_planes = 1` and `row_period = 1` return every step, which phase 2 measures (§7.6,
R12).

**What B adds over C.** Accepting a second implementation of the controller rests on
B's gain over a completed phase 1, not over today (R12). From the same pieces:

| Case | Apple M5 Max | AMD EPYC 7763 |
|---|---|---|
| `he_rdw_averaged` | 45–85 µs per trial, 0.06–0.12 s per run | 125–220 µs per trial, 0.18–0.31 s per run |
| `argon_blueshift_averaged` | 25–65 µs per trial, 0.28–0.73 s per run | 85–150 µs per trial, 0.95–1.7 s per run |

That is 7–14 % of the helium run and 9–23 % of the argon run once the items of §6.9 have
landed. These are estimates; phase 2 measures them with N1–N7 held fixed (§7.6).

#### 6.2 Recommendation: B, built on C

B and A cost the same in time and in C++, since both put the whole loop in native code.
They differ in who owns the numbers:

- **Under A**, a change to a constant, or a default chosen in C++, can drift from the
  Python march. A reviewer has to read C++ to know what a run did.
- **Under B**, there is one source of every number (solprop's record), one specification
  of the algorithm (the kernel contract), and a test that drives both implementations
  over the same norm sequences and compares every output exactly (§7.1).

C is not an alternative to B. It is B's first phase: the tables and the stage scalars
must be native before the loop can be. As a stopping point, C is weaker for three
reasons:

- On the EPYC it leaves 125–235 µs of Python per trial, still 0.5–0.7 source calls on
  the cheap cases.
- It keeps two crossings per accepted trial.
- On a GPU it keeps the Python decision between the norm read-back and the next launch.

**Recommended:** B, built in the phases of §11 with C's pieces first.

#### 6.3 Does B satisfy `native/AGENTS.md`?

The rule has two parts.

**"Never imports Python or solprop concepts."** B satisfies this. The record is plain
data validated at the seam, as a `KernelProblem` is. The march calls no Python, and the
events return to the host instead.

**"Never owns user-facing policy."** By intent, B satisfies this too:

- Every choice a user can see or set is made in solprop: the method and its fields, the
  tolerances, the defaults, the landings, admission and refusal, warnings, the
  monitor's verdicts, and the error messages.
- Native executes an algorithm that the contract specifies in full, as it already
  executes the stage arithmetic the contract specifies.

The rule's wording is ambiguous, though. The controller's *structure* is behaviour a
user observes in the step counts: that the order never rises after a rejection, that
three rejections restart the history. A later reader could take B as a breach.

This SEP therefore amends the rule, in phase 2, to:

> Native code never imports Python or solprop concepts and never owns user-facing
> policy: it may execute an algorithm the kernel contract specifies in full -- the Adams
> march's controller, order choice and landings among them (SEP 39) -- with every
> parameter handed over by solprop and solprop's implementation as the reference, and
> it never chooses or defaults a parameter itself.

"Never tune the step-size controller to gain speed" stays as it is. B tunes nothing.

ADR-043 is amended by a new ADR in phase 2. Its reason, "one controller for every
backend ... tested once", becomes:

- **one specification** of the controller, in the kernel contract;
- **one reference implementation**, solprop's;
- **one native execution**, held to the reference by exact decision-trace tests and by
  accuracy envelopes on whole runs.

Option A would break the rule and need its removal. That is the main reason it is not
recommended.

#### 6.4 Layers and the seam

| Layer | What it holds |
|---|---|
| Physics (P) | Nothing changes. |
| Translation (T) | Uploads the scalar tables once per run, as a new leaf of the native problem (phase 1). |
| Backends (B), solprop | `stepper.multistep` gains `MarchPolicy`, a frozen dataclass of §4.2's fields, formed by `march_policy(method, settings, …)`. `backends.multistep` gains a pure controller core (§11, phase 0), which `AdamsMarch` calls. |
| Backends (B), native adapter | `_NativeAdamsMarch` in `solprop_native.backend`. It configures the native march once per run, calls it per march, applies the host events, and maps events to solprop's exceptions. |
| Backends (B), C++ | The march, beside `GeneralWorkspace`'s Adams methods. It owns the state $\mathcal{S}$ and reuses the resident engine's trial and accept passes. |
| Driver (D) | `run_problem` and the record are unchanged, but for `RunMetadata.adams_march`. |

The seam changes:

- **Phase 1.** Kernel schema +1:
  - the scalar tables leaf;
  - the trial call takes the nodes and the scored orders instead of the tables.
  - A contract tuple `SCALAR_TABLES_FIELDS` is added.
- **Phase 2.** Kernel schema +1 and output schema +1:
  - `MarchPolicy` and its tuple `MARCH_POLICY_FIELDS`;
  - the driver's bulk bookkeeping hooks, which join `DRIVER_SURFACE` (for example
    `_record_attempts`, and a counted form of `_take_step_stats`; the names are the
    implementer's);
  - `RunMetadata.adams_march`.
  - `check_build()` checks the new tuples at import, as it does the others.
- **Phase 3.** The interaction formulation's fused trial needs no seam change. A replay
  passes the schedule's steps and edges, which the stepper settings already carry.

The exact schema numbers are those current at each phase; a change under review at the
time of drafting moves the kernel schema to 38.

#### 6.5 The native march, item by item

##### 6.5.1 The controller

§4.3–§4.5, in binary64 on the host thread (ADR-054), as scalar C++ in a host-only
translation unit of its own. Its inputs are the norms the engine already reduces.

**Strict on every build** (R8). That unit is compiled with contraction off and without
fast-math on every build, through a per-source compile option rather than `fp.hpp`'s
macros. The macros do not suffice here:

- `SOLPROP_NATIVE_STRICT_FUNCTION` alone is empty under Clang, which needs
  `SOLPROP_NATIVE_STRICT_BLOCK` in the body;
- both macros are empty under CUDA and HIP, including their host pass.

A test in the style of `tests/native/test_fp.cpp` checks, on each supported compiler,
that a product and a sum in that unit round separately. The compilers are GCC, Clang,
the Windows toolchain, and in phase 5 the CUDA and HIP host compilers.

**One elementary function.** The controller calls only `pow`, from the platform's C
library, as Python's float power does.

The norm's scale must be the engine's own: `adams_scale` from `PIController.scale`,
which today's `_device_scale` path already forms natively. A controller with another
`scale` runs on the host march. The PI gains are Runge–Kutta's; the Adams march has
always used the elementary factor of §4.4.

##### 6.5.2 Variable order

§4.5. The collar weighting needs only $\rho$, $c$, $P_k$, $Q$ and $h$, all of which the
march holds. The carrier cap needs the reference element's chord phase over
$[z, e]$. Native already evaluates the reference element's $\Phi$ from its resident
terms, or from a z table's slots, so the chord is formed natively. Today the host forms
it for `ExponentialWeights`.

##### 6.5.3 Startup, rejections, hard kinks, restarts

§4.3–§4.6:

- A failure restart and a kink restart empty the history. The next attempt starts it
  again, paying one startup call.
- A rejection leaves the accepted history untouched, as the engine's held trial already
  guarantees (SEP 13 §6.1).

##### 6.5.4 Landings and dense observation

§4.6, with the save planes, kinks, interval edges and run end in $\mathcal{P}$, and the
march target passed per call. Under interpolated planes:

1. the march returns *crossed* when an accepted step passes $z_T$;
2. the host holds the engine, observes and records as today (`_ahead`, the engine's
   `hold`, `observe`, `restore`; ADR-036, ADR-039);
3. the next call resumes from the held state.

##### 6.5.5 Varying operators

- **z tables.** Their native slots (SEP 29 G1) are read in-process instead of through
  `_ZTables.load`. The least-recently-used slot set moves into the march.
- **Rate weights.** For an expansion, the closing-side derivative of the integrated
  weight tables. The host precomputes the derivative tables at translation and uploads
  them with the scalar tables. For a z table, the rate slot native already evaluates.
- **The transported compensation.** The engine's existing passes (ADR-042), with the
  spans and weights of §4.7 formed natively. The sample's rate row is stored by the
  accept from the endpoint rate the trial formed.

##### 6.5.6 Stage scalars along gas profiles

Three ways were weighed:

- **Evaluate natively** (recommended). §4.8: the same strict Clenshaw pass, so the
  bundles are identical.
- **Tabulate** on a fine grid at translation. Rejected: the values would differ beyond
  rounding, the grid would need its own error control, and memory grows with profile
  detail.
- **Keep on the host.** Rejected: it is the largest single host item on the EPYC, and it
  would need a host callback per trial, which `native/AGENTS.md` forbids from native
  code.

The Raman decay rates already drive a native rebuild of the transformed kernel spectrum
when they change (`prepare_raman_stage`), so nothing moves there.

##### 6.5.7 Commits, signals, errors and the monitor

**Commits.** Each accepted step commits in native code, before the next attempt: the
state, the history, the position and the ledger change together. On the host, one
`committing()` guard spans the whole exchange (R3):

1. the native call;
2. the synchronisation of the host's position, proposal, order, history and ledger with
   the native march;
3. the ingestion of every returned row (§4.8, draining).

A signal that arrives during the call is held until all three are done, and only then
raised (ADR-047). Without the guard around the call itself, `solprop._interrupts` would
raise as soon as Python resumed, with native several commits ahead of the host's
records. SEP 24's atomic step guards its native call the same way.

User code runs only after the guard: the progress callbacks, the debug lines and the
event step's work (§4.8). A callback that raises therefore finds the host's records
complete and in step with the native state, and the run loop's terminal checkpoint
writes a consistent state, position, history and ledger. The floor (§4.4) follows the
same rule: the accepted attempt, its kink restart and its ledger are published before
the floor is raised.

**Signals.** A signal waits for the native call to return (ADR-047), so a call is
bounded:

- by its attempts: the size of the row buffer, 256;
- by its wall time, from a monotonic clock read once per attempt: 0.1 s (decided, Q3).

Neither is a user setting. A crossing costs an estimated 20–40 µs.

- In default runs the monitor's cadence returns at least every eighth accepted step, so
  the call bound rarely binds.
- When it does, 256 attempts of a trial of 0.3–1 ms take 0.08–0.3 s, and the crossing is
  at most 0.05 % of that. For a 50 µs trial it would be 0.3 %.

Phase 2 measures the crossings actually made and their cost (§7.6, R12).

A scheduler's `SIGTERM` or a keyboard interrupt is then served within that bound, plus
one trial.

**Errors.** A native failure is caught inside the march: in an evaluation, a reduction,
an accept or a scalar-table evaluation. The march stops at the last commit and returns
an *error* record (R4) that holds:

- **the stage:** the startup evaluation, the trial's evaluation, the trial's reductions,
  the accept, or the stage scalars;
- **the category:** a code for the exception class the host march would raise there.
  These are the classes the adapter raises today: `ConvergenceError` from an envelope
  failure (`_envelope_errors`), `BackendError`, and the built-in errors of the bindings.
  The class is never inferred from the message.
- **the message.**

The ledger returned with it keeps the charges already made. A startup or trial call is
charged before it is evaluated (§4.3), so a failed call is counted, as the reference
counts it. The host:

1. synchronises the ledger;
2. drains the committed rows;
3. raises the class the code names.

An exception thrown through nanobind in the middle of a call would lose the records of
steps already committed.

**`ConvergenceError` and `ResponseBudgetError`** are raised by the host from the *floor*,
*diverged* and *budget* events, with the reference's messages. The budget error carries
the last accepted state.

**The boundary monitor** keeps its policy in Python (`solprop.monitor`): its cadence, its
thresholds, the quiet verdict and the stop.

- The march returns when a check is due (§4.8). Busy edges therefore return every step,
  and quiet ones every `CADENCE` steps or by distance.
- A change under review moves the monitor's reading into the backend, which then hands
  only five numbers and the split's removal to the host. With it, a check costs one
  crossing and a few µs of Python.
- The quiet flag $Q$ is passed at every entry. A change of verdict reaches the stages
  through `_collar_changed`, as today.

##### 6.5.8 The call ledger, records and metadata

- **The ledger.** The ledger counters accumulate natively and are read at every event
  into `CallLedger`. The orders of the accepted steps come in the attempt rows, so the
  ledger counts the calls actually made (SEP 25 §4.4).
- **The step history's rows.** These are `_record`'s rows. Each attempt's wall time is
  the call's start, read on the host with `perf_counter`, plus the native monotonic
  offset.
- **Progress.** The reporter receives one `step` call per attempt, in order, in bulk
  after each native call. Its display therefore updates once per call and lags by at
  most one call (Q12).
- **The per-step debug log** of the Python march is kept. When the `DEBUG` level is
  enabled, the host writes one line per attempt from the rows at each event, guarded by
  `isEnabledFor`, so the lines cost nothing otherwise (decided, Q12).
- **The record and the configuration hash** are unchanged. The metadata adds
  `adams_march`.

##### 6.5.9 Resume snapshots

`MarchSnapshot` keeps its fields (ADR-046; SEP 29 §6.6) and gains one: the collar's
quiet flag at the last attempt, $w_Q$ (R9). At an event the host reads the march state:

- the position and the proposal;
- the order and the rejection count;
- $\lambda$ and $w_Q$;
- the entries' positions, identities and rows.

It reads the engine's rows through the existing `adams_rows_to_host`, losslessly. A load
installs both, restores $w_Q$, then calls `reset` on the held trial. A snapshot is
taken only at an event with nothing ahead of it, as today.

**Why $w_Q$.** It decides the collar's re-entry cap (§4.3). Today's snapshot does not
hold it, so a resumed run starts with it false and skips a cap the uninterrupted run
would take. And `begin_run` does not reset it, so a backend reused for another run
starts with the last run's value. Phase 0 adds the field (output schema +1) and resets
it in `begin_run`, in the Python march first (decided, R9).

**Where a snapshot continues.**

| Snapshot from | Native march | Host march on native | `numpy_ref`, `numba_cpu` |
|---|---|---|---|
| Native march | yes | yes: the same engine and rows | yes, except a frozen-residual coupled flow's history (`applied` rows), which the host engines refuse by name, so it restarts, as today |
| Host march on native | yes | yes | the same as the row above |
| `numpy_ref`, `numba_cpu` | yes, except a frozen-residual flow's history (`weights_<row>`), refused by name, as today | the same as the column to the left | yes |

The existing named refusals stay. No conversion between the two frozen-residual forms is
proposed.

**What a resume promises** (R10).

- **The transfer is exact.** The rows, positions, identities, slots, order, controller
  state and $w_Q$ equal the stored ones after a round trip (SEP 29 R17).
- **The continuation is an adaptive run.** A resumed run meets the case's accuracy
  envelope against a converged reference, as SEP 25 §4.4 rules. It need not take the
  uninterrupted run's steps.
- **A rounding budget applies only to matched comparisons:** a fixed-order schedule with
  matched history and plane handling.

#### 6.6 Backend order, exceptions and refusals

The method is not new. `Multistep()` already runs on `numpy_ref`, `numba_cpu` and
`native`, and the Python `AdamsMarch` is its oracle on all three. This SEP adds a native
*execution* of the existing march, as SEP 16 added the native engine. The backend order
of SEP 0 §7 is met as follows:

- **`numpy_ref` first** (phase 0). The controller is factored into a pure, data-driven
  core and the policy record, with no change in behaviour. The decision-trace corpus of
  §7.1 is generated from it.
- **`numba_cpu`** keeps the Python march. This is the exception this SEP records:
  - Its per-trial Python cost is of the same kind.
  - But a compiled march there would have to call FFTW plans and the stage pipeline from
    nopython code, which is the compiled step driver backends.md §3.5 decided against.
  - Compiling only the decision would save about a tenth of the host cost and keep every
    crossing.
  - `numba_cpu` is being demoted, and backend parity of the march is not needed. No
    roadmap item is added (decided, Q9).
- **Native** gets the march (phases 1–5).

**Admission** (R11). One matrix decides how each configuration runs on the native
backend at `Final`. It covers CPU and device builds alike, since phase 5 comes before
`Final` (Q6). A configuration is matched on its formulation, its corrector and
interpolant, whether its engine is resident, its norm scale and the build, never on the
method's type alone. `ComparisonMultistep` is a subclass of `Multistep`, and its
one-order-higher PEC variants take the same resident path.

| Configuration | At `Final` | Phase |
|---|---|---|
| `Multistep()`, exponential, on a constant, expansion or z-table operator, with the transported compensation, carrier frame, collars, interpolated planes, kinks and budget | native march | 2 (CPU), 5 (device) |
| `Multistep(formulation="interaction")` | native march | 3, 5 |
| `ComparisonMultistep` with the one-order-higher PEC corrector (`corrector="pec"`, `corrector_order = order + 1`), fixed or variable order, with the Adams or the Hermite interpolant | native march: the resident fused path, whose observation already takes the Hermite rule | 3, 5 |
| A prescribed schedule, single run | native march | 3, 5 |
| A coupled flow whose sections all lie on the eigenbasis route, at most 16 channels; a moving transform map | native march | 4, 5 |
| `ComparisonMultistep`'s general path: PECE, explicit Adams–Bashforth, matched pairs | warned host march (host arrays, as today) | — |
| A coupled flow off the eigenbasis route or wider than 16 channels (the host engine, ADR-044) | warned host march | — |
| A norm scale other than `PIController.scale` | warned host march | — |
| Lockstep batch replay (`LockstepRunner`, `replay_step`) | warned host march | — |
| The untransported compensation (a private test witness); the private `adams_march="host"` switch (§5) | host march, no warning | — |
| Everything `_refuse_adams` and `_refuse_unsupported` refuse today: TDSE, single precision, CF4, a batch axis, the interaction formulation beside a coupled flow; on device builds the carrier frame and z tables | refused by name, unchanged | — |

**The warning** (decided, Q2). A run that takes the warned host march warns once, when
the backend is constructed, with a `SolpropWarning` subclass, for example
`HostMarchWarning`. It names the reason and the native march's coverage, and the choice
is recorded in `adams_march`.

This is not a fallback between backends: both marches execute the same method on the
same backend, and each meets the case's accuracy envelope (R10). The same already holds
when `_adams_engine` returns the host engine for a flow off the eigenbasis route
(ADR-044). Warnings are errors in the test suite, so a test of a warned configuration on
native asserts the warning.

**While the phases are under way,** the rows of later phases run the warned host march,
devices until phase 5 included. That is a state of the branch, not of `Final`.

Inside the march, a record with a missing or out-of-range field is refused by name with
a `BackendError` before anything is allocated.

#### 6.7 GPU builds

On a device build the march runs on the host thread. For each trial it:

1. forms the tables and scalars in C++ (a few µs);
2. uploads them in one copy, from a pinned host buffer instead of today's two unpinned
   ones;
3. launches the trial's kernels;
4. reads back at most three norms (one planned synchronisation, already there today);
5. decides, then launches the accept.

The divisor is already formed on the device (`adams_upload_divisor`), so the only
read-back is the norms.

SEP 16 measured 0.13–0.2 ms of host Python per trial on a device, against kernels of
3–9 µs. The march removes nearly all of that. About 26 launches per trial remain, which
kernel fusion (§6.9) or graph capture would address later. A controller resident on the
device is out of scope (§10).

Device qualification follows the standing policy ([backends.md](../backends.md)): in one
pass, in a GPU segment with its own budget, as phase 5 and before `Final` (Q6). Until then
a device build runs the host march, records it and warns (Q2).

#### 6.8 Cost: time, memory, code

- **Memory.** The attempt-row buffer (256 × 7 doubles), the march state (a few hundred
  bytes) and the scalar tables (kilobytes) are all listed in `memory_ledger()`. Nothing
  state-sized is added.
- **Time.** This is the per-trial saving of §6.1. It scales with the number of trials,
  not with the grid, so it matters most on cheap single-mode sources and least on
  expensive multimode ones.
- **Code.**
  - About 1 500–2 500 lines of C++: the march, the tables, the scalar evaluation, the
    events, the bindings and the C++ tests.
  - About 500 lines of Python: the controller core, the policy record and the adapter.
  - A parity test matrix.

The lasting cost is two implementations of one algorithm. Any change to the controller
becomes a seam change: a schema bump, and both halves changed in the same commit. The
host march stays available on native for controller research.

#### 6.9 The native work that needs no SEP, and where the march sits

These items change no algorithm and leave results unchanged to rounding, or exactly.
They follow the working agreement of `AGENTS.md`, each with a benchmark ledger entry.
Two are already under way as ordinary changes.

| Item | Results | Measured or estimated gain |
|---|---|---|
| **N1. Fewer threaded regions.** The trial as three passes: phase, magnitude, moments and prediction as one reduction; the source; correction and defect norms as one reduction, with the divisor inline. The restore copy becomes a view swap, and the accept's mask and store become one pass. Cheap passes stay separate where they do not thread. | rounding | `rdw` at 16 threads: −25 % (M5), −29 % (EPYC). Nothing at one thread, where the passes are compute-bound. |
| **N2. A threading size class for the Adams passes.** Sized by elements × moments: per element the passes cost about ten times a stage entry, so the stage pool's threshold leaves them serial on every single-column case. | rounding (reduction order) | On top of N1 at 16 threads: helium −24 points (M5, a tie with RK), −8 (EPYC); argon 0 (M5), −7 (EPYC). |
| **N3. `Range2` tiling.** Host tiles of (1, 256) instead of Kokkos' default, which gives OpenMP ⌈rows/2⌉ tiles. Under way. | exact | `rdw` at 16 threads: −14 % (M5), −19 % (EPYC). It also reaches Runge–Kutta's `Range2` kernels. |
| **N4. The φ-functions' stopping test and bookkeeping.** Squared magnitudes, exact `2^-k` and `1/n` tables, no `pow`, `log2` or `hypot`. Under way. | rounding (bit-identical on `rdw` on the EPYC) | One thread: −4 to −6 % (M5), −8 to −11 % (EPYC). |
| **N5a. The newest sample's carry is the flow.** $e^{(1-\theta_0)x}$ with $\theta_0 = 0$ is $e^{x}$, already formed, so one of the compensation's six exponentials per element is free. | exact | about 1/6 of the compensation's exponentials (≈40 µs of the helium trial on the EPYC; estimated) |
| **N5b. Vector `exp`/`sincos`.** Over a pass that writes all of a step's exponentials to a buffer: glibc's libmvec on Linux, Accelerate's vForce on macOS. GCC uses libmvec only under fast-math, which is forbidden, so the calls are explicit. | changes values by a few units in the last place per exponential: requalified under SEP 25 (budgets re-derived, work–precision cases re-run) | The largest native lever left on Linux for the gradient cases. On the EPYC, transcendentals are about two thirds of the 0.77 ms helium predict pass: compensation 43 %, the phase and propagator remainder 24 %. On the M5 they are about a third. Estimated −0.25 to −0.35 ms per helium trial on the EPYC. SEP 25 measured libmvec at about 3 % on the RK cases and did not adopt it; here the share is far larger. |
| **N6. `-fno-math-errno`.** | **not value-identical under GCC**: with the same steps the final helium field moved by 1.9e-10 relative and argon's by 9e-14 (GCC 16.2), within SEP 25's rules but to be stated when adopted | −1.9 to −2.7 % alone, −12 to −14 % with N4 (EPYC, one thread) |
| **N7. Host micro-work.** The factored defect rows without `numpy.polynomial.polyfromroots` (25 → 5 µs a call, coefficients within 2 ulp), and the resident trial's stage rows packed once (identical arrays). Subsumed by this SEP's phase 1. | rounding / exact | −19 and −10 µs a step (M5) |

**Where the march sits.** The table below estimates complete runs with everything done:

- the column "N items" applies N1–N7 (§3.3's measured prototype combinations where they
  exist, plus the estimates above for N5);
- the next column adds the native march (B), whose saving is the host cost left after
  N7, less a residual of 5–15 µs (M5) or 15–40 µs (EPYC) per trial, times the trial
  count of §3.3.

Ranges carry the spread of the measurements and of the unmeasured items. These are
estimates, not measurements.

| Case | Machine | Threads | MS today | MS, N items | MS, N + march | RK today | MS / RK after |
|---|---|---:|---:|---:|---:|---:|---:|
| `he_rdw_averaged` | M5 Max | 1 | 0.967 | 0.86–0.90 | 0.69–0.75 | 0.705 | 0.98–1.06 |
| `he_rdw_averaged` | M5 Max | 16 | 0.973 | 0.665–0.69 | 0.50–0.54 | 0.692 | 0.72–0.78 |
| `argon_blueshift_averaged` | M5 Max | 1 | 3.863 | 3.18–3.30 | 2.31–2.54 | 1.451 | 1.6–1.75 |
| `argon_blueshift_averaged` | M5 Max | 16 | 3.840 | 3.25–3.50 | 2.35–2.70 | 1.465 | 1.6–1.85 |
| `rdw` | M5 Max | 1 | ≈7.5 | ≈6.9–7.2 | ≈6.6–6.9 | ≈26.5 | ≈0.25–0.26 |
| `rdw` | M5 Max | 16 | 4.627 | 3.08–3.28 | 2.73–2.94 | 11.60 | 0.24–0.25 |
| `he_rdw_averaged` | EPYC 7763 | 1 | 3.150 | 2.21–2.35 | 1.64–1.81 | 1.389 | 1.2–1.3 |
| `he_rdw_averaged` | EPYC 7763 | 16 | 3.392 | 2.12–2.46 | 1.55–1.93 | 1.654 | 0.95–1.15 |
| `argon_blueshift_averaged` | EPYC 7763 | 1 | 12.61 | 9.8–10.0 | 6.1–6.6 | 3.012 | 2.0–2.2 |
| `argon_blueshift_averaged` | EPYC 7763 | 16 | 12.60 | 8.45–9.1 | 4.8–5.7 | 2.969 | 1.6–1.9 |
| `rdw` | EPYC 7763 | 1 | 22.03 | 16.9–18.3 | 15.8–17.4 | 59.06 | 0.27–0.29 |
| `rdw` | EPYC 7763 | 16 | 21.32 | 12.7–13.8 | 11.6–12.9 | 47.43 | 0.24–0.27 |

Notes on the table:

- **The M5 `rdw` rows at one thread** are derived from the per-trial profile; no
  complete one-thread run was timed.
- **The EPYC sixteen-thread rows for the two averaged cases** come from a node shared
  with another job, so read them against their own baseline.
- **The `rdw` host cost on the EPYC** was not measured cleanly. It is taken as 1.4 times
  helium's, the M5's ratio.

What the table shows:

- **On the Apple M5 Max**, everything together brings the helium mode-averaged case
  level with Runge–Kutta at one thread, and about 1.3 times ahead at sixteen. Argon stays
  1.6–1.9 times behind. `rdw` is four times ahead.
- **On the AMD EPYC 7763**, helium is 1.2–1.3 times behind at one thread and about level
  at sixteen. Argon is about twice as slow. `rdw` is 3.5–4 times ahead.
- **The native march is the largest single item on the single-mode cases on Linux.** It
  takes 0.54–0.57 s from helium's run (17–18 % of today's time) and 3.4–3.7 s from
  argon's (27–29 %). On the M5 it takes 0.15–0.17 s and 0.76–0.90 s.
- **Runge–Kutta gains too, and the table does not count it.** It would gain from N3
  (multi-row cases at sixteen threads), N6 (a few per cent) and N5b on its propagators.
  A native Runge–Kutta march would save its own 10 µs (M5) or 40–50 µs (EPYC) per call:
  about 0.13 s of helium's 1.39 s and 0.39 s of argon's 3.0 s on the EPYC. Such a march
  is not planned (§10, Q8).
- **On short runs the fixed 90–190 ms per run is then a large share** of both methods'
  time: 0.12 s of a 0.7 s helium run.

### 7. Verification and validation

Tolerances follow SEP 25 and [testing.md](../testing.md). Exact equality is used only
for structural contracts, and every rounding budget is fixed before the comparison. All
runs are ten-step windows or short guides that finish in seconds, as `AGENTS.md`
requires.

#### 7.1 Exact structural contracts

- **Decision traces** (R1, R8). The controller is tested on its decisions, separately
  from the arithmetic that feeds them.
  - *The oracle.* The corpus is generated in phase 0 from the reference as it stands
    before the refactor. The current `AdamsMarch` is driven through a stub engine that
    returns prescribed norms. The refactored core and the native march must both
    reproduce it, so a mistake in the extraction cannot be blessed in both halves.
  - *Decisions on supplied scores.* Each decision is exercised on identical scalar
    inputs: an acceptance on its norm, an order adoption on its weights, a landing on
    its gaps and caps, a restart on its counts. On these the two implementations agree
    exactly:
    - the accept;
    - the stored and control orders;
    - the restart;
    - every ledger counter;
    - the landing (step, end, kind, kink).

    The threshold witnesses live here, with no `pow` between input and comparison:
    norms at $1 \pm$ one ulp, $0$, $\infty$ and NaN; weights tied, and at $H$ exactly;
    gaps within $\tau$ of a plane, a kink, a kink on a plane, and at the stretch
    limit.
  - *The scalar formulas.* $F$ and $W_k$ are compared over the corpus's inputs. They are
    exact where both halves call the same C library's `pow` (Linux, macOS). Elsewhere
    the bound is one ulp, stated in the test.
  - *Whole traces.* Sequences through startup, growth, rejection and restart are
    compared with margin-aware expectations. A step whose decision margin is below 8 ulp
    of its threshold is reported, not compared.

  The traces cover:
  - fixed-order startup and post-restart growth at orders 2–6 (R1);
  - variable order with the collar cap binding, and quiet;
  - three and four consecutive rejections;
  - the budget at its edge;
  - the floor, including an accepted landing shorter than $h_{\min}/g$;
  - equal approach;
  - the carrier cap;
  - non-default `max_growth`, `min_factor` and `equal_approach`.
- **Stage scalars.** The native bundles equal `scalars_at`'s field for field on constant,
  gradient, profile, multi-gas, Raman and plasma problems, because they use the same
  strict Clenshaw helper (§4.8).
- **Events** (R5). Each event falls at the accepted step where the host march acts, and
  each accepted step is accounted exactly once. Checked by counting rows, checks and
  stops on runs that take the same steps:
  - statistics rows at `row_period` 1, 3 and 7;
  - natural stops at `step_planes` 1, 2 and 5;
  - monitor checks, busy and quiet;
  - dense crossings, including one where a natural stop stays due;
  - a plane that is also a statistics row and a monitor check;
  - `max_calls = 1`;
  - returns with rejections only, and with accepted steps followed by rejections;
  - a return for the call bound.
- **Lossless state** (R9, R10). After a round trip in either direction, a snapshot's rows,
  positions, identities, slots, order, controller state and $w_Q$ equal the stored ones
  exactly (SEP 29 R17), and a load does not take the restart path. The frozen-residual
  refusals of §6.5.9 stay, with their names. A backend reused for a second run starts
  with $w_Q$ and $\lambda$ cleared.
- **The ledger** (R4) counts the calls actually made, checked against the source's own
  counter. That includes a startup evaluation that fails, a trial evaluation that fails,
  and a failure after the source has returned, each also through a snapshot and resume
  of its ledger.

#### 7.2 Rounding budgets and the table oracle

- **Coefficient tables** (R7). Native and reference tables are both compared with exact
  tables. These are computed in rational arithmetic (`fractions.Fraction`) from the same
  binary64 nodes, for `exponential_tables`, `compensation_tables`,
  `factored_defect_rows` and `fused_tables`.
  - *The node sets* come from real histories, not only random ones: landings that
    shorten a step by $10^{-1}$ to $10^{-10}$ with `equal_approach=False`, rejection
    cascades up to the restart, non-default growth and smallest factors, orders 1–6.
  - *The error* is measured in the weights a trial applies: per row, the sum of absolute
    errors over the sum of absolute exact weights.
  - *The bound* is finite and informative: the native error must not exceed the
    larger of four times the reference's error on the same node set and 16 ulp.
  - *The report.* The reference's own error is reported for every node set. Where it is
    large, that is a finding for the author (§4.7), not a test to loosen.
- **Matched marches.** A prescribed schedule at a fixed order, with matched startup and
  restart history and matched plane handling, on the native march and on the host march
  over the same engine. The fields at every plane must agree to a budget from the step
  count and the per-step rounding of SEP 25 §4.3. This is the R-comparison SEP 25 asks
  for (phase 3, when replay is native).
- **The norms' spread** (R8). Phase 1 records, per qualification case, the distribution
  of $|\Delta\varepsilon|/\varepsilon$ between trials with host and native tables.
  These are the inputs of §4.9's margins.

#### 7.3 Accuracy envelopes for adaptive runs and resume

The native march against the host march on the same native engine. Across the rows of
§6.6's matrix, the cases are:

- a constant operator;
- an expansion gradient with the transported compensation;
- a z table;
- the carrier frame;
- the in-stage collar near its stability limit;
- the quiet collar;
- interpolated planes;
- a hard kink;
- `max_calls`;
- tolerances tight enough to reject;
- the restart after three rejections.

**Adaptive resumes** (R10) run the same cases stopped part-way, including a stop and
resume across a quiet-to-active collar transition (R9).

Each case's envelope is set in its test against a tighter or stored converged reference,
times a safety factor fixed before measuring (SEP 25 §4.4); `rtol` and `atol` alone do
not bound the global error. Step and call counts are recorded, together with the
smallest decision margin of each run. A difference in steps is reported in the phase
report, never asserted against.

#### 7.4 The existing suite

The resident-Adams tests run once under each march:

- `tests/backends/test_resident_adams.py`;
- `test_exponential_varying.py`;
- `test_exponential_routes.py`;
- `test_exponential_march.py`;
- `test_carrier_frame.py`;
- the resume, monitor and signal tests;
- in phase 4, `test_exponential_coupled.py`.

Each is parametrised by `adams_march`, so every check the host march passes on native
is also a check on the native march. Tests that patch the Python march's internals keep
the host march.

#### 7.5 Lifecycle

- **Signals** (R3). Signals are injected:
  - during a long native call;
  - at the return boundary, after native has committed several steps and before the
    host has ingested them.

  Each is honoured within the call bound. The terminal checkpoint's state, position,
  step history and ledger are checked together, and a resume continues the history.
- **Exceptions in user code** (R3). A progress reporter that raises on the first
  ingested row, and an accepted landing that hits the floor, leave the same consistent
  checkpoint.
- **Native errors** (R4). A failure in the middle of a call leaves the committed steps
  recorded, and raises the class its error record names. Startup, trial, reduction and
  accept failures are each injected.
- **Replay** (R6, phase 3):
  - the 160-step schedule that ends 12 ulps short of its length, which must count as
    arriving;
  - kinks on edges;
  - a schedule too short for its target (*exhausted*);
  - a resume part-way through, with the cursor rebuilt from the position.

#### 7.6 Benchmarks

The A/B harness, `python -m benchmarks ab`, compares the two marches on the same build
and revision, with N1–N7 held fixed and the march chosen per arm (R12). The harness
gains a per-arm environment, or the private switch is read from the environment once at
import.

**What each phase reports:**

- end-to-end time;
- attempts, source calls and host events per run, counted, not inferred;
- residual host time per trial.

**Cases:**

- `he_rdw_averaged`, `argon_blueshift_averaged`, `h2_raman_averaged`, `rdw`;
- a short window of `ctc`, for an expensive source;
- frequent events: a busy monitor, `step_planes = 1` and `row_period = 1`.

Each is run at one thread and the production width, interleaved in fresh processes, on
an Apple M5 Max and an AMD EPYC 7763 (or the machines then current, recorded by hardware
model).

**Lengths.** Runs use `--length` fractions chosen so that each arm finishes in seconds
on the slower machine, and named in the ledger entry: short guides, as `AGENTS.md`
requires. Complete tracking runs stay with SEP 38's release procedure and are not part
of this SEP's qualification.

**Baseline.** Phase 0 records the baseline in [benchmarks.md](../benchmarks.md): the
invocation, the lengths, the counted attempts and calls per case, and the host cost by
piece. The decision to go on to phase 2 then rests on public numbers. Each later phase
adds its own entry.

**Acceptance rule** (decided, R12). On every case above, the native march must not be
slower than the host march by more than the A/B's own measured spread, about 2 %. That includes the
expensive-source and frequent-event cases. A regression beyond that blocks `Final` until
it is understood and fixed.

### 8. Documentation

- **[stepper.md](../stepper.md), "Low-call Adams methods".** Where the march runs on each
  backend, the policy record and the events.
- **[kernel-contract.md](../kernel-contract.md).** A new section, "The Adams march":
  §4's algorithm as the normative text, the policy record's fields, the events, the
  attempt rows, the scalar tables leaf. §7 links to it.
- **[backends.md](../backends.md) §4.** The native march, its admission, and the GPU
  behaviour.
- **[decisions.md](../decisions.md).** A new ADR amending ADR-043 (§6.3).
- **`native/AGENTS.md`.** The amended rule (§6.3).
- **The manual page [the Adams integrator](../../guide/numerics/adams.md).** A short
  paragraph under its limitations or costs. On native the march runs in the compiled
  backend and `adams_march` records it; the warning says when it does not; and results
  meet the same accuracy envelope as the host march's. Its resume paragraph, which says
  a resumed run agrees with the uninterrupted one "to rounding", is restated as SEP 25
  §4.4 rules: the transfer is exact, and the continuation meets the accuracy envelope
  (R10).
- **The manual page [choosing a stepper](../../guide/numerics/choosing-a-stepper.md).**
  Re-measured rows once the phases land.
- **`CHANGELOG.md`.** A line citing this SEP.

No tutorial or worked example is needed: nothing a user writes changes.

### 9. Compatibility and impact

- **Defaults.** None changes. `Multistep()` stays the method, with its fields and
  constants.
- **Results.**
  - On the native backend, the native march's tables differ from the host's by rounding
    (measured, §7.2).
  - A decision can flip at a knife edge, after which the run meets the case's accuracy
    envelope rather than the host march's steps (§4.9, R10).
  - Validated results elsewhere are unchanged: `numpy_ref` and `numba_cpu` run the same
    Python march, refactored without behaviour change in phase 0.
- **Seam.** Kernel schema +1 in phase 1 and +1 in phase 2. New contract tuples
  `SCALAR_TABLES_FIELDS` and `MARCH_POLICY_FIELDS`, and additions to `DRIVER_SURFACE`.
- **Output.** Output schema +1 in phase 0, for the collar flag $w_Q$ in
  `MarchSnapshot` (R9), and +1 in phase 2, for `RunMetadata.adams_march`. Older files read
  as before, since breaking changes are allowed before the first release and there are no
  compatibility shims (ADR-051).
- **Rules.** `native/AGENTS.md`'s policy rule is amended, and ADR-043 is amended by a new
  ADR (§6.3).
- **Warnings.** A run on native that the native march does not cover warns
  (`HostMarchWarning` or similar, §6.6).
- **What it breaks:**
  - ADR-043's single executed controller;
  - Python-level introspection of a native run's march: the debug lines arrive per
    call rather than as each step is taken, and monkeypatching `AdamsMarch` methods
    applies only to the host march;
  - signal latency, now up to one call bound (0.1 s) instead of one trial;
  - progress granularity, now per call.
- **Luna.jl.** No difference to record.

### 10. Alternatives considered

- **Option A, the controller owned by C++** (§6.1). The same speed as B, but it puts the
  policy in native code and keeps two sets of constants.
- **Option C alone** (§6.1). It keeps a third of the EPYC's host cost and one crossing
  per trial. It is kept as phase 1.
- **A native loop that calls back into Python for each decision.** It saves nothing,
  since the crossings stay. It also holds the GIL across native work, and
  `native/AGENTS.md` forbids calling Python from native code.
- **A Numba-compiled controller** called from the Python loop. It saves the decision's
  39–47 µs on the EPYC and nothing else.
- **A lagged decision.** ADR-004's lagged controller lets a device run ahead of the host
  by deciding on the previous trial's norm. For Adams it changes the algorithm: a trial
  would start before its predecessor is accepted, which wastes a source call on every
  rejection and changes the step sequence. That breaks the "no change to the method"
  scope.
- **A controller resident on the device**, in a persistent kernel or graph. This is the
  next step for device builds, but the launches and the norm read-back are a separate
  problem, and it would be its own SEP.
- **Tabulating the stage scalars on a fine grid** (§6.5.6).
- **Moving the Runge–Kutta march native.** RK's host cost is 10 µs (M5) and 40–50 µs
  (EPYC) per call, and its controller (the lagged PI controller, the stretch, the start
  rule) is another algorithm to specify. Not pursued: it can be taken up another time,
  reusing this SEP's policy-record pattern, if it is needed (Q8).
- **Retuning the controller for fewer steps** on the argon case. That is a method
  change, outside this SEP, and `native/AGENTS.md` forbids it as a speed measure.

### 11. Phases

Every phase is on one branch, `feature/sep-0039-native-adams-march`, and reaches `main`
in one pull request at `Final` (SEP 0). Each phase runs targeted checks, and the full
gate runs in CI.

**Phase 0: the controller core in solprop.**

- `MarchPolicy` and `march_policy`.
- The general path (`_attempt`, `_correct`, `_commit`, `_next_order`), which serves only
  `ComparisonMultistep`, moved out of `AdamsMarch` into its own class (Q7).
- A pure core of the controller (`stencil`, `decide`, `reject`, `choose_order`, the caps,
  `approach` and `landing`) over a policy and a state. `AdamsMarch` calls it.
- No change in behaviour, except two fixes:
  - the commit order of §4.4, with a regression test for an accepted landing shorter
    than $h_{\min}/g$;
  - the collar flag $w_Q$, kept in `MarchSnapshot` and reset by `begin_run` (§6.5.9;
    output schema +1).

  The existing Adams tests pass unchanged on `numpy_ref` and `numba_cpu`.
- The decision-trace corpus, generated from the reference before the refactor (§7.1,
  R1), and the table oracle's corpus (§7.2, R7).
- The baseline benchmark entry: invocation, lengths, counted attempts and calls, host
  cost by piece (§7.6, R12).
- The kernel-contract section, drafted.

Verified by the existing suite, and by the refactored core reproducing the
pre-refactor corpus exactly.

**Phase 1: native tables and stage scalars** (option C's pieces).

- The scalar tables leaf, with derivative tables for the rate weights.
- Native coefficient tables formed from the nodes inside the trial call.
- Lean trial arguments: positions and phase increment only.
- The trial and the accept stay separate calls (R2).
- Kernel schema +1.
- The Python march still decides.

Verified by §7.1's scalar equality, §7.2's table oracle and the norms' spread, the
existing suite, and the benchmarks. Expected −265 to −335 µs per helium trial on the
EPYC. Phase 1 stays in
this SEP and lands with the rest in its one pull request (decided, Q11).

**Phase 2: the native march**, for the exponential formulation on diagonal operators,
with everything of §6.5:

- the events;
- commits, signals and errors;
- snapshots;
- metadata;
- admission and the host march elsewhere.

Also in phase 2: the new ADR, the `native/AGENTS.md` amendment, the documentation of §8,
and kernel and output schema +1. The two marches are compared on the same revision with
N1–N7 held fixed, under §7.6's acceptance rule. Verified by §7.1, §7.3, §7.4, §7.5 and
§7.6.

**Phase 3: the interaction formulation's fused path (Q4), and replayed schedules
natively,** with replay's own cursor, slack, kink and exhaustion rules (§4.3, R6). Single
runs only; lockstep batches keep `replay_step` on the host (Q14). Verified by §7.2's
matched marches, §7.4 and §7.5's replay cases.

**Phase 4: coupled flows and moving maps**, natively, for every flow the resident engine
executes (Q2). The flow section and weights, the block's end weights and the transform
weights per position. Verified by `test_exponential_coupled.py` under both marches.

**Phase 5: device builds**, inside this SEP and before `Final` (Q6), with a GPU budget
the maintainer approves when the phase starts. The march on CUDA and HIP builds, in one
GPU segment under the standing device policy: the parity tests with `-m gpu`, and
ten-step timings against the host march.

The native items of §6.9 proceed independently. Each phase's benchmarks are measured on
`main` as it then stands, and the report says which of N1–N7 had landed.

### 12. Open questions

The author decided every question below on 2026-10-28, before review. Each decision is
recorded beside its question, and the proposal above follows it.

- **Q1.** Amend `native/AGENTS.md`'s rule with the wording of §6.3, or keep it as it is
  and record B as an exception in the new ADR? **Decided 2026-10-28 (John Travers):
  amend.**
- **Q2.** Configurations outside the native march's admission: run the host march and
  record it, or refuse them by name on native? **Decided 2026-10-28 (John Travers): run
  the host march with a warning, and aim to cover every configuration** (§6.6).
- **Q3.** The call bound. A scheduler allows seconds, but a keyboard interrupt wants
  tenths. **Decided 2026-10-28 (John Travers): 256 attempts and 0.1 s of wall time,
  not a user setting** (§6.5.7).
- **Q4.** Does the interaction formulation's fused path deserve the native march (phase
  3)? It is not the default, but on SEP 37's capillary rows it scored 1.45/1.47 times
  the fastest against the exponential formulation's 2.15/2.26, and the controller is
  shared. **Decided 2026-10-28 (John Travers): yes, in phase 3.**
- **Q5.** Coupled flows and moving maps (phase 4): native, or left on the host march?
  Answered by Q2: native, for every flow the resident engine executes. What has no
  resident engine is Q14.
- **Q6.** Device builds: qualify in this SEP's phase 5 or in a later device SEP?
  **Decided 2026-10-28 (John Travers): in phase 5, before `Final`, with a GPU budget
  approved when the phase starts.**
- **Q7.** Whether phase 0's refactor should also simplify `AdamsMarch` itself. The
  general path, `_attempt`, `_correct`, `_commit` and `_next_order`, which serve only
  `ComparisonMultistep`, could move into their own class. **Decided 2026-10-28 (John
  Travers): yes, in phase 0, with no change in behaviour.**
- **Q8.** A native Runge–Kutta march in a follow-up SEP, reusing the policy-record
  pattern, so that the default comparison is between equally optimised marches?
  **Decided 2026-10-28 (John Travers): no.** It can be taken up another time if it is
  needed.
- **Q9.** A compiled march for `numba_cpu`? **Decided 2026-10-28 (John Travers): no.**
  Backend parity of the march is not needed, and `numba_cpu` is being demoted, so no
  roadmap item is added (§6.6).
- **Q10.** The capillary default. **Decided 2026-10-28 (John Travers): outside this
  SEP's scope; it changes no default.**
- **Q11.** Should phase 1, which needs no SEP on its own, land ahead of acceptance as an
  ordinary optimisation pull request? **Decided 2026-10-28 (John Travers): no.** It
  lands with the whole SEP.
- **Q12.** Progress and logging granularity. Is one progress update and one debug line
  per event enough, or should the attempt rows be logged one by one at debug level?
  **Decided 2026-10-28 (John Travers): progress once per call; one debug line per
  attempt, written from the rows when `DEBUG` is enabled** (§6.5.8).
- **Q13.** Should an accepted landing shorter than $h_{\min}/g$ raise the floor (§4.4)?
  The Adams march forms its next proposal from the step taken, so it does; the
  Runge–Kutta march grows from its proposal and does not. **Decided 2026-10-28 (John
  Travers): phase 0 fixes the commit order. The rule may be changed in this SEP if the
  implementation finds a reason, recorded in the phase report.**
- **Q14.** Which configurations may still run the warned host march at `Final`
  (following Q2)? Proposed: only those with no resident engine or no data form for the
  native march:
  - `ComparisonMultistep`'s general path;
  - a coupled flow whose sections are off the eigenbasis route or wider than 16
    channels (ADR-044);
  - a controller whose norm scale is not `PIController.scale`;
  - lockstep batch replay.

  The untransported compensation is a private test witness and does not warn.
  **Decided 2026-10-28 (John Travers): as proposed.** The proposal also listed "device
  builds, until phase 5". With phase 5 before `Final` (Q6), that is a state of the
  branch, not of `Final`, and it moves to §6.6's matrix (R11).

### 13. References

1. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd ed., Springer (1993), sections II.4 and III.5.
   doi:10.1007/978-3-540-78862-1
2. P. N. Brown, G. D. Byrne and A. C. Hindmarsh, "VODE: a variable-coefficient ODE
   solver," *SIAM J. Sci. Stat. Comput.* **10**, 1038 (1989). doi:10.1137/0910062
3. M. Hochbruck and A. Ostermann, "Exponential integrators," *Acta Numerica* **19**, 209
   (2010). doi:10.1017/S0962492910000048
4. S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff systems,"
   *J. Comput. Phys.* **176**, 430 (2002). doi:10.1006/jcph.2002.6995
5. G. Söderlind, "Automatic control and adaptive time-stepping," *Numerical Algorithms*
   **31**, 281 (2002). doi:10.1023/A:1021160023092
6. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002),
   chapter 22 (Vandermonde systems). doi:10.1137/1.9780898718027
7. D. Goldberg, "What every computer scientist should know about floating-point
   arithmetic," *ACM Comput. Surv.* **23**, 5 (1991). doi:10.1145/103162.103163
8. *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (2019).
   doi:10.1109/IEEESTD.2019.8766229
9. J.-M. Muller *et al.*, *Handbook of Floating-Point Arithmetic*, 2nd ed., Birkhäuser
   (2018), on the accuracy of elementary-function libraries.
   doi:10.1007/978-3-319-76526-6
10. C. R. Trott *et al.*, "Kokkos 3: programming model extensions for the exascale era,"
    *IEEE Trans. Parallel Distrib. Syst.* **33**, 805 (2022).
    doi:10.1109/TPDS.2021.3097283

## Part II — Implementation record

No phase has started.

## Part III — History

- 2026-10-28: Drafted by Opus 5.5 for John Travers, from a profiling study of the
  exponential `Multistep()` on the native CPU backend (Apple M5 Max and AMD EPYC 7763).
  Recommends option B (policy as data, mechanism native), built on option C's native
  tables and stage scalars.
- 2026-10-28: The author's decisions recorded:
  - Q1: amend `native/AGENTS.md`'s policy rule;
  - Q2: uncovered configurations run the host march with a warning, and the SEP aims to
    cover every configuration (§6.6; phase 4 is no longer optional; Q5 answered; Q14
    added);
  - Q11: phase 1 stays in the SEP.
- 2026-10-28: The author's decisions on the remaining questions recorded:
  - Q3: the call bound is 256 attempts and 0.1 s;
  - Q4: the interaction formulation is covered in phase 3;
  - Q6: device builds are phase 5, before `Final`;
  - Q7: phase 0 moves the general path into its own class;
  - Q8: no native Runge–Kutta march;
  - Q9: no `numba_cpu` march and no roadmap item, since `numba_cpu` is being demoted;
  - Q10: no default changes, outside scope;
  - Q12: one debug line per attempt at `DEBUG`;
  - Q13: phase 0 fixes the commit order, and the rule may change if the
    implementation finds a reason;
  - Q14: the warned host-march list as proposed.
- 2026-10-28: AI review by GPT-6 (R1–R12) recorded by the author. The drafting agent
  (Opus 5.5) answered every point under *Response to review* and revised Part I.
- 2026-10-28: The author accepted the two proposals left open by the response: the
  collar flag $w_Q$ is kept in `MarchSnapshot` and reset by `begin_run` (R9), and the
  native march may not be slower than the host march beyond the A/B spread of about 2 %
  (R12).

### AI review — 2026-10-28

Reviewer: **GPT-6 (AI), acting through Codex**. Reviewed all of Part I and the
associated index and roadmap changes on PR #52 at `40c34af`. This review changes
only the reviewer metadata and appends findings here; it does not revise Part I
or record a human review. The author's decisions in §12 remain the author's.

**Assessment:** option B is a reasonable architecture, and keeping the host march
available on the same engine gives it a useful verification path. The proposal
needs the following corrections and contract decisions before acceptance. P1
marks a correctness or lifecycle issue to resolve before implementing the affected
phase; P2 marks a specification or qualification issue to resolve before accepting
the corresponding promise. All findings are open for the author's response.

**Evidence and limits.** The review compared the specification with the current
march, controller, landing rule, stage tables, native adapter, interrupt guard,
monitor and snapshot code, and with SEPs 0, 16, 25 and 29. Small executable probes
confirmed R1 and R6 and the conditioning example in R7; no propagation benchmark
or native CPU/GPU qualification was run for this review. The existing virtual
environment did not contain `solprop-native`. The moment identity in §4.7 agrees
with equation (2.10) of
[Hochbruck and Ostermann's author manuscript](https://na.math.kit.edu/download/papers/acta-final.pdf):
putting `k = p + 1` gives the stated factorial. The Lagrange/Vandermonde identity
was also checked against
[Higham's explanation](https://nhigham.com/2021/06/15/what-is-a-vandermonde-matrix/).
The other bibliography entries and the profiling measurements were not independently
verified. The issues below concern execution and qualification, not a proposed
change to the propagation equations.

1. **R1 — P1: distinguish the stored order from the order controlling startup
   growth (§§4.4–4.5, 7.1, phase 0).** With fixed order, §4.5 sets `q_n = q_max`,
   then §4.4 uses that order in `F`. The current `AdamsMarch._choose_order`
   instead stores `method.order` for the next stencil but returns `current` as the
   factor's exponent order. At startup and after a restart, `current` can be
   smaller than `q_max`. A probe with `Multistep(order=5, variable_order=False)`,
   one available history sample and norm `0.25` gives a factor of `1.6` in the
   reference, versus `1.0079368399158986` under the draft's formula. This changes
   steps away from any rounding threshold. Please specify these two orders
   separately and add fixed-order startup and post-restart traces for orders 2–6.
   Preserve the pre-refactor reference as an oracle: generating the whole corpus
   only from the newly extracted core could bless this same error in both halves.

2. **R2 — P1: define how phase 1 commits an acceptance before the next trial
   exists (§6.1 option C, §6.5.7, phase 1).** Fusing a trial's accept into the next
   trial call leaves the accepted state pending. Today `_decide` calls the
   engine's accept before moving the position, recording statistics, checking the
   monitor or returning a plane. There may be no next trial: a save plane, natural
   stop, exhausted call budget, final endpoint or interruption can intervene.
   A monitor verdict can also change the next trial's collar setting after the
   current state must already have been adopted. Please specify a flush/commit
   operation and its ownership at every such boundary, or retain a separate
   accept until phase 2 can decide and commit inside the native call. Include a
   one-step run ending at a plane, an immediately exhausted budget and a monitor
   change in phase 1's tests; measure the crossings actually left by that design.

3. **R3 — P1: enter the signal guard before native mutation and make host
   publication complete before callbacks (§§4.4, 4.8, 6.5.7, 7.5).** The text
   puts the host's row bookkeeping inside `committing()` after the native call.
   `solprop._interrupts._handle` raises immediately outside that guard, including
   when Python resumes from a native call that has already committed several
   steps. The low-storage precedent guards the native call itself as well as
   the position update. Please require one guard spanning native execution,
   synchronization of the host position/proposal/history/ledger and ingestion of
   all returned rows. Also separate ingestion from user progress callbacks:
   `_record` invokes `_progress.step`, whose exception is not deferred by
   `committing()`. A callback failing on the first replayed row must not leave
   the native state several commits ahead of the host records. The accepted-step
   floor fix needs the same complete publication rule, including the accepted
   attempt row and kink restart, before raising. Inject signals at the return
   boundary and exceptions from callbacks and from the floor path; verify the
   terminal checkpoint's state, position, rows and counters together.

4. **R4 — P1: preserve call charges when source evaluation fails (§§4.3,
   6.5.7–6.5.8, 7.1).** The algorithm adds `startup` after the startup evaluation
   and `predictor` after the trial and its reductions. The reference increments
   them before `engine.start` and `engine.exponential_trial`, respectively.
   A source call that raises, or succeeds before a later reduction fails, has
   still spent work. Rolling back to the last committed ledger or incrementing
   only after success undercounts it, which matters when that ledger is saved
   and a budgeted run resumes. Please distinguish accepted-state commits from
   attempted-call charges, specify the failure stage and exception category in
   the returned error record, and preserve the reference's charging boundary.
   A message alone cannot reliably select the original exception type. Add
   independent call-counter tests for startup failure, predictor failure and
   failure after the source returns, including export/resume of their ledgers.

5. **R5 — P1: make event draining total for empty, rejected and coincident
   returns (§§4.8, 6.4–6.5, 7.1).** The event table is headed "after an accepted
   step", but a call bound can end a rejected attempt, and a budget/error event
   can return without an attempt or without any acceptance. The prescribed
   unconditional final `_take_step_stats`/`_natural_stop` then counts a step that
   did not happen. Conversely, a call can finish after accepted steps followed
   by rejections, so its last attempt is not its last accepted step. Please
   specify entry checks, per-attempt checks, zero-row and zero-acceptance returns,
   and exactly-once accounting of each accepted step. Define how simultaneous
   plane, monitor, statistics, natural-stop and call-bound events are all serviced,
   including a dense crossing whose natural stop remains due. State how
   `_record`'s backend hook is handled: native `_on_accepted_step` currently calls
   `promote_fsal`, so replaying that hook in bulk is not automatically an inert
   record operation. Finally reconcile §6.5.8's per-attempt progress callbacks
   with Q12's "progress once per call". Test `max_calls=1`, rejection-only
   returns, accepted-then-rejected returns, and several coincident events.

6. **R6 — P1: specify replay's separate stopping and kink tolerances
   (§§4.3, 4.6, 4.8, phase 3).** Replacing only the proposal and landing steps
   does not describe the existing replay. `_replay_to` uses
   `replay_tolerance(schedule, target)`, with
   `max(LANDING_ULPS, schedule.n_steps)` ulps, and uses the same slack to identify
   replayed kinks. It also advances `_schedule_index`. The draft otherwise leaves
   the adaptive `reached` test in force and omits the replay cursor from its
   state description. The existing documented witness
   `FixedSchedule(np.full(160, 1e-3 / 160))` ends 12 ulps below `1e-3` in the
   review probe: adaptive `reached` is false, while the replay test succeeds.
   Following the draft literally can report exhaustion for this valid schedule.
   Please specify the cursor, edge arithmetic, replay tolerance, exhaustion and
   restoration rules, with tests for accumulated-edge drift, kinks and a resume
   part-way through the schedule.

7. **R7 — P2: qualify coefficient tables over the nodes the march actually
   permits (§§4.7, 7.2).** The claimed modest conditioning does not follow from
   the default growth and rejection factors. They are configurable, multiple
   rejections compound a shrink, and a forced landing can shorten a step by an
   arbitrary ratio without discarding history at an ordinary save plane.
   `equal_approach=False` is admitted. For example, four preceding unit steps
   followed by a trial shortened to `1e-10` give corrector nodes
   `[1, 0, -1e10, -2e10, -3e10, -4e10]`; their monomial Vandermonde has an
   infinity-norm condition estimate of about `2.05e53` in the probe, despite
   having only six nodes. Multiplying that estimate by operation count and unit
   roundoff gives no useful accuracy test. This is not itself evidence that the
   existing root-product algorithm fails: its error needs analysis appropriate
   to that algorithm and to the weights ultimately applied. Please remove the
   unsupported conditioning claim, include actual landing/rejection histories and
   non-default factors in the corpus, and require finite, informative bounds
   supported by an independent polynomial or higher-precision oracle. Any new
   admission limit or restart rule would be a numerical-policy decision for the
   author, not an implementation convenience.

8. **R8 — P2: make the exact-arithmetic promises consistent with the build
   and threshold contracts (§§4.8–4.9, 6.5.1, 6.7, 7.1).**
   `SOLPROP_NATIVE_STRICT_FUNCTION` alone is empty under Clang; the function body
   also needs `SOLPROP_NATIVE_STRICT_BLOCK`. Both macros are empty under CUDA/HIP,
   including the CUDA host pass (`native/include/solprop_native/fp.hpp`). Thus
   that annotation alone does not establish the promised strict controller on
   every phase-5 build. Also, §7.1 permits a one-ulp `pow` difference while still
   demanding identical order and landing decisions at exact thresholds. A small
   weight or proposal difference can change precisely those branches. Please
   specify the host compilation and elementary-function requirements, and
   separate exact decision tests on identical supplied scalar scores from
   numerical tests that independently compute those scores. Either ensure common
   arithmetic for threshold witnesses or define margin-aware expectations.
   Verify strictness on each supported compiler. The nominal `1e-14` norm
   difference in §4.9 also needs a case-specific basis, especially after R7.

9. **R9 — P2: bound the snapshot portability promise and account for quiet
   collar state (§§4.2, 6.5.9, 7.1, phase 4).** Changing the march on the same
   resident engine is different from changing engines. A frozen-residual native
   history exports `applied` rows; `HostHistory.load_rows` refuses those, while
   `_ResidentAdams.load_rows` refuses the host engine's `weights_<row>` format.
   Therefore retaining the existing format cannot by itself make every phase-4
   snapshot continue on `numpy_ref` or `numba_cpu` without restart. Please give
   a support matrix and retain the existing named refusal, or explicitly design
   and qualify a conversion with its memory cost. Separately, §4.2 lists
   `w_Q` as march state, but neither the listed snapshot fields nor the current
   `MarchSnapshot` stores `_collar_was_quiet`. Its value controls the re-entry
   cap. State whether load restores it, reconstructs it from monitor state or
   deliberately resets it, and test a stop/resume across a quiet-to-active
   transition. Verify that a reused backend starts with the specified flags too.

10. **R10 — P2: use accuracy envelopes for ordinary adaptive resume
    (§§4.9, 6.5.9, 7, 9).** The unconditional promise that a resume agrees with
    an uninterrupted run "to rounding" contradicts SEP 25 §4.4, which explicitly
    classifies ordinary resumed-versus-uninterrupted comparisons as accuracy
    comparisons. The native and host marches can compute different tables and
    proposed steps, and can then choose different orders or acceptances, even
    if the serialized state was copied losslessly. Please keep exact checks for
    the snapshot transfer, reserve rounding budgets for matched fixed-order
    schedules with matched history and plane handling, and add adaptive resume
    accuracy cases against a converged reference. Throughout the interface and
    compatibility wording, replace agreement "to the run's tolerance" with the
    qualified per-case accuracy envelope: local `rtol`/`atol` do not bound the
    global error. This follows the policy already cited by the proposal.

11. **R11 — P2: make admission a single consistent matrix (§§2, 6.6, Q14).**
    Q14 permits the warned host march for `ComparisonMultistep`'s general path,
    but §6.6 lists that same path as "Refused by name, unchanged". The current
    `_refuse_adams` does not impose that blanket refusal; `AdamsMarch` uses
    `HostHistory` when the fused/exponential engine is unavailable. Moreover,
    `ComparisonMultistep` is a subclass of `Multistep`, and some comparison
    configurations use the fused resident path, including the Hermite dense
    interpolant covered by `test_resident_adams.py`. A type-membership check
    alone does not enforce the stated scope. Please enumerate admission by
    formulation, corrector/interpolant, resident-engine capability, scale and
    build, distinguishing native march, warned host march, private unwarned
    witnesses and pre-existing backend refusals. Record device rows at `Final`
    after phase 5, rather than leaving "until phase 5" as a final exception.

12. **R12 — P2: make the performance decision reproducible and independent
    of the other optimisations (§§3.3, 6.1, 6.9, 7.6, 11).** The measurements
    are useful motivation, but the draft links no committed timing ledger or
    profiling artifact for the baseline, and infers some trial counts from
    timings. The estimated gain of B over completed phase 1 is the relevant
    evidence for accepting a second controller implementation. Please name the
    reproducible benchmark invocation/configuration, exact measured window,
    measured attempt and source counts, and public ledger entries. Compare the
    two marches on the same implementation revision with N1–N7 held fixed, and
    report end-to-end time and event frequency as well as residual host cost;
    busy monitors and `step_planes=1` return every step. State the acceptance
    rule for regressions on expensive-source and frequent-event cases. The claim
    that call bounds cost "at most 0.04 %" assumes the time bound is limiting;
    the 256-attempt bound can instead dominate cheap trials. Measure it. Finally,
    §7.6's "then complete runs" must name short-guide lengths that finish in
    seconds, consistently with §7 and the repository's qualification rule,
    rather than authorize full-length tracking problems by implication.

The author should answer R1–R12 in a separate response subsection and revise Part I
thereafter, as SEP 0 requires. This review does not accept the SEP or change its
status from Draft.

### Response to review — 2026-10-28

The drafting agent, Opus 5.5, checked each point against the code before answering. All
twelve are taken. Part I is revised as follows; section numbers are those of the revised
text.

1. **R1: taken.** Confirmed in `AdamsMarch._choose_order`: under a fixed order it stores
   `method.order` but returns the step's own order for the factor.
   - §4.4 (step 6–7) and §4.5 now distinguish the stored order $q_s$ from the control
     order $q_c$ and its norm $\varepsilon_c$. The review's probe is quoted in §4.5.
   - §7.1's corpus is generated from the reference *before* the refactor, through a stub
     engine, and both the refactored core and the native march must reproduce it. Its
     traces cover fixed-order startup and post-restart growth at orders 2–6.
   - Phase 0 is updated to match.
2. **R2: taken, by dropping the fusion.** Option C and phase 1 keep the trial and the
   accept as separate calls (§6.1, §11). The flush that fusion would need belongs at
   exactly the boundaries the review lists, and phase 2 removes those crossings anyway,
   by deciding and committing inside the native call.
   - Option C's figures are recomputed without the fused accept: −265 to −335 µs per
     helium trial on the EPYC, two crossings per accepted trial.
3. **R3: taken.** Confirmed: `committing()` only defers inside its block, and `_record`
   calls the progress reporter.
   - §6.5.7 now requires one guard spanning the native call, the synchronisation of the
     host's position, proposal, order, history and ledger, and the ingestion of every
     row.
   - User code runs only after the guard: progress callbacks, debug lines and the event
     step's work. The floor publishes the accepted attempt, its kink restart and the
     ledger before it raises (§4.4).
   - §7.5 injects signals during a call and at the return boundary, and exceptions from
     a progress reporter and from the floor. It checks the terminal checkpoint's state,
     position, history and ledger together.
4. **R4: taken.** Confirmed: the reference charges `startup` and `predictor` before
   `engine.start` and `engine.exponential_trial`.
   - §4.3 now charges before evaluating.
   - §6.5.7's error record carries the stage, a category code for the exception class
     (never parsed from the message), and the message, with the ledger as charged.
   - §7.1 counts calls independently through startup, trial and post-source failures,
     including a snapshot and resume of their ledgers.
5. **R5: taken.** §4.8 now specifies:
   - the entry checks (as `march_to`'s, so no call is made when no attempt is due);
   - the per-attempt, per-accepted-attempt and pre-attempt tests;
   - returns with no rows, rows without acceptance, and acceptances followed by
     rejections;
   - exactly-once accounting of each accepted row;
   - the single event step, for which the host runs the reference's own
     `_take_step_stats` and `_natural_stop`, so coincident events (including a natural
     stop due on a dense crossing) are serviced in the reference's order.

   The event table is split into accepted-step events and the other returns, which have
   no event step. `_on_accepted_step` is not replayed: on native it calls `promote_fsal`,
   whose quadrature scoring the native march performs itself after each acceptance.
   Progress is reconciled with Q12: one callback per attempt, delivered in bulk, so a
   display updates once per call (§4.8, §6.5.8). The tests the review asks for are in
   §7.1.
6. **R6: taken.** Confirmed: `_replay_to` uses `replay_tolerance` for both stopping and
   kinks, and advances `_schedule_index`; a load rebuilds the cursor with
   `_schedule_index_at`. §4.3 now specifies:
   - the cursor;
   - the precomputed edges;
   - the slack $\max(u, n_{\mathrm{steps}})$ ulps;
   - the stopping, kink and *exhausted* rules;
   - the restoration on load.

   `reached` is explicitly not used on a replay. §7.5 adds the 160-step witness, kinks on
   edges, exhaustion and a resume part-way. Phase 3 refers to them.
7. **R7: taken.** The conditioning claim is removed. §4.7 gives the review's example and
   says why a condition-number budget is uninformative here.
   - §7.2 replaces the budget with an exact rational oracle (`fractions.Fraction`, no
     new dependency).
   - The node sets come from real landing and rejection histories, with non-default
     factors and `equal_approach=False`.
   - The error is measured in the applied weights. The bound is finite and relative to
     the reference's own error (at most four times it, or 16 ulp).
   - No restart or admission rule is proposed. If the reference proves inaccurate on
     admissible histories, that is reported to the author as a numerical-policy question
     (§4.7).
8. **R8: taken.** Confirmed in `fp.hpp`: `SOLPROP_NATIVE_STRICT_FUNCTION` is empty under
   Clang, and both macros are empty when `__CUDACC__` or `__HIPCC__` is defined, the host
   pass included.
   - §6.5.1 puts the controller in a host-only translation unit compiled with
     contraction off by a per-source option on every build, checked on each supported
     compiler. It names `pow` from the platform's C library as its only elementary
     function.
   - §7.1 separates exact decision tests on identical supplied scores (where the
     threshold witnesses live) from numerical tests of $F$ and $W_k$. Those are exact
     with a common C library and at one ulp otherwise, with margin-aware whole traces.
   - §4.9's nominal $10^{-14}$ is removed. Phase 1 measures the norms' spread per case
     (§7.2).
9. **R9: taken, with one proposal for the author.**
   - §6.5.9 now gives a support matrix. It keeps today's named refusals between the
     native `applied` rows and the host engines' `weights_<row>` for frozen-residual
     flows, and proposes no conversion.
   - On $w_Q$, confirmed: neither `MarchSnapshot` nor `load` keeps
     `_collar_was_quiet`, and `begin_run` does not reset it. The SEP now proposes
     restoring it from the snapshot (one new field, output schema +1 in phase 0) and
     resetting it in `begin_run`, fixed in the Python march first.
   - Reconstructing it from the monitor's state at the plane was considered and not
     proposed. The plane's own check may change the flag after the last attempt, so it
     would not reproduce the uninterrupted run's cap.
   - §7.1 and §7.3 test the quiet-to-active resume and a reused backend.

   Whether to snapshot $w_Q$ (proposed) or deliberately reset it is the author's call.
   **Decided 2026-10-28 (John Travers): snapshot it and reset it in `begin_run`, as
   proposed.**
10. **R10: taken.** §6.5.9 states the three levels:
    - the transfer is exact;
    - an adaptive continuation meets the case's accuracy envelope against a converged
      reference;
    - a rounding budget applies only to matched fixed-order schedules.

    "To the run's tolerance" is replaced by the per-case accuracy envelope throughout
    (§4.9, §6.6, §9). §7.3 adds adaptive resume cases. §8 restates the Adams manual
    page's resume paragraph, which today says "to rounding", in those terms.
11. **R11: taken.** Confirmed: `_refuse_adams` does not refuse the general path, and the
    one-order-higher PEC comparison variants take the resident fused path, the Hermite
    interpolant included. §6.6 replaces the list and the wrong "refused by name" line
    with one matrix. The matrix:
    - is keyed on formulation, corrector and interpolant, resident engine, norm scale
      and build;
    - distinguishes the native march, the warned host march, the unwarned private paths
      and the existing refusals;
    - records device builds at `Final`.

    Q14's "device builds, until phase 5" is moved there as a state of the branch. This
    is a consistency correction under Q6, not a change to the author's Q14 decision.
12. **R12: taken.**
    - §6.1 adds the estimate the review asks for: B's gain over a completed phase 1.
      That is 125–220 µs per helium trial and 85–150 µs per argon trial on the EPYC, or
      7–14 % and 9–23 % of the runs once §6.9's items have landed. It also notes that
      frequent events return every step.
    - §7.6 now compares the two marches on the same build and revision, with N1–N7 held
      fixed (a per-arm environment for `ab`). It reports end-to-end time, counted
      attempts, calls and events, and residual host time.
    - The cases add an expensive source (a short `ctc` window) and frequent events (a
      busy monitor, `step_planes = 1`, `row_period = 1`).
    - The acceptance rule is no regression beyond the A/B's measured spread, about 2 %,
      on any case.
    - Runs use `--length` fractions that finish in seconds; complete tracking runs stay
      with SEP 38's release procedure.
    - The "0.04 %" figure is replaced by an estimate that names which bound binds, to be
      measured in phase 2 (§6.5.7).
    - Phase 0 records the baseline publicly (§3.3, §7.6). The profiling study's numbers
      remain the motivation; the counted entry is the evidence.

    The 2 % rule is a proposal; the author may set another. **Decided 2026-10-28 (John
    Travers): the 2 % rule, as proposed.**
