# SEP 39 — A native Adams march: the Multistep step loop in the compiled backend

| | |
|---|---|
| **SEP** | 39 |
| **Title** | A native Adams march: the Multistep step loop in the compiled backend |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-28) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
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
- its host cost per trial is at most 15 µs on the M5 and 40 µs on the EPYC.

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
- the resident coupled-flow sections (phase 4), or a recorded decision to leave them on
  the host march;
- the seam: a march-policy record, the scalar tables, the host events, one metadata
  field;
- the amendment of `native/AGENTS.md`'s policy rule and of ADR-043 (§6.3).

**Not in scope:**

- **Any change to the method, the controller's algorithm, its constants or any
  default.** The native march executes the Python march's algorithm, and
  `native/AGENTS.md` forbids tuning the controller for speed.
- **The capillary default.** It is re-decided by SEP 37's rule after re-measurement
  (§12, Q10).
- **Fewer steps on the argon blue-shift cases.** Multistep takes 11 169 calls there
  against Runge–Kutta's 10 105, so no overhead work wins that case (§4.1). That needs
  a method change, not an execution change.
- **Runge–Kutta's host loop.** Its cost per call is small, but a fair comparison needs
  it later (§10, Q8).
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
per-call cost, so read them as about ±5 %.

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
   - evaluate the source once at the accepted state;
   - set the history to $[(z, \text{accepted}, 0)]$;
   - set $q = 1$ under variable order, otherwise $q = q_{\max}$;
   - add one to `startup`.
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
   - Form the coefficient tables of §4.7 from the nodes.
   - Evaluate the source once.
   - Reduce the scaled defect norms $\varepsilon_k$ for $k \in O$ (SEP 16 §4.2;
     [stepper.md](../stepper.md)). The scale is $r_{\mathrm{tol}}\,w\,\lVert u_n \rVert
     + a_{\mathrm{tol}}$, with $u_n$ the accepted state before the step, over the norm
     mask.
   - Add one to `predictor`.
7. **Decide** (§4.4), and record the attempt (§4.8).

Steps 6 and 7 are the engine's and the controller's; everything else is the march's. A
prescribed schedule (phase 3) replaces steps 3–4 with the schedule's step and edge, and
accepts every finite norm. A non-finite norm on a replay is the event *diverged*, which
the host raises as `ConvergenceError`.

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
6. choose the next order $q_n$ and its norm (§4.5);
7. set $h_p = $ `limit_step`$(h, F(\varepsilon_{q_n}, q_n, g))$ and clear $\lambda$;
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
and position always agree, and the native march follows that order. Whether a short
landing should raise the floor at all is a method question (Q13).

#### 4.5 Order choice

Under a fixed order, $q_n = q_{\max}$ and the norm is $\varepsilon_{q'}$. Under variable
order every scored order $k$ has the weight

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

**Accuracy.** The nodes stay well separated: steps vary by at most $g = 2$ per accepted
step and by at least $f_{\min} = 0.2$ per rejection, and $q_{\max} \le 6$. The
Vandermonde systems are therefore small and modestly conditioned. The native tables use
the algorithm of `stepper.multistep` (`_basis_rows`, already native and strict as
`_core.basis_rows`). They are compared with the Python tables to a rounding budget fixed
from the systems' condition numbers before the comparison ([6], chapter 22).

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

**Host events.** The march returns to the host only after an attempt at which the host
has work. The host passes the bounds at every entry, and the native side tests them
with the host's own expressions:

| Event | Native test after an accepted step |
|---|---|
| target reached | `reached(z, z_T)`: $z \ge z_T - \tau$ |
| crossed (dense observation) | $z > z_T$ with planes interpolated |
| host step | the accepted-step count since entry reaches $n_e$, the smallest of: the steps to the next statistics row (`row_period`), to the next natural stop (`step_planes`), and to the boundary monitor's next cadence check (1 while an edge is busy, otherwise `CADENCE` less the steps since its last check) |
| monitor distance | $z - z_{\mathrm{last}} \ge I_{\mathrm{check}}$, the monitor's own expression, or its first check |
| call bound | the attempts or the wall time since entry reach the call's bounds (§6.5.7) |
| budget, floor, diverged | §4.3 and §4.4 |
| error | a native failure in a trial. The steps committed before it stand, and the error is returned rather than thrown (§6.5.7) |

Every attempt writes one row into a host buffer: end, step, norm (with a non-finite norm
written as $\infty$, as `_record` does), whether it was accepted, its order, and its wall
time since entry. On return, the host:

1. records the rows (`_record`, progress);
2. applies the accepted steps' bookkeeping in bulk. By construction none of them but the
   last had host work: monitor skips, statistics and natural-stop counters;
3. runs the last step's `_take_step_stats` and `_natural_stop` as today.

So every statistics row, monitor check and natural stop falls on the accepted step it
falls on today.

#### 4.9 What "the same steps" means

Both marches read norms from the same native kernels. They differ only in:

- the coefficient tables, which differ by a few units in the last place;
- the order of the scalar operations that combine them.

The controller's own arithmetic is the same IEEE operations on the same inputs: strict
in C++, with the same `pow` [7, 8]. Its outputs are discontinuous functions of the
norms: acceptance at $\varepsilon = 1$, adoption at $W_b = H W_{q'}$, and the landing and
kink tolerances.

A decision flips only when the two norms straddle a threshold, that is, when its margin
is below the norms' rounding difference, about $10^{-14}$ relative here. Across $10^4$
decisions in a run, such flips are rare, but they are not impossible. After one flip the
two trajectories differ by more than rounding, and they agree only to the run's
tolerance.

So, as SEP 25 §4.4 rules for adaptive runs, "the same steps" is:

- **expected:** the same accepted and rejected sequence, orders and landings on every
  qualification case;
- **recorded:** with the smallest decision margin of each run;
- **never asserted:** the assertions are an accuracy envelope on the fields and exact
  structural contracts (§7).

A prescribed schedule with a fixed order has no decisions, so it is a matched
computation that agrees to a rounding budget (§7.2).

### 5. Proposed interface

Nothing public changes. A user's script, configuration and output read as before. On
the native backend, `Multistep()` runs through the native march whenever the
configuration is admitted (§6.6), and otherwise through the Python march as today.

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

**(C) No controller move.** Python keeps the decision. Native does three things:

- **Forms the coefficient tables** from the nodes inside the trial call.
- **Evaluates the stage scalars** from tables uploaded once. Python passes only the
  trial's two positions and its phase increment.
- **Fuses each call.** A trial call also performs the previous trial's accept or
  discard, so a trial costs one crossing.

**What each option takes off a trial**, per trial, from today's host cost (§3.3).

**Option C.** Its parts:

- the tables become a native helper (−15 to −34 µs on the M5, −126 to −168 µs on the
  EPYC with the compensation's tables);
- the stage scalars are evaluated natively (−130 to −156 µs on the EPYC);
- the rate weights go native (−22 µs);
- the accept is fused into the next trial (−15 to −20 µs).

The decision (39–47 µs on the EPYC), the march loop, `_record`, the step bookkeeping
and one crossing stay.

**Options A and B** remove all of it but a residual:

- the controller's scalar work in C++, a few µs;
- one crossing per host event. Events come every eighth accepted step or more often by
  the monitor's distance on a quiet window, and at every step while an edge is busy.
- the bulk records, about 1–2 µs an attempt.

Per-trial savings:

| | C | A | B |
|---|---:|---:|---:|
| Saving per trial, Apple M5 Max (helium / argon), µs | 55–85 / 45–75 | 125–135 / 93–103 | 125–135 / 93–103 |
| Saving per trial, AMD EPYC 7763 (helium / argon), µs | 280–350 / 230–295 | 460–485 / 365–390 | 460–485 / 365–390 |

Host time left per trial, and the run-level effect:

| | C | A | B |
|---|---|---|---|
| Host time left per trial, M5 / EPYC, µs | 35–85 / 110–220 | 5–15 / 15–40 | 5–15 / 15–40 |
| Seam crossings | one per trial | one per host event | one per host event |
| Run-level saving alone, helium (EPYC, of 3.15 s) | 0.40–0.50 s | 0.65–0.69 s | 0.65–0.69 s |
| Run-level saving alone, argon (EPYC, of 12.6 s) | 2.6–3.3 s | 4.1–4.4 s | 4.1–4.4 s |

Ownership, rules and risk:

| | C | A | B |
|---|---|---|---|
| Who owns the parameters | solprop | native, with solprop mirroring it | solprop; native has none |
| `native/AGENTS.md` | kept | broken: native owns policy | kept in intent; its wording amended (§6.3) |
| ADR-043 | kept | replaced | amended: one algorithm, two executions, Python the reference |
| Risk | leaves a third of the EPYC's host cost | two sets of constants that drift apart | a specification and a trace test to maintain |

The M5 savings for C take the M5's measured table cost (15–34 µs) and scale the other
EPYC pieces by the measured 3.5–4 times. The helium figures include the compensation's
tables and rate weights; argon has a constant operator and needs neither.

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

- On the EPYC it leaves 110–220 µs of Python per trial, still 0.4–0.7 source calls on
  the cheap cases.
- It keeps one crossing per trial.
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

§4.3–§4.5, in binary64 on the host thread (ADR-054), written once as scalar C++ in a
`SOLPROP_NATIVE_STRICT_FUNCTION`. Its inputs are the norms the engine already reduces.

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
state, the history, the position and the ledger change together. The host's bookkeeping
for a call's rows (§4.8) runs inside one `committing()` section, so a signal can never
separate a step's records from its state (ADR-047). This is SEP 24's pattern
(`atomic`), applied to a run of steps instead of one.

**Signals.** A signal waits for the native call to return (ADR-047), so a call is
bounded:

- by its attempts (the size of the row buffer; 256 is proposed);
- by its wall time, from a monotonic clock read once per attempt (0.1 s is proposed;
  Q3).

A scheduler's `SIGTERM` or a keyboard interrupt is then served within that bound, plus
one trial.

**Errors.** A native failure in a trial (a non-finite field the source refuses, an
allocation failure) is caught inside the march. It stops at the last commit and returns
the *error* event carrying its message. The host records the committed rows, then
raises the same exception a host march would. An exception thrown through nanobind in
the middle of a call would lose the records of steps already committed.

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
- **Progress** is reported per attempt, in bulk at each event, so it lags by at most one
  call.
- **The per-step debug log** of the Python march is replaced by one debug line per
  event, built from the rows.
- **The record and the configuration hash** are unchanged. The metadata adds
  `adams_march`.

##### 6.5.9 Resume snapshots

`MarchSnapshot` keeps its form and fields (ADR-046; SEP 29 §6.6). At an event the host
reads the march state: the position, the proposal, the order, the rejection count,
$\lambda$, and the entries' positions, identities and rows. It reads the engine's rows
through the existing `adams_rows_to_host`, losslessly.

A load installs both, then calls `reset` on the held trial. Snapshots cross both ways:

- native march to host march, which is also how `numpy_ref` and `numba_cpu` continue a
  native run's history;
- host march to native march.

A resume agrees with the uninterrupted run to rounding, never bitwise (ADR-049, SEP 25
§6.8). A snapshot is taken only at an event with nothing ahead of it, as today.

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
  - A roadmap item records it, to be revisited if `numba_cpu` becomes a production
    backend for cheap capillaries.
- **Native** gets the march (phases 1–4).

**Admission.** The native march runs when all of these hold:

- the method is `Multistep()`;
- the engine is resident (`_adams_engine` returns `_ResidentAdams`);
- the norm's scale is `PIController.scale`;
- the transported compensation is used (not the untransported witness);
- no coupled flow or moving transform map is present (until phase 4).

Otherwise the host march runs, as it does today. This is not a silent fallback between
backends: both execute the same method on the same backend and agree to the run's
tolerance, and the choice is recorded in `adams_march`. The same holds today when
`_adams_engine` returns the host engine for a flow off the eigenbasis route (ADR-044).
Q2 asks whether some of these should be refused instead.

**Refused by name, unchanged:**

- `ComparisonMultistep`'s general path, which has no resident engine;
- everything `_refuse_adams` and `_refuse_unsupported` refuse today: TDSE, single
  precision, CF4, a batch axis, and on device builds the carrier frame and z tables.

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

Device qualification follows the standing policy ([backends.md](../backends.md)): later,
in one pass, in a GPU segment with its own budget (phase 5). Until then a device build
runs the host march and records it.

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
  about 0.13 s of helium's 1.39 s and 0.39 s of argon's 3.0 s on the EPYC (§10, Q8).
- **On short runs the fixed 90–190 ms per run is then a large share** of both methods'
  time: 0.12 s of a 0.7 s helium run.

### 7. Verification and validation

Tolerances follow SEP 25 and [testing.md](../testing.md). Exact equality is used only
for structural contracts, and every rounding budget is fixed before the comparison. All
runs are ten-step windows or short guides that finish in seconds, as `AGENTS.md`
requires.

#### 7.1 Exact structural contracts

- **Decision traces.** A test entry of the native march takes a policy, a state, the
  nodes and a sequence of norms, and returns the decisions. Phase 0's pure controller
  core does the same in Python. Over a corpus of traces the two must agree exactly on:
  - the accept;
  - the next order;
  - the next proposal, to the bit;
  - the restart;
  - every ledger counter;
  - the landing (step, end, kind, kink).

  The corpus is generated from phase 0's core, kept with `pytest-regressions`, and
  covers:
  - norms at $1 \pm$ one ulp, $0$, $\infty$ and NaN;
  - neighbours tied, and at $H$ exactly;
  - the collar cap binding and quiet;
  - three and four consecutive rejections;
  - the budget at its edge;
  - the floor;
  - landings within $\tau$ of a plane, a kink, a kink on a plane, and stretched;
  - equal approach;
  - the carrier cap.

  Exact equality is legitimate because the same IEEE operations on the same inputs give
  the same result. Both halves' scalar arithmetic is strict and calls the same `pow`. A
  platform whose C++ `pow` differs from Python's would compare the proposals at one ulp
  and everything else exactly, and the test says so where it applies.
- **Stage scalars.** The native bundles equal `scalars_at`'s field for field on constant,
  gradient, profile, multi-gas, Raman and plasma problems, because they use the same
  strict Clenshaw helper (§4.8).
- **Events.** Each event at the accepted step where the host march would act:
  - statistics rows at `row_period` 1, 3 and 7;
  - natural stops at `step_planes` 1, 2 and 5;
  - monitor checks busy and quiet;
  - dense crossings.

  Checked by counting rows and checks on runs that take the same steps.
- **Lossless state.** A snapshot's rows, positions, identities, rows' slots, order and
  controller state equal the stored ones exactly after a round trip in either direction
  (SEP 29 R17), and a load does not take the restart path.
- **The ledger** counts the calls actually made. The calls are counted independently
  through the source's own counter.

#### 7.2 Rounding budgets

- **Coefficient tables.** Native against `exponential_tables`, `compensation_tables`,
  `factored_defect_rows` and `fused_tables`, over random admissible node sets (steps
  varying by $[f_{\min}, g]$, orders 1–6). The budget is the Vandermonde system's
  condition number times its operation count times the unit roundoff, derived in the
  test before the comparison [6].
- **Matched marches.** A prescribed schedule at a fixed order, with matched startup and
  restart history and matched plane handling, on the native march and on the host march
  over the same engine. The fields at every plane must agree to a budget from the step
  count and the per-step rounding of SEP 25 §4.3. This is the R-comparison SEP 25 asks
  for (phase 3, when replay is native).

#### 7.3 Accuracy envelopes for adaptive runs

The native march against the host march on the same native engine. Across the routes
and switches of phase 2's admission, the cases are:

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

Each case's envelope is set in its test against a tighter or stored converged reference,
times a safety factor fixed before measuring (SEP 25 §4.4). Step and call counts are
recorded, together with the smallest decision margin of each run. A difference in steps
is reported in the phase report, never asserted against.

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

- **Signals.** A `SIGTERM` or `SIGINT` sent during a long native call is honoured within
  the call bound. The output file is consistent, and a resume continues the history.
- **Errors.** A native failure in the middle of a call leaves the committed steps
  recorded and raises the host march's exception. The floor, budget and diverged events
  raise the reference's messages.

#### 7.6 Benchmarks

The tracking capillaries are run with `python -m benchmarks ab`, in interleaved fresh
processes, with the host and native marches selected through the private switch:

- cases: `he_rdw_averaged`, `argon_blueshift_averaged`, `h2_raman_averaged`, `rdw`;
- one thread and the production width;
- on an Apple M5 Max and an AMD EPYC 7763, or the machines then current, recorded by
  hardware model.

Short windows of about ten accepted steps come first, then complete runs. The per-trial
host cost is reported from the same timers §3.3 used. Each phase adds a ledger entry in
[benchmarks.md](../benchmarks.md).

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
  paragraph under its limitations or costs: on native the march runs in the compiled
  backend, `adams_march` records it, and results agree with the host march to the run's
  tolerance.
- **The manual page [choosing a stepper](../../guide/numerics/choosing-a-stepper.md).**
  Re-measured rows once the phases land.
- **`CHANGELOG.md`.** A line citing this SEP.

No tutorial or worked example is needed: nothing a user writes changes.

### 9. Compatibility and impact

- **Defaults.** None changes. `Multistep()` stays the method, with its fields and
  constants.
- **Results.**
  - On the native backend, results change at the rounding level. That is the coefficient
    tables' few ulp.
  - A decision can flip at a knife edge, after which the run agrees with the host march
    to its tolerance (§4.9).
  - Validated results elsewhere are unchanged: `numpy_ref` and `numba_cpu` run the same
    Python march, refactored without behaviour change in phase 0.
- **Seam.** Kernel schema +1 in phase 1 and +1 in phase 2. New contract tuples
  `SCALAR_TABLES_FIELDS` and `MARCH_POLICY_FIELDS`, and additions to `DRIVER_SURFACE`.
- **Output.** Output schema +1 in phase 2 for `RunMetadata.adams_march`. Older files read
  as before, since breaking changes are allowed before the first release and there are no
  compatibility shims (ADR-051).
- **Rules.** `native/AGENTS.md`'s policy rule is amended, and ADR-043 is amended by a new
  ADR (§6.3).
- **What it breaks:**
  - ADR-043's single executed controller;
  - Python-level introspection of a native run's march: per-step debug logging and
    monkeypatching of `AdamsMarch` methods, which apply only to the host march;
  - signal latency, now up to one call bound (0.1 s proposed) instead of one trial;
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
- **Moving the Runge–Kutta march native in the same SEP.** RK's host cost is 10 µs (M5)
  and 40–50 µs (EPYC) per call. A native RK march matters for a fair default decision
  (§6.9), but its controller (the lagged PI controller, the stretch, the start rule) is
  another algorithm to specify. It is better as a follow-up that reuses this SEP's
  policy-record pattern (Q8).
- **Retuning the controller for fewer steps** on the argon case. That is a method
  change, outside this SEP, and `native/AGENTS.md` forbids it as a speed measure.

### 11. Phases

Every phase is on one branch, `feature/sep-0039-native-adams-march`, and reaches `main`
in one pull request at `Final` (SEP 0). Each phase runs targeted checks, and the full
gate runs in CI.

**Phase 0: the controller core in solprop.**

- `MarchPolicy` and `march_policy`.
- A pure core of the controller (`stencil`, `decide`, `reject`, `choose_order`, the caps,
  `approach` and `landing`) over a policy and a state. `AdamsMarch` calls it.
- No change in behaviour, except the commit order of §4.4, which is a fix: the existing
  Adams tests pass unchanged on `numpy_ref` and `numba_cpu`, and a regression test
  covers an accepted landing shorter than $h_{\min}/g$.
- The decision-trace corpus.
- The kernel-contract section, drafted.

Verified by the existing suite and the corpus's self-consistency.

**Phase 1: native tables and stage scalars** (option C's pieces).

- The scalar tables leaf, with derivative tables for the rate weights.
- Native coefficient tables formed from the nodes inside the trial call.
- Lean trial arguments: positions and phase increment only.
- The accept fused into the next trial call.
- Kernel schema +1.
- The Python march still decides.

Verified by §7.1's scalar equality, §7.2's table budgets, the existing suite, and the
benchmarks. Expected −280 to −350 µs per helium trial on the EPYC. Q11 asks whether
phase 1 lands ahead of the SEP as an ordinary optimisation.

**Phase 2: the native march**, for the exponential formulation on diagonal operators,
with everything of §6.5:

- the events;
- commits, signals and errors;
- snapshots;
- metadata;
- admission and the host march elsewhere.

Also in phase 2: the new ADR, the `native/AGENTS.md` amendment, the documentation of §8,
and kernel and output schema +1. Verified by §7.1, §7.3, §7.4, §7.5 and §7.6.

**Phase 3: the interaction formulation's fused path, and replayed schedules natively.**
Single runs only; lockstep batches keep `replay_step` on the host. Verified by §7.2's
matched marches and §7.4.

**Phase 4: coupled flows and moving maps**, natively, or a recorded decision to leave
them on the host march. The flow section and weights, the block's end weights and the
transform weights per position. Verified by `test_exponential_coupled.py` under both
marches.

**Phase 5 (later, its own budget): device builds.** The march on CUDA and HIP builds, in
one GPU segment under the standing device policy: the parity tests with `-m gpu`, and
ten-step timings against the host march.

The native items of §6.9 proceed independently. Each phase's benchmarks are measured on
`main` as it then stands, and the report says which of N1–N7 had landed.

### 12. Open questions

- **Q1.** Amend `native/AGENTS.md`'s rule with the wording of §6.3, or keep it as it is
  and record B as an exception in the new ADR? Recommended: amend. To settle before
  acceptance.
- **Q2.** Configurations outside the native march's admission: run the host march and
  record it (recommended, as `_adams_engine` already does for the host engine), or
  refuse them by name on native? To settle before acceptance.
- **Q3.** The call bound: 256 attempts and 0.1 s of wall time are proposed. A scheduler
  allows seconds, but a keyboard interrupt wants tenths. Phase 2.
- **Q4.** Does the interaction formulation's fused path deserve the native march (phase
  3)? It is not the default, but SEP 37's capillary rule measured it on every row, and
  the controller is shared. Recommended: yes, since the cost is small once phase 2
  exists. Settle at phase 3.
- **Q5.** Coupled flows and moving maps (phase 4): native, or left on the host march?
  Settle at phase 3 from measurement.
- **Q6.** Device builds: qualify in this SEP's phase 5 or in a later device SEP? It
  depends on the GPU budget.
- **Q7.** Whether phase 0's refactor should also simplify `AdamsMarch` itself. The
  general path, `_attempt`, `_correct`, `_commit` and `_next_order`, which serve only
  `ComparisonMultistep`, could move into their own class. Phase 0.
- **Q8.** A native Runge–Kutta march in a follow-up SEP, reusing the policy-record
  pattern, so that the default comparison is between equally optimised marches.
  Recommended: draft it after phase 2's measurements.
- **Q9.** A compiled march for `numba_cpu`: recorded here as an exception (§6.6), with a
  roadmap item. Settle at `Final`.
- **Q10.** The capillary default. It is not decided here. After phases 1–2 and the §6.9
  items land, SEP 37's rule (the fastest at matched accuracy among the methods every
  route of the interface admits) is re-run on the capillary rows. That decision is the
  maintainers'.
- **Q11.** Should phase 1, which needs no SEP on its own, land ahead of acceptance as an
  ordinary optimisation pull request? Recommended: yes, if the maintainers want its
  saving before the rest. The SEP's later phases then start from it. To settle before
  acceptance.
- **Q12.** Progress and logging granularity. Is one progress update and one debug line
  per event enough, or should the attempt rows be logged one by one at debug level?
  Phase 2.
- **Q13.** Should an accepted landing shorter than $h_{\min}/g$ raise the floor (§4.4)?
  The Adams march forms its next proposal from the step taken, so it does; the
  Runge–Kutta march grows from its proposal and does not. Changing the Adams rule is a
  change to the method, outside this SEP. It is listed for the maintainers, and phase 0
  only fixes the commit order.

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
