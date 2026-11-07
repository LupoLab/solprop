# SEP 45 — A native Runge–Kutta march, and boundary checks inside the compiled marches

| | |
|---|---|
| **SEP** | 45 |
| **Title** | A native Runge–Kutta march, and boundary checks inside the compiled marches |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-07) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-11-07 |
| **Resolution** | Accepted 2026-11-07 by John Travers, after his human review of Part I, the GPT-6 AI review and the response to it, and his decisions Q1–Q7 (§12), in [LupoLab/solprop#563](https://github.com/LupoLab/solprop/pull/563) |
| **Requires** | [SEP 39](archive/sep-0039-native-adams-march.md) (the native Adams march, its call protocol and ADR-060); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); [SEP 21](archive/sep-0021-absorbing-boundaries.md) and [SEP 40](archive/sep-0040-transverse-absorbing-boundaries.md) (the boundary monitor) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0045-native-rk-march` ([#567](https://github.com/LupoLab/solprop/pull/567)), based on #563 and #566: started 2026-11-07; phases 0 and 1 complete |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Remove the Python time that is still spent *inside* a warm run on the native backend.
It sits in two places:

1. **Boundary checks force the native Adams march back to Python.** While the boundary
   monitor is busy (a guard or the collar holds more than $(0.1\,\mathrm{budget})^2$ of
   the field above the floor), it asks for a check at every committed step, and the
   native march of [SEP 39](archive/sep-0039-native-adams-march.md) returns to solprop
   at every one. Each return costs 100–250 µs of Python on an Intel Xeon Gold 6240. On
   the tracking case `argon_blueshift_envelope` the march returns at 1910 of its 2227
   attempts and spends 17–21 % of the run in Python.
2. **The Runge–Kutta march is still a Python loop.** `RungeKutta()` (interaction-picture
   DOPRI5, the default of `prop_gnlse` and `prop_waveguide`) steps through solprop's
   shared `StepDriver._march_to` with one native call per trial, and spends about
   200 µs of Python per step: 22–24 % of `gnlse_soliton`, 4–6 % of `gnlse_raman_shock`.

The proposal has two parts, in this order:

- **B. Boundary checks inside the compiled march.** The native march decides when a
  check is due, takes the reading, predicts the monitor's verdict from thresholds
  solprop computed, and buffers each reading. It no longer returns for a check; it
  returns at the other host events, at its call bound, or when a check's verdict needs
  Python (a warning, a stop, a change of the quiet verdict). solprop replays every
  buffered check through the unchanged `BoundaryMonitor.step` when it drains the call,
  so the monitor's record and its arithmetic stay in one place, and verifies the
  prediction against it. The replay still costs Python time per checked step; what B
  removes is the crossing and the state adoption per check (§4.1).
- **A. A native Runge–Kutta march.** The adaptive (and later replayed) DOPRI5 loop runs
  in `solprop-native` under SEP 39's design: solprop hands over one policy record and
  the events at which to return, native executes an algorithm this SEP and the kernel
  contract specify in full, and solprop's Python march stays the reference.

B comes first because A depends on it: a native RK march that still returned at every
busy check would save little, and `gnlse_soliton`'s monitor asks for a check at about a
third of its steps (§3.4).

Success means: with B, the native Adams march takes the same steps and writes a
bitwise identical boundary-monitor record as before, and no longer returns for a busy
monitor's checks; with A, the native RK march takes the same steps as solprop's Python
march on the same build; no case of the tracking set is slower beyond its A/A spread;
and the Python share of a warm run falls to the targets phase 0 sets from the measured
cost of what stays in Python (§7.6). The provisional targets are 5 % on
`argon_blueshift_envelope` and 4 % on `gnlse_soliton`.

### 2. Motivation and scope

**Who pays.** The Python share of a warm run is largest where a step is cheap:

- small-grid fibre runs (solitons, short supercontinua, soliton-fission studies) on the
  default `RungeKutta()`, and the scans of many such runs that solprop's executors are
  built for;
- envelope-grid and mode-averaged capillary runs whose boundary monitor stays busy;
- every device build, where a trial is cheap and the host's share of it is large
  (removing 41–70 µs of host time a trial gave the native Adams march wall-time gains
  of 3–23 % on an NVIDIA H200; `docs/dev/benchmarks.md`, "The native Adams march").

Python is serial, so its share grows with the thread count: on `he_rdw_averaged` it is
4.1 % at one thread and 7.5 % at sixteen.

**In scope.**

- B on the native Adams march, for every monitor configuration except a transverse
  edge (§6.6).
- A: the adaptive interaction-picture Runge–Kutta march (`integrator is None`) on the
  native backend, then its replay of a fixed schedule, interpolated planes, z tables,
  moving maps and coupled flows, phase by phase (§11).
- A native trial packer and error-norm scorer that the Python loop uses as well, so that
  both marches read the same norms (§6.3).
- Device builds, qualified separately (phase 5).

**Not in scope.**

- **Cold start**: interpreter and import time, lazy imports on first use (CoolProp,
  SciPy's signal module), FFTW planning. These are paid once per process.
- **Setup**: building and translating a problem. The large-grid and graded-index
  setups have Python-side remedies (vectorisation, reuse within a build) that need no
  SEP and go in their own pull request.
- **The boundary monitor's policy.** `CADENCE`, `ACTIVE`, `CHECKS`, the floor and the
  budget are unchanged. This SEP changes where the comparisons are evaluated, not what
  they are.
- **A single march protocol** across all step loops (#110). This SEP adds one compiled
  loop and shares its landings and call machinery with the Adams march (§6.9); it does
  not decide #110.
- **Lockstep batches**, which keep their host loop (SEP 39 Q14), the low-storage march
  (already one native call per step), `numpy_ref` (the reference) and `mlx`.
- **Faster native kernels.** A trial's arithmetic is unchanged.

### 3. Context

#### 3.1 The Runge–Kutta march today

`StepDriver._march_to` (`src/solprop/backends/driver.py`) is the adaptive loop of every
backend. Per attempt it:

1. polls the stop flag;
2. caps the controller's proposal (`_quiet_capped`, `_transverse_capped`,
   `_split_capped`) and lands it on the next kink, save plane or end
   (`solprop.stepper.schedule.landing`, `LANDING_ULPS = 8`);
3. calls `_trial_step(z, dz)`. On `native` that is `_prepare_trial` (stage scalars for
   every tableau node by `_hoist` and `_stage_bundles`, packed by `_trial_arrays` and
   `_weight_rows` into about 25 small arrays), one native `trial_step`, and
   `_score_trial`, which sums the returned norm partials with NumPy and applies
   `controller.scale`;
4. forms the equivalent norm at the proposed step (`PIController.equivalent_norm`) and
   decides acceptance (`PIController.accepts`);
5. records the attempt (`_record`); on acceptance calls `promote_fsal`, `finish`,
   commits, applies the start-up rule on the first accepted step (`_start_up`),
   proposes the next step (`PIController.propose`, `limit_step`), and takes the
   per-step statistics and boundary check (`_take_step_stats`); on rejection scales
   the proposal (`PIController.rejection_factor`).

The method is the Dormand–Prince 5(4) pair [1] in the interaction picture [2]
(ADR-004). The controller is Söderlind's PI.4.2 [3] with ADR-004's lag and reject-free
options, and the start-up rule takes the deadbeat step of [4, §II.4]; its
state is its own proposal, never the clipped or stretched step executed
(`driver.py`, the comment before `landing`). Replay of a fixed schedule is
`StepDriver._replay_to`; interpolated planes use `_dense_record` and `_observe_inside`.
The native trial (`GeneralWorkspace::trial_step`, `native/src/general.cpp`) already
fuses the stages, the running accumulators of ADR-009 and the three norm partials.

#### 3.2 The native Adams march and its events

SEP 39 moved `Multistep()`'s loop into `native/src/adams_march.cpp`. Each call runs
attempts until a host event, at most 256 attempts or 0.1 s (SEP 39 Q3), and returns one
row per attempt. The adapter `_NativeAdamsMarch` (`native/src/solprop_native/backend.py`)
builds the call's bounds (`_bounds`): `host_step` is the fewest accepted steps until the
driver has work, the minimum of the next statistics row, the next natural stop, the
monitor's `steps_to_check()` and, for a transverse edge read every step, one. After a
call it adopts the march's state (`_synchronise`), books each row (`_record_attempt`,
`_skip_step`) and, at the event step, runs `_take_step_stats`. The kernel contract's
§11 specifies all of this.

#### 3.3 The boundary monitor

`BoundaryMonitor` (`src/solprop/monitor.py`) reads the state's energy in the temporal
guards and the collar (`monitor_reading`, ten numbers plus, under the split scheme, 128
walk-off buckets) and judges it:

- `due(z)`: a check is due at every committed step while `_busy`, otherwise every
  `CADENCE = 8` steps or after `interval = length / CHECKS` (256 checks over the guide);
- `step(energies)`: commits the reading, updates the crossing cap (`_read_cap`, split
  scheme), the split's found energies (`_found`), the shares above the floor (`_above`),
  the verdict (`_judge`: `BoundaryError` above `stop_factor × budget`, one
  `BoundaryWarning` above `budget`) and the cadence and quiet flag (`_settle`);
- `skip(z)`: counts an unchecked step.

The removal ledger of the in-stage scheme is a trapezoid over the *checked* positions,
so the record depends on exactly which steps were checked. SEP 39 §6.5.7 and SEP 44
kept judging in Python; SEP 44's option D ("native also judges") was rejected for the
transverse edge because one of its reads costs tens of milliseconds, an argument that
does not carry to the temporal monitor, whose reading costs one inverse transform.

#### 3.4 The measurements this SEP starts from

Intel Xeon Gold 6240 (one socket, `taskset` to its 18 cores), GCC 16.2,
`-march=cascadelake`, commit `4b22e5c9`, the benchmark harness's builders and its
`timed_run`, Linux `perf` sampling the Python main thread at 1 kHz during a warm repeat.
Another user's process shared the socket (load 5–7), so read the shares, not the walls.
"Python" is the interpreter plus NumPy's and SciPy's extension modules; the remainder is
`solprop-native`, FFTW, OpenBLAS and OpenMP, plus 2–12 % unattributed (libc and kernel).

| case (integrator) | threads | steps | warm run (s) | Python share | Python µs per step |
|---|---:|---:|---:|---:|---:|
| `gnlse_soliton` (RK) | 1 | 691 | 0.62 | 24.3 % | 219 |
| | 16 | 691 | 0.60 | 22.0 % | 190 |
| `gnlse_raman_shock` (RK) | 1 | 8469 | 45.1 | 3.7 % | 198 |
| | 16 | 8469 | 34.9 | 5.6 % | 229 |
| `argon_blueshift_envelope` (Adams) | 1 | 2227 | 1.16 | 16.9 % | 88 |
| | 16 | 2227 | 0.90 | 20.7 % | 84 |
| `argon_blueshift_averaged` (Adams) | 1 | 11168 | 4.15 | 7.4 % | 27 |
| `h2_raman_averaged` (Adams) | 1 | 12680 | 6.51 | 6.5 % | 34 |
| `he_rdw_averaged` (Adams) | 1 / 16 | 1437 | 1.26 / 0.74 | 4.1 / 7.5 % | 35 |
| `rdw`, `argon_blueshift`, `radial_filament_256`, `ctc` (0.1) | 1–16 | — | — | 0.7–4.2 % | — |

How busy the monitor is, from the run's `boundary_monitor` record (one thread, the
whole guide unless a fraction is given):

| case | accepted steps | checked | skipped |
|---|---:|---:|---:|
| `argon_blueshift_envelope` | 2227 | 1910 | 321 |
| `gnlse_soliton` | 691 | 217 | 484 |
| `gnlse_raman_shock` (0.3) | 2701 | 342 | 2370 |
| `he_rdw_averaged` | 1434 | 243 | 1201 |
| `argon_blueshift_averaged` (0.3) | 3189 | 403 | 2796 |

(The checked and skipped counts can overlap by a few steps at planes, #133.) A quiet
monitor checks about one step in eight; `argon_blueshift_envelope` is busy for about
85 % of its guide and `gnlse_soliton` for about a fifth. A cProfile of the envelope case
attributes the Python time to the return path: `_call`, `_state`, `_synchronise` with
`march_stepped`, the row booking, `_boundary_energies` and `monitor.step`, each called
1910 times. A cProfile of `gnlse_soliton` attributes it to `_stage_bundles` and
`scalars_at` (Chebyshev reads at each stage), `_trial_arrays`, `_weight_rows`,
`landing`, `_score_trial`, `_boundary_energies` and `monitor.step`.

SEP 39 §6.9 estimated that a native RK march would save 0.13 s of a 1.39 s helium run
and 0.39 s of a 3.0 s argon run on an AMD EPYC 7763, and Q8 left it for another time,
"reusing the policy-record pattern". The table above is the case for taking it up now:
on small-grid fibre runs, where `RungeKutta()` is the default, the host loop is a fifth
of the run.

#### 3.5 The rules this SEP meets

- `native/AGENTS.md`: native holds no physics or numerical defaults; every constant a
  native decision reads arrives in a record that solprop forms. Today that rule names
  the Adams march's policy; this SEP extends it to the RK policy and the watch policy.
- ADR-060: the native Adams march, with solprop's march as its reference. This SEP's
  ADR extends the pattern to the RK march and to boundary checks.
- ADR-047: a signal or stop request is honoured between commits; ADR-049 and SEP 25:
  backends agree to the error bound, not bitwise; "the same steps" is a diagnostic on
  one build.
- ADR-054: the controller runs in binary64 on the host side of the seam.
- ADR-057: the quiet collar.
- Kernel contract §10: a new compiled loop needs a contract section that specifies it in
  full (here §12, and an extension of §11).

#### 3.6 Issues

Fixed or changed by this SEP:

- #273 — SEP 39's frequent-event rows (a busy monitor) were never recorded; phase 1
  records them and removes most of the cost they would show.
- #135 — the no-contraction rule has no working witness. The new strict files share the
  problem; phase 0 adds a witness that can fail where it runs (an optimised build with
  FMA in the target) for all three files.
- #133 — the monitor counts a step checked at its plane twice. Resolved first, in its
  own pull request before phase 1 (Q6), so that this SEP's bitwise record comparisons
  pin the corrected counts; the drain books each step once (§4.10).

Related, not fixed:

- #110 — a SEP for the step loops and the kernel seam (§6.9).
- #351 — `PiecewiseChebyshev` sums every padded coefficient; it affects the native
  stage tables the RK march reads, unchanged.
- #264, #267 — FSAL and the interval cap beside a coupled flow; phase 4 keeps today's
  behaviour.
- #223, #224 — the quiet collar's limits; unchanged.
- #300 — per-step plasma statistics under Adams; such a row is still a host event.

### 4. Theory

There is no new physics. This section specifies the two algorithms precisely enough
that the native code can be written against it and checked against solprop's.

#### 4.1 The cost of a run

Write a warm run's wall time as

$$T = \sum_{\text{attempts}} t_{\text{trial}} + N_{\text{returns}}\, h_{\text{return}}
+ N_{\text{attempts}}\, h_{\text{row}} + N_{\text{checks}}\, h_{\text{check}}
+ N_{\text{attempts}}\, h_{\text{loop}} + T_{\text{fixed}},$$

with $t_{\text{trial}}$ the native trial; $h_{\text{return}}$ the Python cost of one
return (the bounds, the state's adoption, the event step's statistics); $h_{\text{row}}$
the booking of one attempt (its history row, its counters, its progress callback);
$h_{\text{check}}$ one `BoundaryMonitor.step` on a reading; and $h_{\text{loop}}$ the
per-attempt Python of a host loop. Today the RK march has $h_{\text{loop}} \approx
200$ µs and returns at every attempt; the Adams march has $h_{\text{loop}} \approx 0$ and
$N_{\text{returns}}$ equal to the number of host events, which a busy monitor drives to
the number of steps, each return also paying a native reading through Python.

B leaves $N_{\text{returns}}$ the planes, statistics rows, natural stops, call bounds
(256 attempts or 0.1 s, SEP 39 Q3), watch-buffer capacity and verdict changes; it keeps
$h_{\text{row}}$ and $h_{\text{check}}$, now paid in the drain. A makes the RK march's
$h_{\text{loop}}$ zero, as the Adams march's is. The share removed is largest where
$t_{\text{trial}}$ is smallest: small grids, many threads and devices. Phase 0 measures
$h_{\text{row}}$ and $h_{\text{check}}$ (a cProfile of the envelope case attributes about
35 µs to `monitor.step` and 30 µs to the reading's round trip, inflated by the profiler)
and sets §7.6's targets from them.

#### 4.2 The RK march's policy and state

The **policy** is one record, formed once by solprop from the controller and the
stepper settings:

- the controller: `safety`, `beta1`, `beta2`, `min_factor`, `max_factor`, `lag`,
  `reject_free`, the estimator order $q$ and the exponent `1/(q+1)` as one double;
- the start-up rule: `auto_init` and `START_GROWTH`;
- the step limits `dz_min` and `dz_max`, `interval_cap`, and the landing targets
  (kinks, save planes, end) and `LANDING_ULPS`;
- the tableau's nodes $c_i$ and whether it is FSAL;
- the norm scale: `rtol`, `atol` and whether per-bin weights apply.

The **state** crossing each call is: the position $z$, the proposal, the accepted-norm
history (at most `lag + 2` values), the accepted count, and the replay cursor. The RK
march keeps no field history, so nothing else crosses. This is a continuation between
the calls of one march, not a resume: `StepDriver.set_initial` resets the proposal to
`dz_init`, clears the norm history and restarts the statistics and natural-plane
counters, on a fresh launch and on a resume alike, and `march_snapshot()` stays `None`
for the RK march. A disk resume therefore starts the controller afresh, as today, and
takes the start-up rule again (`StepDriver._start_up`). Every `set_initial` also resets
the watch state (§4.8), clears the recycled FSAL stage and any dense step, and the march
is configured again after the run's monitor (a resumed one included) is attached.

#### 4.3 One RK attempt

Given the state, one attempt is, in order, exactly as `StepDriver._march_to` does it.
$p$ is the controller's proposal, its state; it is never replaced by a capped or landed
length.

1. $\hat p \leftarrow$ the landing candidate: $p$ capped by the quiet collar, the transverse
   ceiling and the split crossing cap, in that order (`_quiet_capped`,
   `_transverse_capped`, `_split_capped`). $p$ is unchanged.
2. $(\Delta z, z_{\text{target}}) \leftarrow \mathrm{landing}(z, \hat p, \text{targets})$.
3. The trial at $z, \Delta z$: stage scalars at $z + c_i \Delta z$ and the whole-step
   row (§4.7), one native trial, and the scaled norm $e$ (§4.4).
4. $\tilde e \leftarrow \mathrm{equivalent}(e, \Delta z, p)$, from the uncapped
   proposal $p$, and $a \leftarrow \mathrm{accepts}(e)$, from the executed norm.
5. The row (end, step, $e$, $a$) is recorded **before** the branch; the end is
   $z_{\text{target}}$ when the step landed, else $z + \Delta z$.
6. On acceptance: `promote_fsal` (which also scores the quadrature alarm), `finish`,
   commit $z \leftarrow$ end; append $\tilde e$ to the history; if `auto_init` and this
   is the first accepted step, apply the start-up rule (§4.5) to $p$; then
   $p \leftarrow \mathrm{limit}(p, \mathrm{propose}(\text{history}))$; then the step's
   boundary check (§4.8), statistics row and natural stop, in that order (§4.10), any of
   which may end the call.
7. On rejection: $p \leftarrow \mathrm{limit}(p, \mathrm{rejection}(\tilde e))$, the
   uncapped proposal shrunk, as for an uncapped step.

A cap therefore shortens the executed step and, through the equivalent norm, tells the
controller what its own proposal would have scored; it never becomes the controller's
state. With proposal 0.01 m, a cap of 0.005 m, an executed norm of 0.5 and $q = 4$, the
history receives 16, not 0.5.

The floor (`limit_step` raising `ConvergenceError` below `dz_min`) can fire in three
places, each with its own order of effects: after a rejected attempt was recorded; in
the start-up rule's `limit_step`, before the history's last value is rewritten; and in
the accepted step's proposal, after the commit and the append, before the step's
boundary check and statistics. The native march reports which and solprop raises it
there; §4.10 gives each site's effects in the drain.

Replay (phase 4) follows `StepDriver._replay_to`: it never rejects, records after the
commit, appends the raw norm, and stops within `replay_tolerance`.

#### 4.4 The norm and the controller

The trial returns three partials per norm slot: $\sum |y|^2$, $\sum |y_{\text{new}}|^2$
and $\sum |\varepsilon|^2$, each over the bins of `problem.grids.norm_mask`, which the
controller's `region` selects (`"active"`, the default, or `"band"`). With
$\|y\| = \sqrt{\sum |y|^2}$ and $\|y_{\text{new}}\|$ likewise, both Euclidean, the weak
norm is

$$e = \frac{\sqrt{\sum|\varepsilon|^2}}{\mathrm{rtol}\,\max\!\left(\|y\|, \|y_{\text{new}}\|\right) + \mathrm{atol}}$$

(`weak_norm_scale`), with no division by the number of bins. With per-bin weights $w_i$
the scale is per bin, $s_i = \mathrm{rtol}\, w_i \max(\|y\|, \|y_{\text{new}}\|) +
\mathrm{atol}$ (`weighted_norm_scale`), and each bin's error is divided by its own scale
before the Euclidean reduction, $e = \sqrt{\sum_i |\varepsilon_i / s_i|^2}$; the magnitude
stays global. These are `NativeBackend._score_trial` and `numpy_ref.passes.error_norm`
today.

Today `_score_trial` sums the partials with NumPy (pairwise above eight terms) and
calls the controller's own `scale` on the host. Phase 2 moves that reduction and the
stock scale into one native scorer that **both** marches use, so that the two read
identical norms by construction rather than by transcribing NumPy's summation order.
Only the order of a sum changes; the formula does not. A controller whose `scale` is
not `PIController.scale` keeps today's host `_score_trial`, unchanged, as the Adams
engine already does (`_ResidentAdams` tests `type(controller).scale is
PIController.scale`). That retained host scorer is also the independent reference the
new scorer is tested against, beside `numpy_ref`'s `error_norm`, so that a scaling
mistake shared by the two marches cannot pass unseen (§7.4).

The controller's functions, with $k = q + 1$ and `max_factor`, `min_factor` the clip:

$$\mathrm{equivalent}(e, \Delta z, p) = e \left(\frac{p}{\Delta z}\right)^{k}, \qquad
\mathrm{accepts}(e) = \bigl(e \le 1 \lor \texttt{reject\_free}\bigr) \land e < \infty,$$

$$\mathrm{propose}(e_0, \ldots, e_n) = \mathrm{clip}_{[\texttt{min\_factor},\,\texttt{max\_factor}]}\!\left(
\texttt{safety}\,\bigl(1/e_{j}\bigr)^{\beta_1} \bigl(1/e_{j-1}\bigr)^{\beta_2}\right),
\quad j = n - \ell,$$

$$\mathrm{rejection}(\tilde e) = \mathrm{clip}_{[\texttt{min\_factor},\,1]}\!\left(
\texttt{safety}\,\bigl(1/\tilde e\bigr)^{1/k}\right),$$

with $\ell$ the lag. `propose` returns 1 until the history holds $\ell + 1$ values,
`min_factor` for a non-finite $e_j$, `max_factor` for $e_j \le 0$, and takes $e_{j-1} = 1$
when it is missing, non-finite or not positive; `rejection` returns `min_factor` for a
non-finite norm and 1 for a non-positive one. These are summaries of
`solprop.stepper.controller` (`PIController.equivalent_norm`, `accepts`, `propose`,
`rejection_factor`, `start_factor`, and `limit_step`). The native code transcribes those
functions, not these summaries: the powers are of $1/e$, not $e^{-\beta}$, and each
product is formed left to right. The contract's §12 quotes them with every operation in
the order Python performs it.

#### 4.5 The start-up rule

With `auto_init`, the first accepted step at proposal $p_0$ with norm $e_0$ sets
$f \leftarrow \mathrm{start\_factor}(e_0)$ (at most `START_GROWTH` $= 100$),
$p \leftarrow \mathrm{limit}(p_0, f)$, and rewrites the history's last value to $0$ if
$f \ge$ `START_GROWTH`, else to $\mathrm{equivalent}(e_0, p_0, p)$ (SEP 14 §4.2,
`StepDriver._start_up`). The accepted step's ordinary proposal follows in the same step
and reads the rewritten value; with `lag = 0` that is not a no-op.

#### 4.6 Landings, caps and the floor

Landings, `reached`, the ULP tolerance and Python's `min`/`max` semantics with NaN are
those of the Adams march, which already transcribes them (`landing`, `reached`, `ulp`,
`py_min`, `py_max` in `adams_march.cpp`). They are factored into a record both marches
share (§6.4); the RK march does not use the Adams march's equal approach, kink restarts
or order choice.

#### 4.7 Stage scalars

The stage scalars (operator increments $\Phi(z+c_i\Delta z) - \Phi(z)$, prefactors,
densities, tapers, Raman and dispersive densities, moving-map weights) are already
evaluated natively from the uploaded expansions for the Adams march
(`stage::Tables`, `stage::Anchor`, `stage::ZSlots`, `stage::row` in
`native/include/solprop_native/stage_tables.hpp`), bit for bit with solprop's
`scalars`. The Adams trial reads one row per step; the RK trial needs one row per
distinct node and the whole-step row, in the layout `_trial_arrays` builds. A new
`stage::rows` produces it; phase 2 shows the rows equal `_trial_arrays` bit for bit.

#### 4.8 The boundary check inside a march

Let the monitor's **watch state** be what its decisions read and change: the last check
position (or none), the steps since it, `busy`, the quiet run length, `quiet`, `warned`
and the crossing cap. Let the **watch policy** be numbers solprop computes from the
monitor and its check at the call, so that native repeats no `pow` and holds no
constant: the floor regions already scaled by `FLOOR_MARGIN` (`_floor_regions` times
$[3, 3, 3, 1]$), the launch floor `_FLOOR * _launch`, the activity threshold
`(ACTIVE * budget) ** 2`, the stop threshold `stop_factor * budget` and the warning
threshold `budget`, `QUIET`, `QUIET_CHECKS`, `CADENCE`, `interval`, whether the quiet
collar may rest, and under the split scheme the crossing share and each bucket's step
limit (`split_crossing_step` of `split_bucket_rates(fastest)`, 128 values).

After each committed step that reaches its statistics (not at a floor exit, §4.3),
native does what `_take_step_stats` asks of the monitor, in `BoundaryMonitor.step`'s
order:

1. **Due?** `due(z)`: busy, or the steps since the last check plus one reach
   `CADENCE`, or no check yet, or `z` at least `interval` past the last check. If not,
   count the step as skipped (as `skip`) and continue.
2. **Read** the state (`monitor_reading`, as today) and append (row index, reading) to
   the call's watch buffer.
3. **Predict**, operation by operation as the reference performs it, in binary64
   without contraction:
   1. under the split scheme, the new crossing cap from the reading's occupancy
      (`_read_cap` and `SplitCrossing.limit`, with the reduction specified below);
   2. `_found`, then `_judged`: the total, zero when not positive or not finite;
   3. if the total is not above the launch floor, `busy` is false, the quiet run counts
      a calm check and nothing is judged (`_settle(None)`);
   4. otherwise `above = regions - scaled floor`, its first three clipped at zero
      (`_above`); `shares = above / total`; `busy` is
      `max(shares[0], shares[1], shares[2]) > activity`, calm is
      `regions[2] / total < QUIET`, and the quiet run and `quiet` follow (`_settle`);
   5. `worst = sqrt(max(shares[0], shares[1], 0.0))`, the correctly rounded square root
      as `math.sqrt` takes it, with Python's `max` (`py_max`); a **stop** is predicted
      if `worst > stop threshold`, and otherwise a **first warning** if
      `worst > warning threshold` and not `warned`, which then sets `warned` (`_judge`).
      The thresholds are amplitudes and the comparison is on the amplitude: no
      threshold is squared, since a squared threshold is not equivalent at the doubles
      beside it.
4. **Return** (`Reason::watch`) at this row if it predicts a stop, a first warning, a
   change of the quiet verdict, or, on a replayed schedule under the split scheme, a
   crossing the schedule cannot honour (`_refuse_scheduled_crossing`, read with the
   row's own replay cursor). Otherwise continue: the new cap bounds the next landing
   candidate (§4.3, step 1), as `_split_capped` does in the host loop.

**The split cap's reduction.** `SplitCrossing.limit` takes the total with NumPy's
pairwise `sum` and the cumulative occupancy with a separate `cumsum`, then the first
bucket whose cumulative share exceeds `share * total` (`searchsorted(...,
side="right")`). Disabling contraction does not make a sequential C++ sum equal
NumPy's pairwise total, and the choice of bucket can turn on the last bit: with bucket 0
holding `0x1.0c6f8ba2f9813p-20`, bucket 1 holding 1 and every other bucket `2**-54`, at
the default share of `1e-6`, NumPy's total selects bucket 1 and a sequential one bucket
0, limits about 1.18 times apart. Phase 0 changes the reference (Q7, decided) to take the total as the last element of
the cumulative sum, which NumPy forms sequentially and native reproduces exactly; the
selected bucket can then move one place in such a near-tie, a reference change recorded
in the release notes.

**The buffer.** Its capacity is the call's attempt bound (256 rows, each the ten numbers
and, under the split scheme, the 128 buckets), allocated once with the march and named
in the memory ledger. A call made without an attempt bound, as the call-bound
diagnostics of §7.3 do, returns with `Reason::capacity` when the buffer is full; the
call-bound tests cover that boundary.

When solprop drains a call it replays the buffered readings through the unchanged
`BoundaryMonitor.step` at their own rows and positions (§4.10), so the monitor writes
the record it writes today and raises or warns at the same row. After the drain solprop
compares its monitor's watch state with the march's; any difference, or a verdict the
march did not predict, raises `BackendError` (Q4).

#### 4.9 What "the same steps" means

As in SEP 39 §4.9: on one build, the native and Python marches read the same norms
(from the same native scorer, §4.4), apply the same decisions in the same order in
binary64 without contraction, and so take the same steps and produce the same fields.
That is a diagnostic, recorded and tested on short windows, not a promise: across
builds, compilers and libms the guarantee is SEP 25's agreement to the error bound.

#### 4.10 The drain

A drain books every attempt a call returned, in order, through one driver hook per
accepted step, so that each accepted step is counted once in **each** of its counters
(the boundary monitor's checked or skipped, the statistics-row period, the natural-plane
count) and each buffered reading is consumed once. `_skip_step` today does the counters
and `monitor.skip`; it becomes `_account_step(z, reading)`, which calls `monitor.step`
on the buffered reading (through `_check_at`, which books a reading at a given position
and cursor) or `monitor.skip`, and then the counters. `_take_step_stats` is split into
its monitor part and its statistics part, so that the final row's monitor work is the
buffered reading too and is never taken a second time.

| attempt | inside the call's `committing()` | after it |
|---|---|---|
| rejected | history row | — |
| rejected, at the floor | history row | `ConvergenceError` |
| accepted, not the call's last | history row; `_account_step` (checked or skipped) | progress |
| accepted, last, at a host event (target, plane, statistics row, natural stop) or a watch exit | history row | monitor part with the buffered reading or a skip; statistics row; natural stop; progress |
| accepted, last, at the call bound or the buffer's capacity | history row; `_account_step` | progress |
| accepted, at the floor (Adams; RK's start-up and proposal sites) | history row; no monitor work, no counters, no statistics | `ConvergenceError` |
| a native fault | the rows before it, as above; the faulted attempt is not booked | the error |

The order after the commit is the reference's: the monitor (which may raise
`BoundaryError` or warn), then the statistics row, then the natural stop, with no
second reading. A watch exit puts the row that needs Python last, so a predicted stop
raises with the state, the history and the monitor record complete to that row, which
is the run's last committed position, and the terminal checkpoint is written there as
for any `BoundaryError` today; a predicted warning (or one turned into an error) is
raised at the same point. A signal during the drain is deferred by `committing()` and
delivered after it, as in the Adams adapter today. A monitor exception inside the commit
can only come from a row the march did not predict, and is turned into `BackendError`.
#133's double count at planes comes from `_monitor_plane`; it is fixed before phase 1
(Q6), and the drain is written against the corrected counting.

#### 4.11 Interruptions and callbacks

Two kinds of event reach Python differently.

- **Deterministic events** (the floor, a predicted stop or first warning, a native
  fault, a natural stop, a scheduled crossing refused) reach Python at exactly the step
  at which the host loop reaches them, with the same state, history and record.
- **Asynchronous requests** (the stop flag, a signal) are polled before each call, as
  for the Adams march (#519). A call ends at its bound: 256 attempts, or the end of the
  attempt during which 0.1 s elapsed. A request therefore takes effect at a committed
  step at most 256 attempts later; the time is the call's bound plus one attempt and the
  drain, not an absolute ceiling. What is promised is a coherent state, history, monitor
  record and checkpoint at that committed step, not the host loop's step.

Progress callbacks keep their count (one per attempt) and their exception semantics; they
are delivered after the call's commit, so their timing, not their number, changes, as
for the Adams march today.

### 5. Proposed interface

No public name changes. A user sees:

- faster warm runs on the native backend;
- a run's metadata gains `rk_march` (`"native"` or `"host"` with a reason), beside
  SEP 39's `adams_march`;
- `HostMarchWarning` is widened to name the RK march when a configuration it does not
  cover falls back to solprop's loop (§6.6);
- `boundary_monitor` records unchanged;
- progress callbacks still one per attempt, delivered after each call's commit, and a
  stop request or signal honoured at a committed step within one call's bound (§4.11),
  as for the Adams march today;
- a disk resume of an RK run starts its controller afresh, as today (§4.2).

The private environment variable `SOLPROP_RK_MARCH=host`, read at import, selects
solprop's loop for A/B comparison, as `SOLPROP_ADAMS_MARCH=host` does; after phase 2
that loop scores with the shared native scorer (§4.4).

### 6. Implementation strategies

#### 6.1 Boundary checks: three options

| Option | What moves native | Python per busy step | Risk |
|---|---|---|---|
| B1. Native predicts, Python replays (recommended) | `due`, the reading, the prediction, the crossing cap | the replayed `monitor.step` and the row's booking ($h_{\text{check}} + h_{\text{row}}$, §4.1); no crossing, no state adoption, no reading through Python | the predictions exist twice; verified at every drain |
| B2. The whole monitor native | everything, including records, ledger, plane rows, summary, resume | the row's booking | duplicates the record arithmetic and resume; against SEP 39 §6.5.7 and SEP 44 |
| B3. Cheaper booking only | nothing; one bulk booking per call | one return, one `_synchronise` and one reading through Python per busy step | low, but removes perhaps half the cost |

B1 is recommended. It amortises the crossing and the state's adoption, not the replay;
phase 0 measures what the replay and the booking cost before §7.6's targets are fixed,
and if the replay dominates, a cheaper `monitor.step` is the next lever, not a native
record.

#### 6.2 The RK march: three options

| Option | Python per step | Notes |
|---|---|---|
| A1. Native march, SEP 39's design (recommended) | ~1–3 µs, as Adams | policy record and events; Python march stays the reference |
| A2. Native packer and scorer only, loop in Python | perhaps 80–120 µs | phase 2 of A1 on its own; removes the stage-scalar and packing time but keeps the loop |
| A3. Leave it | ~200 µs | SEP 39 Q8's choice |

#### 6.3 Recommendation and order

B1, then A1 built on A2. Phase 2 (A2) is a step on the way, and gives the Python loop the
native scorer so that both marches read the same norms. That moves the native RK
backend's norms at rounding level once, by the order of a sum and nothing else, which is
recorded (§9); a controller with its own `scale` keeps today's host scorer (§4.4).

#### 6.4 Layers and the seam

- **solprop** (`src/solprop/`): `BoundaryMonitor.watch_policy()` and `watch_state()`;
  `_check_at`, which books a reading at a given position and replay cursor;
  `_account_step` in place of `_skip_step`, and `_take_step_stats` split into its
  monitor and statistics parts (§4.10);
  a pure `solprop/backends/rk_core.py` holding the RK decisions (caps, start-up,
  propose, reject, the floor sites) that `StepDriver._march_to` and `_replay_to` call,
  extracted with no change in behaviour, as `adams_core.py` was for Adams.
- **solprop-native**: `native/src/watch.cpp` (the watch, compiled without contraction);
  `native/src/rk_march.cpp` (the RK march, likewise); a shared landings record factored
  out of `march::Policy`, keeping `MARCH_POLICY_FIELDS` unchanged; `stage::rows`; the
  native scorer; bindings `rk_march_configure`, `rk_march_run`, `watch_configure`; the
  adapter `_NativeRungeKuttaMarch` beside `_NativeAdamsMarch`, reusing `Bounds`, the row
  `Column` layout, `Reason`, `fail()` and `MarchFault`.
- **The seam**: new frozen field tuples `RK_MARCH_POLICY_FIELDS` and
  `WATCH_POLICY_FIELDS` (`native/src/solprop_native/__init__.py`), checked exactly like
  `MARCH_POLICY_FIELDS`; `DRIVER_SURFACE` gains `_account_step` and `_check_at` and
  loses `_skip_step`; `Reason` gains `watch` and `capacity`.
  The frozen tables in `tests/backends/test_driver.py` are rewritten in phases 1 and 3,
  each with a release-note fragment. `KernelProblem` does not change.
- **Kernel contract**: §11 gains the watch; a new §12 specifies the RK march.

#### 6.5 Does this satisfy `native/AGENTS.md`?

Yes, by construction: every threshold, gain, cap and limit arrives in a record solprop
forms; native evaluates comparisons and the controller's formulas on those numbers. The
one literal in `read_due` (`1e-12` in `adams_march.cpp`) is moved into the record in
phase 1.

#### 6.6 Admission and fallback

- **B** applies to every native Adams run (and, from phase 3, RK run) except one with a
  transverse edge, whose reads stay host events (SEP 44), and lockstep batches. Those
  keep returning at each check, without a warning: the algorithm is the same.
- **The scorer** (phase 2) replaces `_score_trial` when `type(controller).scale is
  PIController.scale`; otherwise the host scorer stays, unchanged. **The staged rows**
  replace `_trial_arrays` when `_stage_tables()` is available; otherwise the host
  packer stays.
- **A v1** (phase 3) admits a single native run with `integrator is None`, adaptive
  stepping, no interpolated planes, staged tables available, no coupled flow, a CPU
  build, and a controller that is a `PIController` whose `scale`, `accepts`, `propose`,
  `rejection_factor`, `start_factor` and `equivalent_norm` are all `PIController`'s
  own (`type(controller).<name> is PIController.<name>`, as the Adams engine tests
  `scale`). Being a `PIController` is not enough: a subclass that overrides one of them
  runs the host loop with its own methods. Anything outside the predicate runs
  solprop's loop with `HostMarchWarning` naming the reason. Phase 4 removes the reasons
  except lockstep batches and an overriding controller; phase 5 admits device builds.

#### 6.7 Device builds

The march loop runs on the host and drives the device through the resident engine, as
the Adams march does; the norm partials already come back through pinned memory. A
device build's gain is expected to be larger than a CPU build's (SEP 39 §6.7); phase 5
measures it on an NVIDIA A40, and on a data-centre card only with the maintainer's
approval of the cost.

#### 6.8 Cost

Memory: the watch buffer holds the call bound's 256 readings of 10 (or 138) doubles,
about 280 kB, allocated with the march and listed in the memory ledger; the RK march
adds no field-sized buffer. Python time that stays: the replay and the booking of §4.1,
measured in phase 0. Code: about 1500 lines of C++ and
bindings, 500 of Python, and the tests of §7, by comparison with SEP 39.

#### 6.9 Relation to #110

#110 proposes one march protocol for all step loops. This SEP adds a second compiled
loop but shares its call protocol, row format, landings record, events and adapter
pattern with the first, and puts the RK decisions in a pure core as Adams's are. That is
the shape #110 would unify; it neither decides nor blocks it.

### 7. Verification and validation

#### 7.1 Exact structural contracts

- The policy tuples round-trip exactly, and a missing or extra field refuses by name.
- After every drain the monitor's watch state equals the march's (§4.8).
- Every `set_initial` resets the RK state and the watch: a resumed run, two inputs on
  one runner, reuse after a failed run, and a monitored input followed by an unmonitored
  one take the host loop's steps and records (§4.2).
- The landings record reproduces the Adams march's decisions on SEP 39's trace corpus
  unchanged.

#### 7.2 Decision traces

- **RK corpus**: `tests/data/generate_rk_traces.py` writes scripted-norm traces of
  `_march_to` *before* phase 0's refactor (start-up on and off, `lag` 0–2,
  `reject_free`, landings at planes and kinks, each floor site), including the quiet,
  transverse and split caps each on its own, a cap coinciding with a clipped or a
  stretched landing, a capped first step under the start-up rule, and a rejection under
  a cap. The pure core reproduces them exactly (phase 0); the native march through a
  scripted engine reproduces them at call bounds of one attempt, seven and none
  (phase 3), as `tests/backends/test_native_march_traces.py` does for Adams.
- **Watch corpus**: monitor readings scripted around each threshold, driven through
  solprop's monitor and the native watch, whose predictions agree exactly: the
  activity threshold; the warning and stop thresholds exactly and at the doubles on
  either side, warned and unwarned, with warnings turned into errors; the quiet run;
  the launch floor; and the split cap, with the near-tie witness of §4.8, exact
  cumulative-share ties, empty buckets and a zero total. A Hypothesis test draws
  readings and policies and checks the native predictions against `_read_cap`,
  `_settle`, `due` and `_judge`.

#### 7.3 Same steps on real problems

On ten-step or short windows of `gnlse_soliton`, `gnlse_raman_shock`,
`argon_blueshift_envelope`, a split-collar capillary and a quiet-collar run: native
against `SOLPROP_ADAMS_MARCH=host` and `SOLPROP_RK_MARCH=host`, the same steps, bitwise
fields and bitwise `boundary_monitor` records on one build. The existing
`tests/driver/test_native_march_call_bound.py` (a call's bound changes no number) is
extended to the watch and the RK march, across the watch buffer's capacity return.
The drain's table (§4.10) is tested case by case: a busy check coinciding with a
statistics row, with a natural plane, with the floor and with the call bound, and a
split run with skipped steps and a nonzero removal, each against the host loop's record
and counters.

These windows must exercise a busy monitor: a launch's first ten steps need not. Phase 0
chooses windows (the case, its start through a resumed launch or a shortened guide, and
its length) whose accepted, checked and skipped counts and returns by reason it reports,
and the same windows serve §7.6.

#### 7.4 Accuracy envelopes

Phase 2 moves native RK norms at rounding level. The new scorer is compared, on
scripted partials and on real trials, with the retained host scorer and with
`numpy_ref`'s `error_norm`: nonzero `atol`, non-unit weights, both `region`s, and an
error confined to the active margin outside the band, so that a scaling or masking
mistake shared by both marches is caught. A controller subclass whose `scale` (and,
separately, each of the other five methods) gives a detectably different result runs
the host loop with its own methods. Every RK case of the test suite stays inside its
SEP 25 envelope against `numpy_ref`, and the release benchmark rows are compared within
one revision.

#### 7.5 Lifecycle

Deterministic events (faults injected at each stage with `MarchFault`, the floor at
each of its three sites, `BoundaryError`, and the first `BoundaryWarning` with warnings
turned into errors) reach Python at the same step as on the host loop, with the same
state, history, record and checkpoint. Asynchronous ones (SIGINT and SIGTERM, the stop
flag) leave a coherent state, history, monitor record and checkpoint at the final
committed step, at most one call's attempts after the request; and an exception from a
progress callback propagates as it does today (`tests/driver/test_native_march_lifecycle.py`,
extended).

#### 7.6 Benchmarks

Two measurements, kept apart:

- **Wall time**, unprofiled: `python -m benchmarks ab` (interleaved fresh processes,
  three or more rounds, an A/A control giving the spread) on an idle machine, at one
  thread and the production width, on the windows of §7.3 and complete runs of
  `argon_blueshift_envelope`, `gnlse_soliton`, `gnlse_raman_shock`, `he_rdw_averaged`,
  `argon_blueshift_averaged`, `rdw`, and the frequent-event shapes of #273. No
  configuration may be slower beyond its A/A spread.
- **The Python share**: Linux `perf` sampling the Python main thread at 1 kHz during a
  warm repeat, after one untimed repeat, as in §3.4, with the samples in the
  interpreter and NumPy's and SciPy's extension modules divided by the warm repeat's
  elapsed wall time, and the unattributed samples (libc, kernel) reported beside it.
  The command and the per-library sample counts are recorded with each figure.

Phase 0 measures $h_{\text{check}}$ and $h_{\text{row}}$ (§4.1) and the returns by reason
on these windows, and fixes the share targets from them before phase 1 starts; the
provisional targets are 5 % on `argon_blueshift_envelope` and 4 % on `gnlse_soliton`.
Recorded in `docs/dev/benchmarks.md`.

### 8. Documentation

- `docs/guide/numerics/solver.md`: the RK march runs natively, its fallbacks and the
  `rk_march` metadata; `docs/guide/numerics/boundaries.md`: checks run inside the march,
  warnings arrive at the same step.
- `docs/dev/stepper.md`, `docs/dev/backends.md` (§1 point 3 and §3.5, which today argue
  against a compiled RK loop), `docs/dev/kernel-contract.md` §11 and §12,
  `native/AGENTS.md` (the policy rule), an ADR extending ADR-060.
- `docs/dev/benchmarks.md`: the phase measurements.

### 9. Compatibility and impact

- No default changes. No output schema change.
- Phase 2 changes native RK trajectories at rounding level once (a different summation
  order for the norm); step counts may differ by a few on long runs. Recorded in the
  phase report and the release notes; release history rows compare within a revision.
- Warnings from the monitor are raised during the drain, at the same step but from a
  different stack frame.
- Progress callbacks keep their count and are delivered per call; stop requests and
  signals take effect within a call's bound (§4.11), as already for the Adams march.
- A disk resume keeps today's behaviour: the RK controller starts afresh (§4.2).
- With Q7, the split collar's crossing limit takes its total in a different
  order: in a near-tie its bucket can move one place, which changes the steps of a run
  under the split scheme at rounding level. Recorded in the release notes.

### 10. Alternatives considered

- **Checking less often while busy** (raising the cadence): changes the monitor's
  policy and its record; out of scope.
- **Returning per check but making each return cheap** (B3 alone): keeps a crossing and
  a state adoption per busy step.
- **A native RK march without B**: returns at every check; on `gnlse_soliton` that is
  about a third of the steps.
- **Moving the whole Python driver loop into C++ for both integrators at once** (#110's
  protocol, compiled): larger, and couples this work to decisions #110 has not made.

### 11. Phases

0. **Baseline, corpora and pure core.** The measurement ledger: the busy windows of
   §7.3 with their counts and returns by reason, $h_{\text{check}}$ and $h_{\text{row}}$,
   and §7.6's targets fixed from them; the RK trace corpus generated before
   refactoring; the watch corpus; `rk_core.py` extracted with the corpus reproduced
   exactly; the split limit's reference total (Q7); the strict-compilation witness
   (#135); the contract's §12 drafted. Before phase 1, #133 is fixed on `main` in its
   own pull request (Q6).
1. **B on the Adams march.** Watch policy and state, `watch.cpp`, `Reason::watch` and
   `Reason::capacity`, the drain of §4.10 with `_account_step` and `_check_at`. Exit:
   §7.2 watch corpus; §7.3 bitwise records and steps, and the drain's cases; no return
   for a busy monitor's checks; §7.6 targets on `argon_blueshift_envelope`; #273's rows
   recorded.
2. **Native packer and scorer in the Python loop.** `stage::rows`, the scorer. Exit:
   rows bitwise equal `_trial_arrays`; §7.4 envelopes; the rounding change recorded.
3. **Native RK march v1.** Exit: §7.2 RK corpus at bounds 1, 7 and none; §7.3 same
   steps; §7.5 lifecycle; §7.6 targets on `gnlse_soliton`.
4. **Coverage.** Interpolated planes (the march returns at the crossing step; the
   pre-step state is read after `finish` swaps the buffers), replay, z tables, moving
   maps, coupled flows (the flow's host guards transcribed with their constants in the
   record). Exit: the same tests on each, the fallback list reduced to §6.6's.
5. **Device builds.** `pytest -m gpu` and `benchmarks ab --device` on an NVIDIA A40;
   other devices with approval.

### 12. Open questions

All seven were decided by John Travers on 2026-11-07, under his general rule that
changes at rounding level, and step choices differing by a few steps, are acceptable.

- **Q1.** May phase 2 move native RK numbers at rounding level (§9), by the order of the
  norm's sums alone, or must the scorer reproduce NumPy's pairwise order so that nothing
  moves? *Decided:* allowed. The retained host scorer and `numpy_ref` check the formula
  (§7.4).
- **Q2.** The RK march's call bound: the same 256 attempts or 0.1 s as Adams?
  *Decided:* yes, one bound for both, in the record.
- **Q3.** Widen `HostMarchWarning` or add a sibling for the RK march? *Decided:* widen
  it; one category to filter, with explicit admission reasons.
- **Q4.** A drain whose watch state disagrees with the march's: `BackendError`, or a
  fallback to per-check returns? *Decided:* `BackendError`; a mismatch is a bug, and the
  run does not continue with an unverified cap.
- **Q5.** Does replay of a fixed schedule come in v1 or phase 4? *Decided:* phase 4;
  until then a replayed run takes the host loop with `HostMarchWarning`, and that
  refusal and host path are tested from phase 3.
- **Q6.** Should #133's double count be fixed first, so that this SEP's tests pin the
  corrected record? *Decided:* yes. #133 is fixed in its own pull request before
  phase 1 (§3.6).
- **Q7.** May phase 0 change `SplitCrossing.limit` to take its total as the last
  element of its cumulative sum (§4.8), so that native reproduces the cap exactly, at
  the cost of moving a near-tie's bucket by one place in the reference? *Decided:* yes;
  the split scheme is inside B.

### 13. References

1. J. R. Dormand and P. J. Prince, "A family of embedded Runge–Kutta formulae,"
   *Journal of Computational and Applied Mathematics* **6**, 19–26 (1980).
   [doi:10.1016/0771-050X(80)90013-3](https://doi.org/10.1016/0771-050X(80)90013-3)
2. J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for simulating
   supercontinuum generation in optical fibers," *Journal of Lightwave Technology* **25**,
   3770–3775 (2007). [doi:10.1109/JLT.2007.909373](https://doi.org/10.1109/JLT.2007.909373)
3. G. Söderlind and L. Wang, "Adaptive time-stepping and computational stability,"
   *Journal of Computational and Applied Mathematics* **185**, 225–243 (2006).
   [doi:10.1016/j.cam.2005.03.008](https://doi.org/10.1016/j.cam.2005.03.008)
4. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd ed., Springer (1993), §II.4.

## Part II — Implementation record

### Phase 0 report — baseline, corpora and the pure core

- **Scope:** phase 0, 2026-11-07, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0045-native-rk-march` (LupoLab/solprop#567), commits `f84bb72b`,
  `853245ed`, `801f0def`, `b7ab4194` and `76b358d5`, on top of #563 and #566.
- **Delivered:**
  - `tests/backends/rk_traces.py`, `tests/data/generate_rk_traces.py`,
    `tests/data/rk_traces.json.gz` and `tests/backends/test_rk_traces.py`: the
    Runge–Kutta decision corpus, 29 scenarios generated from the march before the
    refactor (`a0d96912`), and the test that replays it.
  - `src/solprop/backends/rk_core.py`: `RungeKuttaPolicy`, `rk_policy`, `decide`,
    `start_up`, `propose` and `reject`; `StepDriver._march_to` and `_start_up` call them
    through `StepDriver._controller_policy`.
  - `src/solprop/boundaries.py`: `SplitCrossing.limit` takes its total from its
    cumulative sum (Q7), with the review's near-tie witness in
    `tests/test_split_crossing.py` and a release-note fragment.
  - `native/src/adams_march.cpp`: `#pragma GCC optimize("fp-contract=off")` (#135), and
    `AdamsMarch.TheDecisionsStayStrictWhereTheBuildFuses` in
    `native/tests/native/test_adams_march.cpp`.
  - `docs/dev/kernel-contract.md` §12, drafted.
- **Deviations from the proposal:**
  - *The scripted norms model the trial.* A scenario's trial returns
    `noise * (dz / h0) ** (q + 1)`: independent random norms give the controller no
    equilibrium, so its step drifted to the floor within a few millimetres and never
    reached a plane or a cap. Scenarios that need a rejection or a floor at a chosen
    attempt script their norms directly. A native scripted engine computes the same
    with `pow` on the same build.
  - *The watch corpus moves to phase 1*, beside the native watch it checks; its
    readings and thresholds are those §7.2 lists.
  - *#135 is fixed in the source as well as witnessed.* The pragma makes GCC treat the
    file's functions as uninlinable into a contracting caller, which is what the issue
    found missing under link-time optimisation with hidden symbols. The new strict files
    of phases 1 and 3 take the same pragma and witness.
- **Decisions:** the policy hook `StepDriver._controller_policy` (so that the scripted
  harness, which has no `StepperSettings`, passes the runtime type checks);
  `rk_policy` reads the controller, floor, cap and start-up flag, and the landings stay
  with the march's targets as before.
- **Verification:**
  - The corpus replays exactly against the refactored march; it reaches the three
    floor sites, natural stops, every cap alone and meeting landings, and completes
    the default, landing and replay scenarios.
  - On an AMD EPYC 7763 node (Python-only changes from the tree's `src/` over the
    branch's native build): `test_rk_traces`, `test_adams_traces`, `test_driver`,
    `test_split_crossing`, `test_monitor`, `test_dense_output`, `test_native_error_norm`,
    the native march's call-bound, lifecycle and trace tests and `tests/stepper` pass
    (713 tests; 14 setup errors in a first run came from moving its checkout during the
    run and did not recur).
  - The native unit tests in a Release `znver3` build: the #135 witness runs, not
    skipped, its control fuses and the march's probe does not.
- **The ledger** (AMD EPYC 7763, one thread, a shared node at load 80, a warm repeat
  with `perf_counter` wrappers on the monitor and the march's host functions;
  per-call costs, not walls):

  | window | attempts | checked / skipped | native calls (reason) | µs per `monitor.step` | µs per reading | Python per call less its native run (µs) |
  |---|---:|---:|---|---:|---:|---:|
  | `argon_blueshift_envelope` (0.1) | 44 | 13 / 31 | 22 (11 monitor, 11 reached) | 27 | 27 | 98 |
  | `argon_blueshift_envelope` (0.3) | 136 | 56 / 80 | 62 (51 monitor, 11 reached) | 30 | 31 | 113 |
  | `argon_blueshift_envelope` | 2227 | 1906 / 321 | 1910 (1828 host step, 71 monitor, 11 reached) | 26 | 27 | 103 |
  | `he_rdw_averaged` | 1437 | 233 / 1201 | 243 (230 monitor) | 35 | 89 | 208 |
  | `argon_blueshift_averaged` (0.3) | 3191 | 393 / 2796 | 403 (392 host step) | 29 | 50 | 162 |
  | `h2_raman_averaged` (0.2) | 2501 | 307 / 2192 | 316 (305 host step) | 33 | 77 | 204 |
  | `gnlse_soliton` (RK) | 691 | 207 / 484 | — | 30 | 38 | — |
  | `gnlse_raman_shock` (0.1, RK) | 1048 | 127 / 921 | — | 90 | 307 | — |

  A row's booking costs about 1 µs (`_record_attempt`) and 2–3 µs (`_skip_step`, of
  which `monitor.skip` is 1.3 µs). The envelope case is busy only after about a third of
  its guide, so windows from the entrance do not exercise a busy monitor; §7.3's busy
  cases are built for it, and §7.6 times the complete envelope run, about 1.4 s.
- **Targets fixed** (§7.6): B keeps $h_{\text{check}} \approx 30$ µs and
  $h_{\text{row}} \approx 3$ µs, which projects about 4–5 % Python on the complete
  `argon_blueshift_envelope` run after phase 1, so its target stays 5 %; the replayed
  `monitor.step` is then the largest Python cost, and the next lever if the target is
  missed. `gnlse_soliton` projects about 2 % after phase 3, inside its 4 %.
- **Limitations:** the ledger's node was shared; the ratios phases 1 and 3 report will
  come from `benchmarks ab` with an A/A control on an idle node.
- **Handoff to phase 1:** the native watch (`Reason::watch` and `Reason::capacity`, its
  policy and state, the reading through the engine), the drain with `_account_step` and
  `_check_at`, the watch corpus, and the frozen driver surface.

### Phase 1 report — the boundary checks inside the native Adams march

- **Scope:** phase 1, 2026-11-07, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0045-native-rk-march` (LupoLab/solprop#567), commit `6d1855c6`.
- **Delivered:**
  - `native/include/solprop_native/watch.hpp`, `native/src/watch.cpp` (strict, with
    the #135 pragma) and `native/tests/native/test_watch.cpp`: the predictions and
    their unit tests, the #135 witness among them.
  - `native/include/solprop_native/adams_march.hpp`, `native/src/adams_march.cpp`:
    `Engine::reading`, `Reason::watch` and `Reason::capacity`, the watch in the
    bounds, state and result, the check after each accepted step, the watch's cap
    under the split scheme.
  - `native/src/bindings.cpp`: the watch's converters, the readings in the result,
    the resident engine's reading, and `watch_check`, `watch_due` and
    `watch_split_limit`.
  - `native/src/solprop_native/__init__.py`: `WATCH_POLICY_FIELDS`,
    `WATCH_STATE_FIELDS`, and `DRIVER_SURFACE` with `_account_step`,
    `_statistics_step` and `_watched_step` for `_skip_step`.
  - `native/src/solprop_native/backend.py`: `_NativeAdamsMarch._watching`,
    `_watch_policy`, `_readings`, `_verify_watch`, and the drain of §4.10.
  - `src/solprop/monitor.py`: `watch_policy`, `watch_state`.
  - `src/solprop/backends/driver.py`: `_check_at`, `_monitor_step`, `_watched_step`,
    `_statistics_step`, `_account_step`; `_refuse_scheduled_crossing` takes the row's
    cursor and position.
  - Tests: `tests/backends/test_monitor_watch.py` (the watch corpus: Hypothesis draws
    and the thresholds' neighbours against a real monitor),
    `tests/driver/test_native_march_watch.py` (busy runs against the Python march);
    the frozen surface in `tests/backends/test_driver.py`, the trace harness's hook,
    and the floating-point licence's strict files.
  - Documents: kernel contract §11, `docs/dev/stepper.md`, `docs/dev/backends.md`,
    `docs/guide/numerics/boundaries.md`, and the release-note fragment.
- **Deviations from the proposal:**
  - *The watch policy travels with each call's bounds*, not with the march's
    configuration: it depends on the launch's floor and energy and is a few hundred
    numbers, so sending it per call costs nothing measurable and keeps
    reconfiguration out of the adapter.
  - *A buffer's capacity ends a call only where no host event does.* Checked before
    the host step at first, it preempted a statistics row when the call's bound was
    one attempt, which `test_native_march_call_bound.py` caught.
  - *The watch corpus is a test module*, not a data file: the native predictions are
    compared with a live monitor on drawn readings, which needs no recorded oracle.
- **Decisions:** the verdicts are `none`, `stop`, `warn`, `quiet`, `refuse` and
  `error`, the first that applies, `stop` before the warning latch; a step's reading
  happens after its replay cursor advances, so a scheduled crossing is read with the
  row's own cursor.
- **Verification:**
  - Native unit tests in a Release `znver3` build on an AMD EPYC 7763: all pass, the
    watch's and the march's #135 witnesses run, not skipped.
  - The watch corpus and the busy runs pass: on a busy monitor (the 0.65-window
    walker at the default budget), in both formulations and both collar schemes,
    the native march's steps, fields, monitor record and warnings equal the Python
    march's bit for bit, while it returns at under a quarter as many points as it
    checks.
  - `tests/backends` and `tests/driver` with the monitor, split-crossing and
    transverse modules on an AMD EPYC 7763 node: 6189 passed, 20 skipped, 1 xfailed,
    and two failures: the floating-point licence's pinned CMake line (fixed in this
    commit) and `test_dense_planes_interpolate_through_the_table[runge_kutta]`, whose
    envelope (1.17e-9 against 1.03e-9) also fails on a tree of `main` without SEP 45
    and passes on some runs: a marginal bound on this hardware, not this phase.
- **Benchmarks** (`benchmarks ab`, an idle AMD EPYC 7763 node, exclusive, complete
  runs, three rounds, phase 0's head against this phase's; ratio below one is
  faster; the A/A control's ratios lay within 0.996–1.007):

  | case | 1 thread | 16 threads |
  |---|---:|---:|
  | `argon_blueshift_envelope` | 0.881 | 0.837 |
  | `argon_blueshift_averaged` | 0.930 | 0.910 |
  | `h2_raman_averaged` | 0.935 | 0.922 |
  | `he_rdw_averaged` | 0.979 | 0.968 |
  | `gnlse_soliton` (Runge–Kutta, unchanged) | 1.002 | 1.009 |

  Every configuration took the same steps; the fields differ by up to 1e-12, as
  the A/A control's do, from FFTW's measured plans. The averaged capillaries gain
  because their monitor's every-eighth-step checks no longer return either.
- **The Python share** (the phase 0 ledger's method, one thread): on the complete
  `argon_blueshift_envelope` run the native calls fall from 1910 to 17 (11 planes,
  6 call bounds) and the Python time outside the native calls from 14.5 % to 3.3 % of
  the run, of which the replayed `monitor.step` is 2.1 % (11 µs a check on the idle
  node); `argon_blueshift_averaged` (0.3) 1.8 %, `he_rdw_averaged` 0.9 %. The 5 %
  target is met.
- **Limitations:** a transverse edge keeps returning per check (SEP 44), as do
  lockstep batches, which do not take the native march.
- **Handoff to phase 2:** the staged rows for the Runge–Kutta trial and the shared
  scorer; the driver's `_monitor_step` and `_statistics_step` are the pieces the
  native RK march's drain will reuse.

### Phase 2 report — the staged Runge–Kutta rows and the native norm

- **Scope:** phase 2, 2026-11-07, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0045-native-rk-march` (LupoLab/solprop#567), commit `2bdbb672`.
- **Delivered:**
  - `native/include/solprop_native/stage_tables.hpp`: `stage::rows`, a trial's stage
    rows at `z + c_i dz` and the whole step from the uploaded tables.
  - `native/include/solprop_native/rk_march.hpp`, `native/src/rk_march.cpp` (strict,
    with the #135 pragma): `rk::positions`, `rk::score` (each row summed in order, the
    weak scale `residual / (rtol * max(old, new) + atol)`), and `strict_probe`.
  - `native/src/bindings.cpp`: `State.rk_trial(z, dz, nodes, prepare_only)` and the
    module's `rk_score`; stubs in `_core.pyi`.
  - `native/src/solprop_native/backend.py`: `_staged_rows()`, decided on first use,
    which uploads the stage tables for a Runge–Kutta run with no coupled flow and no z
    table; `_prepare_trial` takes `rk_trial` there; `_score_trial` takes `rk_score`
    for `PIController`'s own scale with no weights. `SOLPROP_RK_ROWS=host` is the
    private comparison switch.
  - Tests: `tests/backends/test_native_rk_rows.py`, `native/tests/native/test_rk_march.cpp`.
- **Deviations from the proposal:** the host's `_last_bundle` is not set on a staged
  trial. Native never read it (its quadrature alarm comes from the native state), so
  `test_the_alarm_is_weighted_at_the_stage_it_was_scored_from` now skips native and
  says why.
- **Verification:** a staged trial's partials equal the host-packed trial's bit for bit on
  ten routes, at four anchors each, a breakpoint among them. The native norm equals the
  host's NumPy reduction exactly below NumPy's pairwise blocking (8 slots) and to
  `8 n eps` above it. A controller with its own `scale`, or with weights, keeps the
  host's reduction. Phase 3's benchmarks include this phase: the complete
  `gnlse_soliton` and `gnlse_raman_shock` runs took the same steps as phase 0's head,
  with their fields differing by under 1e-12.
- **Handoff to phase 3:** `BoundState::rk_trial` and `rk::score` are the native
  march's trial.

### Phase 3 report — the native Runge–Kutta march

- **Scope:** phase 3, 2026-11-07, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0045-native-rk-march` (LupoLab/solprop#567), commit `e8ca0a9b`.
- **Delivered:**
  - `native/include/solprop_native/rk_march.hpp`, `native/src/rk_march.cpp`: `rk::Policy`,
    `rk::State` (the position, the proposal, the accepted norms and the watch's state),
    `rk::Engine` (`trial`, `accept`, `reading`), `rk::Result` with the floor's site, the
    controller (`equivalent`, `accepts`, `propose`, `rejection_factor`, `start_factor`,
    `limited`) and `rk::run`, the loop of §4.3 with the boundary watch of phase 1.
  - `native/src/bindings.cpp`: the policy, state and result converters; the resident
    engine (the staged trial, `rk::score`, `promote_fsal` then `finish`, the monitor's
    reading); `State.rk_march_configure` and `State.rk_march_run`, which releases the
    GIL; `ScriptedRkMarch`, the scripted engine the corpus drives.
  - `native/src/solprop_native/backend.py`: `_NativeRungeKuttaMarch` (the policy
    record, the bounds, the drain shared in substance with the Adams adapter's, the
    floor's message) and its admission, `NativeBackend._rk_march`, the run's
    metadata `rk_march`, and `SOLPROP_RK_MARCH=host`. The watch helpers are now
    module functions both adapters call.
  - `src/solprop/exceptions.py`: `HostMarchWarning` names both marches (Q3).
  - Tests: `tests/backends/test_native_rk_traces.py` (the corpus through the adapter
    on the scripted native march, at call bounds of 1, 7 and 256 attempts),
    `tests/driver/test_native_rk_march.py` (native against host, bit for bit, on six
    routes and on a busy monitor in both collar schemes),
    `tests/backends/test_native_rk_admission.py`, and the controller's branches in
    `native/tests/native/test_rk_march.cpp`.
  - Docs: kernel contract §12 normative; ADR-069; `backends.md` §§1 and 3.5,
    `stepper.md`, the solver guide and `native/AGENTS.md`; the changelog fragment.
- **Deviations from the proposal:**
  - *The landings reuse `march::Policy`* rather than a factored `march::Landings`: the
    Runge–Kutta policy holds one, with `interpolate` false. Factoring it out would
    have touched every Adams site for no behavioural gain; phase 4 can still do it.
  - *The corpus's replay and transverse scenarios* (`replay`, `replay_diverged`,
    `transverse_cap`, `caps_at_landings`) are phase 4's and are left to the host
    loop's own test.
  - *A test-wide filter* (`pyproject.toml`) ignores the interim `HostMarchWarning`
    of the Runge–Kutta runs phase 4 covers, so that the many native Runge–Kutta tests
    on those routes keep running unchanged. The admission tests still assert the
    warning, and the filter goes with phase 4.
  - Tests that patch the host loop's trial now pin it with the private switch
    (`test_source_evaluations.py`). On native, the one-plane stop test runs the
    Runge–Kutta march for the Adams marches' length and allowance
    (`test_stop.py`), since it now polls once per call.
- **Decisions:** the march is refused, with the reason named, for a device build,
  interpolated planes, a controller overriding any of `scale`, `accepts`, `propose`,
  `rejection_factor`, `start_factor` or `equivalent_norm`, a weighted norm, a
  transverse edge, or rows the host forms (a coupled flow, a z table). The floor
  raises the reference's `limit_step` message.
- **Verification:**
  - Native unit tests in a Release `znver3` build on an AMD EPYC 7763: all 13 of the
    Runge–Kutta, watch and Adams march tests pass.
  - The corpus: all 25 scenarios the march takes, at each of the three bounds,
    including every floor site.
  - Native against host on one build: equal steps, norms, acceptances, final fields
    and monitor records on the GNLSE, four capillary routes and the constant general
    guide, and on the busy-monitor walker in both collar schemes.
  - `tests/backends`, `tests/driver`, the monitor, split-crossing, transverse, stepper,
    spatial-flux and pulse modules on an AMD EPYC 7763 node: 6790 passed. The 40
    failures, rerun with this phase's fixes, all pass except
    `test_dense_planes_interpolate_through_the_table[runge_kutta]`, the marginal
    envelope on this hardware recorded in phase 1. Of the 40, 35 were the interim
    warning before its filter, and 4 were the tests above.
- **Benchmarks** (`benchmarks ab`, an idle AMD EPYC 7763 node, exclusive, complete
  runs, three rounds, phase 0's head against this phase's, so phases 1–3 together;
  a ratio below one is faster):

  | case | 1 thread | 16 threads |
  |---|---:|---:|
  | `gnlse_soliton` (Runge–Kutta) | 0.784 | 0.787 |
  | `gnlse_raman_shock` (Runge–Kutta) | 0.962 | 0.938 |
  | `argon_blueshift_envelope` (Adams, phase 1) | 0.885 | 0.838 |

  Every configuration took the same steps (691, 8469 and 2227); the fields differ by
  up to 1.1e-12.
- **The Python share** (the phase 0 ledger's method, one thread, the same node):
  `gnlse_soliton`'s 691 steps take 11 native calls, one a plane, and the time outside
  them is 4.2 % of the run (0.403 s). Of that, the drain's per-step accounting is
  0.9 %, the replayed monitor checks 0.7 % (207 checks at 13 µs), and most of the rest
  is the 11 planes' observation, part of which is native work outside the march's
  call; perf would separate it. That leaves the 4 % target met for Python's own share,
  or at worst within 0.2 % of it. `gnlse_raman_shock` 0.24 % (373 calls, 362 at the call
  bound); `argon_blueshift_envelope` 4.7 % in this instrumented run, which wraps more
  functions than phase 1's 3.3 % measurement did.
- **Limitations:** see the refusals above; lockstep batches keep their own loop.
- **Handoff to phase 4:** interpolated planes, replay, z tables, moving maps, coupled
  flows and the transverse edge; then remove the test-wide warning filter.

### Phase 4 report — coverage

- **Scope:** phase 4, 2026-11-07, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0045-native-rk-march` (LupoLab/solprop#567), commit `972ecb93`.
- **Delivered**, each with its tests in `tests/driver/test_native_rk_march.py` (native
  against host, bit for bit) and, where it moved a decision, the corpus:
  - *A coupled flow.* `stage::rows` carries the anchor's section and each stage's residual
    weights; `BoundState::guarded_flow_actions` forms a step's actions for one (section,
    step) behind `_flow_squarings` and `_check_flow_conditioning`, transcribed, with the
    Taylor scale, the squarings admitted, the row sums, the certificate and the rounding
    allowance handed over (`flow_guard_configure`); `native::BackendRefusal` and
    `Category::refused` carry a refusal back as `BackendError`. Each loop resets the other's
    key, so the one set of actions stays coherent. Tested on the moving guide, the coherent
    wall and the taper coupling, with each guard's refusal held to the reference's message.
  - *Interpolated planes.* The landings step across the save planes to the run's end, and
    the march returns the accepted step that crosses its target unfinished
    (`Reason::dense`, with its equivalent norm); `_finish_crossing` completes it with the
    reference's acceptance, so `_dense_record` reads the state before the step.
  - *A z table.* The state loads the stages' positions into the one slot cache, stage by
    stage: the slots differ from the host's, the values in them do not, and the trial's
    partials are the host-packed trial's bit for bit on the walk-off fibre.
  - *The transverse edge.* Its ceiling caps the candidate between the quiet interval and
    the split cap, the call returns before a step would carry past the next read, and
    solprop takes the monitor's checks by its distance test, as for the Adams march. The
    corpus's `transverse_cap` and `caps_at_landings` now run natively.
  - *Replay.* `StepDriver._replay_to` in the march, from the schedule's cursor, with the
    replay's crossing refusal (`_refuse_scheduled_crossing`) as the Adams march takes it;
    a non-finite norm returns `diverged`. The corpus's `replay` and `replay_diverged` run
    natively.
  - *A weighted norm.* `rk::weighted_scale` and `rk::weighted_score`, called by both loops
    through `BoundState::rk_weighted_score`: weighted runs move once at rounding level, as
    the weak norm did in phase 2 (Q1).
  - The interim warning filter is gone (`pyproject.toml`); the native package's surface
    test lists the watch tuples; the CI skip check allows the alarm test's native skip.
- **Deviations from the proposal:** the weighted norm was not in phase 4's list in §11,
  but §6.6 leaves only lockstep batches and an overriding controller on the host by Final,
  so it is covered here. Tests that probe solprop's loop by patching its hooks
  (`_on_accepted_step`, `_stage_bundles`, `_score_trial`) now pin that loop with the private
  switches, as `test_source_evaluations.py` does: the recycled-stage comparison, the frozen
  moving maps, the invalid-norm batch, and a test that simulates a device build. The native
  replay's stop flag is polled between calls, so `test_stop.py` allows the replay to run to
  its next saved plane on native.
- **Decisions:** what still steps the reference, with a `HostMarchWarning`: a device build
  (phase 5) and a controller overriding any of its six methods. Lockstep batches keep their
  own loop.
- **Verification:**
  - Native unit tests in a Release `znver3` build on an AMD EPYC 7763: the 13 Runge–Kutta,
    watch and Adams march tests pass.
  - The corpus: all 29 scenarios at bounds 1, 7 and 256.
  - Native against host, bit for bit in steps, norms, fields and the monitor's record: ten
    routes (the coupled flows and the spatial routes with their edge among them), three with
    interpolated planes, the z table landed and interpolated, two replays, a weighted
    norm, and the busy monitor in both collar schemes.
  - The whole suite on an AMD EPYC 7763 node: 17976 passed before the fixes above, with 21
    failures, every one of which passes in a rerun of their files; CI runs the full gate.
- **Benchmarks** (`benchmarks ab`, an idle AMD EPYC 7763 node, exclusive, complete runs,
  three rounds, phase 3's head against this phase's):

  | case | 1 thread | 16 threads |
  |---|---:|---:|
  | `dense_output` (interpolated planes) | 0.880 | 0.879 |
  | `moving_solid_guide` (coupled flow) | 1.007 | 0.992 |
  | `gnlse_soliton` (control) | 1.005 | 1.000 |

  The same steps in each; the fields differ by up to 2.6e-14. `moving_solid_guide`'s 262
  steps take 0.7 s each, so its host share was already negligible.
- **Limitations:** a device build (phase 5); an overriding controller; lockstep batches.
- **Handoff to phase 5:** `pytest -m gpu` and `benchmarks ab --device` on an NVIDIA A40, then
  admit device builds.

## Part III — History

- 2026-11-07: Drafted (John Travers, with Opus 5.5), from a profile of warm runs on the
  tracking set (§3.4).
- 2026-11-07: Tracking issue [#562](https://github.com/LupoLab/solprop/issues/562) opened.
- 2026-11-07: GPT-6 completed the AI review below; Part I is unchanged. The findings
  await the drafting agent's response and the maintainer's decisions.
- 2026-11-07: Response to the AI review (Opus 5.5): R1–R9 accepted; Part I revised in
  §§1, 4.1–4.4, 4.8, 4.10 (new), 4.11 (new), 5, 6.1, 6.3, 6.4, 6.6, 6.8, 7, 9, 11 and
  12, with Q7 added. The maintainer's decisions on Q1–Q7 are still to come.
- 2026-11-07: John Travers decided Q1–Q7 (§12): rounding-level changes and step choices
  differing by a few are acceptable (Q1, Q7); the Adams call bound for both marches
  (Q2); one widened `HostMarchWarning` (Q3); `BackendError` on a watch mismatch (Q4);
  replay in phase 4 (Q5); #133 fixed first (Q6). Recorded in §§3.6, 4.10, 11 and 12.
- 2026-11-07: John Travers reviewed Part I (human review). Accepted by John Travers;
  Part I is frozen from here, and deviations go in the phase reports.
- 2026-11-07: Implementation started (Opus 5.5 for John Travers) on
  `feature/sep-0045-native-rk-march`.
- 2026-11-07: Phase 0 reported (Part II).
- 2026-11-07: Phase 1 reported (Part II).
- 2026-11-07: Phases 2 and 3 reported (Part II).
- 2026-11-07: John Travers approved the phase 3 choices (the interim warning filter, the
  three adapted tests, the landings reusing `march::Policy`) and ADR-069 as worded.
- 2026-11-07: Phase 4 reported (Part II).

### AI review — 2026-11-07

Reviewer: **GPT-6 (AI)**. Reviewed all of Part I at PR #563 commit `f3efb5ce2b`,
against SEP 0, the conventions, the current driver, controller, monitor, native
adapter and march, the kernel contract, SEP 39 and SEP 44, and the PR discussion.
These are review comments and questions, not revisions to the proposal or acceptance
decisions. All findings below remain open under
[#562](https://github.com/LupoLab/solprop/issues/562). R2 and R3 confirm and incorporate
the two existing automated PR findings so that the response can address them here.

The separation into a native watch, a shared scorer and a native RK march is a
reasonable approach. Keeping Python's monitor as the owner of its record is useful.
However, the algorithm as written changes the norm and controller decisions, and the
buffered drain needs a more complete lifecycle contract. Recommend resolving the
following before acceptance; this does not prevent merging a Draft for discussion.

1. **R1 — P1: preserve the Euclidean norm and its selected bins (§§4.2, 4.4,
   6.3, 7.4, 9; phase 2).** The numerator in §4.4 is
   `sqrt(sum(abs(error)**2) / n)`, whereas both
   [`NativeBackend._score_trial`](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/native/src/solprop_native/backend.py#L1085)
   and `numpy_ref.passes.error_norm` use the Euclidean norm without `/ n`.
   With 16 unit state entries, error entries of 2, `rtol=1` and `atol=0`, a probe
   of the NumPy oracle gives 2 (reject); the written formula gives 0.5 (accept).
   This is a factor-of-four change, not the rounding change Q1 asks permission for.
   Also, the reductions use `problem.grids.norm_mask`, selected by
   `PIController.region` (default `"active"`), rather than always the in-band
   mask named here. Please remove the extra normalisation, explicitly retain the
   selected mask, and write the weighted formula with division by each bin's
   scale before the Euclidean reduction. Include nonzero `atol`, non-unit weights,
   both regions and an error confined to the active margin in the scorer tests.
   Compare the new scorer to the existing scorer and the NumPy oracle: the two
   marches sharing a new scorer cannot detect a common scaling mistake.

2. **R2 — P1: keep the controller proposal separate from the capped trial
   length (§§4.3, 4.5, 7.2).** Step 1 overwrites `p` with the capped value, then
   steps 4, 6 and 7 use it for the equivalent norm and the next proposal. The
   [current loop](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/src/solprop/backends/driver.py#L1569)
   caps only the argument to `landing`; its controller retains `_proposal`.
   For proposal 0.01 m, cap 0.005 m, executed norm 0.5 and estimator order 4,
   `equivalent_norm` gives 16 from the retained proposal, versus 0.5 from §4.3's
   capped value. This changes start-up, later growth, rejection and floor behaviour.
   Please name a separate landing candidate and retain the original proposal
   throughout the controller update, including the first accepted capped step.
   Add independent quiet, transverse and split cap traces, caps coinciding with a
   clipped or stretched landing, and rejection under a cap. This confirms
   [the existing PR finding](https://github.com/LupoLab/solprop/pull/563#discussion_r4501591694).

3. **R3 — P1: compare guard amplitudes with amplitude thresholds (§4.8,
   §§7.2, 7.5).** The native watch is told to compare a guard *share* against
   `budget` and `stop_factor * budget`; the
   [reference judge](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/src/solprop/monitor.py#L923)
   compares `sqrt(max(shares[0], shares[1], 0.0))`. A direct monitor probe with
   total 1 and guard energy `1e-5` warns at the default budget, but comparing
   the share to `1e-3` does not return to Python. A missed stop would likewise
   let native advance past the state whose reading raises during the drain.
   Please specify the reference square root and strict comparisons, with the stop
   tested before the first-warning latch. Squaring the thresholds is not
   automatically bitwise equivalent at neighbouring floating-point values.
   Test the exact thresholds and their adjacent doubles, warned and unwarned
   states, and warnings promoted to errors. This confirms
   [the existing PR finding](https://github.com/LupoLab/solprop/pull/563#discussion_r4501591697).

4. **R4 — P2: specify the split crossing reduction, not only its constants
   (§§4.8, 4.9, 7.1–7.3).**
   [`SplitCrossing.limit`](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/src/solprop/boundaries.py#L1119)
   uses a NumPy sum for the total, a separate cumulative sum and
   `searchsorted(..., side="right")` at `share * total`. Disabling contraction
   does not make a sequential C++ sum equal that total. A probe with 128 positive
   occupancies, bucket 0 equal to `float.fromhex("0x1.0c6f8ba2f9813p-20")`,
   bucket 1 equal to 1 and every other bucket equal to `2**-54`, at the default
   `share=1e-6`, selects bucket 1 with NumPy's total and bucket 0 with a sequential
   total. The selected step limits differ by a factor of about 1.176. The exact
   watch-state comparison would then fail, or the march would have already used
   a different cap. Please choose how native reproduces this decision: preserve
   the reference reduction order, or return to Python before using an ambiguous
   cap. Changing the reference decision needs an explicit compatibility decision;
   Q1 currently permits only the RK norm's rounding change. Add this witness,
   exact cumulative-share ties, empty buckets and zero total to the watch corpus.

5. **R5 — P1: specify a complete drain transaction and coincident-event order
   (§§4.3, 4.8, 6.4, 7.5; phases 1 and 3).** The proposed row walk accounts an
   unchecked accepted step with `_skip_step` and a checked one with `monitor.step`.
   Those are not interchangeable bookkeeping operations:
   [`_skip_step`](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/src/solprop/backends/driver.py#L2398)
   also increments the periodic-statistics and natural-plane counters;
   `monitor.step` does neither. Conversely, calling the current `_take_step_stats`
   after replaying the final checked row can check it again, particularly while
   busy. Its split reading also drains native's removal accumulator. The blanket
   instruction to check after every commit conflicts with the accepted floor
   exits in §4.3, which occur before any step statistics or monitor work.
   Please give the drain a table of effects for checked and unchecked rows,
   rejected attempts, all three RK floor sites, watch exits, host events and
   native faults. Count every eligible accepted step once in *each* counter;
   consume each buffered reading once; run statistics and natural stops in the
   reference order without another check; use the row's own replay cursor when
   testing a scheduled crossing. Keep #133's existing plane behaviour separately
   identified. State which work is inside `committing()` and how complete history,
   monitor records and terminal checkpoints are retained if a warning becomes an
   exception, a callback fails or a signal arrives during the drain. Test a busy
   check coinciding with a statistics row, a natural plane, a floor and a call
   bound, plus a split run with skipped steps and nonzero removal.

6. **R6 — P2: distinguish call continuation from disk resume (§4.2, §9;
   phases 3–4).** The statement that resume restores the same five RK state
   components does not describe the current implementation.
   [`StepDriver.set_initial`](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/src/solprop/backends/driver.py#L1254)
   resets the proposal to `dz_init`, clears accepted norms and restarts the
   statistics/natural-plane counters, including on resume; `march_snapshot()`
   returns `None` for RK. `_start_up` explicitly documents starting afresh after
   resume. Preserving these values across native calls is necessary, but
   preserving them across disk resume would add behaviour and require a storage
   design absent from the no-schema-change proposal. Which meaning is intended?
   Recommend retaining today's disk-resume behaviour and saying so explicitly.
   Specify reset of the RK state and watch on every `set_initial`, reconfiguration
   after attaching the resumed monitor, and invalidation of FSAL/dense caches.
   Test a resumed run and two inputs on one runner, including reuse after failure
   and changing from a monitored to an unmonitored input.

7. **R7 — P2: make the interruption and callback promises achievable (§§5,
   7.5, 9; Q2).** §7.5 promises that signals and stop flags reach Python at the
   same step as the host loop, while §§5 and 9 defer them until a native call
   returns. The
   [existing Adams adapter](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/native/src/solprop_native/backend.py#L2649)
   explicitly permits further accepted steps within that call. Its clock limit
   is checked after an attempt, so 0.1 s is not an absolute latency ceiling: an
   attempt and the subsequent drain/callback work can exceed it. Also, the
   adapter invokes `progress.step` once per recorded attempt during the drain,
   not once per call as §5 could imply. Please distinguish exact positions for
   deterministic boundary/floor failures from call-bound latency for asynchronous
   requests, and distinguish callback count from delivery time. Test coherent
   state/history/checkpoint at the final committed position and a bounded number
   of additional attempts, rather than same-step equality for a wall-clock
   signal. Do callbacks keep their existing per-attempt delivery and exception
   semantics? Recommend yes, with the deferred timing documented.

8. **R8 — P2: make fallback survive phase 2 and controller overrides (§§6.3,
   6.6, 7; phases 2–4).** Moving the scorer into the Python loop also changes
   the path intended to be the fallback. Today `_score_trial` calls the actual
   `controller.scale`; the
   [Adams engine](https://github.com/LupoLab/solprop/blob/f3efb5ce2b4700d5f8382f505e7a6857415cadcb/native/src/solprop_native/backend.py#L2075)
   checks that a `PIController` subclass has not overridden that method before
   selecting native scaling. Merely being a `PIController` does not establish
   that its scale, acceptance, proposal, rejection or start-up methods are the
   formulas in §4. Please state how custom methods retain the current host
   implementation, including the phase-2 scorer, and how unavailable stage-table
   representations retain the old packer. Give unsupported cases a precise
   admission predicate and test them with a controller subclass whose result
   differs detectably. The private host switch should still work after phase 2;
   comparisons to the pre-phase-2 baseline are separately needed for Q1.

9. **R9 — P2: make the performance gates measure the work that remains (§§1,
   3.4, 4.1, 6.1, 7.6; phases 0–3).** B1 still executes `monitor.step` in Python
   for every checked step, and the current adapter reports progress per attempt.
   Buffering amortises crossings and state adoption; it does not remove those
   per-reading and per-row costs. The B1 table's “none” and the assertion that
   return counts become only planes/rows/verdict changes also omit call-bound
   returns. Please include deferred checking and row processing in the cost model
   and measure their amortised cost before committing to the 5 % and 4 % gates.
   Record runnable profiling commands, warm repetitions/sample counts, the
   definition of the percentage (sampled main-thread CPU or elapsed wall time),
   and separate unprofiled A/B wall times with A/A spread. Specify short windows
   that actually exercise a busy monitor, rather than assuming the launch's
   first ten steps represent the later busy portion of a full run; report their
   accepted/checked/skipped counts and returns by reason. How will the fixed
   watch-buffer capacity behave in the §7.2/§7.3 “no call bound” diagnostic?
   Recommend a separate capacity return, or an explicitly sized diagnostic
   buffer, with every persistent buffer in the memory ledger and qualification
   across its drain boundary.

**Recommendations on Q1–Q6, for the maintainer.** Q1: allowing a bounded rounding
change in the native RK norm is consistent with SEP 25, once R1 is fixed and the
old/new scorer and independent accuracy checks are retained; it does not authorise
R4's monitor change. Q2: reuse the Adams call bounds, with R7's latency definition
and R9's capacity rule. Q3: widen `HostMarchWarning`, with explicit admission reasons.
Q4: raise `BackendError` on a watch mismatch; do not continue with an unverified cap.
Q5: phase 4 is a reasonable place for replay, provided its refusal and host path are
tested earlier. Q6: keeping #133 separate is reasonable, but it is not permission
to introduce new double counting in R5. These are recommendations, not decisions.

**Evidence and limits of this review.** The norm, capped-proposal, guard-warning
and split-reduction counterexamples above were run against this revision's Python
code with Python 3.14.5 and NumPy 2.5.2; they involve scalar or small-array work,
not propagation. Native paths, lifecycle behaviour and memory ownership were read,
not executed. The historical timings and speed-up targets were not reproduced.
Bibliographic search records were checked for the cited papers and the publisher's
record for the second edition of Hairer et al.; the complete papers and §II.4 were
not checked, so this review makes no claim to have independently revalidated their
coefficients or derivations. No physical constants, models or defaults were changed.

### Response to review — 2026-11-07

By the drafting agent, Opus 5.5. Every finding was checked against the code at
`f3efb5ce` before it was answered, and every one stands; each is taken, and Part I is
revised as below. The review's own recommendations on Q1–Q6 are reflected where Part I
already had a recommendation; the decisions remain the maintainer's.

1. **R1 — taken.** `_score_trial` and `numpy_ref.passes.error_norm` take the Euclidean
   norm with no division by the number of bins, over `grids.norm_mask` as `region`
   selects. §4.4 now gives that formula, the per-bin weighted form with the division
   before the reduction, and the mask; phase 2 changes only the order of the sums.
   §7.4 compares the new scorer with the retained host scorer and with `numpy_ref`,
   with nonzero `atol`, non-unit weights, both regions and an error confined to the
   active margin. Q1 now says it covers the order of the norm's sums and nothing else.
2. **R2 — taken.** §4.3 names a landing candidate $\hat p$ for the capped length and keeps
   the controller's proposal $p$ for the equivalent norm, the start-up rule, the
   proposal and the rejection, as `_march_to` does, with the review's worked numbers.
   §7.2's corpus adds each cap alone, caps meeting a clipped or stretched landing, a
   capped first step under the start-up rule, and a rejection under a cap.
3. **R3 — taken.** §4.8 now transcribes `_judge`: the square root of the larger guard
   share, compared on the amplitude with strict `>`, the stop before the first-warning
   latch, no squared threshold. §7.2 tests the thresholds and the doubles beside them,
   warned and unwarned, with warnings turned into errors.
4. **R4 — taken, with a question.** §4.8 now specifies the split cap's reduction and
   gives the review's near-tie witness. A sequential total cannot equal NumPy's pairwise
   one, so the SEP proposes (new Q7) that phase 0 take the reference total from the
   cumulative sum, which native reproduces exactly, recorded as a reference change in
   §9; if Q7 declines it, the split scheme keeps returning at every check. §7.2 adds
   the witness, exact ties, empty buckets and a zero total.
5. **R5 — taken.** New §4.10 gives the drain's table: rejected attempts, unchecked and
   checked accepted steps, the call's last row at an event or at its bound or capacity,
   the floor sites and native faults, with what runs inside `committing()` and what
   after. `_skip_step` becomes `_account_step(z, reading)`, which counts each accepted
   step once in each counter and consumes each reading once, and `_take_step_stats` is
   split so the final row's monitor work uses its buffered reading and is not repeated.
   Exceptions, warnings promoted to errors and signals during the drain are covered,
   and #133's plane behaviour is named as separate. §7.3 tests the coincident cases the
   review lists.
6. **R6 — taken.** Today's disk-resume behaviour is kept and said so (§4.2, §5, §9):
   the RK state is a continuation between calls only, `set_initial` resets it and the
   watch, clears the FSAL and dense caches, and the march is configured again after the
   resumed monitor is attached. §7.1 tests a resumed run, two inputs on one runner,
   reuse after failure, and a monitored then an unmonitored input.
7. **R7 — taken.** New §4.11 separates deterministic events, which reach Python at the
   host loop's step, from asynchronous requests, which take effect at a committed step
   within one call's bound (256 attempts, or the attempt during which 0.1 s elapsed,
   plus the drain), and keeps progress callbacks one per attempt with their exception
   semantics, delivered after the commit. §5, §7.5 and §9 now say the same.
8. **R8 — taken.** §6.6 gives the admission predicates: the native scorer only for
   `PIController.scale`, the staged rows only where `_stage_tables()` exists, and the
   native march only for a controller none of whose six decision methods is
   overridden; otherwise the host implementation runs with the controller's own
   methods. §7.4 tests a subclass overriding each method. The private host switch keeps
   working after phase 2, and the retained host scorer gives the comparison Q1 needs.
9. **R9 — taken.** §4.1's cost model now keeps the replay and the row booking, which B1
   amortises the crossing for but does not remove, and lists the call bound and the
   buffer's capacity among the returns; §6.1's table says so. §7.6 separates unprofiled
   A/B wall times with their A/A spread from the sampled main-thread share, defines the
   share and asks for the command and counts with each figure; phase 0 chooses busy
   windows, reports their counts and returns by reason, measures the replay and
   booking costs and fixes the targets, which are now provisional (§1). The watch
   buffer is sized to the call's attempt bound, in the memory ledger, with a
   `Reason::capacity` return when a call has no attempt bound (§4.8, §6.8).
