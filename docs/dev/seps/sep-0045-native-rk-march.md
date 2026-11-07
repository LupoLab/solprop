# SEP 45 — A native Runge–Kutta march, and boundary checks inside the compiled marches

| | |
|---|---|
| **SEP** | 45 |
| **Title** | A native Runge–Kutta march, and boundary checks inside the compiled marches |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-07) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-11-07 |
| **Resolution** | — |
| **Requires** | [SEP 39](archive/sep-0039-native-adams-march.md) (the native Adams march, its call protocol and ADR-060); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); [SEP 21](archive/sep-0021-absorbing-boundaries.md) and [SEP 40](archive/sep-0040-transverse-absorbing-boundaries.md) (the boundary monitor) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
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
  check is due, takes the reading, applies the monitor's comparisons with thresholds
  solprop computed, and buffers each reading. It returns only at the existing events
  or when a check's verdict needs Python (a warning, a stop, a change of the quiet
  verdict). solprop replays every buffered check through the unchanged
  `BoundaryMonitor.step` when it drains the call, so the monitor's record and its
  arithmetic stay in one place.
- **A. A native Runge–Kutta march.** The adaptive (and later replayed) DOPRI5 loop runs
  in `solprop-native` under SEP 39's design: solprop hands over one policy record and
  the events at which to return, native executes an algorithm this SEP and the kernel
  contract specify in full, and solprop's Python march stays the reference.

B comes first because A depends on it: a native RK march that still returned at every
busy check would save little, and `gnlse_soliton`'s monitor asks for a check at about a
third of its steps (§3.4).

Success means: with B, the native Adams march takes the same steps and writes a
bitwise identical boundary-monitor record as before, and returns only at planes,
statistics rows and verdict changes; with A, the native RK march takes the same steps
as solprop's Python march on the same build; and the Python share of a warm run falls
below 5 % on `argon_blueshift_envelope` and below 4 % on `gnlse_soliton`, with no case
of the tracking set slower beyond the A/B spread.

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
- #133 — the monitor counts a step checked at its plane twice. The drain (§4.8) books
  each step once; this SEP keeps the existing counting so the records stay bitwise, and
  fixes nothing here unless #133 is fixed first.

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
+ N_{\text{attempts}}\, h_{\text{attempt}} + T_{\text{fixed}},$$

with $t_{\text{trial}}$ the native trial, $h_{\text{return}}$ the Python cost of one
return (adopting the state, booking rows, the event's statistics and check), and
$h_{\text{attempt}}$ the per-attempt Python cost of a host loop. Today the RK march has
$h_{\text{attempt}} \approx 200$ µs and $N_{\text{returns}} = N_{\text{attempts}}$; the
Adams march has $h_{\text{attempt}} \approx 1$–$3$ µs and $N_{\text{returns}}$ equal to
the number of host events, which a busy monitor drives to the number of steps. B makes
$N_{\text{returns}}$ the number of planes, statistics rows and verdict changes; A makes
the RK march's $h_{\text{attempt}}$ that of the Adams march. The share removed is largest
where $t_{\text{trial}}$ is smallest: small grids, many threads and devices.

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
march keeps no field history, so nothing else crosses (resume restores the same five).

#### 4.3 One RK attempt

Given the state, one attempt is, in order, exactly as `StepDriver._march_to` does it:

1. $p \leftarrow$ the proposal capped by the quiet collar, the transverse ceiling and the
   split crossing cap (each a `min`, in that order).
2. $(\Delta z, z_{\text{target}}) \leftarrow \mathrm{landing}(z, p, \text{targets})$.
3. The trial at $z, \Delta z$: stage scalars at $z + c_i \Delta z$ and the whole-step
   row (§4.7), one native trial, and the scaled norm $e$ (§4.4).
4. $\tilde e \leftarrow \mathrm{equivalent}(e, \Delta z, p)$ and
   $a \leftarrow \mathrm{accepts}(e)$.
5. The row (end, step, $e$, $a$) is recorded **before** the branch; the end is
   $z_{\text{target}}$ when the step landed, else $z + \Delta z$.
6. On acceptance: `promote_fsal` (which also scores the quadrature alarm), `finish`,
   commit $z \leftarrow$ end; append $\tilde e$ to the history; if `auto_init` and this
   is the first accepted step, apply the start-up rule (§4.5); then
   $p \leftarrow \mathrm{limit}(p, \mathrm{propose}(\text{history}))$; then the step's
   boundary check (§4.8), statistics row and natural stop, any of which may end the
   call.
7. On rejection: $p \leftarrow \mathrm{limit}(p, \mathrm{rejection}(\tilde e))$.

The floor (`limit_step` raising `ConvergenceError` below `dz_min`) can fire in three
places, each with its own order of effects: after a rejected attempt was recorded; in
the start-up rule's `limit_step`, before the history's last value is rewritten; and in
the accepted step's proposal, after the commit and the append, before the step's
statistics. The native march reports which (§6.4) and solprop raises it there.

Replay (phase 4) follows `StepDriver._replay_to`: it never rejects, records after the
commit, appends the raw norm, and stops within `replay_tolerance`.

#### 4.4 The norm and the controller

The trial returns three partials per norm slot: $\sum |y|^2$, $\sum |y_{\text{new}}|^2$
and $\sum |\varepsilon|^2$ over the in-band mask. The scaled norm is

$$e = \frac{\sqrt{\sum|\varepsilon|^2 / n}}{\mathrm{rtol}\,\max\!\left(\|y\|, \|y_{\text{new}}\|\right) + \mathrm{atol}},$$

or the weighted form of `weighted_norm_scale`. Today solprop sums the partials with
NumPy (pairwise above eight terms) and applies `PIController.scale` on the host. This
SEP moves that reduction and scale into one native scorer used by **both** marches
(phase 2), so that the two read identical norms by construction rather than by
transcribing NumPy's summation order.

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

Let the monitor's **watch state** be: the last check position (or none), the steps since
it, `busy`, the quiet run length, `quiet`, `warned` and the crossing cap. Let the
**watch policy** be numbers solprop computes from the monitor at the call: the floor
regions scaled by `FLOOR_MARGIN` ($[3,3,3,1]$ times the floor's expectations), the
launch floor `_FLOOR × launch energy`, the activity threshold $(\texttt{ACTIVE} \cdot
\mathrm{budget})^2$, the warning and stop thresholds `budget` and
`stop_factor × budget`, `QUIET` and `QUIET_CHECKS`, `CADENCE`, `interval`, whether the
quiet collar may rest, and under the split scheme the crossing share and the per-bucket
step limits. After each committed step, inside the march:

1. If no check is due by `due` (busy, `CADENCE`, `interval`, or no check yet), count the
   step as skipped and continue.
2. Otherwise read the state (`monitor_reading`, as today), append (row index, reading)
   to the call's buffer, and evaluate, with the policy's numbers, the comparisons of
   `_found`, `_above`, `_judge` and `_settle`: the new `busy`, quiet run and `quiet`,
   whether the worst guard share exceeds the warning (first time) or the stop
   threshold, and under the split scheme the new crossing cap, which caps the very next
   proposal.
3. Return to the host (`Reason::watch`) if the check predicts a stop, the first warning,
   a change of the quiet verdict, or (on a replayed schedule) a crossing the schedule
   cannot honour; otherwise continue.

When solprop drains a call it walks the rows in order: an unchecked accepted step is
booked with `_skip_step`, a checked one with `monitor.step` on its buffered reading at
its own position. `BoundaryMonitor.step` therefore runs exactly as it does today, on the
same readings at the same steps, and writes the same record; it raises the same error
and the same warning at the same step. After the drain, solprop compares its monitor's
watch state with the march's; a difference is a bug and raises `BackendError`.

Native evaluates only comparisons and the products and sums that `_above` and the
crossing limit perform; the thresholds arrive precomputed, so no `pow` and no constant
is duplicated. The comparisons are compiled without contraction.

#### 4.9 What "the same steps" means

As in SEP 39 §4.9: on one build, the native and Python marches read the same norms
(from the same native scorer, §4.4), apply the same decisions in the same order in
binary64 without contraction, and so take the same steps and produce the same fields.
That is a diagnostic, recorded and tested on short windows, not a promise: across
builds, compilers and libms the guarantee is SEP 25's agreement to the error bound.

### 5. Proposed interface

No public name changes. A user sees:

- faster warm runs on the native backend;
- a run's metadata gains `rk_march` (`"native"` or `"host"` with a reason), beside
  SEP 39's `adams_march`;
- `HostMarchWarning` is widened to name the RK march when a configuration it does not
  cover falls back to solprop's loop (§6.6);
- `boundary_monitor` records unchanged;
- progress callbacks and a stop request act once per call, at most 256 attempts or
  0.1 s apart (as for the Adams march today).

The private environment variable `SOLPROP_RK_MARCH=host`, read at import, selects
solprop's loop for A/B comparison, as `SOLPROP_ADAMS_MARCH=host` does.

### 6. Implementation strategies

#### 6.1 Boundary checks: three options

| Option | What moves native | Python per busy step | Risk |
|---|---|---|---|
| B1. Native predicts, Python replays (recommended) | `due`, the reading, the comparisons, the crossing cap | none; one drain per call | the comparisons exist twice; checked after every drain |
| B2. The whole monitor native | everything, including records, ledger, plane rows, summary, resume | none | duplicates the record arithmetic and resume; against SEP 39 §6.5.7 and SEP 44 |
| B3. Cheaper booking only | nothing; one bulk `_record_attempts` per call | still one return and one `_synchronise` per busy step | low, but removes perhaps half the cost |

B1 is recommended, and includes B3's bulk booking, which every drain benefits from.

#### 6.2 The RK march: three options

| Option | Python per step | Notes |
|---|---|---|
| A1. Native march, SEP 39's design (recommended) | ~1–3 µs, as Adams | policy record and events; Python march stays the reference |
| A2. Native packer and scorer only, loop in Python | perhaps 80–120 µs | phase 2 of A1 on its own; removes the stage-scalar and packing time but keeps the loop |
| A3. Leave it | ~200 µs | SEP 39 Q8's choice |

#### 6.3 Recommendation and order

B1, then A1 built on A2. Phase 2 (A2) is a step on the way, and gives the Python loop the
native scorer so that both marches read the same norms. That moves the native RK
backend's norms at rounding level once, which is recorded (§9).

#### 6.4 Layers and the seam

- **solprop** (`src/solprop/`): `BoundaryMonitor.watch_policy()` and `watch_state()`;
  `_check_boundaries` split so a buffered reading can be booked at its own position;
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
  `MARCH_POLICY_FIELDS`; `DRIVER_SURFACE` gains `_record_attempts` and the split check.
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
- **A v1** (phase 3) admits a single native run with `integrator is None`, adaptive
  stepping, no interpolated planes, staged tables available, no coupled flow and a
  `PIController` (weak or weighted scale), on a CPU build. Anything else runs solprop's
  loop with `HostMarchWarning` naming the reason. Phase 4 removes those reasons except
  lockstep batches and a controller other than `PIController`; phase 5 admits device
  builds.

#### 6.7 Device builds

The march loop runs on the host and drives the device through the resident engine, as
the Adams march does; the norm partials already come back through pinned memory. A
device build's gain is expected to be larger than a CPU build's (SEP 39 §6.7); phase 5
measures it on an NVIDIA A40, and on a data-centre card only with the maintainer's
approval of the cost.

#### 6.8 Cost

Memory: the watch buffer holds at most 256 readings of 10 (or 138) doubles per call,
about 280 kB; the RK march adds no field-sized buffer. Code: about 1500 lines of C++ and
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
- The landings record reproduces the Adams march's decisions on SEP 39's trace corpus
  unchanged.

#### 7.2 Decision traces

- **RK corpus**: `tests/data/generate_rk_traces.py` writes scripted-norm traces of
  `_march_to` *before* phase 0's refactor (start-up on and off, `lag` 0–2,
  `reject_free`, landings at planes and kinks, each floor site). The pure core
  reproduces them exactly (phase 0); the native march through a scripted engine
  reproduces them at call bounds of one attempt, seven and none (phase 3), as
  `tests/backends/test_native_march_traces.py` does for Adams.
- **Watch corpus**: monitor readings scripted around each threshold (activity, warning,
  stop, quiet, floor, split crossing), driven through solprop's monitor and the native
  watch; the predictions agree exactly. A Hypothesis test draws readings and policies and
  checks the native verdicts against `_settle`, `due` and `_judge`.

#### 7.3 Same steps on real problems

On ten-step or short windows of `gnlse_soliton`, `gnlse_raman_shock`,
`argon_blueshift_envelope`, a split-collar capillary and a quiet-collar run: native
against `SOLPROP_ADAMS_MARCH=host` and `SOLPROP_RK_MARCH=host`, the same steps, bitwise
fields and bitwise `boundary_monitor` records on one build. The existing
`tests/driver/test_native_march_call_bound.py` (a call's bound changes no number) is
extended to the watch and the RK march.

#### 7.4 Accuracy envelopes

Phase 2 moves native RK norms at rounding level; every RK case of the test suite stays
inside its SEP 25 envelope against `numpy_ref`, and the release benchmark rows are
compared within one revision.

#### 7.5 Lifecycle

Faults injected at each stage (`MarchFault`), SIGINT and SIGTERM between commits, the
stop flag, the floor at each of its three sites, `BoundaryError` and the first
`BoundaryWarning` with warnings turned into errors: each reaches Python at the same step
as on the host loop (`tests/driver/test_native_march_lifecycle.py`, extended).

#### 7.6 Benchmarks

`python -m benchmarks ab` (interleaved fresh processes, A/A control) on an idle machine,
one thread and the production width, `--length` windows: `argon_blueshift_envelope`,
`gnlse_soliton`, `gnlse_raman_shock`, `he_rdw_averaged`, `argon_blueshift_averaged`,
`rdw`, and the frequent-event shapes of #273. With the main-thread Python share measured
as in §3.4. Targets: Python share below 5 % on `argon_blueshift_envelope` and below 4 %
on `gnlse_soliton`; no configuration slower beyond its A/A spread. Recorded in
`docs/dev/benchmarks.md`.

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
- Progress and stop requests act per call (at most 256 attempts or 0.1 s), as already
  for the Adams march.

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

0. **Baseline, corpora and pure core.** The measurement ledger (§7.6 cases, returns per
   reason); the RK trace corpus generated before refactoring; the watch corpus;
   `rk_core.py` extracted with the corpus reproduced exactly; the strict-compilation
   witness (#135); the contract's §12 drafted.
1. **B on the Adams march.** Watch policy and state, `watch.cpp`, `Reason::watch`, the
   drain with `_check_at`, bulk booking. Exit: §7.2 watch corpus, §7.3 bitwise records
   and steps, returns fall to planes, rows and verdict changes; §7.6 targets on
   `argon_blueshift_envelope`; #273's rows recorded.
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

- **Q1.** May phase 2 move native RK numbers at rounding level (§9), or must the scorer
  reproduce NumPy's pairwise order so that nothing moves? *Recommended:* allow it; it is
  within SEP 25's guarantee and makes the norms agree by construction.
- **Q2.** The RK march's call bound: the same 256 attempts or 0.1 s as Adams?
  *Recommended:* yes, one constant for both, in the record.
- **Q3.** Widen `HostMarchWarning` or add a sibling for the RK march? *Recommended:*
  widen it; one category to filter.
- **Q4.** A drain whose watch state disagrees with the march's: `BackendError`, or a
  silent fallback to per-check returns? *Recommended:* `BackendError`; it is a bug.
- **Q5.** Does replay of a fixed schedule come in v1 or phase 4? *Recommended:*
  phase 4; adaptive runs are where the time is.
- **Q6.** Should #133's double count be fixed first, so that this SEP's tests pin the
  corrected record? *Recommended:* no; keep the record bitwise and fix #133 separately.

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

## Part III — History

- 2026-11-07: Drafted (John Travers, with Opus 5.5), from a profile of warm runs on the
  tracking set (§3.4).
- 2026-11-07: Tracking issue [#562](https://github.com/LupoLab/solprop/issues/562) opened.
- 2026-11-07: GPT-6 completed the AI review below; Part I is unchanged. The findings
  await the drafting agent's response and the maintainer's decisions.

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
