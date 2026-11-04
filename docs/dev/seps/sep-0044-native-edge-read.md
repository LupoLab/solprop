# SEP 44 — Reading the transverse edge where the state lives

| | |
|---|---|
| **SEP** | 44 |
| **Title** | Reading the transverse edge where the state lives |
| **Author** | John Travers |
| **Drafting agent** | Claude Opus 5.5 |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-11-04 |
| **Resolution** | — |
| **Requires** | [SEP 40](archive/sep-0040-transverse-absorbing-boundaries.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

The boundary check's transverse edge (SEP 40) is reduced in Python on a host copy of the
state, on every backend. On a device that copy and those reductions take far longer than the
march they watch. This SEP moves the edge's reductions into the native memory space on the
radial, Cartesian, line and planar routes, on CPU and device builds, so that a field-free
native run crosses no state-sized data. Native returns energies per node and per mode, and
judging stays in Python and shared with `numpy_ref`. A requested collar's strip and inward
current move too, unchanged in meaning. Success is judged by parity with `numpy_ref` to
rounding, no state crossing in a field-free run, and a check that costs about a fifth of the
march rather than eleven times it.

### 2. Motivation and scope

A field-free point of a Kerr scan at 768 x 768 x 600 (5.27 GiB), through
`SimulationRunner` with fixed low-storage RK4IP steps, measured on `7a224ea5`
(issue [#385](https://github.com/LupoLab/solprop/issues/385)):

| 8 steps, 4 planes | NVIDIA H200 | AMD Instinct MI325X |
|---|---:|---:|
| `boundary_check=None` | 5.9 s | 4.2 s |
| Default check | 70 s | 29 s |
| 13 crossings of the state | 37.6 s | 6.1 s |
| Single-thread host reductions | 20.8 s | 5.1 s |
| Temporal monitor's occupancy | 1.7 s | 11.2 s |

A longer window with 51 steps and 17 planes on the H200, at 384 x 384 x 600, goes from 3.95 s to
49.8 s. So the default check multiplies a scan's time by about twelve on the card a large scan
is rented for, and the guide tells users to turn it off, which gives up the check of the time
window as well.

[#510](https://github.com/LupoLab/solprop/issues/510) and
[#502](https://github.com/LupoLab/solprop/issues/502) remove the wasted reads, the occupancy's
atomics and most of the crossing's cost without touching the seam. This SEP removes what is
left, which is structural: the state has to reach the host for each read at all.

**In scope.** The default read and the launch on all four routes; a requested collar's strip,
inward current and rate; the seam they need; the removal of `READ_RESOLUTION`; and the bug in
[#512](https://github.com/LupoLab/solprop/issues/512).

**Out of scope.** Judging, warnings, stops, status, the run's record and the read spacing, which
stay in Python. Whether `L/16` should bind on light far from the edge
([#396](https://github.com/LupoLab/solprop/issues/396)), a numerics decision that this makes less
pressing. Whether the inward current survives
([#388](https://github.com/LupoLab/solprop/issues/388)). `mlx`, which keeps its host read. Moving
the march itself ([#110](https://github.com/LupoLab/solprop/issues/110)).

### 3. Context

[`TransverseEdge`](https://github.com/LupoLab/solprop/blob/e49925d1/src/solprop/transverse_monitor.py)
holds the geometry, the reductions and the judgement together, and
[`StepDriver._read_transverse`](https://github.com/LupoLab/solprop/blob/e49925d1/src/solprop/backends/driver.py#L1985)
hands it `_observed_field`, which is the identity on `numpy_ref` and a whole state's copy on
native. A read with no collar:

1. scans every frequency plane for energy, `_occupied_planes`;
2. synthesises the kept planes, with a transverse inverse transform or, radially, a dense product on
   the edge's rows;
3. sums each face, and reads the column energies for the spacing and the safety ramp's ledger;
4. judges.

The design page says this is left to a measurement (`docs/dev/backends.md`, "The transverse edge is
read on the host"), and SEP 40 §7.7 sets a 2 % limit that the measurements above exceed by two orders
of magnitude. [ADR-057](../decisions.md) and `_boundary_energies` are the pattern for a reduction that
runs where the state lives, with `numpy_ref`'s host form as the reference.

**Issues.** This fixes [#385](https://github.com/LupoLab/solprop/issues/385) for CUDA and HIP,
[#396](https://github.com/LupoLab/solprop/issues/396),
[#512](https://github.com/LupoLab/solprop/issues/512) and the time-window item of
[#505](https://github.com/LupoLab/solprop/issues/505). It needs
[#510](https://github.com/LupoLab/solprop/issues/510) first, and bears on
[#388](https://github.com/LupoLab/solprop/issues/388),
[#386](https://github.com/LupoLab/solprop/issues/386) and
[#225](https://github.com/LupoLab/solprop/issues/225).

### 4. Theory

Nothing here changes what is measured. With $\tilde E_j(\omega_p)$ the state's column $j$ at
frequency plane $p$ and $E_n(\omega_p)$ its synthesis at node $n$, a read forms

$$
u_n = \sum_p |E_n(\omega_p)|^2, \qquad
c_m = \sum_{\text{pol}} \sum_p |\tilde E_{m,\text{pol}}(\omega_p)|^2,
$$

unweighted, summed over polarisation, with no Nyquist weighting. A face's share is
$\sum_{n \in F} w_n u_n / U$, with $w_n$ the node quadrature and $U = \sum_n w_n u_n$ at the launch,
and the spacing and the ramp's ledger read $\gamma_m c_m$, with $\gamma_m$ the column quadrature. All
of the masks and weights are the edge's, and neither $u_n$ nor $c_m$ depends on them.

**Removing `READ_RESOLUTION`.** A read leaves out the planes holding together at most $10^{-3}$ of the
warning, so by Parseval a face reads low by at most $10^{-9}$ of the launch energy at the default budget.
With no planes left out, the reference and native both form the sums above exactly, and a face's reading
can only rise, by at most that. No test or verdict at the default budget is near it, and the maintainer has
decided this removal.

**The inward current.** $J = \mathrm{Im}(E^* \partial_s E)/k$, rectified per frequency and per line
before it is summed, as SEP 40 defines it. Rectifying after summing would be a different quantity. Its
trapezoid in $z$ is unchanged.

**Contraction.** $\mathrm{Re}\,v\,\mathrm{Im}\,s - \mathrm{Im}\,v\,\mathrm{Re}\,s$ can't cancel
exactly on any build, and no hard-mask contract rests on it, so it takes no strict block (ADR-049). Its
parity bound is $\varepsilon \sum |v||s|$ times the line's measure over $k\,U$.

### 5. Proposed interface

Nothing public changes, except that `solprop.transverse_monitor.READ_RESOLUTION` goes. A read's
numbers can rise by at most the bound above. `memory_ledger` gains a `transverse_edge` row.

```text
StepDriver._edge_reading(state, edge, *, faces, current, full) -> EdgeReading   # host by default
EdgeReading(faces, clip, totals, columns, strip, current)                       # numbers only
EdgeTables(faces, clip, totals, rows)                 # four regions' weights on the nodes, from the edge
```

### 6. Implementation strategies

**Option A: only a faster crossing.** [#502](https://github.com/LupoLab/solprop/issues/502)'s ring takes
a crossing from 2.9 s to about 0.2 s on an H200. It leaves the host's reductions and a host volume per read,
and on a CPU it changes nothing. Worth having for a saved plane, not enough here.

**Option B: native forms each face's sum with its own masks.** Fewest numbers cross. But native would have to
hold each face's masks and a reducer per route, and a parity failure could lie in any of them.

**Option C: native forms $u_n$ and $c_m$ and returns them.** About 6 MiB per read at 768 x 768, with $u_n$ only
on the edge's rows. Every mask, weight, face and judgement stays in Python. But $u_n$ sums over planes at each
node: against the scratch's order that is a stride of a whole plane per element, and in the scratch's order it
is a histogram with a thread's copy or an atomic per element, which is what step 1 of this work removed from the
temporal monitor's occupancy.

**Option E, recommended: native sums against weight tables Python gives it.** The edge gives native three tables,
each four regions' weights on the nodes, and native runs the temporal monitor's existing kernel (`monitor_sums`, in
the scratch's own order, one reduction of five numbers) against them, returning twelve numbers and $c_m$. Native
holds no face, mask or judgement, only rows of weights; there is no reducer per route, so a parity failure can lie
only in the synthesis. Corners count on two faces because two tables hold them. The price is the tables: 96 bytes a
node, 57 MiB at 768 x 768 against a 5.3 GiB state, and an upload per point until a runner shares them.

**Option D: native also judges.** Returning to Python per read costs microseconds against a read's tens of
milliseconds, and a stop needs Python anyway. Not worth moving.

**Python.** `TransverseEdge` keeps geometry, judgement, `_spacing`, the ramp's ledger, `resume` and
`summary`. `_node_power`, `_column_power`, `_lines`, `_strip_inward` and `_onset_current` move to a
`HostEdgeReduction`, deep-copyable because a lockstep batch copies its monitors, which `numpy_ref` and `mlx`
use through the default `_edge_reading`.

**Launch.** It needs no state crossing. `_input_floor` is already a host array, so the noise's reading stays
the host reducer's. `_judge_ramp_band` needs `state - noise`: `set_initial` computes those clean column
powers before it adds the draw, as $\sum_\text{active} |f|^2 + \sum_\text{inactive} |n|^2$, because a supplied
realisation isn't masked. One corner keeps a single copy: a resume with no monitor record.

**Native, Cartesian, line and planar.** `separable_load_current`, whose scale makes the scratch equal the host's
synthesis, on a scratch that is dead at a committed position on all three marches; then `monitor_sums` on the
faces' table, and at a launch the other two; then a column-energy pass on the state. The kernel takes the plane
count and the node count, so it has no route branch beyond those. Planar's node count is `n_y`, not `n_nodes`.

**Native, radial.** The same sums, with the node's value a dense product per plane of the synthesis's row and the
state, from the first row a read's regions can touch. It needs no scratch, so neither `statistics_time_` nor
`modal_complex_`, whose non-crop bins the retained path relies on staying zero. It is slower at one thread than
the host's BLAS (the cost below); a GEMM remains open to it if that ever matters.

**Collar.** Python computes each line's strides, window, inward bin range and onset triple, captured by value,
so native does no tie-breaking. The strip uses a persistent padded transform per axis. The current accumulates in
the commit path that `finish` and `adams_exponential_accept` share, into a device view, as the split collar's
removal does, and the next reading drains it. Judging happens only at reads, so `steps.append(1)` in the native
march's bounds can go, and a collar run stops returning to Python at every step. A Cartesian collar still pays one
transverse inverse per commit, as it does now on the host.

**Reusing the read's transform as the next step's first pass.** Provable on the chunked route and with input
noise, but not on the Kerr volume route, whose first pass is a rank-3 inverse on spectral scratch, and that is
the route this SEP is mainly for. It's left out unless a CPU measurement asks for it (Q2).

**Kernel contract.** New `StepDriver` members, so a schema bump under today's rule. `KernelProblem` doesn't change.

**Cost, estimated.** 30 to 50 ms a read at 768 x 768 x 600 on a data-centre GPU, against 4.3 s then on an H200 and
against a step of about 230 ms. On CPU a read now transforms every plane, where the host read transformed a fifth to
a third of them, but it loses the copy and the host's single-thread scan.

**Cost, on a prototype.** Phases 1 and 2 were prototyped on a branch to test Option E and that estimate, before review.
On an NVIDIA A40 (CUDA 12.8.1) at 384 x 384 x 600, eight fixed steps and four planes, `python -m benchmarks.device_split`:
the 8 reads took 0.41 s together, about 50 ms each, and the point took 6.6 s against 5.8 s with no check and 9.8 s with
#502's ring alone, with no state crossing. A40 timings say little of a data-centre card, whose reads this hasn't measured. On CPU, against
`main`, with #510 and #502 in, `python -m benchmarks ab`, Intel Xeon Gold 6240, 18 cores of one socket, passive
threads, three rounds of one repeat, exploratory on a shared machine:

| case | threads | wall, s | ratio (range) | field difference |
|---|---:|---|---:|---:|
| `kerr_beam_192` | 1 | 18.74 → 17.62 | 0.934 (0.934–0.955) | 1.6e-16 |
| `radial_filament_256` | 1 | 7.66 → 7.93 | 1.036 (1.024–1.059) | 4.2e-15 |
| `line` | 1 | 1.017 → 1.011 | 0.997 (0.986–1.008) | 1.4e-15 |
| `planar` | 1 | 0.571 → 0.571 | 1.004 (1.001–1.007) | 0 |
| `kerr_beam_192` | 16 | 3.08 → 2.08 | 0.675 (0.662–0.683) | 0 |
| `radial_filament_256` | 16 | 1.763 → 1.722 | 0.984 (0.936–0.986) | 0 |
| `line` | 16 | 0.296 → 0.286 | 0.934 (0.900–1.053) | 0 |
| `planar` | 16 | 0.172 → 0.163 | 0.945 (0.937–0.980) | 0 |

The same work in every row. On a point of 256 x 256 x 48, sixteen fixed steps and two planes at sixteen threads, the
default check went from 2.4 s to 2.0 s against 1.8 s without it, and from 3.9 to 2.1 field volumes above the level before
the run against 1.5 without it. The prototype's read of one state agrees with `HostEdgeReduction`'s on all four routes,
at a launch and a read, within four times `tests/bounds.py`'s `fft_bound`; a read at every step under each march leaves the
final field bit for bit that of the same run read on the host; and a field-free default run finishes with
`_observed_field` refused, on the CPU and the A40. The wheel compiles for `AMD_GFX942` under ROCm 7.2.4.

### 7. Verification and validation

- **Same-state parity.** New `tests/backends/test_native_edge_reading.py`: native's reading of a state against
  `HostEdgeReduction` on that state's own host copy, which removes trajectory rounding and leaves the budgets of
  `tests/bounds.py`. Over route, scheme and march, with adversarial probes: energy in one (polarisation, last
  frequency) bin, an anisotropic box with light in one corner, `n_pol = 2`, `n_to != n_t` radially, a planar
  `p != 0`, a small node chunk, and mixed inward and outward frequencies that tell per-row rectification from
  rectifying a sum.
- **No interference.** A read forced at every commit, on a matched schedule, across the low-storage march with and
  without its stage register, Runge-Kutta with FSAL, the split and in-stage collars and both Adams formulations.
  Final fields agree with the unread run within the rounding budget.
- **Whole runs.** `tests/backends/test_transverse_backends.py` extended to every route and to collars: the same stop,
  read count and record as `numpy_ref`, including `current`, `strip_max` and `returned_max`.
- **No crossing.** A test that fails on `_observed_field` during a field-free watched run: none allowed, and one for a
  resume with no record. A collar run's native calls are far fewer than its commits.
- **Device logic on CPU.** Both the team and host kernels compile everywhere, with `#if` only at the point of use,
  so that CPU, ASan and UBSan run the device-shaped logic. A GTest beside them.
- **By hand on CUDA and HIP.** `pytest -m gpu` with the above mirrored, a probe above the device GEMM's shape, a
  volume-route probe and a collar. No CI lane runs these
  ([#354](https://github.com/LupoLab/solprop/issues/354)).
- **Benchmarks.** `python -m benchmarks.device_split` on an NVIDIA H200 and an AMD Instinct MI325X, against the table
  in §2, and `python -m benchmarks ab` on `kerr_beam_192`, `kerr_beam_384`, `line`, `planar` and
  `radial_filament_256` against 0.1.0 on an AMD EPYC 7763. Targets: under 1.25 times the check-off point on the longer
  window above, and no state crossing.

### 8. Documentation

`docs/dev/backends.md`, "The transverse edge is read on the host"; `docs/guide/setup/running.md`, "The boundary check on a
free-space scan", whose advice to pass `boundary_check=None` goes; `docs/guide/results/output.md` and
`docs/guide/free-space/model.md`, which say the check reads on the host; `docs/dev/kernel-contract.md`; and a ledger
entry in `docs/dev/benchmarks.md`. No gallery.

### 9. Compatibility and impact

`READ_RESOLUTION`, public since 0.2's development, goes. Native's readings rise by at most $10^{-9}$ of the launch at the
default budget against the current behaviour, and match `numpy_ref` to rounding. A resumed collar run changes, because
[#512](https://github.com/LupoLab/solprop/issues/512) is fixed. The record gains `strip_floor`. The output schema's own
compatibility is whatever [#508](https://github.com/LupoLab/solprop/pull/508) settles.

### 10. Alternatives considered

Options A, B and D in §6. **Keeping plane skipping on native:** an extra pass and a transform plan per kept count, for
a saving a device doesn't need. **Reading at a stage's physical field:** between the inverse and the first crop of pass 3
the scratch holds the predictor at $z + \Delta z$, which isn't the committed state the monitor's contract names
(`docs/dev/backends.md`, on reading what a step computed). **Retiring the current to spare the collar's transform:** a
physics decision, left to [#388](https://github.com/LupoLab/solprop/issues/388).

### 11. Phases

1. **Python split.** `EdgeReading`, `EdgeTables`, `HostEdgeReduction`; `READ_RESOLUTION` goes; clean column powers;
   [#512](https://github.com/LupoLab/solprop/issues/512). Native inherits the host path. Verify: the edge's tests, adapted
   to pass readings.
2. **Native read, no collar.** All four routes on CPU, with same-state parity and the no-crossing test. Native falls back
   to the host path when a collar is requested.
3. **Collar, called per step.** Lines, windows, plans, and `_edge_current`, with the host step still one, which pins parity
   on every route.
4. **Collar, accumulated.** The commit hook; the host step goes. Verify the march's traces are unchanged.
5. **Device qualification.** CUDA and HIP compile, `pytest -m gpu`, and the timings of §7 on two data-centre GPUs.
6. **Documents and ledger.** §8, then `Final`.

### 12. Open questions

- **Q1.** Whether `Simulation.estimate` should count a collar's strip buffers, as it does the held plane. Recommended: yes.
- **Q2.** Whether a CPU read, with every plane now transformed, stays within SEP 40's 2 % on a long run. If not, the
  read-as-first-pass reuse of §6 comes back, on routes where it's provable. On the prototype it holds except for the
  radial route at one thread, ×1.036 on a window (§6); a GEMM could take that, if it's worth a second code path.
- **Q3.** Whether an interrupted collar run's record should drain the native current on the way out, or may lag by a few
  steps. Recommended: drain it.

### 13. References

1. SEP 40, [Transverse absorbing boundaries](archive/sep-0040-transverse-absorbing-boundaries.md), §5.3 and §7.7.
2. ADR-049, ADR-057 and ADR-061, [decisions](../decisions.md).

## Part II — Implementation record

No phase has been implemented.

## Part III — History

- 2026-11-04: Drafted. Tracking issue: [#514](https://github.com/LupoLab/solprop/issues/514).
- 2026-11-04: GPT-6 (AI) reviewed the proposal at `a121c73c`; findings and questions below.

### AI review — 2026-11-04

Reviewer: **GPT-6 (AI)**. Reviewed all of Part I against SEP 0, SEP 40, the current
monitor and native implementation, the numerical comparison rules, and the linked
issues. Part I is unchanged. These are requests for the drafting agent and
maintainer to address, not amendments or acceptance decisions. All findings remain
open under [#514](https://github.com/LupoLab/solprop/issues/514).

The separation recommended in Option E is sound in principle: Python supplies the
geometry and judges the readings, while native performs numerical operations on
resident data. Keeping per-line, per-frequency rectification and testing scratch
interference are particularly important. The proposal still needs the following
contract and qualification details before acceptance.

1. **R1 — P1: specify the actual weighted-reduction contract (§§1, 4–6, 11).**
   The abstract promises node and mode energies, but Option E returns twelve
   regional sums and column powers. `EdgeTables.totals` and `EdgeReading.totals`
   have no defined row meanings, units or normalisation; the requested collar's
   removal rate has no named result. The current consumers need distinct values:
   total launch norm, face sums, clip sums, the launch region used by
   `_launch_return`, and the rate-weighted sum used by `_collar_rate`. The last
   must also be available at later reads, although §6 says the other two tables
   run only at launch. See the
   [existing consumers](https://github.com/LupoLab/solprop/blob/a121c73c/src/solprop/transverse_monitor.py).
   These choices determine the measured quantity, rather than just its storage.
   Please give each table row and returned component an equation, shape and
   scaling, define unused rows and the `faces/current/full` flags, and say whether
   `columns` already includes the column quadrature. Include how planar mode and
   polarisation axes are summed and how the strip's winning face retains its own
   companion `windowed` value. Which result carries the collar rate at an ordinary
   read? Add independent weighted-sum tests; agreement between two reducers using
   the same incorrectly constructed tables would not establish correctness.

2. **R2 — P1: make the current accumulator's lifecycle part of the contract
   (§§6, 7, 9, 11, Q3).** Moving accumulation into native memory introduces state
   beyond the scalar `current`: its previous accepted sample, position or interval,
   and the undrained integral. Draining only at an edge read leaves Python's record
   stale after a native march returns between reads. The existing
   [execution-record path](https://github.com/LupoLab/solprop/blob/a121c73c/src/solprop/driver/run.py)
   obtains `backend.metadata()` for saved states, snapshots and failures, and
   [loading a new input](https://github.com/LupoLab/solprop/blob/a121c73c/src/solprop/backends/driver.py#L1148)
   promises to reset everything a reused backend reads. Fixing #512 requires more
   than adding `strip_floor`: losing the preceding current sample drops the first
   trapezoid across a resume, even if the cumulative integral is restored.
   **Needs a maintainer decision:** resolve Q3 before phase 4; recommend a current
   record through the last committed state. Specify an idempotent drain, retention
   of the last sample across drains, reset on `set_initial`, restore on resume,
   and the initial-sample rule. Update only after the final mask/split treatment,
   never for a rejected trial or an interpolated observation. Test repeated metadata
   reads, an interruption between edge reads, snapshot/resume, and two runner
   inputs, including one after failure. How are partially completed low-storage
   steps kept from changing the last committed monitor record?

3. **R3 — P2: derive bounds for the complete readings (§§4, 6, 7).** The stated
   current bound, proportional to `epsilon * sum(abs(v) * abs(s))`, addresses the
   product cancellation at supplied values, but does not account for synthesis,
   the finite-difference slope, the reduction's addition depth or accumulation over
   accepted steps. A slope formed from nearly equal neighbours can have error set
   by their magnitudes divided by their separation, rather than by the resulting
   slope's magnitude. Likewise, the radial path is a dense dot product, not an FFT;
   four times `fft_bound` alone does not justify its error at arbitrary `n_r` or
   under cancellation. Please declare an absolute error budget through synthesis,
   squaring/weighting, reduction, division by the launch norm, rectification and
   the current's trapezoid, using the applicable helpers in `tests/bounds.py`.
   Distinguish unit roundoff from machine epsilon. Include a nearly cancelling
   radial synthesis and an almost-zero inward current. Whole adaptive runs need
   accuracy envelopes; exact stop/read-count parity should be limited to controlled
   witnesses with matched histories and margins from decision thresholds. This
   does not request a strict floating-point block or bitwise parity.

4. **R4 — P2: measure the CPU criterion against the right baseline (§§3, 6, 7,
   Q2).** SEP 40 §7.7's 2 % is the overhead of enabling the ramp and transverse
   monitor on the *same implementation*, compared with disabling them. The table
   in §6 instead compares the prototype with `main`; a ratio below 1.02 does not
   establish that requirement. The separate 2.0 s versus 1.8 s point has about
   11.1 % check-on overhead, although that point alone does not establish a miss of
   SEP 40's more specific transverse-only criterion. Please separate improvement
   over the old code, monitor overhead, and the whole-point GPU target. Measure
   ramp only, transverse reads only and both, reporting construction separately
   and actual steps/source calls as SEP 40 requires. The comparison with 0.1.0 is
   useful historical evidence but cannot replace this control. Include `numpy_ref`
   after removing plane skipping: it retains host synthesis and does not gain
   native's avoided transfer. **Needs a maintainer decision:** retain the 2 % CPU
   criterion or explicitly replace/defer it. Also reconcile “fixes #396” with the
   excluded read-ceiling question; keep the remaining work in that issue or a
   linked successor rather than treating faster reductions as its full closure.

5. **R5 — P2: account for default tables as well as collar buffers (§§5, 6,
   Q1).** Q1 asks only about the collar's strip buffers, but Option E also adds
   tables to the default check. At 768 by 768, twelve binary64 rows occupy
   56,623,104 bytes, **54 MiB** (about 57 MB), per resident copy. Python's tables,
   upload staging, column results, radial rows and transform workspaces can add to
   that. The current
   [`estimate`](https://github.com/LupoLab/solprop/blob/a121c73c/src/solprop/driver/estimate.py#L158)
   reads the ledger before loading the input and never attaches a monitor, so a
   ledger entry allocated at attachment alone would remain invisible to it.
   Recommend answering Q1 yes for all persistent resources this SEP introduces,
   and specify how estimation sees them without reading a field. State ownership,
   allocation and release points, host versus device accounting, and which buffers
   can actually be shared across runner inputs. Test the estimate against the bound
   backend's ledger, and verify that a newly constructed check-off run pays no edge
   allocation. How much temporary memory does a collar's padded transform need on
   each route, at the widest supported strip?

6. **R6 — P2: complete and scope the qualification matrix (§§6, 7, 11).** The
   adversarial probes are useful, but the matrix should explicitly include field
   and envelope grids, DC/Nyquist and inactive bins, input/modified/no noise, a
   supplied unmasked realisation, and the supported binary32 host path. The clean
   launch identity in §6 is correct for a fresh input-noise launch; a resume already
   carries the noise and must follow a separately stated path. Include launch-only
   interruption and resume with and without a monitor record. Scope the no-crossing
   promise to monitor-induced transfers: requested fields, some statistics,
   interpolation and terminal checkpoints have separate transfer requirements.
   Failing `_observed_field` alone cannot detect a direct
   `field_to_host()` call, which the native adapter also contains. Add a lower-level
   transfer count or trace for the stated field-free workload, including launch,
   ordinary reads, collar commits and shutdown. For the CPU exercise of device
   logic, require tests that actually invoke the device-shaped implementation;
   merely compiling both alternatives does not exercise the unused one.

7. **R7 — P2: make the performance evidence and phase gates reproducible
   (§§6–8, 11).** The draft correctly labels the prototype as unpublished and the
   GPU target as unmeasured. However, `benchmarks.device_split` is absent from the
   reviewed tree, and neither the prototype revision nor the exact revisions
   carrying #510/#502 are identified for the prototype comparisons. Before using
   these numbers to accept Option E, provide a public reproducible harness or
   patch, pinned baseline and candidate revisions, full commands/settings and
   enough per-repeat results to assess the shared-machine spread. Keep the
   51-step historical measurement as context; specify the short qualification
   window, or obtain an explicit exception to the standing roughly-ten-step rule.
   Assign the schema bump and protected-surface test update to phase 1, where the
   new driver hook first appears. Put the host-path manual changes and release
   note for removing `READ_RESOLUTION` there too, rather than deferring all
   documentation to phase 6. The maintainer's decision to remove it is respected;
   the compatibility text should retain the general omission bound
   `1e-3 * budget**2` and allow a changed verdict for a state within that distance
   of a threshold, even though the existing witnesses are not near one.

**Evidence and limits.** Read the proposal at `a121c73c`, its base at `e49925d1`,
the existing host/native reduction and commit paths, SEP 40's definition and cost
criterion, `tests/bounds.py`, the existing transverse tests, #396, #510, #512, #514
and PR #508. Ran small, non-propagating NumPy probes: a line-grid sample of amplitudes
2 and 3 in two polarisations at the last frequency gives node power 13 and matching
weighted node/column totals to rounding; a supplied unmasked-noise example gives
327 by both the existing mask-and-subtract expression and the proposed clean-power
identity. Checked the table-byte calculation and the check-on/off ratio above.
No prototype, native build, GPU execution or performance measurement was run for
this review. No new physical constants or literature coefficients were introduced.
