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
radial, Cartesian, line and planar routes, on CPU and device builds, so that in a field-free
native run the monitor brings no state-sized data to the host. Python gives native weight tables;
native returns a sum per region and each mode's energy; and judging stays in Python, shared with
`numpy_ref`. A requested collar's strip and inward current move too, unchanged in meaning. Success
is judged by agreement with `numpy_ref` within a stated rounding budget (§4), no monitor-induced
transfer of the state in a field-free run, and the gates of §7.

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

About 5.0 s and 3.3 s of the first row is the configuration hash, which
[#508](https://github.com/LupoLab/solprop/pull/508) takes out, leaving a march of about 1.9 s and 1.8 s. As context,
not as a gate, a longer window with 51 steps and 17 planes on the H200, at 384 x 384 x 600, goes from 3.95 s to 49.8 s.
So the default check multiplies a scan's time by about twelve on the card a large scan is rented for, and the guide
tells users to turn it off, which gives up the check of the time window as well.

[#510](https://github.com/LupoLab/solprop/issues/510) and
[#502](https://github.com/LupoLab/solprop/issues/502) remove the wasted reads, the occupancy's
atomics and most of the crossing's cost. With both, the default check takes 25 s on the H200 and
9.6 s on the MI325X on the table's window. This SEP removes what is left, which is structural: the
state has to reach the host for each read at all.

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
read on the host"). SEP 40 §7.7 sets a 2 % limit on what the ramp and the edge cost a clean CPU run against the same
implementation without them, which [#396](https://github.com/LupoLab/solprop/issues/396) records as missed
on some cases, with the read ceiling and the read's host cost as the two causes. That limit is
not the table above: a device's cost is far larger, and it is measured against the same build's check-off point. By the maintainer's decision
(R4) this SEP retires that limit: the cost is to be as small as possible, and reported.
[ADR-057](../decisions.md) and `_boundary_energies` are the pattern for a reduction that
runs where the state lives, with `numpy_ref`'s host form as the reference.

**Issues.** This fixes [#385](https://github.com/LupoLab/solprop/issues/385) for CUDA and HIP, and
[#512](https://github.com/LupoLab/solprop/issues/512). It takes the read's host cost out of
[#396](https://github.com/LupoLab/solprop/issues/396), whose read ceiling stays open there. It needs
[#510](https://github.com/LupoLab/solprop/issues/510) first, and bears on
[#388](https://github.com/LupoLab/solprop/issues/388),
[#386](https://github.com/LupoLab/solprop/issues/386),
[#225](https://github.com/LupoLab/solprop/issues/225) and the time-window item of
[#505](https://github.com/LupoLab/solprop/issues/505), which wants a statistic this SEP doesn't add.

### 4. Theory

Nothing here changes what is measured. With $\tilde E_m(\omega_p)$ the state's mode $m$ at frequency plane $p$ and
$E_n(\omega_p)$ its synthesis at node $n$, a read forms

$$
u_n = \sum_{\text{pol}} \sum_p |E_n(\omega_p)|^2, \qquad
c_m = \sum_{\text{pol}} \sum_p |\tilde E_{m,\text{pol}}(\omega_p)|^2,
$$

unweighted, with no Nyquist weighting. A region's sum is $\sum_n W_n u_n$, for weights $W_n$ the edge chooses (§5). A
face's share is that sum over the launch's $U = \sum_n w_n u_n$, with $w_n$ the node quadrature, and the spacing and the
ramp's ledger read $\gamma_m c_m$, with $\gamma_m$ the column quadrature. Where the sums run over one node at a time,
$n$ on the planar route is a $y$ row, its slab channels summed with the polarisations.

The two totals $\sum_n w_n u_n$ and $\sum_m \gamma_m c_m$ are one number by Parseval only as far as the transform is
unitary. An FFT is, to rounding. A discrete Hankel transform isn't: its columns are orthogonal only to its matrix's
own defect, the largest eigenvalue of $\Gamma^{-1/2} S^T W S\,\Gamma^{-1/2} - I$, which is 2.6e-9 at 32 nodes. Nothing
judged depends on their agreement, but a test of it has to allow that.

**Removing `READ_RESOLUTION`.** A read leaves out the planes holding together at most $10^{-3}$ of the warning, so by
Parseval a face reads low by at most $10^{-3}\,b^2$ of the launch energy at a budget $b$, which is $10^{-9}$ at the default
budget. With no planes left out, the reference and native both form the sums above exactly, and a face's reading can
only rise, by at most that. So a state within that distance of a threshold can change its verdict. No existing witness
is that near one, and the maintainer has decided this removal.

**The inward current.** $J = \mathrm{Im}(E^* \partial_s E)/k$, rectified per frequency and per line before it is summed, as
SEP 40 defines it. Rectifying after summing would be a different quantity. Its trapezoid in $z$ is unchanged.

**Rounding budget.** Unit roundoff $u = 2^{-53}$, $\varepsilon = 2u$ and $\gamma_k = ku/(1 - ku)$ (`higham_gamma` in
`tests/bounds.py`). Each bound is absolute, on one reading; a comparison of two readings allows both.

- *Synthesis.* On an FFT route, per plane, $\|\delta E\|_2 \le B \|E\|_2$ with $B$ = `fft_bound(samples)`. Radially it is a
  dense dot product, so per node $|\delta E_n| \le \gamma_{2 n_r + 4} T_n$ with $T_n = \sum_m |S_{nm}||\tilde E_m|$, which
  doesn't shrink when the sum cancels.
- *A region's sum.* Node by node, $|\delta |E_n|^2| \le 2|E_n||\delta E_n| + |\delta E_n|^2 + 2\varepsilon|E_n|^2$. On an FFT
  route Cauchy–Schwarz over the plane turns the first two terms into $2B\sqrt{R_p P_p} + B^2 P_p$, with $R_p$ the region's
  energy in the plane and $P_p$ the plane's: so a dark face's error is set by the plane, not by the face. The terms are
  non-negative, so the reduction adds $\gamma_d \sum_n W_n u_n$, for an addition depth $d$ at most the number of terms.
- *A share.* Dividing by $U$ adds $2u$ and $U$'s own relative error. The judged share, less baseline and allowance, adds
  their absolute errors.
- *The columns.* Non-negative sums of the stored values: $\gamma_{n_\text{pol} n_\omega + 2}$, relative, with no synthesis.
- *The inward current.* With $v$ and $\delta v$ as above and $s = (v_a - v_b)/\Delta$, the slope's error is set by the
  neighbours and not by itself: $|\delta s| \le (|\delta v_a| + |\delta v_b|)/\Delta + u(|v_a| + |v_b|)/\Delta + u|s|$, as
  `stencil_bound` says. Then $|\delta J| \le (|\delta v||s| + |v||\delta s| + 2\varepsilon|v||s|)/k$. Rectifying is
  1-Lipschitz, so a term's error stays below that bound, and an almost-zero current's error is that bound, not a share of the
  current. The non-negative sum over lines adds $\gamma_d$ of it, and its trapezoid accumulates by `prefix_bound`. No
  hard-mask contract rests on the cancellation in $\mathrm{Im}(v^* s)$, so it takes no strict block (ADR-049).

### 5. Proposed interface

Nothing public changes, except that `solprop.transverse_monitor.READ_RESOLUTION` goes. A read's
numbers can rise by at most the bound in §4. `memory_ledger` gains a `transverse_edge` row.

```text
StepDriver._edge_reading(state, edge, *, faces=True, current=False, full=False) -> EdgeReading   # host by default
EdgeReading(faces, clip, totals, columns, strip, current)                                        # numbers only
EdgeTables(faces, clip, totals, rows)                                                            # from the edge
StepDriver.edge_memory(space) -> int                                                             # what reading holds resident
_core.State.crossed_bytes() -> (to_host, from_host)                                              # a state's whole transfers
```

**The tables and what comes back.** $n$ runs over the nodes: $n_r$ radially, $n_y n_x$ with $x$ fastest, $n_x$ on a line and
$n_y$ on the planar kind. $f$ runs over `TransverseEdge`'s faces in their order (`r`; `x-`, `x+`, `y-`, `y+`; `x-`, `x+`;
`y-`, `y+`), and a table's rows past the route's faces are zero. $E_f$ is a face's edge, the outer `GUARD` of the outer
size; $C_f$ is its clip, the outer `GUARD` of the requested size; $O$ is what lies outside the request; and $\alpha_n$ is a
requested collar's power rate, in 1/m. Each table has shape `(4, n_nodes)`. Each returned sum is $\sum_n (\text{weight})\,u_n$,
in the state's units squared times the node measure, not yet divided by the launch.

| Returned | Weight on node $n$ | What reads it | Formed |
|---|---|---|---|
| `faces[f]` | $w_n$ on $E_f$ | the face's share, against its baseline | every reading with `faces` |
| `clip[f]` | $w_n$ on $C_f$ | the launch's clipping warning | `full` |
| `totals[0]` | $w_n$ | $U$, the launch energy | `full` |
| `totals[1]` | $\alpha_n w_n$ | `_collar_rate`, over $U$, in 1/m | `full`, and every read of a collar |
| `totals[2]` | $w_n$ on $O$ | the current's launch baseline | `full`, with a collar |
| `totals[3]` | zero | nothing: reserved, and tested zero | |

Beside the sums, in a reading with `faces`:

| Returned | Shape | Value |
|---|---|---|
| `columns[m]` | `(n_modes,)` | $c_m$, unweighted: the edge multiplies by $\gamma_m$. On the planar kind $m$ is $(y, p)$ |
| `strip[l]` | `(n_lines, 2)` | Each line's `(inward, windowed)` over the launch's measure, not divided by $U$; `None` without a collar. The edge takes the line with the largest inward share, and that same line's windowed value, not the largest windowed value |
| `current` | scalar | $\sum_l M_l \sum_{\text{rows}} \max(J, 0)$, with $M_l$ the line's measure, not divided by $U$; `None` unless asked |

**The flags.** `faces=False` asks for the current alone, and the other fields are `None`. `current=True` asks for it beside
whatever else. `full=True` is a launch: every node is formed, and every row above is meaningful. Otherwise a read may form only
`EdgeTables.rows`, the rows the faces and a collar's rate touch, and leave the others out, so `totals[0]` and `totals[2]`
mean nothing at a read and aren't read. `totals[1]`'s rate is zero outside those rows, so it is exact there.

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
only in the synthesis. Corners count on two faces because two rows hold them. The price is the tables: 96 bytes a
node, 54 MiB at 768 x 768 against a 5.3 GiB state, and an upload per point until a runner shares them. **Memory**, below,
has the rest.

**Option D: native also judges.** Returning to Python per read costs microseconds against a read's tens of
milliseconds, and a stop needs Python anyway. Not worth moving.

**Python.** `TransverseEdge` keeps geometry, judgement, `_spacing`, the ramp's ledger, `resume` and
`summary`. `_node_power`, `_column_power`, `_lines`, `_strip_inward` and `_onset_current` move to a
`HostEdgeReduction`, deep-copyable because a lockstep batch copies its monitors, which `numpy_ref` and `mlx`
use through the default `_edge_reading`.

**Launch.** A fresh launch needs no state crossing. `_input_floor` is already a host array, so the noise's reading stays the
host reducer's. `_judge_ramp_band` needs the launch without its noise: for a fresh input-noise launch `set_initial` computes
those clean column powers before it adds the draw, as $\sum_\text{active} |f|^2 + \sum_\text{inactive} |n|^2$, because a supplied
realisation isn't masked. Other launches take their own path. With no noise, or modified noise, which never enters the
state, the columns are read directly. A resume with a record isn't judged, its state already carries the noise, and it
reads with `full=False`. A resume with no monitor record is judged as a launch, and because its state already carries the
noise and nothing recorded the parts, it keeps a single copy to subtract it: the one crossing left in this path.

**Native, Cartesian, line and planar.** `separable_load_current`, whose scale makes the scratch equal the host's
synthesis, on a scratch that is dead at a committed position on all three marches; then `monitor_sums` on the
faces' table, and at a launch the other two; then a column-energy pass on the state. The kernel takes the plane
count and the node count, so it has no route branch beyond those. Planar's node count is `n_y`, not `n_nodes`.

**Native, radial.** The same sums, with the node's value a dense product per plane of the synthesis's row and the
state, from the first row a read's regions can touch. It needs no scratch, so neither `statistics_time_` nor
`modal_complex_`, whose non-crop bins the retained path relies on staying zero. It is slower at one thread than
the host's BLAS (the cost below); a GEMM remains open to it if that ever matters.

**Collar.** Python computes each line's strides, window, inward bin range and onset triple, captured by value, so native
does no tie-breaking. The strip uses a padded transform per axis, in batches of planes (**Memory**, below). The current is read
at every commit, with `_edge_reading(state, edge, faces=False, current=True)`, and accumulates in Python as it does today. So
the host step stays one and a collar's march returns to Python at every commit, also as it does today: a return takes tens of
microseconds against a step of hundreds of milliseconds on a grid where any of this matters. A Cartesian collar still pays one
transverse inverse per commit, as it does now on the host.

**The edge's record through a run.** Everything accumulated lives in `TransverseEdge`, native holding no integral or sample, so
the record is always through the last committed state, and reading it twice can't change it.

- *When it's updated.* Only from `_take_step_stats`, after the mask and any split treatment and after the position advances, at
  the same point as the periodic statistics. A rejected trial or a low-storage step abandoned in flight never reaches it, and
  an interpolated observation doesn't call it.
- *A fresh launch.* Takes no sample of the current, as today: the first commit takes the first, and the launch's share is the
  baseline.
- *A resume with a record.* `resume` restores `current`, `current_baseline`, `strip_baseline`, `strip_floor`, `strip_max` and
  `returned_max`, and the launch takes a sample from the resumed state, so the next trapezoid continues from it without one being
  dropped ([#512](https://github.com/LupoLab/solprop/issues/512)). A record from before this change has no `strip_floor`, which
  reads as zero.
- *A resume with no record.* Judged as a new launch, as today.
- *A new input to a runner.* A new `TransverseEdge`, whose tables are bound again. Native's allocations are kept, and
  nothing else carries over, including after a stop.

**Reusing the read's transform as the next step's first pass.** Provable on the chunked route and with input
noise, but not on the Kerr volume route, whose first pass is a rank-3 inverse on spectral scratch, and that is
the route this SEP is mainly for. It's left out unless a CPU measurement asks for it (Q2).

**Memory.** Every persistent resource here counts in `memory_ledger`'s `transverse_edge` row and in `Simulation.estimate`.

- *Default read.* Three tables of 32 bytes a node, 96 in all, and 8 bytes a mode for the columns: 56,623,104 bytes (54 MiB) and
  4.5 MiB at 768 x 768. Radially, $n_r^2$ doubles more for the synthesis. They're allocated at the first `bind_edge`, in the
  memory space, kept across a runner's inputs, and released at `close`. A run with no edge binds nothing and pays nothing.
- *On the host.* The edge's own tables are another 54 MiB, the upload's staging a transient 54 MiB, and each read's column result 4.5
  MiB. These aren't resident on a device and aren't in the ledger, which lists the state's memory space. A runner's points can share the
  tables, since they depend only on the grid and the design.
- *`estimate`.* It reads the ledger before loading an input and attaches no monitor, so it asks the backend for these by formula from the grid
  (`StepDriver.edge_memory`, `transverse_edge_bytes`), with no field read and no monitor built, and a requested collar's run adds none until phase 3. A test holds the formula to a bound backend's ledger row.
- *A collar's strip.* Its transform is padded twice over $m$ nodes along the normal. At the widest strip for a collar fraction
  $f$, $m \approx n(1/4 + f)/(2(1 + f))$ across an axis of $n$ nodes, and twice that radially, where $n_r$ nodes span only $r$.
  Transformed all at once, one Cartesian axis would take $2m \times n_\perp \times n_\text{pol} n_\omega \times 16$ bytes: at 768 x 768 x
  600 and $f = 1$, $m = 240$, that is 3.3 GiB, 62 % of the state. So it runs in batches of planes, capped at 256 MiB (45 planes
  there), allocated at a collar's first read. The radial route needs 160 MiB at 1024 x 8192, and the planar 35 MiB at 768 rows
  and 4800 channels and bins, both below the cap. These are formulas, to be measured in phase 3.

**Kernel contract.** New `StepDriver` members, so phase 1 rewrites the frozen table in `tests/backends/test_driver.py`, and takes
the next kernel schema if the number still exists by then ([#508](https://github.com/LupoLab/solprop/pull/508) takes 46 and
[#516](https://github.com/LupoLab/solprop/pull/516) 47).
`KernelProblem` doesn't change.

**Cost, estimated.** 30 to 50 ms a read at 768 x 768 x 600 on a data-centre GPU, against 4.3 s then on an H200 and
against a step of about 230 ms. On CPU a read now transforms every plane, where the host read transformed a fifth to
a third of them, but it loses the copy and the host's single-thread scan.

**Cost, on a prototype.** Phases 1 and 2 were prototyped, before review, to test Option E and that estimate. By the maintainer's
decision (R7) the prototype isn't published, so these numbers are evidence for this proposal and don't stand in for §7: phase 2's pull
request carries the harness and repeats them. Baseline `main` at `e49925d1`; the prototype holds [#508](https://github.com/LupoLab/solprop/pull/508),
[#510](https://github.com/LupoLab/solprop/issues/510) and [#502](https://github.com/LupoLab/solprop/issues/502). Intel Xeon Gold 6240, 18 cores of one socket,
`OMP_WAIT_POLICY=passive`, on a shared machine.

*The monitor's overhead on one implementation.* SEP 40 §7.7's control, `python -m benchmarks.edge_overhead --width 18`: each case run four ways in one process,
interleaved, against itself with neither the ramp nor the reads, keeping the temporal monitor. Medians, of 12 rounds on `line` and `planar`, 6 on `radial_filament_256`
and `kerr_beam_192`, and 3 on half of `cartesian_filament_128`, whose light reaches its edge. Spread is the median absolute deviation of the check-off wall.

| native | check off, s | ramp | reads | both | added, s | on `main`: both, added | source calls or steps | reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `line` | 0.225 ± 0.001 | −0.1 % | +13.2 % | +13.6 % | 0.031 | +19.7 %, 0.046 | 196 → 224 calls | 23 |
| `planar` | 0.124 ± 0.001 | 0.0 % | +5.8 % | +8.4 % | 0.010 | +13.3 %, 0.019 | 41 → 41 steps | 23 |
| `radial_filament_256` | 1.319 ± 0.006 | −0.7 % | +1.9 % | +2.2 % | 0.029 | +1.3 %, 0.022 | 128 → 128 steps | 21 |
| `kerr_beam_192` (a read at each of 4 steps) | 1.835 ± 0.006 | −0.1 % | +5.9 % | +6.3 % | 0.115 | +38.2 %, 1.186 | 16 → 16 calls | 4 |
| `cartesian_filament_128`, half | 54.48 ± 0.49 | −0.2 % | +2.2 % | +3.0 % | 1.61 | not run; +10.5 % in #396 | 301 → 308 calls | 24 |

The percentages aren't comparable across the two builds, whose check-off times differ: `main`'s includes the hash. The added seconds are. Two costs come apart here.

- *The read ceiling's extra steps* are unchanged, and on `line` they are all of it: 28 more source calls at the check-off rate is 0.032 s, against 0.031 s added. They stay in [#396](https://github.com/LupoLab/solprop/issues/396).
- *The read itself* is what this SEP changes. On `kerr_beam_192` it falls from about 296 ms to 29 ms, though `main`'s rounds there spread by 0.25 s on a shared machine. On `planar` it falls from about 0.8 ms to 0.4 ms. On the Cartesian filament, once its 7 more source calls are taken out, it falls from the 180 ms #396 records to about 14 ms by medians, which three rounds' spread of 0.5 s leaves anywhere up to about 35 ms.
- *Radial* may rise, from about 1.0 ms to 1.4 ms a read, where a BLAS product gave way to a plain loop. At 18 threads that's within this table's spread of 0.01 s, but at one thread below it is clear.

Building and translating each variant is reported apart by the harness and isn't in these rows. The ramp costs nothing measurable anywhere.

On `numpy_ref`, 6 rounds on the short cases, after removing plane skipping: `line` +13.8 % (+14.3 % on `main`), the same ceiling; `planar` −4.0 % (−2.5 %), within a spread of 0.1 s; `radial_filament_256` +0.9 %, 3 rounds. So removing plane skipping costs the host path nothing measurable on these cases.

*Improvement over the old code.* `python -m benchmarks ab` against `main`, three rounds of one repeat, passive threads:

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

The same work in every row. With [#510](https://github.com/LupoLab/solprop/issues/510) and [#502](https://github.com/LupoLab/solprop/issues/502) alone, `radial_filament_256` at one thread is 0.999, so its 1.036 is the native read's.

*The whole point.* On an NVIDIA A40 (CUDA 12.8.1) at 384 x 384 x 600, 8 fixed steps and 4 planes, `python -m benchmarks.device_split 384`: the 8 reads took 0.41 s together, about 50 ms each, and the point took 6.6 s against 5.8 s with no
check and 9.8 s with [#502](https://github.com/LupoLab/solprop/issues/502)'s ring alone. A40 timings say little of a data-centre card, which hasn't run this. So §7's gates there are unmeasured.

*Parity.* The prototype's tests pass on CPU, and those it had at the time on the A40; the wheel compiles for `AMD_GFX942` under ROCm 7.2.4. The transfer count hasn't run on a device. There, the evidence so far is the A40's profile of the point above, which shows no `_observed_field` call.

### 7. Verification and validation

- **Same-state parity.** `tests/backends/test_native_edge_reading.py`: native's reading of a state against
  `HostEdgeReduction` on that state's own host copy, which removes trajectory rounding and leaves the budget of §4, stated
  through the helpers in `tests/bounds.py`. Over route, scheme and march.
- **Independent of the tables.** Two reducers agreeing on one wrong table would show nothing, so each region is also held to SEP 40's
  definition and a plain transform, written in the test from the grid alone; the total to the columns by Parseval, a separate path,
  allowing a radial grid's orthogonality defect (§4); one lit node to exactly its own faces and its weight; and a collar's rate to
  its design at a lit node.
- **Adversarial probes.** A nearly cancelling radial synthesis against the absolute budget, whose relative error is large; energy
  in one (polarisation, last frequency) bin; an anisotropic box with light in one corner; DC, Nyquist and inactive bins; `n_pol = 2`;
  `n_to != n_t` radially; a planar `p != 0`; a small node chunk; an almost-zero inward current; and mixed inward and outward
  frequencies, which tell per-row rectification from rectifying a sum.
- **The qualification matrix.** Field and envelope grids; input, modified and no noise, and a supplied unmasked realisation;
  `numpy_ref`'s binary32 host path, which native refuses by name; fresh launches, launch-only interruptions, and resumes with and without a
  monitor record.
- **No interference.** A read forced at every commit, on a matched schedule, across the low-storage march with and
  without its stage register, Runge-Kutta with FSAL, the split and in-stage collars and both Adams formulations.
  The final field is bit for bit that of the same run read on the host, which touches no scratch.
- **The record.** An interruption between edge reads, a snapshot and its resume, and two runner inputs, one of them after a stop, which must
  read as a new runner would.
- **Whole runs.** `tests/backends/test_transverse_backends.py` extended to every route and to collars. Exact stops, read counts
  and records against `numpy_ref` hold only on controlled witnesses, with matched histories and margins from the decision thresholds.
  Other adaptive runs are held to an accuracy envelope ([testing.md](../testing.md) §3).
- **No monitor-induced transfer.** The promise is about what the monitor costs: requested fields, some statistics, interpolation and
  a terminal checkpoint keep their own transfers. Refusing `_observed_field` can't see a direct `field_to_host()`, so native counts the bytes
  through its crossing paths, each way, and a field-free default run through launch, reads and shutdown is held to none to the host,
  and one state from it, the load. Collar commits join in phase 3. On a device, a trace of the same workload confirms it.
- **Memory.** `transverse_edge_bytes` against a bound backend's ledger row; a check-off run's row at zero.
- **Device logic.** Phases 1 and 2 add no device-shaped kernel: the sums are `monitor_sums`' existing branch, and the column pass
  is one code on every build. So their device logic is exercised by running on a device, below. A device-shaped kernel in phase 3 has to be invoked
  by a CPU test, not merely compiled beside the one that runs there. A GTest beside them.
- **By hand on CUDA and HIP.** `pytest -m gpu` with the above mirrored, a probe above the device GEMM's shape, a
  volume-route probe and a collar. No CI lane runs these
  ([#354](https://github.com/LupoLab/solprop/issues/354)).
- **Benchmarks.** Three questions, each with its own control.
  - *The monitor's overhead on one implementation:* `python -m benchmarks.edge_overhead`, SEP 40 §7.7's four-way control, on `line`,
    `planar`, `radial_filament_256`, `kerr_beam_192` and half of `cartesian_filament_128`, on native and on `numpy_ref`, reporting construction
    apart, source calls and reads. It carries no percentage limit: by the maintainer's decision (R4) the cost is to be as small as possible, and
    reported, with the read ceiling's steps kept apart from the read's own cost.
  - *Improvement over the old code:* `python -m benchmarks ab` against `main`, and against 0.1.0 on an AMD EPYC 7763 as history.
  - *The whole point on a device:* `python -m benchmarks.device_split 768`, on an NVIDIA H200 and an AMD Instinct MI325X: 8 fixed steps
    and 4 planes, which is the ten-step rule's window and makes one read a step, the worst case. Gates: no state transfer; a read under
    half a step; and the point under twice its check-off time, with the hash gone. The 51-step window is reported as context.

### 8. Documentation

With the phase that changes what they say. Phase 1: `docs/guide/free-space/model.md`'s account of what a read leaves out, and the release
note for removing `READ_RESOLUTION`. Phase 2: `docs/dev/backends.md`, "The transverse edge is read on the host";
`docs/guide/setup/running.md`, "The boundary check on a free-space scan", whose advice to pass `boundary_check=None`
goes; `docs/guide/results/output.md`; `docs/dev/kernel-contract.md`; and a ledger entry in `docs/dev/benchmarks.md`. Phase 3: a
collar's costs. No gallery.

### 9. Compatibility and impact

`READ_RESOLUTION`, public since 0.2's development, goes. A read's shares rise by at most $10^{-3}\,b^2$ of the launch at a budget $b$
($10^{-9}$ at the default budget) against the current behaviour, and match `numpy_ref` within §4's budget. So a state within that
distance of a threshold can change its verdict, though no existing witness is that near one. A resumed collar run changes, because
[#512](https://github.com/LupoLab/solprop/issues/512) is fixed. The record gains `strip_floor`, which is in the execution
record and outside the configuration hash, so under [#508](https://github.com/LupoLab/solprop/pull/508) it needs no other step; an
older file resumes with it at zero. SEP 40 §7.7's 2 % limit on the default's cost is retired by the maintainer's decision (R4).

### 10. Alternatives considered

Options A, B and D in §6. **Keeping plane skipping on native:** an extra pass and a transform plan per kept count, for
a saving a device doesn't need. **Reading at a stage's physical field:** between the inverse and the first crop of pass 3
the scratch holds the predictor at $z + \Delta z$, which isn't the committed state the monitor's contract names
(`docs/dev/backends.md`, on reading what a step computed). **Retiring the current to spare the collar's transform:** a
physics decision, left to [#388](https://github.com/LupoLab/solprop/issues/388).

### 11. Phases

1. **Python split.** `EdgeReading`, `EdgeTables`, `HostEdgeReduction`; `READ_RESOLUTION` goes; clean column powers;
   [#512](https://github.com/LupoLab/solprop/issues/512). Native inherits the host path. The new driver hook's frozen table, the kernel
   schema if it still exists, the host path's manual changes and the release note land here. Verify: the edge's tests, adapted
   to pass readings.
2. **Native read, no collar.** All four routes on CPU, with same-state parity, the independent tests, the transfer count and the
   memory row. Native falls back to the host path when a collar is requested.
3. **Collar, called per step.** Lines, windows, batched plans, and the current at each commit, with the host step still one, which
   pins parity on every route.
4. **Device qualification.** CUDA and HIP compile, `pytest -m gpu`, and §7's gates on two data-centre GPUs.
5. **Documents and ledger.** What §8 hasn't already moved, then `Final`.

### 12. Open questions

- **Q1.** Whether `Simulation.estimate` should count what this SEP adds. Answered: yes, for every persistent resource (§6).
- **Q2.** Whether a CPU read, with every plane now transformed, stays within SEP 40's 2 %. Answered: the limit is retired (R4), and §6 reports the cost.
- **Q3.** Whether an interrupted collar run's record should drain a native current on the way out. Answered by not having one (R2, §6).

### 13. References

1. SEP 40, [Transverse absorbing boundaries](archive/sep-0040-transverse-absorbing-boundaries.md), §5.3 and §7.7.
2. ADR-049, ADR-057 and ADR-061, [decisions](../decisions.md).

## Part II — Implementation record

No phase has been implemented.

## Part III — History

- 2026-11-04: Drafted. Tracking issue: [#514](https://github.com/LupoLab/solprop/issues/514).
- 2026-11-04: GPT-6 (AI) reviewed the proposal at `a121c73c`; findings and questions below.
- 2026-11-04: Revised in response, with the maintainer's decisions on R2, R4 and R7. Phase 4 goes, and SEP 40 §7.7's 2 % limit is retired.

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

   **Maintainer's decision, 2026-11-04:** the current stays in Python, read natively per commit. There's no native accumulator, and phase 4 goes.

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

   **Maintainer's decision, 2026-11-04:** the 2 % limit is arbitrary, and goes. The cost is to be as small as possible, and reported. The read ceiling's steps stay in #396.

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

   **Maintainer's decision, 2026-11-04:** no public prototype is needed. Only the maintainer's review remains, and it won't read that branch.

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

### Response to review — 2026-11-04

By the drafting agent, Claude Opus 5.5, to the AI review above. Part I is revised as stated. Where a point's premise
concerned resuming, [#508](https://github.com/LupoLab/solprop/pull/508) has since changed part of it: see **What #508 changes**
at the end.

1. **R1: taken.** §1 no longer promises node energies. §5 now gives every table row and returned component a weight, a shape, a
   scale and a reader, says what's unused, and defines `faces`, `current` and `full`. The answers to the questions asked: `columns`
   is unweighted, and the edge applies $\gamma_m$. The planar kind's node is a $y$ row, with its channels summed like polarisations, and its
   column is $(y, p)$. The strip returns each line's pair and the edge keeps the winner's own `windowed`. The collar's rate is `totals[1]`,
   formed at a launch and at every read of a collar, which corrects §6's earlier wording.

   Independent tests are added to the prototype: `test_each_region_reads_what_its_definition_says` writes each region from SEP 40's rule and a
   plain transform, without the edge's tables or reducer, and `test_the_grids_total_is_its_columns_by_parseval` reaches the total
   by a path through no transform. The second found something the draft had wrong: **a radial grid's node total and column total agree only to
   the discrete Hankel transform's orthogonality defect, 2.6e-9 at 32 nodes,** a property of the transform and not of a backend. Nothing judged depends on it. §4
   says so, and the test allows the defect computed from the grid's own matrix rather than a loosened tolerance.

2. **R2: taken, by removing what it concerned, as the maintainer decided.** The lifecycle R2 sets out is what a native accumulator
   would need. The reasons for one don't survive measurement: the saving was the return to Python at each commit, and Python's march control
   costs 10–50 µs a call against a step of hundreds of milliseconds at the sizes where this matters. So the current stays in
   Python, read natively per commit (§6), and old phase 4 goes (§11). The requests then read:

   - *Idempotent drain, retention across drains:* nothing to drain. `metadata()` reads `summary()`, and repeated reads are the same.
   - *Reset on `set_initial`:* a run has a new `TransverseEdge`. Prototype: `test_a_runner_reads_its_next_input_as_a_new_runner_would_after_a_stop`,
     bit for bit, after a stopped input.
   - *Restore on resume, and the first trapezoid:* the point about the lost sample is correct, and the draft's `strip_floor` alone wouldn't
     fix it. On a resume the launch takes a sample at the resumed state, so the next trapezoid starts there
     (`test_a_resume_restores_a_collars_totals_and_continues_its_current`).
   - *The initial-sample rule:* a fresh launch takes none, as today; stated in §6, since "unchanged" had left it implicit.
   - *After the final mask and split treatment, never a rejected trial or an interpolated observation:* the read's one call site is
     `_take_step_stats`, after `_finish`.
   - *Partially completed low-storage steps:* they can't change the record. A step in flight hasn't reached that call site.

   Interruption between edge reads and snapshot/resume are added to §7. I haven't run those two yet.

3. **R3: taken.** §4's new **Rounding budget** carries the error through synthesis, squaring and weighting, reduction, division by the launch norm,
   the slope, rectification and the trapezoid, in $u$ and $\varepsilon$ as named, and it replaces the single product bound R3 criticised. The
   concrete failure is right: four times `fft_bound` had no business bounding a radial dot product under cancellation. The prototype's
   `test_a_nearly_cancelling_radial_read_stays_within_its_absolute_budget` now uses Higham's bound on $|S||\tilde E|$, on a share below 1e-20 of the
   total. The almost-zero current is in §7's probes, for phase 3. Whole adaptive runs move to an accuracy envelope; exact parity is kept
   for controlled witnesses.

4. **R4: taken, with the limit retired, as the maintainer decided.** The criticism is correct: comparing the prototype with `main` says nothing about SEP 40's 2 %.
   §6 now separates the three questions, and `python -m benchmarks.edge_overhead` is SEP 40's four-way control, on native and `numpy_ref`, with construction reported apart. Its table shows the
   two causes #396 named coming apart. The read itself falls, from about 296 ms to 29 ms on `kerr_beam_192`, and on the Cartesian filament from 180 ms to about 14 ms. The ceiling's steps don't, and on `line`
   they're all of the cost. The ramp costs nothing. `numpy_ref` loses nothing measurable from losing plane skipping. Radial at one thread is the one row that rises, ×1.036 against `main`, and §6 says
   it's the native read's; a GEMM would be a second code path for it, which I haven't proposed. Several of these figures are close to their spread on a shared machine, and §6 gives it.

   On #396: "fixes" was wrong. The read's host cost goes, and the read ceiling stays open there, so §3 now says that and the SEP won't close
   it. In re-reading §3 I found a second overclaim the review didn't raise: the SEP doesn't add the statistic that #505's time-window item asks for, so
   that moves to "bears on". I also withdrew the sentence saying the device's cost exceeded SEP 40's limit, since that limit is a CPU control.

5. **R5: taken.** The arithmetic was right and the draft's "57 MiB" wrong: it's 54 MiB. Q1 is answered yes for everything, and §6's new **Memory** states what is
   resident, on the host and transient, who owns it and when it's released, and how `estimate` sees it without a monitor or a field: by
   formula, held to the ledger by a test. A check-off run binds nothing, and the prototype's `test_the_edge_holds_memory_only_once_a_monitor_binds_it` holds that. The temporary memory asked about was larger than I'd assumed: a whole
   axis at 768 x 768 x 600 would take 62 % of a state, so the strip now runs in batches of planes under a 256 MiB cap. Those are
   formulas until phase 3 measures them.

6. **R6: taken,** with one change of form. §7 now has the full matrix, and §6 states each launch path separately, including the resume that keeps a copy. The
   no-crossing promise is scoped to monitor-induced transfers, and §7 replaces the `_observed_field` test's sole authority with a byte count in native's
   crossing paths, each way, now in the prototype.

   **A correction this led to.** The version you reviewed said a field-free default run finishes with `_observed_field` refused "on the CPU and the A40". Adding the count showed that test
   wasn't exercising native at all: it ran the route's default backend, `numpy_ref`. So that statement had no support when I wrote it. With the test fixed, it passes on CPU: nothing to the host, and one state from it. On a device the
   evidence so far is only the A40's profile, which §6 now says, and the corrected test still has to run there.

   On device logic, I didn't take "both kernels compile everywhere" further, because Option E leaves nothing to take: phases 1 and 2 add
   no device-shaped kernel, and §7 says so. The requirement stands for phase 3.

7. **R7: taken, except a public prototype, which the maintainer decided isn't needed.** `benchmarks.device_split` is public with [#516](https://github.com/LupoLab/solprop/pull/516), and `benchmarks.edge_overhead` will be with phase 2's pull request, which repeats §6's numbers.
   So those numbers are evidence for this proposal and not a verdict, and §6 says so. It now pins the baseline and states the commands and each measurement's spread. The qualification window is 8 steps and 4 planes, within the rule, and the 51-step run stays as
   context, so no exception is requested. §7's gates moved from wall times to ratios and a read's share of a step, because #508's hash fix had
   made the old absolute targets too easy. The kernel-schema change, the frozen table, the host manual and the release note move to phase 1 (§8, §11).
   §4 and §9 now state the omission bound as $10^{-3}\,b^2$ and allow a changed verdict within it.

**What #508 changes.** [#508](https://github.com/LupoLab/solprop/pull/508) keeps same-version resume and doesn't touch `TransverseEdge.resume`, so the
resume points above still stand. What differs: a resume now needs an exact configuration hash, and `strip_floor`, in the execution record, is outside
it, so a recorded field needs no other step, as §9 now says; an older file can still resume and reads it as zero. The held plane is gone, which removes
one reader of a low-storage step's state. The hash's cost leaves the check-off row of §2's table. It takes kernel schema 46.
