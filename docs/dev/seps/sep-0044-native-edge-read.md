# SEP 44 — Reading the transverse edge where the state lives

| | |
|---|---|
| **SEP** | 44 |
| **Title** | Reading the transverse edge where the state lives |
| **Author** | John Travers |
| **Drafting agent** | Claude Opus 5.5 |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
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
the host's BLAS (§ Part II); a GEMM remains open to it if that ever matters.

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

**Cost.** Estimated before the work: 30 to 50 ms a read at 768 x 768 x 600 on a data-centre GPU, against 4.3 s then
on an H200 and against a step of about 230 ms. On CPU a read now transforms every plane, where the host read
transformed a fifth to a third of them, but it loses the copy and the host's single-thread scan. Part II has what
was measured (Q2).

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
  read-as-first-pass reuse of §6 comes back, on routes where it's provable. Measured in phase 2: see Part II. It holds
  except for the radial route at one thread, ×1.036.
- **Q3.** Whether an interrupted collar run's record should drain the native current on the way out, or may lag by a few
  steps. Recommended: drain it.

### 13. References

1. SEP 40, [Transverse absorbing boundaries](archive/sep-0040-transverse-absorbing-boundaries.md), §5.3 and §7.7.
2. ADR-049, ADR-057 and ADR-061, [decisions](../decisions.md).

## Part II — Implementation record

Phases 1 and 2 are implemented on the branch `feature/sep-0044-native-edge`, ahead of this SEP's review.

**Phase 1, the Python split.** As designed, with `EdgeTables` in place of `EdgeBinding`. `READ_RESOLUTION` and
`_occupied_planes` are gone. The launch's clean column powers are formed at the load. A resume restores a collar's
current, baselines and peaks ([#512](https://github.com/LupoLab/solprop/issues/512)), and the record gained
`strip_floor`.

**Phase 2, the native read without a collar.** Deviations from the draft: native sums against weight tables (Option E)
rather than returning $u_n$ (Option C), for the reason in §6; the radial read is a dense product per plane rather than
a GEMM. A collar still reduces a host copy.

*Verification, on CPU.* `tests/backends/test_native_edge_reading.py`, all passing: the reading of one state against
`HostEdgeReduction` on all four routes, at a launch and at a read, within four times `tests/bounds.py`'s `fft_bound`;
the state's bits unchanged by a reading; two polarisations on the radial, Cartesian and line routes; one lit node
reaching only its own faces, with the right weight, which white noise can't show; a read at every step on all four
routes under the Runge-Kutta, Adams and low-storage marches, whose final field is bit for bit that of the same run with
the host's read; and a field-free default run with `_observed_field` refused, which has to finish having read the edge.
`tests/backends/test_transverse_backends.py`, `test_transverse_reads.py`, `test_transverse_monitor.py`,
`test_transverse_nonlinear.py`, `test_transverse_resolution.py`, `test_transverse_witnesses.py`, `test_split_crossing.py`
and `test_monitor.py` pass.

*Not yet covered from §7:* the rest of the adversarial probes (`n_to != n_t` radially, a planar `p != 0`, a small node
chunk), the no-interference matrix's collars, and the device rows, which this SEP's phase 5 holds.

*Timing against `main`, Intel Xeon Gold 6240, 18 cores of one socket.* `python -m benchmarks ab`, `--threading
passive`, three rounds of one repeat. This also holds this work's steps 1 and 2, which measured neutral on CPU alone.
Exploratory, on a shared machine:

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

The same work in every row. On a point of 256 x 256 x 48, sixteen fixed steps and two planes at sixteen threads, one
runner, the default check went from 2.4 s to 2.0 s against 1.8 s without it, and from 3.9 to 2.1 field volumes above
the level before the run against 1.5 without it.

*HIP.* The wheel compiles offline for `AMD_GFX942` under ROCm 7.2.4.

## Part III — History

- 2026-11-04: Drafted. Tracking issue: to be opened with this draft.
