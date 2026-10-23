# SEP 28 — Free-space evaluation at the memory bound

| | |
|---|---|
| **SEP** | 28 |
| **Title** | Free-space evaluation at the memory bound |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-19) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-19 |
| **Resolution** | Accepted 2026-10-19 by John Travers, on the revision answering R1–R12 and the decisions recorded in Part III; Part I is frozen from here |
| **Requires** | [SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md) (the low-storage march, whose evaluation this makes cheaper); [SEP 25](sep-0025-rounding-level-reproducibility.md) (rounding-level bounds instead of bitwise identity); related: [SEP 22](sep-0022-memory-lean-spatial-execution.md) (statistics-only planes and the held field-free plane) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI): phase 0, phases 1 and 2 (both CPU backends, with device kernels) and the closing step, all merged 2026-10-20 |
| **Manual pages** | [Tuning and threads](../../../guide/performance/tuning.md); [the free-space model](../../../guide/free-space/model.md) (memory and the node budget) |

## Part I — Proposal

### 1. Abstract

Make the source evaluation of solprop's transverse-separable routes — Cartesian, line and
hybrid planar free space — move as few bytes as the memory system allows, on every
stepper (the Runge–Kutta replay and adaptive march, both `Multistep()` formulations, the
low-storage march) and every CPU backend (`numba_cpu`, Kokkos CPU), without specialising
for one response. Both compiled backends already transform over contiguous transverse
planes (all but `numba_cpu`'s field grids). A drafting profile places the excess outside
the transforms' arithmetic: up to four traversals per evaluation that cross between the
registers' layout (ω fastest) and the transverse scratch (ω slowest), copies and clears
inside every node chunk beyond the pad and the crop, a serial host loop over chunks, and
per-element arithmetic repeated at every pass. That attribution, and which of these
causes the growth with grid size, is a hypothesis that phase 0 tests before any
optimisation. The proposal keeps the canonical state layout and the scratch layout both
backends use. It makes every crossing between them cache-blocked, runs the temporal stage
over node tiles sized from the cache within the existing memory cap, under a stated
buffer-ownership contract, and sets a logical volume-pass target for the register folds
and stage combinations. A whole-volume three-dimensional transform and a
transform-ordered register layout are analysed and recorded as alternatives; a Kerr-only
envelope path was rejected. Success is judged on one quiet Linux node over a case matrix
declared before timing: by the low-storage march's scaling from $384^2$ to $768^2$
columns at 600 frequency bins (target: growth within 1.2 times the $N\log N$ ratio,
measured from the smaller of the baseline's and the candidate's $384^2$ times; about 3
times that ratio today), by time per evaluation against an in-tree
reference on the unpadded case (a threshold agreed after phase 0 measures it), by
[benchmarks](../../benchmarks.md) §4's aggregate rule over every declared case, by rounding
budgets and accuracy envelopes fixed before measurement, and by no added memory beyond a
stated, accounted bound.

### 2. Motivation and scope

**The comparison.** Dedicated three-dimensional envelope codes keep the field on a
$(t\text{ or }\omega, y, x)$ grid and take one in-place three-dimensional transform per
nonlinear evaluation, the standard practice for split-step and interaction-picture
envelope propagation [1]. Between transforms their registers are touched in the
transform's own order, so every elementwise pass is contiguous. In drafting
measurements on a 2-socket Zen 3 node (2 × 32 cores, eight NUMA domains), with the node
held exclusively, statistics-only planes and FFTW planned with `FFTW_ESTIMATE`
throughout, such a code — RK4IP [2] with four field volumes — was timed against
solprop on the same Cartesian envelope Kerr beam. Milliseconds per step:

| shape, placement | the dedicated code | Kokkos low-storage march | `numba_cpu` low-storage march | Kokkos IP-DOPRI5 replay |
|---|---:|---:|---:|---:|
| $384^2 \times 600$, one socket, 32 threads | 1655–1683 | 1769–1797 (1.07×) | 5187–5411 (3.2×) | 4329–4375 (2.6×) |
| $384^2 \times 600$, one NUMA domain, 8 threads | 5929 | 13307–13400 (2.25×) | — | — |
| $768^2 \times 600$, one socket, 32 threads | 6362 | 23724 (3.7×) | 71036 (11×) | — |

These are motivation, not ledger entries, and their causes are not isolated: phase 0
re-measures them in-tree (§7.3) and its report, not this table, sets the baseline. Three
things stand out:

- **Scaling.** From $384^2$ to $768^2$ (four times the points) the dedicated code takes
  3.8 times as long, below the $N\log N$ ratio of 4.30 (§4.2); both solprop backends
  take about 13 times as long. Something in solprop's evaluation costs more per element
  as the transverse grid grows. Strided layout crossings are one candidate (§4.3); page
  placement, cache and TLB effects, the transforms' decomposition and per-chunk
  synchronisation are others, and the timings above do not exclude any of them.
- **Bandwidth.** With one NUMA domain (fewer memory channels per process) the gap at
  $384^2$ grows from 1.07 to 2.25 times. That is consistent with solprop moving more
  bytes per step, but the configuration also changes page placement and each thread's
  cache share.
- **`numba_cpu`** is three times slower than Kokkos at 32 threads. The cause is not
  established. Those runs set the FFTW pools explicitly, which bypasses `numba_cpu`'s
  default four-thread cap (§3, Q1), so the cap is not part of that gap; it binds only a
  run with the default settings.

**Where an evaluation goes.** A drafting profile of the `numba_cpu` march, one thread,
$192^2\times300$ envelope Kerr, split one evaluation of 351 ms as: the two-dimensional
transverse inverse and forward transforms 105 ms; the one-dimensional temporal transforms
about 56 ms; the per-chunk pack and crop copies between the transverse scratch and the
time buffers about 45 ms (four node chunks, which at this shape means a node budget of
about 1 GiB; the default 64 MiB gives 53); the Kerr response about 25 ms; the register
fold 99 ms; about 21 ms unattributed. Two whole-volume in-place three-dimensional
transforms of the same volume took about 178 ms with the same library and planning flag.
The profiled run was instrumented and its host is not the ledger's (the
[benchmark ledger](../../benchmarks.md)'s "SEP 24 phase 2" entry gives 134–144 ms per
evaluation for this case on an 18-core Apple workstation, loaded), so the shares matter
and the absolute numbers do not. The ledger's own stage profile of the same case agrees
in shape: the evaluation's core (transforms, the Kerr column, packs and crops) 69 %, the
fused finish 27 %, the register input 3.6 %. In that profile the transforms' arithmetic
is not the excess: solprop's two transform stages cost about what a three-dimensional
pair costs, and the rest lies in the copies, the fold and the traversals around them. It
is one thread at one small shape; it does not show what grows at $768^2$.

**Why it matters.** The routes are memory-bound by construction: one state at
$768^2\times600$ is 5.66 GB (SEP 24 §2), and the low-storage march exists because only
three or four such volumes fit. On such runs the step's cost is largely the bytes it
moves, and SEP 24's measurements showed the evaluation's shared core, not the
integrator, now dominates every march (ledger entries "SEP 24 phase 2" and "phase 3":
the march met neither of its in-tree speed targets for that reason).

**Generality.** The maintainer's requirement is that this is a general improvement:
every Cartesian, line and planar run, every response (Kerr, Raman, the field-grid and
envelope plasma, SEP 6's dispersive Kerr, the in-stage absorber), every stepper and
both compiled CPU backends. A Kerr-only fast path for envelope free space was considered
and rejected (§10). Every in-scope stepper stays in scope for completion; a narrower
fallback (for example a gain confined to the low-storage march) would need a separate
scope decision and does not close this SEP (§11, Q2).

**In scope.**

1. The transverse transform layout and every traversal that crosses between it and the
   registers (§6.2).
2. A node-tile path for the temporal stage that removes the copies and clears no reader
   needs, under a buffer-ownership contract, with the chunked path kept for memory
   limits and devices (§6.3).
3. The register folds and stage combinations of every stepper: logical volumes read and
   written per pass, per-element arithmetic, and a target in volume-passes per
   evaluation (§6.4). The Adams sample's history ownership is audited here; a copy-free
   sample is a later item (§6.4, Q7).
4. Thread scaling on CPU: serial sections, `numba_cpu`'s pools and `prange`
   granularity, Kokkos policies (§6.5).
5. Optionally, FFTW planning effort and wisdom on quiet hosts, as a measurement (§6.6).

**Out of scope.**

- Device qualification of the march and device tile sizes: a separate item after this
  SEP's layout lands. Nothing here may preclude the device path (§6.7), and the
  roadmap item on a device-sized node chunk stays open.
- NUMA-aware scan executors and page placement policies beyond measuring them: a possible
  follow-up, pending a measurement now running.
- Single or mixed precision: a precision decision, not this SEP's.
- The radial and modal routes, whose transforms are matrix products.
- `numpy_ref`, which stays the unchanged oracle.
- Any change to results beyond the numerical contract of §4.5, to defaults, or to the
  public interface.

### 3. Context

**One core, three steppers.** Every stepper on a separable route evaluates its source
through one shared core per backend:

- On `numba_cpu` that core is `NumbaCpuBackend._free_pass` (`backends/numba_cpu/backend.py`).
  The Runge–Kutta stage (`_stage_free`), the Adams sample (`_adams_lab_source`, a
  zero-length stage) and the low-storage pass (`NumbaLowStorage._free_pass`,
  `backends/numba_cpu/low_storage.py`) all call it. The planar kind's is `_slab_pass`.
- On Kokkos it is `GeneralWorkspace::separable_core` (solprop-kokkos `src/general.cpp`).
  The Runge–Kutta stage and `low_storage_separable_pass` call it.

The core takes the input from the transverse scratch through the transverse inverse
transform. For each node chunk it then packs the chunk into the fine temporal buffer,
transforms it to time, evaluates the response (or the in-stage absorber's rate),
transforms back and crops into the scratch. Last comes the transverse forward
transform. Each stepper then reads the scratch in its own finish. §4.1 states the
operations exactly, including the planar route's synthesis and projection.

**The layouts, as built.** The kernel state is `(n_batch, n_modes, n_pol, n_omega)`
with ω contiguous ([kernel contract](../../kernel-contract.md),
[conventions](../../conventions.md) §6). Every register that every march holds uses that
layout too: stage rows, accumulators, the low-storage registers, propagators and the
flow table. **The transverse scratch is already transposed:**

- `numba_cpu` on envelope grids allocates the scratch's storage as
  `(n_omega, n_y, n_x, n_pol)` and exposes it through a transposed view
  (`fft.build_transverse_plans(spatial_contiguous=True)`, chosen by `_spatial_layout`
  for the free kind on envelope grids). The transverse plans then run over contiguous
  planes. Field grids keep the column layout `(n_y, n_x, n_pol, n_omega)`, so there the
  batched two-dimensional transforms are strided by $n_\mathrm{pol} n_\omega$
  elements. The choice is recorded as `transverse_layout` in the backend's metadata.
- Kokkos allocates `transverse_` as `(n_pol_state·n_omega, n_y, n_x)`, `LayoutRight`,
  for every free-space kind. It notes that trailing transform axes avoid Kokkos-FFT's
  per-execution transpose allocations. The planar kind's is `(n_omega, n_p, n_y)`.

So on envelope grids, the two-dimensional transforms are not what is strided. **What is
strided is every traversal between the two layouts.** Each loops over columns outside
and ω inside, which is contiguous in the registers and steps by
$n_x n_y n_\mathrm{pol}$ elements in the scratch:

| traversal | `numba_cpu` | Kokkos |
|---|---|---|
| register input | `make_free_kernels(...).register_input` (low-storage); `_combine` (Runge–Kutta) | `low_storage_separable_input`, `separable_stage_input` |
| chunk gather | `spatial.pack_columns` | `separable_pack_chunk` |
| chunk scatter | `spatial.crop_columns` | the crop in `separable_core` |
| fold or finish | `make_free_kernels(...).fold`; `spatial.finish_material` | `low_storage_separable_fold`; `separable_finish_transform` |

The low-storage march elides the input on passes 2–4 (SEP 24 Q4), which leaves three
such traversals there. On Kokkos CPU `cartesian_for` runs one task per row with a serial
inner loop, so the same holds there.

**Inside each node chunk**, beyond the gather and the scatter:

- **`numba_cpu`.** `pack_columns` zeroes every bin of every row of the fine buffer,
  including the rows a partial last chunk leaves unused, before writing the coarse ones.
  `_free_material` copies the time samples into the node buffer
  (`self._nodes[:count] = plans.modal_time[:count]`) and the response's polarisation
  into the projection buffer (`plans.projected[:count] = self._polarisation[:count]`).
  `_free_pass` clears all of `plans.projected` every chunk; `_slab_output` clears the
  rows a partial planar block leaves, because the fixed-width forward plan reads them.
  The envelope plasma's streams are copied into their transform buffer
  (`_transform_free_sources`) before `assemble`.
- **Kokkos.** `separable_pack_chunk` clears the whole fine spectrum with `deep_copy`,
  and the projected buffer is cleared every chunk. `separable_nodes` copies time into
  the node buffer; a `polarisation_columns` pass copies the polarisation into the
  projection. The roadmap's item on spatial stage clears records the clears.

Some of these clears also initialise rows a fixed-width plan reads, and some copies
separate buffers that a later reader needs; §6.3 states which may go.

The chunk loop is a host loop: Python on `numba_cpu`, C++ on Kokkos. Each iteration
launches several parallel kernels and plan executions.

**The node chunk.** `free_transform` (`translate/free.py`) sets
`node_chunk = min(n_columns, node_budget // (20 · 16 · n_pol · n_to))` with
`DEFAULT_NODE_BUDGET = 64 MiB` (`translate/modal.py`). At $n_{to}=600$ and one
polarisation that is 349 nodes: 423 chunks at $384^2$ and 1691 at $768^2$. The chunk's
hot working set (fine spectrum, time, nodes, polarisation, projection) is a fraction of
the budget's 20 column-equivalents. The kernel contract defines `node_chunk` as "nodes
per chunk for the response pass". The planar kind takes whole y rows,
`max(1, node_chunk // n_x)` of them (`_chunk_layout` on `numba_cpu`, the same rule in
`numpy_ref`), so a planar chunk holds at least $n_x$ nodes even when `node_chunk` is
smaller.

**Per-element arithmetic.** Every low-storage fold with a material term forms the
separable prefactor $Q(\omega,k_\perp) = N(\omega)/\mathrm{Re}\,k_z$ at every element of
every pass. Here $k_z=\sqrt{k^2(\omega)-k_\perp^2}$ is the principal root of the
longitudinal square, which `bodies.spatial_element` forms from high and low parts,
$(k^2-k_\perp^2)+(k^2_\mathrm{low}-k^2_{\perp,\mathrm{low}})$; $N(\omega)$ is the purely
imaginary per-frequency numerator (`SeparableOperator.prefactor_numerator`), and the
evanescent branch is clamped. The work is one complex square root and one division in
`bodies.longitudinal_rate`, reached through `spatial_element` from seven per-frequency
vectors, three per-column values and one scalar. Without a table, the flow
$\exp(hD/2)$ adds a complex exponential (`propagator_pair`) per element. The march
tabulates by default (`low_storage_flows` is 1 on a constant separable operator), at the
cost of one held volume. The SEP 24 phase 2 ledger names the root and the division as
the finish's per-element cost. Pass 4 also reads the state $A$, which its update
$A\leftarrow QB+\tfrac{h}{6}S$ does not need: `a = state[column, j]` on `numba_cpu` and
`Complex a = state(column, omega)` on Kokkos are unconditional in the source, though a
compiler may already drop the load.

**Threads.** On `numba_cpu`:

- `fft.choose_pools` gives a batched transform at most `FFTW_MAX_THREADS = 4` threads
  once it reaches `FFTW_THREAD_ELEMENTS = 65 536` elements. The transverse plans' pool
  (`_transverse_threads`) uses the same rule unless `SOLPROP_TRANSVERSE_FFTW_THREADS`
  overrides it. That cap was measured on fibre transforms up to 393 216 elements, and
  its docstring notes that more threads pay again above about half a million; one
  Cartesian volume is $10^7$ to $10^9$ elements.
- The fold's `prange` and the chunk kernels' parallel transfers apply only in the
  "large" regime (`_parallel_spatial`, and for the transfers `_spatial_layout`).

Kokkos plans transforms above `threaded_fft_min_samples` on its threaded FFT execution
space (`fft_space_for`). The Zen 3 runs above set `SOLPROP_FFTW_THREADS` and
`SOLPROP_TRANSVERSE_FFTW_THREADS` to the process's thread count, and an explicit count
bypasses the four-thread cap (`numba_cpu/fft.py`), so their `numba_cpu` transforms used
every thread; a run with the default settings keeps the cap (Q1).

**Planning.** `numba_cpu` plans with `FFTW_MEASURE` by default (`SOLPROP_FFTW_PLAN`).
The Kokkos build uses Kokkos-FFT 2.0.0, whose host plans pass `FFTW_ESTIMATE`. The
measurements above planned both with `FFTW_ESTIMATE`. A measured plan of the dedicated
code was reported 2.4 times slower on a shared host. So planning does not explain the
gap; it is a minor, optional item (§6.6). [SEP 4](../sep-0004-execution-autotuning.md)
covers searching over planning policies.

**Memory.** The low-storage march's ledger names four field-sized volumes on a Cartesian
grid with the stage register elided: state, accumulator, transverse scratch and flow
table (three when recomputing). The node-chunk buffers and plan workspace come on top
([backends](../../backends.md), the low-storage march; `low_storage_flows`,
`low_storage_stage_register` in `backends/memory.py`). In the drafting measurements the
Kokkos march's resident set was 5.1–5.4 field volumes with the held field-free plane
(SEP 22 Q15), against the dedicated code's 4.0. The Runge–Kutta and Adams marches hold
more. Memory is not this SEP's problem, except that what it adds is bounded and
accounted (§6.8).

**Other readers of the scratch and the state.**

- `_free_physical` reconstructs physical columns into the transverse scratch for the
  spatial statistics, the collar diagnostic and the split-step absorber.
- The far-field statistics, noise rotation, observation, checkpoint and resume all read
  the canonical state.
- Lockstep batches (`backends/batch.py`, `LockstepRunner`) run members on independent
  workspaces, so any layout is per member.
- `LineGrid` is a `FreeGrid` with one y point, so the line route is the free kind with
  a one-dimensional transverse transform.

**The Adams sample and its history.** On `numba_cpu`, `_adams_lab_source` runs the
Runge–Kutta stage at index 0 with the propagator slot filled with one
(`self._forward[0].fill(1.0)`, likewise the inverse), so the sample also pays the stage
row's clear and the accumulation into both Runge–Kutta accumulators, which the Adams
march does not read; it then returns a `.copy()` of the stage row, an allocation and a
copy of a field volume per evaluation. The history takes its own copies:
`HostHistory.start` writes the startup sample into `_store[0]`;
`HostHistory.exponential_trial` holds the predictor's source in `_ExponentialTrial`, and
`exponential_accept` writes it to `store[free]` and, for dense observation, keeps a
further copy of it and of the samples it used. The interaction formulation's
`_adams_source` on these routes multiplies the lab source by the inverse propagator into
a new array and returns a copy of the forward propagator. Which of these copies the
algorithm needs and which are avoidable depends on the history's ownership across
rejected trials, restarts and dense observation (§6.4). A backend may replace the host
engine through the protected hook `StepDriver._adams_engine(method, compensation)`
(kernel schema 21), as Kokkos does.

**Grids.** An `EnvelopeGrid` has $n_{to}=n_t=n_\omega$ unless it is oversampled for the
third harmonic or a finer `dt` (`EnvelopeGrid.from_limits`). Then, for example, $n_t=2048$
and $n_{to}=4096$. When $n_{to}=n_\omega$ the envelope crop index is the identity.
Field grids always pad: real time samples, and $n_\omega=n_t/2+1$ coarse bins inside a
longer fine grid. The envelope plasma carries two current streams per node
(`EnvelopePlasma.sources`, shape `(2, rows, n_to)`). They are transformed separately and
assembled into the source with per-frequency Drude and loss factors (`assemble`). Raman
convolves on a doubled temporal grid per node.

### 4. Theory

There is no new physics; this section states the operations the design must preserve,
the cost model, and the numerical contract, in the conventions of
[conventions](../../conventions.md) (internal FFT convention, raw DFTs, SI).

#### 4.1 One evaluation

This section describes what the shared core and each stepper's finish compute today;
it changes no normalisation and moves no mask. Let $u$ be the evaluation's input in
$(k_\perp,\omega)$: the stage input (Runge–Kutta), the pass's register (low-storage) or
the lab-frame field (Adams), with the `modified` noise draw added where it applies.
$\mathcal{F}_\perp$ is the two-dimensional (line: one-dimensional) transverse DFT, its
inverse unnormalised and the factor $1/(n_x n_y)$ applied by the input traversal;
$\mathcal{P}_\sigma$ pads the $n_\omega$ coarse bins into the $n_{to}$ fine ones at
`crop_index`, scales by $\sigma$ and, on `field_analytic` grids, doubles the bins
strictly between zero and the fine Nyquist bin; $\mathcal{F}_t$ is the unnormalised
temporal DFT on the fine grid, $\mathcal{C}$ the crop back to the coarse bins.

**Cartesian and line, material term.** With the response $P$ evaluated node by node in
time,

$$
S_\mathrm{m}(\omega,k_\perp) = Q(\omega,k_\perp)\,w(\omega)\,
\mathcal{F}_\perp\,\mathcal{C}\,\mathcal{F}_t\,
P\!\left[\mathcal{F}_t^{-1}\,\mathcal{P}_{\sigma_\mathrm{m}}\,\mathcal{F}_\perp^{-1}u\right],
\qquad \sigma_\mathrm{m}=\frac{\text{pad scale}}{n_{to}},
$$

with $Q$ the prefactor of §3, zero outside the active band, and $w$ the spectral window.

**Cartesian and line, in-stage absorber term.** The absorber's term carries neither $Q$
nor $w$: it is the hard active-band mask $M(\omega)\in\{0,1\}$ times the transformed
product of the field with the rate $r(x,y,t)=-\tfrac12\big(\gamma_\perp(x,y)+\gamma_t(t)\big)$
of the transverse collar and the temporal absorber (`absorber_columns` on both backends;
on field grids $r$ multiplies the real part of the samples),

$$
S_\mathrm{a}(\omega,k_\perp) = M(\omega)\,
\mathcal{F}_\perp\,\mathcal{C}\,\mathcal{F}_t\!\left[r\,
\mathcal{F}_t^{-1}\,\mathcal{P}_{1/n_{to}}\,\mathcal{F}_\perp^{-1}u\right].
$$

`make_free_kernels(...).fold` and Kokkos's `low_storage_separable_fold` select the two
by `term` (`_TERM_MATERIAL`, `_TERM_ABSORBER`), and the Runge–Kutta finishes do the same.
On the Cartesian and line routes the two terms are separate passes of the core, each
gathering its own input. Any fusion must keep the two contributions distinct.

**Responses with more than one output.** The envelope plasma adds two current streams
$J_1,J_2$ per node, each transformed separately and assembled after the crop with
per-frequency factors, $\mathcal{C}\mathcal{F}_tP + d(\omega)\,\mathcal{C}\mathcal{F}_tJ_1
+ \ell(\omega)\,\mathcal{C}\mathcal{F}_tJ_2$ (`assemble`, with its masks). SEP 6's
dispersive response filters the padded spectrum before the temporal inverse
(`SpatialDispersive.prepare`, since the inverse may destroy its input), transforms the
filtered fields, forms its products at the nodes and either adds them to the
polarisation before the stage's forward transform (the borrowed path) or transforms its
own products and adds their spectrum (`combine`); it runs beside the response column or
alone in its place. Raman convolves each node's history on a doubled temporal grid in
buffers of its own.

**The stepper's frame.** Each stepper applies its own propagator after this core: the
Runge–Kutta finish multiplies by the stage's inverse propagator and accumulates, the
low-storage fold combines the source with the flow $\exp(hD/2)$ into its registers, and
the Adams sample returns $S_\mathrm{m}+S_\mathrm{a}$ untransported (exponential
formulation) or carried to the anchor (interaction formulation).

**Planar.** The state's columns are $(p,k_y)$, $n_p$ slab channels per $k_y$. The core
takes them to $(p,y)$ rows by the inverse y transform and processes blocks of $b$ whole
y rows. Each block's $b\,n_p\,n_\mathrm{pol}$ rows are padded with $1/n_{to}$ and taken
to time once. The **material** term synthesises the $b\,n_x$ physical nodes in time with
the real slab matrix $X$ ($n_x\times n_p$, one GEMM per y row, carrying the temporal pad
scale: `_slab_product` on `numba_cpu`, `separable_nodes`' `slab_contract` on Kokkos),
evaluates the response at the nodes, and projects back onto the rows with $\Pi$
($n_p\times n_x$; `_slab_material`, `slab_project`); the envelope plasma's streams are
projected the same way. The **absorber** term acts on the rows themselves, because the
collar's rate depends on y alone, and needs the block's time samples after the material
response has run. Both return to frequency and are cropped into separate halves of the
column volume, one forward y transform takes both, and the finish forms each column's
material term with the slab element's prefactor and its absorber term with the mask.
So a planar block has temporal buffers of $b\,n_p\,n_\mathrm{pol}$ rows and response
buffers of $b\,n_x\,n_\mathrm{pol}$ rows: the synthesis and projection are arithmetic
of the route, not removable copies.

Let $N = n_x n_y n_\mathrm{pol} n_\omega$ be the elements of one state and
$F = 16N$ bytes a field-sized volume (`complex128`).

#### 4.2 Reference quantities

A complex DFT of $N$ points has a conventional cost of $5N\log_2 N$ floating-point
operations, a scaling factor for comparing transform speeds rather than an exact
operation count [3]. The cost of $N\log N$ work grows from $N_1$ to $N_2$ by

$$
r_{N\log N} = \frac{N_2\log N_2}{N_1\log N_1}.
$$

From $384^2\times600$ ($N_1 = 8.85\times10^7$) to $768^2\times600$
($N_2=3.54\times10^8$) that is $4\times\log N_2/\log N_1 = 4.303$; a purely
memory-bound pass grows by 4. Today's measured ratio is about 13 for both backends
(§2), so the growth beyond $N\log N$ is a factor of about 3. $1.2\,r_{N\log N}=5.16$,
written 5.2 below, is a target derived from this scaling model, not a hardware bound.

A **per-evaluation reference** is defined only where a comparable one exists: the
unpadded Cartesian envelope case ($n_{to}=n_\omega$, one polarisation, the Kerr beam of
§7.3). Phase 0 adds the script that measures it, independent of any candidate
implementation, and records its definition beside the measurement:

- **Transform part $T_\mathrm{3D}$.** One `complex128` array of shape
  $(n_\omega n_\mathrm{pol}, n_y, n_x)$, C order, 64-byte aligned, holding the case's
  launch field; a rank-3 complex DFT over all three axes, forward then backward, in
  place and unnormalised, through the backend's own library (pyFFTW's FFTW on
  `numba_cpu`, Kokkos-FFT on Kokkos CPU), with the run's plan flag, thread count and
  placement. Planning and two untimed pairs first; then the median and MAD of at least
  15 pairs in one process, timing the executions only.
- **Response part $T_P$.** The case's response, from phase 0's per-pass timers inside
  the evaluation.
- **Elementwise part.** $n_L\,t_\mathrm{pass}$, with $n_L$ the stepper's logical
  volume-passes per evaluation outside the transforms (§4.3, from the audit, each fused
  pass counted once) and $t_\mathrm{pass}$ the measured time of one streaming
  volume-pass: a scaled copy between two field volumes first touched in parallel, on
  the same threads and placement, median of 15, divided by its two passes.

The reference is $T_\mathrm{ref}=T_\mathrm{3D}+T_P+n_L\,t_\mathrm{pass}$, a roofline [6]
for this evaluation built from measured parts. It is compared
with the **measured time per evaluation**: a ten-step window's time divided by its
source calls, on windows with no observation plane inside and statistics-only output.
The window includes the input, the transforms, the temporal stage, the response, the
folds or finish and the flow (tabulated or recomputed); it excludes translation,
construction, observation, checkpoints and statistics. The target ratio (Q4) is
provisional at 1.5 until phase 0 has measured both sides; the final value is agreed at
the checkpoint after phase 0, before any optimisation starts (§11).

Padded field grids, Raman's doubled grid, the plasma's extra streams and the planar
products are not the work of one coarse three-dimensional pair, so those cases have no
such reference: they are judged by measured A/B ratios against phase 0's baseline
(§7.2), with their per-pass profiles reported.

#### 4.3 Logical volume-passes

Count a **logical volume-pass** as one read or one write of one field-sized array over
its whole extent, as the source code states it. The low-storage march with a tabulated
flow and the stage register elided (the default on the Cartesian and line routes with a
material response and no in-stage collar) does, per step, as built
(`make_free_kernels`, `low_storage_separable_fold`):

| pass | reads | writes | count |
|---|---|---|---|
| input (pass 1 only) | $A$ | scratch | 2 |
| fold 1 | scratch $S$, table, $A$ | $A$, $B$, scratch (next input) | 6 |
| fold 2 | $S$, $A$, $B$ | $B$, scratch | 5 |
| fold 3 | $S$, $A$, $B$, table | $B$, scratch | 6 |
| fold 4 | $S$, $A$ (unneeded), $B$, table | $A$ | 5 |

That is 24 logical volume-passes a step, 6 per evaluation, outside the transforms and
the chunk loop. Removing the unneeded read gives 23. Recomputing the flow instead of
reading the table removes three more reads at the cost of a complex exponential per
element. The Runge–Kutta stage reads the state, every earlier stage row its Butcher row
names, the propagator pair, the scratch and both accumulators, and writes the scratch,
its stage row and both accumulators; both backends also clear the stage row first
(`stage[:] = 0`, `free.clear_stage`). For IP-DOPRI5's full lower-triangular matrix that
grows with the stage index. The Adams sample on `numba_cpu` adds the identity fills, the
stage clear, the unused accumulation and the returned copy of §3. The exact counts per
stepper and backend are audited in phase 0.

Logical passes are not measured traffic. Write-allocation, reuse of a pass's operands
in cache, a compiler dropping a load, and per-column reductions all change the bytes
that reach memory. §6.4's target is stated in logical passes, a property of the code
that review and the audit can check; what is accepted is elapsed time (§7.3). Phase 0
reports measured traffic beside it where the host exposes memory-controller counters,
and otherwise infers it from time against $t_\mathrm{pass}$, labelled as an inference.

These counts are only the floor if every pass is contiguous in every operand. A pass
that crosses layouts element by element touches, for each column, $n_\omega$ cache lines
$n_x n_y n_\mathrm{pol}$ elements apart in the scratch. Whether those lines are reused by
the neighbouring column depends on $n_\omega$ lines (and as many pages) staying resident
between columns. The stride grows with the transverse grid, so the cost per element of
such a pass can grow with $n_x n_y$. That is one candidate mechanism for growth beyond
$N\log N$; page placement, TLB reach, the transforms' decomposition and per-chunk
synchronisation are others, and phase 0 separates them (§11). A cache-blocked crossing,
which moves a tile of $b_c$ columns by $b_\omega$ bins through cache with both extents
chosen to fit, costs one read and one write of each operand whatever the stride [4].

#### 4.4 Tiles, groups and cache domains

The temporal stage's schedule uses five separate quantities:

- **node tile**, $m$ nodes whose rows one thread owns from the gather to the scatter;
- **group**, the $g$ nodes one host iteration of the temporal stage processes;
- **concurrent threads** $n_\mathrm{thr}$, the threads the backend's parallel region uses
  after the process's CPU affinity (on `numba_cpu` numba's thread count, on Kokkos the
  host execution space's concurrency);
- **cache domain**, a set of cores sharing one cache level: a core's private levels, or
  a last-level cache shared by a core complex or a socket;
- **live group buffers** $L(g)$, every buffer allocated for one group at once: fine
  spectrum, time samples, the node and polarisation buffers where kept, the projection
  and its spectrum, response-specific storage (Raman's doubled grid, the plasma
  streams, SEP 6's filtered spectra and products) and the plans' workspace.

In the response path a tile's hot working set is about $16\,c\,m\,n_\mathrm{pol}\,n_{to}$
bytes with $c\approx3$ (fine spectrum, time samples, response output); the Raman
histories and the plasma streams add to $c$. Two rules bound the schedule:

1. **The cap.** $L(g)\le$ `node_budget` always, as the chunk buffers are bounded now.
   When the budget admits fewer than one tile per thread, the group shrinks, down to one
   node (one whole y row on the planar kind).
2. **Residency.** A tile runs from cache only if its hot set fits the cache share of the
   thread that owns it: $m \approx \text{share}/(16\,c\,n_\mathrm{pol}\,n_{to})$.

Under strategy (ii) of §6.3 each thread owns its tile through every operation, so rule 2
applies per thread and per private cache share. Under strategy (i) it does not: FFTW's
pool, numba's `prange` and Kokkos's host space partition a group independently, and
nothing makes the thread that transforms a row the one that responds it. Summing private
caches is then not enough; the whole group's hot set must fit the shared cache of the
domain its threads span, less interference from other traffic. At $n_{to}=600$, one
polarisation and $c=3$ a node needs about 29 kB, so a 32 MB last-level cache holds a
group of about a thousand nodes. Thirty-two threads on one Zen 3 socket span four or
more such caches, depending on the part, none of which holds the whole group, so there
(i) may save host iterations without saving traffic.

**What follows if residency holds.** The temporal stage then moves only the gather's
read of the scratch and the scatter's write back, one logical pass each over the volume:
the traffic of one in-cache axis pass of a three-dimensional transform, for padded and
unpadded grids alike, since the pad is written into the tile while it is in cache. This
is conditional: phase 0 measures the temporal stage's traffic (or its time against
$t_\mathrm{pass}$) as a function of group size under (i), and §6.3 states when (ii) is
tried instead.

#### 4.5 The numerical contract

The proposal changes how bytes move, not which operations are performed. Blocking,
tiling, reordering a fold's operations or changing an FFT's decomposition changes each
discrete operation by rounding only. SEP 25 then gives two kinds of comparison, and this
SEP uses both:

- **Matched discrete computations** — one source evaluation, one step, and short
  prescribed schedules with a fixed-order method and matched startup, restarts, noise
  arrays and observation policy (SEP 25 §4.3) — are compared with a **rounding budget**
  fixed per test before measurement, from operation and reduction counts and a stated
  margin (`testing.md` §3).
- **Adaptive runs** (the Runge–Kutta adaptive march, the adaptive Adams march),
  variable-order Adams replay and resumed-against-uninterrupted runs may take different
  steps, rejections or orders after a rounding change. They are compared by a per-case
  **accuracy envelope** against a tighter-tolerance or stored converged reference, with
  a safety factor fixed before the candidate runs (SEP 25 §4.4). Source calls, steps and
  rejections are recorded as work, not required to match.

The budget for the matched kind is normwise, not per bin. A DFT's computed result
satisfies $\lVert\hat y-y\rVert_2\le c\,u\log_2 n\,\lVert y\rVert_2$ for a modest
constant [5]; a bin that cancels to nearly zero has only that absolute error, so the
tests use absolute per-bin bounds scaled by the band's norm, with relative figures
reported only above a stated floor. The budget also carries the response's sensitivity
to its input, the propagation's amplification over the compared window (SEP 25 §4.3),
and the prefactor's amplification near cutoff, where $|Q|=|N|/\mathrm{Re}\,k_z$ is
largest among propagating bins: an error in the transformed polarisation reaches the
source multiplied by $|Q\,w|$.

**Exact structural assertions stay exact.** Bins outside the active band are exactly zero
in both terms (the hard mask); metadata that records identity or layout
(`transverse_layout`, `node_chunk`, the tile record) is compared exactly; a ledger counts
the source calls actually made and a prescribed schedule executes exactly its steps.
No path keeps or acquires a bitwise requirement beyond these.

### 5. Proposed interface

None for users: no public name, parameter or default changes. Results stay within the
numerical contract of §4.5: matched discrete computations within rounding budgets,
adaptive and resumed runs within accuracy envelopes.

- **The node budget.** `SimulationSettings.node_budget` keeps its meaning, a memory cap
  for node buffers, now stated as a bound on all simultaneously live group buffers
  (§4.4). The backend chooses its tile and group within it (§6.3). The kernel contract's
  `node_chunk` row is clarified to "the largest node chunk the response pass may use;
  the planar kind takes at least one whole y row, `max(1, node_chunk // n_x)` rows",
  which states the existing semantics of every backend and the oracle and changes no
  validation.
- **Metadata.** Each compiled backend's `metadata()` records the tile and group it chose
  (nodes, threads, and the cache size and source it was derived from) beside the
  existing `node_chunk` and `transverse_layout`.
- **Overrides.** For measurement only, an environment override sets the per-thread
  tile's hot working set, in the style of the existing `SOLPROP_*_THREADS` overrides:

```text
SOLPROP_TILE_BYTES=4194304 uv run python run.py   # one thread's tile hot set, bytes
```

  The cap wins: if the group this implies at the run's thread count would exceed
  `node_budget`, construction raises an error naming both values, rather than clipping
  silently. At the default 64 MiB and 32 threads the example's 4 MiB tiles need
  128 MiB for their hot sets alone and are refused.

### 6. Implementation strategies

#### 6.1 Layers

The work lives in the backends, `numba_cpu` in solprop and the Kokkos CPU build in
solprop-kokkos ([architecture](../../architecture.md) §3). Translation changes only the
documented meaning of `node_chunk`. Physics and `numpy_ref` do not change. The kernel
contract keeps its fields and its protected `StepDriver` surface; under the
recommendation `SCHEMA_VERSION` does not change (§9). Any later change to a protected
hook or a stage bundle (for example a destination argument for the Adams sample, Q7)
would bump it.

#### 6.2 Item 1: the transverse transform layout

Three options, each analysed against every reader of the state and the scratch.

**(a) One in-place three-dimensional transform of the whole scratch.** When
$n_{to}=n_\omega$ (an envelope grid not oversampled), the temporal transform needs no
pad. Transforming the scratch in place along all three axes then replaces the
transverse pair and the chunk loop. The response runs on the whole volume in time.

**(b) Transform-ordered registers.** Move the march's registers, and with them the state
the backend holds, to the scratch's order (ω slowest), so that input and fold are
contiguous in every operand. The canonical layout is formed only at the seam.

**(c) Keep both layouts and block every crossing.** Keep the canonical registers and the
ω-slowest scratch both backends use. Make the input, gather, scatter and fold
cache-blocked, and run the temporal stage over node tiles (§6.3).

| reader or constraint | (a) whole-volume 3-D | (b) transform-ordered registers | (c) blocked crossings, tiles |
|---|---|---|---|
| padded grids (every field grid, oversampled envelope) | not applicable: needs a second path | unchanged temporal stage | the tile writes the pad in cache |
| envelope plasma (two streams, per-bin factors) | two more volumes, or chunks again | unchanged | tile-sized streams, as now |
| Raman (doubled grid per node) | chunks again | unchanged | tile-sized |
| SEP 6 dispersive Kerr | frequency products on the volume; possible | unchanged | unchanged; its preserved inputs kept (§6.3) |
| in-stage absorber's term | per-node rate on the volume; possible | unchanged | unchanged; a separate term (§4.1) |
| statistics, collar diagnostic, split-step absorber (`_free_physical`) | unchanged | must read the new order or transpose into the scratch | unchanged on envelope grids; re-tested where `numba_cpu`'s field-grid scratch changes layout |
| observation, checkpoint, resume, noise, far field | unchanged | transposed copy at each field plane (through the scratch, free between steps); the noise draw stored in the new order | unchanged |
| Runge–Kutta and Adams | unchanged | every stage row, accumulator, propagator and history in the new order, or the march alone gains | unchanged; every stepper gains through the shared core |
| lockstep batches | per member | per member | per member |
| planar hybrid (y-transform only) | not applicable | its own order | blocked gather and scatter on its `(p, y)` rows; the synthesis and projection GEMMs remain |
| line grids | trivially 2-D | as Cartesian | as Cartesian |
| kernel contract, schema | none | none if private to the backend; a contract change if the state's layout changes | none (`node_chunk` clarified) |
| Kokkos-FFT on CPU and devices | a rank-3 plan (Kokkos-FFT 2.0.0 allows up to three axes); cuFFT and rocFFT plan it natively | as today | as today; a device tile can be every node |
| memory | none added; frees the chunk buffers | none added if the plane copy uses the scratch | bounded tile and staging buffers (§6.8) |

**Recommendation: (c).** It is the only option that serves every grid, response and
stepper with one design. It touches no reader outside the evaluation except where
`numba_cpu`'s field-grid scratch changes layout, needs no schema change, and addresses
the strided-crossing mechanism of §4.3 if phase 0 confirms that it matters. (a) applies
only to unpadded envelope grids whose responses run in place on the volume; the envelope
plasma and Raman are not among them, and oversampled grids (third harmonic) are
excluded. It duplicates no response code, but it would leave every padded case on the
path (c) improves anyway, so it is kept as a later backend policy (§10). (b) removes the
crossings outright rather than blocking them. Its cost is a second register layout
through every stepper and reader listed above. It is kept as the escalation if phase 2's
measurement misses the scaling target after (c), and only for every stepper (Q2).

On field grids `numba_cpu` keeps the column layout for its scratch. Under (c) it moves
to the ω-slowest layout that envelope grids and Kokkos already use. The padded temporal
stage then reads the scratch only through the gather and the scatter, so nothing favours
the column layout there. Phase 0 checks this on the filament case first, and §7.1
re-tests the scratch's other readers on field grids.

#### 6.3 Item 2: the tile path and its buffer ownership

Per group (§4.4), on both backends:

1. **Gather.** One fused, blocked kernel reads the group's columns from the scratch and
   writes the fine spectrum: coarse bins at `crop_index` with the scale and the
   analytic-signal doubling of §4.1 folded in, zeros in the pad bins, and zeros in every
   row a fixed-width plan reads beyond a partial group's own (or that group runs an
   actual-width tail plan). The whole-buffer `deep_copy` on Kokkos and the full zero
   fill in `pack_columns` become exactly these writes.
2. **To time.** SEP 6's filter reads the fine spectrum first, as now; then the inverse,
   which may destroy its input.
3. **Response.** On Cartesian and line grids the response reads the time samples where
   they lie and writes its output into the buffer the forward transform reads, so the
   node copy (`modal_time → nodes`) and the polarisation copy
   (`polarisation → projected`) go, wherever the ownership table below allows. On the
   planar route the synthesis, the response and the projection stay as §4.1 states.
4. **To frequency**, then **scatter**: SEP 6's `combine` and the envelope plasma's
   `assemble` add into the projected spectrum, and one fused, blocked kernel crops it
   into the scratch.

The chunked path does not disappear: it is this path with the group set by the node
budget instead of the cache, for memory-limited cases, devices (§6.7) and planar blocks
of whole y rows. With $n_{to}=n_\omega$ the gather and scatter become a pure blocked
transpose with scale.

**Buffer ownership.** Removing a clear or a copy is allowed only where this table says
so. It holds for each admitted transform family: `field_real` (complex fine spectrum of
$n_{to}/2+1$ bins, a complex-to-real inverse that may destroy its input, real time
samples, polarisation and projection, a real-to-complex forward); `field_analytic`
(complex throughout, the gather writes the doubled bins); envelope (complex throughout).

| buffer, per group | first writer | last reader | may alias | write rule |
|---|---|---|---|---|
| fine spectrum | the gather, every element any plan reads | SEP 6's filter, then the temporal inverse | none | overwrite; may be destroyed by the inverse |
| time samples | the temporal inverse | the response (Cartesian) or the synthesis (planar); on the planar route, the absorber term after the material response | the node buffer, when the response does not write its input | read-only while a later reader remains |
| node buffer | planar: the synthesis; Cartesian: none, unless the response writes its input (SEP 6's lone symmetric response) | the response | the time samples, as above | overwrite |
| polarisation | the response | the forward transform (Cartesian, as the projection) or the projection (planar) | the projection, Cartesian only | overwrite every node row |
| projection (forward input) | the response, the projection or the absorber term; the rows beyond a partial group zeroed, or a tail plan | the temporal forward | the polarisation, Cartesian only | overwrite every element the plan reads |
| projected spectrum | the temporal forward | the scatter | none | SEP 6's `combine` and the plasma's `assemble` accumulate into it |
| plasma streams | the envelope response | their own forward transforms | none | overwrite; transformed separately |
| Raman histories; SEP 6's filtered spectra, fields and products | their response | their response or `combine` | none | as today |

Three rules follow. Every element a plan reads is written in the same group before the
plan executes, or the tail runs an actual-width plan. Both plan shapes (full group and
tail) are created, and their buffers allocated within $L(g)$, at construction, before the
first step. A buffer a later operation reads — the planar time samples before the
absorber term, a response's preserved input for Raman, the plasma or SEP 6's products —
is never the target of an earlier write in the same group. The `numba_cpu` copy of the
plasma streams into their transform buffer is kept unless phase 1 shows the streams can
be transformed where they lie under these rules.

**The host loop.** At $768^2$ a cache-sized group means hundreds to thousands of host
iterations per evaluation. They cannot each pay a Python call and a parallel launch on
`numba_cpu` or several launches on Kokkos. Two strategies:

- **(i) Threaded groups.** One host iteration processes one group; its batched plan is
  threaded as now and its kernels run in parallel. This keeps today's structure and
  needs no new mechanism; the iterations drop by the group's size. It saves traffic only
  where §4.4's shared-cache residency holds, which phase 0 measures.
- **(ii) Tile-parallel loop.** One parallel loop over tiles, each thread running a serial
  plan on its own tile buffers, so each thread owns its tile through every operation. On
  `numba_cpu` a `prange` body cannot call pyFFTW, so this needs FFTW's new-array execute
  called from compiled code through the library pyFFTW loads: a new binding. On Kokkos
  it needs either Kokkos-FFT executions issued from a host parallel region, which the
  installed version must be shown to allow, or FFTW called directly from solprop-kokkos's
  CPU build.

Recommend (i) first on both backends. (ii) is tried only if phase 0 or phase 2 shows,
at either shape, that the temporal stage under (i) moves more than three logical
volume-passes per evaluation (the gather's read, the scatter's write and one pass's
allowance, measured or inferred as §4.3 states), or that synchronisation between groups
exceeds a tenth of the evaluation. Before (ii) is chosen, a feasibility probe must
establish, on each supported `numba_cpu` platform and the Kokkos CPU platforms: the
FFTW library and ABI that pyFFTW actually loads (bundled in a wheel or a local build) and
its exported new-array execute; plan lifetime and aligned buffers satisfying the
new-array contract (shape, strides, in-place relation, alignment); serial planning at
construction, with tail plans; the concurrency mechanism and its nested pool limits (no
threaded FFTW plan inside a parallel loop); and, on Kokkos, the installed Kokkos-FFT's
execution and temporary-storage behaviour under concurrent calls, rather than assuming
reentrancy. Where a platform does not expose what the probe needs, that platform keeps
(i). This SEP does not design (ii) further; the binding itself is a maintainer decision
(Q3).

#### 6.4 Item 3: the register folds and the stage combinations

**Target.** Each evaluation moves no more logical volume-passes than its algorithm needs
(§4.3), and every one is contiguous in every operand or cache-blocked:

- **The low-storage march**: at most 5.75 logical volume-passes per evaluation outside
  the transforms and the tile loop, tabulated (23 a step, the unneeded read in pass 4
  gone), and 5 recomputing.
- **The Runge–Kutta stage**: each operand of its Butcher row read once, the propagator
  pair formed or read once per stage, the stage-row clear dropped where the finish
  overwrites the row, and the finish and accumulation fused where SEP 25 has not
  already fused them. Phase 0's audit gives the per-stage count for IP-DOPRI5, and the
  target is that count with no crossing unblocked.
- **The Adams sample**: the shared core's gains, plus removal of the passes that are
  private to the backend and read by nothing: on `numba_cpu`, the accumulation into the
  Runge–Kutta accumulators that the zero-length stage performs for the Adams march, and
  the identity fills where the slot already holds the identity. The returned copy and
  the history's own copies stay. Removing them needs an ownership design across rejected
  trials, startup and restarts, corrected-field evaluations and dense observation (§3),
  for both formulations: either a `numba_cpu` Adams engine through the existing
  `_adams_engine` hook (no schema change), or a destination argument to
  `_adams_lab_source` (a protected-surface change and a schema bump). That is a later
  item, not part of this SEP's phases (Q7).

**Per-element arithmetic.** The prefactor $Q$ cannot be read from a precomputed volume
without adding one (§6.8). Nor is it separable in $\omega$ and $k_\perp$, since $k_z$
couples them, and interpolating it would change results beyond rounding. So the target
is one complex root per element per evaluation, shared by the rate and the prefactor as
`spatial_element` already shares it, and no more. Phase 0 measures whether the fold is
bandwidth- or arithmetic-bound on the quiet node. If it is arithmetic-bound, a real
square root may replace the complex one, under these conditions only:

- the longitudinal square is formed exactly as now, from both its high and low parts,
  before any test;
- the shortcut applies only when that compensated square has an imaginary part exactly
  zero and a real part strictly positive; every other element, including any with a
  small nonzero imaginary part, keeps the complex path;
- the propagation, attenuation, evanescent clamp and active-band mask rules of
  `longitudinal_rate` are unchanged, and the clamped branch never takes the shortcut;
- no threshold defining "lossless" is introduced.

For the flow, tabulated against recomputed stays the existing `propagator_mode` choice;
the measurement records its cost at both shapes.

**Both backends.** On `numba_cpu` the fold becomes a blocked kernel: an outer `prange`
over tiles of columns, an inner loop over ω-blocks, and the scratch's block staged
through a small per-thread buffer, or read along its contiguous axis with the
registers' block held in cache. On Kokkos CPU the same tiling is expressed as a team or
`MDRangePolicy` with tile extents. Device builds keep their two-dimensional launch, whose
coalescing already favours the scratch's contiguous axis (§6.7).

#### 6.5 Item 4: thread scaling on CPU

- **`numba_cpu`'s FFTW pools.** The `FFTW_MAX_THREADS = 4` cap applies by default to the
  temporal and transverse plans of Cartesian volumes, two to four orders of magnitude
  larger than the fibre transforms it was measured on. Measure the transverse plan and
  the group's plans against thread count at both shapes, and give the free kinds a size
  rule of their own: a constant in `fft.py` with its measurement in the docstring, as
  the existing ones have. This changes runs with the default settings; it does not
  explain the explicitly uncapped comparison of §2.
- **Serial sections.** The Python chunk loop (§6.3), the host-side clears, and every
  non-`prange` traversal under the "small" regime when the volume is large. The regime
  is read off `_regime_elements`; check that every Cartesian volume above a few million
  elements runs every traversal in parallel.
- **`prange` granularity.** Outer loops over tiles of columns, not single columns, so
  that a thread's iterations share cache lines in the scratch and no two threads write
  the same line.
- **Kokkos policies.** `cartesian_for`'s one-task-per-row CPU policy is replaced, for
  the crossing traversals, by a tiled policy (Kokkos's multidimensional range and team
  policies [7]). Kokkos launches per group stay few.
- **Page placement.** Record whether each field volume is first touched in parallel, so
  that its pages spread across the NUMA domains the threads run on. Measure; NUMA-aware
  executors are out of scope.

#### 6.6 Item 5 (optional): planning effort and wisdom

On the quiet node, time `FFTW_ESTIMATE`, `FFTW_MEASURE` and `FFTW_PATIENT` plans
(and imported wisdom, `SOLPROP_FFTW_WISDOM`) for the transverse plan and the group plans
at both shapes, and record planning time beside execution time. Kokkos-FFT's host plans
fix `FFTW_ESTIMATE`; a planning option there is a solprop-kokkos change and only worth
making if the `numba_cpu` measurement shows a gain beyond the A/A spread. This item
changes no default by itself; any default it motivates goes through SEP 4.

#### 6.7 Devices (later, not precluded)

On CUDA and HIP the scratch's ω-slowest layout already gives the batched two-dimensional
transform its trailing axes. A device's group is the whole node set when memory allows
(the roadmap's device-sized node chunk). The fused gather and scatter of §6.3 serve
there unchanged, and the blocking in §6.4 is a CPU policy behind the same kernels. (a)'s
rank-3 plan remains available to a device phase if it measures better there. Device
qualification of the march and the device tile rule are a separate item after this SEP,
but phase 0 takes one **device baseline** first (§11): the CPU measurements of §2 do not
say where a device stands, and that position decides whether (a) matters there.

#### 6.8 Memory budget

The budget is relative to a **matched configuration**: the same backend, stepper, route,
shape, responses, flags and thread count before the change. It is stated so that it can
be tested:

- **Total added bytes.** Outside the group buffers, which replace today's chunk buffers
  under the same cap, the sum over every unique backing allocation the change adds
  (each counted once whatever views alias it) stays below one field volume, and its
  rows are named in the backend's memory ledger (`backends/memory.py` on `numba_cpu`,
  the native ledger on Kokkos). This closes the loophole of several smaller
  allocations adding up to a volume.
- **Group buffers and plan workspace** are counted at their simultaneous peak:
  $L(g)\le$ `node_budget` for the largest group and its tail plan together, with the
  plans' workspace included.
- **Per-thread and team staging** for the blocked crossings is real memory even on the
  stack or in team scratch: it is bounded explicitly, at most $n_\mathrm{thr}$ times
  one staging block of at most 256 KiB, and counted in the ledger.
- **Removed allocations** (none is promised for the Adams sample; §6.4) are recorded as
  removals, not offset against additions.

Resident set size is reported separately, as a measurement: the peak during the timed
steps, after construction, with the construction peak noted and the reference script's
own buffers excluded. The 5.1–5.4 field volumes of §3 is the baseline only for the
Kokkos low-storage march on the Cartesian envelope Kerr beam with the held field-free
plane; every other configuration's baseline is its own phase-0 measurement.

### 7. Verification and validation

#### 7.1 Numerical verification

Every changed path is checked against `numpy_ref` and against its own backend before the
change, under the two kinds of §4.5, with every bound fixed before measurement.

**Matched discrete comparisons, rounding budgets**, from the smallest unit up, so that a
failure is localised before any ten-step run:

- one source evaluation (material, absorber, and both) on Cartesian, line and planar
  grids, for each stepper's entry (the Runge–Kutta stage, the low-storage pass, both
  Adams samples);
- one step of each stepper, then ten steps on a prescribed schedule with a fixed-order
  method and matched startup, noise and observation;
- for each response family on a Cartesian grid — Kerr, Raman, the field-grid plasma,
  the envelope plasma, SEP 6's dispersive Kerr alone and beside the response column, and
  the in-stage absorber — the same ladder;
- groups of 1 node, of a size that leaves a partial last group, and of every node; a
  budget smaller than one proposed group, and on the planar route smaller than one y
  row;
- an oversampled envelope grid (third harmonic) and both field families, the padded
  paths;
- `modified` noise on the march, whose input carries the draw;
- the existing Cartesian, line and planar parity tests of each route and stepper, and
  the low-storage march's tests (`tests/backends/test_low_storage.py` and the backend
  modules beside it).

**Buffer-lifetime tests** (§6.3): fill every reusable buffer with NaN before evaluating
full and partial groups repeatedly, then check the padding bins, the analytic doubling's
endpoints, the scale, finiteness and the final source. Cover two polarisations,
material plus absorber, SEP 6's response alone and combined with ordinary responses,
the plasma streams, and the `field_real`, `field_analytic` and envelope families. These
catch stale data and aliasing that a smooth ten-step comparison can miss.

**Real-root tests** (only if §6.4's shortcut is taken): elements just below, at and above
cutoff, small nonzero loss, inactive bins and the supported evanescent rates, against
the existing `longitudinal_rate` and the reference, with condition-aware absolute
bounds and finite outputs required.

**Accuracy envelopes**: short adaptive Runge–Kutta and adaptive Adams windows (both
formulations), variable-order Adams replay and a resume against an uninterrupted run,
each against a tighter-tolerance or stored converged reference with a per-case envelope,
recording steps, rejections and source calls without requiring them to match.

**Exact assertions**: hard-mask zeros, the metadata records, the ledger's call counts and
prescribed schedules (§4.5).

**Readers of a changed layout**: where `numba_cpu`'s field-grid scratch changes layout,
the spatial statistics, the collar diagnostic, the split-step absorber, observation and
checkpoint-resume are exercised on a field grid.

**Memory**: the ledger tests assert §6.8 — alias accounting, the staging bound's scaling
with thread count, and the cap at budgets smaller than one group or one planar row.

#### 7.2 Benchmark cases and the acceptance rule

The benchmark set includes, beside the Cartesian envelope Kerr beam:

- **a non-Kerr Cartesian case**: the Cartesian filament of `benchmarks/sep16/free_space.py`
  (`cartesian_filament_large`: argon, Kerr and ionisation, a field grid, $128^2$),
  and the same physics on a larger grid named as a new case; it exercises the padded
  path and the plasma column;
- **a line case**: `line` from the same module;
- **the planar control** (`planar`), for the y-row blocking.

**One timing rule**, the [benchmarks](../../benchmarks.md) §4 aggregate rule, applies to
every phase and to completion: over the cases declared in §7.3 before timing, the
equal-weight geometric mean of per-case ratios, taken separately for the production
thread policy and for one thread; the change is adopted when the production saving is
above 5 % and the geometric mean of the conservative per-case ratios
`(candidate median + 2 MAD) / (before median − 2 MAD)` is below one. Small regressions
on individual cases, or at one thread, do not veto a phase; they stay visible in the
complete per-case table, reported with each case's A/A spread. Because an equal-weight
mean can hide one route's loss, each phase report names every case whose production-shape
ratio is worse than its A/A spread and says why; that is a reporting duty, not a veto. Correctness (§7.1) and
memory (§6.8) are independent of the timing aggregate and are never traded against it.

#### 7.3 Performance acceptance

On **one quiet Linux node**, exclusive, both sockets' cores stated, with thread counts,
placement (`OMP_PLACES`, `OMP_PROC_BIND`) and plan flags recorded per configuration, and
alternating fresh-process A/B pairs with the A/A spread reported (benchmarks §5).
Ten-step windows, never full-length runs, and no window is extrapolated.

**The case matrix**, declared here and fixed in phase 0's ledger entry before any
candidate is timed; a case that does not fit the node's memory at a shape is replaced by
a named smaller shape, never by an infeasible allocation:

| case | steppers | shapes |
|---|---|---|
| Cartesian envelope Kerr beam | low-storage march | $384^2$ and $768^2\times600$ |
| Cartesian envelope Kerr beam | IP-DOPRI5 replay | $384^2$; $768^2$ where it fits |
| Cartesian envelope Kerr beam | adaptive IP-DOPRI5, both `Multistep()` formulations | a named smaller shape at which their registers and history fit, at matched accuracy |
| Cartesian filament (field grid, plasma), both sizes | the default stepper and the low-storage march | as defined in §7.2 |
| `line` | the default stepper and the low-storage march | as defined |
| `planar` | the default stepper and the low-storage march | as defined |

Each row is timed on both compiled backends, at the production thread policy and at one
thread, recording end-to-end time, source calls and time per call.

**Targets and which stepper they apply to**:

- **Aggregate** (every row): §7.2's rule.
- **Scaling headline** (the low-storage march, each backend, one socket): the candidate's
  per-step time at $768^2$ at most $1.2\,r_{N\log N}=5.2$ times the smaller of the
  candidate's and the baseline's per-step times at $384^2$, against about 13 today.
  Taking the smaller of the two means that slowing the smaller case cannot satisfy it.
  The candidate's own ratio is reported beside it. For the other steppers the per-call
  time's ratio between the two shapes is reported where both fit, without a target.
- **Per-evaluation reference** (the unpadded Kerr beam, low-storage march and
  IP-DOPRI5): time per evaluation within the ratio to $T_\mathrm{ref}$ (§4.2) agreed at
  the checkpoint after phase 0; 1.5 is provisional (Q4).
- **Bandwidth sensitivity**: the one-NUMA-domain configuration at $384^2$, reported,
  with no target.
- **Memory**: the ledger and resident set at both shapes, against §6.8.

**Phase 0's baseline** is recorded first, as a ledger entry with the case definitions,
source revisions, plan flags, placement and raw per-repeat timings, so that every later
phase is a measured ratio against it and can be reproduced.

The Apple workstation's timings are used only as same-process A/B ratios, labelled; the
loaded host is not an acceptance host.

### 8. Documentation

- [backends](../../backends.md): the scratch layouts, the tile and group rule, the buffer
  ownership table, the blocked crossings and the free kinds' thread rule; the
  low-storage section's volume-pass count.
- [kernel contract](../../kernel-contract.md): the clarified `node_chunk` row.
- [architecture](../../architecture.md) §13: "space-domain work runs over node tiles chosen
  from the cache within the memory budget".
- [benchmarks](../../benchmarks.md): one ledger entry per phase, the baseline first.
- The user guide's free-space performance notes
  ([Cartesian](../../../guide/free-space/cartesian.md)) gain a paragraph on node budget,
  tile and threads. No tutorial or worked example: nothing a user calls changes.
- `CHANGELOG.md`: one line per phase, citing the SEP.

### 9. Compatibility and impact

- **Results** stay within §4.5's contract: rounding budgets for matched computations,
  accuracy envelopes for adaptive, replayed and resumed runs; no validated result moves
  beyond its bound.
- **Defaults and interface** are unchanged. `node_budget` keeps its meaning as a cap.
- **Kernel schema**: no field is added or removed and no protected hook changes. The
  clarified `node_chunk` row states existing semantics and changes no validation. If
  (b) is ever taken for the canonical state, or the Adams sample gains a destination
  argument (Q7), that is a contract change with a schema bump; (b) private to a backend
  is not.
- **Output schema**: unchanged; the metadata gains the tile record.
- **Benchmarks**: the new cases take new names (benchmarks §4).
- **Luna.jl**: no deliberate difference to record.

### 10. Alternatives considered

- **A Kerr-only fast path for envelope free space** (rejected by the maintainer): one
  three-dimensional transform and an elementwise Kerr term. It duplicates the response
  machinery that Raman, the plasma and dispersive Kerr would each need again in a second
  path, and serves one application.
- **(a) A whole-volume three-dimensional transform when no pad is needed** (§6.2):
  correct and memory-neutral, and unlike the fast path above it duplicates no response
  code: it is a transform policy with the blocked path as its fallback. It is not
  recommended first because it applies only where the temporal grid needs no pad (the
  envelope runs without Raman or the envelope plasma), so it would leave the field-grid,
  Raman and plasma cases on the path (c) improves anyway. Kept as a candidate backend
  policy, for devices especially, if phase 0 shows the blocked path cannot approach the
  reference on the unpadded cases.
- **(b) Transform-ordered registers** (§6.2): the clean end state for the march. It
  costs a second register layout through every stepper and reader, and is kept as the
  escalation behind Q2, for every stepper or not at all.
- **A larger node budget**: SEP 24 §2 measured 1.3 times from 1 GiB against 64 MiB at
  $192^2$. That is a symptom of per-chunk overheads, which §6.3 addresses. Memory spent
  to hide them is the wrong trade on these routes.
- **Precomputing the prefactor volume**: one more field volume; refused by §6.8.
- **Single precision**: halves the bytes, but is a precision decision outside this SEP.

### 11. Phases

Each phase is independently mergeable on its own `feature/sep-0028-…` branch.
`numpy_ref` is the oracle and is not changed. Each phase runs the slice checks of
`AGENTS.md` (ruff, basedpyright and the test modules that exercise what it changed) and
its own verification below; the full gate and lanes run once, at the closing step.

1. **Phase 0 — instrumentation and baseline**, an accepted measurement phase with no
   behaviour change.
   - Per-pass timers on `numba_cpu`, and on Kokkos native timers or Kokkos Tools regions
     around the core's stages.
   - The reference script (§4.2).
   - The logical volume-pass audit of every stepper, the Adams sample's included (§4.3).
   - The new benchmark cases and the case matrix (§7.2, §7.3).
   - The FFTW pool sweep (§6.5) and the planning sweep (§6.6).
   - **A device baseline**, taken before the CPU optimisation so that §6.7's choices
     rest on device numbers: the low-storage march and the IP-DOPRI5 replay built for
     CUDA, timed on one current data-centre GPU at $768^2\times600$ and $384^2\times600$
     over a short fixed-schedule window, with the memory high-water mark, against a
     dedicated envelope code's single-transform CUDA engine on the same card where one is
     available (about 159 ms a step at $768^2\times600$ in that code's published
     measurement). It doubles as the march's first device build check, the start of
     [SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md)'s device
     qualification. A rental needs the maintainer's approval and a budget cap.
   - Tests of the hypotheses on the quiet node at both shapes, each isolated where it
     can be: crossing cost against stride; copies and clears; per-chunk
     synchronisation; residency and traffic of strategy (i) against group size (§4.4);
     the fold bandwidth- or arithmetic-bound; page placement and TLB effects; the
     transforms' decomposition; on the device, the share of the transforms and whether
     a rank-3 plan of the whole volume beats the batched two-dimensional one.
   - **Deliverable**: the baseline ledger entry and a report in Part II that confirms or
     revises §6's priorities.
2. **Checkpoint after phase 0** (a maintainer decision, recorded in Part III):
   - the final per-evaluation threshold (Q4) and §6.3's (ii) criterion, agreed before
     any optimisation;
   - Q5's tile rule and Q6;
   - whether phase 0's evidence supports recommendation (c). If it does not — for
     example if the crossings are not where the growth lies — the SEP returns to Draft
     for revision and re-acceptance under SEP 0, rather than proceeding on a recommendation
     the evidence has invalidated.
3. **Phase 1 — `numba_cpu`.**
   - The tile path (§6.3, strategy (i)) under the ownership contract, the blocked input,
     gather, scatter and folds (§6.4), and the ω-slowest scratch on field grids.
   - The pass-4 read removed; the Runge–Kutta clears and the Adams sample's unread
     passes removed (§6.4).
   - The free kinds' FFTW pool rule (§6.5).
   - **Verify**: §7.1 at the levels phase 1 changes, §6.8's ledger, and a ledger entry
     under §7.2's rule.
4. **Phase 2 — Kokkos CPU** (solprop-kokkos): the same on `separable_core`, the fold
   kernels and `separable_pack_chunk`, with tiled policies; the clears go where §6.3
   allows. **Verify**: as phase 1, then §7.3's acceptance on both backends. If the
   scaling target is missed, Q2 is decided here.
5. **Phase 3 (optional) — strategy (ii) or (b)**, only if phase 2's measurement asks for
   it: (ii) after its feasibility probe and Q3; (b) for every stepper under Q2.
6. **Closing step (always)**, whether or not phase 3 runs: the full gate, the slow lane,
   the coverage lane and the Kokkos-selected slow lane, once; the whole-suite aggregate
   of benchmarks §4; then Final.

### 12. Open questions

- **Q1.** *Answered while drafting:* the Zen 3 measurements of §2 set the FFTW pools
  explicitly, which bypasses the four-thread cap, so the cap is not part of their gap to
  Kokkos. It does bind a run with the default settings on these shapes; item 4's thread
  rule for the free kinds is what changes that, and phase 0 measures by how much.
- **Q2.** If phase 2 misses the scaling target, is (b) taken for every stepper (the
  layout through every reader)? A low-storage-only (b) cannot close this SEP's
  generality requirement; any narrower fallback needs a separate scope decision by the
  maintainer. *Decided at phase 2 with the measurement.*
- **Q3.** Is binding FFTW's new-array execute from compiled `numba_cpu` code acceptable
  (no new package, but a new ABI dependency on the library pyFFTW loads, which differs
  between wheels and local builds)? A maintainer decision, taken only after §6.3's
  feasibility probe and before phase 3; this SEP does not approve the binding.
- **Q4.** The per-evaluation target is 1.5 times $T_\mathrm{ref}$ provisionally (the
  maintainer's decision of 2026-10-19). Phase 0 defines and measures the comparable
  reference (§4.2); the final threshold is agreed at the checkpoint after phase 0,
  before optimisation starts.
- **Q5.** Should the tile rule read the cache sizes from the host or be a measured
  constant per platform family? Either way it needs a portable fallback: where the
  operating system does not report the cache hierarchy (or `numba_cpu` runs on a
  platform whose query is not implemented), a conservative fixed per-thread share; under
  a restricted CPU affinity (a cpuset or a batch scheduler's allocation) the thread
  count and the cache domains are those of the allowed CPUs; a shared cache's share is
  its capacity divided by the threads placed in it. *At the checkpoint after phase 0.*
- **Q6.** The drafting profile's host and instrumentation are not the ledger's. Phase 0
  replaces it. If phase 0 disagrees with §2, its report in Part II supersedes the
  motivation table; Part I is not edited after acceptance. *At the checkpoint after
  phase 0.*
- **Q7.** Should a copy-free Adams sample follow this SEP, and through which route: a
  `numba_cpu` Adams engine via the existing `_adams_engine` hook (no schema change), or a
  destination argument to `_adams_lab_source` (a schema bump)? Either needs lifetime and
  rejection tests across startup, restarts, corrected-field evaluations and dense
  observation, for both formulations. *Deferred to a later item; recorded on the
  roadmap at Final unless the checkpoint brings it in.*

### 13. References

1. A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and
   M. Kolesik, "Practitioner's guide to laser pulse propagation models and simulation,"
   *Eur. Phys. J. Spec. Top.* **199**, 5–76 (2011). doi:10.1140/epjst/e2011-01503-3
2. J. Hult, "A fourth-order Runge–Kutta in the interaction picture method for
   simulating supercontinuum generation in optical fibers," *J. Lightwave Technol.*
   **25**, 3770–3775 (2007). doi:10.1109/JLT.2007.909373
3. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301
4. M. Frigo, C. E. Leiserson, H. Prokop and S. Ramachandran, "Cache-oblivious
   algorithms," in *Proc. 40th Annual Symposium on Foundations of Computer Science*,
   285–297 (1999). doi:10.1109/SFFCS.1999.814600
5. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002),
   chapter 24 (the fast Fourier transform). doi:10.1137/1.9780898718027
6. S. Williams, A. Waterman and D. Patterson, "Roofline: an insightful visual
   performance model for multicore architectures," *Commun. ACM* **52**(4), 65–76
   (2009). doi:10.1145/1498765.1498785
7. C. R. Trott *et al.*, "Kokkos 3: programming model extensions for the exascale era,"
   *IEEE Trans. Parallel Distrib. Syst.* **33**, 805–817 (2022).
   doi:10.1109/TPDS.2021.3097283

## Part II — Implementation record

### Phase 0 report — instrumentation and baseline

- **Scope:** phase 0 (§11), 2026-10-20, Opus 5.5 (AI), branch
  `feature/sep-0028-phase-0` in solprop and solprop-kokkos, both with main (SEP 27)
  merged. An accepted measurement phase: no behaviour, default, schema or interface
  changes. The SEP's status is left for the maintainer's checkpoint.
- **Delivered:**
  - Per-pass timers on `numba_cpu` (`src/solprop/backends/numba_cpu/pass_timers.py`;
    `NumbaCpuBackend.pass_timers`), enabled by `SOLPROP_PASS_TIMERS=1` and absent
    otherwise, around every pass of `_free_pass`, `_slab_pass`, the Runge–Kutta stage,
    the low-storage input, fold and table, and the Adams sample; tests in
    `tests/backends/test_numba_pass_timers.py`. On Kokkos, profiling regions
    (`solprop_kokkos.free.*`) around the separable core's stages, read by the Kokkos
    Tools timer `benchmarks/cpu_timer.cpp` (solprop-kokkos `src/general.cpp`).
  - The reference script (§4.2): `benchmarks/sep28/reference.py`, with
    `reference_fft3_pairs` and `reference_stream_copies` in solprop-kokkos's private seam
    (`src/core.cpp`, tested in `tests/test_api_contract.py`).
  - The volume-pass audit (below), the cases and the case matrix
    (`benchmarks/sep28/cases.py`), the window harness (`window.py`), the driver
    (`run.py`), the sweeps (`sweeps.py`), the hypothesis tests (`hypotheses.py`) and the
    tables (`analyse.py`); records in `benchmarks/sep28/data/`.
  - The baseline ledger entry ([benchmarks](../../benchmarks.md), "SEP 28 phase 0"), a
    paragraph in the [tuning guide](../../../guide/performance/tuning.md), changelog lines
    in both repositories.
- **The logical volume-pass audit** (§4.3; read from the source, passes outside the
  transforms and the chunk loop, the Cartesian envelope route with a material response):

  | stepper | `numba_cpu` per step / per evaluation | Kokkos per step / per evaluation |
  |---|---:|---:|
  | low-storage march, tabulated | 26 / 6.5 | ≈25 / ≈6.25 |
  | low-storage march, recomputed | 23 / 5.75 | ≈22 / ≈5.5 |
  | IP-DOPRI5 replay (6 evaluations) | 129 / 21.5 | ≈96 / 16 |
  | `Multistep()`, exponential (1 evaluation) | ≈102 | ≈60 |
  | `Multistep()`, interaction (1 evaluation) | ≈62 | ≈53 |

  §4.3's march table is confirmed (24 a step, pass 4's read of `A` unneeded on both
  backends); both backends add the hard mask after pass 4 (`numba_cpu` a full read and
  write, Kokkos the inactive bins only). On the IP-DOPRI5 replay `propagator_mode="auto"`
  resolves to *recompute* on separable routes outside the march, so `numba_cpu` writes
  the propagator pair to a volume at stages 1–5 and Kokkos forms it per element; the
  stage-row clear is overwritten by the material finish on both backends; `numba_cpu`'s
  step start costs 14 passes (clears plus NumPy temporaries) against Kokkos's fused 4;
  Kokkos writes an error residual that only a weighted controller reads. The Adams
  sample on `numba_cpu` makes 16 passes of which 4 are needed (identity fills, stage
  clear, unused accumulation, returned copy); its exponential formulation also forms
  the phase on the host in NumPy (≈40 passes, an estimate) and its Adams kernels are
  serial. Kokkos keeps both Adams formulations resident. The ≈ entries count reductions
  and temporaries that are estimates.
- **Deviations from the proposal:**
  1. *Kokkos's T_3D.* §4.2 asks for the rank-3 pair through each backend's own library.
     Kokkos-FFT 2.0.0's rank-3 plan runs single-threaded on the Linux build (1194 ms at
     384² against pyFFTW's 45 ms; unchanged without OpenMP binding), so the reference for
     both backends uses the node's best measured parts: pyFFTW's `FFTW_MEASURE` pair at
     the node's threads and the streaming pass. Each backend's own parts are reported
     beside it.
  2. *Windows and repeats.* Two production repeats, not three; five steps at 768²; three
     steps and one repeat at one thread at 384² and on the 256² filament; no one-thread
     768² row. All declared in `cases.py` before the baseline; the A/A spreads are below
     0.5 % on the Kerr rows and below 5 % on every Cartesian row.
  3. *Hosts.* No two-socket host was measured: the SEP's one-socket headline used the
     whole of a one-socket, one-NUMA-domain node, and the one-NUMA-domain configuration
     is therefore the only one. No GPU was rented (the device baseline was taken before
     this phase).
  4. *The `numba_cpu` reference pass.* The first `reference.py` started Numba's pool
     after pyFFTW's OpenMP runtime had bound the main thread, giving a 31.6 ms streaming
     pass on few cores; the script now starts the pool on the whole grant (as the backend
     does) and the corrected pass, 5.8 ms, is in `follow-up-c8a/`.
- **Decisions:** the per-evaluation reference's parts are the node's best (deviation 1);
  the production policy binds OpenMP (`OMP_PLACES=cores OMP_PROC_BIND=close`), whose
  effect was measured separately (4–13 % on `numba_cpu`'s Kerr march, negative on its
  filament, none measurable on Kokkos). No ADR: nothing lasting is decided here.
- **Verification:** `tests/backends/test_numba_pass_timers.py` 12 passed;
  solprop-kokkos `tests/test_api_contract.py::test_the_reference_parts_time_what_they_are_asked_to`
  passed; solprop-kokkos `scripts/check.sh` passed (Mac); coverage of the changed
  modules: every timer line covered by the new tests, the off branches by the existing
  suite (100 % of `pass_timers.py`); `ruff`, `basedpyright`, `poe docs` clean;
  `scan_public.py` on every changed file. Measurements: the ledger entry. Not run: the
  full gate and lanes (the closing step's), a two-socket host, a workstation comparison beyond
  two labelled windows (load 3–5).
- **Findings against §2–§6** (numbers in the ledger entry):
  1. *Growth beyond N log N is not reproduced here.* 384² → 768² costs 4.1–4.5 times per
     step on both backends and steppers (the N log N ratio is 4.30); §2's 13 came from a
     two-socket Zen 3 node with eight NUMA domains. The devices still grow 8.7–9.1 times.
     The candidates left for §2's growth are those this node cannot show: NUMA placement
     across domains and the smaller last-level cache per thread.
  2. *The strided crossing does not grow with stride.* At fixed size it costs 1.4 times
     a contiguous copy at 384² and 768² and more at small grids; blocking recovers most
     of it. Today's Kokkos fold already moves its 6.25 logical passes at the streaming
     rate on 64 threads (31.6 against 33.6 ms), so blocked folds have little to gain at
     full threads; at one thread the fold is arithmetic-bound (1114 ms, 29 % of the
     evaluation) and the real-root shortcut of §6.4 matters there.
  3. *Kokkos's excess is the temporal stage's movement and its per-chunk cost.* Kokkos
     is 2.2–2.6 times its reference at both shapes. At the default node budget the
     temporal stage moves about 20 streaming volume-passes beyond its transforms and
     response (gather, scatter, the fine-spectrum and projection clears, the node and
     polarisation copies) and pays about 260 µs per chunk; with groups of 1391 nodes
     (256 MiB) the movement drops to 8 and the stage from 152 to 56 ms; at 5461 nodes
     (1 GiB, 158 MB of hot set against 8 × 32 MiB of L3) it rises again (68 ms): the
     residency limit of §4.4 is visible.
  4. *`numba_cpu`'s excess is a threading pathology, before any layout.* Its temporal
     pair inside the evaluation takes 3.1 ms against 0.17–0.55 ms in isolation and
     0.69 ms at one thread; its production march is only 1.9 times its one-thread speed
     (Kokkos 17) and its filament is 2.7 times slower than one thread. One-thread
     temporal plans with an uncapped transverse pool take the march from 7.75 to
     4.43 s a step; an active OpenMP wait makes it 54 s. The interference between FFTW's
     threaded executions and Numba's pool, repeated per chunk, is the leading
     hypothesis; the four-thread cap costs a further 0.15 s per evaluation at 384² in
     the transverse pair (21.7 against 168 ms).
  5. *Planning.* `FFTW_MEASURE` makes pyFFTW's transverse pair 3.2–3.4 times faster than
     `FFTW_ESTIMATE`; `FFTW_PATIENT` adds nothing for 504 s of planning. Kokkos-FFT's
     fixed `FFTW_ESTIMATE` transverse pair (30.8 ms) is 1.4 times pyFFTW's measured one.
  6. *Transforms' decomposition* (batched transverse plus temporal against rank-3) costs
     the same (45.5 against 43.5 ms at 384²); *transparent huge pages* change Kokkos by
     0–7 %; *Adams* per call costs 2.2 times (Kokkos) and 7.4 times (`numba_cpu`,
     exponential) an IP-DOPRI5 call on the 192² beam, outside the core.
- **Checkpoint recommendation** (for the maintainer, §11 item 2):
  - *Q4, the per-evaluation threshold.* Keep 1.5 times `T_ref`, with `T_ref` built from
    the node's best parts (deviation 1) and measured per host. Kokkos stands at
    2.2–2.6; removing the temporal stage's excess movement (finding 3) at today's other
    costs would put the march at about 1.5, so the threshold is demanding but reachable.
    `numba_cpu` (12–13) cannot be judged against it until finding 4 is fixed.
  - *§6.3's (ii) criterion.* Keep it as drafted, applied after (i) is built. The
    baseline's temporal stage already moves 8–20 passes and spends half the Kokkos
    evaluation in per-chunk cost at the default budget, so (i) must first reach groups
    of about a thousand nodes before the criterion can be read.
  - *Q5, the tile rule.* Read the cache sizes from the host (Linux sysfs, macOS sysctl)
    with a fixed fallback; size the group so that its hot set fits a share of the
    last-level cache of the domains its threads span (here groups of 1000–1500 nodes,
    30–45 MB, against 5461 nodes losing), with a floor that amortises the per-chunk
    cost. Note for the checkpoint: at the default 64 MiB budget the rule
    `node_chunk = budget / (20 · 16 · n_pol · n_to)` admits only 349 nodes at 600 bins,
    below that optimum, because its 20 column-equivalents are about three times the
    live group buffers.
  - *Q6.* Phase 0 supersedes §2's motivation table on this host: no growth beyond
    N log N on one Zen 5 socket; the per-evaluation excess is the temporal stage on
    Kokkos and threading on `numba_cpu`. The scaling headline (5.2 times) is already met
    by the baseline here (4.1–4.5), so it cannot discriminate on this node; it needs a
    multi-domain host (or the device) to mean anything.
  - *Recommendation (c).* The evidence supports (c)'s mechanism on Kokkos (finding 3)
    and refutes the stride-growth premise on this host (findings 1–2). It does not
    invalidate (c): the tile path and the removal of copies and clears are where the
    measured excess lies; the blocked crossings and folds are worth less than §6
    assumed at full threads. Revised priorities: (1) `numba_cpu`'s thread pools and
    per-chunk FFTW execution (§6.5, item 4) before anything else on that backend;
    (2) the tile path, larger groups and copy/clear removal (§6.3, item 2) on both;
    (3) the folds' arithmetic (§6.4's real root) for low thread counts; (4) blocked
    crossings last. Whether the scaling motivation's host dependence needs a revision
    under SEP 0 rather than a checkpoint record is the maintainer's call.
- **Sources checked:** none transcribed; the reference's definition is §4.2's.
- **Limitations:** one host type; inferred traffic only (no memory-controller counters
  on EC2); the `numba_cpu` threading diagnosis is a hypothesis with four measurements
  behind it, not a proof; the Kokkos-FFT rank-3 behaviour was not investigated in the
  library.
- **Open questions:**
  - Q8. Kokkos-FFT's rank-3 plan on Linux runs single-threaded: a solprop-kokkos issue
    if (a) or a rank-3 reference is ever wanted there. *To solprop-kokkos's issues.*
  - Q9. Should the node-budget accounting count the live group buffers (§6.8) instead
    of 20 column-equivalents per node? *At the checkpoint, with Q5.*
  - Q10. Is the scaling headline re-measured on a two-socket host before phase 2's
    acceptance? *At the checkpoint.*
  - Q11. The ledger's SEP 24 phase 2 and 3 entries describe their shapes as 300 and 600
    bins; `benchmarks/sep24/phase2_costs.py`'s grid gives 128 and 256 (`fft_size`
    `"pow2"`). *A ledger correction, for the maintainer.*
- **Two-socket follow-up (Q10, 2026-10-20).** One Spot `c6a.48xlarge` (two Zen 3 EPYC
  7R13 sockets, 96 cores with SMT, four NUMA domains of 24 cores), the Kerr beam at
  384² and 768², the march and IP-DOPRI5, three placements (whole node, one domain,
  pages interleaved), profiles, the node budget at 768² and the stride test; numbers in
  the ledger entry ("Two sockets"), data in `benchmarks/sep28/data/numa-c6a/`. The
  instance's NUMA nodes are half-sockets, so the bound runs are one domain (24 cores),
  not a whole socket; `numba_cpu`'s IP-DOPRI5 rows were dropped for the budget.
  - *Growth beyond N log N does not appear on this host class either*: 384² → 768²
    costs 4.59 (Kokkos march, whole node), 4.12 (one domain), 3.77 (interleaved), 4.02
    and 4.06 on `numba_cpu`. §2's ×13 (two Zen 3 sockets with eight domains, one socket
    at 32 threads, an earlier tree) remains unreproduced; what this node does not
    cover is four domains per socket and §2's exact configuration.
  - *What grows with the node is the temporal stage's per-chunk cost*, not the
    crossings: on the whole node the Kokkos march spends 83 % of an evaluation at
    768² in the temporal stage (about 1 ms of movement and synchronisation per chunk,
    against 0.27 ms on the Zen 5 node), and groups of 5564 nodes (1 GiB) make the step
    3.7 times faster; in one domain 3.1 ms per chunk and 2.3 times. The crossing costs
    1.1–1.6 times a contiguous copy and does not grow with stride; blocking does not
    help there.
  - *Page placement matters at 768²*: interleaving the pages over the four domains
    makes the march 27 % faster than the default placement, so first touch by one
    domain's threads, or the per-chunk buffers' placement, costs bandwidth across
    sockets.
  - *Checkpoint recommendation restated.* The two-socket growth §2 reported does not
    exist on this host class in today's tree; the excess that does exist, and grows
    with threads and domains, is the temporal stage's per-chunk cost and movement, with
    NUMA page placement second and the crossings last. Recommendation (c) stands with
    the revised priorities above, with the tile path's group size made the first
    lever on CPU (a cache-sized group that is still large enough to amortise a
    per-chunk cost of 0.3–3 ms) and first touch of the field volumes and group buffers
    spread over the threads' domains (§6.5's page placement, now a measured cost). The
    scaling headline (5.2) is met by the baseline on both host classes and cannot
    judge the SEP; the maintainer may replace it with the per-evaluation reference and
    the aggregate, or keep it as a non-regression check. Q10 is answered for this host
    class; four domains per socket stays open.
- **Handoff:** phase 1 starts from finding 4: `numba_cpu`'s pools for the free kinds
  (one-thread temporal plans per group or a passive Numba pool, then the transverse
  pool's size rule), measured with `benchmarks/sep28/run.py --groups binding,threads`;
  then the tile path. Every later phase compares against `benchmarks/sep28/data/baseline-c8a/`
  on a `c8a.16xlarge` with the same thread policy; `analyse.py` produces the tables.
- **Merge (2026-10-20).** Merged into `main` in both repositories with `--no-ff` as a
  measurement phase: timers that are absent unless `SOLPROP_PASS_TIMERS=1`, Kokkos
  profiling regions, the benchmark scripts and data, the ledger entry, this report and
  a tuning note. No default, schema or interface changed. Lanes on the branch head:
  the full gate, the slow lane and the Kokkos-selected lanes on Linux failed only on
  the two band-plasma parity cases that fail on `main` since SEP 27 (Linux only) and a
  timing test on a shared host; coverage on macOS reached 100 % (16618 passed). The
  phase 1 handoff above is not started; see the checkpoint record in Part III.

### Phases 1 and 2 report — the separable core on both CPU backends

- **Scope:** phases 1 (`numba_cpu`) and 2 (Kokkos CPU) in the order the checkpoint set,
  2026-10-20, Opus 5.5 (AI), branch `feature/sep-0028-kokkos` in solprop and
  solprop-kokkos (the `numba_cpu` Adams passes on `feature/sep-0028-numba-adams`, merged
  into it). Timed on a quiet Zen 5 node (the phase 0 instance type), a two-socket
  Zen 3 node and, as same-process A/B only, an 18-core Apple silicon laptop; numbers in
  the ledger entry "SEP 28 phases 1 and 2".
- **Delivered:**
  - *Kokkos, the chunk loop* (`separable_core`, `separable_pack_chunk`): the fine
    spectrum's bins outside the crop are zeroed per chunk instead of the whole buffer;
    the projected rows are cleared only when nothing overwrites them; on plain free
    space (no planar slab, no dispersive response) the node field and polarisation are
    views of the chunk's time rows and projected rows, so their two copies and two
    buffers go; a material stage no longer clears the stage row its finish assigns.
  - *Kokkos, the crossings*: the traversals between the registers `(column, omega)` and
    the transverse scratch run on CPUs in tiles of 32 columns by 16 bins, columns
    fastest, with flat scratch offsets (`TransverseLayout::cell`), and the fold's
    reducing pass sums each column in bin order. The scatter into the scratch uses
    streaming stores on x86. On devices the gather, the scatter and the fold pass
    through 32 × 32 tiles of team scratch, so both layouts are read and written
    coalesced, and the final fold's norms are reduced in a pass of their own.
  - *Kokkos, arithmetic and plans*: a lossless medium's real `q` takes the real root
    (the same bits); free-space transforms are planned at `SOLPROP_FFTW_PLAN`'s effort
    (`measure` by default) by seeding FFTW wisdom, which Kokkos-FFT's fixed estimate
    plans then use.
  - *`numba_cpu`*: a spatial transform of at least `FFTW_WIDE_ELEMENTS` (2^20) elements,
    the transverse pair and a large chunk's temporal batch, takes every allowed thread;
    plain free space aliases its node and polarisation buffers as Kokkos does; the
    Adams passes (both formulations) run in parallel above the regime threshold, and
    the separable exponent is formed by a compiled kernel instead of host NumPy.
  - *Both*: free-space, line and planar grids default to a 1 GiB node budget
    (`translate.DEFAULT_SPATIAL_NODE_BUDGET`; `SimulationSettings.node_budget=None`);
    modal routes keep 64 MiB.
  - *Process*: solprop keeps the process's CPU grant, and sizes Numba's pool from it,
    when `solprop_kokkos` was imported first and its OpenMP runtime bound the main
    thread to one CPU (`solprop_kokkos.PROCESS_CPUS`, `solprop._process`); before, a
    Linux run with `OMP_PROC_BIND` set ran `numba_cpu` on one core.
- **Deviations from Part I:** the blocked crossings are not a separate pass: the tiles
  carry the crossing inside the existing input, gather, scatter and fold kernels, and
  phase 0 showed the stride costs little. No new FFTW binding (Q3) and no strategy (ii):
  the per-chunk cost fell with larger groups. A dynamic schedule for the crossings was
  tried (6 % on Apple silicon's mixed cores) and reverted: on Zen 5 it slowed the
  gather and scatter by 2 to 2.6 times.
- **Findings:**
  1. On the Zen 5 node at the new default budget the Kokkos march is 2.3 (384²) and
     2.4 (768²) times faster than phase 0, IP-DOPRI5 1.7 and 1.8 times; `numba_cpu`'s
     march 10 times. On the same node the march is 2.05 (384²) and 1.67 (768²) times
     faster than the dedicated single-transform engine's Numba path on the
     comparison case of phase 0.
  2. The node budget is the largest single lever once the per-chunk copies are gone:
     1 GiB beats 256 MiB on Zen 5 (where phase 0 found the reverse) and on Apple
     silicon; a group's fixed cost, not cache residency, dominated.
  3. On two Zen 3 sockets, interleaving pages no longer helps the Kokkos march at
     1 GiB (0.679 against 0.645 s a step at 384²): phase 0's 27 % came from the small
     groups.
  4. `numba_cpu`'s temporal pool flips sign with the group size: one thread beat four
     at 64 MiB in phase 0, and every core beats one by 4.3 times at 1 GiB.
  5. On the same Zen 5 node the Kokkos march beats the dedicated single-transform
     engine's Numba path by 2.07 times at 384² and 1.67 at 768²; `numba_cpu` is 2.4
     times slower than that engine.
  6. On an H200 the march went from 73.7 to 51.6 ms a step at 384² and from about 337
     to 227 ms at 768² (the device baseline's 768² rows were doubled by its timing
     script); the dedicated CUDA engine takes 30.8 and 168.7 ms, so the gap is 1.67
     and 1.35 times. The rest of it is the gather and scatter (32 ms a step at 768²)
     and a fold already near the card's bandwidth. Option (b) (registers in the
     scratch's order) would not remove the gather and scatter: the temporal transforms
     want the frequency axis contiguous, as the registers have it. Keeping that order
     everywhere and transforming the plane with strides, as the dedicated engine does,
     would; that is a different option from (a), (b) and (c).
  7. `numba_cpu` on Linux has a threading problem outside this SEP's passes: the CtC
     window takes 16.8 s on 64 Zen 5 cores against 2.3 s on the 18-core laptop, and
     under `OMP_PROC_BIND` on two Zen 3 sockets its parallel loops run nearly serially.
- **Verification:** the full gate, the slow lane, the coverage lane and the
  Kokkos-selected lanes on one Zen 5 node (solprop at d8e749b, solprop-kokkos at
  545a768): 16632 passed in each of the gate, coverage and Kokkos lanes, the slow
  lanes passed, and the only failures are the two band-plasma parity cases that fail
  on `main` on Linux since SEP 27; coverage 100 % of lines, its one partial branch
  (`solprop._process`) covered afterwards. On an H200 the Kokkos GPU tests (31), the
  device tests of solprop (2), the march on the device (15) and the Kokkos free-space
  suite (384, with seven CUDA record rows the held-current cases lacked, regenerated
  on the card) pass. Targeted suites on Apple silicon throughout.
- **Merge (2026-10-20).** Merged into `main` in both repositories with `--no-ff`.
  Spend: AWS about US$42 of the US$60 cap (with phase 0), RunPod US$8.83 of US$30.
- **Open:**
  1. *`numba_cpu`'s OpenMP waiting on Linux.* On the Zen 5 node `OMP_WAIT_POLICY=passive`
     (or `GOMP_SPINCOUNT=0`) takes the CtC window from 16.7 to 5.2 s and the 384² march
     from 1.91 to 0.80 s a step; idle Numba OpenMP workers spin on the cores FFTW
     needs. solprop already makes pyFFTW's runtime passive, not Numba's; a fix needs
     care where Kokkos shares the runtime. Binding (`OMP_PROC_BIND`) is ruinous for
     `numba_cpu` on Linux (44-47 s and 5.7-6.2 s).
  2. *The device gap.* A probe after this phase (cuFFT on an H200): strided transverse
     transforms cost 3.7-7.1 times contiguous ones, and one rank-3 transform of the
     volume saves only 1.6-2.2 ms an evaluation against today's planes, temporal
     batch and two transposes. The transform structure is near its best; the rest of
     the gap to the dedicated engine is the march's register traffic.
  3. *Vendor FFTs on Zen 5.* AMD's FFTW fork is 3 % faster than FFTW 3.3.11 on the
     transverse pair and 6-27 % on the temporal batch; MKL is slower on the transverse
     pair at 768². Not worth a second library for now.
  4. *The SEP's closing step* (benchmarks §4's whole-suite aggregate and the
     per-evaluation reference re-measured) is not run; Q7 stays deferred.

### Closing step — the case matrix at the package's defaults

- **Scope:** §11 item 6, 2026-10-20, Opus 5.5 (AI): the case matrix (production and one
  thread), the per-evaluation reference and the matrix at the settings' default node
  budget, on the phase 0 instance type, against phase 0's baseline. Numbers in the
  ledger entry "SEP 28 closing"; data in `benchmarks/sep28/data/closing-c8a/`.
- **Results against §7.3's targets:**
  - *Aggregate:* the package's defaults now against phase 0's: 3.00 times faster over
    the 24 rows (Kokkos 2.02, `numba_cpu` 4.45); three millisecond-scale rows lose
    (the line march on Kokkos, 0.89; the planar rows on `numba_cpu`, 0.96 and 0.98).
    At phase 0's fixed 64 MiB the gain is 1.62.
  - *Scaling:* the march grows 4.33 (Kokkos) and 4.20 (`numba_cpu`) times from 384² to
    768², under 5.2.
  - *Per-evaluation reference:* within 1.5 times `T_ref` for the Kokkos march (1.02,
    0.94) and IP-DOPRI5 (1.21) and the `numba_cpu` march (1.37, 1.29); missed by
    `numba_cpu`'s IP-DOPRI5 (1.80).
  - *Memory:* the spatial default node budget adds at most 1 GiB of chunk buffers; the
    Kokkos aliasing removes two chunk buffers on plain free space.
- **Decision:** the maintainer asked for the SEP to be closed (2026-10-20). The one
  missed reference row is recorded, not hidden; `numba_cpu`'s future is SEP 29's.
- **Recorded on the roadmap:** the copy-free Adams sample (Q7), and the march's register
  traffic on devices (the remaining gap to a dedicated single-transform engine on an
  H200, 1.35 to 1.67 times).

## Part III — History

- 2026-10-19: Drafted.
- 2026-10-19: Reviewed by GPT-6 (AI), with John Travers's human review recorded
  at his request. Findings and his decisions follow. Review status is Reviewed;
  lifecycle status remains Draft. Part I has not been edited by the reviewer.
- 2026-10-19: Part I revised by the drafting agent (Opus 5.5) in response to R1–R12,
  incorporating the maintainer's decisions on R6, R7 and R8; see *Response to review*
  below. Status remains Draft; Reviewers unchanged.
- 2026-10-19: Maintainer decisions on the revision (John Travers): phase 0 runs after
  acceptance as a measurement phase ending in a recorded checkpoint, with a return to
  Draft if its evidence undercuts recommendation (c); the per-evaluation threshold (Q4)
  is fixed at that checkpoint, 1.5 kept unless the reference proves unreliable; the
  copy-free Adams sample (Q7) stays deferred; no new FFTW binding now (Q3), decided only
  if the feasibility probe passes and phase 2 meets the §6.3 criterion; strategy (i)
  first, with the drafted trigger for (ii); the layout escalation (Q2) applies to every
  stepper or none; the staging, budget and tile-override numbers stand as drafted; the
  tile rule (Q5) is chosen at the checkpoint.
- 2026-10-20: Phase 0 merged. **Checkpoint record, provisional** (taken by the
  implementing agent under the maintainer's delegation of 2026-10-19 to decide and
  record from phase 0's evidence, and to stop rather than proceed if the evidence
  undercuts the design; the maintainer confirms or replaces every item):
  - *Finding.* The evidence meets §11 item 2's named trigger. On one Zen 5 socket and
    on two Zen 3 sockets (four domains), $384^2 \to 768^2$ costs 3.8–4.6 times per
    step, within 1.1 times $N\log N$; the crossings cost 1.1–1.6 times a contiguous
    copy, do not grow with stride and gain nothing from blocking. The growth §2
    reported is not reproduced in today's tree, and the crossings are not where the
    excess lies. The measured excess is the temporal stage's per-chunk cost and
    movement (83 % of a Kokkos evaluation at $768^2$ on two sockets; larger groups make
    a step 2.3–3.7 times faster), NUMA page placement (interleaving 27 % faster), and
    `numba_cpu`'s thread pools (a temporal transform 6–18 times slower in situ than
    isolated).
  - *Consequence.* §3's scaling target (within 1.2 times $N\log N$) is met by the
    baseline on both host classes and cannot judge the SEP, and §6's ordering (blocked
    crossings first) is contradicted. Under §11 item 2 the SEP should return to Draft
    for a revision of §2, §3, §6 and §11 and re-acceptance under SEP 0. That status
    change is the maintainer's: the status stays Accepted and no optimisation phase
    starts.
  - *Provisional answers, for the revision.* Q4: keep 1.5 times `T_ref`, measured per
    host, as the per-evaluation target, and make it with benchmarks §4's aggregate the
    acceptance test in place of the scaling headline (kept as a non-regression check).
    §6.3's (ii) criterion: keep it, read only after (i) reaches groups of about a
    thousand nodes. Q5: read the cache sizes from the host with a fixed fallback and a
    floor that amortises the per-chunk cost; Q9 (count the live group buffers, not 20
    column-equivalents per node) belongs with it, since today's rule caps a group at
    349 nodes at 600 bins. Q6: phase 0's report supersedes §2's table. Q10: answered
    for two sockets with four domains; eight domains per socket stays open.
  - *Revised priorities proposed for the revision.* (1) `numba_cpu`'s pools for the
    free kinds and per-chunk FFTW execution; (2) the tile path with cache-sized but
    large groups, and the copies and clears removed, on both backends; (3) first touch
    of the field volumes and group buffers spread over the threads' domains; (4) the
    folds' arithmetic at low thread counts; (5) blocked crossings last, or dropped.
    All of them are general, with no response-specific path, so the SEP's scope and
    generality requirement are unchanged.
- 2026-10-20: **Checkpoint decided** by John Travers. The SEP stays Accepted and does
  not return to Draft: phase 0's report supersedes §2's motivation table, and the
  provisional record above is adopted as the checkpoint. Acceptance is the
  per-evaluation target of 1.5 times `T_ref` with benchmarks §4's aggregate; the scaling
  target is kept as a non-regression check. Work proceeds in the revised order:
  Kokkos first (group size and the node-budget accounting, copies and clears), then
  `numba_cpu` (the free kinds' pools, then the tile path), then page placement and the
  transverse plans; blocked crossings only if a later measurement asks for them. Part I
  is not edited.

### AI review — 2026-10-19

**Reviewer:** GPT-6 (AI). **Disposition:** revisions requested before acceptance.
The proposed blocked crossings are a reasonable first experiment, and retaining
the canonical registers limits the change well. The draft does not yet establish
that its recommended tile schedule delivers the claimed cache traffic within the
memory cap, or define an unambiguous performance and correctness acceptance test
for its full scope. These are design questions to resolve, not objections to
measuring the approach.

This review covers all of Part I against SEP 0, the conventions, kernel contract,
SEP 25 and the current implementations. Code evidence is from solprop commit
`c67aa34fd` and solprop-kokkos commit `391ca7d`. No propagation benchmark was run
for this review, and the drafting timings have not been independently reproduced.
P1 identifies a correctness, feasibility or acceptance-contract issue to settle
before acceptance; P2 identifies a specification or qualification gap to address
in the revision. The drafting agent should answer every R-number in a separate
`Response to review` subsection, with the changed sections or a reason for keeping
the text. The human decisions recorded below settle policy, not the remaining
technical findings.

#### R1 [P1] — State the actual source pipeline, especially for planar runs

**Sections:** §4.1, §6.2's comparison table, §6.3.

The equation in §4.1 is a useful sketch of the Cartesian/line material source in
the lab frame, but it is labelled an interaction-picture source without its
return propagator. More seriously, replacing its response by an absorber rate
would retain the material prefactor and spectral window. The implementations
instead use the hard mask for that term, without the material prefactor:
`numba_cpu/low_storage.py:make_free_kernels.fold` distinguishes
`_TERM_MATERIAL` and `_TERM_ABSORBER`, as does Kokkos's
`low_storage_separable_fold`. That distinction must survive any fusion.

The planar route also has a nontrivial synthesis from `n_p` modal rows to `n_x`
physical nodes and projection back. `NumbaCpuBackend._slab_material` performs
both products; Kokkos's `separable_nodes` and `slab_project` do the same.
These are not the removable Cartesian node and polarisation copies. A planar
tile's temporal input and response output generally have different row counts,
so §6.3 cannot describe them as a single copy-free node buffer.

**Requested change:** delimit the sketch's route, frame and response assumptions;
write the material and absorber contributions separately; and give the planar
pipeline its synthesis/projection and buffer dimensions. Include the envelope
plasma's separate streams and SEP 6's spectral products in the operation/lifetime
description. State exactly which planar traversals can be blocked or removed and
which products remain. This is a clarification of the existing equations, not
permission to change their normalisation or placement of masks.

#### R2 [P1] — A group of threaded tiles does not establish cache residency

**Sections:** §4.4, §6.3 strategy (i), §6.5, Q5.

Strategy (i) still gathers a whole group, calls a batched FFT over that group,
runs the response and calls another batched FFT. Nothing specified makes an
FFTW worker process the same tile as a Numba or Kokkos response worker. Summing
private cache capacities is therefore insufficient to justify §4.4's claim that
the only memory traffic is the scratch read and write. It could work from a
shared cache, but its capacity, sharing domains and interference then determine
the bound. Even common affinity settings do not prove common row ownership across
the distinct runtime pools.

There is also a direct budget ambiguity: if the 4 MiB override means a tile per
worker, 32 workers require 128 MiB before additional response and plan buffers,
already above the default 64 MiB node budget. If it means a whole group, the
per-worker tile size and iteration-count argument must use that interpretation.

**Requested change:** define tile, group, concurrent worker count and cache domain
separately. Give a rule bounding the total simultaneously live group buffers,
including response-specific storage, by the node budget; define the override's
scope and its precedence over that cap. In phase 0 measure locality and total
traffic for strategy (i), not just fewer host iterations. Keep the two-volume
traffic claim conditional on demonstrated residency, with a stated criterion for
trying strategy (ii) if it fails. Q5 must also specify a portable fallback and
behaviour under restricted CPU affinity and shared caches.

#### R3 [P1] — Removing clears and copies needs a buffer-lifetime contract

**Sections:** §6.3 steps 1–4, §7.1.

The current copies also separate the ownership of FFT input, preserved time
samples and accumulated response. `pack_columns` initializes unused rows as well
as pad bins. `_free_pass` clears the full projection, and `_slab_output` explicitly
clears unused rows because the fixed-width forward plan reads them. The proposed
write of pad bins alone does not initialize a partial group's unused rows.
`SpatialDispersive.prepare` preserves/filter-transforms the spectrum before the
inverse destroys it; its borrowed and separate-product paths then have different
write rules. The planar material pass can also be followed by an absorber pass
that still needs the original time samples.

**Requested change:** give an ownership table for each admitted transform family
(`field_real`, `field_analytic`, envelope), showing which arrays may alias, the
first writer and last reader, and whether a response overwrites or accumulates.
Specify actual-width tail plans or initialization of every unused element a plan
reads. Preallocate and plan the normal and tail shapes before stepping. A
preserved input for Raman, plasma or dispersive products must be identified even
when redundant Cartesian copies are removed.

Add focused tests that poison reusable buffers before evaluating full and partial
tiles repeatedly; cover two polarisations, material plus absorber, dispersive
response alone and combined with ordinary responses, and the real/analytic field
families. Check padding, endpoint treatment, scale and finiteness as well as the
final source. This catches stale-data and alias errors that a ten-step smooth-beam
comparison can miss.

#### R4 [P1] — Explain how Adams writes into history without changing the seam

**Sections:** §3, §6.1, §6.4, §9, phase 1.

The `.copy()` in `_adams_lab_source` is real, but removing it does not implement
“the sample is written where the history reads it.” `HostHistory.start` copies
the returned value into `_store[0]`; `HostHistory.exponential_trial` retains the
source in `_ExponentialTrial`; `exponential_accept` later writes it to
`store[free]` and may preserve another copy for dense output. The interaction
formulation's `_adams_source` additionally allocates the transported source and
copies the forward propagator. Thus the promised no-allocation/no-copy sample
requires an engine/history design, not only an evaluation-kernel edit.

**Requested change:** identify the existing engine or hook through which a
destination slot is selected and filled. Specify ownership across rejected
trials, startup/restarts, corrected-field evaluations and dense observations;
do not let a borrowed stage buffer overwrite a live history sample. State which
copies are avoidable and which belong to the integrator's algorithm. Cover both
Adams formulations or explicitly justify a narrower target. If a protected
`StepDriver` hook or stage bundle changes, require a schema bump under the kernel
contract; unchanged `KernelProblem` fields alone do not establish compatibility.
If the existing engine interface suffices, name that implementation route and
demonstrate it with lifetime and rejection tests.

#### R5 [P1] — Apply both halves of SEP 25's numerical contract

**Sections:** §4.5, §5, §7.1, §9.

The unconditional promise of equal fields to rounding, including every adaptive
march, is stronger than SEP 25 permits. A different FFT or fold can change an
adaptive step size, rejection or Adams order. Even ordinary fixed-schedule Adams
replay does not fix the order/history; SEP 25 §4.4 explicitly calls that an
accuracy-envelope comparison. Conversely, “no bitwise requirements” must not
remove exact structural checks for hard-mask zeros, lossless storage, identity
metadata and actual work accounting.

**Requested change:** separate matched discrete source/step/fixed-order-history
tests, which use rounding budgets, from adaptive/replay/resume tests, which use
per-case accuracy envelopes against a tighter or stored converged reference.
For the former specify matched method, orders, startup, restarts, noise and
observation policy. For the latter record error and actual work without requiring
identical trajectories. Retain the exact structural assertions.

State the norm and scale of the FFT error bound: an `O(u log n)` estimate is not a
uniform relative bound for individual Fourier bins that can cancel to zero.
Include source and propagation sensitivity, and amplification by the longitudinal
prefactor near cutoff, in the budget. Add one-source and one-step comparisons to
localise errors before the ten-step comparisons, and exercise the statistics,
split-step absorber and observation/checkpoint readers affected by changing the
field-grid scratch layout.

#### R6 [P1] — Define an executable and comparable performance baseline

**Sections:** §4.2–4.3, §6.4, §7.3, Q4.

The “transform bound” adds a measured FFT-pair time to an elementwise floor that
§4.3 expresses only as a count of reads and writes. It does not specify how those
counts become time, or whether the comparison includes the response, prefactor
root/division, noise, propagator construction, folds and observations. A 3-D
complex pair on the coarse volume is also not the work of a padded real-field
evaluation, a doubled-grid Raman response, extra plasma streams or planar matrix
products. Without a defined numerator and denominator, 1.5 times this quantity is
not a reproducible acceptance test.

The pass table is useful as a logical operand count, but is not itself measured
DRAM traffic: cache reuse, write allocation, compiler elimination of the
pass-4 load and per-column reductions change the physical traffic. The
[FFTW3 paper, §III](https://fftw.org/fftw-paper-ieee.pdf) likewise treats
`5 N log2 N` as a conventional performance scaling factor, not an exact operation
count. The calculated scaling ratio is approximately 4.303; 5.2 is a reasonable
rounded presentation of 1.2 times that ratio, but is not a hardware lower bound.

**Requested change:** specify a runnable baseline with exact arrays, transform
families, normalization, response and elementwise work, thread/library/plan
settings, warmup and timed boundaries. Restrict the 3-D comparison to the
unpadded Cartesian envelope case, or define separate comparable baselines for
the other cases. Distinguish logical passes, measured traffic and elapsed time;
avoid counting the same fused work twice. Pair the scaling ratio with absolute
A/B improvement at both sizes, so slowing the smaller case cannot satisfy it.
Keep the baseline definition independent of the candidate implementation.

**Maintainer decision — John Travers, 2026-10-19:** the 1.5-times target is
provisional until phase 0 defines and measures the comparable baseline. Agree
the final threshold before optimisation starts. The drafting agent should
incorporate this decision into Q4 and the phase/acceptance wording.

#### R7 [P1] — Measure the scope promised by the abstract and keep it at completion

**Sections:** §1–2, §7.2–7.3, §11, Q2.

The performance matrix lists the low-storage march at both large shapes and
IP-DOPRI5 replay only at the smaller shape. It does not explicitly qualify Adams
or adaptive execution, although both are promised beneficiaries and Adams gets a
separate ownership change. Sharing a source core does not establish end-to-end
improvement when the steppers have different copies, histories, source counts,
rejections and observation work. Q2's low-storage-only escalation would also
quietly narrow the stated completion criterion.

**Requested change:** declare the affected-case matrix and aggregation before
timing. Include both Adams formulations where supported and short adaptive
controls at matched accuracy, recording source calls and timing per call beside
end-to-end time. Use smaller named cases where a stepper's existing memory
footprint precludes the large shapes; do not require an infeasible large Adams
allocation. Include the planar control explicitly in §7.3 and run the production
and one-thread comparisons required by the benchmark policy. Explain which
headline target applies to which stepper.

**Maintainer decision — John Travers, 2026-10-19:** keep all in-scope steppers in
scope for SEP completion. A low-storage-only fallback cannot close the SEP's
generality requirement; any narrower fallback requires a separate scope decision.
This does not require every case to meet the same speed-up ratio; R8 governs
performance acceptance across the declared cases.

#### R8 [P2] — Reconcile the per-case veto with the aggregate benchmark policy

**Sections:** §1, §7.2, phase verification lists.

§7.2 first prohibits any control regression beyond A/A spread, then invokes
`benchmarks.md` §4. That policy explicitly permits small individual regressions
when the declared aggregate improves significantly. These rules can give opposite
decisions for the same measurement.

**Requested change:** use one acceptance rule consistently, preserving the full
per-case table and A/A resolution. Under the existing policy, evaluate the
equal-weight geometric mean, the production saving above 5 percent and the
conservative median/MAD ratio, with one-thread results reported separately.
Correctness and memory requirements remain independent of that timing aggregate.

**Maintainer decision — John Travers, 2026-10-19:** use the existing aggregate
rule in `benchmarks.md` §4. Revise the unconditional non-regression language in
the abstract and §7.2 accordingly.

#### R9 [P2] — Protect the compensated cutoff arithmetic in the real-root path

**Sections:** §6.4, §7.1.

The proposed real square root is mathematically appropriate for a positive real
longitudinal square, but its eligibility test is part of the numerical contract.
`bodies.spatial_element` forms that square from both high and low parts before
`longitudinal_rate` chooses propagation, attenuation and the clamped prefactor.
Discarding the residual, testing only the high part, treating a small imaginary
part as zero, or applying a real root to the clamped branch can change behaviour
beyond rounding near cutoff.

**Requested change:** restrict the shortcut to an exactly real, strictly positive
compensated longitudinal square; retain the established complex path otherwise.
Preserve the residual evaluation and the loss/clamp/mask rules. Test just below,
at and above cutoff, small nonzero loss, inactive bins and the supported
evanescent rates against the existing helper and reference. Give those tests
condition-aware absolute bounds and verify finite outputs. No new threshold
defining “lossless” should be introduced as an optimisation.

#### R10 [P2] — Make the memory limit relative to each configuration and count scratch

**Sections:** §5, §6.1–6.2, §6.8, §7.1, §9.

The 5.1–5.4-volume RSS figure describes one low-storage configuration; it cannot
be a universal ceiling for Runge–Kutta or Adams, whose existing registers already
exceed it. Saying no added field-sized *row* also leaves a loophole for several
smaller allocations whose total is a field volume. Per-thread staging is still
real memory even if stack or team scratch rather than a persistent array, and
the repository requires work buffers to be accounted for.

There is an existing planar exception to the proposed literal `node_chunk` cap:
`_chunk_layout` uses `max(1, node_chunk // n_x)` whole y rows, so a chunk can contain
`n_x` nodes even if `node_chunk < n_x`. Rewording the seam row as an absolute
maximum would contradict that implementation and the unchanged oracle.

**Requested change:** assert the total additional bytes against a matched
configuration baseline, with named unique backing allocations, plan workspace,
all simultaneous tiles and an explicit bound for per-thread/team scratch. Report
RSS separately with its measurement boundary, accounting for construction peaks
and any transform-baseline buffers. Apply the 5.1–5.4 figure only to its named
configuration. Define the planar minimum whole-row behaviour and its budget
admission before calling `node_chunk` a hard cap; state whether the documentation
clarifies existing semantics or changes the contract. Test alias accounting,
thread-count scaling and budgets smaller than one proposed group or planar row.

#### R11 [P2] — Require a supported concurrent FFT execution mechanism for strategy (ii)

**Sections:** §6.3 strategy (ii), §6.7, Q3.

The new FFTW binding is more than symbol discovery. FFTW documents that execution
is thread-safe, but planning and other operations must be serialized; concurrent
execution on different data requires the new-array interface or separate plans.
That interface also constrains shape, strides, in-place relation and alignment.
See the primary [thread-safety documentation](https://www.fftw.org/fftw3_doc/Thread-safety.html)
and [new-array execution contract](https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html).
Safe FFTW execution does not by itself establish that a Kokkos-FFT wrapper may be
called concurrently from a host parallel region with the proposed execution
space and shared plan/workspace.

**Requested change:** make a small feasibility probe a prerequisite to selecting
(ii). Name the owning FFTW library and ABI, lifetime of plans and aligned buffers,
serial planning policy, tail-plan policy, concurrency mechanism and nested pool
limits. Verify the installed Kokkos-FFT version's execution/temporary-storage
behaviour rather than assuming serial calls are reentrant. Test the supported
Numba platforms and Kokkos CPU platforms; define a supported fallback when a wheel
does not expose the necessary ABI. Q3 remains a separate maintainer decision
before that implementation; this review does not approve a new binding.

#### R12 [P2] — Make phase 0 and final qualification unambiguous

**Sections:** §2, §4.3, §11, Q4–Q6; roadmap summary.

Q4/Q6 require phase-0 evidence at acceptance, but the phases are otherwise framed
as implementation of an accepted, frozen proposal. Define whether phase 0 is
pre-acceptance investigation or an accepted measurement phase followed by an
explicit decision before optimisation. Also move the end-of-SEP gates out of the
optional phase-3 bullet: they remain mandatory when phase 2 succeeds and phase 3
is skipped. Align “each phase ... with a green gate” with the repository's
per-slice fast checks and final full qualification.

The evidence also needs more cautious causal language until that investigation.
§2 calls the FFTW pool cap a verified reason for the reported Numba/Kokkos gap,
while Q1 correctly says those runs bypassed it. Layout traffic is a plausible
cause of the large-size slowdown, but the presented timings do not exclude page
placement, cache/TLB effects, plan decomposition or synchronization as
contributors. The abstract and roadmap currently sound more conclusive than
phase 0's hypothesis tests warrant.

**Requested change:** state the decision checkpoints, always-required closing
lanes and revision process if evidence invalidates the recommendation. Treat the
causes as measured hypotheses until isolated; distinguish the default-pool issue
from the explicitly uncapped comparison. Preserve reproducible case definitions,
revisions, planning flags, placement and raw timing evidence for the replacement
baseline. Do not infer an end-to-end result from a microbenchmark or extrapolate
the ten-step windows.

**Sources and limits of this review.** The implementation evidence, SEP 25's
rounding/accuracy distinction and the benchmark acceptance rule were checked
directly. The FFTW3 paper's transform-cost convention and FFTW's current primary
documentation for concurrency and new-array execution were checked at the links
above. The numerical scaling ratio in §4.2 was recalculated. The remaining
literature references were not independently checked in full, and the exact
Kokkos-FFT 2.0.0 concurrency guarantee was not established; R11 asks for that
evidence. No physical coefficient or new numerical method was transcribed or
approved. The review adds this history and the two review-header fields only.

#### Response to review

The drafting agent (Opus 5.5) revised Part I; the review text above is unchanged. Every
code citation in the review was checked against the current trees (solprop `fba32c6`,
solprop-kokkos `391ca7d`) and found accurate. All twelve points are accepted; none is
declined. Where a promise needed a seam change, a new binding or evidence not yet in
hand, the revision narrows or defers it rather than designing it now.

- **R1, accepted.** §4.1 now delimits its scope (what the core and each finish compute
  today, with no change of normalisation or mask), writes the material term
  $S_\mathrm{m}$ (with $Q$ and $w$) and the absorber term $S_\mathrm{a}$ (hard mask
  $M$, rate $r$, no $Q$ or $w$) separately, as `make_free_kernels(...).fold` and
  `low_storage_separable_fold` select them by `term`, and states each stepper's frame.
  It describes the plasma streams' assembly, SEP 6's filter-before-inverse and its
  borrowed and separate-product paths, and Raman's own buffers. A planar paragraph gives
  the synthesis $X$ and projection $\Pi$ (`_slab_material`, `separable_nodes`,
  `slab_project`), the absorber acting on rows after the material response, and the
  buffer dimensions ($b\,n_p\,n_\mathrm{pol}$ temporal rows, $b\,n_x\,n_\mathrm{pol}$
  response rows). §6.2's planar row and §6.3 step 3 say the GEMMs remain and only the
  gather, scatter and the clears are reworked there.
- **R2, accepted.** §4.4 defines tile, group, concurrent threads, cache domain and live
  group buffers separately, with two rules: the cap ($L(g)\le$ `node_budget`, the group
  shrinking to one node or one y row) and residency. It states that strategy (i) does
  not give common row ownership across FFTW's, numba's and Kokkos's pools, so its
  residency needs the whole group in one shared cache, which 32 threads spread over
  several Zen 3 last-level caches do not provide; the two-pass claim is now
  conditional. §5 defines `SOLPROP_TILE_BYTES` as one thread's tile and makes the
  cap win, by an error. §6.3 gives the criterion for trying (ii): more than three
  logical volume-passes in the temporal stage, or synchronisation above a tenth of the
  evaluation. Phase 0 measures traffic against group size (§11). Q5 now specifies the fallback, restricted affinity
  and shared-cache shares.
- **R3, accepted.** §3 records which clears also initialise rows a fixed-width plan
  reads (`pack_columns`, `_free_pass`, `_slab_output`) and the plasma streams' copy.
  §6.3 adds a buffer-ownership table for the three transform families (first writer,
  last reader, admissible aliases, overwrite or accumulate) and three rules: every
  element a plan reads is written, or a tail plan runs; both plan shapes are made at
  construction; preserved inputs (planar time samples before the absorber, Raman, the
  plasma, SEP 6) are never overwritten early. §7.1 adds the NaN-poisoning tests on full
  and partial groups over the cases the review lists.
- **R4, accepted by narrowing.** §3 now traces the Adams sample and the history's copies
  (`HostHistory.start`, `exponential_trial`, `exponential_accept`, the interaction
  formulation's `_adams_source`), and notes that the sample also pays the Runge–Kutta
  stage's clear and accumulation. §6.4 keeps only the passes private to `numba_cpu` that
  nothing reads; the returned copy and the history's copies stay. A copy-free sample
  becomes a later item, Q7, naming both routes: a `numba_cpu` engine through the
  existing `_adams_engine` hook (no schema change) or a destination argument (a schema
  bump). §6.1 and §9 say that any protected-hook change bumps the schema; §6.8 no longer
  counts an Adams removal.
- **R5, accepted.** §4.5 is rewritten around SEP 25's two kinds: rounding budgets for
  matched discrete computations, and accuracy envelopes for adaptive runs, variable-order
  replay and resume, with work recorded rather than matched. It states the normwise FFT
  bound and absolute per-bin bounds, the response and propagation sensitivity and the
  prefactor's amplification near cutoff, and keeps the exact structural assertions. §5
  and §9 drop the unconditional "same fields to rounding". §7.1 orders the checks from
  one source and one step up to ten steps, and re-tests the field-grid scratch's readers.
- **R6, accepted, with the maintainer's decision.** §4.2 defines the per-evaluation
  reference for the unpadded Cartesian envelope case only: exact arrays, library,
  normalisation, plan, warm-up and timed boundaries for $T_\mathrm{3D}$, the response
  part, and the elementwise part as logical passes times a measured streaming pass,
  each fused pass counted once. It also defines what the measured time per evaluation
  includes. Other cases are judged by A/B ratios only. §4.3 separates logical passes
  from measured traffic and elapsed time. $r_{N\log N}$ is 4.303, the target
  $1.2r=5.16$ ("5.2"), and $5N\log_2N$ a conventional scaling. The scaling headline
  (§7.3) is taken against the smaller of the baseline's and the candidate's $384^2$
  times, so slowing the smaller case cannot meet it. Per the decision, 1.5 is
  provisional (§4.2, §7.3, Q4) and the final threshold is agreed at the checkpoint after
  phase 0, before optimisation (§11).
- **R7, accepted, with the maintainer's decision.** §7.3 declares the case matrix before
  timing: the low-storage march at both shapes, IP-DOPRI5 replay, adaptive IP-DOPRI5
  and both `Multistep()` formulations on a named smaller shape where their memory
  requires it, and the filament, line and planar controls. It covers both backends, the
  production and one-thread policies, and source calls and time per call. It says which
  target applies to which stepper. §1, §2 and Q2 now state that all in-scope steppers
  stay in scope for completion: a low-storage-only (b) cannot close the SEP, and a
  narrower fallback needs a separate scope decision.
- **R8, accepted, with the maintainer's decision.** §7.2 states one rule, benchmarks §4's
  aggregate (geometric mean, saving above 5 %, conservative median/MAD ratio, one thread
  separately). The full per-case table and A/A spreads are kept, and correctness and
  memory are independent of it. The per-case veto is gone from §1 and §7.2, and the
  phase verification lists cite this rule.
- **R9, accepted.** §6.4 restricts the real root to a compensated longitudinal square
  with an imaginary part exactly zero and a real part strictly positive. The square is
  formed as now, the clamp and mask rules stay, and no lossless threshold is added. §7.1
  adds the cutoff, loss, inactive-bin and evanescent tests with condition-aware absolute
  bounds. §3 states how `spatial_element` forms the square and corrects the operand
  count to seven per-frequency vectors, three per-column values and one scalar.
- **R10, accepted.** §6.8 measures against a matched configuration. It counts unique
  backing allocations, group buffers with plan workspace at their peak, and per-thread
  staging with an explicit bound. Resident set is reported separately with its
  boundary. The 5.1–5.4 figure is limited to its named configuration. §3 and §5 state
  the planar whole-row minimum (`_chunk_layout`, and the same rule in `numpy_ref`). The
  `node_chunk` row is now a clarification of existing semantics, not a contract change.
  §7.1 adds alias, thread-scaling and small-budget tests.
- **R11, accepted.** §6.3 makes a feasibility probe a prerequisite to (ii) and lists
  what it must establish: the library and ABI, plan and buffer lifetime under the
  new-array contract, serial planning and tail plans, the concurrency mechanism and
  nested pools, and Kokkos-FFT's behaviour under concurrent calls (or direct FFTW from
  solprop-kokkos). A platform that lacks them keeps (i). (ii) is not designed further,
  and Q3 remains a maintainer decision after the probe.
- **R12, accepted.** The causal language is softened in §1, §2, §4.3, §6.2 and §6.5 (and
  the roadmap summary). Layout traffic is a candidate among page placement, cache/TLB,
  transform decomposition and synchronisation. The §2 claim that the FFTW cap is a
  verified reason for the Numba/Kokkos gap is withdrawn, because those runs bypassed it.
  §11 makes phase 0 an accepted measurement phase followed by a recorded checkpoint
  (threshold, the (ii) criterion, Q5, Q6, and whether the evidence supports (c), with a
  return to Draft if it does not). It aligns phases with the slice checks and moves the
  closing lanes into an always-run closing step. §7.3 requires reproducible baseline
  evidence and forbids extrapolating windows.
- 2026-10-19: Accepted (John Travers); Part I frozen.
- 2026-10-20: Phases 1 and 2 merged, then the `numba_cpu` OpenMP defaults on Linux
  (outside this SEP's passes, measured with it). Closing step run; **Final** at the
  maintainer's request, with `numba_cpu`'s IP-DOPRI5 reference ratio (1.80) above the
  1.5 target recorded in the closing report.
