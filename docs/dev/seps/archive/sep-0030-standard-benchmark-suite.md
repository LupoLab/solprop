# SEP 30 — Standard benchmark suite

| | |
|---|---|
| **SEP** | 30 |
| **Title** | Standard benchmark suite |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 (2026-10-21) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Process |
| **Created** | 2026-10-21 |
| **Resolution** | Accepted 2026-10-21 by John Travers after reading the revised proposal; maintainer decision recorded in Part III. |
| **Requires** | [SEP 25](sep-0025-rounding-level-reproducibility.md) (numerical comparison policy); [SEP 28](sep-0028-free-space-evaluation.md) (bounded spatial workloads) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Phase 1 merged; phases 2–4 implemented on `feature/sep-0030-phase-2`, `feature/sep-0030-phase-3` and `feature/sep-0030-phase-4`; consolidated for merge on the phase 4 branches after closing validation |
| **Manual pages** | [Running the benchmark suite](../../benchmarking.md) |

## Part I — Proposal

### 1. Abstract

Provide one benchmark command that developers will use while changing solprop.
Its default is a quick Kokkos CPU check of four core workloads, with warm time,
startup and peak memory in a small CSV bundle. A separate standard mode collects
five fresh-process samples at one thread and the machine's declared production
width, extending noisy comparisons to at most eleven rounds. Adaptive
`Multistep()` and `RungeKutta()` measure time to a checked result; the Kerr beam
uses fixed low-storage RK4IP only. Fixed windows elsewhere diagnose per-step cost.
Ship and exercise this core runner first, then add the broader case sets, detailed
telemetry and native Kokkos coverage. Fixed workloads, explicit failures and basic
numerical checks are essential; a comprehensive scientific campaign is not a
prerequisite for a useful development run.

### 2. Motivation and scope

The immediate questions are whether a change runs faster, uses more memory or
delays the first result. A developer should get useful feedback in a few minutes,
without preparing a campaign or interpreting several incompatible scripts.

Two collection modes serve different decisions:

| Mode | Default selection | Intended use and wall-time target |
|---|---|---|
| `quick` (default) | Core headline tracks, production width, one sample | Everyday feedback; aim for about five minutes on a qualified machine; no adoption verdict |
| `standard` | Core headline and fixed diagnostic tracks, one and production widths, five samples | Repeatable comparison; aim for about thirty minutes per arm on a quiet qualified machine before noise extensions |

A paired standard collection therefore budgets about one hour before extensions.
These are qualification targets, not fabricated measurements or hard cutoffs. Report
construction, reference preparation, guard waiting and measurement separately in
the estimate. An uncached reference preparation is a separate, explicit operation.
`list` and dry-run planning show pilot-based estimates and unknown costs. If a core
run misses the target, first remove harness overhead or excessive repetition; any
change to a window is an explicit new revision, never machine-dependent resizing.
The broader suite is opt-in and may take hours.

Warm performance is primary. Memory and startup stay beside it as secondary
results. The first delivery needs reliable measurements, simple records and cheap
checks. Detailed process inventories, coverage, publication exports and additional
cases follow without delaying that delivery. Unsupported optional telemetry is
reported as unavailable; it does not prevent ordinary timing.

Initially qualify on an Apple-silicon macOS host, then a dual-socket x86-64 Linux
workstation under an exclusive allocation. Design portable records and guarded OS
probes from the start; Windows, cloud and GPU qualification remain later work.
Workloads may exceed ten seconds and keep their declared resolution. No routine
run regenerates a full-length capillary or fibre propagation. TDSE/haCC studies
remain specialised campaigns. This Process SEP changes no simulation API, physical
default, numerical method or kernel-contract schema.

### 3. Context

The [benchmark protocol](../../benchmarks.md) already requires short windows,
machine records and an aggregate criterion, but existing scripts measure different
boundaries and use different sampling rules. Consolidate their useful definitions:

| Public project source | Reuse |
|---|---|
| `benchmarks/cases.py`, `bench_window.py`, `bench_adams.py` | RDW/CtC inputs, fixed windows, bounded adaptive lead-ins |
| `benchmarks/bench.py`, `ledger.py`, `record.py`, `compare.py` | Existing command, machine aliases, ledger and performance policy |
| `benchmarks/sep25/cases.py`, `parts.py` | Mode-averaged route, output boundaries, general modal inputs |
| `benchmarks/sep28/cases.py`, `window.py`, `run.py` | Spatial resolutions, low-storage RK4IP, memory and thread experiments |
| `benchmarks/sep16/free_space.py`, `coupled_window.py`, `capability_audit.py` | Radial/Cartesian, line/planar, wall and taper definitions |
| `benchmarks/bench_fft.py`, `bench_stats.py`, `sep27/kernel_timer.cpp` | Optional component and Kokkos Tools diagnostics |

[SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md) defines
RK4IP, [SEP 25](sep-0025-rounding-level-reproducibility.md) numerical
comparison, and [SEP 28](sep-0028-free-space-evaluation.md) the spatial
workloads. The [kernel contract](../../kernel-contract.md), `build_info()` and
`memory_ledger()` already supply the main interfaces. The suite does not depend on
accepting [SEP 29](../sep-0029-compiled-backend-distribution.md); its optional Numba
adapter can follow the default Kokkos adapter without changing simulation code.

### 4. Theory

#### 4.1 Tracks, boundaries and resets

Every configuration records the resolved case revision, grid, physical inputs,
integrator including formulation, controller, window, output policy, backend and
thread policy. Defaults are resolved and saved; their names alone are insufficient.

| Track | Integrator | Timed work and role |
|---|---|---|
| `adaptive-multistep` | `Multistep()` as publicly constructed; currently exponential | Warm propagation over a fixed interval, declared observations/statistics and in-memory output; primary time to result |
| `adaptive-rk` | `RungeKutta()`; currently interaction-picture DOPRI5 | Same boundaries as adaptive Multistep; primary time to result |
| `fixed-rk` | `RungeKutta()` on a frozen schedule | Warm propagation only; diagnostic cost of identical steps |
| `fixed-rk4ip` | `LowStorageRungeKutta(scheme="rk4ip")`, tabulated flow | Warm propagation only; the Kerr cases' sole track |

Fixed observation/statistics costs are recorded separately. Construction,
translation, warm-up, reset, lead-in and reference validation are outside warm
timing. Adaptive step counts may change: record actual accepted and rejected steps
and source evaluations, not inferred multiples of a step count. Time per source
call is diagnostic. Kerr has no adaptive Multistep or adaptive RK row in this suite;
the 192-square case is also fixed RK4IP only. Large adaptive Kerr investigations
remain separate roadmap work.

Normally a sample measures one window. If a short-window calibration selects
$k>1$, time each window separately and store every $t_{i,r}$. Report
$t_i=\sum_{r=1}^{k}t_{i,r}/k$, not elapsed time around resets. Keep each repetition's
accepted/rejected/evaluation counts and their totals/ranges; never present averaged
counts as the work of one adaptive trajectory.

Reset restores the declared field, z, noise realisation, output/statistics state,
controller and method history. For Multistep this includes its step-size proposal,
error/controller memory and Adams history at the end of the declared lead-in; RK
also restores its controller proposal and method caches. Reinitialising at launch
and rerunning a bounded lead-in is the initial implementation; a later snapshot
optimisation must pass the same field-and-controller equivalence checks. No private
history-copy API is required. Reusable FFT plans and allocated buffers remain warm.
A field-only restore is not a mature-history reset. Changing the reset protocol
changes measurement identity.

Keep the original guide and longitudinal profiles when extracting a window.
Shortening a pressure-gradient guide, even with recomputed endpoint pressure, is
not assumed equivalent. The new helium cases use the original 1.5 m guide; argon
uses the original constant-fill 0.4 m guide. SEP 25's `truncated_gradient` remains
historical. Reusing such a transformation would require demonstrating equality of
the selected gas-flow law, density, index and translated longitudinal coefficients
throughout the window, not just matching endpoint pressures. Initially do not use
it. A later-state window needs a versioned witness; routine runs never make one by
repeating a long propagation.

#### 4.2 Sampling and contention

Quick mode takes one fresh-process sample per selected configuration. It records
raw time, startup, peak memory, checks and contention, with dispersion marked
unavailable. It does not manufacture a MAD of zero or issue a clean adoption verdict.
Standard mode takes five fresh-process samples, one configuration per worker.
Shuffle configurations in balanced rounds with a recorded scheduling seed; paired
arms stay adjacent and reverse A/B order across rounds. Only one measurement worker
runs at a time.

There is no three-second minimum. Use $k=1$ unless an untimed pilot is shorter than
$\tau=\max(0.1\,\mathrm{s},1000\delta)$, where $\delta$ is measured timer resolution
and call overhead. Then use $k=\lceil\tau/\min(t_A,t_B)\rceil$ for paired arms (the
single pilot for an unpaired run), capped at ten. Record when the cap cannot meet
the target. Freeze the same k across arms, rounds and resume. Ordinary propagation
windows should need no repetition, and quick mode uses k=1. This is an overhead
control, not a statistical accuracy claim.

For clean sample times let $m=\operatorname{median}(t_i)$ and
$d=\operatorname{median}(|t_i-m|)$. If $2d/m>0.05$, collect complete paired rounds
up to eleven total; remaining noise is unresolved. These are descriptive bounds,
not confidence intervals. Also extend if clean eligibility counts below are short.
Do not remove a slow sample because of its time or select a best repeat.

A load-guard miss defers the configuration (the whole A/B pair in paired mode) to
the end of its round. Allow one deferred retry by default. If it still misses,
measure anyway and mark it `contended`. Contention during a measurement is flagged
the same way. All valid measurements remain in all-sample summaries and raw tables;
separate clean summaries use only unflagged measurements. A contended sample does
not fail a run. `--force` skips waiting and marks the entire run exploratory.

A clean comparison needs at least five clean paired rounds per configuration and
at most 20% contended measurements in either arm's completed valid measurements. If either
member is flagged, omit that whole pair from clean statistics. At five rounds,
one flagged pair therefore calls for a further complete round; at the eleven-round
ceiling, inadequate counts, more than 20% contention or unknown guard confidence
make the verdict unresolved. Unpaired standard comparisons need five clean samples
per arm under the same fraction rule. A crash, timeout or failed accuracy check is
a failed attempt, distinct from contention. Retrying preserves the original row;
unresolved execution failures make the comparison incomplete, and accuracy failures
make it numerically invalid. None of these erase the useful measured rows.

#### 4.3 Comparison rules

Pair only equal case revisions, resolved methods, boundaries, thread/measurement
policies and machine-profile revisions. Source/build identity normally differs
between arms; record that intended axis. Explicit library/build experiments name
their changed axis. A default spelling change resolving to identical values is
compatible; an actual changed formulation, tolerance, output or window is not.

For candidate and baseline medians define $r_j=m_{c,j}/m_{b,j}$ (below one is faster)
and $G=\exp(n^{-1}\sum_j\log r_j)$. Equal-weight sets are deduplicated and aggregates
stay separate for each thread width and track. Report the two adaptive-track
aggregates and their equally weighted combined adoption aggregate. Fixed diagnostic
rows, including Kerr, are beside them and outside that aggregate. An optimisation
affecting only fixed RK4IP is assessed with a separately declared fixed-track set;
it cannot claim an adaptive improvement. Never hide missing rows by scoring the
surviving intersection.

Preserve [benchmarks.md §4](../../benchmarks.md): declare the affected set before
timing; production improvement must exceed 5%, and the geometric mean of
$(m_{c,j}+2d_{c,j})/(m_{b,j}-2d_{b,j})$ must be below one. A non-positive denominator
is unresolved. Report one-thread results critically and flag individual timing
regressions over 10%; small individual losses do not automatically veto an
aggregate improvement. The machine profile fixes the production width; it cannot
be picked after looking at results. Adoption requires that production row; an
explicit one-thread-only run can report a diagnostic comparison, not adoption.

For a standard paired run, let P contain original round indices i for which both
round i and round i+1 are clean. Define adjacent baseline log changes
$a_i=|\log(t_{b,i+1}/t_{b,i})|$ for $i\in P$, and the per-configuration
repeatability band $S_j=2\operatorname{median}_{i\in P}(a_i)$. At least four such
pairs are needed; extend up to eleven or report repeatability unavailable. Do not
join nonadjacent rounds across contamination. An explicit A/A control uses the same
prepared executable for both arms and at least five clean paired rounds. Report
its median ratio and $S_j^{AA}=2\operatorname{median}_i|\log(t_{A2,i}/t_{A1,i})|$.
A saved A/A band with the same profile, baseline build, case and measurement policy
may supply an unpaired comparison; without either source, a clean adoption verdict
is unavailable.

For the affected set use $S=n^{-1}\sum_j S_j$ and require $-\log G>S$ in addition
to both existing adoption criteria. If these disagree, label the claimed gain
unresolved and show which condition failed. This conservative descriptive rule
does not claim a confidence level. A/A campaigns run when qualifying a machine or
changing its profile, not before every quick run.

Memory and startup have separate flags. Compare maximum observed worker peak RSS
and median fresh-process startup per configuration. Flag memory when its increase
exceeds 5% **or** 64 MiB (67,108,864 bytes), and startup when it exceeds 10%; show
absolute differences and ratios, with undefined ratios explicitly unavailable.
Use matched sample counts for maximum-memory comparisons and show all-sample and
clean values so contention cannot hide a resource problem. These advisory flags do
not veto a performance adoption automatically; the maintainer decides the tradeoff.
A candidate memory-admission refusal where baseline succeeds is an explicit resource
regression: no clean adoption verdict, even if other cases improve. Symmetric
refusals yield an incomplete selected set. Missing secondary telemetry is labelled
and does not invalidate a warm-time verdict by itself.

#### 4.4 Startup, memory and numerical validity

Startup is parent-observed launch through the first completed declared window,
including any lead-in. Record import, construction/translation, backend creation
and first-window components where existing interfaces expose them; keep unassigned
process/IPC overhead. Record FFTW planner flags and wisdom source/hash, Numba cache
state and hits when exposed, Kokkos initialisation, and filesystem-cache state as
unknown unless controlled. Fresh-process does not mean cold-disk latency. Optional
empty-cache experiments use isolated directories, never delete users' caches.

Record worker lifetime OS peak RSS through measurement, including setup, in bytes.
Backend allocation-ledger bytes are a separate breakdown. Phase samples are lower
bounds; high-water differences are not phase peaks. Reference comparisons occur
after the measurement receipt or in a separate worker, so they cannot inflate the
recorded peak. Never sum unrelated process lifetime maxima.

Routine checks are finite output, shape/identity, endpoint reached, exact work
accounting and comparison with an existing qualified reference when available.
Reference preparation and convergence studies happen once per case/reference
revision, separately from timing. Quick runs without a reference still give useful
results labelled `not-qualified`; standard adoption requires a valid reference
check for each arm. Repeat samples need only cheap sanity checks after the first
validated result for that arm/configuration. A case uses SEP 25 rounding bounds
for matched computations and a frozen accuracy envelope for adaptive results.
Never derive an envelope from a candidate's speed or loosen one to obtain a pass.

### 5. Proposed interface

#### 5.1 Registry, cases and sources

Keep typed case builders under `benchmarks/`, outside the simulation API. Each
definition declares physical inputs, resolved dimensions, guide/profile, initial
state, tracks, steps/intervals, tolerances, outputs, warnings, numerical reference
and capability tags. Record a case revision, suite version and result-schema
version separately. Stable CLI aliases such as `ctc` resolve to immutable revisions;
changing physics, resolution or measurement semantics makes a new revision. This
replaces the old rule that every change must retire the CLI name (§9).

| Set | Contents |
|---|---|
| `core` | Four-mode VUV RDW; full-resolution six-mode CtC; 384-square Kerr beam (fixed RK4IP only); 256-node radial filament |
| `modal-capillary` | Resolved one/four-mode VUV RDW; reduced/full CtC; resolved argon blueshift; resolved hydrogen Raman shift |
| `free-space` | Radial filaments 256/512; Cartesian filaments 128-square/256-square; Kerr beams 192/384/768-square (fixed RK4IP only); line, hybrid planar and spatial absorber |
| `mode-averaged` | Kerr-only GNLSE soliton; Raman/shock GNLSE; mode-averaged helium RDW, argon blueshift and hydrogen Raman shift |
| `advanced` | Scalar and TE/TM walls; taper with/without wall; general step-index pair; linear two-channel and nonlinear four-channel GRIN; moving solid guide |
| `backend-exercise` | Vector Raman; envelope plasma; mixtures/gradients; dispersive Kerr; plasma integration variants; batching; noise; dense output; absorber treatments; ordinary/far-field statistics |

`--all` is the sole spelling for the deduplicated union; there is no set named
`all`. Sets select workloads; modes, widths and admitted tracks are separate choices.
Unimplemented or unqualified definitions remain visible as pending, not fabricated
successful results. Core delivery does not wait for the other sets.

**Core track matrix.** A means both `adaptive-multistep` and `adaptive-rk` with the
constructors/formulations in §4.1; F means `fixed-rk`; K means `fixed-rk4ip`.
Original full guides and spectral/spatial grids are retained.

| Case and source | Grid/route | Primary interval, lead-in and local tolerances | Fixed diagnostic | Qualification reference |
|---|---|---|---|---|
| `rdw`, `benchmarks/cases.py::CASES["rdw"]` | Resolved four-mode field, 4096/8192, original 1.5 m helium guide | A: 0.02–0.04 m; lead-in 0–0.02 m; `rtol=1e-6`, `atol=1e-10` | F: ten 2 mm entrance steps | Tight RK over the same lead-in/window; matched-method NumPy parity; retained RDW band metrics |
| `ctc`, `benchmarks/cases.py::CASES["ctc"]` | Resolved six-mode field, 32768/65536, 4 ps, original 0.2 m hydrogen guide | A: 0.00017–0.00034 m; lead-in 0–0.00017 m; `rtol=1e-6`, `atol=1e-10` | F: ten 17 micrometre entrance steps | Tight RK on the same short interval; NumPy parity and Raman/energy metrics |
| `kerr_beam_384`, `benchmarks/sep28/cases.py` | Cartesian envelope 384-square, 600 bins, original box/material | K only: one 0.5 micrometre lead-in then ten 0.5 micrometre steps; no adaptive tolerance | Same K row, never duplicated | Existing SEP 24/28 fixed RK4IP qualification and same-schedule NumPy parity |
| `radial_filament_256`, `benchmarks/sep16/free_space.py::build("radial_filament")` | Radial field, 256 nodes, 250 fs, 350 nm–2 micrometres, original 0.03 m medium | A: 0–0.03 m, no lead-in; `rtol=1e-6`, `atol=0`; four source-case observation planes | F: ten steps from a pilot schedule, frozen during phase 1 | `free_campaign.py` tight RK reference and its worst-plane field metric; NumPy parity |

The radial adaptive row includes solver startup inside its warm interval and does
not claim a mature-history window. Its existing span is a starting definition for
phase 1 cost qualification, not a measured promise to fit quick mode. If it misses
the target, freeze an explicitly shorter revision before adoption comparisons.
No draft-only timing is needed to choose unmeasured constants.

Adaptive observations are the interval endpoint for RDW/CtC and the four declared
planes for the radial case, with ordinary statistics and in-memory output. Record
the exact statistics spec and field-retention policy. Core accuracy envelopes are
the source cases' existing metrics evaluated on these intervals; the numeric bounds
for the newly combined default-Multistep windows are **not yet qualified**. Phase 1
must freeze their reference receipts and bounds before a standard adoption verdict.
Use the source protocols' tight RK settings (starting at `rtol=1e-10`) and existing
analytic/parity evidence; perform refinement only where that evidence is inadequate.
Publishing a made-up global error ceiling here would not make the suite safer.

**Remaining case sources.** These rows identify what is extracted, not commands to
execute a complete gallery. A is the same two adaptive tracks; F is added only
after a useful short schedule is frozen. K is the Kerr-only fixed track.

| Cases | Public definition and grid/route | Tracks and check |
|---|---|---|
| Resolved one-mode RDW; reduced CtC | `benchmarks/cases.py`, RDW with `modes=1`; reduced CtC unchanged, two modes and 1 ps | A/F; short RK reference and NumPy parity; reduced CtC F is ten 120 micrometre steps |
| Resolved and averaged argon blueshift | `docs/examples/processes/plot_06_ionisation_blue_shift.py`: radius 13 micrometres, original 0.4 m constant 1.7 bar Ar guide, 800 nm, 30 fs, 3 microjoules, loss off, 400 fs, 300 nm–4 micrometres; resolved one mode or explicit mode-averaged route | A/F entrance windows frozen in phase 2; separate plasma/field reference for each route, not the entire SEP 25 21-plane run |
| Resolved and averaged hydrogen Raman shift | `docs/examples/processes/plot_05_raman_self_frequency_shift.py`: radius 20 micrometres, original 2 m, 5 bar H2, 1030 nm, 40 fs, 4 microjoules, loss/plasma off, Raman on, 400 nm–4 micrometres; new named `h2_raman_6ps` variants use 6 ps | A/F short windows; route-specific Raman/field checks. The 6 ps variant is a longer-grid workload, not a claim that the gallery's 3.5 ps is invalid |
| Averaged helium RDW | Four-mode RDW physical inputs above, `modes=1`, explicitly mode-averaged response, original 1.5 m profile; use `benchmarks/sep25/cases.py` only for its route construction | A/F; independent averaged-route reference; no truncated-gradient substitution |
| Kerr-only GNLSE soliton | `docs/introduction/tutorials/plot_02_soliton.py`: envelope, 835 nm, 4 ps, 500 nm–2 micrometres, N=3, Kerr/second-order dispersion only | A/F short interval, not a soliton-period run; analytic/NumPy soliton tests and short RK reference |
| Raman and self-steepening GNLSE | `benchmarks/sep13/gnlse.py`: its BETAS, gamma, launch and 12.5 ps envelope grid, 400–2000 nm | A/F short interval of original 0.15 m guide; tight RK, no full-length campaign |
| Radial 512, spatial absorber | `benchmarks/sep16/free_space.py`: `radial_filament_large`, `radial_boundary`; preserve 512-node/250 fs and 64-node/0.3 mm aperture definitions respectively | A/F; `free_campaign.py` reference and absorber balance |
| Cartesian filaments, line, hybrid planar | `benchmarks/sep28/cases.py`, underlying `sep16/free_space.py`: 128-square/256-square field filaments; 128-point line; two slab modes by 32 transverse points | A/F where admitted; short RK/NumPy checks, preserved grids |
| Kerr 192 and 768 | `benchmarks/sep28/cases.py`: 192-square/130 fs; 768-square/600 bins/250 fs, original boxes | K only; ten steps for 192, five for 768, both at 0.5 micrometres after one lead-in step; same-schedule parity |
| Scalar/TE–TM walls; taper with/without wall | `docs/tutorials/capillaries/plot_07_coherent_wall.py`, `plot_08_coherent_wall_taper.py`, `plot_09_vector_wall.py`; `benchmarks/sep16/coupled_window.py` for bounded intervals | A/F with original channel/grid/profile choices; coupled-flow/taper tests and tight RK |
| Step-index pair; linear GRIN | `docs/tutorials/waveguides/plot_05_general_modal.py`, `plot_06_graded_index.py`; `tests/modal_provider_cases.py` specifies the HE11 parity pair used by SEP 25 | A/F; pair parity and analytic GRIN spacing/self-imaging; preserve guide knots and temporal grids |
| Nonlinear GRIN; moving solid guide | `docs/examples/waveguides/plot_04_graded_index_multimode.py`, `plot_05_moving_solid_guide.py`; preserve four-channel Kerr/dispersive and moving-basis definitions | A/F short windows; existing modal/connection tests and short RK references |
| Vector Raman, envelope plasma, mixtures/gradients, dispersive Kerr | `docs/tutorials/capillaries/plot_06_vector_raman.py`, `plot_05_envelope_plasma.py`, `plot_04_air_mixture.py`, `plot_03_gas_flow.py`, `plot_11_dispersive_kerr.py` | A/F where supported; matching vector/plasma/dispersive oracle tests |
| Plasma integration, batching, noise, dense output, absorber treatments, statistics | `tests/backends/test_band_plasma_parity.py`, `tests/backends/test_batch_lockstep.py`, `tests/test_noise.py`, `tests/backends/test_free_noise.py`, `tests/backends/test_dense_output.py`, `tests/backends/test_spatial_absorber_schemes.py`, `benchmarks/bench_stats.py`, `docs/tutorials/free_space/plot_07_far_field_statistics.py` | Extract positive small workloads with explicit dimensions and supported tracks; existing invariants/parity, no import of test fixtures at runtime |

The 6 ps hydrogen definitions change only the stated grid window and the explicitly
named route relative to the public example; resolved and averaged results are not
paired with each other or with the 3.5 ps gallery. Broader cases freeze dimensions,
windows and bounds when added; they do not delay core use. Builders extract public
definitions without importing executable galleries or test-only helper modules.

Each case declares admitted widths/tracks and exclusion reasons. The 768-square
Kerr case excludes one thread from standard selection and uses five measured steps,
matching SEP 28's production window. An explicit diagnostic override may run that
same window at one thread. The 384-square case keeps ten steps at every admitted
width; SEP 28's three-step single-thread rows stay historical and incompatible.
An exclusion is a visible planning/result row, never a silent clamp or smaller grid.

#### 5.2 Commands and collection modes

During development, use `uv run python -m benchmarks`; switch `poe bench`
after the core runner is usable and qualified in phase 1. The module entry point
remains valid thereafter. These are proposed commands:

```text
uv run python -m benchmarks run --set core
uv run poe bench list
uv run poe bench run
uv run poe bench run --mode standard
uv run poe bench run --case ctc --mode standard --threads 1,production
uv run poe bench run --set advanced --threads 4,16
uv run poe bench run --all --backend kokkos --dry-run
uv run poe bench run --mode standard --baseline-python BASE --candidate-python CANDIDATE
uv run poe bench run --out RESULTS
uv run poe bench run --resume RESULTS
uv run poe bench compare BASELINE CANDIDATE
uv run poe bench qualify --set core
uv run poe bench coverage --set core
uv run poe bench coverage --all
```

No selector means `core`; no mode means `quick`; backend defaults to `kokkos`.
`--case`/`--set` repeat and deduplicate; `--all` selects the union. NumPy and Numba
are explicit backends; absent requested backends fail clearly. Quick uses headline
tracks (six adaptive rows and one Kerr row for core); standard adds admitted fixed
diagnostics. `--track` selects the named tracks in §4.1 without multiplying cases
by every integrator variant. `qualify` prepares or checks references separately.

Quick defaults to production width; standard defaults to `1,production`, deduplicated
when production is one. `--threads` accepts arbitrary positive counts and the
`production` token. `--seed` affects only scheduling; physical noise seeds are fixed
by case revision. `--timeout` and `--memory-limit` bound attempts (stored seconds
and bytes); support `--out`, `--machine-profile`, `--force` and diagnostic placement.
Pair prepared environments without modifying/installing into them. Use one harness
version and verify each arm's actual imported packages/builds.

Planning uses registry metadata and stored pilot receipts, with no translation or
scientific imports. Print the full admitted matrix, exclusions, memory estimates,
missing references and estimated setup/warm/guard/total cost for the selected mode;
show standard noise-extension cost separately. A missing or stale pilot is unknown,
not zero. Do not invent cross-machine scaling. Collect and refresh pilots explicitly
or from completed compatible runs, never inside `list`.

Resume validates the saved plan/identities and skips completed sample IDs. Different
options or builds require a new run. Retries get new attempt IDs and preserve earlier
rows. A successful quick run exits zero with advisory flags; crashes/invalid results
exit nonzero. `compare` reports exploratory, passed, regression, unresolved,
incomplete, incompatible or numerically invalid, with nonzero exit unless a standard
comparison cleanly passes. Contention by itself does not fail `run`.

#### 5.3 Machine profiles and thread policies

A small versioned JSON machine profile, keyed by alias, declares production width,
allowed allocation/affinity, CPU topology fingerprint and thread policy. Embed it
in every run. A first quick run may create a local profile with the detected eligible
CPU count, printing that choice; standard mode requires an explicit saved production
width. No silent reduction of unsupported budgets. Different profile revisions are
incompatible for adoption; an explicit profile experiment reports its own ratios.

The standard `budget` policy sets backend budget N, preserves size-based internal
pool choices and caps external BLAS at N. At one thread constrain every relevant
pool. Portable controlled runs use unbound OpenMP workers and passive waiting within
the recorded allocation. Separate diagnostics are `as-installed` (one defaults row),
`all-pools-N`, `blas1` and explicit binding. They never share a timing distribution.

Set the child environment before scientific imports; record incoming and effective
settings. Cover Kokkos kernels, temporal/transverse FFTs, Raman FFTs/tails, FFT
thresholds/planning/wisdom, Numba, OpenMP, OpenBLAS, MKL, BLIS and Accelerate,
including binding, wait/spin policy and affinity. Record requested versus observed
values; unobservable library caps remain unverified. On heterogeneous CPUs record
worker/core-class residency where available, otherwise explicitly unavailable.
A width alone does not imply performance-core placement. Detailed residency tracing
is diagnostic and never silently enabled during ordinary timing.

### 6. Implementation strategies

#### 6.1 Deliver a small runner first

Use standard-library scheduling, subprocesses, CSV/JSON and typed records, plus the
approved benchmark-only `psutil` dependency [3]. Keep telemetry outside runtime
dependencies. Reuse `threadpoolctl` for available introspection, not after-import
retuning. Set all thread variables first and preserve Kokkos-before-pyFFTW import
order. One measurement worker runs at a time; no reference job overlaps it.

The first slice needs the four core cases, selection, quick/standard modes, correct
timing/reset boundaries, NumPy checks, Kokkos timing, raw CSV/manifest, simple compare,
startup/RSS, a lightweight guard and recoverable sample writes. Do not build a plugin
framework or comprehensive topology service first. Add optional fields compatibly.
Detailed provenance and coverage can improve an already used runner.

#### 6.2 Monitoring and memory admission

Start with CPU utilisation, available memory, swapping when observable, worker RSS,
thread counts, affinity, build identity and before/after snapshots. Detailed mode adds
process inventories, sockets/SMT/NUMA/cache maps, heterogeneous core classes, CPU
features, power/frequency, cgroup quotas and commit limits. Probe Linux via
`lscpu`/sysfs and macOS via `sysctl`, with guarded Windows probes. All missing values
have reasons, not zero sentinels. Process identity is PID plus creation time; do not
collect command arguments or full environments for contention diagnosis.

Use two one-second CPU-counter intervals before a standard sample, maximum five
seconds of waiting per visit, and the bounded deferral rule in §4.2. Quick takes
one interval and never waits for quiet. The guard's initial threshold is median
external use at most 10% of eligible CPU capacity and no observed swapping; record
thresholds and confidence. Unavailable swap counters are unknown, not evidence of
no swapping; do not promote an unknown guard to clean. Exclude benchmark CPU where
counters permit. Lightweight during/after sampling flags external contention.
Native rolling load averages are context only, so recent benchmark load does not
block the next sample. Windows uses CPU counters, not its initially meaningless
emulated load average [3]. Heavy process enumeration stays outside timed regions.

Record OS worker high-water RSS (Darwin bytes, Linux kibibytes converted to bytes,
Windows peak working set). Detailed phase polling defaults to 100 ms and records
missed polls. Before/after process RSS remains available in quick mode. Measure
monitor overhead during qualification; expensive topology/residency probes are
optional diagnostics, not per-sample prerequisites.

Before expensive construction, compare a conservative estimate with the smaller
of the explicit limit and effective available allocation, with a recorded OS/controller
reserve. Include translation, retained output, scratch, histories and simultaneous
copies. Use existing estimators and qualified empirical peak-plus-margin receipts
initially; a complete symbolic estimator is not a phase 1 prerequisite.

An unestimated configuration is refused with a result row unless explicitly selected
with `--allow-unestimated` and a memory limit. Such a run is exploratory and monitored;
best-effort process limits do not guarantee avoiding an OS kill. Missing estimates
must not launch a known memory-heavy path automatically. Any observed peak above the
estimate is an estimator defect, retained and used to invalidate that estimate for
future admission until revised. Test this with a synthetic 384-square/default-Adams
configuration corresponding to the known excessive-history case; do not run that
unwanted adaptive Kerr case to prove it can exhaust memory.

Refusal, timeout, crash and missing receipt produce durable rows. Kill only the
attempt's own process tree. Without evidence, an abnormal exit is not labelled OOM.

#### 6.3 Records and provenance

The first bundle contains `manifest.json`, `samples.csv`, `summary.csv`,
`accuracy.csv` and raw logs. Include case/configuration/profile identities, requested
and effective widths, raw repetition times and work counts, all flags, startup/RSS,
reference status, source revisions/dirty fingerprints, package versions and complete
available `build_info()`. Hash the installed native extension on every native run;
hash the original wheel when available, otherwise record unknown. Never invent a
wheel identity from an installed directory. Record actual linked FFT/BLAS/OpenMP
libraries and compiler flags as exposed; unavailable metadata does not stop quick use.

As the suite grows add `runs.csv`, `cases.csv`, `configurations.csv`, `system.csv`,
`processes.csv`, `allocations.csv`, `libraries.csv` and `coverage.csv` as normalised
views of the same records. Use SI units, byte counts, UTC timestamps, UTF-8 and
documented missing-value/status columns. Nested configuration belongs in JSON, with
schema/hash relationships, not ambiguous CSV strings. Inventory dependencies and
source/build receipts once per arm/session, not per sample.

The parent writes one immutable completed-attempt shard using a temporary file,
flush and same-directory atomic rename. Derive aggregate CSVs from shards and
regenerate them on resume. Use a run lock; no resume deletes or overwrites an
earlier attempt. Short numeric artifact names permit eventual Windows support.
Incremental export is an optional later hook, not a requirement to run locally.

Local detailed bundles stay outside version control. Reuse `SOLPROP_BENCH_MACHINE`
as the alias default; explicit `--machine-profile` wins and records any mismatch.
Aliases identify physical machines without embedding hostnames and are not inferred
solely from hardware models. Sanitised exports live in
`benchmarks/results/<alias>/suite/<run-id>/` as summary/sample CSVs plus a small JSON
manifest with public hardware/build/profile identity. Strip private paths, hostnames,
process details and private source locations, recording omissions. Existing version-2
JSON ledger entries in `benchmarks/results/<machine-id>/` are preserved alongside
them, not rewritten. Publication export is explicit and never automatically commits.

#### 6.4 Separate native coverage

Coverage follows the usable runner. Use an isolated Clang build on both macOS and
Linux; Linux's GCC timing build is not repurposed. Record compiler/tool versions,
optimisation level and the coverage build's OpenMP runtime. Use matching
`llvm-profdata`/`llvm-cov`, `-fprofile-instr-generate -fcoverage-mapping` and
`-fprofile-update=atomic`, with the profile runtime linked [4, 5]. A `BuildInfo`
marker makes timing reject instrumented builds.

Instrument native core, bindings and project headers. Filter reports by project
source paths so Kokkos, Kokkos-FFT, Kokkos Kernels and other dependency templates
instantiated in project translation units are outside the denominator. Record
compiled-out and uninstantiated paths separately. Use process/module-unique profile
names per case/attempt, verify binary/profile identity and merge per-case, per-set
and union reports. Export HTML and line/function/branch CSV summaries. A failed
profile run is incomplete, not a successful zero-hit result.

Coverage uses declared dimensions and enough startup/steady execution, without
timing repetitions. Pair native reports with case capability tags and observed
Kokkos Tools kernel/region labels collected separately. Static tags alone are not
execution evidence. Distinguish exercised, supported-but-uncovered and unsupported
features; do not sum inclusive region times. Establish the first measured CPU
baseline, with no invented 100% target or claim about GPU code. Optional Numba
interpreted-body coverage is lower priority and labelled accordingly.

### 7. Verification and validation

#### 7.1 Test the runner quickly

Synthetic workers test selection/deduplication, identity compatibility, profile
changes, pre-import environment setup, scheduling, resets/lead-ins, raw repetition
times and exact work totals. Cover contention deferral and bounded retry, clean-pair
selection, five-to-eleven extension, A/A criteria, advisory memory/startup flags,
missing telemetry/units, unestimated admission, estimator defects, child failures,
timeouts, interrupted writes/resume and instrumented-build refusal. These tests
must not require expensive simulations or native coverage builds.

Add one small end-to-end NumPy case, then a Kokkos core run as soon as the runner
works. Exercise quick mode during development rather than waiting for every feature.
Detailed native coverage later tests known hits/misses, source filters and set unions.

#### 7.2 Reuse numerical qualification

Reuse public builders, existing analytic soliton/linear/GRIN checks, compiled-backend
parity and short-window references. Validate both field and controller/history state
after reset against fresh construction plus the same lead-in, within established
bounds. Check active intended responses; a launch window with little developed
plasma must not claim coverage of a later plasma regime. Only add a developed-state
witness where useful coverage requires it.

Reference generation is separately requested and cached by case, method, material
and reference-code identity. Changing relevant physics/reference code invalidates
the receipt; unrelated runner or candidate backend edits do not regenerate truth.
A lightweight candidate result check uses the cached reference. Three-level
refinement is needed for a new unbounded numerical comparison, not on every run or
for every already-qualified case. Do not assert conservation for lossy/ionising
cases or invent tolerances before qualification. Standard adoption checks cannot
be bypassed by quick-mode labels.

Replace inherited category-wide warning suppression with case-specific expected
warnings: category plus narrow message pattern and phase. Capture with an always-emit
warning policy and count every occurrence, including expected ones, in
`accuracy.csv`/logs. Repeated warnings are aggregated, not hidden. Unexpected warnings
mark validation `warning-unreviewed`;
quick timing continues visibly flagged, while standard adoption is unresolved until
the warning is understood. Non-finite output or a failed numerical check is invalid,
not merely a warning. Backend diagnostics emitted through logging retain their
level/message counts too. Warning recording overhead stays visible and outside
timing where feasible; no blanket warning filters enter benchmark builders.

Extracted definitions retain their primary citations. Phase reports state which
inputs were inherited and which newly transcribed constants were checked. This
measurement project does not repeat the scientific validation of every existing
material model before allowing a core performance check.

#### 7.3 Qualify use, cost and machines

First demonstrate the core on an Apple-silicon macOS host: quick turnaround,
standard wall-time target, reset/parity, thread caps, startup/RSS and a short A/A
control. Record actual costs; optional missing hardware counters stay visible.
Contention does not fail execution, but clean adoption follows §4.2.

Then use a dual-socket x86-64 Linux workstation under exclusive allocation. Set its
profile to a verified physical-core allocation within one socket, measure one and
production widths serially, retain unbound OpenMP within the process cpuset, and
record NUMA placement. Detached launch and incremental copying are useful recipes,
not dependencies of local execution. Maintain distinct machine baselines. Detailed
telemetry and the broad matrix follow core usability; no Windows/cloud/GPU resources
are required for this SEP's initial qualification.

### 8. Documentation

Deliver `docs/guide/performance/benchmarking.md` with the first usable runner:
quick command, standard comparison, machine profile, estimated cost, flags and
failure recovery. Link it from the performance index. Update
[benchmarks.md](../../benchmarks.md) at command migration to name the profile's
production width, adaptive adoption tracks and explicit fixed-track assessments,
case revision policy and historical format limitations. Record delivered commands
in `CHANGELOG.md` and keep help/examples short enough to use directly.

Later extend that same page for case qualification, source coverage, public export,
thread/binding diagnostics and remote recipes. A benchmark suite needs no new
full-length gallery. This revision changes the SEP and roadmap only; documentation
must not claim a runner or coverage build already exists.

### 9. Compatibility and impact

No physics/default, propagation API, simulation-output schema or kernel-contract
`SCHEMA_VERSION` change is planned. Coverage introspection is additive. No scientific
difference from Luna is introduced.

On adoption, replace the old retire-the-name rule with immutable
`(case_id, revision, track, configuration)` pairing. CLI aliases may point at newer
revisions, but old identities never change. Code provenance and intended experimental
build axes are recorded separately from workload identity. Legacy version-2 records
map to a revision only through an explicit audited mapping of dimensions, method,
boundaries, outputs and policy; otherwise they have `legacy-unmapped` identity and
cannot participate in a new-suite comparison. Minima are never reinterpreted as
medians. A matched resolved default is compatible even if its spelling changes.

Keep the old `poe bench` workflow during initial construction. Switch it at the end
of phase 1 after core qualification, without waiting for native coverage or all
case sets. Preserve legacy `bench-run`, `bench-record`, `bench-compare` through
documented wrappers or retirement messages where meaning cannot be preserved.
Historical SEP/TDSE/haCC campaigns and results remain accessible. Extract reusable
definitions from SEP 16/25/28, but do not equate altered gradients, different step
counts or old timing boundaries with new revisions.

### 10. Alternatives considered

- **Deliver every set, telemetry field and coverage tool before switching commands.**
  This postpones feedback and risks an unused suite. Ship the Kokkos core first and
  add later phases to the same runner.
- **Make five samples at two widths the everyday default.** Keep it as explicit
  standard mode; a quick default lowers the cost of checking a small change.
- **Treat quick results as adoption evidence.** One sample cannot measure noise.
  Quick gives useful measurements, and standard supplies the stronger verdict.
- **Use only pytest-benchmark, ASV or pyperf.** Keep existing component tests, but
  a small domain controller is needed for case identity, reset semantics and
  fresh thread environments. Avoid another environment manager or framework until
  it removes demonstrated work.
- **Force every pool to N or resize each machine's cases.** These change what is
  measured. Keep separate pool diagnostics and immutable workloads.
- **Refuse all contended developer runs.** Keep flagged measurements and defer
  briefly; reserve quiet requirements for clean adoption claims.
- **Regenerate convergence references and exhaustive inventories on every run.**
  Reuse qualified receipts and collect cheap sanity checks; deeper diagnostics are
  explicit. This preserves useful correctness without turning timing into a study.

### 11. Phases

Each implementation slice tests what it changes and reports its
actual verification in Part II, following [testing.md](../../testing.md). Prioritise
an end-to-end core check over completion of optional bookkeeping.

| Phase | Deliverables | Acceptance evidence |
|---|---|---|
| 0 — Revise proposal | This draft, review responses and current roadmap | Diff/spelling, SEP tests and documentation build |
| 1 — Usable core | Registry and module CLI; synthetic worker tests; NumPy reference/check path first, then Kokkos adapter; four cases with expected warnings, quick/standard, profiles, startup/RSS, basic guard, CSV/resume and comparison; first manual/changelog; switch `poe bench` after core qualification | A real quick core run early; frozen core windows/reference bounds for standard adoption; measured quick/standard cost, basic admission and A/A; no wait for full telemetry/coverage |
| 2 — Broaden and refine | Remaining sets incrementally, optional Numba adapter, warning declarations, richer process/topology/library records, improved estimates, exports and Linux workstation qualification | Each added case has source, resolved inputs, admitted tracks/widths and checks; matched profiles, memory-estimator defects and missing data tested |
| 3 — Native coverage | Separate Clang build including Linux OpenMP/atomic counters; instrumented marker; per-case/set source reports and capability map | Known hits/misses, dependency-source exclusion, correct unions and timing refusal; first measured baseline |
| 4 — Close qualification | Remaining broad-suite checks, documentation and historical entry-point cleanup; publish useful baselines and remaining gaps | Required solprop final gate, slow/coverage lanes and Kokkos-selected slow lane; Kokkos gate for native build changes; public exports and phase reports |

The adapter order is a deliberate exception to SEP 0's usual backend order for
new numerical features: NumPy remains the oracle, but Kokkos timing precedes optional
Numba timing because it is the routine target. No new algorithm is implemented in
either backend. Qualification of Kokkos memory occurs with its adapter in phase 1,
not as a promise of an earlier NumPy-only phase. Do not add final-gate work to each
development slice beyond the repository's phase rules.

### 12. Open questions

1. **Core measured bounds/cost, phase 1:** freeze the radial fixed schedule and the
   core adaptive accuracy receipts before standard adoption; measure turnaround and
   correct excess harness cost. Quick runs may precede this with explicit unqualified
   status. No numerical envelope or runtime was measured by this draft revision.
2. **Broader cases, phase 2:** freeze short windows and reference checks as each set
   is added. Assess whether later-state witnesses exercise a useful missing path.
3. **Memory/telemetry, phases 1–2:** begin with conservative existing/empirical
   estimates and explicit unknowns; fix measured underestimates. Record monitor cost
   and optional counters rather than blocking core use on exhaustive accounting.
4. **Coverage, phase 3:** establish the compiled CPU denominator and useful uncovered
   capabilities. GPU, Windows and cloud qualification remain separate roadmap work.

### 13. References

1. solprop, [benchmark policy](../../benchmarks.md), [testing](../../testing.md),
   [SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md),
   [SEP 25](sep-0025-rounding-level-reproducibility.md) and
   [SEP 28](sep-0028-free-space-evaluation.md).
2. solprop-kokkos, [threads, introspection and profiling](https://github.com/LupoLab/solprop-kokkos/blob/main/README.md);
   solprop, [kernel contract](../../kernel-contract.md).
3. psutil, [CPU, memory and process APIs](https://psutil.readthedocs.io/stable/).
4. Clang, [source-based coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html)
   and [profile-update flags](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fprofile-update).
5. LLVM, [llvm-cov guide](https://llvm.org/docs/CommandGuide/llvm-cov.html).

## Part II — Implementation record

### Phase 1 — Core runner development, 2026-10-21

The core development runner is delivered; the SEP remains in progress. Part I
remains frozen. Both repositories use a `feature/sep-0030-phase-1` topic branch.
The maintainer's revised handoff scope below separates trying the tool from
completing machine/adoption qualification.

**Delivered paths.** `benchmarks/` separates registry, builders, planning,
system observations, worker execution, collection, records, comparison and reference
preparation. Metadata-only planning exposes pending broader cases. The core builders
retain the original guides. Local bundles use `.benchmarks/suite/`; `poe bench`
selects the core quick runner and `poe bench-legacy` preserves the version-2
controller. The manual chapter is linked in the header.

`StepDriver.metadata()` adds ordinary RK `source_evaluations`, incremented at actual
dispatch and cleared by `set_initial`. NumPy and Numba count dispatch directly;
solprop-kokkos adds a private native counter for scalar and general states and exposes
it through its adapter. Adams and low-storage ledgers retain their existing meanings.
No public signature or kernel/output schema changes are involved.

**Engineering measurements.** An initial Apple-silicon 18-core, 128 GiB CPU pilot
completed all six adaptive rows. A separate Kerr pilot completed its original
ten-step window after revising an excessively conservative admission estimate.
These are contended development measurements, not qualified baselines or adoption
evidence. They precede final reference and harness identities.

| Case / track | Warm window, s | Worker peak RSS, bytes |
|---|---:|---:|
| RDW / RK | 0.0820 | 439255040 |
| RDW / Multistep | 0.0539 | 454868992 |
| CtC / RK | 2.8577 | 767950848 |
| CtC / Multistep | 1.0138 | 809877504 |
| Radial 256 / RK | 11.1474 | 565575680 |
| Radial 256 / Multistep | 5.3421 | 629260288 |
| Kerr 384 / fixed RK4IP | 11.6597 | 10740252672 |

The six adaptive workers consumed 95.4 s including construction, lead-in, first
window and measurement; the separate Kerr worker consumed 28.0 s. These totals
exclude controller inventory and guard overhead and are not a qualified quick-cost
claim. The radial interval completed without shortening. The fixed RDW diagnostic
at one thread completed ten steps with 61 actual source evaluations. Kerr's ten
measured steps recorded 40 evaluations; its lifetime metadata also includes the
four evaluations in its lead-in.

**Decisions and defects found.** Strict JSON needed an explicit tagged representation
for legitimate infinite material-domain endpoints; numerical samples still reject
non-finite values. The initial Cartesian estimate allowed 48 complex field volumes
plus 2 GiB and refused the Kerr pilot. The revised provisional RK4IP allowance is
24 volumes plus 4 GiB. SEP 28's retained low-storage ledger has five field volumes;
the extra allowance covers construction, NumPy temporaries and qualification copies.
The pilot peak was 10.74 GB, below the revised 38.27 GB budget. This allowance never
admits an adaptive Adams Kerr history. Native pool metadata reported 18 Kokkos and
FFTW threads. Accelerate's internal BLAS width remains system-managed/unavailable.
macOS sandbox-denied swap counters are recorded as unknown; outside the sandbox they
were observable, and actual contention remained flagged.

**Reference pilot and reviewed policy.** Separate NumPy RK runs at `rtol=1e-10`
and `1e-9` covered the exact adaptive windows and lead-ins. Matched-method NumPy
and Kokkos runs used the declared `rtol=1e-6`; no timing-driven tolerance search
was used. Both default integrators were compared against the tighter reference.
RDW retained full-field, weak-bin and 100–150 nm scores; CtC retained field and
weak-bin scores; radial retained the worst-plane in-band field score.

| Case | Largest observed field/band error, both methods | Field refinement floor | Reviewed ceiling |
|---|---:|---:|---:|
| RDW | 2.04464e-5 (UV band) | 1.52789e-11 | 1e-4 |
| CtC | 2.05525e-6 (weak bins) | 4.32305e-11 | 1e-5 |
| Radial 256 | 1.20627e-6 (field) | 2.47262e-10 | 1e-4 |

The maintainer approved measured limits with headroom for ordinary rounding and
solver changes. These use existing source-campaign target levels, approximately
4.9, 4.9 and 83 times the largest pilot errors respectively. They are global result
checks, separate from the unchanged local controller tolerance. Energy differences
remain diagnostic only and use the ordinary field-grid Parseval endpoint weights.
They have no additional acceptance ceiling. A future implementation or reference
bug may require a new reference or accuracy-policy revision; neither renames an
unchanged workload. Frozen measurement hashes prevent unreviewed windows from
inheriting these defaults. Reference preparation is separate from routine timing.

**Maintainer-directed extension.** After acceptance, the maintainer requested an
explicit override so numerical checks do not prevent exploratory timing. The
`--allow-accuracy-failure` option preserves failed checks and measured errors while
allowing collection to finish and exit zero when execution succeeds. It records the
override in the manifest and marks the collection exploratory; crashes, timeouts
and memory refusals remain failures. This is an explicit exception to Part I's
default invalid-result exit rule, not an edit to frozen Part I. Both comparison
arms must use the same reference/policy receipt for adoption. `qualify --recheck`
reassesses retained evidence without propagation; `qualify --supersedes` prepares
a corrected reference. Both preserve old receipts and require a replacement note.

**Verification so far.** The latest affected slice passed 45 tests covering
metadata-only planning, selection, scheduling, calibration, comparison, contention,
atomic writes, locks, resume, crash/timeout recovery, synthetic excessive-history
admission, reference hash invalidation, policy revisions and the accuracy override.
A real NumPy worker was given a deliberately incorrect finite reference: ordinary
mode retained an invalid result and exited nonzero; override mode retained the same
failed check and timing while exiting zero. Six of the tests exercise source counts
across NumPy, Numba and Kokkos, including a forced rejected trial, stage reuse and
reset. The native affected backend module passed 82 tests. The small NumPy/Kokkos
reference workflow and all six adaptive reference pilots completed. The manual
build passed an earlier slice. Ruff and basedpyright are rerun as the slice changes.
Required final lanes and the complete native gate have not run. These checks do
not establish broad-suite or clean machine qualification.

**Sources checked.** Core recipes were checked against `benchmarks/cases.py`,
`benchmarks/sep16/free_space.py` and `benchmarks/sep28/cases.py`; reference refinement
follows `benchmarks/sep16/free_campaign.py`. The existing Malitson recipe and its UV
range warning are preserved; no new optical coefficient was introduced or independently
re-qualified. psutil's upstream release notes were consulted for its development
dependency floor.

**Maintainer-directed developer handoff.** The maintainer requested a runnable
Phase 1 checkpoint before further qualification and bookkeeping. Consequently,
`poe bench` now selects the core quick suite before clean-machine qualification,
an explicit sequencing deviation from Part I. The historical command remains as
`poe bench-legacy`. Missing references do not block exploratory timing, and quick
comparison reports descriptive percentage changes without an adoption verdict.
The terminal table shows warm time, startup, peak RSS, clean-sample counts and
accuracy status; CSV and immutable attempt records remain available for inspection.

The radial fixed diagnostic freezes the first ten accepted steps of the NumPy
default-RK pilot at `rtol=1e-6`. Its first step is 3 micrometres and its endpoint
is 0.00027452363203191686 m. The full 0.03 m guide remains intact. Edges are stored
in the registry and enter the fixed measurement identity without changing the
adaptive identities. Standard planning now admits all twenty core configurations.

A complete seven-configuration Kokkos quick collection finished in 141.07 s,
including environment inventory, startup and guard sampling. All seven workers
completed, with no estimator defects or unreviewed warnings. Warm windows were
0.0546/0.0829 s for RDW Multistep/RK, 0.9742/2.9928 s for CtC, 5.2830/11.3095 s
for radial, and 11.4932 s for Kerr RK4IP. Kerr measured ten steps and forty source
evaluations, with a 10.73 GB lifetime peak. The guard marked every sample contended;
this demonstrates a usable quick cost below the five-minute target on this run,
not a clean performance baseline or a runtime guarantee.

The three fixed-RK diagnostics each completed exactly ten accepted steps with no
rejections on NumPy and Kokkos at one thread. Kokkos dispatched 61 sources and
NumPy 70, reflecting FSAL reuse. Observed relative endpoint field differences were
8.68e-17 (RDW), 7.15e-17 (CtC) and 4.79e-16 (radial). Fresh-versus-reset field,
controller-history and statistic differences were zero in these runs; these are
measurements, not new bitwise-agreement requirements. The fixed envelopes remain
unreviewed. Kerr's fixed window was exercised by the complete quick collection.

Ignored local benchmark storage retains the compact
measurement ledger, hardware/build identities and source fingerprints. Local paths
and host names are omitted. A paired RDW smoke run at one and eighteen threads
also passed after fixing a discovered identity bug: process-local pool widths in
Kokkos `BuildInfo` must not change the extension's build identity. Actual widths
remain in each sample. This is functional paired-run evidence, not a clean A/A
control. Native static typing also exposed an existing macOS affinity guard;
the Linux-only API now has an explicit platform guard.

**Handoff verification.** Ruff fix/format/check and the full solprop basedpyright
scope passed. The affected modules passed 51 tests, and SEP lifecycle checks
passed 224 with two expected skips. The native backend module passed 82 tests;
its pinned Pyright check and C++/Python formatting passed. The manual build,
codespell, whitespace checks and solprop licence check passed. The full gate,
slow/coverage lanes, Kokkos-selected slow lane and complete native gate remain
deferred to the end of SEP implementation, as required for phased slices.

**Remaining qualification and later phases.** These are explicit follow-ups after
the maintainer has tried the tool, not prerequisites for exploratory timing:

1. Promote reviewed reference receipts, including all required reset/band evidence,
   before enabling adoption. Current generated receipts keep adoption disabled.
2. Collect clean A/A controls, full standard costs and monitor-overhead evidence;
   finish thread-cap qualification and compatible empirical memory/cost estimates.
   Current admission uses conservative provisional budgets. Reference checks still
   run after every sample rather than only the first sample of each arm/configuration.
3. Deliver broader workloads, Linux qualification, publication export and native
   coverage in the remaining phases. Stop at this developer checkpoint before
   starting Phase 2. Run final lanes and the native gate before the SEP becomes Final.

### Phase 2 — Broader workloads and Numba, 2026-10-21

The maintainer approved continuing after trying the usable Phase 1 runner and
its local merge. Ordinary benchmark use remains independent of reference review
and machine adoption qualification.

The first increment adds nine definitions to `modal-capillary` and
`mode-averaged`, extracted into `benchmarks/builders.py` without gallery,
legacy warning-filter or test-fixture imports. One resolved mode retains its
transverse quadrature; the averaged helium, argon and hydrogen cases explicitly
select the effective-area route and its statistics. Original full guide lengths,
pressure laws, launches and temporal grids are retained. The two hydrogen variants
use the accepted 6 ps grids. The GNLSE definitions preserve the introduction's
ideal N=3 soliton and SEP 13's Raman/shock inputs.

**Window decision.** The maintainer approved first-ten-accepted-step NumPy RK
pilots for new windows without a specified interval. Pilots use `rtol=1e-6`,
`atol=1e-10`, the original full guide and no intervening observation landings;
a progress callback stops immediately after the tenth accepted step. The stored
edges in `benchmarks/schedules.py` freeze the resulting entrance intervals
and fixed diagnostics. Reduced CtC retains its already specified ten uniform
120 micrometre diagnostic steps. Every new adaptive interval saves its endpoint.
No ordinary run executes a pilot or chooses new edges. All nine pilots completed
without rejection or model warnings; the selected end positions and precision
are retained in the registry, schedules and manual. This defines bounded work,
not an assertion of physical convergence or an accuracy ceiling.

**Numba path.** The existing fresh-process worker is shared by all three CPU
backends. Numba startup includes compilation/cache loading before its first-window
receipt; warm samples reset the existing backend and reuse compiled kernels,
allocations and FFT plans. The existing disk cache remains explicitly uncontrolled.
The real one-thread collection passed all 22 adaptive configurations in these two
sets, including the original core RDW/CtC rows. Its contended measurements are
development evidence, not a clean baseline.

**Verification so far.** Full basedpyright: zero errors. The directly affected
controller, workload, reference, publication and SEP lifecycle modules: 296 tests
passed, with two intentional process-SEP skips. Ruff, codespell, diff checks and
the strict manual build passed. The new builder
checks cover full-guide preservation, dimensions, resolved/averaged distinction,
frozen schedules, field retention and named refusals. Small worker integration
checks exercise both default methods on NumPy and Numba, including reset fields,
controller/history/statistics and exact work counts. Separate tight-RK (`rtol=1e-10`),
refinement (`1e-9`), matched NumPy and Kokkos runs completed for all 18 added
adaptive configurations. The largest relative field error was 6.013e-7 and the
largest reference-refinement difference was 2.897e-10. All reset field differences
were zero in this measurement. The maintainer approved a deliberately loose
`1e-4` relative field ceiling for these nine cases. Literal measurement hashes
bind the review to the unchanged windows; rechecking retained evidence writes
replacement receipts without running simulations. Adoption remains disabled in
these receipts. The capillary/GNLSE settings now apply the declared field-retention policy
for the fixed diagnostic as well as the adaptive path.

**Sources.** Inputs were checked against the public source files named by each
registry entry. The GNLSE higher-order coefficients retain SEP 13's primary
Dudley–Genty–Coen citation (doi:10.1103/RevModPhys.78.1135); no new material values
were introduced and no independent remeasurement of those material data is claimed.

**Free-space increment.** Six more first-ten-step pilots froze the radial-512,
Cartesian-128/256, line, planar and radial-absorber windows. Each took ten accepted
steps without rejections or warnings. The Cartesian-256 NumPy pilot peaked at
15,975,038,976 bytes; admission budgets distinguish its Adams histories from RK
registers. Kerr-192 and Kerr-768 retain the accepted ten/five-step RK4IP windows
after one lead-in step, with the explicit one-thread exclusion for 768. The full
guides and source grids remain unchanged. Reference checks for these additions
remain in progress; timing does not require invented ceilings.

**Resource views and export.** Normalized local tables now separate runs, cases,
configurations, systems, allocations, processes and libraries. Failed process
polls retain unknown CPU/RSS values with reasons. Pool introspection after timing
also captures lazily loaded backend libraries. Explicit publication export writes
a new whitelisted JSON/CSV snapshot and replaces the local machine alias; it
does not copy raw logs, paths, process identities or environment values, and it
does not commit or upload anything. The Linux CPU build completed in an isolated
allocation; exclusive timing qualification has not yet run.

**Advanced increment.** Eight definitions now retain the wall/taper tutorials,
general step-index pair, linear/nonlinear GRIN and moving solid guide. Independent
NumPy pilots freeze the first ten accepted steps. Linear GRIN had three rejected
trials; the others had none. Original wall diagnostics remain ten uniform
20 micrometre steps. The linear continuous-wave launch stays configurable until
translation; the nonlinear GRIN input noise retains seed 41. Moving-guide schedule
tails land on the original 5 mm taper corner. Expected weak-absorber and grazing
warnings are declared narrowly. Inputs were extracted from the named public
examples and checked against them; their material laws are reused, not refitted.

Qualification exposed an intentionally undefined spectral centroid in the empty
step-index mode. Reset reports now retain undefined statistic masks and compare
their finite entries; non-finite histories, fields and infinities still fail.
Regression tests cover undefined masks, missing values and rejected infinities.
Richer topology records and a separate `diagnose` command record per-thread CPU
time, Linux allowed CPU sets and NUMA pages outside ordinary timing. Missing
telemetry retains a reason, and affinity is not presented as execution residency.
The advanced increment passed full basedpyright, Ruff and 314 affected tests
(two intentional process-SEP skips). All sixteen advanced adaptive configurations
then completed on Numba. Tight RK, refinement, matched NumPy and Kokkos checks
completed for the eight advanced cases plus line, planar and radial absorber.
Across these 22 configurations, the worst field error was 1.013e-6 and the largest
reference-refinement difference was 1.549e-8. The maintainer approved a loose
`1e-4` field ceiling for these eleven cases. Rechecked receipts preserve the old
evidence and advance its policy/reference lineage; all 22 passed. Literal
measurement identities prevent changed workloads from inheriting this approval.

**Backend exercises.** Separate builders extract vector rotational Raman,
envelope plasma, a helium/argon pressure gradient, dispersive Kerr, the band and
trapezoid plasma paths, noise, dense output, split-step absorbers and ordinary and
far-field statistics. The maintainer approved current optmat rotational hydrogen
data with vibration disabled, preserving the propagation fixture's circular
pump and delayed oblique probe. Its low-density linewidth warning is recorded
by phase/category/message. Other source inputs and models remain as declared by
the registry; material laws are inherited without refitting.

Ten accepted default-RK steps freeze each new entrance window, except the original
8 micrometre envelope-plasma guide, which ends after four 2 micrometre steps.
The maintainer explicitly approved retaining those four steps. The remaining
pilots had no rejections; only vector Raman emitted a model warning. Dense output
uses eight interpolated observations. Far-field statistics preserve the source
224-by-64 spatial grid, apertures, spectral regions and periodic absorbed power,
with eight statistics planes but only the measured exit field retained.

The batch exercise uses two distinct GNLSE members through the real batch backend
and ten fixed RK steps. Its base lockstep controller and individual members reset
through existing initialisation, retaining native/Numba buffers and FFT plans.
Each member has independent field checks, statistics and actual source counts.
Reset evidence now includes periodic statistics as well as ordinary statistics
and method history. These are harness changes, with no numerical default or seam
schema change. Pilot and reference evidence remains in ignored local benchmark storage.

The exercise increment passes Ruff, full basedpyright and 321 affected tests,
with two intentional process-SEP skips. Tests include real batch reset on all
three CPU backends, statistics-only field retention, the four-step schedule and
dense observation without changing the accepted trajectory or source counts.
Full-sized benchmark campaigns remain outside ordinary test lanes. All 23 exercise
reference configurations completed on NumPy and Kokkos, including the fixed batch.
The eleven adaptive cases had a worst field error of 3.181e-7 and a largest
refinement difference of 5.374e-11. The maintainer approved the same loose `1e-4`
ceiling; all 22 adaptive configurations passed a retained-evidence recheck.
All 23 Numba exercise configurations then completed; the 22 adaptive checks
passed the reviewed ceiling, while batch retained its matched-method diagnostic.

**Storage decision.** The maintainer clarified that ignored `.benchmarks/` storage
is appropriate, but this is a development tool rather than a repository archive.
Generated timing, reference evidence and coverage reports are local artifacts;
implementation snapshots were moved out of the tracked tree. The registry,
accuracy policies, tests and concise implementation notes remain versioned.
Coverage is an explicit on-request diagnostic with no scheduled notifications.
Historical version-2 ledgers are preserved until the follow-up migration to the
separate development repository requested by the maintainer. That migration covers
historical scripts, ledgers and development evidence after this implementation;
the roadmap records the import/link audit and entry-point guidance it needs.
Part I remains frozen; this later
maintainer direction supersedes any planned commitment of generated baselines.

**Remaining work.** Broader backend/reference checks; exclusive Linux qualification; native
coverage; and the final SEP-wide gates and closing report remain in progress.
Part I is unchanged.

### Phase 3 — Native coverage implementation

**Delivered paths.** The separate native Clang CPU build instruments project core,
bindings and instantiated headers with atomic counters, without instrumenting its
Kokkos Tools label collector. Additive `BuildInfo` fields identify instrumentation
and the source digest. Normal timing refuses an instrumented binary; coverage
refuses a normal binary. No simulation, saved-output or kernel-contract schema
changes are involved.

`benchmarks/coverage_worker.py` executes construction, the declared lead-in,
one window and observations without timing samples or reset repetitions.
`native_coverage.py` admits memory before construction, retains failures, verifies
binary/source/profile identities and uses matching LLVM tools. Each worker has a
process/module-unique raw profile destination. Case, set and union reports contain
project-only HTML and line/function/branch CSV views. `coverage_reports.py` excludes
dependency templates and keeps LLVM's line counts. A separate execution pass records
Kokkos labels beside declared capability tags; those profiles do not enter coverage
unions. Reports are local and ignored, with a redirectable terminal summary.

Uncovered mapped CPU code is counted. Unsupported execution is reported separately;
CUDA/HIP are compiled out of this CPU build. LLVM does not enumerate uninstantiated
templates or every compiled-out source path: these counts remain unavailable with
a reason rather than being reported as zero or missed executable code. No coverage
target has been invented.

**Verification so far.** Thirteen focused tests pass with an actual Clang fixture
containing known hits and misses, including failed profiles, dependency exclusion,
unions and changed binary/source refusal. The normal-environment controller and
coverage tests pass together (44 tests); full basedpyright and Ruff pass. Native
binding-marker and concurrent label-collector tests pass. End-to-end smoke and
batch/dense collections generated local reports and separate labels. These validate
the reporting paths, not a whole-suite coverage baseline. Full native and final
SEP-wide gates remain outstanding.

**Sources checked.** Build flags and profile/tool handling follow the official
[Clang source-based coverage workflow](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html)
and its
[atomic counter option](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fprofile-update).
The label ABI was checked against the pinned Kokkos Tools C interface. Coverage
describes this CPU binary and selected workloads, not unsupported device execution.

### Phase 4 — Closing validation and integration

Phase 3's strict documentation build and full type check passed. The affected
controller, coverage and SEP checks passed 240 tests with two expected process-SEP
skips. The native gate passed formatting, licensing and typing, then identified
an existing redundant empty lambda parameter list in the general workspace.
The final native branch removes that syntax without changing its arithmetic;
the gate continues from the failed build step.

**Reference use.** The first successful result for each arm/configuration retains
its independent numerical check. Later samples check current output shape and
finiteness and identify the original validation sample. They do not repeat its
measured errors as if they were newly measured. Changed reference/policy identities,
failed samples and overridden checks require a new full check. Environment identity
checks continue for every worker.

`qualify --recheck --reviewed` now closes the earlier temporary adoption block for
retained evidence that has been reviewed. All four reference runs must have succeeded
and retained reset fields, histories/statistics and work counts; reset fields must
meet the approved field ceiling. The explicit review covers the case-dependent
history and statistic comparisons, without inventing a universal rounding tolerance
or requiring identical adaptive trajectories. Changed policies do not inherit an
earlier review automatically. This is separate from ordinary timing and its override.

The following closing evidence completes the local lanes and reports the scope
of the Linux and native coverage collections. Historical benchmark migration
remains the separate follow-up recorded in the roadmap; no generated result is
being committed.

The native gate completed debug/tidy and sanitizer builds, 338 Python tests with
six expected skips, and the distribution wheel and clean smoke test. The Python
gate identified missing parameter/return sections in three existing private
OpenMP helpers; only their documentation changed before continuing the gate.

A real cancellation exposed an interrupted-worker recovery defect: a request file
could survive without its completed attempt shard, and the collector reused that
attempt number. Resume now advances beyond existing shard numbers and occupied raw
directories. The regression interrupts collection after a completed sample, retains
the unfinished request unchanged, and checks fresh attempt numbers and skipped
completed samples on repeated resume. The failed attempt and diagnostics from the
observed failure were retained; the resumed Linux broader quick selection completed.

The parent now records its launch/monitor CPU time separately from worker timing.
Two short GNLSE diagnostic workers used 6.02 ms and 5.25 ms of parent CPU across
2.83 s and 1.97 s elapsed, with 31 and 23 process samples respectively. These
contended local observations measure controller CPU cost, not a causal bound on
measurement disturbance. Their generated receipts remain in ignored storage.

Metadata-only planning now prints conservative memory budgets and distinguishes
missing, compatible, reviewed, incompatible and malformed reference receipts.
Array hashes and numerical results remain worker checks; no scientific import or
simulation construction occurs during planning. Cost forecasts remain explicitly
unknown when no compatible estimator is available.

The default lane completed 16,762 passing tests and 30 expected skips, with 23
queue/Slurm failures caused by sandbox denial of the `ps` ownership query. Rerunning
the two affected modules with process inspection permitted passed all 62 tests.
The changed benchmark/reference and SEP modules passed 279 tests with two expected
process-SEP skips. The latest full type check passed without errors or warnings.
The strict documentation build, including tutorials, completed successfully.

**Maintainer-approved closing scope, 2026-10-21.** Finish the existing Linux
comparison and native coverage collections, review the refreshed core references,
and repair the targeted Python coverage collection. Additional clean macOS
standard/A/A campaigns, the largest spatial reference checks, runtime forecasts
and automatic reuse of measured memory receipts move to the roadmap. The
maintainer requested that a fresh macOS baseline and a Numba-versus-Kokkos CPU
comparison wait until concurrent performance changes have merged to `main` and
the benchmark checkout and native build have been updated. Numerical reference
preparation is separate from that performance baseline. No new long macOS timing
campaign is required to close this implementation. This later direction changes
the delivery scope without editing frozen Part I.

All six refreshed core adaptive references passed the approved envelopes and
review of reset fields, controller histories, statistics and work counts. Together
with the 62 broader adaptive references already reviewed, 68 adaptive
configurations have reviewed local reference evidence. Timing still works without
references, and changed numerical assumptions require reassessment rather than
inheriting a previous review.

The default slow lane passed 148 tests with one expected skip, followed by its
statistics-cost test. The Kokkos-selected slow lane passed all 149 tests and its
statistics-cost test. The Python coverage run encountered the same sandbox-blocked
queue/Slurm process queries as the default lane; the targeted rerun passed all 62
tests and restored their coverage. The remaining uncovered paths were Windows
pipe typing and process-handle declarations, addressed with focused portable tests.
That module passed 35 tests with one expected real-Windows-query skip. Combined
coverage reached 100% of 43,748 statements and 13,296 branches without exclusions
or changes to the runtime code.

**Linux collection.** The exclusive, verified one-socket allocation completed
seven core quick configurations and 38 broader quick configurations, all with
successful clean samples. The broader bundle retains the interrupted-attempt
failure and its successful retry. This collection used the phase 2 advanced-guide
increment; it does not establish Linux qualification of the later backend
exercises or native coverage build. Sanitized exports were checked and remain
in ignored local storage.

The standard A/A calibration completed 40 successful pilots. Summed startup and
warm measurements cost 43.6 minutes; the single-thread 384-square Kerr warm
window alone took about 384 seconds. Five comparable rounds plus calibration
were therefore likely to exceed the four-hour allocation and did not meet the
original turnaround target. At the maintainer's request, the A/A job was stopped
after six successful measurement samples. Completed attempts and the interrupted
request were preserved; comparison reports `incomplete`, not a repeatability
baseline. The allocation was released. A fresh baseline waits for performance
integration, as recorded in the roadmap; this implementation does not shorten the
workload or change its meaning to meet the target.

**Native coverage baseline.** The Apple-silicon Clang CPU collection attempted
114 configurations at two threads. It completed 106 profile windows and their
separate label passes. A successful one-thread 384-square Kerr retry adds one
configuration; the combined report covers 38 cases and retains all 115 outcomes.
Its 45 per-case, per-set and union reports passed source, binary and profile checks.

| Compiled project coverage | Covered / mapped | Fraction |
|---|---|---|
| Lines | 12,901 / 19,014 | 67.85% |
| Functions | 713 / 1,039 | 68.62% |
| Branch outcomes | 3,365 / 5,828 | 57.74% |

Six two-thread configurations reached the 600-second limit: the 384-square Kerr
beam, all three 128-square Cartesian tracks, and the 256-square Cartesian RK and
fixed-RK tracks. The 256-square Cartesian Multistep and 768-square Kerr cases
were refused before construction under the 48 GiB memory cap. These outcomes
leave the selection explicitly incomplete; they are not missing profiles treated
as successful coverage. The one-thread retry supplements the union without erasing
the original Kerr timeout. All 107 successful configurations produced labels.

This diagnostic baseline uses Python `a05380b9` and native `1e92e27`, with matching
LLVM 23.1.3 tools. It describes that compiled CPU binary, not the later performance
integration or device code. Dependency sources are excluded; uninstantiated and
compiled-out paths retain the limitations described in phase 3. No coverage target
is imposed. The complete reports, profiles, binary snapshots and failure records
remain in ignored local storage; only this concise result is versioned.

**Integration for merge.** The final Python branch contains the remaining phase
histories and integrates `main` at `00a9524d`; the final native branch integrates
`main` at `45cc8e0`. A fresh normal native wheel built successfully against this
Python tree. The affected benchmark, reference, memory and resident-Adams modules
passed 287 tests; three native API/build-metadata tests passed. Ruff and the full
Python type check passed. These are focused integration checks: the full lanes
reported above describe the earlier implementation tree and were not repeated
after integration, in accordance with the maintainer's gate-confirmation policy.

A separate instrumented wheel of the integrated native tree also built and
completed a two-thread diagnostic coverage window with source reports and labels.
Four native build-metadata/label tests passed, including concurrent label counting.
An actual instrumented import was refused by the timing path. This checks coverage
support after integration; its profiles are separate from the earlier broad CPU
baseline because the source and binary identities differ.

The closing documentation build exposed broken section links in the integrated
performance notes and the archive move. Correcting the targets produced a clean
strict build. The lifecycle checks passed 224 tests with two expected process-SEP
skips, and the default metadata-only benchmark plan completed successfully.
The subsequent documentation-only `main` update at `5a7d93d8` is also integrated;
its changelog include supports repository-relative links, resolving the overlapping
link repair without changing the implementation tested above.

The remaining work is recorded in the [roadmap](../../roadmap.md):

1. Fresh machine baselines and the requested Numba/Kokkos CPU comparison after
   performance integration, with campaign scope confirmed by the maintainer.
2. Independent references for the largest spatial cases, plus optional runtime
   forecasts and reuse of empirical memory receipts.
3. Linux native coverage and separate Windows, cloud and GPU qualification; CPU
   source coverage does not establish device coverage.
4. Migration of historical benchmark material to the separate development
   repository, with import/link audits and retirement guidance.

These agreed follow-ups do not block ordinary development timing or on-request
coverage. Merge the native final branch before the Python final branch; the latter
contains the phase 2 and phase 3 histories, so separate merges of those Python
branches are unnecessary. No main-branch merge or publication is claimed here.

## Part III — History

- 2026-10-21: Completed the implementation within the maintainer-approved closing
  scope and prepared the Final archive transition on the final topic branch.
  Remaining machine campaigns, large spatial references, estimator conveniences
  and historical-material migration are recorded in the roadmap. Main-branch
  integration and maintainer approval of the Final transition remain pending.
- 2026-10-21: Drafted from the agreed benchmark-suite design. Kokkos CPU is the
  routine timing backend; fixed workloads, both timing tracks, startup and peak memory,
  native coverage and benchmark-only psutil were selected. Initial qualification is
  macOS on Apple silicon followed by a dual-socket x86-64 Linux workstation. At
  drafting, the proposal was unreviewed and Draft.
- 2026-10-21: Reviewed by Opus 5.5 (AI), with John Travers's human review recorded at
  the maintainer's request; the maintainer's decisions on R1–R4 are recorded beside
  those points. Findings follow. Review status is Reviewed; lifecycle status remains Draft. Part I has not
  been edited by the reviewer.
- 2026-10-21: Revised Part I in response to R1–R16 and the maintainer's usability
  priority. Quick mode is now the default; standard mode is explicit; Kerr uses
  fixed RK4IP only. The first phase delivers a useful Kokkos core before broad
  telemetry and coverage. The response below records every disposition. Status
  remains Draft; the revised text awaits further review.
- 2026-10-21: John Travers read and accepted the revised proposal (`c34fead5`) and
  requested its merge to main. Status is Accepted and Part I is frozen. The
  qualification questions assigned to implementation phases in §12 remain there;
  acceptance does not claim measured timings or completed implementation.

### AI review — 2026-10-21

**Reviewer:** Opus 5.5 (AI). **Disposition:** revisions requested before acceptance.
The draft is a sound and carefully bounded measurement design: fixed workloads,
interleaved fresh-process samples, explicit timing boundaries, honest memory
semantics and crash-safe records answer real problems of today's scattered
harnesses. Its main gaps are what the core actually times (which integrator, which
thread width counts as production), what the suite costs to run, and a few
conflicts with existing rules and harnesses that it means to absorb.

This review covers all of Part I against SEP 0, [benchmarks.md](../../benchmarks.md),
the roadmap and the harnesses the draft names. Code evidence is from solprop commit
`774d0ef5` and solprop-kokkos commit `e62c6f2`. No benchmark was run for this
review. **P1** marks an acceptance-contract or design issue to settle before
acceptance; **P2** a specification gap to close in the revision; **P3** a
clarification. The drafting agent should answer every R-number in a separate
`Response to review` subsection, naming the changed sections or giving the reason
for keeping the text. The maintainer's decisions settle policy; they are recorded
under the points they answer.

#### R1 [P1] — Name the integrator of every track; make the adaptive tracks primary

**Sections:** §1, §4.1, §5.1 (set and window tables), §5.2, §7.2.

Neither table says which integrator a case is timed with. The existing fixed windows
(`benchmarks/cases.py`, `Case.simulation`) replay `RungeKutta()` on a frozen
schedule, and the Adams march cannot replay a schedule at all (`adams_simulation`
in the same file; `bench_adams.py` pins `Multistep(formulation="interaction")` and an
adaptive lead-in). The package default is `Multistep()`
([tuning guide](../../../guide/performance/tuning.md)). As drafted, `core`'s headline
"warm performance" could therefore be the per-step cost of a non-default stepper,
and an optimisation of the default Adams path could go unmeasured by the core.

The two tracks answer different questions, and the SEP should say so. A fixed window
does identical work on every repeat, so it measures per-step cost (source, linear
propagator, stage arithmetic) free of controller effects: the low-noise measure for
kernel and source work. An adaptive window measures the time to a qualified result,
which is what a user gets, but its work changes whenever rounding changes the
controller's accepted and rejected steps.

**Maintainer decision (2026-10-21):** the priority is to benchmark the default
`Multistep()` and `RungeKutta()`, both adaptive. Every core case carries both
adaptive tracks (time to the qualified result, with actual work counts), and these
are the primary metrics and the ones the adoption aggregate is computed over. The
fixed RK (or low-storage march) window stays in `core` as a per-step-cost diagnostic,
reported beside the adaptive rows but excluded from the adoption aggregate.

**Requested change:** add integrator and formulation columns to the case table, per
track, using the default `Multistep()` as a user constructs it (not a pinned
formulation, unless a second row names it); give each core case's adaptive interval,
lead-in, tolerance and accuracy envelope; and restate §1, §4.1 and §4.3 so that the
adaptive tracks are primary and the fixed window is diagnostic. Where the default
stepper is refused or impractical (for example the large Kerr beams, where the
resident Adams history exhausted 64 GB on Kokkos, as the roadmap records), the case
says so and names its adaptive substitute, or has no adaptive track.

#### R2 [P1] — Declare the production width per machine; default to two widths

**Sections:** §1, §4.3, §5.2, §5.3, §7.3.

[benchmarks.md §4](../../benchmarks.md) judges adoption at the "production thread
policy", which it and SEP 28's matrix take as every core of the node with the
backends' default pools. The draft lets the caller declare a production width before
collection, but its default matrix (1, 4 and 16 threads) contains no all-cores row,
and nothing ties a declaration to a machine. On a heterogeneous Apple-silicon host
(for example 6 + 12 cores in two performance levels, 18 in all), 16 unbound threads
span both core classes and 4 land wherever the scheduler puts them, so a "4" or "16"
row means different hardware on different Macs.

**Maintainer decision (2026-10-21):** a per-machine profile declares the production
width. By default a run measures two widths only: one thread (critical) and the
machine's declared production width. Other widths, including 4 and 16, are explicit
selections.

**Requested change:** introduce a versioned machine profile, keyed by the machine
alias (§6.3), that declares the production width and is recorded with the run; make
`--threads` default to `1,<production>`; refuse an adoption verdict when the two arms
were collected under different profiles; record on heterogeneous hosts how many
workers ran on each core class where the platform exposes it, or that it does not;
and amend the benchmarks.md criterion in §8's documentation plan to name the profile
as the production configuration.

#### R3 [P1] — State a cost budget; reduce the default sample count

**Sections:** §2, §4.2, §5.1, §5.2, §11.

The draft says the core "must be affordable during optimisation" but sets no budget.
Each sample is a fresh process with imports, construction, translation (the full CtC
grid is 32 768/65 536 points over six modes), warm-up, reset and a calibrated
measurement of at least three seconds, after up to five seconds of guard intervals.
At four cases, three widths and seven samples that is roughly 45–60 minutes before
noisy configurations extend; `all` (some forty cases) would take many hours. It also
schedules the 768² Kerr beam at one thread for ten steps, which SEP 28's matrix
excluded because those windows alone would exceed a phase's budget (and ran five
steps, not ten, at production width).

**Maintainer decision (2026-10-21):** the core should complete in about 30 minutes on
a quiet machine; the default is five fresh-process samples per configuration,
extended in complete paired rounds when the noise indicator exceeds its threshold
(the reviewer suggests a ceiling of eleven); and two widths by default (R2).

**Requested change:** state the core target and require `list`/`--dry-run` to print a
per-set wall-time estimate from stored pilot timings; let each case declare the
widths and tracks it admits (so the 768² beam need not run at one thread), with a
declared exclusion recorded rather than silently omitted; reconcile the 768² window's
step count with SEP 28's; and re-derive §4.2's sampling and extension rules for five
samples. Consider whether the three-second calibration floor is needed for windows
already longer than the timer resolution needs: it multiplies the cost of every
short case.

#### R4 [P1] — A refused guard defers a sample; it never fails the session

**Sections:** §4.2, §6.2, §7.3.

As drafted, a guard failure after its sixty-second maximum wait refuses a clean
sample, and a refused or contaminated scheduled attempt leaves the session
"incomplete for a clean verdict". On a developer machine in daytime use most samples
would be refused, so the macOS qualification of §7.3 could not complete.

**Maintainer decision (2026-10-21):** a sample whose guard is not met within the wait
is deferred to the end of its round and retried (a bounded number of times). If it
still cannot be taken quietly it runs anyway and is kept, flagged as contended. A
contended sample never fails the run: it is useful to record. Clean baselines are
expected from quiet sessions; a flagged sample is excluded from a clean adoption
verdict but stays in every table.

**Requested change:** rewrite §4.2's "failed or contaminated" sentence and §6.2's
admission paragraph to this rule, distinguish *contended* (kept, flagged) from
*failed* (no valid measurement: crash, timeout, validation failure); define what
fraction of flagged samples turns a comparison's verdict into "unresolved" rather
than invalidating it; and make `--force` an explicit "exploratory" label for a
whole run rather than the only way to obtain a contended sample.

#### R5 [P2] — Reconcile case revisions with the rule that a changed case retires its name

**Sections:** §5.1, §9.

[benchmarks.md §4](../../benchmarks.md) says that a changed definition retires a case's
name, and the existing harnesses follow it (SEP 28's `cases.py` cites it). The draft
instead keeps stable CLI names (`ctc`) that resolve to explicit revisions. Both work,
but the SEP must say that it replaces the old rule, that `compare` pairs on
(case, revision, track, configuration) and refuses a pairing across revisions, and
how a historical version-2 ledger row is mapped to a revision (or declared to have
none). State also which identity changes are compatible for comparison, for example a
changed default that the case resolves to the same value.

#### R6 [P2] — Specify warning handling instead of the category filters it inherits

**Sections:** §5.1, §7.2.

§7.2 says "no blanket warning suppression is part of the suite", but every harness it
consolidates ignores `MaterialRangeWarning` and `SolpropWarning` by category
(`benchmarks/cases.py`, `sep25/cases.py`, `QUIET`). Say what replaces it: each case
declares its expected warnings (category and a message pattern), the worker records
every warning raised with its count, an unexpected warning marks validation as
failed or flagged (state which), and the expected ones stay visible in
`accuracy.csv` rather than being silenced.

#### R7 [P2] — SEP 25's truncated-gradient windows conflict with §4.1

**Sections:** §4.1, §5.1, §9 (migration table).

§4.1 rightly forbids shortening a pressure-gradient guide with its endpoint
pressures retained. The SEP 25 cases that §9 folds into the standard tracks do
something different but still changed: `sep25/cases.py:truncated_gradient` shortens
the guide and recomputes the end pressure from the quasi-static flow law. That is the
same problem only when the case's gas model is exactly that law. State whether such
windows are admissible (and under what demonstration of identity) or are redefined
as windows of the original guide under new revisions. The resolved
`mode-averaged` helium RDW and argon blue shift cases in §5.1 should name which form
they take.

#### R8 [P2] — Give the sources of the cases that have none in the tree

**Sections:** §5.1, §7.2.

Several §5.1 cases have no definition the reviewer could find: the hydrogen Raman
workload "with a 6 ps window" (the worked example
`docs/examples/processes/plot_05_raman_self_frequency_shift.py` uses 3.5 ps, and no
benchmark defines 6 ps), the "resolved argon blueshift" in `modal-capillary` (SEP 25's
`ar_blueshift` covers 4 cm with 21 planes, which is an end-to-end run, not a window), and
the mode-averaged argon and hydrogen variants (SEP 25 defines a mode-averaged helium
RDW only). The GNLSE pair presumably comes from
`benchmarks/sep13/gnlse.py`; say so. The drafting agent should name each
case's source (example, tutorial, test or witness), or give its parameters in the
SEP, and say why a 6 ps revision is needed alongside the 3.5 ps one. A table giving
each case's source, route, grid, integrator(s) and qualification reference would
answer this and R1 together.

#### R9 [P2] — Define the repeat-within-sample and reset semantics precisely

**Sections:** §4.1, §4.2.

$t_i = T_i/k$ with reset and lead-in excluded from $T_i$ implies that each of the $k$
windows is timed on its own and the times summed. Say so, and store the $k$
per-window times, since the first window after a reset can differ from later ones
(plan reuse, caches, first touch of history buffers). Say what "reset" restores for
each stepper: for the adaptive Multistep track, the step-size controller state and
the Adams history after the declared lead-in; for RK, the controller's step proposal.
State that a reset must restore exactly the declared state, and that §7.2's "state
resets against fresh backend construction" check covers both the field and the
controller state, within numerical bounds. Since adaptive windows become primary
(R1), define how their accepted and rejected steps are reported when they differ
between repeats of one sample.

#### R10 [P2] — Define comparison criteria for memory and startup

**Sections:** §2, §4.3, §4.4.

The motivation asks three questions (faster, more memory, slower first result), but
§4.3 defines a verdict for timing only. Specify how peak RSS and startup are
compared: per-case ratios of the maximum (memory) and median (startup), a reporting
threshold for a regression (the reviewer suggests flagging a peak-memory increase
over 5 % or 64 MiB, and a startup regression over 10 %), whether either can block
adoption, and how admission refusals in one arm but not the other are reported.

#### R11 [P2] — Specify the A/A statistic

**Sections:** §4.3.

"A difference inside the measured repeatability spread is unresolved" needs a
definition: which statistic of the adjacent baseline-to-baseline ratios is the
spread (for example the median absolute log-ratio, or a quantile), over how many
rounds, and how it combines with the conservative ratio of the adoption criterion so
that the two cannot disagree silently.

#### R12 [P2] — Memory admission when no estimate exists

**Sections:** §6.2, §11 (phases 1–2), §12 Q2.

Phase 1 promises "resource qualification of frozen core windows", but §6.2 admits
against an estimate that existing estimators do not fully provide, and Kokkos adapters
arrive only in phase 2. State the rule when a configuration has no estimate (refuse
without an explicit selection, or admit and record "unestimated"), and that an
estimate below an observed peak is recorded as an estimator defect. The roadmap's
384² Kerr beam on the default stepper, killed for memory on 64 GB machines on Kokkos,
is the test case this rule must handle.

#### R13 [P2] — The coverage toolchain on Linux

**Sections:** §6.4, §11 phase 3.

Linux timing builds use GCC 13 by default (solprop-kokkos README), whereas
source-based coverage needs a Clang build with its own OpenMP runtime. State that
the coverage build on Linux is a separate Clang configuration, record its
optimisation level and OpenMP runtime, and use atomic profile counters
(`-fprofile-update=atomic`) for threaded runs. Say that Kokkos, Kokkos-FFT and
Kokkos Kernels templates instantiated inside project translation units are filtered
from the denominator by source path, so "instantiated project headers" does not take
in third-party template code.

#### R14 [P3] — The runner's name before migration

**Sections:** §5.2, §9.

`poe bench` already runs `benchmarks/bench.py`, and §9 switches it only after
qualification. Name the command that phases 1–3 use (for example
`python -m benchmarks`), so that the existing ledger workflow is untouched
until phase 4.

#### R15 [P3] — `--seed` is ambiguous

**Sections:** §4.1, §5.2.

§4.1 makes a case's noise realisation part of its identity, and §5.2 offers `--seed`
without saying what it seeds. State that it seeds only the schedule shuffle, and that
a case's physical noise seed is fixed by its revision.

#### R16 [P3] — Smaller points

- §5.2: `--all` and a set named `all` are both mentioned (§5.1 "`all` is the
  deduplicated union"); keep one spelling.
- §5.2: the example `--threads 1,8,16` should follow the new default (R2).
- §6.3: say where the sanitised public export lives and in what format, and how its
  machine alias relates to `SOLPROP_BENCH_MACHINE` and today's
  `benchmarks/results/<machine-id>/` ledger.
- §8: name the manual page (for example `docs/guide/performance/benchmarking.md`), as
  SEP 0's acceptance criteria ask.
- §4.4: list FFTW wisdom, the Numba cache and the Kokkos initialisation as parts of
  the recorded cache state, since each moves startup latency.
- §7.3, §11 and Part III: the public-tree scan flags the draft's names for the two
  qualification hosts as references to particular machines (SEP 0 §7). Name host classes
  instead (an Apple-silicon macOS host; a dual-socket x86-64 Linux workstation under
  an exclusive allocation).
- §11 phase 2: given SEP 29's direction for `numba_cpu`, consider ordering the Kokkos
  adapter before the optional Numba one.

### Response to review — 2026-10-21

**Drafting agent:** GPT-6. Part I has been revised; the review above is retained.
Lifecycle status stays Draft and the recorded reviews remain in the header. The
revision has not been independently re-reviewed. No benchmark or numerical
qualification was run for this documentation change.

**Maintainer direction for this revision:** prioritise a fast, useful core runner
over scientific bookkeeping. The maintainer selected quick mode as the everyday
default and standard mode explicitly, and directed that Kerr cases use fixed RK4IP
only. That later decision supersedes the adaptive-Kerr part of R1. New numerical
envelopes are not invented here: they are a phase 1 qualification prerequisite for
standard adoption, not for a visibly unqualified quick measurement.

1. **R1 — Addressed, with the maintainer's Kerr exception (§§1, 4.1, 4.3, 5.1, 7.2).**
   Named adaptive Multistep (public default, currently exponential) and RK (DOPRI5)
   tracks and their core intervals, lead-ins and local tolerances. Fixed RK is
   diagnostic. All Kerr sizes use only low-storage RK4IP, with no adaptive substitute.
   Adaptive tracks determine the adoption aggregate; an affected fixed-only set has
   a separate assessment. The core table names reference metrics and source protocols;
   numeric envelopes for the newly combined windows remain explicitly unqualified
   until phase 1, and cannot support a standard verdict before then.
2. **R2 — Addressed (§§4.3, 5.2–5.3, 7.3, 8).** Versioned machine profiles fix
   production width. Standard defaults to one plus production, quick to production
   alone per the later maintainer choice. Profile mismatches refuse adoption.
   Core-class residency is reported when exposed, otherwise unavailable. The
   documentation plan updates the old production-width criterion.
3. **R3 — Addressed (§§2, 4.2, 5.1–5.2, 11).** Added five-minute quick and
   thirty-minute-per-arm standard targets, honest pilot-based estimates, five
   standard samples and an eleven-round ceiling. Removed the three-second minimum;
   only unusually short windows repeat, with a capped 0.1-second timer-based target.
   Restored five steps for 768-square RK4IP and declared its one-thread exclusion.
   New-suite 384-square windows keep ten steps at both widths; historical shorter
   rows are explicitly incompatible. These costs are targets awaiting measurement.
4. **R4 — Addressed (§§4.2, 6.2).** One bounded deferral/retry, then measure and flag
   contention. Quick never waits. Failed measurements and contended valid results
   are distinct. Clean verdicts need five clean paired rounds and no more than 20%
   contention per arm, otherwise unresolved at the ceiling. All measurements remain
   visible; `--force` marks the whole run exploratory.
5. **R5 — Addressed (§§4.3, 5.1, 9).** Explicitly replaced retire-the-name with
   immutable revision pairing. Identical resolved defaults remain compatible; actual
   workload changes do not. Legacy ledger mapping requires a semantic audit, otherwise
   the row is `legacy-unmapped` and excluded from new-suite comparisons.
6. **R6 — Addressed (§§5.1, 7.2, 11).** Cases declare expected category/message/phase
   patterns; every emitted warning and count is retained. Unexpected warnings permit
   flagged quick results but leave standard adoption unresolved. Numerical failures
   remain invalid. Core warning declarations arrive with the core runner.
7. **R7 — Addressed (§§4.1, 5.1, 9).** Original-guide windows are the initial policy.
   SEP 25's quasi-static truncated-gradient substitution is not inherited. Any later
   equivalence claim must check the selected flow law and translated coefficients
   throughout the window. Averaged helium and constant-fill argon forms are explicit.
8. **R8 — Addressed (§5.1).** Added a source/route/grid/track/reference inventory,
   including public definitions for argon and both hydrogen routes. The 6 ps hydrogen
   variant is explicitly a grid workload derived from the 3.5 ps public example,
   not a private dependency or an assertion of physical necessity. The Kerr-only
   soliton comes from the introductory lesson; Raman/shock GNLSE from SEP 13.
   Additional short windows are frozen as each broader case is delivered.
9. **R9 — Addressed (§§4.1–4.2, 7.2).** Store every repeated window time and actual
   work counts; sum only timed windows. Reset includes controller proposals, caches
   and Multistep history, initially by a bounded replay from launch. Check both field
   and controller/history against fresh construction. Work variation stays visible.
10. **R10 — Addressed (§§4.3–4.4).** Compare maximum RSS and median startup; flag
    memory above 5% or 64 MiB and startup above 10%. Flags are advisory rather than
    automatic performance vetoes. Asymmetric candidate admission refusal is a resource
    regression and prevents a clean adoption verdict; missing telemetry is explicit.
11. **R11 — Addressed (§4.3).** Defined the adjacent-baseline absolute-log statistic,
    minimum four adjacent clean pairs, compatible saved A/A evidence and its
    aggregate band. Adoption must pass the repeatability band as well as both
    existing performance tests; disagreement is explicitly unresolved.
12. **R12 — Addressed (§§6.2, 11, 12).** Start with conservative existing or empirical
    estimates. Unestimated work needs explicit override plus a memory limit and is
    exploratory; underestimates are defects. Kokkos core admission is qualified with
    its phase 1 adapter. Use a synthetic excessive-history regression test, not an
    actual memory-exhausting adaptive Kerr benchmark.
13. **R13 — Addressed (§6.4).** Separate Linux Clang coverage configuration, recorded
    optimisation/OpenMP runtime, atomic counters and dependency-template filtering
    by source path. Timing continues to reject instrumented builds.
14. **R14 — Addressed (§§5.2, 9, 11).** Named `uv run python -m benchmarks`.
    To meet the usability priority, migrate `poe bench` after the qualified phase 1
    core rather than waiting for all coverage and platform work. Keep the old
    workflow intact until that switch and preserve historical records thereafter.
15. **R15 — Addressed (§5.2).** `--seed` only shuffles execution; physical noise seeds
    are immutable case inputs.
16. **R16 — Addressed (§§4.4, 5.2–5.3, 6.3, 7.3, 8, 11).** One `--all` spelling;
    standard-width example uses `1,production`, with 4/16 explicit diagnostics;
    concrete sanitised CSV/JSON export location and alias/environment mapping;
    named manual page; FFTW/Numba/Kokkos cache states; public qualification host
    classes; Kokkos adapter before optional Numba, with the Process-SEP exception
    to the usual numerical-feature implementation order stated explicitly.

### Acceptance — 2026-10-21

John Travers confirmed that he had read the revised proposal and requested that it
be marked Accepted and merged to main. This accepts Part I with the R1–R16 responses,
quick mode as the default, fixed RK4IP only for Kerr, and a usable Kokkos core before
broader telemetry and coverage. The recorded human and AI reviews remain in the
header. Implementation has not started; the phase-specific qualification work in
§12 remains required before the corresponding standard adoption claims.
