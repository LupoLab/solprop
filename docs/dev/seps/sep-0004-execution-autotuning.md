# SEP 4 — Bounded execution autotuning

| | |
|---|---|
| **SEP** | 4 |
| **Title** | Bounded execution autotuning |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add an explicit preflight that measures existing CPU execution policies on bounded
windows of a user's simulation and produces a validated, serialisable execution
plan. Each policy runs in a fresh interpreter; the production launcher establishes
the selected environment before importing numerical libraries. The preflight has
a wall-clock budget covering discovery, translation, compilation, planning,
measurement and validation, and may report that no improvement was established.
Ordinary runs retain their current heuristics and backend selection. Success means
preserving the numerical problem, reliably declining inconclusive measurements,
and finding useful improvements on a declared set of short, representative
workloads. Native GPU policy selection, chunk-size search and automatic decisions
about whether a whole propagation will repay tuning remain separate work.

### 2. Motivation and scope

**Draft; implementation not started.** The source audit for this proposal is
solprop commit `c099545cf4a2b2f433f4bd5e1303315352d77b62` and solprop-kokkos commit
`87ac816190ffb0c3978de7e236b6e30d7c068ab8`. It covers the registry, driver,
translation and memory contracts, both CPU implementations, Kokkos native policy
selection, scan launchers, output provenance and existing benchmark harnesses.
These are source findings, not new performance measurements. No implementation
or speed-up is qualified by this draft.

The concrete use case is a costly simulation, or repeated executions of one
simulation, for which a user already has an allocation and wants to select kernel,
FFT and BLAS concurrency without conducting a manual environment sweep. The
[performance guide](../../guide/performance/tuning.md) currently asks users to time
about ten steps at several settings. The proposed tool makes that procedure
repeatable and records both its evidence and its limits.

The motivation survives improvements to the defaults. Numba sizes different
operations independently, Kokkos now exposes separate host pools, and vendor
libraries use different threading mechanisms. No single count describes the
execution. Existing measurements in [backends](../backends.md) and
[benchmarks](../benchmarks.md) justify investigating this space; they do not
establish how often a tuner will beat the current defaults.

Included:

- Explicit tuning of a single simulation under a declared CPU and memory allocation.
- Existing Numba regime, FFT planning and pool controls; existing Kokkos CPU pool
  controls; supported OpenMP waiting/placement and controllable vendor BLAS pools.
- A bounded pilot, numerical checks, a conservative search, a plan/report format,
  strict launch-time validation, and local reuse of measured plans and FFTW wisdom.
- Explicit comparison of installed CPU backends in the last phase. NumPy is the
  oracle and supports the common launch/pilot protocol; it is not an automatic
  performance candidate.
- Application to a preselected scan worker allocation, with the executor retaining
  control of resource partitioning and scan semantics.

Excluded:

- Physics, grids, oversampling, quadrature, response approximations, table accuracy,
  controller or integrator choice, tolerances, precision, fast math, and save or
  statistics cadence. They remain identical across candidates.
- Live retuning during propagation; hidden notebook environment mutation; a
  `tuning="auto"` convenience keyword; runtime estimates extrapolated from a prefix.
- Selecting processes versus lockstep batches, worker counts or devices; concurrent
  candidate timings; cross-host plan transfer.
- Native rebuilds, GEMM-provider selection, GPU launch/reduction geometry, Kokkos
  FFT-planner/provider selection, chunk-size search and propagator-mode search.

This is a narrower, independently useful feature than a general optimiser of
every backend decision. The remaining execution-tuning work stays on the
[roadmap](../roadmap.md). SEPs [1](sep-0001-raman-response.md),
[2](sep-0002-two-photon-absorption.md) and [3](archive/sep-0003-optmat-materials.md) are
independent: a tuner may eventually execute their supported problems but never
chooses between their physical models or numerical approximations.

### 3. Context

#### 3.1 The execution controls that actually exist

The distinction between a requested setting, a resolved pool and the operation
that uses it is essential. A setting with no effect is not another candidate.

| Owner and source | Current behaviour | Treatment in this SEP |
|---|---|---|
| `backends/registry.py`, `SimulationSettings.__post_init__` | `auto` resolves to installed Kokkos, otherwise Numba; a broken installed backend raises rather than falling back | Preserve this meaning; backend competition requires an explicit list |
| `numba_cpu/fft.py:choose_pools`, `numba_cpu/backend.py` | Regimes are named `small` and `large`; the default cutoff is 1,000,000 elements, with 262,144 for guided transforms declaring `transform.n_nodes > 1` | Search existing specialisations, respecting their actual operation shapes |
| Same modules | Temporal FFTW gets up to four threads from 65,536 elements; BLAS gets up to two from 262,144; kernel count follows the regime and available CPUs | Search nearby legal pool counts, not new thresholds |
| Same modules | Raman full-chunk and partial-tail FFT pools have separate size decisions; transverse FFTs have another. `SOLPROP_FFTW_THREADS` overrides temporal and both Raman choices together; transverse FFTW has its own override | Preserve that coupling; do not claim independent Raman knobs in Numba |
| `NumbaCpuBackend.__init__` | Accepts `threads`, `fftw_threads`, `transverse_fftw_threads`, `blas_threads`, `plan_flag`, `regime`; the registry passes none of these | The launcher can use the existing environment path; no generic constructor-keyword dictionary is needed |
| `numba_cpu/fft.py` | `measure` is the default; `estimate` and `patient` exist. Wisdom is opt-in through `SOLPROP_FFTW_WISDOM`, imported at construction and exported at close | Candidate-private wisdom, immutable promotion, strict reuse verification |
| `numba_cpu/fft.py:apply_pools` | Numba and supported BLAS concurrency are changed at construction; FFTW counts belong to plans. The Numba import-time ceiling can be below solprop's chosen count | Validate both limits; never request more than `NUMBA_NUM_THREADS` [3] |
| `solprop_kokkos/__init__.py`, `src/core.cpp` [8] | Kokkos initialises once at import. A process cap and four optional kernel, FFT, Raman and Raman-tail pools are resolved there | Search existing startup settings; no reinitialisation |
| Kokkos `general.cpp`, `general_modal.cpp`, `raman_fft.hpp` [8] | Actual FFT/provider decisions depend on operand size, route and provider. The general workspace serialises selected FFTs below 8,192 real-side samples; Accelerate Raman schedules provider tasks | Inspect realised metadata; a cap is not every plan's concurrency |
| Kokkos `CMakeLists.txt`, `cmake/HostLibraries.cmake` [8] | GEMM and Raman FFT providers, architecture/build profile and FFT library are build choices. Host FFT planning records `FFTW_ESTIMATE`, or an MKL compatibility path that ignores those flags | Record and fingerprint; never search as runtime settings |
| Kokkos `general.cpp:response_team_size` [8] | Device team size has a compiled ceiling; CUDA caps it at 256 below `n_to = 8192`. The audited CMake default ceiling is 256, so a default build does not automatically use 512 above that size. Reduction width is compiled, default 512 | No GPU geometry candidates until runtime capabilities are designed and qualified |
| `translate/modal.py:Transform`, `translate/problem.py`, backend general-modal workspaces | Static/modal and spatial routes use `node_chunk` subject to memory/geometry rules. A general modal transform requires the placeholder `node_chunk=1`; its workspaces use `general_node_chunk` separately | No universal `replace(transform, node_chunk=...)` operation |
| `SimulationSettings.propagator_mode`, `Simulation.metadata` | `auto` resolves differently by backend; the requested value is in the configuration hash. Spatial routes refuse `tabulate` | Preserve the request and record the resolved mode; defer policy search |

The numeric cutoffs above are the current implementation's policies, not new
physical coefficients or recommended search constants. The relevant solprop code
is identified in [7]; the manual explanation is
[performance tuning](../../guide/performance/tuning.md) and
[backends §§3–4](../backends.md).

The Kokkos startup variables are `SOLPROP_KOKKOS_THREADS` for the cap and
`SOLPROP_KOKKOS_KERNEL_THREADS`, `SOLPROP_KOKKOS_FFT_THREADS`,
`SOLPROP_KOKKOS_RAMAN_FFT_THREADS` and `SOLPROP_KOKKOS_RAMAN_TAIL_FFT_THREADS`
for the four pools. Numba uses `SOLPROP_NUMBA_THREADS`, `SOLPROP_NUMBA_REGIME`,
`SOLPROP_FFTW_THREADS`, `SOLPROP_TRANSVERSE_FFTW_THREADS`,
`SOLPROP_BLAS_THREADS` and `SOLPROP_FFTW_PLAN`, alongside its import ceiling
`NUMBA_NUM_THREADS`. Retain the loaded Numba threading layer rather than searching
across `omp`, `tbb` and `workqueue` in this first implementation.

Numba's serial/parallel factory choice is made before compilation; selecting
`large` is not inherently a conflict with an existing binary. However, parallel
closures and closures capturing dispatchers deliberately avoid the disk cache.
Fresh candidate processes can therefore pay substantial compilation cost each
time. The policy enforced by `tests/backends/test_numba_cache_policy.py` remains
unchanged. A thirty-second search is a budget, not a promise that several cold
Numba candidates can finish.

#### 3.2 Existing infrastructure and its limits

`driver/executors/pool.py` already starts fresh interpreters with thread variables
established before startup, uses importable builders, bounds worker resources and
probes peak RSS. `Slurm` and the scan CLI already build on this execution model
([scans](../scans.md)). It is a useful design precedent, not a generic launcher:
`Pool._spawn` temporarily changes the parent's environment under a lock, replays
the main module through multiprocessing, and applies one count to many pools.
A tuner needs a different candidate environment and must not race with unrelated
threads creating processes.

Resource detection is also incomplete: the current probes read root cgroup files,
not necessarily the process's actual nested cgroup. The roadmap records this
limitation. Resource identity for tuning must describe the effective allocation,
including affinity, scheduler limits, quotas and explicit worker partitions;
physical machine size alone is insufficient.

`import solprop` is lazy; pyFFTW is not loaded until a Numba backend is imported.
Kokkos's macOS guard nevertheless refuses import after pyFFTW. Neither lazy imports
nor notebook discipline make it safe to compare different numerical runtimes in
one interpreter. Build/capability discovery that imports a backend also belongs
in a child.

`stepper.schedule.discover` consumes recorded history, and `FixedSchedule` stores
positive accepted step lengths. `StepDriver._replay_to` never rejects; it checks
finiteness and records the embedded norm. Replaying identical edges therefore
fixes work, but identical replay acceptance flags are not a correctness test.
The shared driver has no bounded accepted-step pilot operation today.

`RunMetadata.backend` is excluded from `configuration_hash()`; the physical
configuration, initial/noise state, stepper and output choices are included.
`driver/run.py` records realised backend metadata after construction. This is the
right provenance boundary for a plan, provided it does not change fields such as
the requested propagator mode that already enter the resume key.
There is no per-resume execution history today: `HDF5Output` in append mode
retains the original metadata, including its original backend record. A tuning
plan must not be represented as though that record describes later segments.

The supported measurement discipline is [benchmarks §5](../benchmarks.md): short
windows, warmed arithmetic, alternating fresh-process pairs, an A/A resolution
indicator and no full-run extrapolation. Current reusable precedents include
`benchmarks/ledger.py` and solprop-kokkos's `benchmarks/provider_sweep.py` and
`statistics_cost.py`; research harnesses are not imported by production code.

### 4. Theory

#### 4.1 What is being minimised

There is no new propagation equation. Every candidate advances the same internal
complex128 state and the same translated operators, following
[conventions](../conventions.md). Fourier signs, SI amplitudes, losses, response
normalisation and deterministic reduction requirements are unchanged.

Let $p$ be a legal execution policy and $w$ a supplied pilot window. In a timed
repeat, let $t_{p,w}>0$ be elapsed seconds for propagation on fixed edges,
requested periodic statistics, and the declared observation/statistics calls.
Translation, construction, compilation, FFT planning, validation comparisons and
file output are excluded from this interval and reported separately. For declared
non-negative weights $a_w$ with $\sum_w a_w=1$, minimise

$$
T_p = \sum_w a_w\,\operatorname{median}(t_{p,w}).
$$

The default is one entrance window with weight one. Additional windows require
existing saved states and explicit weights; they are not generated by running a
long propagation. The score describes only these windows. A late ionising state,
a taper or changing Raman width may have different costs despite identical array
extents. No claim of global representativeness follows from shape alone.

The feasible set requires backend admission, numerical validation, allocation
compliance and measured memory admission. A faster answer outside that set cannot
win. FFTW's measured planning is an established example of selecting execution
algorithms for fixed mathematics [1]; this SEP applies the principle to a small
set of existing policies, not to generating kernels.

#### 4.2 Noise and selecting a winner

Screening measurements only rank candidates for further work. They do not justify
a plan. For each finalist, repeat baseline/candidate pairs in alternating order,
with independent baseline repetitions providing an A/A check. For pair $j$ define

$$
g_{p,j}=1-\frac{T_{p,j}}{T_{0,j}},\qquad
\widetilde g_p=\operatorname{median}_j g_{p,j},\qquad
d_p=\operatorname{median}_j|g_{p,j}-\widetilde g_p|.
$$

Here $T_{p,j}$ is the same weighted window bundle in every pair. Define $d_0$ as
the median absolute relative difference of the independent A/A bundles and use
the conservative decision rule

$$
\widetilde g_p > \max(g_{\min}, 2d_p, 2d_0).
$$

Require at least three complete pairs plus an independent A/A pair. The proposed
default $g_{\min}=0.05$ and factor two are selection policies, not confidence
levels or physical constants. Correlated load and small samples make a formal
confidence claim inappropriate. Publish the raw timings and ordering; a missing
sample, tied score or inadequate budget yields no established improvement.
Correctness has separate bounds (§7) and is never relaxed to obtain a winner.

Load before work is a useful guard; load after work includes the tuner's own
activity. Follow the ledger's distinction rather than rejecting a successful
CPU-intensive search because it raised load. Record both; interleaving and A/A
spread detect drift. If the guard fails, retain only a previously validated
baseline, with a named `machine_busy` or `inconclusive` result. Missing load
telemetry is recorded as unavailable, never zero.

#### 4.3 Budget and amortisation

Let $B$ be the requested search budget in seconds and $b$ the monotonic elapsed
time from entry to the preflight. All child startup, resource/build probes,
factory construction, translation, oracle work, JIT, planning, warm-up, validation
and cache checks count toward $b$. A candidate may not start unless the remaining
budget allows the scheduler's reserved validation and shutdown work. At the
deadline terminate unfinished children, then reap them and atomically record the
completed evidence. An OS process creation or termination need not complete at an
exact deadline [6]; record actual elapsed time and any cleanup overrun, without
describing it as time spent successfully tuning.

The budget does not hide a preliminary untimed long run. Thirty seconds is an
initial user-interface default to evaluate in phase 4, not an acceptance guarantee.
With no validated baseline, return a report with no plan. With a validated
baseline but no validated improvement, return that baseline and the reason.

For repeated *identical measured windows*, if tuning costs $b$ seconds and the
selected policy saves $T_0-T_p>0$ seconds per window, the algebraic break-even
count is

$$
R > \frac{b+\max(0,C_p-C_0)}{T_0-T_p},
$$

where $C_p-C_0$ is any additional production construction cost. This explains the
trade-off but does not estimate a propagation's duration or future step count.
The report gives measured costs and gains separately. Users decide whether their
campaign merits the overhead; no one-hour threshold is inferred from a prefix.

### 5. Proposed interface

The public implementation belongs in `solprop.driver.tuning`; the CLI is
`python -m solprop.tune`. The following sketches are proposed, not runnable APIs.

```text
from solprop.driver.tuning import RunSpec, TuneSettings, tune, launch

spec = RunSpec(factory="campaign:build", parameters={"pressure_bar": 2.0})
result = tune(spec, settings=TuneSettings(budget=30.0, min_gain=0.05))
result.write("tuning-report.json")
if result.plan is not None:
    result.plan.write("execution-plan.json")
    launch(spec, plan=result.plan, output="run.h5")
```

```text
python -m solprop.tune search campaign:build --parameters case.json \
    --budget 30 --report tuning-report.json --plan execution-plan.json
python -m solprop.tune run campaign:build --parameters case.json \
    --plan execution-plan.json --output run.h5
```

`RunSpec` names an importable `module:function` receiving JSON-representable
keyword parameters and returning an unexecuted `Simulation`. It never contains
a live backend or pickled `KernelProblem`. Translation contains callables and
user-defined providers; JSON-serialising the entire simulation is not assumed.
The factory may construct those objects inside the child but must be deterministic,
use explicit seeds and declared input files, and perform no propagation, output
creation or scan dispatch. Lambdas, notebook closures and arbitrary script replay
are refused with guidance to create an importable factory. Different translated
inputs from the same specification are a factory error, not measurement noise.

`TuneSettings` is a frozen record. `budget` is finite positive seconds;
`min_gain` lies in $[0,1)$; a pilot takes one to ten accepted steps, default ten,
with a finite attempt limit of 100 by default. The latter is a resource guard,
not a changed controller. Pilot windows may end earlier at the simulation boundary.
An optional resource record supplies positive CPU and byte limits and an allowed affinity set;
it can narrow but never enlarge an actual allocation. Explicit policy variables
in the inherited environment are fixed constraints unless named for search in
`vary`; the allocation and device visibility are never overridable search axes.
Inconsistent fixed constraints fail before measurements.

By default search only the concrete backend already resolved by the simulation.
In phase 4, `backends=("numba_cpu", "kokkos")` explicitly authorises CPU backend
comparison; it does not redefine `backend="auto"`. Probe Kokkos's execution space
in isolation: an installed device build is not a CPU candidate. A baseline import
or numerical failure is fatal. An explicitly requested alternative that cannot
execute the problem is listed as refused, with its original diagnostic; the
report must not call the comparison complete for that backend.

`TuneResult` contains a status, complete report and optional `ExecutionPlan`.
Statuses distinguish `improved`, `baseline`, `inconclusive`, `machine_busy`,
`budget_exhausted` and `cancelled`. An exception is reserved for invalid requests,
baseline failures and implementation errors; it uses the `SolpropError` hierarchy.
Cancellation saves completed evidence but never launches production automatically.

An `ExecutionPlan` is immutable and has its own format version, independent of
the kernel and output schemas. Its JSON contains:

- A concrete backend and validated policy record, including environment settings
  and explicitly removed policy variables, resource constraints and resolved pools.
- The simulation/input and pilot identities, software/build and allocation
  fingerprints, creation time and validation status.
- Wisdom content digests and companion artifact identities where applicable;
  no executable commands or arbitrary backend constructor keywords.
- The selected candidate ID, report digest, measured objective, baseline comparison
  and why the search stopped. Full candidate diagnostics live in the report.

The launcher validates the plan and specification, starts one interpreter with
an explicit environment, imports the selected backend in its required order,
rebuilds the simulation, checks identities and realised policy, and only then
creates or resumes production output. It calls the existing `Simulation.run`
path. `validation_budget` (CLI `--validation-budget`) is positive finite seconds,
defaulting to the search budget saved in the plan; it bounds launch validation
separately from production. Applying half a plan and merely logging the unapplied settings is not
supported. Interactive callers may launch a child; no `plan.apply()` mutates a
live notebook's pools.

### 6. Implementation strategies

#### 6.1 Process boundary and allocation

Use standard-library subprocesses with explicit `env`, argument lists and a small
importable worker entry point. The parent need not import any backend. Set Linux
affinity in the worker bootstrap before importing the factory or numerical
libraries, so native pools see the intended allocation. Preserve the caller's
environment apart from an allowlist of execution variables; do not clear unrelated
variables or serialize credentials into provenance. Check OpenMP runtime/provider
compatibility and report requested versus realised placement. Unsupported topology
information disables topology-specific candidates rather than inventing socket
or performance-core counts.

Extract shared resource/environment concepts from the scan implementation only
where useful, in domain-named driver modules with scan regression tests. Do not
call private `Pool._spawn` from the tuner or add a second incompatible allocation
policy. Resolve nested cgroups where supported; otherwise require an explicit
conservative allocation for a constrained environment and name the uncertainty.
Environment variables such as `KMP_BLOCKTIME` require a matching runtime;
`OMP_WAIT_POLICY` itself is a hint, not a guarantee of how threads wait [4–5].

Candidates run sequentially, at most one resident numerical worker at a time.
This costs repeated setup but avoids resident workspaces, spinning pools and
cross-runtime interference changing the measured problem. Resource probes and
the NumPy oracle obey the same bound. Record the backend memory ledger, process
peak RSS and controller/artifact overhead. A ledger is not a hard RSS limit and
does not include every compiler or library allocation; reject a candidate whose
measured footprint exceeds the admitted budget and use available OS enforcement
without promising universal prevention of an out-of-memory kill.

#### 6.2 A bounded pilot through the shared driver

Add a shared bounded-advance operation that stops after at most the declared
accepted steps or attempted steps, or at the next requested endpoint. It returns
the normal committed observation and a reason for stopping. It uses the existing
adaptive and replay loops, never a copied integrator or a fabricated long-distance
estimate. The stop is checked after state, position, history and periodic
statistics agree. A reporter exception is unsuitable: the adaptive loop currently
reports acceptance before committing the advanced state.

This extends the backend contract and shared driver's protected control surface:
**phase 1 bumps `SCHEMA_VERSION`**, updates its frozen literals and documents the
bounded operation. The exact public method/record names are settled in phase 1;
the bounds and commit semantics above are acceptance requirements. It introduces
no changes to `StageTerms` or `StageScalars`, and ordinary `propagate_to` retains
its behaviour. An external backend built for the preceding schema refuses by
name until updated. Each implementation phase qualifies unbounded calls too.
Phase 1 includes the minimal compatibility update and rebuild in solprop-kokkos
needed to keep existing propagation usable under the new schema, with tests of
its inherited driver operation. This does not admit Kokkos policies to the tuner
before phase 3. Leaving every installed CPU backend incompatible until phase 3
would prevent the earlier phases from satisfying their full gate.

Discovery translates the original full-domain simulation without shortening a
pressure profile or taper. It advances only the bounded prefix at the user's
production tolerances, preserving existing save-plane and hard-kink landings.
Record every attempt and retain the accepted edges as a pilot `FixedSchedule`.
For an already fixed production schedule, take its bounded prefix without a
second adaptive solve. Do not invoke a full `Simulation.run()` to discover it.

Candidates reconstruct the original problem and execute the same bounded schedule
and output-call pattern. If replay validation requires endpoint/save-plane changes,
make a separately identified pilot view using the original translated operators;
never re-derive a longitudinal profile on a shortened physical length. Preserve
noise realisations and boundary/flow sections. Load saved witnesses through the
existing optics-to-internal and power-amplitude conversions; distinguish a bare
entrance field from one already carrying input noise, so reset never adds it twice.

Warm a complete window before timing and reload the identical initial state,
noise, controller/schedule cursor, FSAL state and statistics counters. Reset and
host comparisons occur outside the interval; observation and public statistics
occur inside at the declared positions. Existing lazy statistics buffers must be
warmed as well. Pilot allocations such as history are reported; the existing
allocation restrictions on compiled stage kernels are unchanged.

An entrance pilot is always labelled as such. Later windows require a supplied
saved witness whose configuration, grid, noise, position and field digest agree
with the simulation; they carry separate schedule identities. Never manufacture
a late state by placing the entrance pulse at a new z, and never regenerate a
long run for tuning. The caller must acknowledge the window coverage when using
a plan for a varying or strongly evolving problem.

#### 6.3 Candidate generation and search

The current heuristics under the declared allocation, including inherited explicit
overrides, define the baseline. Impose required resource caps before constructing
it: a machine-wide default outside a worker's allocation is not an admissible
baseline. An explicit override conflicting with that allocation is an error,
not an opportunity to silently change the request. Record this resource
normalisation separately from searched choices. Measure the baseline first.
Generators use the translated operation inventory and isolated backend capability
record. Candidate enumeration is deterministic and versioned. De-duplicate
resolved policies, not only their input strings.

| Family | Initial candidate axes | Restrictions |
|---|---|---|
| NumPy reference | Baseline only | Supports the common launcher and validation; BLAS is constrained to the allocation |
| Numba kernels | Current regime; the other existing regime; current, one, half-allocation and allocation-sized pool | `small` has no useful parallel node pool; all counts obey the import-time ceiling and actual allocation |
| Numba FFTs | Current, one, two and four threads, clipped to allocation; independent transverse candidates when applicable | Temporal/Raman/tail overrides remain coupled; skip absent transforms |
| Numba BLAS | Current, one and two where the loaded provider can be controlled | An uncontrolled provider is fingerprinted and not claimed as tuned |
| Numba planning | Current effort, `estimate`, `measure`; `patient` only when explicitly included | Construction and wisdom generation count toward budget; no planning in trials |
| Kokkos CPU | Process cap; kernel, FFT, full Raman and tail pools independently | Counts are positive and within cap/allocation; omit absent/provider-inapplicable work |
| OpenMP and vendor BLAS | Current waiting/placement, active/passive, close/spread within known allowed places; supported small BLAS pools | Respect fixed environment constraints; do not assume one vendor's variables control another |

Search a bounded neighbourhood, not the Cartesian product. First compare a small
set of pool allocations near the heuristic, including one explicitly serial
policy where legal. Then explore the relevant FFT/Raman and waiting-policy
neighbours of the best screened policies. Allow a small number of joint moves:
kernel and FFT pools can interact, and one-axis screening alone can miss them.
Record every proposed, skipped, failed and completed candidate and the reason.
The number of candidates is constrained by remaining wall time, not a promise
that every row above is visited.

Use remaining time to validate and repeat the best one or two candidates against
baseline under §4.2. A screen winner that fails numerical or resource checks is
removed. A noisy finalist does not force an early end to all exploration if a
different validated candidate can still be assessed within budget. The report
states whether the search was exhausted, pruned or interrupted.

Keep both an effective candidate policy and its realised per-operation metadata.
For example, a requested Kokkos FFT count can resolve to serial work on a small
operand, and a Numba FFT override can change the Raman tail along with the main
transform. No benefit may be attributed to a knob whose effect was not verified.

#### 6.4 Plans, cache and wisdom

Use canonical JSON with explicit validation and atomic replacement. Malformed,
partial or unknown-format entries are misses during search and errors when
explicitly supplied to `run`. A report is never silently substituted for a plan.
Store under the platform's user cache directory in `solprop/plans`, or at a
caller-specified directory. Cache use is explicit in the tuning API; ordinary
simulation construction does not consult it.

Use a two-stage identity. Before rebuilding a simulation, an index based on the
factory, parameters and environment may locate a possible entry; it grants no
permission to execute. Before measurement or use, confirm:

- Package versions, source/build content identities (including editable or dirty
  code), `SCHEMA_VERSION`, Python ABI, NumPy/SciPy/Numba/pyFFTW versions, FFTW/BLAS/
  OpenMP libraries, backend build profile, compiler flags and capability version.
- Host identity and CPU topology, OS, effective CPU affinity/quota/scheduler and
  worker allocation, memory limit and relevant environment values, including unset
  values. Mutable load is validation telemetry, not an ever-changing cache key.
- The simulation configuration hash plus a digest of actual translated arrays,
  response/table identities, initial field and noise. Include all operation shapes:
  grid family, geometry, components, modes, nodes, chunk/tail extents, FFT lengths,
  response families/channels, table/flow representation, statistics and cadence.
- Pilot start-state digests, z extents, step edges, observation pattern and weights;
  search settings, generator version and relevant baseline wisdom digest.

A same-shape simulation with different data is not an exact hit. A different
allocation is not the same host for this purpose. If reliable software or input
identity cannot be established, allow an explicitly uncached search but refuse
persistent plan reuse. Paths and mtimes alone do not identify a rebuilt extension
or modified input file. Shape-based reuse as an untrusted search prior can be
added later; it is not automatic application of a plan.

Before `tune` returns a cached policy as a current measured choice, run a bounded
baseline/selected comparison with the same correctness and noise rules, charged
to the search budget. Budget exhaustion does not turn an unvalidated entry into
a winner. Explicit `run --plan` instead verifies applicability, strict plan
reconstruction and one bounded numerical replay against the stored, oracle-checked
witnesses within its launch validation budget; it does not repeat the performance
search. Record the age of its last performance measurement. No `trust=True`
bypass of applicability or numerical checks is proposed. Explicit stale plans fail
with changed identity fields named; they do not silently fall back to another backend.

FFTW wisdom is a separate content-addressed artifact, bound to hardware, library
build, transform layout, precision, thread count and planning effort [2]. Each
candidate receives a private copy of the same declared initial wisdom, or an
empty store. Losing candidates must not change what later candidates inherit or
overwrite the user's wisdom file. Export the winner's wisdom after clean close,
then revalidate the exported artifact in a fresh process before promotion.
Budget for this step; incomplete exports cannot enter a plan.

For a planned production run, verify import success for the needed double-precision
plans and their recorded identities; a missing/incompatible artifact is an error.
Do not accept the ordinary best-effort import's `any(...)` success as proof that
every required plan was reproduced. Use a strict wisdom-only construction path
where FFTW supports it, with a named refusal if a required plan is missing [2].
This is an explicit plan-mode addition in phase 2; ordinary Numba construction
retains its current behaviour. If reproducing a measured plan cannot be established,
do not call the policy frozen merely because its planner flag is the same.

#### 6.5 Provenance, production and scans

Record requested and realised policy, plan/report digests and pilot coverage in
`BackendRecord.settings` for a newly created output. Keep tuning timings distinct
from production timings. The complete report remains a companion artifact; a
digest without the retained artifact is insufficient for an auditable campaign.

Add a versioned companion execution journal for planned HDF5 outputs, keyed to
the output's configuration hash and a persistent run identifier. It records each
segment's starting saved-plane index and z, realised backend/plan, completion or
interruption, and report digest. Create/update it atomically without replacing
the original HDF5 metadata. The ordinary resume path must continue an existing
journal even when no plan is supplied, recording the actual default execution;
legacy segments without evidence are explicitly unknown. A journal/output
mismatch fails before append. Memory outputs carry the same segment records in
memory. Phase 2 defines crash recovery and artifact relocation semantics and
tests interruption between output and journal writes. The journal has its own
format version and does not change historical HDF5 configuration hashes.

The tuner never changes the production stepper to its pilot schedule. The pilot
schedule is measurement evidence only. Keeping backend execution policy outside
the configuration hash lets the same physical run resume with or without a plan,
subject to existing compatibility checks and the plan's stricter applicability
checks. Do not delete `propagator_mode` from historical hashes to make tuning fit.

Scan resource partitioning happens first. A plan for a worker is keyed to that
worker's allowed CPUs, memory, backend and complete point identity, and is applied
before its interpreter starts. `Pool` must preserve the plan's separate pools
instead of overwriting all of them with `threads`. A one-run plan cannot silently
change a lockstep batch or enlarge a worker partition. Tuning runs before concurrent
scan workers start, or in a separately declared allocation; it is not a nested
oversubscribed search inside every scan point. Broad reuse across scan points
and native batch tuning remain outside this SEP.

#### 6.6 Architectural cost

Policy, subprocess supervision, cache and reports belong in D. Existing B modules
remain the owners of pool resolution and realised metadata; T still owns problem
validation. The new bounded driver operation is the only intended kernel-seam
extension. Environment policies and plan records do not become physics fields
of `KernelProblem`. No new third-party dependency is proposed.

With $K$ completed candidates and $R$ repeated window bundles, cost is the sum of
their setup and $R$ short executions; the deadline can truncate that sum. Peak
memory is the controller and saved validation data plus one worker, not $K$
resident backends. Full field witnesses can be large: stage them in temporary
array artifacts and count their resident copies and disk usage. Recomputing a
translation per child is intentional; only demonstrated immutable, versioned
translation artifacts should justify a later reuse optimisation.

### 7. Verification and validation

#### 7.1 Oracles and numerical gates

Qualify policy adapters against `numpy_ref` before admitting them to candidate
generation. Use the existing ladder in [accuracy §7](../accuracy.md): one linear
application, nonlinear evaluation, one step, eight steps, statistics and a short
adaptive run. Cover scalar/vector field and envelope families, plasma, scalar and
tensor Raman, labelled mixtures, coupled flows, general modal tables and spatial
routes where the backend admits them. Refused routes remain refused.

Derive bounds before execution from machine epsilon, FFT depth, contraction and
reduction lengths, conditioning and accumulated steps, following the corresponding
route's existing tests. Compare full fields and relevant weak channels as well
as energies/statistics; a global relative norm can hide a wrong weak component.
Near zero, use an absolute scale derived from the reference/input norm and the
operation bound. Do not invent one universal tolerance or copy a milestone's
looser full-run bound onto every ten-step window.

At tuning time the baseline and every promotable candidate must match a NumPy
oracle window within the applicable prequalified bounds. Store baseline-relative
deviations too, but the baseline is not the oracle. If the actual workload lacks
a justified bound or the oracle cannot complete within budget, no new policy is
validated. This cost may make a short budget unsuitable for a large problem;
the result must say so.

Replay checks require identical edges (a prescribed schedule executes its specified
steps) and compare fields, finite embedded error estimates, periodic statistics and
saved diagnostics within a rounding bound fixed before measurement. Replays always accept; their flags
are only a work-consistency check. An embedded norm violating the predeclared
pilot acceptance bound is not ignored because replay cannot reject.
Separately qualify short adaptive runs. Near an acceptance boundary, rounding
can change a decision; require identical decisions only where a derived margin
from the threshold warrants it. Otherwise compare achieved numerical accuracy
and record differing trajectories, rather than forcing controller identity by
widening tolerances. Same-policy repeated runs and runs at different thread counts
agree within a rounding bound fixed before measurement on a matched fixed schedule,
and within the case's accuracy envelope against a tighter or stored reference when
adaptive ([SEP 25](archive/sep-0025-rounding-level-reproducibility.md)).

The bounded operation must return the committed state of an ordinary run stopped
at the same accepted edge, for adaptive and fixed stepping, including FSAL,
rejected attempts, save planes, kinks, modified/input noise, periodic statistics
and a resumed witness. On fixed stepping or replayed edges, edges, counts, save
planes and identities are exact (structural) and the field, FSAL stage and
statistics agree within a rounding bound fixed before measurement; an adaptive
stop agrees with the ordinary adaptive run within the case's accuracy envelope,
with its counts in a loose band. A bounded stop must not change later continuation.
Retain analytic dispersion/loss, soliton and causal-response tests and their
existing convergence gates; no new numerical order is claimed.

#### 7.2 Orchestration and failure tests

Use deterministic fake clocks/workers for search logic, not noisy wall-clock
assertions. Test known costs, interactions missed by one-axis search, tied and
misleading screening winners, drift, A/A noise, busy hosts, cancellation, deadline
overrun reporting, import failure, crashes, child reaping and baseline failure.
No unit test promises a speed-up on the CI machine.

Test JSON round trips, unknown fields, identity changes one at a time, dirty builds,
same-shape/different-physics inputs, altered witnesses, environment precedence,
partial writes, concurrent cache writers, stale plans, wisdom corruption and
failed strict plan reconstruction. A child must observe its environment and
affinity before its first numerical import; parent pools and environment must be
unchanged. Exercise macOS runtime ordering and both controllable and uncontrolled
BLAS reports. Test resource probes with nested-cgroup and scheduler fixtures.

For each adapter verify requested versus resolved counts, Numba's import ceiling,
small/large specialisations, missing Raman and tail work, transverse FFTs, serial
Kokkos builds and provider task limits. A setting rejected or ignored by the
backend must not silently become a successful candidate. Test plan provenance
through memory/HDF5 output and resume, and planned workers against the existing
Local/Pool/Slurm execution semantics. Keep all persisted test artifacts in
`tmp_path` and all numerical windows bounded.

#### 7.3 Performance evidence and acceptance

Record a ledger entry for each hot-path change, including the bounded-driver
guard's effect on ordinary execution. Perform an offline finite-candidate study
on short windows: representative single-mode GNLSE, multimode field/plasma/Raman,
coherent wall/taper, general modal and radial/Cartesian/planar workloads. Include
small shapes where baseline should remain and shapes with Raman tail work.
Use at least two CPU architectures/provider combinations and an allocation-limited
case. Unsupported combinations are named, never counted as passes.

For each case publish the candidate set, full operation shapes, allocation/build
record, validation deviations, warm window medians/dispersion, A/A spread,
construction/JIT/planning costs, RSS and ledger, windows and timed boundaries.
Compare independent tuning searches at 10, 30 and 120 seconds with a separately
measured exhaustive result over that *declared finite set*. Cold starts and reuse
of verified wisdom are separate experiments. The offline sweep is not hidden
inside the online budget and does not measure a full propagation.

If $T_*$ is the fastest validated policy in that set and $T_s$ the selected time,
define recovered throughput as $q=T_*/T_s$. Report $q$ and improvement over the
baseline separately: $q\ge0.9$ can hold even when tuning achieved nothing because
the baseline was already good. The proposed performance target is $q\ge0.9$ on
each declared case that finishes validation in budget; also report the fraction
of all cases that obtained a validated plan and all budget failures. Independent
confirmation must show no selected regression beyond the measured resolution.
The target is an empirical acceptance question in phase 4, not an algorithmic
guarantee, and a failed target is not removed by silently pruning difficult cases.

All implementation phases run `uv run poe check`, `uv run poe test-slow` and
`uv run poe cov` before commit/merge. Kokkos changes additionally run that
repository's gate and solprop qualification with its fresh CPU build. Do not
defer complete gates until the last phase. The CPU-only tuning exception is
explicit: GPU arithmetic remains separately qualified by existing policy; new
GPU autotuning is not delivered by this SEP.

### 8. Documentation

Add `docs/model/tuning.md` as the manual for the factory, explicit search, pilot
coverage, costs, plans, cancellation, validation, cache and launch/resume semantics.
Add `docs/dev/tuning.md` for process boundaries, identities, candidate capabilities
and measurement protocol. Update the performance/backend/scan guides and
`docs/guide/setup/running.md` so the existing environment controls remain documented
beside their new measured application path. Update `README.md`, `docs/introduction/first-run.md`
and `CHANGELOG.md` when functionality lands, not to advertise this draft as an API.

Update architecture/structure, backends, kernel contract and output descriptions
in the phases that change them. Document all public records and methods with
units, ranges, refusal behaviour and runnable examples. A quick tutorial should
show a tiny deterministic search/report and baseline outcome; it must not depend
on one setting winning on a particular host. The manual's proposed API sketches
remain `text` fences until implemented. No new full-size worked example is
required solely to demonstrate tuning. At Final, produced manual pages link
back to this SEP and the changelog cites it.

### 9. Compatibility and impact

Ordinary backend selection, numerical defaults and environment overrides retain
their meanings. No automatic search, cache lookup or extra child process appears
in ordinary `prop_*` or `Simulation.run` calls. Explicit launch plans apply all
their settings or fail before writing production output.

The bounded pilot extends the kernel contract in phase 1 and requires a schema
bump and external-backend update; this is not hidden as a constructor-only change.
Plan schema and candidate-generator versions are independent. The existing
saved-output schema can carry plan metadata inside `BackendRecord.settings`
without changing the resume hash; if implementation needs a different structural
output representation, it must version and qualify that separately.

Changing an execution policy can change rounding or an adaptive trajectory, as
existing cross-backend runs already can. The promise is unchanged physics and
qualified numerical accuracy, not bitwise equality across FFT planners or BLAS
libraries: a rounding bound fixed before measurement on a matched fixed schedule,
and the case's accuracy envelope for adaptive runs
([SEP 25](archive/sep-0025-rounding-level-reproducibility.md)). A plan records the relevant realised execution and artifacts; it does
not promise identical performance under future system load. No new deliberate
physical or numerical difference from Luna is introduced.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Keep manual environment sweeps only | Lowest implementation cost; still a valid choice if the acceptance study finds insufficient benefit. Loses repeatable budgets, strict application and provenance |
| More global thresholds | Useful zero-cost defaults, retained; cannot adapt to the user's allocation/provider and all operation shapes |
| Reconfigure a live backend | Cannot reproduce import-time OpenMP/Kokkos state, mixes native runtimes and wisdom, and conflicts with process-wide pools |
| Persistent resident workers for every policy | Amortises JIT but multiplies workspace memory and permits idle runtime interference. Defer until separately measured and bounded |
| Arbitrary-script interception or pickling a live simulation | Risks running outputs/scans repeatedly and does not define reconstruction for closures and native state; use an explicit factory |
| Microkernel-only tuning | Cheap screening, but misses response/FFT/statistics interactions and finite-window behaviour. Cannot select a production plan without whole-window confirmation |
| Bayesian or learned optimisation | Adds machinery and potentially a dependency before a small legal search space has been measured. Deterministic staged neighbourhoods are sufficient to test the premise |
| Automatic backend race inside `auto` | Changes existing semantics and failure behaviour; compare backends only on explicit request |
| Search all chunks, propagators and native GPU geometries now | Adds route-specific memory contracts, hash migration questions and unimplemented native capabilities to an otherwise useful CPU feature; retain as follow-up |

### 11. Phases

Each phase has its own `feature/sep-0004-execution-autotuning-...` branch and
appends a report to Part II when merged. No phase may label an unsupported
backend as tuned. Acceptance of this SEP includes the explicit CPU-only tuning
scope; there is no implied GPU completion requirement hidden in Final status.

1. **Reference, bounded pilot and process contract.** Implement records, factory
   reconstruction, resource validation, subprocess supervision, the bounded shared
   driver operation and the schema bump, with NumPy baseline/oracle support and
   the manual. Validate commit/reset/noise semantics and fake search outcomes.
   Include the minimal Kokkos compatibility bridge described in §6.2, proving its
   consumed driver surface remains supported and rebuilding its CPU package;
   older binaries fail clearly. Policy search remains reference-only. No speed-up claim.
2. **Numba policy search and strict plans.** Implement the existing Numba policy
   adapter, numerical validation, search/refinement, plan launcher and cache,
   private wisdom artifacts and strict reconstruction. Preserve cache restrictions
   on parallel closures. Verify cold-budget failure, all applicable pool families,
   fresh-process parity, provenance and same-configuration resume. Add bounded
   ledger evidence before promoting the interface as useful.
3. **Kokkos CPU and shared allocation integration.** Build on phase 1's compatible
   bounded driver contract, expose/verify any missing realised
   host-policy metadata, and add its generator. Search only existing import-time
   host controls. Complete nested-allocation handling and scan worker plan
   application without changing worker count or route. Run both repositories'
   gates and short parity/plan tests with and without the optional CPU package.
4. **Explicit CPU backend comparison and acceptance study.** Admit explicit
   backend lists, qualify default-resolution and refusal semantics, complete
   independent finite-set studies and report success rate, recovered throughput,
   setup costs and budget failures. Resolve or move remaining questions to named
   roadmap items, complete documentation and archive only the scope delivered.

### 12. Open questions

- **Q1 — Maintainers, before acceptance:** Is existing-control CPU tuning a useful
  first deliverable, with GPU policies, chunking, propagator selection and implicit
  convenience tuning explicitly deferred as proposed?
- **Q2 — Phase 1:** Choose the bounded-advance method/record names and final attempt
  budget interface. The stop must occur after a committed step; reporter-based
  interruption cannot satisfy the contract. Record the durable decision in an ADR.
- **Q3 — Phase 1:** Which constrained platforms can be detected reliably, and which
  require an explicit allocation? Resolve nested-cgroup behaviour jointly with
  scans rather than accepting a silently oversized machine count.
- **Q4 — Phase 2:** Can strict reconstruction cover every FFTW plan family, including
  Raman tails and spatial plans, with the existing bindings? If not, name the
  refused family and return no frozen plan until a supported solution is qualified.
- **Q5 — Phase 4:** Do the proposed thirty-second default, five-percent minimum gain
  and repeated-pair requirements produce useful validated results after real setup
  costs? Report failures and recommend a documented change if evidence warrants it;
  never move setup outside the budget to preserve a headline target.
- **Q6 — Phase 4:** Do entrance-only policies transfer usefully to supplied later
  windows? If not, restrict the documented applicability and require multiple
  witnesses; do not add an unmeasured trajectory extrapolator.
- **Q7 — Roadmap follow-up:** Design capability-enumerated native GPU/GEMM controls,
  geometry-specific chunk budgets and propagator policy identity before admitting
  them as candidates. None is delivered by the four phases above.

### 13. References

1. M. Frigo and S. G. Johnson, “The Design and Implementation of FFTW3,”
   *Proceedings of the IEEE* **93**(2), 216–231 (2005).
   [doi:10.1109/JPROC.2004.840301](https://doi.org/10.1109/JPROC.2004.840301).
   [Author-hosted paper](https://www.fftw.org/fftw-paper-ieee.pdf) and
   [author bibliography](https://klein.mit.edu/~stevenj/papers/papers_bib.html).
   Checked for the planning/execution distinction and bibliographic details;
   its historical speed measurements are not tuning targets here.
2. FFTW manual, [Planner flags](https://www.fftw.org/fftw3_doc/Planner-Flags.html)
   and [Caveats in using wisdom](https://www.fftw.org/fftw3_doc/Caveats-in-Using-Wisdom.html).
   Checked for destructive planning, planning effort, wisdom-only refusal,
   approximate planner time limits and hardware/binary dependence.
3. Numba manual, [The threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html).
   Checked for startup selection, import-time thread ceiling, runtime masking
   and process-safety restrictions.
4. OpenMP 5.1, [§6.7, `OMP_WAIT_POLICY`](https://www.openmp.org/spec-html/5.1/openmpse64.html).
   Checked for active/passive semantics and their status as hints.
5. OpenMP 5.1, [§6.4, `OMP_PROC_BIND`](https://www.openmp.org/spec-html/5.1/openmpse61.html).
   Checked for binding policy semantics; actual placement remains runtime-dependent.
6. Python standard library, [subprocess](https://docs.python.org/3/library/subprocess.html).
   Checked for explicit child environments and timeout limitations around process
   creation; shutdown must record actual elapsed time.
7. solprop, source at `c099545cf4a2b2f433f4bd5e1303315352d77b62`:
   [backend implementations and registry](https://github.com/LupoLab/solprop/tree/main/src/solprop/backends),
   [driver and scan executors](https://github.com/LupoLab/solprop/tree/main/src/solprop/driver),
   [translation](https://github.com/LupoLab/solprop/tree/main/src/solprop/translate),
   [output schema](https://github.com/LupoLab/solprop/blob/main/src/solprop/output/schema.py).
   Companion specifications: [kernel contract](../kernel-contract.md),
   [stepper](../stepper.md), [accuracy](../accuracy.md),
   [backends](../backends.md), [benchmarks](../benchmarks.md).
8. solprop-kokkos, source at `87ac816190ffb0c3978de7e236b6e30d7c068ab8`:
   [Python adapter and native workspaces](https://github.com/LupoLab/solprop-kokkos/tree/main/src),
   [native contracts](https://github.com/LupoLab/solprop-kokkos/tree/main/include/solprop_kokkos),
   [CMake build choices](https://github.com/LupoLab/solprop-kokkos/blob/main/CMakeLists.txt),
   [host libraries](https://github.com/LupoLab/solprop-kokkos/blob/main/cmake/HostLibraries.cmake).
   See also the [Kokkos guide](../../guide/performance/native.md).

The repository sources above were inspected at the stated revisions. Primary
documentation in references 1–6 was checked when drafting; it does not establish
runtime behaviour for every installed library build. No new hardware timings,
GPU experiments or prototype tuning study were performed for this proposal.

## Part II — Implementation record

## Part III — History

- 2026-09-27: Drafted from an audit of current execution controls. Propose bounded,
  explicit CPU tuning; retain native GPU policies, chunking, propagator selection
  and automatic whole-run decisions as separate work.
- 2026-10-17: Restated for [SEP 25](archive/sep-0025-rounding-level-reproducibility.md).
  In §7.1, replay checks keep identical edges (exact accounting of a prescribed
  schedule) and compare fields, error estimates, statistics and diagnostics within a
  rounding bound fixed before measurement; the repeated-run and thread-invariance
  promises become that bound on a matched fixed schedule and the case's accuracy
  envelope for adaptive runs; the bounded operation's agreement with an ordinary
  run keeps edges, counts, save planes and identities exact and bounds the values.
  §9's disclaimer of bitwise equality across FFT planners and BLAS libraries stays,
  naming the two kinds of bound. Configuration hashes, plan and report digests and
  witness digests remain exact identity checks.
