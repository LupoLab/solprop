# SEP 48 — A Julia backend: one engine for CPUs and GPUs, in both precisions

| | |
|---|---|
| **SEP** | 48 |
| **Title** | A Julia backend: one engine for CPUs and GPUs, in both precisions |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-10) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-11-10 |
| **Resolution** | — |
| **Requires** | SEPs 25, 33, 46 and 47 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add `backend="julia"`, an optional execution backend written in Julia. One engine runs
on CPUs and on NVIDIA, AMD and Apple GPUs, in binary64 and in binary32. It implements
SEP 47's execution session in full: preparation, the march, its controller and history,
the boundary monitor, the statistics and export are all owned in Julia, behind the same
`PropagationProblem` and `RunRequest` that the native backend receives. A plain
`pip install solprop` is unchanged and installs no Julia. `pip install "solprop[julia]"`
adds only the Python bridge, `juliacall`. An explicit `solprop julia setup` then installs
a pinned Julia and a pinned package environment. Until that has run, the backend refuses
by name and downloads nothing.

The backend enters at Tier 3 (ADR-061). It is kept runnable and correct for what it
runs, and refuses everything else by name. Success is judged in four ways:

- parity with `numpy_ref` on every capability it admits;
- the session conformance suite passing over it;
- a qualification pass on each GPU vendor;
- recorded performance and start-up costs against native.

Performance is recorded, not a gate. Replacing native is not proposed.

### 2. Motivation and scope

**A second complete implementation is a cross-check.** The native backend is solprop's
production engine. Its correctness is decided against `numpy_ref`, which is unthreaded
and slow, and its performance has had no comparable alternative since `numba_cpu`
(SEP 43) and `mlx` (SEP 47) were retired. An experiment, PR #559, ran solprop's focus
routes in Julia from the same translated problem. It repeatedly showed native where it
could be faster or was wrong:

- the thread budget counted second-tier cores on Apple silicon (#333);
- the topology was re-read on every small pass (#556);
- the benchmark harness timed metadata (#557);
- seven optimisation ideas, some since adopted by native (#560, #584).

Native's CPU push in #576 then moved the Apple M5 Max comparison from 0.82 to about
1.05 (Julia time over native time). The comparison had paid for itself.

**GPUs without compiling.** Native's device builds are built from source, with a CUDA or
ROCm toolkit ([building native](../../guide/performance/building-native.md)). Julia's
GPU packages fetch their own runtime libraries, so a user with only a driver can run on
an NVIDIA, AMD or Apple GPU.

**Single precision beyond the oracle.** With `mlx` retired, `precision="single"` runs
only on `numpy_ref`. Native's reduced precision was withdrawn (SEP 36). The experiment
implements SEP 33's contract on every device. On an Apple GPU, which has no binary64, it
forms the wide quantities in double-float arithmetic.

**Apple GPUs.** Kokkos has no Metal backend. On the M5 Max GPU, at its branch's head,
the experiment took 0.66 of `mlx`'s time over seven cases.

**Compactness.** The experiment's engine is about 3,500 lines of Julia: one source for
every device and precision, with each device added as a package extension. Native is
about 31,000 lines of C++. The two do not cover the same features. Even so, a feature
costs far less to write in the Julia engine, which also makes it a quick place to try an
optimisation before porting it to C++.

**This SEP's aim is a viable, mergeable, optional backend.** It is not a decision about
native's future. Whether Julia could later displace native depends on completeness and
measured performance across machines. §7.6 records what would bear on that, without
proposing it.

**In scope:**

- the Julia engine and session, on the routes, responses, integrators and session
  services of §5.3;
- binary64 and binary32 (SEP 33), with binary64 or double-float as the wide type;
- the CPU, and the CUDA, ROCm and Metal package extensions, each qualified once by hand
  before merge;
- distribution as an optional extra, with an explicit setup command;
- the rules, tools and CI lane for the Julia code;
- documentation, and a Tier 3 ADR.

**Out of scope:**

- `auto` selecting `julia`, or any change to native or its wheels;
- Tier 2 or Tier 1 status, which need a later ADR;
- a system image, which is deferred by the maintainer;
- a trimmed shared library (`juliac --trim`);
- a Julia-facing user API (calling solprop from Julia);
- new physics;
- the capabilities §5.3 refuses: they come later, each by work proposed for it (Tier 3);
- distributed runs.

### 3. Context

#### 3.1 Backends after SEP 47

SEP 47 ([#605](https://github.com/LupoLab/solprop/pull/605),
[#607](https://github.com/LupoLab/solprop/pull/607)) makes every backend the sole
owner of a run:

- **The descriptors.** Translation produces an immutable `PropagationProblem`. Its route
  is a closed union: `ModalRoute`, `GeneralRoute`, and `SpatialRoute` with the kinds
  radial, Cartesian, planar and line. The run's policies are a `RunRequest`
  (`translate/run_request.py`), resolved by Python before preparation.
- **Preparation.** A backend prepares a `PreparedExecution`, which starts a `RunSession`
  (`backends/session.py`).
- **The session.** It advances until it has an event from a closed set
  (`backends/events.py`). The service loop (`driver/service.py`) handles output
  transactions, warnings, snapshots and callbacks.
- **Who runs it.** `numpy_ref` runs on a host session with independent numerical code.
  Native runs its own session in C++ (`native::session::Session`). The inherited
  `StepDriver` remains only as the native bridge, and is deleted at SEP 47's Final.
- **Settled rules.** SEP 47 rules out any host loop driving compiled kernels, and
  retires `mlx`.

[backends.md](../backends.md) §§9–11 list what native and `numpy_ref` run today.
ADR-061 sets the tiers: a new backend starts at Tier 3, and is kept runnable and correct
for what it runs. ADR-064 and ADR-068 make `native` the primary backend and Tier 2
empty.

#### 3.2 The experiment

PR #559 (branch `feature/julia-backend-experiment`, head `b84d3ab8`; its
[status page](https://github.com/LupoLab/solprop/blob/b84d3ab81df863669e57aeca6c0ac3abaf6820ac/julia/STATUS.md))
continues the CPU prototype of #10, recorded in #368.

**How it is built.** Every state-sized operation is a scalar body launched by a small
device layer:

| Device | Each pass | Reductions and scans | FFT | Products |
|---|---|---|---|---|
| CPU | Polyester tasks over contiguous SIMD loops | the same tasks | FFTW, as blocks of single-threaded plans | Accelerate on macOS, OpenBLAS elsewhere |
| GPU | KernelAbstractions kernels | AcceleratedKernels | cuFFT, rocFFT or Metal's FFT | the vendor's BLAS |

CUDA.jl, AMDGPU.jl and Metal.jl attach as package extensions. Binary32 follows SEP 33:
the state is held as $a/s$, and the operator's exponents, the propagators and the Adams
$\varphi$ functions are formed in a wide type and rounded once. On Metal the wide type is
`DF32`, a value held as two binary32s.

**How fast.** Window times over native's, from `python -m benchmarks ab` with both arms in
one interpreter, as recorded on the branch and not re-measured:

| Machine | Precision | Julia ÷ native |
|---|---|---|
| Apple M5 Max CPU, 6 threads | binary64 | 0.82 over whole runs; about 1.05 after native's #576 |
| Intel Xeon Gold 6240, 16 threads | binary64 | 1.13 |
| NVIDIA A40 | binary64 | 1.15 |
| NVIDIA A40 | binary32, double-float wide | 0.79 (against native's binary64) |
| NVIDIA H100 | binary64 | 1.25 |
| AMD Instinct MI300X | binary64; binary32, double-float wide | 1.28; 1.48 |
| Apple M5 Max GPU | binary32 | 0.66 of `mlx`'s time |

On GPUs with full-rate binary64 the remaining gap is host time between launches and
between bridge calls. On an A40 the device work equals native's. SEP 47's coarse session
boundary removes most of the bridge share.

**What it covers.** The resolved and mode-averaged capillaries, radial, Cartesian and
line free space, and lockstep batches. Kerr; field and envelope plasma with several
channels; scalar Raman with several components, varying along a gradient; tapers.
DOPRI5, TSIT5, the exponential Adams march and RK4IP.

**What it refuses.** Vector polarisation, tensor Raman, noise, dispersive Kerr, the
planar and general modal routes, coupled flows, TDSE, the interaction formulation, the
carrier frame, interpolated planes and binary64 on Metal.

**What did not carry over.**

- **The adapter.** About 2,400 lines of Python ride `StepDriver`, which SEP 47 deletes.
- **The configuration surface.** Thirteen environment variables.
- **The compile trace.** A generated precompile trace of 1,600 lines.
- **The build check.** A source digest checked when the engine loads.
- **The bootstrap.** It works only from a source checkout.

**What it taught.** None of these has a workaround in SEP 47's design yet:

- **Flush-to-zero is hazardous embedded in Python.** Julia's first thread is the
  interpreter's, so a flag set there flushed NumPy's subnormals too (#561).
- **The toolchain has bugs.** Julia's LinearAlgebra gave wrong concurrent
  `Complex{Float32}` × `Float32` products; the experiment worked around them.
- **Some GPU costs are structural.** On Metal, each value the host waits for costs about
  0.2 ms.
- **Start-up is slow.** The first engine in a process compiled for 11 to 14 s on a CPU,
  and for about 28 s on Metal, whose kernels compile at run time.
- **The install is large.** The package depot is about 1.9 GB.

#### 3.3 Issues

- **To resolve first:** #588 (SEP 47). This SEP builds on its session contract and
  descriptors, and needs at least its native-session milestone merged.
- **Fixed or closed by this SEP:**
  - #368, the prototype's record: closed when this SEP is Final.
  - PR #559, the experiment: closed in favour of this SEP's implementation PR.
  - #557, the harness timing metadata: the fix on the experiment's branch lands here, or
    in its own PR first.
- **Related, and changed by this SEP:**
  - #560, native ideas from Julia: Julia's lane keeps supplying them.
  - #590, FFTL as a CPU FFT provider: an adapter for Julia's plans would be an option
    here.
  - #561, the flush-to-zero policy: Julia keeps SEP 25's rule.
  - #333, Apple's core tiers: Julia reads the same thread budget.
  - #356, lockstep throughput on GPUs: Julia's groups are member sessions, as native's
    are.
  - #596, RK checkpoint continuation: Julia follows the same deferral.

### 4. Theory

The backend adds no physics, and the numerical problem is unchanged. Everything below
is the meaning a Julia session must preserve, and the arithmetic it uses to do so.

**What must agree.** A Julia session integrates the problem that `PropagationProblem` and
`RunRequest` specify, with the methods, tolerances, observation schedule and boundary
policies they name. Under [SEP 25](archive/sep-0025-rounding-level-reproducibility.md)
it agrees with `numpy_ref` and with native to rounding, never bit for bit across
backends. Contraction (`muladd`) and per-kernel reassociation are permitted. Umbrella
fast-math, approximations beyond a function's stated accuracy, and flush-to-zero are
forbidden.

SEP 47 §4.3 sets the identity of control: the scalar decisions of the controller, the
monitor and the landing are the same on the same recorded readings. Array reductions and
transcendentals are each backend's own reading, within SEP 25's bounds.

**Binary32** follows [SEP 33](archive/sep-0033-single-precision-mlx.md) §§4.1–4.4
exactly:

- **The state.** It is held as $a/s$, with $s$ a power of two fixed at launch.
- **The wide quantities.** The operator's exponents $\Phi$, the propagator pair
  $e^{\pm M\Delta\Phi}$ and the Adams $\varphi_k$ are formed in a wide type, then
  rounded once to binary32.
- **Everything else.** The other buffers obey SEP 33's range rules.
- **The admitted set.** It is `numpy_ref`'s single-precision admission, including the
  `rtol` floor and cap of `translate/precision.py`.

**The wide type** is binary64 by default. Where binary64 is absent (Apple GPUs) or slow
(GPUs whose binary64 rate is a small fraction of binary32's), it may be the
**double-float** $x = x_h + x_\ell$, an unevaluated sum of two binary32 values with
$|x_\ell| \le \tfrac{1}{2}\,\mathrm{ulp}(x_h)$:

- **Its arithmetic** uses the error-free transformations:
  - TwoSum (Knuth; Dekker [2]);
  - TwoProd with a fused multiply–add (Dekker [2]).
- **Its precision** is close to $u_{\mathrm{df}} = 2^{-48}$, with the rigorous bounds of
  Joldes, Muller and Popescu [3] for addition, multiplication and division. Here
  $u_{\mathrm{df}} = u_{32}^2/2$, and $u_{32} = 2^{-24}$ is binary32's unit roundoff.

The experiment's double-float $\varphi_k$ agree with binary64's to within $2^{-36}$, and
its propagator pair to 8 binary32 ulps up to a phase of $10^6$ rad. §7.3 makes these
recorded bounds.

A binary64 run never uses double-float. A binary32 run's wide type is an execution
option recorded in the run's metadata. It cannot change the admitted problem, only the
rounding within SEP 33's bounds.

**Transcendentals on GPUs.** A device's library `sin`, `cos`, `exp` and `log` may be
less accurate than the CPU's: Metal's `sin` and `cos` are 2 to 3 ulp. Wherever an
iteration's convergence depends on such a function, the engine uses functions of stated
accuracy, as the experiment's quadrant-reduced polynomial `sincos` (about 1 ulp) does for
the envelope plasma's cycle quadrature. A function's accuracy is stated beside it.

### 5. Proposed interface

#### 5.1 Installation and setup

```text
pip install "solprop[julia]"        # adds juliacall; no Julia is downloaded
solprop julia setup                 # Julia and the CPU environment, precompiled
solprop julia setup --device cuda   # also the CUDA environment (rocm, metal likewise)
solprop julia status                # what is set up, where, and at which versions
```

- **`pip install solprop` installs no Julia.** The `julia` extra adds only `juliacall`
  (with its dependency `juliapkg`), pinned to a tested range. The Julia sources ship
  inside the `solprop` wheel as package data (§6.1). Importing solprop never imports
  `juliacall` or starts Julia.
- **Before setup, `backend="julia"` refuses by name.** The `BackendError` says what is
  missing (the extra, or the setup) and gives the command to run. Nothing downloads
  implicitly. `registry.available()` lists `julia` only when `juliacall` is importable
  and a set-up environment for this solprop version exists. It checks this from files,
  without starting Julia.
- **`solprop julia setup`** installs the pinned Julia release, unless a matching one is
  found (§6.5). It then instantiates the committed environment of each requested device
  and precompiles it, including the engine's compile workload.
  - The packages go to a private depot below the calibration store directory
    (`solprop.calibration.store_directory`, so `SOLPROP_CACHE_DIR` moves it).
    `SOLPROP_JULIA_DEPOT` names another.
  - The setup can run on an HPC login node and be used offline on compute nodes that
    share the file system.
  - `--cpu-target` sets Julia's CPU target for a heterogeneous cluster's precompiled
    code.
  - `--cuda-local` makes CUDA.jl use the system toolkit instead of its own, which is
    needed beside a native CUDA build in the same process (§6.5).
  - Setup is idempotent, prints its time and the depot's size, and removes nothing it did
    not create.
- **`solprop julia status`** reports, for each environment:
  - the Julia version, and the engine's and the device packages' versions;
  - the depot's location and size;
  - whether the precompiled code matches this machine.

#### 5.2 Selecting the backend

```text
from solprop import Simulation, SimulationSettings

sim = Simulation(..., settings=SimulationSettings(backend="julia"))
sim_single = Simulation(..., settings=SimulationSettings(backend="julia", precision="single"))
```

- **The device.** `SOLPROP_JULIA_DEVICE` (`cpu`, the default, `cuda`, `rocm` or `metal`)
  is read once, when the process starts its Julia runtime. A process holds one Julia
  runtime, whose environment is fixed when it starts, so the device is a per-process
  choice (Q2). A run in a process whose runtime holds another device's environment
  refuses by name. `SOLPROP_REQUIRE_DEVICE` gains `metal` and applies to `julia` as it
  does to `native`: `cuda` accepts Julia's CUDA, `hip` its ROCm, and `any` every
  device.
- **Precision.** `precision="single"` is admitted: `julia` joins
  `SINGLE_PRECISION_BACKENDS`. Metal is binary32 only, and binary64 refuses there by
  name. The wide type defaults to double-float on Metal and binary64 elsewhere, and is
  chosen by `ExecutionOptions.backend_options["wide"]` (`"binary64"` or `"df32"`).
- **Threads.** The runtime starts with the process's thread budget (ADR-052; SEP 46's
  calibration). A run's width is at most that budget, and Julia's tasks are its only
  pool. Host BLAS is held to one thread while an engine lives, as native does.
- **FFT plans.** Julia keeps the FFTW plans it measures, by transform, in the
  calibration store, as native does (ADR-076).
- **Metadata.** Every run records `julia` as its backend, and:
  - the Julia version and the engine's version (solprop's);
  - the device and its package versions;
  - the environment's manifest digest;
  - the wide type, the thread count and the CPU target.

  It records no schema change (§9).

#### 5.3 What it runs at merge

| Capability | CPU | CUDA, ROCm | Metal |
|---|---|---|---|
| Modal route, mode-averaged and multimode, field and envelope grids | yes | yes | binary32 |
| Radial, Cartesian and line free space | yes | yes | binary32 |
| Kerr, field and envelope plasma (scalar, several channels), scalar Raman (several components, constant or varying along a gradient), tapers | yes | yes | binary32 |
| Temporal absorbers and collars, including the split collar (SEP 21); the transverse collar (SEP 40) | yes | yes | binary32 |
| `Multistep` (exponential formulation), the embedded Runge–Kutta tableaux adaptive and replayed, RK4IP, the low-storage replay | yes | yes | binary32 |
| Session services: planes, the periodic-capable statistics, the temporal monitor, the transverse edge, dense output, snapshots and resume, stop tokens | yes | yes | binary32 |
| The input noise model (SEP 37's default) | yes | yes | binary32 |
| Fixed-schedule groups, as member sessions | yes | yes | binary32 |
| `precision="single"` | yes, binary64 or double-float wide | yes, either | yes, double-float wide |

Refused by name at merge, each naming `native` or `numpy_ref`:

- the general modal route, the planar kind and coupled flows;
- TDSE responses;
- vector polarisation and tensor Raman;
- the dispersive electronic response;
- the modified noise model;
- the interaction Adams formulation and the carrier frame;
- operators that took the z table;
- the far-field statistics (SEP 23);
- comparison methods and custom controllers, which are reference-only under SEP 47;
- binary64 on Metal.

Each later addition is proposed for this backend (Tier 3), or arrives with a feature's
SEP when that SEP says so.

#### 5.4 Process model

- **Starting the runtime.** The runtime starts at the first `julia` preparation in a
  process, never on import. It refuses if `juliacall` was already initialised by
  something else, which would hold another environment.
- **Workers.** Scans, the service client and the GUI's jobs already use spawned worker
  processes. Each spawned process starts its own runtime and pays its start-up once.
  `fork` after the runtime has started is unsupported, and refused where solprop
  controls it.
- **Stopping.** A run's stop token is a shared flag that the Julia session polls between
  attempts, as SEP 47 §5.5 requires. Julia never calls back into Python while advancing.
- **The GIL.** An advance releases it.
- **First-run cost.** The first run in a process pays the runtime's start, package
  loading and any remaining compilation. §7.5 records these costs, and the manual states
  them.

### 6. Implementation strategies

#### 6.1 Layout

The code lives in four places:

- **`julia/`, the Julia project.** It sits at the repository root, beside `native/`:
  - the package `SolpropJulia`: `Project.toml`, `src/`, the device extensions in `ext/`
    and the Julia tests in `test/`;
  - `julia/environments/{cpu,cuda,rocm,metal}/`, the committed environments setup
    instantiates (§6.5);
  - `julia/AGENTS.md` (§6.7).
- **`src/solprop/backends/julia/`, the Python adapter:**
  - the registry entry and admission (`refusal`);
  - the lowering of descriptors to transfer records;
  - `JuliaPreparedExecution`, `JuliaSession` (a `RunSession`) and `JuliaGroupSession`;
  - the runtime's start;
  - the `solprop julia` commands.
- **The wheel.** `julia/` is force-included into the `solprop` wheel as package data,
  a few hundred kilobytes. The `julia` extra is added by `hatch_build.py` beside `gui`.
- **The environments.** None is created until setup runs.

#### 6.2 The session in Julia

The Julia session owns everything native's session owns (SEP 47 §§5.2–5.5):

- the field and the march: Adams, the embedded tableaux and the low-storage replay;
- the controller and the method history;
- the cursors and the row cadence;
- the monitor's decisions and records, and the transverse edge;
- the periodic statistics, computed in the engine;
- the snapshot latch and the export generation;
- the pending batch.

The Python adapter only translates. It lowers the descriptors once, prepares, forwards
`advance`, `acknowledge` and `export`, and builds the closed event types from compact
records. It keeps no numerical state and implements no host-over-Julia trial or
statistics path: SEP 47's rule against a host executor holds for every compiled backend.

**Independence.** The control logic may be transcribed from native's session, because its
decisions are specified to be identical on recorded readings. The experiment's Adams
march was transcribed from native's in the same way. The numerical kernels stay
independent implementations, and `numpy_ref` remains the oracle for numbers.

**The adapter's dependencies.** It imports solprop's backend-neutral descriptors,
translation and stateless numerical modules, never `solprop_native`. A piece of native's
Python lowering that is in fact backend-neutral moves into solprop, as a stateless
module, when Julia needs it.

**The boundary.**

- **Arrays.** On the CPU, NumPy arrays cross as zero-copy views. Preparation copies
  them once into storage the engine owns, establishing SEP 47's ownership at preparation.
- **Export.** Julia writes into the destination arrays the service loop provides
  (SEP 47 §5.4). On a GPU, the field crosses to the destination once.
- **The engine does not depend on Python.** `SolpropJulia` has no PythonCall dependency.
  The bridge's entry points, which unpack Python objects, are a PythonCall package
  extension that loads when `juliacall` starts the runtime. The engine therefore tests
  under `Pkg.test` with no Python, its quality tools see no bridge types, and a later
  C-ABI library stays possible.

#### 6.3 The device layer

The experiment's device layer carries over (§3.2):

- **Kernel bodies** capture only values and arrays, and are inlined into the kernel. On
  CUDA, not inlining them cost up to twenty times.
- **Graphs.** A source's passes replay as CUDA graphs where measured to pay.
- **Long transforms.** Long binary64 transforms take the four-step split on CUDA.
- **Cartesian planes** are transposed before the transverse transform on CUDA only,
  where measured.
- **Device choices stay measured.** Each is A/B'd per vendor, and none is assumed from
  another vendor.

What the experiment found against, and reverted, stays out unless re-measured: a
whole-trial graph, an order-independent one, and fused device reductions.

#### 6.4 Precision

Binary32 admission is `numpy_ref`'s single-precision admission intersected with §5.3.
The state scale, the narrowed tables and the exponent checks follow SEP 33 and
`translate/precision.py`, shared with the reference through the descriptors. The
double-float primitives live in one Julia file, with their error bounds stated beside
them (§4).

The experiment's open performance question, binary32 at two to three times binary64's
time on an x86 CPU (subnormals ruled out), is investigated in phase 3. It does not
block merge.

#### 6.5 Distribution

There are two ways to deliver option A, a run-time Julia through `juliacall`:

**A1, juliapkg-managed.**

- A `juliapkg.json` in the package declares Julia and the engine, the engine by `path`.
- juliapkg resolves compat ranges into a project in the virtual environment, merging the
  declarations of every installed Python package, and installs Julia through `juliaup`
  or a download.
- Offline mode, `PYTHON_JULIAPKG_OFFLINE`, gives the refuse-until-setup behaviour.
- The standard route, with the least code. But the resolved versions are whatever the
  ranges admit on the day, one depot of about 2 GB per virtual environment.
- Another `juliacall` package in the same environment shares, and can conflict with,
  the project.

**A2, solprop-managed (recommended).**

- Setup installs a pinned Julia: an existing `juliaup` channel or a `julia` on the
  `PATH` of the exact pinned minor, otherwise the official release, checked against its
  recorded SHA-256 as the experiment's bootstrap does.
- It instantiates the committed `Manifest.toml` of each device environment into the
  private depot.
- At run time solprop points `juliacall` at that Julia and environment through
  `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT`, which bypasses juliapkg's
  resolution.
- **Why A2:**
  - It gives refuse-until-setup naturally.
  - It runs exactly the package versions the tests ran, as native's wheels do.
  - One depot serves every virtual environment of a user. Each solprop version has its
    own environment directories in it, and they share package files.
  - It cannot be disturbed by other `juliacall` packages.
- **The cost:**
  - A manifest per device environment to keep current (a scheduled update PR).
  - The table of Julia releases and digests that setup downloads.

**Julia version.** Julia 1.13 is the minimum (the experiment's version; FFTL's compiled
library misbehaved on 1.12.6, #590). The patch release is pinned and recorded, and
raised by a PR that reruns the Julia lane.

**Platforms.** Setup and the CPU lane cover Linux x86-64 and macOS arm64. Linux aarch64
and Windows x64 are attempted by setup but unqualified at merge (Q5). Metal needs macOS
on Apple silicon; CUDA and ROCm need Linux.

**Coexistence with native in one process** is supported. The experiment's A/B harness
ran both arms in one interpreter, on Linux and macOS.

- **On a CPU,** Julia's FFTW and OpenBLAS come from its own artifacts. Native's wheel
  bundles its own, and on Linux its OpenBLAS symbols carry a `solprop_` prefix. Phase 1
  repeats the check on the distribution wheel.
- **On CUDA,** CUDA.jl's runtime libraries clash with a native build linked to a system
  CUDA (an undefined cuBLASLt symbol on an H200). `--cuda-local` makes CUDA.jl use the
  system toolkit.

**Licences.** Julia and the engine's packages are MIT or BSD licensed. FFTW is GPL,
compatible with solprop's GPL-3.0-or-later. The vendor runtimes that CUDA.jl and
AMDGPU.jl download are fetched on the user's machine under their own terms; solprop
redistributes none of them. Every file in `julia/` carries REUSE headers.

**Later, out of scope.**

- A system image, deferred by the maintainer, would remove most of the first-run
  compilation. GPU kernels would still compile at run time: GPUCompiler keys its cache on
  the device and keeps none on disk.
- A trimmed shared library (JuliaC, `--trim`) could ship the CPU engine in a wheel with
  no Julia install. It needs trim-safe code throughout and has no GPU path.

The C-ABI-friendly boundary of §6.2 keeps both possible.

#### 6.6 Tests and CI

**Julia tests**, `Pkg.test` in `julia/`, in two groups:

- **`Core`:**
  - the shared helpers (Chebyshev rows, Adams basis) bit for bit against recorded
    values;
  - the double-float primitives against their bounds;
  - every FFT plan form the engine uses, on each array type present;
  - a warmed step that allocates nothing on the CPU (`@allocated`).
- **`Quality`:** `Aqua.test_all` and `JET.test_package` (error analysis), both blocking.

**Python tests**, `tests/backends/julia/`:

- parity with `numpy_ref` on each capability of §5.3 (§7.1);
- the SEP 47 conformance suite (`tests/sessions/`) parametrised over `julia`;
- the refusals;
- availability and setup detection without starting Julia;
- metadata;
- a subprocess test that `import solprop`, and a `native` run, never import `juliacall`.

**Bounds checked, and device-safe on the CPU.** Tests run with bounds checking on. On the
CPU they run with the experiment's device-safety check, which refuses any kernel body a
GPU could not take.

**CI.** A `julia` job on Linux x86-64:

- runs setup with the depot cached by manifest digest, the Runic check, `Pkg.test`, and
  the Python Julia modules;
- runs on every pull request whose diff is not documentation alone (Q3), and on macOS
  arm64 weekly and on changes to `julia/`;
- is independent of the coverage lane, which omits `backends/julia/` as a recorded
  omission (as `mlx` was).

GPU tests (`pytest -m gpu` with `SOLPROP_JULIA_DEVICE`) run by hand, once per vendor
before merge and at each qualification pass, as native's device builds do.

#### 6.7 Rules for the Julia code

`julia/AGENTS.md` is short. It adopts the parts of solprop's rules that carry over and
the Julia ecosystem's tools, and nothing more.

- **Formatting.** Runic, with no configuration, checked in CI. Lines of at most 92
  characters.
- **Tools.**
  - **Blocking:** Runic, Aqua, and JET's `test_package` error analysis.
  - **Advisory:** `JET.report_opt` on the hot code a change touches; a straightforward
    instability is fixed.
  - **Not adopted:** ExplicitImports; explicit imports remain a written rule.
- **Docstrings.**
  - Only the engine's public surface gets them: the bridge's entry points and the types
    and functions declared `public` or exported. They are in the form of the Julia
    ecosystem's conventions: a hand-written signature line, an imperative summary, then
    `# Arguments`, `# Keywords`, `# Returns` and `# Throws` as they apply, with units
    and ranges for every numerical quantity.
  - Internal functions and helpers have none. A comment explains what is not obvious.
  - Physics comments cite the manual, the SEP or the paper rather than re-deriving them.
- **Hot paths.**
  - A kernel body is a scalar function of its index that captures only values and
    arrays.
  - Advancing allocates nothing on the CPU, which is tested. Preparation may allocate.
  - Struct fields are concrete or parametric.
  - Code is generic over `Float32`, `Float64`, `DF32` and the device array types; it is
    written against `similar`, `eachindex` and broadcasting, never `Array` alone.
  - `@inbounds` is allowed only where measured, on the narrowest expression, with the
    reason in a comment. The bounds-checked test runs catch a slip.
- **Floating point.** SEP 25: `muladd` permitted, no `@fastmath`, no flush-to-zero.
- **Errors.** Admission refuses in Python before allocation. Julia throws specific
  exception types for invariants and failures. The adapter turns them into
  `BackendError` or `ConvergenceError`, keeping the cause.
- **Dependencies.**
  - Each has a `[compat]` bound.
  - Adding one is a design decision, recorded in [dependencies.md](../dependencies.md).
  - GPU packages are weak dependencies with package extensions.
  - The manifests are committed, a deliberate difference from a Julia library's
    practice, because setup instantiates exactly them (§6.5).
- **Fast checks while iterating.** One `Pkg.test` group (`GROUP=Core`), the Python Julia
  module in question, and the Runic check.

#### 6.8 Tier 3 obligations

- **Runnable and correct.** `julia` is kept runnable and correct for what it runs, and
  refuses the rest by name (ADR-061).
- **Breaking changes.** A change that breaks the Julia lane does one of two things in the
  same pull request: it fixes the adapter, or it makes `julia` refuse the affected case
  by name, with an issue to restore it. It never leaves the lane red.
- **New features.** A new feature's SEP need not implement it on `julia` unless it says
  so.
- **Promotion** to Tier 2 is a later ADR. §7.6 lists the evidence it would weigh.

#### 6.9 Cost

**Code.**

| Part | Size |
|---|---|
| Engine | about 4,000 to 5,500 lines of Julia: the experiment's 3,500, plus the input noise and the remaining statistics |
| Session | about 2,000 to 3,000 lines of Julia, mostly scalar host logic |
| Python adapter | about 1,000 to 1,500 lines |

**Install.** About 2 GB of depot with one GPU environment.

**Start-up.** A process's first run pays the runtime start, package loading and the
compilation the precompiled workload missed: 11 to 14 s on a CPU in the experiment,
before this SEP's workload, and about 28 s on Metal. §7.5 records the costs. Reducing
them is phase 2 and phase 4 work; a system image is the later remedy.

**Memory.** A device state is native's, and the depot does not count toward a run's
memory ledger.

#### 6.10 Dependencies (to agree before acceptance)

- **Python:** `juliacall`, in the `julia` extra only.
- **Julia, engine:**
  - FFTW.jl, Polyester.jl, KernelAbstractions.jl, AcceleratedKernels.jl, Atomix.jl,
    LinearAlgebra and PrecompileTools.jl;
  - PythonCall.jl, as a weak dependency for the bridge extension;
  - CUDA.jl, AMDGPU.jl and Metal.jl, as weak dependencies.
- **Julia, test:** Aqua.jl and JET.jl.
- **Formatter:** Runic.jl, in its own environment.

Accelerate on macOS is reached through libblastrampoline's forwarding, with no extra
package.

### 7. Verification and validation

#### 7.1 Parity in binary64

Each admitted capability is compared with `numpy_ref` on a short window (about ten steps,
or a fibre short enough for seconds):

- **Fixed schedules** and fixed-order methods take the same steps.
- **Adaptive runs** take the same accepted steps, with the same control trace on equal
  readings (SEP 47 §4.3).
- **Fields and every statistic** agree within bounds derived from the operations'
  summation counts ([testing.md](../testing.md)). The experiment's windows agreed to
  $10^{-13}$ or better.
- **Groups** agree member by member with their solo runs.
- **Repeat runs** agree within SEP 25's bounds.
- **A resume** continues the history as native's does.

#### 7.2 Session conformance

The lifecycle conformance suite runs over `julia`, covering:

- receipts, barriers and close;
- bounded buffers and retained arrays;
- failing consumers;
- stop timing;
- snapshots between planes and field-free export;
- warnings as errors;
- yield invariance;
- group failure semantics.

A Julia exception inside an advance poisons the session as native's failures do. It
never leaves a half-committed state reported as valid.

#### 7.3 Binary32

On the CPU, with both wide types, and on each GPU:

- **Classes R and P** (SEP 33 §7.2) against `numpy_ref`'s single-precision path, within
  the binary32 bounds of SEP 33 §7.1.
- **Class T, survival** (SEP 33 §7.3), with its frozen envelopes, against binary64
  references on the same grids, on the soliton, dispersive-wave, plasma blue-shift,
  Raman, radial and Cartesian cases.
- **The propagator-drift record** for each wide type, with its stated constant and step
  count.
- **Double-float unit bounds** against binary64 and against Joldes–Muller–Popescu's
  bounds [3]:
  - the $\varphi_k$ to $2^{-36}$;
  - the propagator pair to 8 binary32 ulps up to $10^6$ rad;
  - the primitives themselves.
- **The envelope-plasma quadrature** converges within its frozen refinement on Metal's
  and the CPU's functions alike.

#### 7.4 Devices

Before merge, one qualification pass per vendor:

- **NVIDIA:** an A40 or an H100, CUDA.
- **AMD:** an Instinct MI300X or MI325X, ROCm.
- **Apple:** an M-series GPU, Metal.

Each pass runs the Julia Python suite with `-m gpu`. Repeat runs and group members must
agree within SEP 25's bounds. Each pass records the GPU package versions and the driver.
No sanitizer lane is required at Tier 3.

#### 7.5 Performance and start-up records

These are not merge gates. They are rows in the ledger of
[benchmarks.md](../benchmarks.md).

- **Windows.** `python -m benchmarks ab` with `julia` (and `julia32`) arms against
  `native`, on the standard cases' windows:
  - CPUs at 1 thread and the default budget: Apple M5 Max, Intel Xeon Gold 6240 and AMD
    EPYC 7763;
  - each GPU of §7.4, binary64 and binary32.
- **Whole runs** on two cases per machine.
- **Start-up:**
  - setup's time and the depot's size;
  - a process's first-run latency: the runtime start, package loading and compilation,
    on a CPU and on each GPU;
  - a spawned scan worker's overhead.
- **Regressions** against the experiment's last rows are explained in the phase report.

#### 7.6 Merge criteria, and evidence for later

**The backend is mergeable when all of these hold:**

- every capability of §5.3 passes §§7.1–7.4;
- everything else refuses by name;
- the Julia lane is green;
- `pip install solprop`, `import solprop` and native runs are unchanged (§6.6's
  subprocess test, and a wheel-metadata check that the base install requires no Julia
  package);
- §7.5 is recorded;
- the manual page exists.

**Not required here**, but kept current for a later tier decision:

- a coverage table against native's [backends.md](../backends.md) §9;
- the performance ratios per machine;
- the first-run costs.

### 8. Documentation

- **User guide,** a new page `docs/guide/performance/julia.md`:
  - installing and setting up, offline and HPC use, devices and precision;
  - what it runs and refuses;
  - the first-run cost;
  - troubleshooting: another `juliacall` user in the process, a native CUDA build beside
    it, a heterogeneous cluster's CPU target.

  It ends with a link to this SEP. The backends page's choice table and the installation
  page's extras gain `julia`.
- **Design pages:**
  - [backends.md](../backends.md) gains a `julia` section (layout, session, device layer,
    precision), and rows in §§9–11;
  - [dependencies.md](../dependencies.md) gains the Julia dependencies;
  - [tooling.md](../tooling.md) gains the Julia job.
- **`julia/AGENTS.md`,** and a pointer from the root `AGENTS.md`.
- **Records:**
  - an ADR at Final: `julia` enters Tier 3, distributed by explicit setup with a
    solprop-managed environment;
  - a `changelog.d/` fragment.
- **Docstrings** on the engine's public surface and the adapter (§6.7).

### 9. Compatibility and impact

**Unchanged:**

- defaults, `auto`, native and its wheels, and `pip install solprop`;
- the output schema: the new metadata keys are additive and need no bump between
  releases;
- the kernel contract, beyond SEP 47's session contract. The Julia backend is a
  consumer of it, and a reason to keep it backend-neutral.

**Added:**

- the `julia` registry name;
- `julia` in `SINGLE_PRECISION_BACKENDS`;
- `SOLPROP_REQUIRE_DEVICE=metal`;
- the `solprop julia` commands;
- a `julia` extra;
- a few hundred kilobytes of package data in the wheel.

**The cost to other work:**

- **CI time.** One more job on most pull requests.
- **Seam changes.** A change to the descriptors or the session contract keeps the Julia
  lane green, or refuses by name (§6.8). Contributors may otherwise ignore `julia/`.

### 10. Alternatives considered

- **Keep it an experiment.** Cheapest, but its adapter dies with `StepDriver`. Its value
  as a cross-check then needs a branch kept alive beside a moving main. At its last head
  it was 127 commits behind.
- **Delegate the environment to juliapkg (A1).** Standard, but its versions are unpinned
  and its depot is per environment (§6.5). Kept as the fallback if A2's maintenance
  proves heavier than expected.
- **A system image now.** Faster start-up, but a release artefact per platform, and GPU
  kernels still compile at run time. Deferred by the maintainer.
- **A trimmed library in a wheel.** No Julia install, but CPU only, and the whole engine
  and its dependencies would need to be trim-safe on a young toolchain. Revisit when
  JuliaC is mature, perhaps for the CPU engine alone.
- **A separate `solprop-julia` distribution, or registering `SolpropJulia` in Julia's
  General registry.** Either adds version coordination across two releases for code that
  only solprop calls. Package data versioned with solprop is simpler.
- **A host session driving Julia kernels.** It would reintroduce the host-over-compiled
  executor that SEP 47 removes, and keep the bridge costs the experiment measured.
- **Julia at Tier 1 now.** Premature: completeness and cross-machine performance are not
  yet shown, and every feature would pay a third implementation.
- **Native single precision, or MLX again.** SEP 36 was withdrawn and `mlx` retired;
  Julia covers both roles with one engine.

### 11. Phases

The SEP is implemented on one branch, `feature/sep-0048-julia-backend`, in one draft
pull request (SEP 0). The branch starts from main once SEP 47's native-session milestone
has merged, or stacks on that pull request if it is still open. The engine's numerical
code is moved from the experiment's branch and reshaped to §6.7. Its adapter is
rewritten.

1. **Package, distribution and rules.**
   - **Deliverables:**
     - the `julia/` package with its environments and `julia/AGENTS.md`;
     - Runic, Aqua and JET;
     - `solprop julia setup` and `status`, and the private depot;
     - availability, refusal and the runtime's start;
     - the wheel's package data and the extra;
     - the CI job.
   - **Verify:**
     - setup on Linux x86-64 and macOS arm64;
     - `pip install solprop` and `import solprop` unchanged;
     - refusal before setup;
     - a trivial engine call through the bridge.
2. **The CPU session in binary64.**
   - **Deliverables:**
     - preparation and lowering;
     - the Julia session with every service of §5.3 and groups;
     - the precompiled workload in place of the trace;
     - FFTW plans kept in the store;
     - the thread budget.
   - **Verify:** §§7.1–7.2 on the CPU, and the first CPU rows of §7.5.
3. **Binary32 on the CPU.**
   - **Deliverables:**
     - SEP 33's contract in both wide types;
     - the investigation of x86 binary32 speed.
   - **Verify:** §7.3 on the CPU.
4. **Devices.**
   - **Deliverables:**
     - the CUDA, ROCm and Metal extensions on the session;
     - graphs and the other measured device choices.
   - **Verify:** §7.4 for each vendor, §7.3 on each GPU, and the device rows of §7.5.
5. **Records and Final.**
   - **Deliverables:**
     - the manual and design pages;
     - the ADR;
     - the changelog fragment;
     - the remaining §7.5 rows.
   - **Final when:** §7.6's criteria hold. Open questions left go to issues.

### 12. Open questions

To settle before acceptance unless deferred:

- **Q1. The environment's manager.** Should it be A2, solprop-managed with committed
  manifests (recommended), or A1, juliapkg (§6.5)?
- **Q2. Device selection.** Should the device be a process-wide `SOLPROP_JULIA_DEVICE`
  (recommended, since the runtime's environment is fixed at start), or a per-run setting
  refused when it differs from the running environment?
- **Q3. CI trigger.** Should the Julia job run on every non-documentation pull request
  (recommended: the seam is wide and Tier 3 must stay runnable), or only on changes to
  `julia/`, `backends/`, `translate/` and `driver/service.py`?
- **Q4. The command's spelling.** `solprop julia setup` and `status` beside
  `solprop calibrate` (recommended), or a Python function only?
- **Q5. Platforms at merge.** Linux x86-64 and macOS arm64 qualified, with Linux aarch64
  and Windows x64 attempted but unqualified (recommended)? Or should Windows be refused?
- **Q6. The merge scope.** Is §5.3's table the right bar? In particular: groups, dense
  output and the low-storage replay at merge, and vector polarisation and tensor Raman
  left out.
- **Q7. The Julia version policy.** Should it be 1.13 as the minimum, with a pinned patch
  raised by pull request (recommended)?

### 13. References

1. J. Bezanson, A. Edelman, S. Karpinski and V. B. Shah, "Julia: A fresh approach to
   numerical computing," *SIAM Review* **59**, 65–98 (2017). doi:10.1137/141000671
2. T. J. Dekker, "A floating-point technique for extending the available precision,"
   *Numerische Mathematik* **18**, 224–242 (1971). doi:10.1007/BF01397083
3. M. Joldes, J.-M. Muller and V. Popescu, "Tight and rigorous error bounds for basic
   building blocks of double-word arithmetic," *ACM Transactions on Mathematical
   Software* **44**, 15 (2017). doi:10.1145/3121432
4. T. Besard, C. Foket and B. De Sutter, "Effective extensible programming: Unleashing
   Julia on GPUs," *IEEE Transactions on Parallel and Distributed Systems* **30**,
   827–841 (2019). doi:10.1109/TPDS.2018.2872064
5. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proceedings of
   the IEEE* **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301
6. PythonCall.jl and JuliaCall documentation,
   <https://juliapy.github.io/PythonCall.jl/stable/>.
7. JuliaPkg, <https://github.com/JuliaPy/PyJuliaPkg>.
8. JuliaC.jl, <https://github.com/JuliaLang/JuliaC.jl>.
9. Runic.jl, <https://github.com/fredrikekre/Runic.jl>; Aqua.jl,
   <https://github.com/JuliaTesting/Aqua.jl>; JET.jl, <https://github.com/aviatesk/JET.jl>.
10. KernelAbstractions.jl, <https://github.com/JuliaGPU/KernelAbstractions.jl>;
    AcceleratedKernels.jl, <https://github.com/JuliaGPU/AcceleratedKernels.jl>;
    Polyester.jl, <https://github.com/JuliaSIMD/Polyester.jl>.
11. solprop: SEP 25, SEP 33, SEP 43, SEP 46 and SEP 47; ADR-052, ADR-061, ADR-064,
    ADR-068 and ADR-076; PR #559 and its
    [status page](https://github.com/LupoLab/solprop/blob/b84d3ab81df863669e57aeca6c0ac3abaf6820ac/julia/STATUS.md);
    PRs #605 and #607; issues #333, #368, #556, #557, #560, #561, #588, #590 and #596.

## Part II — Implementation record

## Part III — History

- 2026-11-10: Drafted by Opus 5.5 for John Travers, after a feasibility analysis of
  PR #559. The maintainer chose an explicit setup over a first-use install, all three
  GPU vendors and binary32 at merge, and Runic, Aqua and JET as the blocking Julia tools.
  The tracking issue is still to be opened.
