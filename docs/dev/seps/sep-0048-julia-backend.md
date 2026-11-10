# SEP 48 — A Julia backend: one engine for CPUs and GPUs, in both precisions

| | |
|---|---|
| **SEP** | 48 |
| **Title** | A Julia backend: one engine for CPUs and GPUs, in both precisions |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5; GPT-6 (acceptance revisions, 2026-11-10) |
| **Status** | Accepted |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-11-10 |
| **Resolution** | Accepted by John Travers on 2026-11-10, including the implementation-plan revisions; acceptance and merge delegated in [PR #609](https://github.com/LupoLab/solprop/pull/609) |
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
ROCm toolkit ([building native](../../guide/performance/building-native.md)). Julia
compiles its kernels at run time, so nothing is built from source. What each vendor
still needs differs:

- **NVIDIA:** only a driver. CUDA.jl fetches the CUDA runtime and its libraries as
  artifacts.
- **Apple:** nothing. Metal.jl uses macOS's own frameworks.
- **AMD:** a system ROCm installation. AMDGPU.jl discovers ROCm's libraries, rocBLAS and
  rocFFT among them, rather than supplying them [11]. That is still less than native's
  HIP build needs, which is a toolkit and a compile.

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

- **To resolve first:** #588 (SEP 47), whose dependencies differ by phase (§11):
  - phase 1 needs nothing from it;
  - phase 2 needs its session contract, descriptors and conformance suite, which merged
    with #605, followed by its completed descriptor/request split and generic
    preparation entry point in #607;
  - production adapter integration and phase 4 wait for those changes to merge.
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
policies they name. It never agrees bit for bit with another backend. How closely it
agrees depends on what is compared, in three classes (§7.1 gives the tests):

- **Class M, a matched discrete history.** Both sides run the whole discrete computation
  identically. That means the same prescribed schedule, a method that makes no discrete
  choice on it (a fixed-order method, a Runge–Kutta tableau, the low-storage replay),
  matched startup and restart history, and matched save-plane handling. Fields and
  statistics then agree to rounding. The budget follows
  [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) §§4.3–4.4: the operation
  counts, together with the conditioning of what is computed and the sensitivity that the
  steps propagate.
  - A replay of `Multistep` is not class M. `FixedSchedule` records step lengths only,
    so the replay still chooses orders and may follow another history (SEP 25 §4.4).
- **Class A, independent adaptive runs, and ordinary variable-order replay.** Their
  reductions round differently, so an adaptive run may accept different steps and
  orders, and a replay may choose different orders. SEP 47 §4.3 permits this. Each run
  is then compared with a tighter or converged reference, within an accuracy envelope set
  by the run's tolerance. Each backend counts its own work exactly, and the two counts
  are not required to be equal.
- **Class C, control identity.** The scalar decisions of the controller, the monitor and
  the landing are the same, bit for bit, when fed the same recorded readings (SEP 47
  §4.3). Array reductions and transcendentals are each backend's own reading, within
  SEP 25's bounds.

Contraction (`muladd`) and per-kernel reassociation are permitted (SEP 25). Umbrella
fast-math, approximations beyond a function's stated accuracy, and flush-to-zero are
forbidden. The error-free transformations below are exempt from contraction (see their
hypotheses).

**Binary32** follows [SEP 33](archive/sep-0033-single-precision-mlx.md) §§4.1–4.4
exactly:

- **The state.** It is held as $a/s$, with $s$ a power of two fixed at launch.
- **The wide quantities.** The operator's exponents $\Phi$, the propagator pair
  $e^{\pm M\Delta\Phi}$ and the Adams $\varphi_k$ are formed in a wide type, then
  rounded once to binary32.
- **Everything else.** The other buffers obey SEP 33's range rules.
- **The admitted set.** It is `numpy_ref`'s single-precision admission, including the
  `rtol` floor and cap of `translate/precision.py`, narrowed for double-float by the
  phase domain below.

**The wide type** is binary64 by default. Where binary64 is absent (Apple GPUs) or slow
(GPUs whose binary64 rate is a small fraction of binary32's), it may be the
**double-float** $x = x_h + x_\ell$, an unevaluated sum of two binary32 values with
$|x_\ell| \le \tfrac{1}{2}\,\mathrm{ulp}(x_h)$.

- **Precision.** Let $u = 2^{-24}$ be binary32's unit roundoff. A double-float carries
  about $2p = 48$ bits, so its representation error is of order $u^2 = 2^{-48}$. That
  is not one bound shared by every operation: each operation has its own, a small
  multiple of $u^2$ that depends on the algorithm.
- **Error-free transformations.** Each is exact under its hypotheses:
  - TwoSum (Knuth), for any two binary32 values;
  - Fast2Sum (Dekker [2]), when $|a| \ge |b|$ (or $a$'s exponent is at least $b$'s);
  - 2MultFMA, the product with a fused multiply–add, when $ab$ neither underflows nor
    overflows.
- **Operations.** Each production operation is one that Joldes, Muller and Popescu [3]
  analyse, with the relative error bound proved there. Those theorems were formally
  re-verified, with corrections, by Muller and Rideau [12]. The candidates are
  AccurateDWPlusDW for addition, an FMA-based DWTimesDW for multiplication, DWTimesFP
  for a binary32 factor, and DWDivDW or DWDivFP for division. Phase 3 transcribes each
  bound and its hypotheses beside the code, after checking it against [3] and [12].
- **The experiment does not meet this yet.**
  - Its addition is the sloppy form: TwoSum on the high parts, then the low parts added
    in binary32. [3] shows that form has no useful relative bound when the operands
    nearly cancel. Production uses the accurate addition wherever cancellation can
    occur, or proves that it cannot.
  - Its division of two double-floats is a three-quotient long division that [3] does
    not analyse. Production replaces it with an analysed algorithm, or derives a bound
    for it.
- **Hypotheses.**
  - Binary32 arithmetic rounds to nearest even.
  - The fused multiply–add is correctly rounded.
  - Nothing inside a primitive is reassociated or contracted. Julia does neither without
    `@fastmath` or `muladd`, and a device compiler is checked by probe (§7.3).
  - No intermediate underflows in 2MultFMA or overflows. A device that flushes binary32
    subnormals loses the low word near the bottom of the range, so the operands' range
    is stated per use.
- **Transcendentals.** Each has its own bound, over the domain where it is used:
  - the series and recurrence of the $\varphi_k$, including their branch boundary;
  - the propagator pair, whose angle is reduced in double-float and then evaluated with
    binary32 `cos` and `sin` and one renormalising Newton step, so its target is a few
    $u$ by design;
  - the exponential's series.

  Errors are relative to the exact value, with an absolute floor at binary32's smallest
  normal where the value may vanish.
- **The experiment's witnesses**, which are evidence on their sampled domain and not
  proofs:
  - the $\varphi_k$ (orders 1, 3, 6 and 9) agree with binary64's within
    $2^{-36}\max(1,|\operatorname{Im}x|)\,|\varphi_k(x)|$, the bound growing with the
    imaginary part;
  - the propagator pair's complex error is within $8u\,|e^{\pm z}|$, for real parts in
    $[-20, 5]$ and phases up to $10^6$ rad.

**The double-float phase domain.** The bound covers every exponent argument the method
forms, including transport of Adams history. Let $h_{\max}$ be the longest prescribed
step for replay, or `dz_max` for an adaptive run. Replay is not assumed to obey
`dz_max`. For a single-step method the transport span is at most $h_{\max}$. For
Adams, a conservative fresh-run span is $(q_{\max}+1)h_{\max}$, where $q_{\max}$ is
the maximum configured order. A resumed run adds the retained history age
$z_{\mathrm{start}}-z_{\mathrm{oldest}}$; cap either span by the distance from the
oldest represented point to the run's end. Call the resulting bound $H$.
Then $\theta_{\max} = H\sup |\operatorname{Im} L|$. The supremum is bounded over
the represented frequencies and z intervals, including interior extrema; a maximum
of samples alone is insufficient. Admission checks the restored history as well as
the request. Two things limit accuracy at a large phase $\theta$:

- **The phase's own representation.** In any wide type, $\theta$ carries an absolute
  error of about $|\theta|$ times the wide type's unit roundoff, and $e^{i\theta}$
  inherits it. In double-float that reaches one binary32 ulp at
  $|\theta| \approx 2^{24}$ rad. In binary64 it does so at about $2^{29}$ rad.
- **The range reduction.** The experiment reduces by a two-word $2\pi$ and a binary32
  turn count, so its error grows with the turn count. An external probe found an error of
  $1.1\times10^{-6}$ at $\theta = 10^{10}$ rad, against an $8u$ budget of
  $4.8\times10^{-7}$ (Part III).

Production reduces in the manner of Cody and Waite [13], by a three-word $2\pi$. The turn
count is an exact integer, and its products with each word are formed exactly by
2MultFMA. The reduction's error is then negligible beside the phase's own up to
$\theta_{\mathrm{df}} = 2^{24}\cdot 2\pi \approx 1.05\times10^8$ rad. Up to there the
propagator pair's error, of the form $c_1 u + c_2 |\theta| u^2$, is expected to stay
within $8u$. Phase 3 derives and records $c_1$ and $c_2$, and lowers
$\theta_{\mathrm{df}}$ if they require it. A run on the double-float wide type whose
$\theta_{\max}$ exceeds $\theta_{\mathrm{df}}$ is refused by name at admission. Off
Metal, the refusal names `wide="binary64"`.

Exponents are assembled where cancellation is least harmful. A separable or frame-shifted
exponent is assembled in binary64 on the host and split once, wherever it is
constant. Where it must be assembled on the device, it is assembled with the accurate
addition.

A binary64 run never uses double-float. A binary32 run's wide type is an execution option
recorded in the run's metadata. Within the domain above, it changes only the rounding,
within SEP 33's bounds; beyond the domain, the run is refused rather than degraded.

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
  (with its dependency `juliapkg`). It pins `juliacall` exactly, to the release whose
  PythonCall.jl the committed manifests hold, because JuliaCall's
  existing-environment mode requires the two to match [6]. The Julia sources ship
  inside the `solprop` wheel as package data (§6.1). Importing solprop never imports
  `juliacall` or starts Julia.
- **Before setup, `backend="julia"` refuses by name.** The `BackendError` says what is
  missing (the extra, the setup, a device's environment, or a matching bridge) and gives
  the command to run. Nothing downloads implicitly. `registry.available()` lists `julia`
  only when `juliacall` is importable and a published setup matches this installation's
  identity (§6.5). It checks this from files, without starting Julia.
- **`solprop julia setup`** installs the pinned Julia release, unless a matching one is
  found (§6.5). It then instantiates the CPU environment and the environment of each
  device requested with `--device` (repeatable: `cuda`, `rocm`, `metal`), and
  precompiles them, including the engine's compile workload. A device can be added by
  a later setup. `solprop.backends.julia.setup()` and `status()` are the same
  operations as Python functions, for scripts and notebooks; the commands are thin
  wrappers over them (Q4). `setup()` returns the published setup identity and
  `status()` lists the identities available for selection.
  - The packages go to a private depot below the calibration store directory
    (`solprop.calibration.store_directory`, so `SOLPROP_CACHE_DIR` moves it).
    `SOLPROP_JULIA_DEPOT` names another.
  - The setup can run on an HPC login node and be used offline on compute nodes that
    share the file system.
  - `--cpu-target` sets Julia's CPU target for a heterogeneous cluster's precompiled
    code.
  - `--cuda-local` makes CUDA.jl use the system toolkit instead of its own, which is
    needed beside a native CUDA build in the same process (§6.5).
  - `--cuda-runtime` names the CUDA runtime release to fetch. It defaults to the newest
    that the pinned CUDA.jl supports. Setup can then fetch the runtime on a node with no
    GPU or driver, which CUDA.jl otherwise cannot choose [14].
  - **Setup is complete or absent.** It fetches every artifact an environment can need,
    lazy ones included, and publishes their platform- and preference-specific inventory
    after verification. Filesystem-only checks verify that inventory before importing
    JuliaCall and before loading another device extension. A missing artifact is a
    setup refusal before a loader can download it. Julia's package manager also runs
    offline, but that setting alone does not prevent lazy-artifact downloads.
  - Setup holds a lock on the depot. It builds each environment in a temporary
    directory and publishes it by one atomic rename, so an interrupted or concurrent
    setup never leaves a partial environment that availability would accept.
  - Setup is idempotent, prints its time and the depot's size, and removes nothing it did
    not create.
  - For `--device rocm` it checks that ROCm's libraries are discoverable, and says so
    when they are not. On an HPC system the check runs again on the compute node when a
    run starts.
- **`solprop julia status`** reports, for each published setup:
  - its identity (§6.5): the engine digest, the Julia patch release, the bridge pair, the
    preferences and the CPU target;
  - the engine's and the device packages' versions;
  - the depot's location and size;
  - the CPU target and CPU name used for precompilation; Julia validates cache
    compatibility and recompiles when needed at load;
  - for a GPU, whether the device prerequisites are met here.

#### 5.2 Selecting the backend

```text
from solprop import Simulation, SimulationSettings

sim = Simulation(..., settings=SimulationSettings(backend="julia"))
sim_gpu = Simulation(..., settings=SimulationSettings(backend="julia", device="cuda"))
sim_single = Simulation(..., settings=SimulationSettings(backend="julia", precision="single"))
```

- **The device** is a per-run setting, a new `SimulationSettings.device` (Q2): `None`
  (the default), `"cpu"`, `"cuda"`, `"rocm"` or `"metal"`.
  - **On `julia`,** `None` means the CPU. A GPU package loads only when a run first asks
    for its device, and SolpropJulia's extension for it then activates by itself
    (§6.5). A CPU run never loads a GPU package. Runs on different devices may share a
    process; the first run on a device pays its package's loading and kernel
    compilation. A device whose environment is not set up refuses by name, giving the
    setup command.
  - **On `native`,** the device is the build's. `None` accepts it, and any other value
    must match it (`"cuda"` a CUDA build, `"rocm"` a HIP build, `"cpu"` a CPU build), or
    the run refuses before allocating anything. `numpy_ref` accepts `None` and `"cpu"`.
  - `SOLPROP_REQUIRE_DEVICE` stays as a job's guard. It gains `metal` and applies to
    `julia` as it does to `native`: `cuda` accepts either backend's CUDA, `hip` native's
    HIP or Julia's ROCm, and `any` every GPU.
- **Precision.** `precision="single"` is admitted: `julia` joins
  `SINGLE_PRECISION_BACKENDS`. Metal is binary32 only, and binary64 refuses there by
  name. The wide type defaults to double-float on Metal and binary64 elsewhere, and is
  chosen by `ExecutionOptions.backend_options["wide"]` (`"binary64"` or `"df32"`).
- **Execution options.** A new `SimulationSettings.backend_options` is an owned mapping,
  empty by default, forwarded to `ExecutionOptions.backend_options` at preparation.
  Julia accepts `wide` and `setup_id`, both strings; unsupported keys or values refuse
  before runtime startup. Other backends refuse options they do not support. The
  mapping and `device` travel with settings through runners, recipes and spawned scans.
  They may change after translation, are recorded as execution metadata, and stay
  outside the scientific configuration and resume hashes. This adds no separate
  `run()`, runner or scan keyword and no second node-budget setting.
- **Threads.** The runtime starts with the process's thread budget (ADR-052; SEP 46's
  calibration). A run's width is at most that budget, and Julia's tasks are its only
  pool. Host BLAS is held to one thread while an engine lives, as native does.
- **FFT plans.** Julia keeps the FFTW plans it measures, by transform, in the
  calibration store, as native does (ADR-076).
- **Metadata.** Every run records `julia` as its backend, and:
  - the setup's identity: the engine digest, the Julia patch release, the bridge pair
    and the manifests' digests;
  - the device, its package versions and its effective toolkit (CUDA.jl's runtime or
    the local toolkit, the ROCm release);
  - the preferences in effect, the wide type, the thread count and the CPU target.

  It records no schema change (§9).

#### 5.3 What it runs at merge

| Capability | CPU | CUDA, ROCm | Metal |
|---|---|---|---|
| Modal route, mode-averaged and multimode, field and envelope grids | yes | yes | binary32 |
| Radial, Cartesian and line free space | yes | yes | binary32 |
| Kerr, field and envelope plasma (scalar, several channels), scalar Raman (several components, constant or varying along a gradient) | yes | yes | binary32 |
| Tapers and gradients on the mode-averaged and single-mode routes; on a multimode route only an uncoupled taper (`taper_coupling=False`) | yes | yes | binary32 |
| Temporal absorbers and collars, including the split collar (SEP 21); the transverse collar (SEP 40) | yes | yes | binary32 |
| `Multistep` (exponential formulation), the embedded Runge–Kutta tableaux adaptive and replayed, RK4IP, the low-storage replay | yes | yes | binary32 |
| Session services: planes, the periodic-capable statistics, the temporal monitor, the transverse edge, dense output, snapshots and resume (below), stop tokens | yes | yes | binary32 |
| The input noise model (SEP 37's default in binary64) | binary64 only | binary64 only | refused |
| Fixed-schedule groups, as member sessions | yes | yes | binary32 |
| `precision="single"` | yes, binary64 or double-float wide | yes, either | yes, double-float wide |

**Resume.** It follows SEP 47 §5.5 exactly, with no promise beyond native's:

- **Field restart.** Every admitted method restarts from a saved field.
- **Adams history.** It continues through `MarchSnapshot`'s portable history, across
  backends of the same precision:
  - in binary64, between `julia`, `native` and `numpy_ref`;
  - in binary32, between `julia` and `numpy_ref`, and across wide types.
- **No continuation.** Runge–Kutta and the low-storage replay restart from the field.
  Their history reset is recorded in the restart record and warned as SEP 47 specifies;
  full continuation is deferred to #596, as for native.
- **Refused.** A resume in the other precision is refused, as everywhere.

§7.1 gives each case's comparison class.

Refused by name at merge, each naming `native` or `numpy_ref`:

- the general modal route, the planar kind and coupled flows, which include a multimode
  taper's default coupling (`taper_coupling=True`, the
  [taper guide](../../guide/waveguides/taper-coupling.md));
- TDSE responses;
- vector polarisation and tensor Raman;
- the dispersive electronic response;
- the modified noise model;
- every noise model in binary32, including input noise on Metal; the existing
  single-precision default remains no noise;
- the interaction Adams formulation and the carrier frame;
- operators that took the z table;
- the far-field statistics (SEP 23);
- comparison methods and custom controllers, which are reference-only under SEP 47;
- binary64 on Metal;
- a double-float run whose largest phase exceeds the qualified domain (§4).

Each later addition is proposed for this backend (Tier 3), or arrives with a feature's
SEP when that SEP says so.

#### 5.4 Process model

- **Starting the runtime.** The runtime starts at the first `julia` preparation in a
  process, never on import. Its load path holds the CPU environment and every device
  environment that setup installed, so a device's package can load later in the same
  process. It refuses if `juliacall` was already initialised by something else, which
  would hold another environment.
  Initialization is serialized and must first occur on Python's main thread; an
  initial preparation on another thread refuses before importing JuliaCall. The first
  preparation fixes the runtime identity and installed environment stack. Subsequent
  preparations with an incompatible identity or a newly installed device environment
  require a fresh process, rather than changing the running environment.
- **Loading a device.** The first run on a device loads its GPU package, which stays
  loaded for the life of the process. Loading it can invalidate some compiled CPU code,
  which then recompiles once; phase 4 measures the cost (§7.5).
- **Workers.** Scans, the service client and the GUI's jobs already use spawned worker
  processes. Each spawned process starts its own runtime and pays its start-up once.
  `fork` after the runtime has started is unsupported, and refused where solprop
  controls it.
- **Stopping.** A run's stop token has a lifetime-owned atomic byte shared by both
  sides. The Python session retains its Julia owner and attaches a precompiled,
  non-blocking atomic setter to `StopToken`; it detaches before releasing the owner.
  Julia reads the flag atomically, without the GIL, between attempts (SEP 47 §5.5),
  and any Python thread may set it. Julia never calls back into Python while advancing.
- **Signals.** Threaded Julia needs JuliaCall's signal handling on, for the
  safepoints of its garbage collector [6]. The maintainer accepts its process-wide
  SIGTERM limitation for this optional backend; [#610] tracks revisiting it with a
  stable Julia 1.14 and compatible device packages.
  - Before startup, the bridge saves the application's actual SIGINT handler,
    including an ignored signal, and restores it afterwards. It does not unblock
    SIGINT in Julia's installed thread mask: on macOS that duplicates delivery through
    Julia's forwarding path. Phase 1 qualifies this behavior on both CPU platforms.
  - An advance runs with the GIL released. SIGINT is recorded by Python's C handler,
    and solprop's handler (`solprop._interrupts`) runs at the advance's bounded return,
    a safe boundary, exactly as SEP 47 retains for Python signals.
  - Julia's `InterruptException` is never used.
  - Once Julia starts, SIGTERM may terminate the entire process, even with an ignored
    or custom Python handler and even while no Julia session is advancing. It need not
    service pending output or write a final checkpoint. Closing a session does not
    remove this lifetime limitation. Graceful cancellation uses the stop token or
    qualified Python Ctrl-C handling; callers must not rely on SIGTERM for it.
  - Phase 1 must show by probes that threaded garbage collection stays safe, that
    Python Ctrl-C handling runs at a bounded return, and that a stop set from another
    thread is honoured, before any session code builds on it. Separate subprocesses
    record SIGTERM's behavior. Disabling Julia's signal handling is not a workaround.
    Failure of these remaining requirements still needs a maintainer decision.
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
  - `julia/environments/`, the committed environments setup instantiates: one joint
    resolution, and the CPU and per-device environments derived from it (§6.5);
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
It plugs in through the generic `Backend.prepare(...).start(...)` entry point after
SEP 47's descriptor/request split merges. Method and statistics policy come only from
`RunRequest`, never duplicated fields on `PropagationProblem`. The final merged
signature is the source of truth; neither a Julia special case in the transitional
dispatcher nor any dependency on `StepDriver` is introduced.

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

There are two ways to deliver option A, a run-time Julia through `juliacall`. The
maintainer chose A2 (Q1).

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

**A2, solprop-managed (chosen).**

- **Julia.** By default setup downloads the pinned patch release, from the official
  release and checked against its recorded SHA-256, as the experiment's bootstrap does.
  It accepts an installed Julia of the same minor version, from a `juliaup` channel or
  the `PATH` (Q7). The patch release actually used is part of the setup's identity
  (below); CI qualifies the pinned one, and `status` marks any other as untested.
- **Packages.** It instantiates the committed manifests (below) into the private depot.
- **At run time** solprop points `juliacall` at that Julia and environment through
  `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT`, which bypasses juliapkg's
  resolution. Before starting the runtime, it checks that the installed `juliacall` is
  the release paired with the manifest's PythonCall.jl, and refuses by name otherwise
  (§5.1). The check makes several virtual environments that share one setup safe: one
  with a different `juliacall` is refused, not started.
- **Why A2:**
  - It gives refuse-until-setup naturally.
  - It runs exactly the package versions the tests ran, as native's wheels do.
  - One depot serves every virtual environment of a user. Each solprop version has its
    own environment directories in it, and they share package files.
  - It cannot be disturbed by other `juliacall` packages.
- **The cost:**
  - The manifests to keep current (a scheduled update PR).
  - The table of Julia releases and digests that setup downloads.

**The environments (Q2).** One joint resolution, a project whose dependencies are the
engine and all three GPU packages, fixes a single version of every package, the shared
GPU dependencies included. The CPU environment and each device environment are
committed as subsets of that resolution, each closed under its dependencies, so any
combination of them agrees on every shared package. At run time the CPU environment
heads the load path and the set-up device environments are stacked after it. Loading a
GPU package then finds it, with versions consistent with what is already loaded, and
SolpropJulia's extension for it activates. Setup instantiates and precompiles only the
environments asked for, so a CPU-only user downloads no GPU package. CUDA.jl's
runtime selection (`--cuda-runtime`) and `--cuda-local` are preferences of the CUDA
environment.

**The device prerequisites** differ by vendor (§2):

- **CUDA** needs a driver. Setup fetches the runtime it names, so it can be prepared on
  a node with no GPU. A first GPU run with networking disabled is part of the CUDA
  qualification (§7.4).
- **ROCm** needs a system ROCm installation, which AMDGPU.jl discovers. Setup and
  `status` check that its libraries are found, and a run on a node without them is
  refused by name with what was not found. A driver-only AMD route is not claimed.
- **Metal** needs macOS on Apple silicon, and nothing else.

**What a setup owns, and its identity.** A setup must not depend on the virtual
environment that created it, and must never be mistaken for another.

- **The engine's sources.** Setup copies the Julia sources it is given, from the wheel's
  package data or an editable checkout, into the depot. They live in a directory named
  by their digest, and the environments refer to that copy. Removing the originating
  virtual environment therefore breaks nothing, and two trees with the same solprop
  version but different sources get different engines.
- **The identity.** A published setup records, in one file written last:
  - the engine digest, over the Julia sources and the adapter's bridge contract;
  - the manifests' digests;
  - the exact Julia patch release;
  - the `juliacall` and PythonCall.jl pair;
  - the machine-dependent preferences (`--cuda-local`, `--cuda-runtime`) and the CPU
    target;
  - the platform.
- **Matching.** Availability matches an installation to a published identity:
  - **A wheel's** engine digest is computed at build time and shipped beside the
    package data.
  - **An editable checkout's** is computed from its files when the runtime starts.
    That is a few hundred kilobytes to hash, and it restores the experiment's
    source-digest check where sources can change. A mismatch is refused, naming the
    command that sets up the changed sources.
- **Preferences.** A second setup with different preferences publishes a second
  identity beside the first, rather than changing the first in place. A run selects
  `ExecutionOptions.backend_options["setup_id"]`, the exact identity returned by
  setup and listed by status. Without that option, it uses the unique compatible
  identity; ambiguity refuses with the available IDs and how to select one. It never
  chooses the most recent setup implicitly. The run's metadata records the identity
  (§5.2), fixed for the process as §5.4 specifies.
- **Tests.** Phase 1 covers four cases:
  - a removed originating virtual environment;
  - two editable trees that differ;
  - an interrupted setup;
  - a second setup with different preferences.

**Julia version.** Julia 1.13 is the minimum (the experiment's version; FFTL's compiled
library misbehaved on 1.12.6, #590). Start with Julia 1.13.3 and matching JuliaCall and
PythonCall 0.9.37. The patch release is pinned, and raised by a PR that reruns the
Julia lane. A minor-version bump re-resolves the manifests and qualifies device
package compatibility; the signal follow-up [#610] is not a promise that an upgrade
alone fixes the limitation.

**Platforms (Q5).** Linux x86-64 and macOS arm64 are qualified at merge. Setup also
installs on Linux aarch64 and Windows x64, which the manual lists as unqualified.
Windows arm64, which Julia supports only at its lowest tier, is refused by name. Metal
needs macOS on Apple silicon; CUDA and ROCm are qualified on Linux only.

**Coexistence with native in one process** is supported. The experiment's A/B harness
ran both arms in one interpreter, on Linux and macOS.

- **On a CPU,** Julia's FFTW and OpenBLAS come from its own artifacts. Native's wheel
  bundles its own, and on Linux its OpenBLAS symbols carry a `solprop_` prefix. Phase 1
  repeats the check on the distribution wheel.
- **On CUDA,** CUDA.jl's runtime libraries clash with a native build linked to a system
  CUDA (an undefined cuBLASLt symbol on an H200). `--cuda-local` makes CUDA.jl use the
  system toolkit.

**Licences.** Julia and the engine's packages are MIT or BSD licensed. FFTW is GPL,
compatible with solprop's GPL-3.0-or-later. The CUDA runtime that CUDA.jl downloads is
fetched on the user's machine under NVIDIA's terms, and ROCm is the user's own
installation; solprop redistributes neither. Every file in `julia/` carries REUSE headers.

**Later, out of scope.**

- A system image, deferred by the maintainer, would remove most of the first-run
  compilation. GPU kernels would still compile at run time: GPUCompiler keys its cache on
  the device and keeps none on disk.
- A trimmed shared library (JuliaC, `--trim`) could ship the CPU engine in a wheel with
  no Julia install. It needs trim-safe code throughout and has no GPU path.

The C-ABI-friendly boundary of §6.2 keeps both possible.

#### 6.6 Tests and CI

**Julia tests**, in two groups with separate dependency environments:

- **`Core`**, through `Pkg.test` in the prepared CPU environment:
  - the shared helpers (Chebyshev rows, Adams basis) against the plain Python forms,
    within operation-derived bounds, as `tests/test_helpers.py` does for the native
    records. Bitwise agreement is recorded as information, not asserted. Exact
    comparison is kept for structural results: refusals, the refused position and NaN
    propagation;
  - the double-float primitives against their bounds;
  - every FFT plan form the engine uses, on each array type present;
  - a warmed step that allocates nothing on the CPU (`@allocated`).
- **`Quality`**, through the same test entry point in the pinned quality environment:
  `Aqua.test_all` and `JET.test_package` (error analysis), both blocking. Quality tools
  are absent from the runtime dependency closure; JET analysis must actually run,
  rather than silently becoming unavailable on an unsupported Julia version.

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

- runs setup with the depot cached by manifest digest, the Runic check, both Julia
  test groups, and the Python Julia modules;
- runs on every pull request whose diff is not documentation alone (Q3). A macOS arm64
  job runs weekly and on changes to `julia/`;
- is independent of the coverage lane, which omits `backends/julia/` as a recorded
  omission (as `mlx` was).

GPU tests (`pytest -m gpu`, with the test run's device named by an environment
variable) run by hand, once per vendor before merge and at each qualification pass, as
native's device builds do.

#### 6.7 Rules for the Julia code

`julia/AGENTS.md` is short. It adopts the parts of solprop's rules that carry over and
the Julia ecosystem's tools, and nothing more.

- **Formatting.** Runic, with no configuration, checked in CI. Lines of at most 92
  characters are an author and reviewer rule: Runic does not enforce line length.
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

- **Python:** `juliacall`, in the `julia` extra only, pinned exactly to the release
  paired with the manifests' PythonCall.jl (§6.5).
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

Each admitted capability is compared with `numpy_ref` on a short window: about ten steps,
or a fibre short enough to finish in seconds. Each comparison belongs to one of §4's
classes, and its tolerance comes from that class.

- **Class M, a matched discrete history:** the same prescribed schedule with a method
  that makes no discrete choice on it (`Multistep(variable_order=False)`, the Runge–Kutta
  tableaux, RK4IP and the low-storage replay), with matched startup, restart history and
  save-plane handling (§4). Fields and every
  statistic agree within SEP 25's rounding budget: the operation counts, with the
  conditioning and propagated sensitivity of what is compared (SEP 25 §§4.3–4.4 and
  [testing.md](../testing.md)). Each budget is derived in a comment beside its test. The
  experiment's windows agreed to $10^{-13}$ or better; that is a record, not a bound.
- **Class A, independent adaptive runs and variable-order replay:** `Multistep` and
  the adaptive Runge–Kutta tableaux, each run as in production, and `Multistep` replayed
  on a recorded schedule with its orders chosen afresh.
  - Accepted steps and orders may differ from the reference's.
  - Each run is compared with a converged reference: a `numpy_ref` run at a tolerance at
    least a hundred times tighter, or a stored witness computed that way. The envelope
    is set by the run's own `rtol` and recorded per case.
  - Each backend's attempt records count its own work exactly, and they are checked for
    internal consistency, not for equality with the other backend's.
- **Class C, control identity:** the controller, the monitor, the transverse edge's
  judgement and the landing. Recorded readings are replayed into the Julia session's
  decision functions and into the reference's. Their decisions and records must agree
  bit for bit (SEP 47 §4.3).
- **Groups and repeat runs.**
  - Members of a fixed-schedule group are compared with their solo runs on `julia`.
  - Repeat runs are compared on the same device, thread count and setup.
  - Each is class M, within SEP 25's repeatability bound, when its method makes no
    discrete choice on the schedule. It is class A when the method chooses orders, as
    variable-order `Multistep` does.
  - Bitwise equality is never required.
- **Resume** (§5.3):
  - an Adams history resume is class M only with a fixed-order method on the
    prescribed schedule, its history matched;
  - a variable-order or adaptive resume, and every field restart, are class A against
    the uninterrupted run (SEP 25 §4.4);
  - a cross-backend history resume (between `julia` and `native` or `numpy_ref`) is
    class A against the uninterrupted reference run;
  - the restart record's retained, restored and reset items are checked exactly;
  - a multimode taper is tested both ways: refused under its default coupling and
    admitted with `taper_coupling=False`.

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
never leaves a half-committed state reported as valid. The suite runs on each GPU in
§7.4's passes, not only on the CPU.

**Signals.** Phase 1's probes become tests (§5.4):

- threaded garbage collection under load;
- the application's actual SIGINT handler, including ignored signals, is restored;
  Python Ctrl-C handling during an advance runs at its next bounded return;
- separate subprocesses record SIGTERM with default, custom and ignored Python
  handlers, both during GIL-free work and while idle after initialization; process
  termination without final output is an accepted limitation, not graceful cancellation;
- a stop token set from another Python thread is honoured;
- first initialization on a worker thread refuses before importing JuliaCall,
  concurrent initialization is serialized, and subsequent worker-thread calls are safe;
- an already-set stop token, concurrent cancellation and close/detach retain the
  atomic flag until no caller can access it;
- cooperative stops and returned failures never report a half-committed state as valid;
  SIGTERM termination has no final-output or checkpoint guarantee.

#### 7.3 Binary32

On the CPU, with both wide types, and on each GPU:

- **Classes R and P** (SEP 33 §7.2) against `numpy_ref`'s single-precision path, within
  the binary32 bounds of SEP 33 §7.1.
- **Class T, survival, on short windows.**
  - SEP 33 §7.3's frozen metrics and envelopes are applied to windows of a few
    millimetres, or about ten steps. The cases are the soliton, the dispersive wave, the
    plasma blue shift, the Raman case, the radial case and the Cartesian case.
  - The binary64 references are stored witnesses on the same grids, each computed once
    at $\mathrm{rtol}=10^{-10}$ and checked against $10^{-9}$ as SEP 33 requires, and
    regenerated by a recorded script.
  - The full-length survival study is not repeated for merge (§7.6). It may be run later
    by hand as a worked example.
- **The propagator-drift record** for each wide type, with its stated constant and step
  count.
- **The double-float bounds** (§4), each in its own test:
  - each primitive (TwoSum, Fast2Sum, 2MultFMA, the analysed addition, multiplication
    and division) against `BigFloat`, within its transcribed bound from [3] and [12],
    including near-cancelling sums and operands at the bottom of their stated range;
  - an exactness probe of TwoSum and 2MultFMA on each device's compiler, which shows
    that nothing was reassociated, contracted or flushed;
  - the $\varphi_k$ against `BigFloat` over the orders and the argument domain the Adams
    march admits, including the branch boundary between series and recurrence and large
    imaginary parts, within the derived relative bound and its absolute floor;
  - the propagator pair over real parts in the admitted range and phases up to
    $\theta_{\mathrm{df}}$, within $c_1u + c_2|\theta|u^2$;
  - exponents assembled with cancellation, against binary64 assembly;
  - the admission refusal just above $\theta_{\mathrm{df}}$;
  - phase admission for replayed steps longer than `dz_max`, an interior extremum of
    a varying expansion, and a resumed, incompletely filled Adams history retaining
    older samples from larger steps.
- **Noise refusals.** Both noise models refuse in every binary32 configuration,
  including Metal; no single-precision noise default changes.
- **The envelope-plasma quadrature** converges within its frozen refinement on Metal's
  and the CPU's functions alike.

#### 7.4 Devices

Before merge, one qualification pass per vendor:

- **NVIDIA:** an A40 or an H100, CUDA.
- **AMD:** an Instinct MI300X or MI325X, ROCm, on a system ROCm installation.
- **Apple:** an M-series GPU, Metal.

**Each pass runs:**

- the Julia Python suite with `-m gpu`: parity, the conformance suite and binary32;
- repeat runs and group members, which must agree within SEP 25's bounds.

**Each pass records:**

- the GPU package versions;
- the driver, and the toolkit or ROCm release.

**The CUDA pass also:**

- prepares its setup on a node with no GPU;
- makes its first GPU run with networking disabled, which must download nothing;
- checks that a missing artifact is refused by name.
  The missing-artifact test also asserts zero attempted network requests, before
  runtime startup or device-extension loading.

No sanitizer lane is required at Tier 3.

#### 7.5 Performance and start-up records

These are rows in the ledger of [benchmarks.md](../benchmarks.md). Some must exist before
merge (§7.6), but none is a gate: no ratio has to reach a threshold.

- **Mandatory windows.** `python -m benchmarks ab` with `julia` (and `julia32`) arms
  against `native`, on the standard cases' windows:
  - at 1 thread and the default budget, on one CPU of each qualified platform: an Apple
    M5 Max and an AMD EPYC 7763 or Intel Xeon Gold 6240;
  - on each GPU of §7.4, in binary64 and binary32. Binary64 on Metal is not
    applicable, so Metal records binary32 against native on the CPU.
- **Mandatory start-up records:**
  - setup's time and the depot's size, for the CPU environment and each device's;
  - a process's first-run latency: the runtime start, package loading and compilation,
    on a CPU and on each GPU;
  - the cost of loading a device package into a process that has already run on the
    CPU, including any recompilation it causes;
  - a spawned scan worker's overhead.
- **Not part of this SEP's qualification:** whole runs. Like the release tracking set
  (`python -m benchmarks track`), they are run separately by hand, on a quiet machine,
  never as a change's qualification (AGENTS.md). Further machines are optional
  windows.
- **Regressions** against the experiment's last rows are explained in the phase report.

#### 7.6 Merge criteria, and evidence for later

**The backend is mergeable when all of these hold:**

- every capability of §5.3 passes §§7.1–7.4 in its class;
- everything else refuses by name;
- the Julia lane is green;
- `pip install solprop`, `import solprop` and native runs are unchanged (§6.6's
  subprocess test, and a wheel-metadata check that the base install requires no Julia
  package);
- §7.5's mandatory records exist;
- the manual page exists.

**Not required for merge:**

- the full-length binary32 survival study;
- whole runs;
- any performance threshold.

**Kept current for a later tier decision:**

- a coverage table against native's [backends.md](../backends.md) §9;
- the performance ratios per machine;
- the first-run costs.

### 8. Documentation

- **User guide,** a new page `docs/guide/performance/julia.md`:
  - installing and setting up, offline and HPC use, devices and precision;
  - what it runs and refuses;
  - the first-run cost;
  - the process-wide SIGTERM limitation after first initialization, including custom
    and ignored handlers, the lack of a final-checkpoint guarantee, and using a stop
    token or qualified Ctrl-C handling for graceful cancellation;
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
- `SimulationSettings.device`, which native and `numpy_ref` check against what they
  run (§5.2);
- `SimulationSettings.backend_options`, forwarded to the existing execution options
  for explicit setup and wide-type selection, outside scientific identity (§5.2);
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
pull request (SEP 0). The branch starts from main, without stacking on any open pull
request. The engine's numerical code is moved from the experiment's branch and reshaped
to §6.7. Its adapter is rewritten.

What each phase needs from SEP 47:

- **Phase 1** needs nothing, and starts as soon as this SEP is accepted.
- **Phase 2** needs SEP 47's contract: `RunSession`, the events, `RunRequest`, the route
  union and the conformance suite. They merged with #605. Production adapter
  integration also requires the completed descriptor/request split and generic
  preparation entry point from #607; inspect the merged code, not a prospective report.
- **Phase 2 integrates once #607's preparation changes have merged.** #607 is SEP 47's native session,
  the source from which the session's control logic is transcribed: the monitor, the
  transverse edge and the statistics plans. Until it merges, its branch can be read but
  not built on.
- **Phase 4** waits for #607 in any case, because `SimulationSettings.device` and the
  registry change files that SEP 47 is still editing.
- **SEP 47's later phases** may still change the contract. While `julia` is at Tier 3,
  each such change keeps the Julia lane green or refuses the affected case by name
  (§6.8). That cost is smallest while the Julia session is small.

1. **Package, distribution and rules.**
   - **Deliverables:**
     - the `julia/` package, the joint resolution with the environments derived from
       it, and `julia/AGENTS.md`;
     - Runic, Aqua and JET;
     - `solprop julia setup` and `status`, and the private depot, with the setup's
       identity, its locking and atomic publication, and the copied engine sources
       (§6.5);
     - the exact `juliacall` pin and its check before start-up;
     - complete artifact fetching, an offline package manager at run time, and the
       vendor prerequisite checks;
     - availability, refusal, and the runtime's start with the actual Python SIGINT
       handler restored and the documented process-wide SIGTERM limitation;
     - the wheel's package data and the extra;
     - the CI job.
   - **Verify:**
     - setup on Linux x86-64 and macOS arm64;
     - `pip install solprop` and `import solprop` unchanged;
     - refusal before setup, and on a mismatched bridge;
     - §6.5's four identity cases;
     - ambiguous and explicit setup selection; an incompatible setup or a device
       installed after runtime startup requires a fresh process;
     - concurrent and interrupted publication, and a missing artifact refused with
       zero network attempts before JuliaCall import or device-extension loading;
     - the signal and stop probes of §5.4, which gate everything after this phase;
     - a trivial engine call through the bridge.
2. **The CPU session in binary64.**
   - **Deliverables:**
     - preparation and lowering;
     - the Julia session with every service of §5.3 and groups;
     - the precompiled workload in place of the trace;
     - FFTW plans kept in the store;
     - the thread budget.
     - `SimulationSettings.backend_options`, forwarded through central preparation,
       runners, recipes and spawned workers, with effective execution metadata.
   - **Verify:** §§7.1–7.2 on the CPU, and the first CPU rows of §7.5.
     Also verify copied option mappings, pickle/spawn round trips, unchanged science
     and resume hashes, changed options after translation, and unsupported-option
     refusals.
3. **Binary32 on the CPU.**
   - **Deliverables:**
     - SEP 33's contract in both wide types;
     - the double-float primitives replaced by analysed algorithms, with their bounds
       transcribed after checking [3] and [12];
     - the three-word range reduction, and the admission refusal above
       $\theta_{\mathrm{df}}$;
     - the stored binary64 witnesses for the short survival windows;
     - the investigation of x86 binary32 speed.
   - **Verify:** §7.3 on the CPU.
4. **Devices.**
   - **Deliverables:**
     - the CUDA, ROCm and Metal extensions on the session, loaded on demand by a run's
       `device`;
     - `SimulationSettings.device`, and native's check of it against its build;
     - graphs and the other measured device choices.
   - **Verify:** §7.4 for each vendor, §7.3 on each GPU, and the device rows of §7.5.
     Extend phase 2's settings propagation and identity tests to `device`, including
     changed devices after translation and named refusals for device mismatches.
5. **Records and Final.**
   - **Deliverables:**
     - the manual and design pages;
     - the ADR;
     - the changelog fragment;
     - the remaining §7.5 rows.
   - **Final when:** §7.6's criteria hold. Open questions left go to issues.

### 12. Open questions

None remain open. The maintainer settled the drafted questions on 2026-11-10, and Part I
records each decision where it applies:

- **Q1. The environment's manager:** solprop-managed (A2), with committed manifests and
  a private depot shared across virtual environments (§6.5).
- **Q2. Device selection:** a per-run `SimulationSettings.device`. GPU packages load on
  demand through package extensions, from stacked environments derived from one joint
  resolution. Native checks the setting against its build (§§5.2, 5.4, 6.5).
- **Q3. CI trigger:** the Linux job on every pull request that is not documentation
  alone, and a macOS arm64 job weekly and on changes to `julia/` (§6.6).
- **Q4. Commands:** `solprop julia setup` and `status`, as thin wrappers over
  `solprop.backends.julia.setup()` and `status()` (§5.1).
- **Q5. Platforms:** Linux x86-64 and macOS arm64 qualified at merge. Linux aarch64 and
  Windows x64 install but are unqualified, and Windows arm64 is refused (§6.5).
- **Q6. Merge scope:** §5.3 as drafted, including snapshots and resume, fixed-schedule
  groups, dense output and the low-storage replay. Vector polarisation and tensor Raman
  stay out.
- **Q7. Julia version:** 1.13 minimum, with the exact patch pinned and raised by pull
  request. A minor-version bump re-resolves the manifests (§6.5).

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
11. AMDGPU.jl documentation: requirements and installation,
    <https://amdgpu.juliagpu.org/stable/>.
12. J.-M. Muller and L. Rideau, "Formalization of double-word arithmetic, and comments on
    'Tight and rigorous error bounds for basic building blocks of double-word
    arithmetic'," *ACM Transactions on Mathematical Software* **48**, 9 (2022).
    doi:10.1145/3484514
13. W. J. Cody and W. Waite, *Software Manual for the Elementary Functions*
    (Prentice-Hall, 1980).
14. CUDA.jl documentation: installation, including precompiling without CUDA,
    <https://cuda.juliagpu.org/stable/installation/overview/>.
15. solprop: SEP 25, SEP 33, SEP 43, SEP 46 and SEP 47; ADR-052, ADR-061, ADR-064,
    ADR-068 and ADR-076; PR #559 and its
    [status page](https://github.com/LupoLab/solprop/blob/b84d3ab81df863669e57aeca6c0ac3abaf6820ac/julia/STATUS.md);
    PRs #605 and #607; issues #333, #368, #556, #557, #560, #561, #588, #590 and #596.

## Part II — Implementation record

## Part III — History

- 2026-11-10: Drafted by Opus 5.5 for John Travers, after a feasibility analysis of
  PR #559. The maintainer chose an explicit setup over a first-use install, all three
  GPU vendors and binary32 at merge, and Runic, Aqua and JET as the blocking Julia tools.
  Tracking issue: [#608](https://github.com/LupoLab/solprop/issues/608).
- 2026-11-10: The maintainer settled Q1–Q7 (§12). Q2 changed the design: the device is
  a per-run setting, with GPU packages loaded on demand from stacked environments of
  one joint resolution, in place of a process-wide variable.
- 2026-11-10: AI review by GPT-6 (R1–R10, recorded below), with the automated review's
  matching finding on §7.1. The maintainer decided R3, R8 and R10. Opus 5.5 answered
  every point and revised Part I (§§2, 4, 5.1–5.4, 6.5, 6.10, 7, 11 and the
  references). The SEP stays Draft, In review.
- 2026-11-10: §§3.3 and 11 state what each phase needs from SEP 47. Phase 1 needs
  nothing, phase 2 needs #605 (merged), and phases 2 and 4 are better begun after #607.
  The branch starts from main rather than stacking.
- 2026-11-10: The automated review's findings on PR #609 were answered (Part III).
  §§4 and 7.1 place variable-order replay in class A, §6.6 compares the helpers within
  rounding bounds, and §7.5 takes whole runs out of the qualification.

### AI review — 2026-11-10

Reviewer: **GPT-6 (AI)**. Reviewed all of Part I at PR #609 head `742857093`,
against SEPs 25, 33 and 47, the current design and precision code, the session contract
on PR #607 at `83c333464`, and the experiment at the cited `b84d3ab8`.
This review changes only the review metadata and this record; Part I is unchanged.
The comments belong to the work tracked by [#608](https://github.com/LupoLab/solprop/issues/608).

The optional Tier 3 engine and the backend-owned session boundary are coherent choices.
The points below need responses before acceptance, particularly the numerical bounds,
the reproducible installation contract and Python signal handling. They do not reopen
the maintainer's Q1–Q7 decisions. **High** identifies a correctness or usability contract
that is presently contradicted or unsupported; **Medium** identifies a scope correction
or a question whose implementation and qualification need to be specified. No point
records a maintainer decision.

1. **R1 — High: distinguish adaptive accuracy from matched-history rounding
   (§§4, 7.1).** Section 4 promises agreement to rounding without limiting that promise
   to matched discrete histories; §7.1 then requires independent adaptive runs to take
   the same accepted steps and bounds every field and statistic by summation counts.
   [SEP 47 §4.3](sep-0047-run-ownership.md)
   explicitly permits different accepted counts when numerical readings differ.
   [SEP 25 §§4.3–4.4](archive/sep-0025-rounding-level-reproducibility.md)
   also requires conditioning and propagated sensitivity in rounding budgets, and a
   tighter or converged reference for independent adaptive runs and ordinary replay.

   **Requested change:** separate matched-method/history comparisons, independent
   adaptive accuracy envelopes and exact controller/monitor traces fed identical
   recorded readings. Give resume, groups and repeat runs the appropriate class rather
   than requiring equal adaptive work. Preserve exact accounting for the work each
   actually performs. This confirms and extends the
   [existing automated finding](https://github.com/LupoLab/solprop/pull/609#discussion_r4524812132),
   rather than treating it as a second independent discovery.

2. **R2 — High: state the actual double-float error bounds (§§4, 7.3).** The two
   definitions of the putative unit roundoff disagree:
   $(2^{-24})^2/2 = 2^{-49}$, not $2^{-48}$. More fundamentally, the precision of a
   two-word representation is not one error bound shared by its addition,
   multiplication, division and transcendental implementations. Name the algorithms
   and the applicable bounds and hypotheses from reference [3], including protection
   against reassociation, the required fused operation and underflow conditions.

   The experiment's cited test also does **not** assert an unqualified $2^{-36}$
   error for the $\varphi_k$. At `b84d3ab8`,
   [`test_julia_double_float_phi_is_binary64s`](https://github.com/LupoLab/solprop/blob/b84d3ab81df863669e57aeca6c0ac3abaf6820ac/tests/backends/test_julia.py#L843)
   uses $2^{-36}\max(1,|\operatorname{Im}x|)|\varphi_k(x)|+10^{-300}$.
   The adjacent propagator test uses $8u_{32}|e^{\pm z}|$, a relative complex-error
   budget, rather than componentwise ulps.

   **Requested change:** quote these witnesses accurately, define absolute versus
   relative errors near zeros, and derive the production bounds over the actual
   argument and order domain. A passing sample is evidence for that domain, not a
   proof for all admitted inputs.

3. **R3 — High: connect the wide-type domain to admission (§§4, 5.2, 6.4, 7.3).**
   The text says choosing `df32` cannot change the admitted problem, but only records
   the propagator bound through $10^6$ rad. The shared single-precision admission does
   not impose that phase limit; its exponent guard concerns the real part and overflow.
   The experiment's `reduce_phase` uses a two-word approximation to $2\pi$ and a
   binary32 turn count, so the error grows with the unreduced phase.

   **Probe run:** using only the cited `julia/src/dfloat.jl`, Julia 1.13.3 on macOS
   arm64, and a 256-bit `BigFloat` reference for $e^{i\theta}$, the forward pair's
   absolute error was $1.14\times10^{-6}$ at $\theta=10^{10}$ and
   $1.56\times10^{-3}$ at $10^{12}$; $8u_{32}=4.77\times10^{-7}$. This is outside
   the recorded qualification range, not a failure of its existing test.

   **Question for the maintainer:** must double-float support the entire binary64-wide
   domain? If so, require range reduction and conditioning bounds that establish it.
   Otherwise specify named refusal or a qualified alternative for out-of-domain
   arguments, especially on Metal. Include cancellation in the assembled exponent,
   large imaginary phases and the $\varphi_k$ branch boundaries in qualification.

4. **R4 — High: bind the Python bridge to the committed environment (§§5.1, 6.5,
   6.10).** A tested range of `juliacall` versions can select a different release from
   the PythonCall.jl pinned in the shared manifest. With the two override variables,
   juliapkg will not reconcile them. The upstream
   [existing-environment contract](https://juliapy.github.io/PythonCall.jl/stable/juliacall/#Using-existing-environments)
   requires matching versions; otherwise preparation can fail after successful setup.

   **Requested change:** pin matching pairs, check compatibility before startup, and
   test mismatches across virtual environments sharing one setup.

5. **R5 — High: make offline GPU setup establish artifact completeness (§§5.1,
   6.5, 11 phase 1).** Instantiating manifests and precompiling on a login node does
   not by itself establish that GPU runtime artifacts exist. Julia's
   [lazy artifacts](https://pkgdocs.julialang.org/v1/artifacts/#Artifact-types-and-properties)
   can download on first use. CUDA.jl specifically documents that
   [precompiling without a driver](https://cuda.juliagpu.org/stable/installation/overview/#Precompiling-CUDA.jl-without-CUDA)
   requires an explicit runtime selection to fetch the runtime/compiler artifacts.
   The proposed workflow otherwise risks its first network access or failure on the
   offline compute node.

   **Requested change:** define how setup selects the target toolkit and materialises
   every required artifact without a GPU, and when it records setup as complete. Test
   a first real GPU run with networking disabled after setup on a node without a GPU.
   Missing artifacts must produce the promised setup refusal, not a download attempt.

6. **R6 — Medium: correct the AMD installation prerequisite (§§2, 6.5, 8).** The
   general claim that Julia's GPU packages supply their runtime libraries and need
   only a driver is true of the documented CUDA route, but is not the documented
   AMDGPU route. The experiment pins AMDGPU 2.11.0; the upstream
   [requirements](https://amdgpu.juliagpu.org/stable/#Requirements) and
   [installation guidance](https://amdgpu.juliagpu.org/stable/install_tips)
   require ROCm and describe discovery of system libraries, including rocBLAS and
   rocFFT.

   **Requested change:** state vendor-specific prerequisites and check them in setup
   or status. If a different pinned AMDGPU release is intended to supply all required
   libraries, identify and demonstrate that route on a host with only the driver.
   An already provisioned ROCm image does not establish the driver-only claim.

7. **R7 — Medium: define what a shared setup owns and identifies (§§3.2, 5.1,
   5.2, 6.5).** The source-digest check is explicitly not carried over, availability
   checks only for this solprop version, and the environment's reference to the engine
   shipped in the Python wheel is unspecified. Two editable trees can share a version
   while their Julia sources or adapter differ. A shared environment pointing into one
   virtual environment can also break when that environment is removed. Matching only
   the Julia minor version contradicts the later promise of an exact patch pin.

   **Requested change:** define immutable setup identity and engine-source ownership,
   covering the engine/adapter contract, manifests, exact Julia patch and bridge pair.
   Clarify the treatment of machine-dependent preferences such as `--cuda-local` and
   the CPU target; a manifest digest alone does not identify them. Record effective
   toolkit/preferences in execution metadata. Specify concurrent setup locking and
   atomic publication so availability cannot accept a partial environment. Test a
   removed originating virtual environment, differing editable sources, an interrupted
   setup and a second setup with different preferences.

8. **R8 — High: resolve threaded Julia's signal policy before adopting the session
   contract (§§5.4, 6.2, 7.2, 11).**
   [JuliaCall's signal caveat](https://juliapy.github.io/PythonCall.jl/stable/juliacall/#Caveat:-Signal-handling)
   recommends `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` for threaded GC safety, but says
   this interferes with Python signal handling, including `KeyboardInterrupt`.
   The experiment sets that option. SEP 47 §5.5 instead relies on Python signals at
   bounded returns and introduces no process-wide native signal handler. Releasing
   the GIL does not settle this conflict.

   **Requested change:** prove threaded GC safety, Ctrl-C and thread-issued cancellation
   in phase 1. Specify token ownership, GIL-free polling and signal handlers. Preserve
   SEP 47's bounded returns; any signal-policy change needs a maintainer decision.
   Threaded bridge and signal probes were not run in this review.

9. **R9 — Medium: qualify the capability table and restart promises (§§5.3, 7.1–7.4).**
   “Tapers” and “multimode” appear admitted while coupled flows are refused. A varying
   multimode capillary normally carries the coupling by default
   ([taper guide](../../guide/waveguides/taper-coupling.md)); only its explicit
   `taper_coupling=False` alternative fits this backend. Similarly, “snapshots and
   resume” must preserve SEP 47's distinction between field restart and portable
   Adams history, including the existing RK continuation deferral, rather than imply
   full history continuation for every admitted method.

   **Requested change:** state these intersections and test the default coupled-taper
   refusal plus the explicitly uncoupled case. Name which same-precision cross-backend
   and cross-wide-type field/history resumes are admitted, which reset history, and
   their warnings and comparison classes. Confirm that GPU qualification includes the
   session conformance cases, not only kernel parity selected by `-m gpu`.

10. **R10 — Medium: reconcile the qualification workload and merge checklist
    (§§7.3–7.6, 11).** Section 7.5 says its records are not merge gates but §7.6 requires
    all of them, including two whole runs per machine. It also requests binary64 on
    every GPU even though Metal refuses it. Section 7.3 imports SEP 33's survival
    studies without saying whether their original full-length runs must be repeated;
    some took hours. The current `AGENTS.md` qualification rule is about ten steps or
    seconds, with complete release tracking a separately invoked activity.

    **Requested change:** identify short qualification windows and stored converged
    witnesses for the survival comparisons; distinguish them from any separately
    authorised full-run study. State which performance/start-up records are mandatory
    while leaving performance ratios non-blocking, and mark binary64 Metal timings
    inapplicable. This lets an implementer follow the merge checklist without guessing
    which workload rule wins.

**Evidence and limits.** Read the experiment's implementation, tests and benchmark
ledger; its reported performance was not re-measured. The only new numerical execution
was R3's isolated scalar propagator probe, with no propagation or GPU run. The JuliaCall,
Julia artifact and AMDGPU documentation linked above was checked on 2026-11-10. No
installation, multithreaded bridge, CUDA/ROCm/Metal or restart qualification was run.
Reference [3]'s full paper could not be retrieved; its algorithm-specific theorems
remain to be checked, rather than claimed verified here. All R1–R10 remain open for the
drafting agent's point-by-point response and the maintainer's decisions.

### Maintainer decisions on the AI review — 2026-11-10

John Travers decided the three points that asked for a decision, after Opus 5.5 put each
to him with its context and a recommendation:

- **R3, the double-float domain.**
  - Production uses a better range reduction: Cody–Waite with a three-word $2\pi$ and an
    exact turn count, with exact products.
  - It is qualified up to $\theta_{\mathrm{df}} \approx 2^{24}\cdot2\pi$.
  - A double-float run whose largest per-step phase exceeds that is refused by name at
    admission, and off Metal the refusal names `wide="binary64"`.
  - Covering the whole binary64 domain is not required.
- **R8, signals.**
  - Julia starts with JuliaCall's signal handling on. Python's SIGINT and SIGTERM
    handlers are then re-installed, so signals are delivered at an advance's bounded
    return.
  - Julia's `InterruptException` is not used.
  - Phase 1's probes are a gate. If they fail, any other signal policy comes back to the
    maintainer.
- **R10, the survival study.**
  - Class T runs in tests on short windows against stored, converged binary64
    witnesses. The full-length study is not required for merge.
  - The mandatory §7.5 records are the window ratios on each qualified platform and GPU
    vendor and the start-up records. They must exist, but none needs to reach a
    threshold.
  - Whole runs are optional, and binary64 on Metal is inapplicable.

The other points needed no decision. The earlier decisions Q1–Q7 stand.

### Response to review — 2026-11-10 — Opus 5.5

Each point was checked against the cited code and documents before answering. Part I is
revised; the review's own text above is unchanged. The SEP stays Draft, and its review
status stays In review: no human review is recorded yet.

- **R1, accepted.**
  - §4 now defines three comparison classes:
    - **M:** matched discrete histories, with SEP 25 §§4.3–4.4 rounding budgets that
      include conditioning and propagated sensitivity;
    - **A:** independent adaptive runs, against a converged reference within an
      `rtol`-set envelope, each backend counting its own work exactly;
    - **C:** control identity on recorded readings.
  - §7.1 assigns every comparison to a class, including groups, repeat runs and each
    kind of resume. The promise of equal accepted steps for adaptive runs is gone.
  - This also answers the automated review's finding on the same lines (PR #609, the
    comment on §7.1). It was the same point, and is not counted as a second one.
- **R2, accepted, and it found more.**
  - $u^2/2$ was wrong: with $u = 2^{-24}$ the double-float scale is $u^2 = 2^{-48}$, and
    §4 now says so without the stray factor. It also states that each operation has its
    own bound, not one shared precision.
  - §4 names the error-free transformations with their hypotheses: TwoSum; Fast2Sum
    with its ordering condition; 2MultFMA with no underflow or overflow.
  - It names the analysed algorithms production must use, with [12], the formal
    re-verification of [3], added. It states the hypotheses: rounding to nearest even,
    a correctly rounded FMA, no reassociation or contraction inside a primitive, and the
    underflow range, including a device's flushing.
  - Checking the experiment's `dfloat.jl` found two gaps, now recorded in §4:
    - its addition is the sloppy form, with no useful relative bound under
      cancellation;
    - its division of two double-floats is a long division that [3] does not analyse.
  - The witnesses are now quoted exactly:
    - the $\varphi_k$ bound $2^{-36}\max(1,|\operatorname{Im}x|)|\varphi_k(x)|$, which
      grows with the imaginary part;
    - the propagator's $8u|e^{\pm z}|$ relative complex error.

    They are labelled as evidence on a sampled domain. Errors are defined as relative,
    with an absolute floor at binary32's smallest normal.
  - The constants of [3] and [12] are not transcribed here. This response could not
    retrieve either paper's full text either, so phase 3 transcribes each bound after
    checking the primary source. §7.3 now tests each primitive against `BigFloat` and
    probes each device compiler for exactness.
- **R3, accepted, and decided as above.**
  - §4 derives the domain:
    - The per-step phase is bounded at admission by $\Delta z_{\max}\max|\operatorname{Im}L|$.
    - The phase's own representation limits accuracy in any wide type: one binary32 ulp
      at about $2^{24}$ rad in double-float, and $2^{29}$ in binary64. No reduction can
      remove that.
    - With the reduction decided above, the pair's error is $c_1u + c_2|\theta|u^2$,
      expected within $8u$ up to $\theta_{\mathrm{df}}$; phase 3 derives the constants
      and lowers $\theta_{\mathrm{df}}$ if they require it.
  - §§4, 5.3, 7.3 and phase 3 now carry:
    - the refusal above $\theta_{\mathrm{df}}$;
    - assembly of the exponent in binary64 on the host where it is constant, and with
      the accurate addition otherwise;
    - qualification cases for cancellation, large imaginary phases and the $\varphi_k$
      branch boundary.
  - The reviewer's probe values are cited in §4.
- **R4, accepted.**
  - §§5.1, 6.5 and 6.10 pin `juliacall` exactly, to the release paired with the
    manifests' PythonCall.jl.
  - The pair is part of the setup's identity, and is checked before the runtime starts.
    A mismatch is refused by name.
  - Several virtual environments sharing one setup are covered by that check, and by
    §6.5's tests.
- **R5, accepted.**
  - Setup gains `--cuda-runtime`, so it can fetch CUDA's runtime on a node with no GPU.
  - It fetches every artifact, lazy ones included, and verifies them before publishing.
  - Runs keep Julia's package manager offline, so a missing artifact is a setup
    refusal, never a download.
  - The CUDA pass of §7.4 prepares the setup on a node with no GPU, then makes its first
    GPU run with networking disabled.
- **R6, accepted.**
  - §2 now states each vendor's prerequisites: a driver for CUDA, nothing for Metal, and
    a system ROCm installation for AMD, which AMDGPU.jl discovers.
  - Setup and `status` check ROCm's libraries, and a run without them is refused by
    name. §7.4's AMD pass runs on a system ROCm installation.
  - No driver-only AMD route is claimed. The licence paragraph is corrected to match.
- **R7, accepted.** §6.5 now defines what a setup owns:
  - Its engine sources are copied into the depot, under their digest.
  - Its identity covers the engine digest over the sources and the bridge contract, the
    manifests, the exact Julia patch release, the bridge pair, the preferences, the CPU
    target and the platform.
  - A wheel's digest is computed at build time. An editable checkout's is computed when
    the runtime starts, which restores the source-digest check where sources can change.
  - A second setup with other preferences publishes a second identity.
  - Setup locks the depot and publishes atomically.
  - The run metadata records the identity and the effective toolkit.
  - The four tests the reviewer asked for are in phase 1.
  - On the Julia patch release, the maintainer's Q7 decision accepts an installed 1.13.x
    beside the pinned default. There is now no contradiction: the patch release actually
    used is part of the identity, CI qualifies the pinned one, and `status` marks any
    other as untested.
- **R8, accepted, and decided as above.**
  - §5.4 specifies the signal policy and the stop token's ownership: a one-byte shared
    flag, read without the GIL between attempts and settable from any Python thread.
  - SEP 47's bounded returns are kept for Python signals.
  - §7.2 turns the phase 1 probes into tests, and phase 1 makes them a gate.
  - No threaded bridge probe was run for this response either.
- **R9, accepted.**
  - §5.3 now admits tapers on the mode-averaged and single-mode routes, and on a
    multimode route only with `taper_coupling=False`. The default coupled multimode
    taper is refused as a coupled flow.
  - §7.1 tests both cases. The coupled default is confirmed in `driver/interface.py` and
    `driver/simulation.py`, where `taper_coupling` defaults to `True`.
  - §5.3's resume paragraph follows SEP 47 §5.5:
    - a field restart for every method;
    - Adams history across backends of the same precision, and across wide types;
    - for Runge–Kutta and the low-storage replay, a recorded reset, with #596 deferred;
    - a resume in the other precision refused.

    §7.1 gives each its comparison class.
  - §§7.2 and 7.4 now run the conformance suite in every GPU pass.
- **R10, accepted, and decided as above.**
  - §7.3's class T uses short windows and stored converged witnesses, regenerated by a
    recorded script.
  - §7.5 separates the mandatory records from the optional ones, and marks binary64 on
    Metal inapplicable.
  - §7.6 lists what is not required for merge.
  - The merge checklist no longer conflicts with AGENTS.md's qualification limits.

**Evidence.** This response read the experiment's `dfloat.jl` and its double-float tests
at `b84d3ab8`, solprop's taper defaults, and the review's cited documentation. It ran no
Julia, no GPU and no bridge probe.

### Response to the automated review — 2026-11-10 — Opus 5.5

The Codex connector left six findings on PR #609. Each is answered on its own thread
and summarised here; none needed a maintainer decision.

- **Adaptive accuracy against control identity** (§7.1, P1). The same point as R1,
  fixed in the response above: classes M, A and C.
- **Helpers compared within rounding bounds** (§6.6, P1). Accepted. The `Core` group
  compared the Chebyshev rows and the Adams basis bit for bit with recorded values.
  `tests/test_helpers.py::test_points_agree_with_the_plain_form` instead asserts an
  operation-derived bound and records bitwise agreement as information. §6.6 now does
  the same, and keeps exact comparison for structural results only.
- **Whole runs out of the qualification** (§7.5, P1). Already fixed by the R10 decision,
  where whole runs became optional and not required for merge. §7.5 now says outright
  that whole runs belong to separately invoked tracking, never to this change's
  qualification.
- **The JuliaCall and PythonCall pair** (§6.5, P1). The same point as R4, fixed above:
  an exact pin, part of the setup's identity, checked before start-up.
- **CUDA artifacts on a node with no GPU** (§5.1, P2). The same point as R5, fixed
  above: `--cuda-runtime`, every artifact fetched before publication, an offline
  package manager at run time, and a first offline GPU run in the CUDA pass.
- **Variable-order replay is class A** (§§4 and 7.1, P1). Accepted. SEP 25 §4.4 says
  that `FixedSchedule` stores step lengths only, and that a replay of `Multistep` still
  chooses orders. §4 now reserves class M for a computation matched in method, orders,
  startup, restart history and save-plane handling. §7.1 places variable-order replay,
  and groups, repeat runs and resumes whose method chooses orders, in class A.


### Implementation planning review and acceptance — 2026-11-10

John Travers reviewed the proposal and accepted it, delegating its final acceptance
revision and merge to GPT-6. GPT-6 then reviewed Part I with independent planning
reviews against the session contract, the experiment and current Julia tooling.
The maintainer approved the resulting implementation plan, including the changes
below. These revisions precede the freeze of Part I.

- **R11, binary32 noise, resolved by the maintainer.** The capability table promised
  input noise on Metal while the inherited single-precision admission refuses every
  noise model. The maintainer chose binary64-only input noise, rather than extending
  the reference oracle. Sections 5.3 and 7.3 now state that restriction; no default changes.
- **R12, double-float admission, accepted.** Fixed-schedule RK and Adams can use steps
  larger than `dz_max`, and Adams transports samples from its retained history. A
  resumed history can predate the current request's step limit. Sections 4 and 7.3
  bound every exponent span, include restored history age, and require a conservative
  bound between the samples of a varying representation. The new cases are explicit
  regression requirements; the numerical constants remain phase 3's source-checked work.
- **R13, execution choices, accepted.** Sections 5.2, 5.4 and 6.5 specify the owned
  `backend_options` mapping, explicit `setup_id`, unique-selection rule and process
  identity. These execution choices travel to spawned workers without entering the
  scientific identity. They add no separate per-call interface or node-budget control.
- **R14, offline artifacts, accepted.** Julia 1.13.3's artifact loader can attempt a
  missing lazy artifact even with package-manager offline mode set. Section 5.1 now
  requires a published inventory and filesystem checks before loading. Section 7.4
  checks zero attempted network requests as well as the first disconnected GPU run.
- **R15, initialization and stop ownership, accepted.** Python signal-handler
  installation requires its main thread. Sections 5.4 and 7.2 require first
  initialization there, serialized startup, preservation of the application's actual
  handlers, and an atomic stop byte retained through concurrent use and detachment.
  The agreed signal/GC probes remain a prerequisite for session implementation; a
  failure requires a maintainer decision rather than a substituted signal policy.
- **R16, the SEP 47 dependency, accepted.** Sections 3.3, 6.2 and 11 require the
  completed descriptor/request split and generic preparation entry point before
  production adapter integration. At the inspected #607 head `6a94b3cc`, those changes
  are still pending in code. Julia consumes only the eventual request-based interface,
  without inheriting the legacy driver or extending its transitional dispatcher.
- **R17, formatting, accepted.** Section 6.7 distinguishes Runic's blocking format
  check from the 92-character author/reviewer rule, which Runic does not enforce.

**Acceptance.** Status is Accepted and review status Reviewed, with John Travers's
human review recorded alongside the earlier GPT-6 review. Part I is now frozen.
The implementation remains one branch and one draft pull request. Completion will
leave that pull request ready for maintainer review, without an automatic merge.
The proposal's merge does not close implementation tracking issue #608.

**Evidence and limits.** The review inspected the current contracts, source and
primary tool documentation. It ran no propagation, signal probe or GPU qualification.
The full primary papers for the double-word arithmetic bounds remain to be checked
in phase 3, as the earlier review already requires.

### Signal-policy decision before merge — 2026-11-10

Phase 1 feasibility probes found that Julia 1.13.3 on macOS arm64 terminates the
process on SIGTERM even after restoring a custom or ignored Python handler. Two
independent subprocess investigations reproduced this. Disabling Julia's signal
handling instead crashed during shutdown in a two-thread probe. The proposal merge
was held for the maintainer's decision, as R8 and R15 required.

John Travers chose to keep Julia 1.13.3 in process and accept the SIGTERM limitation,
rather than adopt a prerelease runtime before device packages are qualified. This
decision explicitly supersedes the SIGTERM-preservation promises in the earlier R8
and R15 responses and implementation response above. Sections 5.4, 7.2, 8 and phase 1
now describe the limitation and its tests. The actual Python SIGINT handler remains
preserved, Julia's SIGINT thread mask remains installed, and lifetime-owned atomic
stop tokens remain the independent graceful cancellation mechanism. Threaded GC,
bounded Python Ctrl-C handling and concurrent cancellation are still blocking probes.

[#610] stays open to revisit signal ownership with stable Julia 1.14 and compatible
device packages. Its resolution is not required for this SEP's implementation or
Final status under the policy accepted here. No propagation or GPU qualification is
claimed by these runtime feasibility probes.

Two implementation clarifications also precede the freeze: filesystem-only status
reports the prepared CPU name and target, leaving native-cache compatibility checks
to Julia at load; and the quality test group uses its own pinned environment so its
tools never enter the runtime dependency closure. Both groups remain blocking.

Independent review found no blocking issue in this amendment. Its documentation
checks passed: spelling, whitespace, 349 SEP consistency tests (three expected skips),
the Sphinx build with warnings as errors, and GitHub CI. The separate documentation
preview was cancelled and is being rebuilt before merge, as the repository requires.

[#610]: https://github.com/LupoLab/solprop/issues/610
