# SEP 36 — Single and mixed precision on the native backend

| | |
|---|---|
| **SEP** | 36 |
| **Title** | Single and mixed precision on the native backend |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-25); Opus 5.5 (2026-10-25, revision after review) |
| **Status** | Accepted |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Feature |
| **Created** | 2026-10-25 |
| **Resolution** | Accepted 2026-10-25 by John Travers after his human review, the GPT-6 AI review and the revision answering it, recorded in Part III ([LupoLab/solprop#27](https://github.com/LupoLab/solprop/pull/27)). |
| **Requires** | [SEP 33](archive/sep-0033-single-precision-mlx.md) (the precision setting, the binary32 contract, its oracle and survival record); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (the floating-point policy and the bound classes); [SEP 29](archive/sep-0029-compiled-backend-distribution.md) (the native backend in solprop and its wheels); [SEP 30](archive/sep-0030-standard-benchmark-suite.md) (the benchmark suite) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

*Revised on 2026-10-25 after the GPT-6 AI review. The response to the review (Part III)
says what changed for each finding; the maintainers' decisions on the changes are in
§12.*

### 1. Abstract

Bring reduced precision to the native (Kokkos) backend, on its CPU builds and on its
CUDA and HIP device builds, for speed and throughput. Two arithmetic modes are
proposed.

- **`precision="single"`** is SEP 33's binary32 contract, implemented on native. The
  contract's exponential is stated here with bounded range reductions for its real and
  imaginary parts (§4.3). Native may keep more in binary64 than the contract requires
  where profiles show it is worth it, provided its comparison bounds still hold.
- **`precision="mixed"`** is new. Everything that touches the propagated state is
  binary64: the state, the stage values, the propagators, the Adams history, the
  once-per-step boundary treatments, and the norms and statistics. The source
  evaluation is binary32: the transforms, the responses with their own running
  integrals, the in-stage absorber terms, and the transverse and modal matrix products.

Mixed precision is possible only where binary64 exists, which is every native target
but not Apple GPUs. It removes single precision's coherent propagator drift and its
damping limit, and keeps the error floor near binary32's rounding times a step's
nonlinear phase. That makes it a candidate for noise-seeded ensembles, which single
precision refuses. `numpy_ref` gains a mixed path as the oracle of the new mode. Its
single-precision path stays the oracle of the other.

Success is judged in three ways:
- agreement of each native mode with its oracle within the bounds of §7;
- a survival record for the mixed mode, including noise-seeded cases, against
  criteria fixed in advance;
- recorded speed-ups over binary64 native on the targets that motivate it: NVIDIA GPUs
  with reduced binary64 rates such as the A40, an AMD Instinct GPU with full binary64
  rates, and multi-core x86-64 and aarch64 CPUs.

### 2. Motivation and scope

**Reduced-FP64 GPUs are plentiful and slow in binary64.** NVIDIA's workstation and
graphics parts, such as the A40 and L40S and the RTX series (GA102, AD102 and their
successors), execute binary64 at 1/64 of their binary32 rate. Research clusters hold many
of them; the A40 nodes available to this project are an example. The native device
builds compute in binary64 throughout, so on these cards they run at the binary64 rate.

**An indicative measurement on one A40** (2026-10-25: 48 GB, 84 SMs, CUDA 12.8.1 with
its cuFFT and cuBLAS, `nvcc -O3 -arch=sm_86`) timed the building blocks of an
interaction-picture step. It ran in one process, with medians of seven rounds of 10–50
repetitions timed by CUDA events. Inputs were finite deterministic pseudo-random
values in $[-0.5, 0.5)$. Every library status was checked, each transform pair's round
trip was validated, and one GEMM element was checked against a host dot product
(binary32 errors about $10^{-7}$ and $10^{-9}$ respectively). These figures motivate the
proposal; they are not qualification evidence (§7.4).

Transforms, forward and inverse complex-to-complex, each pair followed by a $1/n$
scaling pass:

| shape | binary64, µs | binary32, µs | ratio |
|---|---|---|---|
| one line, $2^{14}$ | 108 | 18 | 6.2 |
| one line, $2^{16}$ | 108 | 17 | 6.3 |
| one line, $2^{18}$ | 256 | 29 | 8.8 |
| 204 × 2048 (multimode) | 202 | 22 | 9.1 |
| 64 × $2^{14}$ (a batch) | 768 | 88 | 8.8 |
| 256 × 4096 (radial) | 614 | 85 | 7.2 |
| 384 × 384 × 256 (Cartesian, 3-D) | 32 418 | 8771 | 3.7 |

Real matrix products ($m \times k$ by $k \times n$), binary32 in cuBLAS's pedantic
mode (no TF32):

| shape | DGEMM, µs | SGEMM, µs | ratio |
|---|---|---|---|
| 256 × 256 × 8192 (a radial Hankel transform) | 2296 | 66 | 35 |
| 64 × 64 × 32 768 | 647 | 40 | 16 |
| 1024 × 1024 × 4096 | 17 255 | 428 | 40 |

Element-wise passes, µs per pass, at $n$ complex samples:

| $n$ | stage combination: binary64 / binary32 | propagator from three terms: binary64 / binary32 / binary64 argument, binary32 trigonometry |
|---|---|---|
| $2^{16}$ | 6.5 / 3.2 | 17.6 / 2.8 / 7.8 |
| $2^{18}$ | 50 / 24 | 55 / 16 / 29 |
| $2^{20}$ | 203 / 88 | 198 / 59 / 103 |
| 384 × 384 × 256 | 7363 / 3319 | 6895 / 2040 / 3563 |

The last column forms the exponent in binary64 and evaluates `exp` and `sincos` in
binary32 with a single-constant reduction of the phase only. It is a timing kernel,
not the accurate form of §4.3.

Reductions and running sums, µs:

| $n$ | sum of squares: complex128 / complex64 with a binary32 accumulator / complex64 with a binary64 accumulator | inclusive scan: binary64 / binary32 / binary32 data with a binary64 accumulator |
|---|---|---|
| $2^{16}$ | 6.8 / 3.9 / 7.4 | 17 / 14 / 17 |
| $2^{20}$ | 35 / 17 / 37 | 44 / 27 / 41 |
| 384 × 384 × 256 | 1046 / 505 / 755 | 1074 / 549 / 806 |

Five observations shape this proposal:
- The transforms and the matrix products are where binary64 hurts on this card, by
  6–9× and 16–40×.
- Streaming element-wise passes are bandwidth-bound in either precision and cost about
  the byte ratio, 2.2×, in binary64.
- A full binary64 propagator costs about 1.9× a form with a binary64 argument and
  binary32 trigonometry at $2^{20}$, and 2.3× at $2^{16}$, where the binary64
  transcendentals make it compute-bound.
- Binary64 accumulation of binary32 data is not free on this card: a reduction or scan
  costs 1.5–2.2× its binary32 form, close to the full binary64 cost.
- At $2^{14}$ the binary64 transform pair on the A40 (108 µs) is about as slow as the
  same pair in FFTW on one laptop core (118 µs under load, below).

So mixed precision on this card keeps single precision's gain on the transforms and
matrix products. It keeps binary64's cost on everything that touches the state (§4.2),
which on the A40 is the bandwidth-bound passes at their binary64 rate, plus the full
binary64 propagator (§6.4). Whether that leaves a worthwhile mixed gain on a whole step
is what phase 4's profiles establish.

**On CPUs and full-rate GPUs the gain is the byte and SIMD factor.** The same transform
pair, with FFTW's built-in normalisation, was timed through pyFFTW 0.15.1 on an Apple
M5 Max CPU. It used FFTW 3.3.11 built with its NEON codelets in both precisions, the
configuration of native's local macOS builds. Plans were made with `FFTW_MEASURE`,
inputs were refilled after planning, and outputs were checked finite and round trips
validated:

| shape | threads | ratio, binary64 over binary32 |
|---|---|---|
| $2^{14}$ | 1 | 1.26 |
| $2^{16}$ | 1 | 1.51 |
| $2^{18}$ | 4 | 1.43 |
| 204 × 2048 | 4 | 1.34 |
| 256 × 4096 | 8 | 1.62 |

The machine was under other load (load average about 9.5), so only the ratios within
each pair are meaningful. On x86-64-v3 (AVX2) and aarch64 (NEON, SVE) builds, the
memory-bound passes should approach 2×. Data-centre GPUs with full binary64 rates
(H100, H200, MI300-series) gain by the byte factor too, which also doubles the problem
that fits in device memory.

These measurements are indicative. Their kernels and scripts are kept with the
project's development records, as SEP 30 does for campaign harnesses. The phases
replace them with public records made by the standard suite's harness under the
benchmark rules, which carry compiler and library provenance (§7.4).

**Native differs from MLX in one way that matters: binary64 is always there.** SEP 33
had to put everything in binary32 on Apple GPUs and emulate binary64 only for the
propagator's argument. On native, binary64 is available on every target, at a cost
that depends on the target and the pass. The design therefore makes two kinds of
choice:
- a *mode* choice for users: full binary32 for the largest speed-up, or mixed for
  binary64 accuracy in the linear part and a much lower error floor;
- an *implementation* choice inside each mode: binary64 is kept where the mode's
  contract needs it (the propagator's argument and host scalars under single
  precision; everything that touches the state under mixed precision), and elsewhere where
  profiles show it is worth its cost, provided the mode's bounds still hold (§4.5).

**Noise ensembles are the throughput workload.** Many A40s run many independent
noise-seeded shots, for coherence and ensemble spectra. SEP 33 refuses noise under
single precision: binary32 rounding of the state can approach a one-photon-per-mode
seed at microjoule energies (SEP 33 §4.5). With the state in binary64, the only
binary32 rounding is the source's, scaled by the step's nonlinear phase. The mixed mode
can therefore qualify noise where single cannot (§4.4).

**In scope.**
- The precision value `"mixed"`: the `Precision` record's mixed form, admission,
  validation, configuration and schema (§5).
- `numpy_ref`'s mixed path, as the mixed oracle.
- Native single and mixed precision, on CPU (OpenMP) and device (CUDA, HIP) builds, on
  SEP 33's routes and responses. These are capillary and fibre runs (mode-averaged and
  multimode, field and envelope grids, every SEP 33 response including field- and
  envelope-grid plasma), radial free space, and Cartesian free space including a line.
- The C++ precision policy: kernels templated on a precision policy, the
  single-precision FFTW, Kokkos-FFT and Kokkos Kernels paths, strict blocks in every
  type, the arithmetic probes, and a CMake option choosing the compiled precisions.
- The mixed survival study, with noise-seeded cases, and the admission of noise under
  mixed precision on what it qualifies.
- The Adams integrator's admission under mixed precision, measured by a rule fixed in
  advance (Q2).
- An A40 (Ampere, `AMPERE86`) build preset and a local GPU qualification lane on A40
  nodes.
- Qualification of both modes on an AMD Instinct GPU (CDNA 3, `AMD_GFX942`, such as the
  MI325X) through the HIP build, on rented hardware within a budget agreed with the
  maintainers before the phase starts: the same class R tests and confirmation cases
  as on the A40, the rocBLAS and rocFFT probes, and profiled benchmarks.
- Benchmarks of both modes against binary64 native on the A40, the AMD GPU, multi-core
  x86-64 and aarch64 CPUs, and against `mlx` on an Apple-silicon Mac. These include the
  throughput of concurrent independent runs on one GPU, and per-pass profiles on both
  GPUs, so that where each mode's time goes is understood, not only its total.

**Not in scope.**
- `numba_cpu`. It keeps refusing both reduced modes, and its refusal message names
  `native`.
- Routes and responses that SEP 33 refuses: the general modal route, the planar kinds,
  coupled flows, the moving guide, TDSE responses, the dispersive electronic response,
  far-field statistics and `"cf4"` sections. They stay refused under both reduced
  modes.
- Noise under full single precision. It stays refused (SEP 33 §4.5).
- A fused batch axis on the device. This SEP measures the gap that one would close and
  records a rule for proposing it (Q8).
- Reduced-precision matrix products of any kind (TF32, BF16, FP16, FP8), approximate
  transcendentals, and fast-math flags (SEP 25 §5). Emulated binary64 matrix products
  on tensor cores are discussed in §10 and not proposed.
- Binary32 output files (SEP 33 Q6, still deferred).
- A mixed mode on `mlx`, which Apple GPUs cannot provide.
- AMD GPUs other than the qualified CDNA 3 part (other architectures need their own
  build and lane), and AMD consumer GPUs.

### 3. Context

- **SEP 33 built the setting and the oracle.** `solprop.translate.precision` defines
  `Precision`, a frozen record with fields `state` and `source`. Its `__post_init__`
  refuses a record whose fields differ, and its docstring reserves that form for a
  later mixed mode. The same module holds:
  - the constants `DOUBLE` and `SINGLE`;
  - `SINGLE_PRECISION_BACKENDS = ("numpy_ref", "mlx")` and the floor
    `SINGLE_RTOL_FLOOR = 1e-6`;
  - the shared admission `admit_precision` and the study context `study_admission`;
  - the range and reduction helpers `block_exponent`, `narrow`, `state_scale`,
    `launch_peak`, `Coefficient`/`combine`, `pairwise_sum`, `prefix_sum` and
    `scaled_square_sum`.

  `KernelProblem.precision` (kernel schema 36) has `_check_precision`, which refuses
  the out-of-scope routes and noise under single precision, and
  `_check_single_exponent`, which refuses a step's damping above $\ln(\mathrm{FLT\_MAX})$.
  `NumpySingleBackend` (`backends/numpy_ref/single.py`) is the binary32 oracle, reached
  through hooks of the reference (`_upload`, `_propagator`, `_rounded`, `_error_norm`,
  `_finish`, `_envelope_sources` and others). ADR-054 records single precision as an
  opt-in arithmetic validated against its oracle. The manual page
  [Reduced precision](../../guide/performance/reduced-precision.md) carries the survival
  record.
- **What SEP 33 measured** (its Part II):
  - Every survival case survives in binary32 at the floor except the reduced
    comb-to-continuum case. That case survives only over 0.05 m, because noise-free
    stimulated Raman scattering amplifies any rounding-sized Stokes seed (a physical
    limit, not a defect).
  - The floor stays at $10^{-6}$.
  - The propagator drift constant is $c \approx 0.5$, so the field envelope
    $10^{-3}$ is reached after about $3 \times 10^4$ steps.
  - Adams was not admitted by the rule then fixed, although binary64 Adams has the same
    errors at the same tolerances.
- **Stage values and boundary treatments.** Stage values and Adams source rows are held
  in the interaction variable: the reference's `_nonlinear` returns the source
  multiplied by the inverse propagator $e^{-\Delta\Phi}$. Absorbers come in two schemes
  (ADR-025). The in-stage scheme evaluates the temporal or spatial absorber as a term of
  the stage (`_absorber_term`, and the Cartesian `_nonlinear_free`). The split-step
  scheme applies $e^{-\mathrm{rate}\,\Delta z}$ to the accepted state in `_finish`,
  through temporal (and on the Cartesian route, transverse) transforms. Noise comes in
  two models: `"input"`, added to the launch state, and `"modified"`, kept apart,
  phase-transported and added to every stage input.
- **The native backend today** ([backends.md §4](../backends.md),
  [The native backend](../../guide/performance/native.md)):
  - *Language and size.* C++20 on Kokkos 5.2, Kokkos Kernels 5.2 and Kokkos-FFT 2.0,
    all fetched at configure time. It is bound with nanobind, in `native/`, about
    25 000 lines of C++, half of them in `src/general.cpp`.
  - *Types.* The arithmetic type is fixed: `Complex = Kokkos::complex<double>` and the
    view aliases in `include/solprop_native/core.hpp`, with further view aliases
    declared per translation unit. There are about 1300 uses of `double` and 1200 of
    the complex type. No kernel is templated on its scalar type.
  - *Transforms.* All go through Kokkos-FFT (FFTW on the host, cuFFT or rocFFT on
    devices), which is templated on the view's value type. The CMake host-library
    module already locates FFTW's single-precision libraries. FFTW wisdom seeding and
    the optional Accelerate Raman path are written for `double`.
  - *Matrix products.* All go through `KokkosBlas::gemm` (host BLAS, cuBLAS or rocBLAS),
    with a tiled host kernel for real binary64 products. Nothing sets a math mode.
  - *Strict blocks.* `fp.hpp` defines the strict blocks of SEP 25 (contraction off
    around the phase increments, the ionised fraction, the multistep defects and
    compensation) and strict complex helpers, all in binary64.
  - *Propagators.* The stage propagator is formed per bin and stage by a strict phase
    sum and one `exp`, `cos`, `sin`, shared with the inverse propagator. Nothing is
    cached across steps.
  - *Reductions.* Norms and statistics are reduced in binary64 by Kokkos
    `parallel_reduce` and summed on the host.
  - *Plasma.* The field-grid plasma forms $1 - (1 - p)e^{-X}$; the envelope-grid
    fraction already uses the stable form $p + (1-p)(-\mathrm{expm1}(-X))$.
  - *Python side.* The class is `NativeBackend(StepDriver)` with a lockstep batch
    runner (`NativeBatchBackend`). Its build record reports `precision = "complex128"`,
    and the backend refuses `problem.precision.single` by name.
  - *Device builds.* These use `SOLPROP_NATIVE_ENABLE_CUDA`/`_HIP` with one Kokkos
    architecture per build (default `HOPPER90`, `AMD_GFX942`). There is no Ampere
    preset.
- **The roadmap item** "Floating-point options and reduced precision" lists two open
  items. One is a single-precision source evaluation inside a double-precision solver,
  which is this SEP's mixed mode. The other is binary64 transcendentals on reduced-FP64
  GPUs, which is §6.4 here. SEP 33's open questions are also listed there: the Adams
  rule, single and mixed precision on native and `numba_cpu`, and noise.
- **Decisions.**
  - ADR-020 orders accuracy and reliability before CPU performance before GPU, and
    builds GPU paths only for what the CPU backends have validated. This SEP follows
    it: each native mode is qualified on CPU against its oracle before its device
    lane.
  - ADR-051 keeps no compatibility with earlier configurations and schemas.
  - ADR-052 sets the thread budget, which applies unchanged.
  - ADR-054 restricts binary32 to an explicit per-run choice; §9 says how this SEP
    extends it.

### 4. Theory

#### 4.1 What carries over from SEP 33

SEP 33 §4 is adopted unchanged for `precision="single"` on native:
- the four rounding contributions of an interaction-picture step (the source's and the
  state's, random; the propagator's and the coefficients', coherent);
- the contract of §4.2, including the binary64-accurate propagator rounded once, whose
  binary32 evaluation §4.3 states with its range reductions and error;
- the range rules of §4.3: block exponents for narrowed buffers, the launch's
  power-of-two state scale, coefficients combined on the host, per-term homogeneity
  factors and scaled reductions;
- the stable plasma forms of §4.4;
- the floor $\mathrm{rtol} \ge 10^{-6}$.

The contract fixes what must be at least binary32-accurate and what must be
binary64-accurate. It does not forbid more precision. §4.5 says what native keeps in
binary64 under it.

Notation: $u_{32} = 2^{-24}$ and $u_{64} = 2^{-53}$ are the unit roundoffs. $n$ is the
transform length, $\phi_{\mathrm{NL}} = \Delta z\,\lVert\mathcal{N}\rVert/\lVert
A\rVert$ is a step's nonlinear phase, $\Phi_{\mathrm{NL}}$ is the run's total, and $M$
is the number of steps.

#### 4.2 The mixed contract

**The stage, written out.** In the interaction picture a stage value is the source
pulled back to the step's start:
$$k_i = e^{-\Delta\Phi_i}\,\mathcal{N}\big(e^{\Delta\Phi_i}(A + \Delta z \textstyle\sum_j a_{ij} k_j)\big),$$
as `NumpyRefBackend._nonlinear` returns it (the source multiplied by the inverse
propagator) and as the Adams source rows store it. The inverse factor $e^{-\Delta\Phi_i}$
is large wherever a bin is strongly damped. A lab-frame source of $10^{-10}$ multiplied
by $e^{150}$ is $1.4 \times 10^{55}$: binary64 holds it, binary32 does not. That is why
SEP 33 refuses single-precision steps whose damping exceeds $\ln(\mathrm{FLT\_MAX})$.

**The operation order under mixed precision**, per stage $i$:

1. *Stage input, binary64:* $y_i = e^{\Delta\Phi_i}(A + \Delta z \sum_j a_{ij} k_j)$,
   with $A$, every $k_j$ and the propagator binary64. When the modified noise model is
   active (§4.4), its phase-transported realisation is added here in binary64.
2. *Narrowing, one rounding:* $\tilde y_i = \mathrm{fl}_{32}(y_i / s)$, with $s$ the
   launch's power-of-two scale of SEP 33 §4.3. Division by $s$ is exact.
3. *Source, binary32:* $\tilde n_i = \mathcal{N}_{32}(\tilde y_i)$, SEP 33's
   single-precision source evaluation of the scaled input. This is every transform, the
   responses with their running integrals in SEP 33's orders, the in-stage absorber
   terms (below), and the transverse and modal matrix products. SEP 33's per-term
   homogeneity factors make $\tilde n_i = \mathcal{N}(s\tilde y_i)/s$ up to binary32
   rounding, held in range by its block exponents and combined coefficients.
4. *Widening and pull-back, binary64:*
   $k_i = e^{-\Delta\Phi_i} \cdot s \cdot \mathrm{wide}(\tilde n_i)$. Widening and the
   multiplication by $s$ are exact; the inverse propagator is applied in binary64. The
   stage value is stored in binary64.

**Stored rows are binary64.** Steps 1 and 4 keep every row the march stores in the
interaction variable in binary64, where the inverse flow's range is binary64's. This
covers the Runge–Kutta stage values (FSAL included), the low-storage registers, the
Adams source and history rows with their compensation, the dense-output rows and every
snapshot row. Mixed precision therefore needs no damping restriction:
`_check_single_exponent` applies to single precision only, and the binary32 values of
step 3 are bounded by SEP 33's lab-frame range rules, which never see
$e^{-\Delta\Phi}$. The cost is that the stage combinations read and write binary64,
which on the A40 is the binary64 rate of a bandwidth-bound pass (§2). Binary32 stage
rows would need SEP 33's damping refusal back and are not proposed (§10).

**The linear part, precisely.** Two kinds of linear operation exist, and they differ in
how their rounding enters the state:

- *Once-per-step treatments act on the whole state:* the diagonal flow $e^{\Delta\Phi}$,
  the band mask, and the split-step absorber of `_finish`, which transforms the state to
  the time (and, on the Cartesian route, transverse) domain, applies
  $e^{-\mathrm{rate}\,\Delta z}$ and transforms back. A binary32 rounding there would
  enter the state unscaled, at $u_{32}$ per step, which is single precision's state
  rounding. These are **binary64 under mixed precision, including their transforms**.
  Mixed precision therefore keeps binary64 transform plans and buffers for the
  split-step absorber when one is active, and the memory estimate and benchmarks count
  them.
- *In-stage terms are part of the source:* the in-stage temporal absorber
  ($-\tfrac12\,\mathrm{rate}\,E$) and the spatial absorber of `_absorber_term` and the
  Cartesian `_nonlinear_free` are evaluated on the stage input and enter the state
  through $\Delta z\,k_i$, like any response. They are **binary32 under mixed
  precision**, inside step 3. Their rounding is a source error scaled by the step's
  absorber exponent $\Delta z\,\mathrm{rate}$.

A source-free mixed run with an inactive collar therefore agrees with binary64 to
binary64 rounding. A source-free run with an in-stage absorber carries the in-stage
absorber's binary32 error, within the bound of §7.1. Both are tested (§7.2), under each
absorber scheme.

**Reductions.** Reductions over the state, meaning the error norm and the statistics,
are formed from the binary64 state in binary64. Running integrals and reductions
*inside* the source (the plasma's exposure and free-current integrals, the Raman
convolution's sums, the quadrature projections) are part of step 3 and follow SEP 33:
binary32, with SEP 33's bounded orders on the oracle and a stated depth bound on
native. Sums inside the transforms and matrix products are the libraries' binary32
sums, which no mode controls; the probes and bounds cover them (§6.3, §7.1).

**Error.** The linear part is binary64, so contributions 2 and 3 of SEP 33 §4.1 (the
state's rounding and the propagator's coherent drift) fall to binary64 levels. What
remains is binary32 rounding of the source and of its narrowed input. It enters the
state scaled by the step. §7.1 states the comparison bound. As a scale for design, the
heuristic per-step relative error is
$$\delta \approx c_s\,u_{32}\,\log_2 n\;\phi_{\mathrm{NL}},$$
random in sign from step to step, plus the coefficients' coherent error of about
$u_{32}\,\Phi_{\mathrm{NL}}$ over the run. These are estimates, not bounds. Responses with
large condition numbers (a plasma rate near its cutoffs, a source formed by
cancellation) multiply $\delta$ by their condition number, which §7.1's bound carries
explicitly. Linear propagation contributes nothing at binary32 level, whatever the
band-edge phase, the damping or the number of steps.

**The controller's floor.** The embedded estimate $\Delta z \sum_i (b_i - \hat b_i) k_i$
sees the source's rounding scaled by the weights' differences, which for DOPRI5 are of
order $10^{-2}$–$10^{-1}$. Its rounding noise is about
$10^{-1} c_s u_{32} \log_2 n\,\phi_{\mathrm{NL}}$. That is below $10^{-8}$ for
$n = 2^{16}$ and $\phi_{\mathrm{NL}} \le 0.1$, so the floor is $\mathrm{rtol} \ge 10^{-8}$,
raised by the study's rule (§7.3, Q1).

Admission does not enforce a small $\phi_{\mathrm{NL}}$, and it need not for the
controller to be safe. The rounding term of the estimate scales with $\Delta z$, while
the truncation term scales with $\Delta z^5$. A step whose rounding noise exceeds the
target is rejected and shortened, and the noise falls in proportion until it is below
the target, which happens at $\phi_{\mathrm{NL}} \lesssim 10\,\mathrm{rtol}/(c_s u_{32}\log_2 n)$.
The controller therefore cannot stall on rounding; at worst it takes more, shorter
steps than binary64 would at the same tolerance. That cost is measured: the floor rule
of §7.3 counts accepted and rejected steps against binary64's, not only final errors. A
source that nearly vanishes contributes rounding in proportion and needs no special
case. The absolute terms of SEP 33 §4.3 item 7 cover a source whose buffers underflow.

**Range under mixed precision.** Range is as step 2–4 above: the binary32 values are
SEP 33's scaled lab-frame quantities, and everything outside the source has binary64's
range.

#### 4.3 The propagator in each mode

- **Single.** SEP 33's contract asks for a binary64-accurate argument and one rounding
  of the propagator. A binary32 evaluation of the exponential meets that only if *both*
  parts of the argument are range-reduced in binary64 before the binary32 functions
  run. Rounding the real part alone to binary32 costs $|x|\,u_{32}$ relative: at
  $x = -80.000003$ the binary32 argument is $-80$, and $e^{-80}$ differs from $e^{x}$
  by $3.0 \times 10^{-6}$, about $50\,u_{32}$. Native therefore evaluates, from the
  binary64 argument $x + i\theta$:
  - the real part as $e^{x} = 2^{m} e^{r}$, with $m = \mathrm{round}(x / \ln 2)$ and
    $r = x - m \ln 2$ formed in binary64 with a two-part $\ln 2$ (Cody–Waite), so
    $|r| \le \tfrac12 \ln 2$. Then $e^{r}$ is evaluated in binary32 and scaled by
    $2^{m}$ exactly. Overflow and underflow are decided by $m$, within single
    precision's existing damping limit $|x| \le \ln(\mathrm{FLT\_MAX})$;
  - the phase as $\theta = 2\pi q + \rho$, with $q$ an integer and $\rho$ formed in
    binary64 with a three-part $2\pi$ (Cody–Waite), so $|\rho| \le \pi$, then
    `sincos` of $\rho$ in binary32. A single binary64 $2\pi$ constant is not enough:
    at $|\theta| = 10^{14}$ rad it leaves a phasor error of $4 \times 10^{-3}$ to
    $9 \times 10^{-3}$, depending on whether the subtraction is fused. With
    the leading parts short enough that $q$ times each is exact, the three-part
    reduction leaves an absolute error in $\rho$ below $10^{-14}$ for
    $|\theta| \le 2^{30}$ rad. That limit is more than $10^{5}$ times the largest
    per-step phase met so far ($10^{2}$–$10^{4}$ rad at band edges). Above $2^{30}$ rad the phase is
    reduced by the binary64 library `sin` and `cos`, which are exact in their
    reduction. That is rare, so its cost is immaterial.

  The stated accuracy is a relative error of the binary32 propagator of at most
  $10\,u_{32}$ for $|x| \le \ln(\mathrm{FLT\_MAX})$ and any finite $\theta$. That is
  the documented accuracies of the binary32 `expf`, `sinf` and `cosf` used (at most
  2 ulp on CUDA and HIP, 1 ulp in the CPU libraries; never the fast intrinsics) plus
  the product's rounding. The tests of §7.2
  check it at the extremes of both ranges, including the inverse propagator and the
  modulus's drift over many steps. On the A40 the timing kernel of §2, which reduced
  the phase only, cost 1.75× the all-binary32 form and 0.52× the all-binary64 form;
  the accurate form adds a few binary64 operations per bin and is timed in phase 2.
  MLX's double-float kernel (SEP 33) reduces both parts too. This makes explicit what
  "binary64-accurate, then rounded" requires of any binary32 evaluation.

- **Mixed.** The propagator multiplies a binary64 state, so a binary32 propagator would
  bring back SEP 33's coherent drift, $c\,M\,u_{32}$. It is formed and kept in binary64
  by the library's binary64 functions, as in the binary64 path. Its cost on
  reduced-FP64 GPUs is the mode's main performance risk (§6.4).

#### 4.4 Noise under mixed precision

**The two noise models.** `noise="input"` adds its realisation to the launch state
once, so under mixed precision the seed lives in the binary64 state.
`noise="modified"` keeps the realisation apart, transports its phase with the linear
flow, and adds it to the stage input at every source evaluation, so the propagating
field never carries it. Under mixed precision both are drawn in binary64. The modified
model's realisation is phase-transported and added in binary64 at step 1 of §4.2,
before the single narrowing of step 2. Both models are qualified separately (§7.3);
one that does not qualify is refused under mixed precision by name.

**Where the seed can be lost.** A one-photon-per-mode seed has a spectral amplitude,
relative to the pulse's peak spectral amplitude, of about
$\sigma_q \approx \sqrt{\hbar\omega_0 n_{\mathrm{occ}}/E_p}$ over $n_{\mathrm{occ}}$
occupied bins (SEP 33 §4.5): $1.6 \times 10^{-5}$ for 1 µJ over 1000 bins and
$5 \times 10^{-7}$ at 1 mJ.

- *Narrowing* the stage input rounds each spectral sample relative to its own
  magnitude. In an empty bin the seed is the sample and survives with relative error
  $u_{32}$. In an occupied bin it survives if $\sigma_q \gg u_{32}$, which holds by
  18 dB at 1 mJ and 48 dB at 1 µJ.
- *The source's own rounding* is spread over the bins [3]. Relative to the seed's
  contribution to the source, it is about $u_{32}\log_2 n/\sigma_q$ per evaluation.
  This is what decides whether amplified noise statistics are the quantum seed's or
  binary32's.
- *Accumulation.* Rounding is injected at every step and amplified from where it is
  injected by the same gain that amplifies the seed. A heuristic accumulated rounding
  amplitude per bin, for constant steps and undifferentiated growth, is
  $\sigma_r \approx c_s u_{32}\log_2 n\,\phi_{\mathrm{NL}}\sqrt{M n_{\mathrm{occ}}/n}$.
  For $c_s = 1$, $n = 2^{14}$, $\phi_{\mathrm{NL}} = 0.05$, $M = 10^4$ and
  $n_{\mathrm{occ}}/n = 1/4$, this is $2 \times 10^{-7}$: 38 dB below a 1 µJ seed and
  about 8 dB below a 1 mJ seed.

These estimates locate the risk but qualify nothing. In binary64 the same estimate is
$10^{-15}$ and always negligible. The study (§7.3) measures the margin directly, with
matched-discrete runs that isolate rounding from truncation, at every saved plane while
the dynamics are linear in the seed, and with source-response witnesses at identical
states. It separates the coherent bias (the noise-free difference) from the random part.

**Chaos.** Noise-seeded runs are chaotic where modulation instability or Raman gain
amplify the seed. Individual shots with the same seed then diverge between precisions
even when the ensemble is right, as they would under any perturbation, including a
binary64 one. Beyond the linear regime, noise is therefore qualified on ensemble
statistics, never on individual shots.

#### 4.5 What native may keep in binary64 under single precision

Extra precision only shrinks native's side of a comparison bound, and a class R bound
is the sum of both sides' bounds. So native may compute more in binary64 than the
single contract requires without changing the oracle or the bounds. Whether it should
is a performance question, answered per target by measurement. The A40 shows that
binary64 is not free on reduced-FP64 GPUs even for accumulation (§2). The choices:

- **The propagator's argument** and its range reductions: binary64 always (§4.3).
- **Host scalars and combined coefficients:** binary64 always, as in SEP 33.
- **Norms and statistics** of the binary32 state: accumulated in binary64 on CPUs and
  full-rate GPUs. On reduced-FP64 GPUs the phase 4 profiles decide between binary64
  accumulation and SEP 33's scaled pairwise forms. They run once per step or plane, so
  the choice is expected to matter little either way.
- **Running integrals inside the source** (plasma exposure and free current): binary32,
  in native's existing scan structure, with its depth bound (SEP 25 §4.6) stated at
  $u_{32}$. This matches the oracle's bound but not its order. Binary64 accumulation is
  an option if the bound is not informative for a case, at the measured cost.
- **Evaluated tables** (z tables, rate-table lookups): formed in binary64 and rounded
  once, as the oracle does on the host, unless the profiles show the binary64 lookup
  is a significant share of the step on a reduced-FP64 GPU. In that case the binary32
  form is adopted only where its added error is within the oracle comparison's bound.

Binary32 stays mandatory under single precision for the state, the stage values, the
transforms, the responses and the matrix products. Those are what the mode is for.

**Existing oracle exceptions.** SEP 33's oracle computes some quantities in binary64
from the binary32 state: the envelope-grid plasma's cycle moments, and the statistics.
Comparisons of those quantities between native and the oracle are therefore not
matched binary32 operation sequences. The tests say so and bound them as different
algorithms on the same input, not as rounding of one sequence.

#### 4.6 Integrators

- **Mixed.** The Adams history holds binary64 combinations of binary32-accurate source
  values. Its stored defects and compensation are binary64, so the multistep machinery
  is unchanged apart from the source's rounding. The default integrator under mixed
  precision is `Multistep()` if the study admits it under the rule of §7.3. The rule
  compares mixed Adams with binary64 Adams at the same tolerance, like with like. The
  default is `RungeKutta()` until then (Q2).
- **Single.** `RungeKutta()` stays the default and `Multistep` stays refused, as in
  SEP 33. Native contributes binary32 Adams measurements on the same like-with-like
  rule to the open question SEP 33 left on the roadmap, without deciding it (Q3).

### 5. Proposed interface

```text
from solprop.driver import prop_gnlse

shots = [
    prop_gnlse(
        ...,                     # the physics, unchanged
        precision="mixed",       # "double" (default), "single" or "mixed"
        backend="native",        # "auto" resolves to native for "mixed"
        rtol=1e-7,               # at or above the mixed floor (1e-8, Q1)
        noise="input",           # admitted under "mixed" from phase 5, on what it qualifies
        seed=k,
    )
    for k in range(32)
]
# each configuration records precision="mixed"; outputs are binary64 arrays
```

- **`precision`**: `Literal["double", "single", "mixed"]` on `SimulationSettings`,
  `Simulation` and every keyword interface that takes it. The default stays
  `"double"`, and no binary64 result changes.
- **The record.** `Precision(state="double", source="single")` becomes the valid
  `MIXED` constant, with `name == "mixed"`. A record with a binary32 state and a
  binary64 source stays refused.
- **Backends:**
  - `native` admits `"single"` and `"mixed"` when its build compiled them (§6.3). A
    build without them refuses by name and says how to build one.
  - `numpy_ref` admits both as oracles.
  - `mlx` admits `"single"` only, and refuses `"mixed"` because Apple GPUs have no
    binary64.
  - `numba_cpu` refuses both, naming `native`.
- **`backend="auto"`.** Under `"single"`, `mlx` when MLX and a Metal device are present
  (unchanged, Q6); otherwise `native` when its build has single precision; otherwise
  the existing refusal, now also naming `native`. Under `"mixed"`, `native` when its
  build has it, otherwise a refusal naming the `native` extra and `numpy_ref` (the
  oracle, never chosen by resolution).
- **Tolerances.**
  - Single keeps its $10^{-6}$ floor.
  - Mixed refuses an explicit `rtol` below its floor ($10^{-8}$, Q1), stating the floor.
  - Under each mode, every interface's default `rtol` is the larger of its binary64
    default and the floor, and its docstring says so.
- **Integrators.** Under `"single"`, `RungeKutta()` is the default and `Multistep` is
  refused (SEP 33). Under `"mixed"`, `RungeKutta()` is the default and `Multistep` is
  refused until phase 5 admits it (§4.6, Q2). The low-storage march is admitted under
  both.
- **Noise.** Refused under `"single"`. Under `"mixed"`, refused until phase 5, then
  admitted per noise model (`"input"`, `"modified"`) on the routes and cases the study
  qualifies, with a refusal or a documented limit where it does not (§7.3, Q4, Q11).
- **The rounding-margin indicator.** A noise-seeded mixed run records in its output
  metadata a heuristic indicator of its accumulated rounding, labelled as such:
  - per accepted step, the step's nonlinear phase
    $\phi = \Delta z\,\lVert k_{\mathrm{last}}\rVert / \lVert A\rVert$, from the last
    stage value of the accepted step and the state, in the norm the controller already
    uses (for multimode and radial states, the route's energy norm), computed in the
    existing norm reduction;
  - the running sum $\Sigma\phi^2$ over accepted steps only, since rejected trials do
    not reach the state; FSAL and Adams evaluations count once, with the accepted step
    they belong to;
  - at each saved plane, $\sigma_r = c_s u_{32}\log_2 n \sqrt{\Sigma\phi^2\,
    n_{\mathrm{occ}}/n}$ beside the seed's $\sigma_q$ of §4.4, with $n_{\mathrm{occ}}$
    the bins above −30 dB of the state's peak spectral energy, and their ratio in dB.

  The constant $c_s$ is calibrated by the study (§7.3). The indicator omits gain and
  coherent bias, and the metadata says so. It is recorded as absent, not as a number,
  for a zero state norm or a run without a noise model. The running sum is stored in
  snapshots, restored on resume and reset when a runner is reused (SEP 22 §4.7).
- **The build record.** `BuildInfo.precision` becomes `precisions`, the tuple of
  compiled modes (for example `("double", "single", "mixed")`). `solprop_native.build_info()`
  and the backend table report it.
- **Outputs, resume and snapshots.**
  - `observe` returns binary64 arrays.
  - The configuration's precision field takes `"mixed"`.
  - A resume in another precision is refused; a resume in the same precision on another
    backend is a class T comparison (SEP 33 §5.3).
  - A mixed snapshot stores a binary64 state and history, so its round trip is class P
    with no scaling.
- **Memory.** `estimate_memory` counts each buffer at its mode's element size. Under
  mixed precision, every stored row is binary64 and the source workspaces are binary32
  (§6.7).
- **Batches.** `select_batch` with `backend="native"` admits members of one precision.
  Members of different precisions are refused.

#### 5.1 Capabilities by backend and phase

"Phase $n$" is the phase that admits the capability (§11).

| capability | `numpy_ref` mixed | `native` single, CPU | `native` mixed, CPU | `native` single and mixed, CUDA and HIP |
|---|---|---|---|---|
| fibre route, every SEP 33 response, field-grid plasma, field and envelope grids | 1 | 2 | 3 | 4 |
| envelope-grid plasma | 1 | 3 | 3 | 4 |
| radial and Cartesian free space, including a line | 1 | 3 | 3 | 4 |
| Runge–Kutta, the low-storage march | 1 | 2 | 3 | 4 |
| Adams | study (1, 5) | refused (Q3) | 5, if admitted | 5, if admitted |
| fixed-schedule lockstep batches | 1 | 2 | 3 | 4 |
| noise | study (5) | refused | 5, where qualified | 5, where qualified |
| SEP 33's refused routes and responses | refused | refused | refused | refused |

### 6. Implementation strategies

#### 6.1 Outside the backends

- `Precision` admits the mixed form, and the `MIXED` constant and its name are added.
- `admit_precision` gains the mixed floor, the mixed integrator rule and per-mode
  backend lists. `SINGLE_PRECISION_BACKENDS` becomes a mapping from mode to backends.
- `KernelProblem._check_precision` refuses SEP 33's routes under both reduced modes.
  It refuses noise under single, and under mixed until phase 5.
- `_check_single_exponent` applies to single only: under mixed precision every row in
  the interaction variable is binary64 (§4.2).
- `SCHEMA_VERSION` and the output schema change in phase 1 for the new value, under
  ADR-051.
- The SEP 33 range helpers serve both modes unchanged.
- `tests/bounds.py` gains the mixed bound of §7.1, with its condition numbers, transport
  factor and two-sided composition.

#### 6.2 `numpy_ref`: the mixed oracle

`NumpyMixedBackend` implements §4.2's operation order explicitly. It does not merely
compose the two existing paths: the single path's source returns $N(A)/s$ in binary32
and its stage rows are binary32, and both differ from what mixed precision needs.

- **The source** is the single path's (`single_sources`, `kerr_single`, `raman_single`,
  `running_integral`, `plasma_current_single` and the narrowed tables), called on
  $\mathrm{fl}_{32}(y_i / s)$. Its running integrals stay binary32 in SEP 33's orders,
  because they are inside the source (§4.2).
- **The pull-back** widens the result exactly, multiplies by $s$ and by the binary64
  inverse propagator, and stores the stage value in complex128. These are the hooks
  `_nonlinear`, `_finish` and `_adams_source`, which the single path overrides
  differently.
- **The state path** is the reference's binary64 path: `_propagator`, the stage
  combinations, `_finish` with its split-step transforms, `_error_norm` and the
  statistics.
- **The in-stage absorber terms** move into the binary32 source. The split-step
  absorber stays in binary64.
- **Noise.** The modified model's realisation is added in binary64 before narrowing.
- **Dtype guards** assert complex128 on the state, the stage values and every stored
  row, complex64 or float32 inside the source, and that the source's input is the
  narrowing of a binary64 array. Adversarial tests place a binary64 array inside the
  source and a binary32 array in a stored row, and check that each is caught.

The oracle's own exceptions (§4.5: envelope-grid plasma moments and statistics in
binary64) carry over unchanged.

#### 6.3 Native: a precision policy in the C++

**The options.**

- **A. Kernels templated on a precision policy** (recommended). A policy type names
  the roles:

  | policy | state and stored rows | source, with its scans | state reductions | propagator |
  |---|---|---|---|---|
  | `Double` | `double` | `double` | `double` | `double` |
  | `Single` | `float` | `float` | per target (§4.5) | `float`, from a `double` argument (§4.3) |
  | `Mixed` | `double` | `float` | `double` | `double` |

  The view aliases (`StateView`, `SourceView`, `TableView`, `ReductionView`) are
  defined once per policy, replacing the per-file aliases. Each kernel and class is
  templated on the policy and explicitly instantiated in one translation unit per
  policy, so builds stay parallel. The strict helpers of `fp.hpp` become templates.
  The types make "no binary64 array in a binary32 pass" a property of the compiled
  code, not a runtime guard.
- **B. One source compiled three times** with preprocessor type aliases, into three
  extension modules. There is less template syntax, but the mixed mode still needs two
  types in one pass. Each `double` must still be assigned a role. There would be three
  modules to load, version and report, three times the link work, and no shared host
  code.
- **C. Hand-written binary32 kernels for the hot passes only.** This gives the smallest
  change for single precision. It would duplicate the passes, drift from the binary64
  ones, and give mixed precision no structure.

Option A is recommended. The `Double` instantiation must compile to the present
arithmetic: it is the same code with the same types, and the existing native tests and
a same-host timing comparison show it (§7.2).

**Assigning roles.** Most of the 1300 `double` uses are host scalars, indices or
bundle values, and stay `double`. The view aliases and the per-pass value types are
what change. Every arithmetic block declares the role of each operand. A binary32
source pass reads binary64 state only through the one narrowing step that forms its
input.

**Libraries.**
- *Transforms.* Kokkos-FFT on `Kokkos::complex<float>` views selects FFTW's `fftwf_`
  (single-precision FFTW, already located by `HostLibraries.cmake`), cuFFT's C2C/R2C or
  rocFFT's single-precision path. The FFTW wisdom seeder and the build check (which
  probes `cufftExecZ2Z`) gain single-precision branches. The optional Accelerate Raman
  path gains a `float` branch or falls back to Kokkos-FFT under binary32.
- *Matrix products.* `KokkosBlas::gemm` on `float` selects SGEMM and CGEMM. The tiled
  host kernel stays binary64-only unless measurement justifies a binary32 one.
- *Wheels.* The wheels and CI images build FFTW's single-precision libraries with the
  same SIMD codelets as the binary64 ones (AVX2 on x86-64-v3, NEON on aarch64).

**Arithmetic probes**, as SEP 33 §6.3 does for MLX:
- *The math mode, set.* cuBLAS computes binary32 products in TF32 when the handle's
  math mode allows it [5]. At construction, native sets the math mode explicitly on the
  cuBLAS handle that Kokkos Kernels' library layer uses for its products (its handle
  singleton): `CUBLAS_PEDANTIC_MATH` under single and mixed precision. On HIP it sets
  rocBLAS's default (non-`xf32`) math mode on the corresponding handle. A product
  issued through another handle or library is not admitted under the reduced modes.
- *The math mode, probed.* Construction then runs a GEMM probe for each dispatch family
  the run uses:
  - real and complex;
  - each transpose and layout combination that native issues;
  - the full node-chunk shape and the tail chunk's shape.

  Each probe multiplies small Gaussian factors with the run's inner dimension and at
  most 64 rows and columns, against a binary64 host reference, with SEP 33's
  discriminating threshold $2^{-17}$. Probe buffers are counted in the memory estimate
  and released before the march. A failed probe refuses the run and reports the
  observed relative error and the family, without guessing the cause.
  `NVIDIA_TF32_OVERRIDE=0`, which the cuBLAS documentation describes as disabling TF32
  [5], is mentioned in the message as one remedy, not as the diagnosis. The negative
  tests select TF32 (and on HIP `xf32`) through each library's API in a test fixture
  (§7.2).
- *Transforms.* cuFFT and FFTW in single precision are characterised against
  `fft_bound` at $u_{32}$ at each length and kind the routes use, in the test suite,
  and rocFFT likewise on the AMD lane.
- *Subnormals.* No build sets flush-to-zero (SEP 25). CUDA's binary32 arithmetic keeps
  subnormals unless `-ftz=true`, which the build must not pass; HIP builds must not pass
  `-fgpu-flush-denormals-to-zero`. Each device lane checks that a subnormal survives a
  pass.

**The Python side.**
- The host-side range work of SEP 33 (the launch scale, block exponents and combined
  coefficients) moves from `NumpySingleBackend` into a shared helper module that
  `numpy_ref`, `mlx` and `native` all use, so the three compute the same exponents.
- Narrowed tables are formed on the host by `translate.precision.narrow` and uploaded as
  complex64 or float32, which halves the upload. Binary64 tables of the linear part are
  uploaded as now.
- The nanobind signatures gain complex64 and float32 overloads. `exact_array` keeps
  refusing implicit conversion, so a binary64 array passed to a binary32 binding fails
  loudly.
- Norm partials stay binary64 and are summed on the host as now.

**Build options.** `SOLPROP_NATIVE_PRECISIONS` (default `double;single;mixed`) selects
the instantiated policies. A build without `double` is refused, since binary64 is the
default arithmetic. Phase 2 measures what the reduced policies add to compile time and
wheel size. The distributed wheels carry all three (Q5), unless the reduced policies
add more than 60 % to a wheel's size or take it past PyPI's per-file limit; then single
and mixed become an option of source builds and the guide says so.

#### 6.4 Propagators on reduced-FP64 GPUs

Under mixed precision, the binary64 propagator is formed per bin and stage, with its
inverse sharing the trigonometry. On the A40 it costs about one binary64
bandwidth-bound pass at $2^{20}$ (198 µs, against 203 µs for a binary64 stage
combination), but 6× the binary32 propagator at $2^{16}$. Phase 4 measures its share
of a mixed step on the A40.

**The 15 % share is a trigger to investigate, not a performance requirement.** Plain
binary64 evaluation is always valid and stays the fallback. Above the trigger, phase 4
evaluates the following remedies. Each is adopted only if it meets the binary64
contract with its error included in the comparison bound, and only if it is faster:

1. **Reuse while $\Delta z$ is unchanged.** Fixed schedules and constant-operator
   routes repeat the same propagators. They can be cached, keyed by the step and the
   terms' weights, within the memory budget. The values are identical, so this adds no
   error.
2. **Derive stage propagators by products.** On a constant operator, the propagators at
   the stage nodes are products of a few base propagators. Binary64 complex
   multiplication is bandwidth-bound, not transcendental-bound. Each product adds a few
   $u_{64}$ relative, and on a damped bin the products of partial factors stay within
   binary64's range, since every factor's modulus is between the step's endpoint values.
   The bound counts the products per stage (at most the number of stages) and charges
   $\gamma_{k}$ at $u_{64}$ for $k$ products. That stays negligible against the
   source's $u_{32}$ terms, but the bound states it.

A double-float evaluation on binary32 units is **not** proposed as a substitute. A
two-term binary32 expansion carries about 48 significant bits, below binary64's 53,
before its polynomial and range-reduction errors [8]. SEP 33's MLX kernel used one only
to form a binary32 result. A double-float exponential would need its own error analysis
over the full damping and phase ranges, and would be a new SEP's subject if the profiles
ever justified it.

Single precision needs none of these (§4.3). On the AMD GPU, whose binary64 vector
rate is half its binary32 rate, the binary64 propagator is expected to be
bandwidth-bound; phase 4 measures its share there too, as the contrast that separates
the A40's compute cost from the format's byte cost.

#### 6.5 Envelope-grid plasma

Native's envelope plasma refines quadrant rules per sample (a $2 \times 10^{-6}$
budget, a $10^{-30}$ floor). Under both reduced modes it runs in the source's binary32,
with quadrature sums accumulated in binary64. The refinement budget stays above binary32
rounding: $2 \times 10^{-6}$ is about $30\,u_{32}$. SEP 33's phase 4 lessons from its
device formulation are carried over: cutoffs honoured, panels past the low cut skipped,
the field clamped, and refinement failures surfaced as `ConvergenceError`. This is
qualified as class R against the oracle's binary32 moments where the operation
sequences match, and class T where they do not.

#### 6.6 Batches and throughput

The existing `NativeBatchBackend` fuses only the accepted update. It admits members of
one precision. Phase 4 measures the A40's throughput in samples times steps per second
on the fibre cases at $2^{13}$–$2^{16}$ samples, three ways:
- solo runs;
- the lockstep batch;
- 1, 2, 4 and 8 concurrent processes on one GPU, under the thread budget for their host
  threads.

The result informs the guide's advice on how to fill a GPU with an ensemble. It also
decides by rule whether a fused batch axis is proposed as a later SEP (Q8): only if, on
a noise ensemble of $2^{14}$–$2^{15}$-sample fibre runs, the best of the three ways
reaches less than half the throughput per GPU of the same case at $2^{18}$. If
concurrent processes already pass, the guide advises them and nothing is built. The
same measurement is repeated on the AMD GPU.

#### 6.7 Cost and memory

Resident bytes are derived from the memory ledgers (`estimate_memory` and the native
backend's own ledger), not asserted. The proposal's expectations, to be replaced by the
ledgers' figures in phases 2–3:

- **Single precision** holds the state, the stage values, the history and the source
  workspaces in binary32. The linear tables (operator terms and propagators as the
  route stores them), the accumulators, the library workspaces and the retained outputs
  keep their own sizes. The ratio to binary64 is therefore above one half. How far
  above depends on the route: the linear tables are a small share on large spatial
  grids and a larger one on small fibre runs.
- **Mixed precision** keeps every stored row binary64 (§4.2). It halves only the
  source's workspaces: the fine time grid, the node chunks, the transform buffers of
  the source and the narrowed response tables. It adds the binary32 source plans
  beside binary64 plans for the split-step absorber when one is active. On routes where
  the source workspaces dominate (radial and Cartesian node chunks, multimode
  quadratures), mixed precision saves a substantial fraction. On a single-mode fibre
  run it saves little, and it may need slightly more if both plan sets are resident.

Each phase records representative ratios from the ledgers for the benchmark cases, and
the device lanes check the measured peak against the estimate, including library
workspaces.

#### 6.8 Dependencies

None new. FFTW's single-precision libraries are part of the same FFTW distribution and
licence. cuFFT, cuBLAS, rocFFT and rocBLAS already include single precision.

### 7. Verification and validation

#### 7.1 Bounds

**Single precision** uses SEP 33's binary32 bounds and limits of use (its §7.1), with
§4.3's exponential bound in place of "one rounding of the propagator".

**Mixed precision.** The bound is written per stage, then per step, then over the run.
Each term is evaluated from quantities known before the comparison, never fitted to
results.

1. *Source error at a stage.* For the binary64 stage input $y_i$, the mixed stage value
   differs from the binary64 one by, per bin,
   $$\lvert \hat k_i - k_i \rvert \le \lvert e^{-\Delta\Phi_i}\rvert \Big( \kappa_{\mathcal N}\,u_{32}\,\lvert \mathcal N(y_i)\rvert_{\mathrm{nw}} + E_{\mathrm{pass},i} + E_{\mathrm{abs},i} \Big) + O(u_{64}).$$
   - $\kappa_{\mathcal N}$ is the response's relative condition number with respect to
     its input, which carries the narrowing of step 2: 3 for Kerr and Raman. For the
     plasma it is $1 + \max \lvert d\ln W / d\ln\lvert E\rvert \rvert$ over the samples'
     field range, read from the rate table on the host. The rate's steepness near its
     cutoffs therefore enters explicitly.
   - $E_{\mathrm{pass},i}$ is SEP 33's binary32 bound of the source's passes, from the
     `tests/bounds.py` formulas at $u_{32}$: transforms, matrix products, scans and one
     narrowing of each table entry per use.
   - $E_{\mathrm{abs},i}$ is SEP 33 §4.3 item 7's absolute underflow term.
   - $\lvert\cdot\rvert_{\mathrm{nw}}$ marks normwise terms (transforms, products)
     spread over bins, as SEP 33's bounds spread them.
   - The inverse propagator multiplies bin by bin in binary64, so it scales the absolute
     bound and leaves the relative one unchanged. It is not an amplification of the
     error relative to the stage value.
2. *Transport through the step.* A stage error enters later stages through the source's
   dependence on its input. To first order, the step's state error is
   $\Delta z \sum_i \lvert b_i\rvert\,\lvert\hat k_i - k_i\rvert$, multiplied by
   $(1 + \kappa_{\mathcal N}\phi_{\mathrm{NL}})^{s}$ for $s$ stages to cover
   propagation into later stages. Here $\phi_{\mathrm{NL}}$ is the step's nonlinear
   phase, computed from the binary64 reference's stage values. Products in the binary64
   state path add $u_{64}$ terms (and §6.4's product count, if adopted).
3. *Over the run.* On a fixed schedule the per-step bounds are summed and multiplied by
   the existing `propagation_growth`, as SEP 33 does.
4. *Two sides.* Native and the oracle each have their own bound, from their own
   accumulators: native's scan depth bound, the oracle's SEP 33 orders. A comparison's
   bound is their sum. Where one side's accumulator is binary64, its term is the
   binary64 one.

A bound larger than a tenth of a case's class T envelope is uninformative, as in
SEP 33. That comparison then runs on a shorter window, and the test says so. Responses
whose $\kappa$ makes every window uninformative (a plasma very close to a cutoff) are
recorded as such, and qualified by the class T study alone.

#### 7.2 Class P, class R and targeted tests

- **No binary64 change.** The existing native suite passes unchanged on the `Double`
  instantiation. A same-host A/B on the SEP 30 core cases shows no slowdown beyond the
  suite's noise (SEP 30's rules).
- **The oracles against binary64, one step.** As SEP 33 did for single precision:
  `numpy_ref` mixed against `numpy_ref` binary64 over one step, within §7.1's bound, on
  every capability of §5.1. This qualifies the mixed oracle independently of any
  native comparison.
- **Independent response witnesses, both reduced modes:**
  - dispersion-free SPM's phase, $P\int\gamma\,dz$;
  - the Raman convolution against a binary64 convolution of the same kernel;
  - plasma populations at small and saturated exposure against their closed forms;
  - photon-number conservation in lossless Kerr propagation.

  These check the oracle against physics, not against itself.
- **Native single against `numpy_ref` single**, and **native mixed against `numpy_ref`
  mixed**, on fixed schedules with a fixed-order method, for every capability of §5.1
  in its phase. These run on the CPU build in CI, on the CUDA build in the local A40
  lane, and on the HIP build on the AMD GPU.
- **Native single against `mlx`** on an Apple-silicon Mac. Both are bounded against the
  oracle, so this is a consistency check, not a gate.
- **The linear part of mixed precision.** Under each absorber scheme:
  - a source-free run with no collar agrees with binary64 within binary64 bounds;
  - a source-free run with a nonzero split-step temporal absorber agrees within binary64
    bounds;
  - a source-free run with nonzero in-stage temporal and spatial absorbers agrees within
    §7.1's bound for the absorber terms.

  All three run on the fibre route and on the Cartesian route.
- **Damping and scale witnesses, mixed precision.**
  - A strongly damped bin, with $|\mathrm{Re}\,\Delta\Phi| = 150$ per step, carrying a
    small source, runs without overflow and agrees with binary64.
  - A launch with scale $s \ne 1$ (a peak of $10^{9}$ V/m, so $s = 2^{30}$) gives the
    physical $\mathcal N(A)$ within the bound, for Kerr, Raman and the plasma.
- **The single-precision exponential.**
  - Relative error at most $10\,u_{32}$ against a high-precision reference over
    $x \in [-\ln \mathrm{FLT\_MAX}, \ln \mathrm{FLT\_MAX}]$, including $x$ just past
    multiples of $\ln 2$ and the witness $x = -80.000003$.
  - $\theta$ up to and beyond $2^{30}$ rad, including just past multiples of $2\pi$.
  - The inverse propagator's error.
  - Modulus drift over $10^4$ constant steps, against $c\,M\,u_{32}$ with SEP 33's $c$.
- **SEP 33's targeted tests, repeated on native in both modes:**
  - scaling, class P where only powers of two differ;
  - the Cartesian energy that would overflow unscaled;
  - the vanished molecular Raman response;
  - plasma populations;
  - the propagator drift, single only;
  - failure injection in fixed, low-storage and adaptive runs;
  - resume across backends in the same precision (class T) and refusal across
    precisions;
  - snapshot round trips, which are class P under mixed with binary64 rows;
  - memory against the estimate.
- **Probes (§6.3):**
  - the GEMM probe on each dispatch family;
  - a negative test that selects TF32 explicitly on the cuBLAS handle native uses, in a
    test fixture, and expects the refusal; on HIP, the same with rocBLAS's `xf32` math
    mode set through its API;
  - transform characterisation at $u_{32}$;
  - a subnormal surviving a device pass.
- **Batches.** Each member agrees with its solo run within the mode's bound.

#### 7.3 Class T: the survival studies

**Single precision.** The oracle's survival record (SEP 33 Part II) transfers to native
through the class R tests above. Phase 4 confirms it on each GPU, on the CUDA build on
the A40 and on the HIP build on the AMD GPU, by rerunning three of its cases against
their binary64 references: `ctc` at its 0.05 m limit, `argon` and `radial_filament`.
Each is judged by SEP 33's envelopes. This is a confirmation, not a new study.

**Mixed precision: coherent cases.** The criteria are SEP 33 §7.3's, recorded in
Part II before the first comparison:
- references in binary64 on the same grid at $\mathrm{rtol} = 10^{-10}$, checked
  against $10^{-9}$;
- the same metrics and envelopes (field $10^{-3}$, bands 1 %, energy $10^{-4}$, peak
  $10^{-3}$, ionised fraction $10^{-3}$, width $10^{-3}$);
- the same verdicts;
- SEP 33's cases, with candidates at $10^{-8}$, $10^{-7}$ and $10^{-6}$, each beside a
  binary64 run at the same tolerance to separate precision from truncation.

The mixed `rdw` and `argon` cases are also confirmed on both GPUs in phase 4.

- **The floor (Q1).** The floor rises from $10^{-8}$ to $10^{-7}$ if either of two
  things holds. One: on every case, no metric's error at $10^{-8}$ is smaller than at
  $10^{-7}$ by more than 25 %. Two: on any case at $10^{-8}$, the mixed run takes more
  than 1.25 times the accepted steps, or 1.5 times the rejected steps, of the binary64
  run at the same tolerance. The same test is then applied from $10^{-7}$ to $10^{-6}$.
- **Adams under mixed precision (Q2).** Admitted, and made the mixed default, if on the
  `rdw` and `argon` cases at each candidate tolerance its error is within every envelope
  and at most twice that of binary64 Adams at the same tolerance in every metric.
- **Expected.** Everything survives at the floor. `ctc` is "to measure": the source's
  rounding still seeds empty Stokes bins, at a level $\phi_{\mathrm{NL}}$ smaller than
  single's, so its limit should lengthen.

**Mixed precision: noise-seeded cases.** Every case's full definition is frozen in
Part II before the first candidate comparison, as SEP 33 froze its cases: pump, fibre
or medium parameters, energy, grid and window, length, saved planes, noise model,
seeds, integrator and tolerances. The cases are:

| id | route | definition (frozen in Part II) | expected |
|---|---|---|---|
| `scg_coherent` | fibre, envelope | the photonic-crystal-fibre supercontinuum of [6] (its dispersion, $\gamma$, Raman and shock; 835 nm, 50 fs FWHM sech, 10 kW, 15 cm) | survives |
| `scg_incoherent` | fibre, envelope | the same fibre with a longer pump chosen in Part II so that the binary64 ensemble's band-averaged $\lvert g_{12}\rvert$ is below 0.5 over most of the band | to measure |
| `rdw_noise` | fibre, field | `rdw` with a seed | survives |
| `ctc_noise` | fibre, field | `ctc` with a seed, Raman-gain dominated | to measure |
| `radial_filament_noise` | radial | `radial_filament` with a seed | to measure |

Every case runs under `noise="input"`. `scg_coherent`, `scg_incoherent` and `rdw_noise`
also run under `noise="modified"`. A model is admitted under mixed precision only on
what it passes.

**The criteria**, recorded before the first comparison:

1. **Rounding margin, in the linear regime (primary).** A binary64 run and a mixed run
   are made on the same discrete steps: the binary64 run's accepted schedule, replayed
   as a `FixedSchedule` in mixed precision. Their difference $D(z)$ then holds rounding
   and its amplification only, not truncation or different step histories. Beside it,
   the seed's response is $S(z) = A_{64}(z; \eta) - A_{64}(z; 0)$, for a seed $\eta$, on
   the same schedule. Both are linear responses while they are small. Both carry
   injection and amplification from everywhere along the run, rounding from every step
   and the seed from its entrance.
   - Per band above −30 dB of the mean spectrum, at every saved plane, the margin is
     $10\log_{10}(\overline{|S|^2}/\overline{|D|^2})$, averaged over 8 seeds. It must be
     at least 20 dB.
   - The margin is evaluated from the entrance up to the plane where
     $\overline{|S|^2}$ in that band first exceeds −20 dB of the band's mean energy. Up
     to that plane both are perturbations of the same flow. Beyond it the dynamics are
     no longer linear in the seed, and criterion 3 judges.
   - The coherent part of $D$ is the noise-free matched-discrete difference. It is
     reported separately as a bias and judged by the coherent-case envelopes. The random
     part is $D$ minus that bias, and is what the margin uses.
2. **Source-response witness at identical states.** At four planes of the binary64 run,
   for 8 seeds, the source's response to the seed,
   $\Delta\mathcal N = \mathcal N(A + \eta) - \mathcal N(A)$, is evaluated in mixed and
   in binary64 on the same binary64 state. The difference must be at most −20 dB of
   $\Delta\mathcal N$ in every band above −30 dB. This checks directly that narrowing
   and the binary32 source keep the seed's effect. Under `"modified"` it uses that
   model's stage-input addition.
3. **Ensemble statistics, $N = 64$ shots, matched seeds,** at the last plane, against
   the binary64 ensemble with the same seeds, in the 32 bands of SEP 33's spectral
   metric above −30 dB of the binary64 ensemble's mean spectrum (the reference selects
   the bands):
   - *the ensemble-mean spectral energy* per band;
   - *the band coherence*: `processing.coherence.g12` evaluated per bin, then averaged
     over the band's bins weighted by the binary64 ensemble's mean spectrum.

   Each statistic is compared on the paired difference, mixed minus binary64, over the
   matched seeds. Its standard error is estimated by a paired bootstrap that resamples
   seeds, with 2000 resamples. The tolerance is the larger of the absolute floor (1 %
   of the band's energy; 0.02 in band coherence) and $z$ times the paired standard
   error. Here $z$ holds the family-wise false-rejection rate at 5 % over every band and
   both statistics of a case, by Bonferroni ($z = 3.4$, two-sided, for 64
   comparisons).

**Verdicts** are as for the coherent cases. A failure becomes a refusal where its
condition can be computed from the problem before the run; otherwise it becomes a
documented limit (Q4). The study records the measured margins against §4.4's
heuristic, and states whether a problem-level predicate separates the passing cases
from the failing ones.

**The recorded indicator (§5).** The study calibrates $c_s$ for the run-time indicator
from the margins measured under criterion 1, and records how well the indicator
predicts them. The indicator is labelled a heuristic wherever it is reported.

#### 7.4 Benchmarks

Each phase records time per accepted step on short windows, under
[benchmarks.md](../benchmarks.md#rules), with build options, library versions and
measured peak memory:
- the SEP 30 core cases on CPU, single and mixed against binary64 native, on an x86-64
  host (2× AMD EPYC 7763 nodes and the x86-64-v3 wheel) and an aarch64 host (Apple
  M5 Max, and a Linux aarch64 host where available);
- the same cases on the A40 and on the AMD GPU, with the batch and concurrency
  throughput of §6.6;
- per-pass profiles of one step in each mode on both GPUs (Nsight Systems on the A40,
  `rocprofv3` kernel traces on the AMD GPU), attributing the time to transforms,
  matrix products, element-wise passes, propagators, reductions and launch overhead,
  and comparing the attribution with the indicative figures of §2;
- native single against `mlx` on the same Mac;
- the hypotheses of §2 and §4.5, each confirmed or refuted by the profiles: that mixed
  precision keeps most of single precision's transform and product gain on a whole
  step; the share of the binary64 propagator; and the cost of binary64 accumulation
  for norms and statistics on each GPU.

The records are made by the standard suite's harness (SEP 30), which captures the
compiler, the Kokkos, Kokkos Kernels and Kokkos-FFT versions, the FFT and BLAS libraries
and the build options. Their inputs are finite and representative, outputs are
validated, and library statuses are checked. The recipe for each GPU lane is part of
the building guide (§8). The user guide's advice comes from these records. Where a mode gains less than 1.3× on
a target and route, the guide says so. Benchmarks are not merge gates.

### 8. Documentation

- The user guide page *Single precision* becomes *Reduced precision*. It gains:
  - the two modes and when to use each;
  - the capability table by backend;
  - the mixed survival and noise records;
  - each mode's floor and limits;
  - the benchmark tables.
- [The native backend](../../guide/performance/native.md): running reduced precision on
  CPUs and GPUs, the A40 and AMD examples, and the TF32 and `xf32` refusals.
- [Building the native backend](../../guide/performance/building-native.md):
  `SOLPROP_NATIVE_PRECISIONS`, the Ampere (`AMPERE86`) preset, FFTW's
  single-precision libraries for source builds, and the recipe for each GPU
  qualification lane (A40 and CDNA 3).
- [Choosing a backend](../../guide/performance/backends.md): the precision column.
- The normative pages:
  - [architecture.md §1](../architecture.md): binary64 by default, single or mixed
    opt-in;
  - [conventions.md §6](../conventions.md): the kernel state is complex128 under
    double and mixed, complex64 under single;
  - [kernel-contract.md](../kernel-contract.md): the mixed record;
  - [backends.md](../backends.md) §4 and §9: native's precision policy and what runs
    where;
  - [testing.md §3](../testing.md): the mixed bound;
  - [decisions.md](../decisions.md): an ADR numbered at acceptance (§9).
- Docstrings for `precision` on every interface.
- A worked example: a noise ensemble's coherence in double and mixed precision on CPU,
  with timings stated, run by hand with `poe docs-examples`.

### 9. Compatibility and impact

- **Defaults unchanged.** No binary64 result changes. The templated `Double` path is
  the present arithmetic (§7.2).
- **Schemas.** `SCHEMA_VERSION` and the output schema change in phase 1 for the new
  precision value. Under ADR-051 every configuration's hash changes, and earlier files
  are not resumed. Native's build record changes from `precision` to `precisions`.
- **`mlx`** is unchanged, apart from refusing `"mixed"` by name and sharing the
  host-side range helpers.
- **The ADR**, written at acceptance: *Reduced precision on the native backend: a mixed
  mode, and binary64 where the contract or the profiles call for it*. It extends ADR-054 in four ways:
  - binary32 is admitted on native under SEP 33's contract, validated against the same
    oracle;
  - a mixed mode with its own oracle;
  - the principle that a backend may compute in more precision than its mode's contract
    requires, provided its comparison bounds hold;
  - noise admitted under mixed precision only, on what the study qualifies.

  It amends ADR-054's statement that binary64 is the only arithmetic of `native`.
  ADR-020 is followed: each mode is qualified on CPU before its device lane.
- **Luna.jl** has no reduced-precision mode, so there is nothing to record on the
  differences page.

### 10. Alternatives considered

- **Single precision only, as SEP 33.** This is less work, but it leaves the
  propagator's step-count limit, a floor of $10^{-6}$, and no route to noise ensembles,
  which are the A40 workload. Not chosen.
- **Mixed precision only.** This loses up to about 1.5× on the A40's element-wise
  passes and the 2× memory saving that lets larger problems fit. It also leaves native
  without the mode MLX runs, so the two could not be cross-checked. Not chosen.
- **Compiling the source three times** (§6.3, B) and **hand-written binary32 kernels**
  (§6.3, C). Not chosen, for the reasons given there.
- **Binary32 stage values under mixed precision**, as the first draft had them. They
  halve the stage rows' traffic, but they hold $e^{-\Delta\Phi}\mathcal N$, which
  overflows binary32 on strongly damped bins (§4.2). They would need SEP 33's damping
  refusal back, or a per-bin exponent, under the one mode meant to have neither. Not
  chosen. The profiles of phase 4 record the stage rows' share of a mixed step; if it
  dominates, a later proposal can revisit this with an exponent scheme of its own.
- **A binary32 propagator under mixed precision.** It costs the same as under single,
  and it reintroduces the coherent drift that is the mode's reason for being (§4.3).
  Not chosen.
- **One accumulation policy for every target.** The A40 measurements (§2) show binary64
  accumulation costing 1.5–2.2× on a reduced-FP64 GPU, while it is close to free on
  CPUs. §4.5 therefore fixes what each mode's contract needs, and leaves the remaining
  accumulator choices to per-target profiles within the same bounds.
- **A double-float substitute for the binary64 propagator** (§6.4). It carries fewer
  than binary64's bits before its own approximation errors [8], so it cannot meet the
  mixed contract without a separate analysis. Not proposed.
- **Emulated binary64 matrix products on tensor cores** (the Ozaki scheme, offered by
  recent cuBLAS releases on some architectures). It could make a full binary64 radial
  route fast on new GPUs. Its accuracy depends on the operands' exponent range, it is
  not available on Ampere, and binary64 products are not where mixed precision spends
  its time. It is left on the roadmap.
- **A fused batch axis now.** It is new infrastructure, and its value is not yet
  measured. It is measured first (§6.6, Q8).
- **`numba_cpu` single and mixed precision.** Excluded by the maintainers. Native's CPU
  build covers the CPU targets.
- **TF32, BF16 or FP16 anywhere.** Their unit roundoffs ($2^{-11}$ and coarser) are
  above every envelope of §7.3. They are forbidden by SEP 25 §5 and by the probe.

### 11. Phases

All phases are made on one branch and reach `main` in one pull request, opened when the
SEP is set Final. Continuous integration runs the full gate on it. Each phase is
qualified on its own and reports in Part II. The local A40 lane is part of the
definition of done of phases 4 and 5, and so is an AMD lane in each (§12, Q9 and Q10).

**Phase 1 — The mixed setting and its oracle.**
- `Precision`'s mixed form, admission, validation, schemas, the mixed bound, and
  `numpy_ref`'s mixed path with its dtype guards.
- The mixed survival criteria are recorded in Part II. The coherent-case study then
  runs on `numpy_ref` mixed. References may run on binary64 native, as SEP 33 did for
  the envelope case. The study gives a provisional floor and a provisional Adams
  measurement.
- Verifies: no binary64 change; the oracle against binary64 over one step, and the
  independent response witnesses; the linear-part, damping and scale witnesses; SEP 33's
  targeted tests in mixed precision; the dtype guards; the refusals.

**Phase 2 — The native precision policy; single precision on the CPU fibre route.**
- The policy types and view aliases, templated kernels, explicit instantiations,
  strict helpers, `SOLPROP_NATIVE_PRECISIONS`, FFTW single-precision libraries in the
  build and the wheels, wisdom and build-check branches, and the bindings' overloads.
- The shared host-side range helpers. Native single on the fibre capabilities, with the
  §4.3 propagator and §4.5 accumulators. The batch runner.
- Verifies: the `Double` instantiation unchanged, by the native suite and a same-host
  A/B; class R against `numpy_ref` single; the targeted tests; compile time and wheel
  size recorded for Q5; CPU benchmarks at $2^{14}$, $2^{16}$ and the multimode shape.

**Phase 3 — The remaining CPU capabilities and mixed precision on native.**
- Native single on radial and Cartesian free space and envelope-grid plasma.
- Native mixed on every capability of §5.1.
- Verifies: class R against both oracles; the linear-part, damping and scale witnesses on
  native; the single-precision exponential's tests; CPU benchmarks on the suite's
  radial and Cartesian cases; the memory ledgers' representative ratios.

**Phase 4 — GPUs: CUDA on the A40, HIP on an AMD Instinct GPU.**
- The `AMPERE86` preset, the TF32 and `xf32` probes and the subnormal checks.
- Both modes on the CUDA build, and the local A40 lane running native's GPU tests for
  both modes.
- Both modes on the HIP build (`AMD_GFX942`), with the same tests on a rented CDNA 3
  GPU, within the agreed budget, each rental destroyed as soon as its evidence is
  recorded.
- The propagator share on both GPUs and §6.4's remedies if needed. The throughput
  measurements of §6.6 and the per-pass profiles of §7.4 on both.
- Verifies: class R on both devices against the oracles; the confirmation cases of
  §7.3 (single: `ctc`, `argon`, `radial_filament`; mixed: `rdw`, `argon`) on both
  GPUs; the probes' negative tests; benchmarks and per-pass profiles against binary64
  on each GPU and against the CPU hosts; the Q8 rule applied.

**Phase 5 — Noise, Adams and the final records.**
- The noise-seeded study on native mixed (validated in phases 3–4) against binary64
  ensembles, with its criteria recorded first.
- Noise admitted under mixed precision on what it qualifies, with the refusals or limits
  it implies.
- The mixed floor and the Adams admission decided by the rules of §7.3, with the
  defaults set accordingly. The binary32 Adams measurement for Q3.
- The rounding-margin indicator (§5), calibrated by the study.
- On both GPUs, after these admissions: class R of noise-seeded mixed runs under each
  admitted noise model, of Adams if admitted, and of their batches and snapshot round
  trips. On the AMD GPU this is a second rental within the agreed budget.
- The user guide, the worked example and the final benchmark records. The SEP becomes
  Final when both native modes, with every capability admitted by phase 5, agree with
  their oracles within their bounds on CPU, on the A40 and on the AMD GPU.

### 12. Open questions

Decided by the maintainers on 2026-10-25, when this draft was commissioned:
- the two modes, single and mixed;
- noise qualified under mixed precision only;
- SEP 33's routes;
- a fused batch axis measured, not built;
- `numba_cpu` excluded.

Decided by the maintainers on 2026-10-25, on the draft's questions. The revision after
the AI review changed the details of four of these decisions and added two new
questions. Those are marked **revised** or **new**; the maintainers confirmed all six
as written on 2026-10-25 (Part III).

- **Q1 — The mixed floor. Decided:** $10^{-8}$, raised by §7.3's rule. **Revised:** the
  rule also raises the floor when the mixed run needs markedly more accepted or
  rejected steps than binary64 at the same tolerance (R4). *Open:* the outcome (phases 1
  and 5).
- **Q2 — Adams under mixed precision. Decided:** the like-with-like rule of §7.3, with
  `Multistep()` becoming the mixed default if admitted. *Open:* the admission (phase 5).
- **Q3 — Adams under single precision. Decided:** refused on native as on `mlx`. Native
  records its binary32 Adams measurements under the like-with-like rule. If phase 5
  admits Adams under mixed precision by that rule, applying the same rule to single
  precision on `mlx` and native is proposed on the roadmap as one small follow-up.
- **Q4 — Noise criteria. Decided:** $N = 64$; a margin of at least 20 dB as the primary
  criterion; the absolute floors of 1 % (mean spectrum) and 0.02 (coherence); a refusal
  only for a predicate computable from the problem, otherwise a documented limit; an
  indicator recorded in each run's metadata. **Revised** (R7–R9), keeping those numbers:
  - the margin is measured on matched-discrete runs, at every saved plane up to the end
    of the linear regime, against the seed's own response, with the coherent bias
    separated out; this replaces "at 1 % of the length";
  - a source-response witness at identical states is added;
  - the ensemble tolerances use the paired standard error of the difference by
    bootstrap over seeds, with Bonferroni control of the family-wise false-rejection
    rate at 5 % ($z = 3.4$), in place of twice the reference's standard error, which
    would falsely reject about 16 % of correct comparisons;
  - band coherence is defined as `g12` per bin, averaged over the band weighted by the
    reference's mean spectrum;
  - the cases' parameters are frozen in Part II before the first comparison;
  - the metadata indicator is a defined, labelled heuristic (§5).
- **Q5 — The wheels. Decided:** the distributed CPU wheels carry all three modes, unless
  phase 2 measures more than 60 % growth or a size past PyPI's per-file limit (§6.3).
- **Q6 — `auto` under single precision on an Apple-silicon Mac. Decided:** `mlx`, as
  now, falling back to native's CPU build. *Open:* whether phase 4's benchmarks justify a
  size threshold, recorded for the roadmap.
- **Q7 — The mixed propagator on reduced-FP64 GPUs. Decided:** always binary64.
  **Revised** (R6): the 15 % share triggers an investigation and does not impose a
  requirement. Only caching and products are evaluated, each within a stated bound.
  Plain binary64 evaluation remains the fallback. The double-float remedy is withdrawn.
- **Q8 — A fused batch axis. Decided:** the rule of §6.6, measured in samples times
  steps per second on a noise ensemble of small fibre runs (phase 4).
- **Q9 — AMD GPUs. Decided:** qualified in this SEP, not deferred: both modes on the
  HIP build on a rented CDNA 3 GPU, with class R tests, the confirmation cases, the
  probes, benchmarks and per-pass profiles. **Revised** (R11): this happens in phase 4,
  and again after phase 5's admissions (noise, Adams if admitted, batches and
  snapshots), which needs a second rental within the budget agreed before phase 4.
- **Q10 — Continuous integration. Decided:** CI runs both modes on the CPU build; the A40
  lane runs locally and the AMD lane on its rentals, each part of the definition of done
  of phases 4 and 5, with results recorded in Part II.
- **Q11 — The two noise models (new, R9). Decided:** qualify both, `"input"` on every
  noise case and `"modified"` on three. A model is refused under mixed precision on any
  route where it does not pass.
- **Q12 — Binary64 stage values under mixed precision (new, R1). Decided:** every row
  stored in the interaction variable is binary64 under mixed precision, so the mode
  needs no damping restriction (§4.2). The cost is that stage combinations run at the
  binary64 rate, which on the A40 is about 2.2× a binary32 pass. Mixed precision's gain
  there then comes from the transforms, the products and the responses alone. The
  alternative, binary32 rows with SEP 33's damping refusal, is in §10.

### 13. References

1. IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (2019).
   doi:10.1109/IEEESTD.2019.8766229
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   doi:10.1137/1.9780898718027
3. J. C. Schatzman, "Accuracy of the discrete Fourier transform and the fast Fourier
   transform," *SIAM J. Sci. Comput.* **17**, 1150–1166 (1996).
   doi:10.1137/S1064827593247023
4. NVIDIA, *NVIDIA A40 datasheet* (2021): peak FP32 37.4 TFLOPS, FP64 0.585 TFLOPS,
   696 GB/s GDDR6, https://www.nvidia.com/en-us/data-center/a40/ (accessed 2026-10-25).
5. NVIDIA, *cuBLAS library documentation*, CUDA 12.8, "cublasSetMathMode",
   "cublasComputeType_t" and `NVIDIA_TF32_OVERRIDE`,
   https://docs.nvidia.com/cuda/archive/12.8.1/cublas/ (accessed 2026-10-25).
6. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135–1184 (2006). doi:10.1103/RevModPhys.78.1135
7. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301
8. M. Joldes, O. Marty, J.-M. Muller and V. Popescu, "Arithmetic algorithms for
   extended precision using floating-point expansions," *IEEE Trans. Comput.* **65**,
   1197–1210 (2016). doi:10.1109/TC.2015.2441714

## Part II — Implementation record

*One report per phase, appended as each phase is completed (SEP 0 §8).*

### Phase 1 — coherent-case criteria (recorded 2026-10-25, before the study's first comparison)

The criteria are §7.3's, as the maintainers decided them (Q1, Q2). This section fixes the
cases, the runs and the way each metric and rule is computed before any study comparison
runs. The class R tests of §7.2 compared the mixed oracle with binary64 on fixed
schedules while the path was built; none of them is a study comparison, and none set a
criterion.

**Cases.** SEP 33's eleven, with their definitions unchanged (SEP 33 Part II, phase 1
criteria): `spm`, `soliton`, `rdw`, `ctc`, `argon`, `argon_envelope`, `nitrogen`,
`radial_kerr`, `radial_filament`, `cartesian_diffraction` and `cartesian_kerr`.
`argon_envelope` runs its 0.1 m window, as SEP 33's verdict for it did: the oracle's
envelope-grid plasma is evaluated in Python and took 6 h over 0.4 m at `1e-6` in SEP 33,
and `1e-8` takes more steps. Expected, as §7.3 says: every case survives at the floor;
`ctc` is to be measured. If `ctc` does not survive over 0.1 m, it is rerun over SEP 33's
0.05 m and 0.025 m windows to state its limit.

**Runs.** Every run is a `RungeKutta()` run unless stated, on the same grid,
discretisation, plasma rule and quadrature as its reference:

- the reference in binary64 at `rtol = 1e-10`, checked against binary64 at `1e-9`; the
  check passes when every metric below is under a tenth of its envelope;
- candidates in mixed precision on `numpy_ref` at `rtol = 1e-8` (the floor), `1e-7` and
  `1e-6`;
- beside each candidate, binary64 on `numpy_ref` at the same tolerance, which separates
  precision from truncation;
- `argon_envelope`'s binary64 runs (reference, check and the three tolerances) on
  `native`, whose envelope plasma is the same discretisation and quadrature, as SEP 33
  did; its mixed candidates on `numpy_ref`, the only mixed path in this phase;
- every run records its accepted and rejected steps.

**Metrics and envelopes** are SEP 33's, computed by the same code against the reference
at the last plane: the field (10⁻³), the 32 spectral bands above −40 dB of the largest
(1 %), the total energy (10⁻⁴), the peak power or intensity (10⁻³), the ionised fraction
where the reference's exceeds 10⁻⁶ (10⁻³) and the beam width (10⁻³), all relative. The
mixed candidate's difference from binary64 at the same tolerance is reported beside its
error, as the precision's part of it.

**Verdicts.** A case *survives* if its mixed runs are within every envelope at the floor
the rule below settles and at every looser candidate; it *survives with a limit* if
that holds only over a stated shorter length; it *fails* otherwise.

**The floor (Q1).** Starting at `1e-8`, the floor rises to `1e-7` if either holds:

1. on every case, no metric's error at `1e-8` is smaller than at `1e-7` by more than
   25 % (the gain `(e(1e-7) − e(1e-8)) / e(1e-7)` is at most 0.25 for every case and
   metric; a metric with zero error at `1e-7` has gain zero);
2. on any case at `1e-8`, the mixed run takes more than 1.25 times the accepted steps of
   the binary64 run at `1e-8`, or more than 1.5 times its rejected steps; where binary64
   rejects no step, its count is taken as one, so that a single mixed rejection does not
   decide the rule.

If it rises, the same test is applied from `1e-7` to `1e-6`. The phase 1 outcome is
provisional; phase 5 decides the floor with the noise study (Q1).

**Adams (Q2), provisional.** `rdw` and `argon` with `Multistep()` through the study's
private admission, in mixed precision and in binary64 at each of `1e-8`, `1e-7` and
`1e-6`. Adams is admissible under mixed precision if, on both cases and at each
tolerance, the mixed run is within every envelope and its error is at most twice
binary64 Adams's at the same tolerance in every metric (an error below 10⁻¹² counts as
10⁻¹²). Phase 5 decides the admission and the default; phase 1 records the measurement.

**Amendment (2026-10-25, before any comparison was scored).** The maintainers moved
`argon_envelope`'s mixed candidates to phase 3: they run on native mixed once native is
qualified against the oracle, since the oracle's envelope-grid plasma, evaluated in
Python, takes hours per run. Its binary64 runs on `native` are made in phase 1 as above;
its verdict, and its part in the floor rule, are recorded in phase 3. The floor rule in
phase 1 is applied over the other ten cases and is provisional as stated.

**Where the record lives.** Accuracy runs only, one thread per run, on an Apple M5 Max;
no timing here is a benchmark. The harness and raw results are kept with the project's
development records (SEP 30's rule); the verdicts and limits are recorded here and, in
phase 5, on the manual page.

### Phase 1 — report (2026-10-25)

**Delivered.**

- `solprop.translate.precision`: `Precision` admits a binary64 state beside a binary32
  source (`MIXED`, `name == "mixed"`, the properties `mixed`, `reduced` and
  `source_complex_dtype`); `PrecisionName` takes `"mixed"`; the per-mode backend lists
  `REDUCED_PRECISION_BACKENDS` (replacing `SINGLE_PRECISION_BACKENDS`), the floor
  `MIXED_RTOL_FLOOR = 1e-8` and `rtol_floor`; `admit_precision` applies each mode's
  backends, Adams rule and floor; `study_active` beside `study_admission`.
- `KernelProblem` (kernel schema 37): `_check_precision` refuses SEP 33's routes and
  responses under both reduced modes, noise under single, and noise under mixed outside
  the study's admission; `_check_single_exponent` runs under single only. The output
  schema is 26. `numba_cpu`, `native` and `mlx` refuse `"mixed"` by name; the native
  build record reports schema 37.
- Interfaces: `precision` takes `"mixed"` on `SimulationSettings` and every keyword
  interface; each reduced mode's default `rtol` is the larger of the interface's
  binary64 default and the floor; `RungeKutta()` is the default; `backend="auto"`
  resolves `"single"` to `mlx`, else to `native` when its build compiles it, and
  `"mixed"` to `native` when its build compiles it (`registry.native_precisions`), and
  is refused otherwise. Lockstep batches admit members of one precision.
- `backends/numpy_ref/mixed.py`: `NumpyMixedBackend` (§6.2, below); the memory rows
  `mixed_precision_rows`.
- `tests/bounds.py`: the mixed bound of §7.1 (`source_pass_bound`, `rate_condition`,
  `mixed_step_bound`, `mixed_run_bound`). `tests/backends/test_mixed_precision.py`
  (64 tests) and `tests/precision_cases.py`, the cases both reduced modes' oracle tests
  now share.
- Normative pages: architecture.md §1, conventions.md §6, kernel-contract.md,
  testing.md §3, backends.md (§2, §9, §11); the output guide's schema note; the
  changelog.

**How the operation order is implemented on the oracle.** `NumpyMixedBackend`
subclasses `NumpySingleBackend`: it keeps the single path's narrowed source tables,
combined coefficients, binary32 passes and host evaluations (SEP 33's oracle
exceptions included), and restores the binary64 reference for every hook that touches
the state: `set_initial` (noise drawn and held in binary64), `_propagator`,
`_error_norm`, `_finish`, `_trial_step`, `_adams_adopt`, `stats` and the observed
field. Its `_nonlinear` is §4.2's steps 2–4 around the reference's step 1 (the stage
input with the modified noise added, in `passes.stage_input`): `fl32(y / s)`; the
single path's source called with a unit inverse propagator, which returns `N(s y) / s`
with the in-stage collars among its terms; `exp(-dPhi) * (s * wide(.))` in binary64.
So every stored row (stages, low-storage registers, Adams history, dense-output rows,
snapshots) is complex128, the low-storage and Adams marches and the lockstep batch need
no change, and the split-step collar keeps its binary64 transforms. The dtype guards
check the stage input and every stored row complex128 and the source's value
complex64; the adversarial tests put a binary64 prefactor inside the source and a
binary32 stage value in a stored row, and each is refused by name.

**Class R record** (fixed schedules, the mixed oracle against binary64; observed
departure over §7.1's bound, every bound below a tenth of the field envelope):

| case | departure | bound | fraction |
|---|---|---|---|
| fibre, Kerr, Raman, shock: one 5 mm step | 7.1e-9 | 2.8e-6 | 2.6e-3 |
| the same, ten steps | 3.0e-8 | 2.8e-5 | 1.1e-3 |
| low-storage march | 2.8e-8 | 1.2e-5 | 2.3e-3 |
| multimode capillary (four modes) | 1.3e-8 | 4.0e-6 | 3.1e-3 |
| envelope capillary | 5.7e-8 | 2.1e-5 | 2.7e-3 |
| radial, Cartesian, line Kerr beams | 4.7e-8, 5.8e-8, 2.2e-11 | 7.3e-5, 6.8e-5, 7.4e-9 | 6.5e-4–2.9e-3 |
| field-grid plasma: argon, nitrogen (with Raman) over 0.25 mm | 3.9e-8, 2.9e-8 | 3.9e-5, 3.5e-5 | 1.0e-3, 8.4e-4 |
| radial filament | 2.5e-9 | 1.9e-6 | 1.3e-3 |
| in-stage temporal collar, no source | 1.7e-9 | 1.1e-5 | 1.6e-4 |
| damped bin, `Re dPhi` = 150 per step | 2.1e-8 | 9.3e-6 | 2.2e-3 |

The source-free runs, with no collar or a split-step one, on the fibre and Cartesian
routes, agree within binary64 bounds. Each stage-level source evaluation is within
`kappa u32 + E_pass` of binary64's at the launch's scale and at `s = 2**30`. The
witnesses hold: dispersion-free SPM against its closed form and photon-number
conservation, in both reduced modes; the damped bin's stored stage values exceed
binary32's range, as §4.2 says they may. A non-finite source stops the adaptive,
replayed and low-storage marches; an Adams snapshot round trip is exact.

**Coherent-case study** (criteria above; every reference passed its `1e-9` check;
relative errors at the last plane against the reference; *precision* is the mixed run
against binary64 at the same tolerance):

| case | mixed `1e-8`: field | `1e-7` | `1e-6` | precision part (field, every tolerance) | largest other metric at `1e-7` | verdict |
|---|---|---|---|---|---|---|
| `spm` | 8.2e-6 | 9.3e-6 | 7.8e-6 | 7.9e-6–9.3e-6 | 8.3e-5 bands | survives |
| `soliton` | 1.8e-6 | 1.7e-6 | 1.9e-6 | 1.7e-6–1.8e-6 | 5.6e-7 bands | survives |
| `rdw` | 6.5e-7 | 6.1e-7 | 2.3e-6 | 6.5e-7–1.3e-6 | 2.6e-6 bands | survives |
| `ctc`, 0.1 m | 1.5e-3 | 8.9e-4 | 1.5e-3 | 8.9e-4–1.5e-3 | 9.7e-5 peak | survives with a limit: 0.05 m (field 1.9e-4; 0.025 m 1.9e-6) |
| `argon` | 1.5e-5 | 1.6e-5 | 2.1e-4 | 1.5e-5–6.0e-5 | 1.6e-5 ionised | survives |
| `nitrogen` | 6.7e-7 | 7.1e-7 | 7.8e-7 | 6.7e-7–7.3e-7 | 2.9e-5 bands | survives |
| `radial_kerr` | 7.6e-7 | 5.4e-7 | 8.2e-6 | 6.2e-7–8.2e-7 | 9.0e-6 bands | survives |
| `radial_filament` | 2.1e-7 | 5.2e-7 | 2.3e-6 | 2.2e-7–4.5e-7 | 1.3e-6 peak | survives |
| `cartesian_diffraction` | 0 | 0 | 0 | 0 | 0 | survives |
| `cartesian_kerr` | 2.6e-6 | 2.6e-6 | 2.0e-6 | 2.6e-6–2.8e-6 | 1.0e-5 bands | survives |
| `argon_envelope` | phase 3 | | | | | |

- *The floor* is provisionally **`1e-7`**. The step clause of Q1 raises it from `1e-8`:
  at `1e-8` `ctc` takes 411 accepted steps against binary64's 276 (1.49×; 351 against
  219 over 0.05 m), and `rdw` rejects two steps against one. The gain clause alone would
  not have raised it (`spm`, `soliton`, `rdw` and `radial_filament` gain 56–97 % on some
  metric). From `1e-7` to `1e-6` neither clause holds: every case is within 1.11× the
  accepted steps and 1.0× the rejections of binary64 at `1e-7`.
- *The precision part is a coherent floor.* The mixed run's difference from binary64 at
  the same tolerance is the same at every candidate (above): §4.2's coherent term, not
  the controller's. On `spm` it is 8.2e-6, about four times `u32 Phi_NL` (10π rad), and
  it equals single precision's error there (SEP 33: 7.9e-6), since `spm`'s error is the
  source's rounding, not the state's or the propagator's. Where the state's rounding or
  the propagator's drift dominated single precision, mixed precision is much better:
  `soliton` 1.8e-6 against 2.7e-5, `argon` 1.5e-5 against 1.8e-4, `radial_kerr` 7.6e-7
  against 8.5e-6. Tolerances below `1e-7` therefore buy no accuracy on these cases.
- *`ctc` keeps SEP 33's limit, 0.05 m.* Its error is precision throughout (binary64 at
  `1e-8` is 2.1e-8 from the reference): the source's binary32 rounding seeds the empty
  Stokes bins, which noise-free stimulated Raman scattering amplifies, as single
  precision's state rounding did. §7.3 expected the limit to lengthen; at these
  windows it does not (0.05 m: 1.9e-4 mixed at `1e-7`, against single's 4.3e-5;
  0.025 m: 1.9e-6 against 6.6e-6). The seed is one rounding of the source per step in
  either mode, of similar size, and its amplification is the case's.
- *Adams (Q2), provisional: not admissible by the rule as fixed.* On `rdw` and `argon`
  every mixed Adams run is within every envelope, and its errors match mixed
  Runge–Kutta's at each tolerance (`rdw` `1e-8`: field 6.2e-7 against 6.5e-7; `argon`
  `1e-7`: 1.2e-5 against 1.6e-5). The rule compares it with binary64 Adams, whose
  errors at `1e-8` and `1e-7` are below mixed precision's coherent floor (`rdw` field
  1.9e-8; `argon` 2.1e-6), so the ratio reaches 13–590 at those tolerances and 3.5–74 at
  `1e-6`. The rule measures the precision floor, which mixed Runge–Kutta would fail by
  the same ratios, rather than Adams. At `1e-8` mixed Adams also takes 1.58× binary64's
  accepted steps on `rdw` (5165 against 3264), 1.02× at `1e-7`. Phase 5 decides (open,
  below).

**Findings for later phases.**

- *For the maintainers before phase 5 (Q2; decided below):* the like-with-like Adams rule, as fixed,
  cannot admit Adams at tolerances where binary64 is more accurate than mixed
  precision's coherent floor, whatever the integrator. A rule comparing mixed Adams
  with mixed Runge–Kutta at the same tolerance (SEP 33's form), or one asking only for
  the envelopes at the settled floor, would admit it on this record. Part I is frozen;
  this is put to the maintainers for phase 5, where the admission is decided.
- *The plasma's condition number.* §7.1 reads `kappa` over "the samples' field range".
  The tests take it over the fields the binary64 reference evaluated a rate at where
  the rate is at least `u32` of its largest, since smaller rates contribute below
  binary32's rounding of the largest term to the normwise bound; this gives 17
  (argon) and 19 (filament). Over every field the logarithmic slope grows without
  limit as the field falls, which would make every plasma bound uninformative.
- *Informative windows.* The transport factor `(1 + kappa phi)**s` makes the ionising
  capillary's 0.25 mm steps (`phi` = 0.5, `kappa` = 17) and one 5 cm fibre step
  (`phi` = 0.55) uninformative; their class R tests run on 0.25 mm in 64 steps and on
  one 5 mm step. The default capillary collar in the in-stage scheme (6.3e4 1/m, an
  absorber exponent of 2.6 per step at the scheme's stability limit) is uninformative
  over any window; the class R test uses a 126 1/m collar, and the class T study
  qualifies the strong one. In-stage collars enter the bound by the step's absorber
  exponent `dz rate / 2` (§4.2), since their binary32 error is the transform's normwise
  rounding times the rate, not a fraction of the term.
- *The state's units.* The launch scale is `2**1`–`2**16` on these cases (the state is
  not in V/m), so the non-unit-scale witness sets `s = 2**30` on the oracle.
- *Schema numbers.* SEP 35's open pull request also moves the kernel schema from 36;
  whichever merges second takes the next number.
- *Memory.* `numpy_ref`'s constructed ledger counts binary64 rows in every precision,
  as SEP 33 left it; `estimate_memory` counts the mixed rows. Phase 3 records the
  native ledgers' ratios.

**Deviations.** The envelope-plasma case's mixed candidates moved to phase 3 (the
amendment above). The class R tests run on the shorter windows, the forced scale and
the weaker in-stage collar above, as §7.1 provides for uninformative bounds; the
Cartesian split-step source-free case takes a grid without a spatial collar, which is
always an in-stage term. The criteria section fixed two operational details §7.3 leaves
open: a binary64 run with no rejection counts as one in the step clause, and an error
below 10⁻¹² counts as 10⁻¹² in the Adams ratio. None changes Part I.

**Verification.** The new and changed modules pass: `test_mixed_precision.py` (64),
`test_single_precision.py`, `test_precision.py`, the registry, memory, batch,
`numpy_ref`, record-stability, schema and display modules and the schema pins
(1557 in one run, after two doctests' schema numbers), native's API-contract tests
and the native backend test that reads the schema, `tests/test_seps.py`, and the docs
build. CI runs the full gate on the SEP's pull request.

### Maintainer decision after phase 1 (2026-10-25)

John Travers, on the phase 1 report: for Q2, phase 5 compares mixed Adams with mixed
Runge–Kutta at the same tolerance (SEP 33's form), in place of binary64 Adams, keeping
the rest of §7.3's rule (within every envelope, at most twice in every metric, at each
candidate tolerance on `rdw` and `argon`). Phase 2 proceeds.

### Phase 2 — report (2026-10-26)

**Delivered.**

- `native/include/solprop_native/precision.hpp`: the policies `Double`, `Single` and
  `Mixed` (§6.3) with role-typed views per policy (`StateView*`, `SourceView*`,
  `TableView*`, `ReductionView*`, `PropagatorView*`, 1–3-D, real and complex) replacing
  the per-file aliases of `core.hpp`, `general.cpp`, `core.cpp`, `dispersive.cpp`,
  `general_modal.cpp` and `coupled_flow.cpp`. A sixth role, **argument**, is binary64
  under every policy and not a policy parameter: the propagator's argument, the
  operator terms, the Adams exponents, the tableau and per-stage scalars, the Raman rates
  and tensor weights, the spatial and flux coefficients and the z tables (§4.3, §4.5).
  Each buffer's role is listed in comment tables beside the policies and the workspace.
  The binary64 seam types moved there from `core.hpp`.
- `Workspace<Policy>` (the single-column fibre route) and `GeneralWorkspace<Policy>`,
  `GeneralBatchWorkspace<Policy>` and the envelope and field-current streams are class
  templates, their definitions in `general_impl.hpp` and one translation unit per policy
  (`general.cpp`, `general_single.cpp`; `general_mixed.cpp` in phase 3). The routes both
  reduced modes refuse (general modal, coupled flow, dispersive Kerr, far-field
  statistics, the slab kind) keep binary64 types and exist in the `Double` instantiation
  only; `require_policy_route<Policy>` refuses them by name otherwise.
- `fp.hpp`'s strict helpers are templates on the real type; a binary32 helper inlined
  into a protected block stays strict (tested).
- `SOLPROP_NATIVE_PRECISIONS` (default `double;single` on CPU in this phase; device
  builds default to `double` until phase 4, with
  `SOLPROP_NATIVE_DEVICE_REDUCED_COMPILE_ONLY` to compile the reduced translation units
  while the runtime refuses them). A list without `double` is refused.
- The §4.3 binary32 exponential (`propagator.hpp`): `exp(x) = 2^m e^r` with a two-part
  `ln 2` (head of 29 significant bits), the phase reduced by a three-part `2π` (heads of
  25 bits) for `|θ| <= 2^30` and by the binary64 library above, then `expf`, `sinf` and
  `cosf`; the inverse shares the phasor with its own modulus. (Replaced in phase 4 by the binary64
  functions rounded once, after the A40's biased `sinf` and `cosf`; see the phase 4 report.) The constants were checked
  against 300-bit references (the `ln 2` split within 2e-27, the `2π` split within
  5e-32). C++ tests check the 10 `u32` bound over both ranges' ends, the −80.000003
  witness, phases past `2^30`, the inverse and the modulus drift over 10⁴ steps.
- Single precision on the CPU fibre route (§5.1): mode-averaged and multimode, field
  and envelope grids, Kerr, Raman (scalar and tensor), the field-grid plasma with SEP 33's
  stable populations, both absorber schemes, Runge–Kutta and the low-storage march,
  lockstep batches. The state is held at the launch scale; the tables are narrowed on
  the host by `solprop.backends.reduced` (the module phase 2 moved SEP 33's host range
  work into, shared by `numpy_ref`, `mlx` and `native`), and the per-stage Kerr and Raman
  coefficients are bit-identical to `combine(...).binary32()` (class P over 452 cases,
  `_core.combined_binary32`); out-of-range coefficients and tables are refused with the
  oracle's messages. Kokkos-FFT plans single-precision FFTW for binary32 views, the
  wisdom seeder has its `fftwf_` branch, and the optional Accelerate Raman provider stays
  binary64's (binary32 views take Kokkos-FFT). `KokkosBlas::gemm` runs binary32;
  `native_order_gemm` stays binary64-only.
- Python: `native` admits `"single"` when its build compiled it (`BuildInfo.precisions`
  replaces `precision`; `registry.native_precisions` reads it) and refuses it by name,
  saying how to build one, otherwise; `REDUCED_PRECISION_BACKENDS["single"]` includes
  `native`; the ledger counts element sizes.

**§4.5 choices kept in binary64 under single precision on native:** the argument role
(above); norms and statistics' reductions; and the plasma-integration diagnostics
(`plasma_integration_diagnostics`), whose band scan uses binary64 rows and taps, reads
the physical field and is allocated only when the diagnostics are requested — binary32
rows would put a floor near 10⁻⁷ on the work-defect diagnostic. The statistics still
synthesise the field in binary32: the electron density and undershoot carry about 10⁻⁶
of transform rounding against the oracle's binary64 synthesis, within the plasma
bounds.

**Verification.**

- *The `Double` instantiation is unchanged.* The native suites pass unchanged. The
  SEP 30 core set A/B on a 2× EPYC 7763 node (16 threads, interleaved, four rounds):
  geometric mean 0.9997, every field identical to the bit but `kerr_beam_384`'s
  (5×10⁻¹⁹). One case, `radial_filament_256` with Adams, read 1.14 there and 1.31–1.42 in
  a rerun on another node, with each arm's rounds varying 0.5–2.4× on an otherwise idle
  node; a single-thread rerun with both builds on node-local disk, six rounds, gives
  0.995 (Adams) and 1.008 (Runge–Kutta) with stable rounds. The multi-thread variance is
  the cluster filesystem's and the node's, not the build's.
- *Tests* (a 2× AMD EPYC 7763 cluster node, GCC 16.2, the `LOCAL` profile, `double;single`): the 93
  solprop test modules that run the native backend, 3282 passed and 13 skipped; native's
  own Python tests, 389 passed and 6 skipped; the C++ tests (with the new precision, fp
  and propagator tests) pass. `tests/backends/test_native_single_precision.py` (49):
  class R native single against `numpy_ref` single on fixed schedules for every phase-2
  capability, bounds summing both sides; the class P power-of-two scaling, the vanished
  molecular kernel, plasma populations, the propagator drift, failure injection on every
  march, resume across backends in the same precision (class T) and refusal across
  precisions, batches, memory and table refusals.
- *Devices.* The A40 (CUDA 12.8.1, `AMPERE86`, GCC 14.4 host, `-Werror`) builds the
  templated binary64 code and passes native's 31 GPU tests and 142 of 143 solprop tests
  (below); with the compile-only flag it also compiles the single and mixed translation
  units. nvcc's front end found what the CPU compilers did not: an out-of-class template
  member definition whose return type it mis-compares (fixed by trailing return types
  throughout), and device lambdas that first capture a variable inside an `if constexpr`
  branch, including through a discarded branch (fixed by naming the captures before the
  branch); a lint for the latter now runs beside a check that forces every device branch
  on for a host syntax check.
- *Q5, size and compile time* (the same node, GCC 16, `LOCAL`): adding `single` grows the
  extension from 7.35 MB to 12.01 MB (+63 %), compressed in the wheel from 2.36 MB to
  3.54 MB (+50 %); the build takes 98 s against 79 s on 16 cores. The extension is 17 %
  of the released Linux wheel (13.99 MB, with OpenBLAS), 58 % of the macOS wheel
  (3.69 MB) and 86 % of the Windows wheel (4.85 MB) at main's head, so `single` projects
  to +8 %, +29 % and +43 %. Phase 3 measures `mixed` beside it; Q5's 60 % rule is decided
  there.

**Findings for later phases.**

- A process's first binary32 Raman run can differ from later identical runs by about
  4×10⁻⁹ relative: its Kokkos-FFT plan is made before FFTW's process wisdom holds the
  measured plans later runs reuse. Under SEP 25 there is no bitwise requirement; the
  bitwise class P tests warm up first.
- `fft_plan_heap_growth` in the ledger depends on what the process planned before; the
  ledger comparisons leave it out.
- `test_the_march_agrees_across_thread_counts` asserts the host thread count and is not
  guarded for device builds; it fails on any CUDA build (not this SEP's change).
- Native single against `mlx` (a consistency check, not a gate) is not yet run.
- The build check's `cufftExecC2C` probe and the device forms of the reduced modes are
  phase 4's.

**Deviations.** None from Part I. The `argument` role is an implementation detail of
§4.3 and §4.5 (binary64 in every mode). The device builds' default stays `double` until
phase 4 qualifies the reduced modes there.

### Phase 3 — report (2026-10-26)

**Delivered.** Native single and mixed precision run every CPU capability of §5.1.

- *Single* on the radial, Cartesian and line routes and the envelope-grid plasma.
- *Mixed* on every route: the `Mixed` policy (`general_mixed.cpp`), the default CPU
  `SOLPROP_NATIVE_PRECISIONS` now `double;single;mixed`, and `native` in
  `REDUCED_PRECISION_BACKENDS["mixed"]`. The state is held physical in binary64, as the
  oracle holds it; only the source works at the launch scale, narrowed once as
  `fl32(y · pad / s)` and widened exactly times `s` before the binary64 inverse
  propagator (§4.2), so the Adams rows, snapshots, noise and tolerances cross the seam
  in physical units and a snapshot round trip is class P.
- The split-step collar on free space under mixed precision transforms the binary64
  state in its own binary64 transverse volume with binary64 plans, allocated at planning
  and counted in the ledger and `estimate_memory` (`mixed_split_transverse`).
- The envelope-grid plasma (§6.5): the quadrature in binary32 (angles, `cos`/`sin`,
  denominator, field), each node's rate a binary64 table lookup rounded once, the level
  sums in binary64 with the 2×10⁻⁶ budget and 10⁻³⁰ floor, and SEP 33's MLX lessons (cut
  angles formed in binary64, the first panel at the ceiling rate, the panel past the low
  cut skipped, the field clamped between the cuts, refinement failures raised as
  `ConvergenceError`). The currents are divided by `s` and rounded once.
- Adams under mixed precision is the binary64 machinery with the binary32 source and a
  binary64 history, reached only through the study's admission until phase 5; noise
  under mixed precision is added in binary64 before narrowing, also behind the study's
  admission.
- Refused under both reduced modes, by name: the slab kind and the far-field
  statistics (the far-field pass is not built under them).

**§4.5 choices kept in binary64:** the spatial statistics read the physical state
widened to binary64 (the radial route through a binary64 copy of the synthesis matrix,
free space through the binary64 transverse volume; the on-axis spectrum on free grids
then comes from the spatial pass); the envelope plasma's ellipse decomposition from the
exactly widened field and its populations, once per sample.

**Verification.**

- *Tests* (the same cluster node, `double;single;mixed`): the native test modules and the
  reduced-precision oracle modules, 534 passed; native's own tests, 389 passed and 6
  skipped. `tests/backends/test_native_mixed_precision.py` (53): class R against
  `numpy_ref` mixed with §7.1's bound summed over both sides, on the fibre (one and ten
  steps, Kerr, taper, low-storage, four modes, envelope, harmonic), the field-grid plasma
  (band and trapezoid rules, Raman, a mixture), tensor Raman, the radial, Cartesian and
  line routes with and without collars, both filaments; the linear-part witnesses
  (source-free and split-step within binary64 bounds, in-stage collars within the mixed
  bound) on the fibre and Cartesian routes; the damped bin (`Re dPhi` = 150) and the
  `s = 2**30` witness; class P scaling, reload and Adams snapshots; Adams and the
  modified noise under the study's admission; failure injection on every march;
  batches; ledgers; the envelope plasma within its class T envelope. The single module
  adds the spatial routes and the envelope plasma.
- *The A40* builds all three policies (443 s) and passes native's 31 GPU tests; the
  reduced modes stay refused on the device until phase 4.
- *The deferred coherent-case study* (`argon_envelope` over 0.1 m, the phase 1
  amendment), on native in both precisions, every run on one node: the reference passed
  its check (field 6.8×10⁻¹⁰). Mixed precision is within every envelope at `1e-8`,
  `1e-7` and `1e-6` (field 1.6×10⁻⁶, 1.4×10⁻⁶, 2.4×10⁻⁶; ionised 1.6×10⁻⁶ at `1e-8`):
  **survives**. Its precision part is flat across the tolerances (field 1.4–1.6×10⁻⁶),
  as on every other case. It takes 1.28 times binary64's accepted steps at `1e-8` and
  1.06 times at `1e-7`, so it joins `ctc` in the step clause and confirms the
  provisional floor of `1e-7` over all eleven cases.
- *Memory* (native ledgers, totals without the planners' heap growth; the estimate's
  ratios beside them):

  | case | double | single / double | mixed / double | estimate, mixed / double |
  |---|---|---|---|---|
  | fibre, 1024 bins | 0.73 MB | 0.71 | 0.87 | 0.90 |
  | capillary with plasma | 7.5 MB | 0.53 | 0.55 | 0.78 |
  | four-mode capillary | 11.0 MB | 0.55 | 0.60 | 0.81 |
  | envelope plasma | 1.2 MB | 0.64 | 0.67 | 0.84 |
  | radial Kerr, 32 nodes | 2.0 MB | 0.62 | 0.86 | 0.93 |
  | radial filament | 9.2 MB | 0.59 | 0.70 | 0.89 |
  | Cartesian 16 × 16 | 12.1 MB | 0.56 | 0.85 | 0.86 |
  | Cartesian, split step | 13.1 MB | 0.55 | 0.90 | 0.92 |
  | line | 0.8 MB | 0.57 | 0.86 | 0.86 |

  These test cases are small; the source's share, which mixed precision halves, grows
  with the transverse grid.
- *CPU time per step* (2× AMD EPYC 7763, one NUMA domain, the three precisions
  interleaved in one process after a warm-up, three windows each, median; departure from
  binary64 at the end of the window in brackets; the node shared, so only the ratios are
  meaningful):

  | case | threads | binary64, ms | single | mixed |
  |---|---|---|---|---|
  | fibre, 2^14 bins | 1 | 9.41 | 1.31× (6.0×10⁻⁷) | 1.36× (3.4×10⁻⁸) |
  | fibre, 2^16 bins | 1 | 39.3 | 1.29× | 1.30× |
  | four-mode capillary | 1 | 4.46 | 1.37× (1.1×10⁻⁶) | 1.39× (1.6×10⁻⁸) |
  | radial Kerr, 32 nodes | 1 | 1.79 | 1.10× | 1.15× |
  | Cartesian 16 × 16 | 1 | 16.7 | 1.04× | 1.01× |
  | fibre, 2^14 bins | 16 | 6.72 | 1.45× | 1.42× |
  | fibre, 2^16 bins | 16 | 23.9 | 1.34× | 1.26× |
  | four-mode capillary | 16 | 5.70 | 1.47× | 1.56× |
  | radial Kerr, 32 nodes | 16 | 1.80 | 1.12× | 1.17× |
  | Cartesian 16 × 16 | 16 | 8.75 | 1.74× | 1.54× |

  The SEP 30 suite's records replace these in the final benchmarks (§7.4).
- *Q5.* With the three policies the macOS extension grows from 6.26 MB to 13.66 MB
  (+118 %; +60 % with single alone); the compile wall time is unchanged at 13–14 s on
  eight cores, since each reduced translation unit compiles beside `general.cpp`.
  Applied to main's released wheels (the extension 17 % of the Linux wheel, 58 % of the
  macOS one, 86 % of the Windows one), all three policies project to about +20 %, +68 %
  and +101 %: past Q5's 60 % on macOS and Windows, which would make the reduced modes an
  option of source builds there. The wheels themselves are to be built and measured
  before the rule is applied; the maintainers decide.

**Findings for later phases.** The envelope plasma on the spatial routes runs the same
code but has no test of its own yet. The oracle-comparison bounds charge native's
binary64 running integrals at the oracle's binary32 rate, which is conservative.

**Deviations.** None from Part I. The envelope-plasma case of the coherent study ran
here, as the phase 1 amendment provides; its binary64 reference ran on `native` on the
same node, replacing the phase 1 records made elsewhere.

### Phase 4 — report (2026-10-26; the NVIDIA half)

**Delivered on CUDA.** Both reduced modes run on the CUDA build (`AMPERE86`, the
`cuda-ampere` preset), qualified on an NVIDIA A40.

- The device builds instantiate all three policies. cuBLAS runs in its pedantic math
  mode, set on Kokkos Kernels' handle, and rocBLAS in its default mode. A GEMM probe
  (threshold $2^{-17}$) and a transform probe run at construction, and the build record
  carries `fft_binary32_library` and `gemm_probe_error`. A CMake guard refuses
  flush-to-zero and fast-math flags.
- Binary32 sources get a device team form: one team per node, with threads over the
  samples. It covers the pointwise Kerr, harmonic and Raman drive, and the trapezoid
  plasma, whose two running integrals are team scans in binary64 over binary32 samples
  (§4.2's reduction role). Until then they ran the host column on the device, one thread
  per node marching over every sample. That made A40 single and mixed throughput flat
  at about $1.7 \times 10^{6}$ samples·steps/s from $2^{13}$ to $2^{18}$ samples, a
  tenth of binary64's. Multiple-channel mixtures stay on the host column, as under
  binary64.
- The band-rule plasma of the reduced policies takes the team column everywhere.

**Verification on the A40.**

- *Tests:* ctest passes; native's GPU tests 31; the API contract 45; the reduced
  GPU lane (`test_native_reduced_gpu.py`, class R against the oracles on the device)
  123; the reduced arithmetic module 25, including the TF32 negative test (cuBLAS in
  TF32 mode fails the GEMM probe, 2 tests) and a subnormal surviving a device pass.
- *The confirmation cases of §7.3*, against their binary64 references, judged by SEP
  33's envelopes:

  | case | mode | field | bands | energy | peak | ionised / width | verdict |
  |---|---|---|---|---|---|---|---|
  | `ctc`, 0.05 m | single | 6.8e-4 | 3.4e-6 | 9.6e-7 | 3.7e-6 | — | within |
  | `argon` | single | 1.8e-4 | 2.7e-4 | 4.0e-6 | 9.7e-5 | 5.4e-5 | within |
  | `radial_filament` | single | 2.4e-6 | 1.5e-6 | 4.9e-8 | 1.2e-5 | 1.8e-6 | within |
  | `rdw` | mixed | 9.2e-7 | 1.2e-5 | 2.1e-7 | 3.8e-7 | 5.4e-6 | within |
  | `argon` | mixed | 3.5e-5 | 2.6e-5 | 1.2e-6 | 3.1e-6 | 3.4e-5 | within |

  Binary64 on the A40 at the same tolerance is within too (`rdw` 7.9e-8, `argon`
  1.1e-5 field).
- *Throughput* (§6.6; the soliton fibre with Raman, 64 fixed steps, samples·steps/s;
  the lockstep batch of 8; concurrent processes on one GPU):

  | samples | mode | solo | batch × 8 | best of 2, 4, 8 processes |
  |---|---|---|---|---|
  | $2^{14}$ | single | 2.7e7 | 4.2e7 | 2.3e7 |
  | $2^{14}$ | mixed | 2.6e7 | 4.1e7 | 2.2e7 |
  | $2^{14}$ | binary64 | 8.9e6 | 1.0e7 | 8.6e6 |
  | $2^{15}$ | single | 3.8e7 | 5.9e7 | 3.4e7 |
  | $2^{15}$ | mixed | 3.4e7 | 5.3e7 | 3.1e7 |
  | $2^{15}$ | binary64 | 1.2e7 | 1.3e7 | 1.1e7 |
  | $2^{18}$ | single | 6.4e7 | 1.1e8 | 6.1e7 |
  | $2^{18}$ | mixed | 5.6e7 | 9.6e7 | 5.2e7 |
  | $2^{18}$ | binary64 | 1.6e7 | 1.8e7 | 1.5e7 |

  At $2^{18}$ single is 4.0 times binary64's solo throughput and mixed 3.5 times.
  Concurrent processes never beat the lockstep batch, so the guide advises batches for
  ensembles on one GPU.
- *Per-pass profile* (nsys, $2^{16}$ samples, 16 steps, share of GPU time):

  | pass | single | mixed | binary64 |
  |---|---|---|---|
  | accepted update and error norm | 36.6 % (407 µs) | 38.7 % (480 µs) | 10.9 % (482 µs) |
  | response | 27.1 % (50 µs) | 24.4 % (50 µs) | 66.3 % (483 µs) |
  | stage entry (propagator and stage combination) | 19.7 % (36 µs) | 20.9 % (43 µs) | 6.1 % (44 µs) |
  | transforms | 8.7 % (4.0 µs) | 8.0 % (4.1 µs) | 14.3 % (26 µs) |
  | projection, synthesis, scaling, finish | 6.9 % | 6.6 % | 2.0 % |

  A single-node fibre gives the accepted update and the binary64 response one team, so
  one of the A40's 84 multiprocessors does their binary64 arithmetic. That costs about
  0.48 ms per call in every mode. It is the largest remaining cost of the reduced
  modes, and of binary64's response; it predates this SEP.

PENDING-PROPAGATOR

**The Q8 rule** (§6.6), on mixed precision, the noise study's mode. The best way at
$2^{14}$ and $2^{15}$ samples is the lockstep batch: 4.1×10⁷ and 5.3×10⁷
samples·steps/s. Against the batch at $2^{18}$ (9.6×10⁷) these are 0.43 and 0.56;
against a solo run at $2^{18}$ (5.6×10⁷), 0.73 and 0.95. The rule's "the same case at
$2^{18}$" does not say which, so $2^{14}$ falls short of half under the first reading
and passes under the second, and $2^{15}$ passes under both. These runs carry no noise
model; a noise ensemble adds one binary64 addition per sample at the entrance. The
reading is the maintainers' to decide.

PENDING-CTC

**Not done: the AMD half.** The HIP build compiles all three policies, but no CDNA 3 GPU
has been rented, so §7.3's confirmations, the probes' negative tests, the throughput and
the profiles on an AMD GPU wait on a budget agreed with the maintainers (Q9).

**Deviations.**

- *The binary32 propagator (§4.3).* The range-reduced binary32 functions meet the
  bound but are biased. On the A40, CUDA's `sinf` and `cosf` make the squared modulus
  $\cos^2\rho + \sin^2\rho$ average $1 + 0.30\,u_{32}$ over the phases, where the CPU
  libraries' and the rounded binary64 values average 1. The propagator multiplies the
  state once per step, so that bias is an energy drift linear in the step count, and
  Kerr phase turns it into a field error that grows faster than the step count. In the
  `argon` case (2500 steps) native single on the A40 reached a field error of
  $1.6 \times 10^{-3}$ and an ionised-fraction error of $1.4 \times 10^{-3}$, outside
  the envelopes, against $1.3 \times 10^{-4}$ and $5 \times 10^{-5}$ on the CPU. The CPU
  form drifted too, by about $+0.25\,u_{32}$ per step against the oracle's
  $-0.24\,u_{32}$, and its Kerr-only error after 1000 steps was four times the
  oracle's. Native single therefore evaluates $e^{x}$, $\cos\theta$ and $\sin\theta$
  with the binary64 library and rounds each once, as the oracle does: the relative
  error is at most $3\,u_{32}$ and the rounding unbiased. On the CPU the drift and the
  Kerr error now match the oracle's (−1.40 against −1.44 × 10⁻⁵ over 1000 steps; 1.7
  against 1.9 × 10⁻⁵), and on the A40 the `argon` field error fell to
  $1.8 \times 10^{-4}$ (table above). A C++ test measures the mean modulus bias in the
  execution space the build runs on (limit $0.02\,u_{32}$; CUDA's functions fail it),
  and a Python test compares the drift with the oracle's over 1000 steps. The cost is
  the binary64 transcendentals per bin and stage, which single now shares with mixed
  (stage entry 36 µs against 23 µs before, at $2^{16}$ on the A40).

## Part III — History

- 2026-10-25: Drafted (Opus 5.5), after an indicative A40 measurement of transforms,
  matrix products and element-wise passes in both precisions, and a CPU transform
  measurement on an Apple M5 Max. The maintainers chose two modes (single and mixed),
  noise qualified under mixed precision only, SEP 33's routes, measuring a fused batch
  axis before proposing one, and no `numba_cpu` work.
- 2026-10-25: Revised (Opus 5.5) with the maintainers' decisions on Q1–Q10: the
  recommendations were accepted, with the noise criteria tightened (64 shots, the
  rounding margin primary, standard-error-aware envelopes, the margin recorded in each
  run), and with AMD GPUs qualified in this SEP rather than deferred (Q9), which adds a
  HIP lane, benchmarks and per-pass profiles to phase 4.

- 2026-10-25: Reviewed by GPT-6 (AI). Findings R1–R12 below concern Part I at
  `e0eaf2a`. Only the review metadata and this history were changed; Part I and Part II
  were left untouched.
- 2026-10-25: Revised after the AI review (Opus 5.5). Every finding is accepted and
  answered in the response below. The indicative A40 and CPU measurements were rerun.
  Four decided questions are revised in detail (Q1, Q4, Q7, Q9) and two are new (Q11,
  Q12); all six await the maintainers' confirmation.
- 2026-10-25: John Travers confirmed the revised Q1, Q4, Q7 and Q9 and the proposed Q11
  and Q12 as written; recorded in §12. Every question of §12 is now decided, leaving
  only what the phases measure.
- 2026-10-25: John Travers confirmed that he has read the proposal and requested that
  his human review be recorded. Review status is `Reviewed`; proposal status remains
  `Draft`.
- 2026-10-25: Accepted by John Travers. Part I is frozen. The decision record of §9 is
  written as ADR-055.
- 2026-10-25: Phase 1 implemented and reported (Part II). John Travers decided that
  phase 5's Adams rule (Q2) compares mixed Adams with mixed Runge–Kutta.

### AI review — 2026-10-25

**Reviewer:** GPT-6 (AI). **Recommendation:** revise before acceptance. The two-mode
scope and the CPU-before-device sequence are reasonable. The main unresolved issues
are the exact mixed arithmetic boundary, the propagator accuracy claims, and the
experiment that will justify noise admission. The maintainers' Q1–Q10 decisions are
not changed by this review; the questions below identify where their implementation
or qualification needs a more precise contract, or an explicit reconsideration.

The review checked the proposal against SEP 33 and its implementation record, the
current precision and reference-backend code, the solver and storage conventions,
and the supplied indicative benchmark sources and A40 output. The A40 table values
agree with that output. Small scalar probes were run for R1 and R5; no propagation
study, native build or device experiment was rerun. The supplied evidence did not
include captured CPU timing output. Priorities: **P1** needs resolution before
acceptance because it changes a numerical guarantee or its qualification; **P2**
needs a specified answer before the affected phase is implemented. All findings are
open for a maintainer response.

1. **R1 — P1: Define the stored stage's frame, scale and range before removing the
   damping restriction (§§4.2, 5, 6.1–6.3, 6.7).** The displayed stage equation omits
   the inverse interaction-picture propagator. In the present implementation,
   `NumpyRefBackend._nonlinear` returns the source multiplied by that inverse, and
   `_adams_source` documents the same convention. Thus retaining a binary64
   propagator does not establish that the stored complex64 stage fits. For example,
   a lab-frame source of `1e-10` multiplied by `exp(150)` gives about `1.394e55`:
   binary64 holds it, narrowing overflows, and multiplication by `exp(-150)` would
   have recovered `1e-10`. This is precisely the kind of strongly damped bin that
   motivated SEP 33's exponent restriction. Does mixed store lab-frame source rows,
   interaction-picture rows with separate exponents, or bounded rows under a new
   admission rule? Specify the complete operation order, including inverse flow,
   source coefficients, widening and the conversion from `N(A / s)` to physical
   `N(A)`. Reusing the single path returns `N(A) / s`, so an exact widening alone is
   insufficient. Carry the chosen representation through FSAL, low-storage steps,
   Adams history, dense output, snapshots and memory estimates. Add a damped-bin
   witness and a non-unit-launch-scale nonlinear witness before dropping the guard.

2. **R2 — P1: State which absorber operations retain binary64 accuracy (§§4.2, 6.2,
   7.2).** Keeping absorber tables in binary64 does not keep their action in
   binary64. The current `_absorber_term` transforms the field even when there is no
   material response; the Cartesian `_nonlinear_free` also transforms the transverse
   field for its spatial absorber. The split-step `_finish` uses temporal transforms,
   and transverse transforms on the Cartesian route. If these reuse binary32 source
   workspaces, a physically linear run acquires binary32 error. If they retain
   binary64 transforms, the workspace and performance accounting must include those
   plans and buffers. Please define whether “linear part” means only the diagonal
   exponential or all admitted linear boundary treatments. Test the chosen guarantee
   with nonzero temporal and spatial absorbers, under both temporal schemes, as well
   as a source-free diagonal flow. A zero-source shortcut with inactive collars cannot
   establish the broader guarantee currently stated.

3. **R3 — P1: Make the mixed oracle implement the declared accumulation policy
   (§§4.2, 4.5, 6.2–6.3).** The mixed contract puts every running integral and
   reduction in binary64, but §6.2 reuses the single source and its reduction orders.
   In `backends/numpy_ref/single.py`, `running_integral` forms binary32 increments,
   calls the binary32 `prefix_sum`, and returns binary32; `plasma_current_single`
   calls it for exposure and free current. These require a deliberate policy change
   in the mixed oracle, not just composition of state hooks. Also distinguish
   binary64 norm/scan accumulators from the internal binary32 sums in FFTs and
   SGEMM/CGEMM: the abstract's “every accumulation” cannot literally cover both.
   Specify where products, increments and partial sums are widened, and where results
   are narrowed again. Add adversarial integral and dtype tests for those boundaries.
   Separately identify the existing oracle exceptions (binary64 envelope moments and
   statistics) so that their comparisons are not mistaken for matched binary32
   operation sequences.

4. **R4 — P1: Write the mixed comparison bound and qualify the oracle independently
   (§§4.2, 6.1, 7.1–7.3).** “The source's terms at `u32` and the state's at `u64`”
   does not yet specify how input narrowing, response conditioning, coefficient
   rounding, inverse-flow amplification and source scaling compose into a state
   error. The scalar `c_s u32 log2(n)` heuristic is not such a bound, especially for
   plasma near rate cutoffs or a source formed by cancellation. State the bound's
   inputs and absolute terms, including how a source error is transported through
   each stage and how the two sides' different accumulator precisions are charged.
   Restore SEP 33's explicit one-step oracle-versus-binary64 test, and add independent
   response witnesses before using native-versus-oracle agreement as qualification.
   For the proposed controller floor, check acceptance/rejection behaviour and
   progress as well as final metric improvement: the argument assumes a small
   per-step nonlinear phase that admission does not enforce. How are large nonlinear
   phases and nearly vanishing sources handled without a bound fitted to the results?

5. **R5 — P1: Bound both parts of the single-precision exponential (§§4.1, 4.3,
   7.2).** Reducing the imaginary argument does not protect the real exponent. For
   `x = -80.000003`, a binary32 cast produces `-80`; even an exact exponential of that
   cast argument differs from `exp(x)` by about `3e-6`, roughly `50 u32`. A NumPy
   scalar probe measured `2.97e-6`. This exponent is inside the stated damping limit,
   so the claim of a propagator within a few `u32`, and the claim that SEP 33's
   binary64-accurate exponential followed by one rounding is unchanged, need repair.
   Specify range reduction for the real exponential or retain its binary64 evaluation,
   and state the allowed error. For the phase, the indicative reduction using one
   binary64 `2*pi` constant is also not uniformly accurate: at `1e14` radians its
   phasor error is about `3.90e-3`, checked against a 70-digit reference. Either bound
   admitted phase arguments or specify robust reduction and test its extremes,
   cancellation, damping and modulus drift. The timing kernel alone is not an
   accuracy witness.

6. **R6 — P1: Do not require an unqualified double-float substitute for binary64
   transcendentals (§6.4, Q7).** A two-binary32 expansion generally supplies about
   48 significant bits, below binary64's 53, before polynomial and range-reduction
   errors. The MLX precedent targets a binary32 output; it does not establish this
   proposal's binary64 propagator contract. The arithmetic error analysis must cover
   the chosen expansion, polynomial, reconstruction and damping range; see the
   primary treatment of floating-point expansions by
   [Joldes et al.](https://doi.org/10.1109/TC.2015.2441714).
   Please make the third remedy conditional on a proved binary64 error budget, with
   ordinary binary64 evaluation remaining valid if none of the remedies meets it.
   Is the 15% share a trigger to investigate, or a performance requirement? As written,
   it can force an unproved numerical scheme solely because the first two remedies
   do not meet a timing threshold. Any product-derived propagators also need their
   product count and amplification included in the bound.

7. **R7 — P1: The 1%-of-length margin does not establish the noise margin of the
   run (§§4.4, 7.3, Q4).** Even under the proposal's constant-step random-walk model,
   measuring only the first 1% underestimates the full-run rounding amplitude by a
   factor of ten, or 20 dB. Early passage at the chosen threshold can therefore mean
   no margin at the end. Nor is “before amplification” guaranteed by a fraction of
   length: gain may start immediately or be concentrated later. The difference of
   two independently adaptive trajectories also includes truncation and different
   histories, even at equal tolerance, so it does not isolate rounding. Define a
   matched-discrete diagnostic (or a converged subtraction control), evaluate source
   perturbations at identical states, and account for injection and amplification
   throughout the run. Include a witness that the narrowed nonlinear source retains
   the physical seed's response: a binary64 stored seed can still disappear when
   `A + noise` is narrowed before evaluation. How will the study separate coherent
   coefficient bias from random rounding, and connect either to the ensemble verdict?
   The early-plane measurement can remain useful, but cannot alone be the stated
   primary qualification of accumulated noise.

8. **R8 — P2: Specify the uncertainty of the ensemble difference and the coherence
   estimator (§7.3, Q4).** A tolerance based only on the reference ensemble's standard
   error omits the candidate's sampling error and the covariance of matched seeds.
   If chaotic evolution decorrelates the two and their variances are equal, the
   difference has standard error `sqrt(2)` times either one's. In the Gaussian limit,
   when the standard-error term sets the threshold, that gate falsely rejects
   about 15.7% of individual comparisons before testing 32 bands and several metrics. Use a paired estimate or
   paired bootstrap of the difference, and state how simultaneous bands are handled;
   any change to the maintainers' numeric thresholds should be an explicit decision.
   Define how `processing.coherence.g12` is aggregated into a band (averaging fields,
   complex coherence or its modulus gives different observables), the bootstrap
   resampling unit, and how dark bands are selected from the reference. Freeze the
   pump, dispersion, energy, grid, propagation length, noise model, seeds and numerical
   controls of the new supercontinuum cases before candidate comparisons. The current
   qualitative case descriptions do not yet define a repeatable experiment.

9. **R9 — P2: Complete noise admission and the recorded margin's semantics (§§4.4–5,
   7.3, 11).** The package has both `noise="input"` and `noise="modified"`.
   `StepDriver` puts the first into the initial state but injects the second at source
   evaluations after unitary phase transport. The proposal's broad noise admission
   and its example do not say which are qualified. State the precision of the draw,
   its phase transport and addition, and either test both models or refuse the
   unqualified one. For metadata, replace the constant-step `phi_NL sqrt(M)` expression
   by a defined rule for variable steps, accepted versus rejected trials, FSAL and
   Adams evaluations; specify `c_s`, the occupied-bin definition and the noise metric
   for multimode and radial states under their energy measures. Persist any cumulative
   diagnostic through snapshot/resume and reset it for a reused runner. A scalar
   heuristic omitting correlated bias and gain should be labelled as such, rather than
   presented as a measured per-band safety margin. What value is recorded when its
   assumptions do not apply or the signal norm is zero?

10. **R10 — P2: Enforce the vendor math policy and test the actual dispatch
    (§§6.3, 7.2).** The cited
    [CUDA 12.8 cuBLAS documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublascomputetype-t)
    documents `NVIDIA_TF32_OVERRIDE=0` as disabling TF32; it does not establish the
    claimed environment switch that forces TF32. Document a supported mechanism for
    the failure test, such as explicitly selecting TF32 on the handle in a test
    fixture, and specify the production math mode on the handle Kokkos actually uses.
    A Gaussian probe is useful evidence, but does not configure that handle or cover
    another dispatch. Define the tested real/complex, transpose and layout families,
    including node-chunk tails, and how construction avoids a full problem-sized
    reference GEMM and extra allocations outside the memory estimate. Use a supported
    rocBLAS control for the corresponding negative test. A detected arithmetic
    mismatch should report the observed mode or probe failure without attributing it
    to an environment variable that may not have caused it.

11. **R11 — P2: Close the AMD qualification gap after phase 5 (§§7.3, 11–12).**
    Phase 4 qualifies both GPU modes before noise and possibly Adams are admitted.
    Phase 5 changes those capabilities, but the opening of §11 names the AMD lane
    only for phase 4, while the final criterion claims both modes qualified on AMD.
    Q10 instead requires device evidence for phases that touch device code. Specify
    the post-phase-5 HIP checks for noise, Adams if admitted, their batches and
    snapshots, or explicitly limit the AMD claim until those checks run. Schedule any
    required second rental within the separately agreed budget. Also make clear which
    mixed confirmation cases run on CUDA: §7.3 explicitly assigns `rdw` and `argon`
    to AMD, whereas its CUDA confirmation list is introduced as single precision.

12. **R12 — P2: Keep performance and memory claims at the scope the evidence
    supports (§§2, 4.5, 6.7, 7.4, 10).** The recorded A40 results support the motivation,
    but at `2^16` the mixed combination is slower than binary64 (8.1 versus 6.5 us),
    and the full binary64 propagator is about 1.9 times the reduced-argument variant
    at `2^20` (198 versus 103 us), not 1.4 times. The indicative kernels do not time
    reductions or plasma scans, so they cannot establish that binary64 accumulation
    is free, or that compensation is never faster on any native target. Record these
    as hypotheses for the planned profiles. Likewise single precision does not
    literally halve all resident bytes while linear tables and accumulators remain
    binary64; derive representative ratios from the named ledgers, including library
    workspaces and output retention. For reproducibility, provide a public timing
    recipe and CPU result record with compiler and library provenance. The supplied
    GPU FFT/GEMM timing inputs were zero, the CPU script does not restore input after
    `FFTW_MEASURE`, and timing-loop FFT/GEMM statuses are not checked. Use finite
    representative inputs, validate outputs and check library errors when replacing
    these explicitly indicative numbers with qualification evidence.

### Response to review — 2026-10-25 — Opus 5.5

Every finding was checked against the code and, where it made a numerical claim, by
measurement before it was answered. All twelve are accepted. The A40 measurements of §2
were rerun on the same card with finite inputs, checked statuses and validated outputs,
and with reductions and scans added (R12). The CPU measurement was rerun with refilled
inputs. The scalar witnesses of R5 and the false-rejection rate of R8 were reproduced.
Decisions that the changes touch are marked in §12 for the maintainers.

- **R1 (accepted; §4.2, §5, §6.1, §6.2, §6.7, §7.2, §10; Q12).** Confirmed:
  `_nonlinear` returns the source multiplied by the inverse propagator, so a binary32
  stage row can overflow on a damped bin even with a binary64 propagator. §4.2 now
  writes out the complete operation order: binary64 stage input, one narrowing at the
  launch scale, SEP 33's binary32 source returning $\mathcal N(A)/s$, exact widening
  and multiplication by $s$, and a binary64 pull-back. It keeps every stored row binary64
  (stages, FSAL, low-storage registers, Adams history, dense output, snapshots), so
  no damping restriction is needed. The memory claims are corrected (§6.7). Damped-bin
  and non-unit-scale witnesses are added. Binary32 rows with a damping refusal are
  recorded as the alternative, and the performance cost is put to the maintainers as
  Q12.
- **R2 (accepted; §3, §4.2, §6.2, §7.2).** Confirmed in `_absorber_term`,
  `_nonlinear_free` and `_finish`. The linear part is now defined precisely.
  Once-per-step treatments on the whole state are binary64 including their transforms:
  the diagonal flow, the mask and the split-step absorber. Mixed precision therefore
  keeps binary64 plans for the split-step absorber, which the memory and benchmark
  accounting count. In-stage absorber terms are part of the binary32 source, with
  rounding scaled by $\Delta z\,\mathrm{rate}$. The tests cover both schemes, temporal
  and spatial absorbers, the fibre and Cartesian routes, and a collar-free source-free
  run.
- **R3 (accepted; §1, §4.2, §4.5, §6.2).** Confirmed: the single source's running
  integrals are binary32 in SEP 33's orders. Rather than widening them, the contract now
  says what the first draft should have said. Reductions over the state are binary64.
  Running integrals and sums inside the source are part of the binary32 source. Library
  sums inside transforms and products are binary32 and covered by the probes and
  bounds. The abstract no longer claims "every accumulation". The oracle is specified
  as an explicit implementation of the operation order with dtype guards and adversarial
  boundary tests. The oracle's existing binary64 exceptions are listed (§4.5).
- **R4 (accepted; §4.2, §7.1, §7.2, §7.3).** §7.1 now writes the mixed bound per stage,
  per step and per run. It includes the input narrowing through each response's
  condition number (the plasma's read from its rate table, so cutoffs enter
  explicitly), the passes' binary32 bounds, the absolute underflow term, the inverse
  flow (bin by bin, which scales the absolute bound and leaves the relative one
  unchanged), transport into later stages, and the two sides' accumulators. The
  one-step oracle-against-binary64 test and independent response witnesses are
  restored. The controller argument now shows why it cannot stall: rounding in the
  estimate scales as $\Delta z$ against truncation's $\Delta z^5$. The floor rule now
  also counts accepted and rejected steps.
- **R5 (accepted; §4.1, §4.3, §7.2).** Reproduced: the binary32 cast of $-80.000003$ is
  $-80$, with $3.0 \times 10^{-6}$ relative error in the exponential. A single
  binary64 $2\pi$ leaves a phasor error of $4$–$9 \times 10^{-3}$ at $10^{14}$ rad.
  §4.3 now reduces the real part by $\ln 2$ (Cody–Waite, exact power-of-two scaling)
  and the phase by a three-part $2\pi$ for $|\theta| \le 2^{30}$ rad, falling back to
  the binary64 library above. It states a $10\,u_{32}$ bound from the functions'
  documented accuracies. Tests cover both ranges' extremes, the inverse propagator and
  modulus drift. The timing kernel is labelled as such in §2.
- **R6 (accepted; §6.4, §10; Q7 revised).** The double-float remedy is withdrawn: two
  binary32 terms carry about 48 bits, and its approximation errors would need their own
  analysis. The 15 % share is now a trigger to investigate. Plain binary64 remains the
  fallback. Product-derived propagators carry their product count in the bound.
- **R7 (accepted; §4.4, §7.3; Q4 revised).** Agreed that an early-plane measurement on
  independently adaptive runs neither isolates rounding nor captures accumulation. The
  primary criterion is now a margin on matched-discrete runs (the binary64 schedule
  replayed in mixed precision) against the seed's own response on the same schedule. It
  is evaluated at every saved plane until the band's dynamics stop being linear in the
  seed, with the coherent bias separated as the noise-free difference. A
  source-response witness at identical states tests directly that narrowing keeps the
  seed's effect. §4.4 shows where the seed could be lost and why it survives narrowing
  (per-sample relative rounding), and labels its estimates as heuristics.
- **R8 (accepted; §7.3; Q4 revised).** Reproduced the 15.7 % per-comparison false
  rejection. The ensemble tolerances now use the paired standard error of the
  difference by bootstrap over seeds, with Bonferroni family-wise control at 5 %
  ($z = 3.4$ for 64 comparisons per case), keeping the maintainers' absolute floors.
  This changes the decided rule and is flagged in §12. Band coherence is defined
  (per-bin `g12`, mean-spectrum-weighted band average), the bands are selected from the
  reference, and the cases' parameters are frozen in Part II before comparison, as
  SEP 33 did.
- **R9 (accepted; §3, §4.4, §5, §7.3; Q11 new).** Confirmed: the modified model adds its
  phase-transported realisation at every stage input. Both models are now specified,
  drawn, transported and added in binary64 before the one narrowing. Both are qualified,
  and an unqualified model is refused by name. The metadata indicator is defined
  (accepted steps only, FSAL and Adams counted with their step, the route's energy
  norm, the occupied-bin rule, $c_s$ calibrated by the study, absent rather than a
  number when undefined, persisted through snapshots and reset on reuse) and labelled a
  heuristic that omits gain and bias.
- **R10 (accepted; §6.3, §7.2).** Corrected: `NVIDIA_TF32_OVERRIDE=0` disables TF32,
  and nothing in the cited documentation makes it force TF32. Native now sets the math
  mode explicitly on the handle Kokkos Kernels uses (and rocBLAS's default mode on HIP).
  It probes each dispatch family (real and complex, transposes, layouts, tail chunks)
  with small slices against a host reference, inside the memory estimate. It reports
  the observed error without attributing a cause. The negative tests select TF32 or
  `xf32` through the libraries' APIs.
- **R11 (accepted; §7.3, §11, §12; Q9 revised).** Phase 5 now reruns the device checks
  for its admissions (noise models, Adams if admitted, batches, snapshots) on both GPUs,
  with a second AMD rental within the agreed budget. The Final criterion covers every
  capability admitted by phase 5. The mixed `rdw` and `argon` confirmations run on both
  GPUs.
- **R12 (accepted; §2, §4.5, §6.7, §7.4, §10).** Corrected: the full binary64 propagator
  is 1.9× the reduced-argument form at $2^{20}$, not 1.4×. The rerun shows binary64
  accumulation of binary32 data costing 1.5–2.2× on the A40, so the "free" claim is
  withdrawn. §4.5 now keeps binary64 where the contract needs it and leaves other
  accumulator choices to per-target profiles. The first draft's 2.2×-cheaper mixed
  stage combination no longer applies once stage rows are binary64 (R1), and §2 says so.
  The memory claims now come from the ledgers. The benchmark inputs are now finite, the
  outputs validated and the statuses checked. The rerun changed the transform ratios
  from 7–12× to 6–9×, since each pair now includes a scaling pass. The public records
  will come from the standard suite's harness with full provenance (§7.4). The
  indicative kernels stay with the development records, as SEP 30 keeps campaign
  harnesses.
