# SEP 36 — Single and mixed precision on the native backend

| | |
|---|---|
| **SEP** | 36 |
| **Title** | Single and mixed precision on the native backend |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-25) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-25 |
| **Resolution** | — |
| **Requires** | [SEP 33](archive/sep-0033-single-precision-mlx.md) (the precision setting, the binary32 contract, its oracle and survival record); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (the floating-point policy and the bound classes); [SEP 29](archive/sep-0029-compiled-backend-distribution.md) (the native backend in solprop and its wheels); [SEP 30](archive/sep-0030-standard-benchmark-suite.md) (the benchmark suite) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Bring reduced precision to the native (Kokkos) backend, on its CPU builds and on its
CUDA and HIP device builds, for speed and throughput. Two arithmetic modes are
proposed. `precision="single"` is SEP 33's binary32 contract, unchanged, implemented on
native. Native may keep in binary64 whatever the contract allows and measurement shows
costs nothing. `precision="mixed"` is new: the propagated state, the propagators, the
Adams history and every accumulation stay in binary64, and only the source evaluation
is binary32. That covers the transforms, the responses and the transverse and modal
matrix products. Mixed precision is possible only where binary64 exists, which is every
native target but not Apple GPUs. It removes single precision's coherent propagator
drift and keeps the error floor near binary32's rounding times a step's nonlinear
phase. That makes it a candidate for noise-seeded ensembles, which single precision
refuses. `numpy_ref` gains a mixed path as the oracle of the new mode. Its
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

**An indicative measurement on one A40** (2026-10-25: 48 GB, 84 SMs, CUDA 12.8.1,
cuFFT and cuBLAS) timed the building blocks of an interaction-picture step. It ran in
one process, with medians of seven rounds of 10–50 repetitions timed by CUDA events.

Transforms, forward and inverse complex-to-complex:

| shape | binary64, µs | binary32, µs | ratio |
|---|---|---|---|
| one line, $2^{14}$ | 106 | 16 | 6.7 |
| one line, $2^{16}$ | 105 | 15 | 7.0 |
| one line, $2^{18}$ | 249 | 26 | 9.7 |
| 204 × 2048 (multimode) | 182 | 18 | 10.4 |
| 64 × $2^{14}$ (a batch) | 707 | 58 | 12.2 |
| 256 × 4096 (radial) | 553 | 58 | 9.6 |
| 384 × 384 × 256 (Cartesian, 3-D) | 30 251 | 7718 | 3.9 |

Real matrix products ($m \times k$ by $k \times n$), binary32 in cuBLAS's pedantic
mode (no TF32):

| shape | DGEMM, µs | SGEMM, µs | ratio |
|---|---|---|---|
| 256 × 256 × 8192 (a radial Hankel transform) | 2296 | 65 | 35 |
| 64 × 64 × 32 768 | 647 | 40 | 16 |
| 1024 × 1024 × 4096 | 17 251 | 404 | 43 |

Element-wise passes, µs per pass, at $n$ complex samples:

| $n$ | stage combination: binary64 / binary32 / mixed | propagator from three terms: binary64 / binary32 / binary64 argument, binary32 trigonometry |
|---|---|---|
| $2^{16}$ | 6.5 / 3.2 / 8.1 | 17.6 / 2.8 / 7.8 |
| $2^{18}$ | 49 / 24 / 34 | 55 / 16 / 29 |
| $2^{20}$ | 206 / 88 / 124 | 198 / 59 / 103 |
| 384 × 384 × 256 | 7360 / 3316 / 4622 | 6894 / 2039 / 3562 |

Here "mixed" reads a binary64 state and propagator with three binary32 stage values
and writes a binary32 source input. The last column forms the exponent and its range
reduction in binary64 and evaluates `exp` and `sincos` of the reduced argument in
binary32: SEP 33's contract (§4.2 there).

Four conclusions shape this proposal:
- The transforms and the matrix products are where binary64 hurts on this card, by
  7–12× and 16–43×.
- Streaming element-wise passes are bandwidth-bound in either precision and cost only
  the byte ratio, about 2.2×, in binary64.
- A propagator with a binary64-accurate argument costs about half of a full binary64
  one; full binary64 transcendentals make the pass compute-bound.
- At $2^{14}$ the binary64 transform pair on the A40 (106 µs) is slower than the same
  pair in FFTW on one laptop core (66 µs, below).

So a mixed mode on this card keeps most of single precision's gain on the transforms
and matrix products. It pays about 1.4× on the element-wise passes, and on the
propagator if that is formed in full binary64 (§6.4).

**On CPUs and full-rate GPUs the gain is the byte and SIMD factor.** The same transform
pair, timed in FFTW 3.3.11 through pyFFTW on an Apple M5 Max CPU, measured binary64
over binary32 as follows:

| shape | threads | ratio |
|---|---|---|
| $2^{14}$ | 1 | 1.30 |
| $2^{16}$ | 1 | 1.53 |
| $2^{18}$ | 4 | 1.17 |
| 204 × 2048 | 4 | 1.43 |
| 256 × 4096 | 8 | 1.47 |

The machine was under other load (load average about 3.5), and this FFTW build is not
the native wheel's. On x86-64-v3 (AVX2) and aarch64 (NEON, SVE) builds, the
memory-bound passes should approach 2×. Data-centre GPUs with full binary64 rates
(H100, H200, MI300-series) gain by the byte factor too, which also doubles the problem
that fits in device memory. The phases' own measurements replace all of these figures
(§7.4).

**Native differs from MLX in one way that matters: binary64 is always there.** SEP 33
had to put everything in binary32 on Apple GPUs and emulate binary64 only for the
propagator's argument. On native, binary64 is available on every target, at a cost
that depends on the target and the pass. The design therefore makes two kinds of
choice:
- a *mode* choice for users: full binary32 for the largest speed-up, or mixed for
  binary64 accuracy in the linear part and a much lower error floor;
- an *implementation* choice inside each mode: binary64 is kept wherever it is free or
  nearly so, such as accumulators, host scalars and the propagator's argument, provided
  the mode's error bounds still hold.

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
  [Single precision](../../guide/performance/single-precision.md) carries the survival
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
- the contract of §4.2, including the binary64-accurate propagator rounded once;
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

Write a stage as $k = \mathcal{N}(e^{\Delta\Phi}(A + \Delta z \sum_j a_j k_j))$, as in
SEP 33 §4.1. Under `precision="mixed"` the work splits three ways.

*Binary64:*
- the state $A$ and every stage combination
  $e^{\Delta\Phi}(A + \Delta z \sum_j a_j k_j)$, with each $k_j$ widened exactly before
  it is scaled;
- the propagators: the combination $\sum_t s_t \Phi_t$, its exponential and every
  product with the state;
- the accepted update and the error estimate's array;
- the Adams history, its compensation rows and its propagators;
- the controller, $z$, $\Delta z$, and every host scalar;
- every running integral and reduction: the norms, the statistics, the plasma's
  exposure and its free-current integrals.

*Binary32*, with SEP 33's range rules for the source's buffers:
- the source input, formed once per stage by rounding the binary64 stage state, divided
  by the launch's power-of-two scale, to binary32;
- every transform inside $\mathcal{N}$;
- the pointwise responses: Kerr, the third-harmonic term, scalar and tensor Raman, and
  the field- and envelope-grid plasma's rates, populations and currents;
- the radial Hankel GEMMs and the Cartesian transverse transforms;
- the stage value $k$, stored in binary32.

*Narrowed at upload with block exponents:* every table the source reads (Raman kernels,
rate tables, prefactors, quadrature and transform matrices). Tables of the linear part
(operator terms, masks, absorbers, collars) stay binary64.

**Error.** The linear part is binary64, so contributions 2 and 3 of SEP 33 §4.1 (the
state's rounding and the propagator's coherent drift) fall to binary64 levels. What
remains is binary32's rounding of the source. It enters the state scaled by the step,
as a relative error per step of about
$$\delta \approx c_s\,u_{32}\,\log_2 n\;\phi_{\mathrm{NL}},$$
random in sign from step to step, plus the coefficients' coherent error of about
$u_{32}\,\Phi_{\mathrm{NL}}$ over the run. On the heuristic of SEP 33 §4.1,
undifferentiated growth gives an accumulated error of about
$u_{32}(c_s \log_2 n\,\phi_{\mathrm{NL}}\sqrt{M} + \Phi_{\mathrm{NL}})$, multiplied
by the propagation's sensitivity where the dynamics amplify. Linear propagation
contributes nothing at binary32 level, whatever the band-edge phase or the number of
steps. That removes the step-count limit of single precision (SEP 33's
$\approx 3 \times 10^4$ steps).

**The controller's floor.** The embedded estimate
$\Delta z \sum_i (b_i - \hat b_i) k_i$ sees the source's rounding scaled by the
weights' differences, which for DOPRI5 are of order $10^{-2}$–$10^{-1}$. Its rounding
noise is therefore about $10^{-1}\,c_s\,u_{32}\log_2 n\,\phi_{\mathrm{NL}}$. For
$n = 2^{16}$ and $\phi_{\mathrm{NL}} \le 0.1$ that is below $10^{-8}$. The proposed
floor for mixed precision is $\mathrm{rtol} \ge 10^{-8}$, to be raised by the study's
rule (§7.3, Q1) if tolerances below the raised value buy nothing.

**Range under mixed precision.** The state needs no scale. The source input is the
state divided by the launch's power-of-two scale $s$, so it is exact apart from the one
rounding to binary32, and every source buffer follows SEP 33 §4.3. The source's output
is multiplied by its combined coefficient, which includes $s$'s homogeneity factor, as
it is widened. Overflow of a binary64 stage combination is not a binary32 concern.

#### 4.3 The propagator in each mode

- **Single.** The contract asks for a binary64-accurate argument rounded once. Native
  forms $\sum_t s_t\Phi_t$ and its range reduction modulo $2\pi$ in binary64 and
  evaluates `exp` and `sincos` of the reduced argument in binary32, giving a binary32
  propagator with a relative error of a few $u_{32}$. The argument is reduced to
  $[-\pi, \pi]$, so binary32 `sincos` loses no accuracy to its size. This is one code
  path on every target. On the A40 it costs 1.75× the all-binary32 form and 0.52× the
  all-binary64 form (§2). On CPUs, binary64 trigonometry costs little more, and the
  same path keeps one oracle comparison.
- **Mixed.** The propagator multiplies a binary64 state, so a binary32 propagator would
  bring back SEP 33's coherent drift, $c\,M\,u_{32}$. It is therefore formed and kept in
  binary64. Its cost on reduced-FP64 GPUs is the one open performance risk of the mode
  (§6.4).

#### 4.4 Noise under mixed precision

A one-photon-per-mode seed has a spectral amplitude, relative to the pulse's peak
spectral amplitude, of about $\sigma_q \approx \sqrt{\hbar\omega_0 n_{\mathrm{occ}}/E_p}$
over $n_{\mathrm{occ}}$ occupied bins (SEP 33 §4.5). That is $1.6 \times 10^{-5}$ for
1 µJ over 1000 bins and $5 \times 10^{-7}$ at 1 mJ.

Under single precision, the state's own rounding reaches $u_{32}\sqrt{M} \approx
6 \times 10^{-6}$ after $10^4$ steps. That is the reason noise stays refused there.

Under mixed precision, only the source's rounding is binary32. A transform's rounding
error is spread over its bins [3], so a normwise relative error $\delta$ puts about
$\delta\sqrt{n_{\mathrm{occ}}/n}$ into each occupied bin, relative to the peak spectral
amplitude. The accumulated rounding "seed" per bin is then about
$$\sigma_r \approx c_s\,u_{32}\log_2 n\;\phi_{\mathrm{NL}}\sqrt{M\,n_{\mathrm{occ}}/n}.$$
For $c_s = 1$, $n = 2^{14}$, $\phi_{\mathrm{NL}} = 0.05$, $M = 10^4$ and
$n_{\mathrm{occ}}/n = 1/4$, this gives $\sigma_r \approx 2 \times 10^{-7}$. That is
38 dB below a 1 µJ seed and about 8 dB below a 1 mJ seed.

The heuristic is not a bound. The ratio $\sigma_r/\sigma_q$ is what decides whether
amplified noise statistics are the quantum seed's or binary32's, so the study measures
it directly (§7.3) and does not rely on this estimate. In binary64 the same estimate is
$10^{-15}$ and always negligible.

Noise-seeded runs are chaotic where modulation instability or Raman gain amplify the
seed. Individual shots with the same seed then diverge between precisions even when
the ensemble is right. Noise is therefore qualified on ensemble statistics and on the
rounding margin $\sigma_q/\sigma_r$, never on individual shots.

#### 4.5 Binary64 that native keeps under single precision

These choices cost nothing measurable, or close to it, and only reduce error. Each
stays within the class R bounds against the binary32 oracle: a comparison's bound is
the sum of both sides' bounds, and native's side only shrinks.

- **Accumulators.** Norm partials, statistics and the plasma's running integrals load
  binary32 values and accumulate in binary64. These passes are bandwidth-bound; on the
  A40 a binary64 add per loaded binary32 complex sample uses about a third of the card's
  binary64 rate. SEP 33's compensated binary32 scans become unnecessary on native.
  SEP 33's Q5 ("the oracle defines the orders") still holds for the comparison: the
  oracle's binary32 orders set the bound native is tested against.
- **The propagator's argument**, as §4.3.
- **Host scalars and combined coefficients**, as in SEP 33.
- **Evaluated tables** (z tables, rate-table lookups) are formed in binary64 on the
  device and rounded once, as the oracle does on the host. Where a binary64 lookup
  costs more than a measured threshold on a reduced-FP64 GPU, the implementation
  records it and keeps the binary64 form unless the oracle's bound allows the binary32
  form.

Binary32 stays mandatory under single precision for the state, the stage values, the
transforms, the responses and the matrix products. Those are what the mode is for.

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
  admitted on the routes and cases the study qualifies, with a refusal or a documented
  limit where it does not (§7.3, Q4). A mixed run with a noise model records in its
  output metadata an estimate of its accumulated rounding floor, $\sigma_r$ of §4.4,
  from the per-step nonlinear phase (the norm of one stage value, folded into the
  existing norm reduction), beside the seed's $\sigma_q$, so a user can read the margin
  of their own run.
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
- **Memory.** `estimate_memory` counts each buffer at its mode's element size:
  binary64 state, history and propagators, and binary32 source workspaces and stage
  values under mixed.
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
- `_check_single_exponent` applies to single only: the mixed propagator is binary64.
- `SCHEMA_VERSION` and the output schema change in phase 1 for the new value, under
  ADR-051.
- The SEP 33 range helpers serve both modes unchanged.
- `tests/bounds.py` gains a mixed bound: the source's terms at $u_{32}$ and the state's
  at $u_{64}$, composed as SEP 33 §7.1 composes its formulas.

#### 6.2 `numpy_ref`: the mixed oracle

`NumpyMixedBackend` reuses the binary32 source path of `NumpySingleBackend` and the
binary64 state path of the reference. The hooks already separate them (`_upload` for
the narrowed source tables, `_column_sources`, `_envelope_sources`, `_raman_spectrum`
and `_held_streams` for the source; `_propagator`, `_finish` and `_error_norm` for the
state). The source input is rounded once per stage at the launch scale. Stage values
are stored in complex64 and widened exactly where they are combined. The dtype guards
assert complex128 on the state and complex64 in the source. The oracle's reduction
orders are SEP 33's in the source, and the reference's in binary64. Most of the work is
in the class's composition, not in new arithmetic.

#### 6.3 Native: a precision policy in the C++

**The options.**

- **A. Kernels templated on a precision policy** (recommended). A policy type names
  the roles:

  | policy | state | source | accumulator | propagator |
  |---|---|---|---|---|
  | `Double` | `double` | `double` | `double` | `double` |
  | `Single` | `float` | `float` | `double` | `float`, from a `double` argument |
  | `Mixed` | `double` | `float` | `double` | `double` |

  The view aliases (`StateView`, `SourceView`, `TableView`, `AccumulatorView`) are
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
- *TF32.* cuBLAS computes binary32 products in TF32 when an application asks for it,
  and `NVIDIA_TF32_OVERRIDE` can force it [5]. Construction runs a GEMM probe on each
  product shape the run uses, with Gaussian factors and SEP 33's discriminating
  threshold $2^{-17}$. It refuses if TF32 is detected, naming the variable. On HIP
  builds the same probe covers rocBLAS's reduced-precision `xf32` compute type, which
  CDNA 3 offers for binary32 products.
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
bandwidth-bound pass at $2^{20}$ (198 µs, against 206 µs for a binary64 stage
combination), but 6× the binary32 propagator at $2^{16}$. Phase 4 measures its share
of a mixed step on the A40. If the share exceeds 15 %, the remedies below are evaluated
in order, each binary64-accurate and each tested against the oracle:

1. **Reuse while $\Delta z$ is unchanged.** Fixed schedules and constant-operator
   routes repeat the same propagators. They can be cached, keyed by the step and the
   terms' weights, within the memory budget.
2. **Derive stage propagators by products.** On a constant operator, the propagators at
   the stage nodes are products of a few base propagators. Binary64 complex
   multiplication is bandwidth-bound, not transcendental-bound. The error grows by one
   rounding per product, within the class R bound.
3. **Binary64 range reduction with a double-float polynomial** on binary32 units,
   as SEP 33's MLX kernel does for the argument. This is the most work, and is used
   only if 1 and 2 do not bring the share below the threshold.

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

- **Single precision** halves resident bytes against binary64.
- **Mixed precision** keeps the state, the history, the propagators and the linear
  tables in binary64 and halves the source's workspaces and stage values. A DOPRI5 run
  with seven stage values is about 0.6 of binary64's resident bytes; an Adams run
  depends on its history length.

`estimate_memory` reports both, and the device lane checks the measured peak against
the estimate.

#### 6.8 Dependencies

None new. FFTW's single-precision libraries are part of the same FFTW distribution and
licence. cuFFT, cuBLAS, rocFFT and rocBLAS already include single precision.

### 7. Verification and validation

#### 7.1 Bounds

Single precision uses SEP 33's binary32 bounds and limits of use (its §7.1). Mixed
precision uses the bound of §6.1: binary32 terms for the source's operations and
binary64 terms for the state's. A class R bound larger than a tenth of a case's class T
envelope is uninformative, as in SEP 33, and the comparison then runs on a shorter
window.

#### 7.2 Class P, class R and targeted tests

- **No binary64 change.** The existing native suite passes unchanged on the `Double`
  instantiation. A same-host A/B on the SEP 30 core cases shows no slowdown beyond the
  suite's noise (SEP 30's rules).
- **Native single against `numpy_ref` single**, and **native mixed against `numpy_ref`
  mixed**, on fixed schedules with a fixed-order method, for every capability of §5.1
  in its phase. These run on the CPU build in CI, on the CUDA build in the local A40
  lane, and on the HIP build on the AMD GPU.
- **Native single against `mlx`** on an Apple-silicon Mac. Both are bounded against the
  oracle, so this is a consistency check, not a gate.
- **The linear part of mixed precision.** A source-free run under mixed precision
  agrees with binary64 within binary64 bounds. This witness shows that no binary32
  arithmetic reached the linear path.
- **SEP 33's targeted tests, repeated on native in both modes:**
  - scaling, class P where only powers of two differ;
  - the Cartesian energy that would overflow unscaled;
  - the vanished molecular Raman response;
  - plasma populations at small and saturated exposure;
  - the propagator drift, single only, with $c$ recorded per target;
  - failure injection in fixed, low-storage and adaptive runs;
  - resume across backends in the same precision (class T) and refusal across
    precisions;
  - snapshot round trips;
  - memory against the estimate.
- **Probes.** The TF32 GEMM probe on each product shape, with a forced-TF32 environment
  refused; transform characterisation at $u_{32}$; a subnormal surviving a device pass.
- **Batches.** Each member agrees with its solo run within the mode's bound.

#### 7.3 Class T: the survival studies

**Single precision.** The oracle's survival record (SEP 33 Part II) transfers to native
through the class R tests above. Phase 4 confirms it on the A40 by rerunning three of
its cases on the CUDA build against their binary64 references: `ctc` at its 0.05 m
limit, `argon` and `radial_filament`. Each is judged by SEP 33's envelopes. The same
three cases, and the mixed study's `rdw` and `argon`, are rerun on the AMD GPU. This is
a confirmation, not a new study.

**Mixed precision: coherent cases.** The criteria are SEP 33 §7.3's, recorded in
Part II before the first comparison:
- references in binary64 on the same grid at $\mathrm{rtol} = 10^{-10}$, checked
  against $10^{-9}$;
- the same metrics and envelopes (field $10^{-3}$, bands 1 %, energy $10^{-4}$, peak
  $10^{-3}$, ionised fraction $10^{-3}$, width $10^{-3}$);
- the same verdicts;
- SEP 33's cases, with candidates at $10^{-8}$, $10^{-7}$ and $10^{-6}$.

The study also records each case's error against the binary64 run at the same
tolerance. This separates precision from truncation, which SEP 33's Adams finding showed
is needed.

- **The floor (Q1).** The floor rises from $10^{-8}$ to $10^{-7}$ if, on every case, no
  metric's error at $10^{-8}$ is smaller than at $10^{-7}$ by more than 25 %. The same
  test is then applied from $10^{-7}$ to $10^{-6}$.
- **Adams under mixed precision (Q2).** Admitted, and made the mixed default, if on the
  `rdw` and `argon` cases at each candidate tolerance its error is within every envelope
  and at most twice that of binary64 Adams at the same tolerance in every metric.
- **Expected.** Everything survives at the floor. `ctc` is "to measure": the source's
  rounding still seeds empty Stokes bins, at a level $\phi_{\mathrm{NL}}$ smaller than
  single's, so its limit should lengthen.

**Mixed precision: noise-seeded cases.** The criteria are recorded before the first
comparison. Cases run on reduced windows, with ensembles of $N = 64$ shots with
matched seeds against binary64 ensembles:

| id | route | definition | expected |
|---|---|---|---|
| `scg_coherent` | fibre, envelope | GNLSE supercontinuum in a silica photonic-crystal fibre, 50 fs pump in anomalous dispersion, length in the coherent regime | survives |
| `scg_incoherent` | fibre, envelope | the same with a 1 ps pump, modulation-instability dominated, $\lvert g_{12}\rvert$ below 0.5 over most of the band | to measure |
| `rdw_noise` | fibre, field | `rdw` with a one-photon-per-mode seed | survives |
| `ctc_noise` | fibre, field | `ctc` with a seed, Raman-gain dominated | to measure |
| `radial_filament_noise` | radial | `radial_filament` with a seed | to measure |

Metrics, at the last plane, in the 32 bands of SEP 33's spectral metric above −30 dB
of the largest:
- **the ensemble-mean spectral energy** per band. Envelope: the larger of 1 % and
  twice the binary64 ensemble's standard error of the mean.
- **the modulus of the first-order coherence**, $\lvert g_{12}\rvert$ [6], per band.
  Envelope: the larger of 0.02 absolute and twice the binary64 ensemble's standard
  error of $\lvert g_{12}\rvert$ in that band (by bootstrap over its shots), so that a
  partly coherent band does not fail on sampling alone.
- **the rounding margin** $\sigma_q/\sigma_r$ per band, from the difference between
  mixed and binary64 shots with identical seeds, measured at 1 % of the length (before
  amplification), against the seed's own spectral amplitude. Envelope: at least 20 dB
  in every band. This is the primary criterion: it tests directly that binary32 is not
  the source of the noise, and it needs a few shots, not an ensemble.

Verdicts are as for the coherent cases. A failure becomes a refusal where its
condition can be computed from the problem before the run; otherwise it becomes a
documented limit (Q4). The study records the measured $\sigma_r$ against §4.4's
estimate, and the metadata estimate of §5 against the measured margin. It also states
whether a problem-level predicate, such as one in pulse energy, occupied bins and
expected nonlinear phase, separates the passing cases from the failing ones.

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
- native single against `mlx` on the same Mac.

The user guide's advice comes from these records. Where a mode gains less than 1.3× on
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
  `SOLPROP_NATIVE_PRECISIONS`, the Ampere (`AMPERE86`) preset, and FFTW's
  single-precision libraries for source builds.
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
  mode, and binary64 kept where it is free*. It extends ADR-054 in four ways:
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
- **A binary32 propagator under mixed precision.** It costs the same as under single,
  and it reintroduces the coherent drift that is the mode's reason for being (§4.3).
  Not chosen.
- **Compensated binary32 accumulation on native, as on `mlx`.** It is no faster than
  binary64 accumulation of binary32 loads on any native target and is harder to bound.
  Not chosen (§4.5).
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
definition of done of phases 4 and 5, and the AMD lane of phase 4.

**Phase 1 — The mixed setting and its oracle.**
- `Precision`'s mixed form, admission, validation, schemas, the mixed bound, and
  `numpy_ref`'s mixed path with its dtype guards.
- The mixed survival criteria are recorded in Part II. The coherent-case study then
  runs on `numpy_ref` mixed. References may run on binary64 native, as SEP 33 did for
  the envelope case. The study gives a provisional floor and a provisional Adams
  measurement.
- Verifies: no binary64 change; the source-free witness; SEP 33's targeted tests in
  mixed precision; the refusals.

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
- Verifies: class R against both oracles; the mixed source-free witness on native; CPU
  benchmarks on the suite's radial and Cartesian cases.

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
  §7.3; benchmarks against binary64 on each GPU and against the CPU hosts; the Q8 rule
  applied.

**Phase 5 — Noise, Adams and the final records.**
- The noise-seeded study on native mixed (validated in phases 3–4) against binary64
  ensembles, with its criteria recorded first.
- Noise admitted under mixed precision on what it qualifies, with the refusals or limits
  it implies.
- The mixed floor and the Adams admission decided by the rules of §7.3, with the
  defaults set accordingly. The binary32 Adams measurement for Q3.
- The user guide, the worked example and the final benchmark records. The SEP becomes
  Final when both native modes agree with their oracles within their bounds on CPU, on
  the A40 and on the AMD GPU.

### 12. Open questions

Decided by the maintainers on 2026-10-25, when this draft was commissioned:
- the two modes, single and mixed;
- noise qualified under mixed precision only;
- SEP 33's routes;
- a fused batch axis measured, not built;
- `numba_cpu` excluded.

Decided by the maintainers on 2026-10-25, on the draft's questions (Part III). What
remains open is what the phases measure within these decisions.

- **Q1 — The mixed floor. Decided:** $10^{-8}$, raised by §7.3's rule. *Open:* the rule's
  outcome (phases 1 and 5).
- **Q2 — Adams under mixed precision. Decided:** the like-with-like rule of §7.3, with
  `Multistep()` becoming the mixed default if admitted. *Open:* the admission (phase 5).
- **Q3 — Adams under single precision. Decided:** refused on native as on `mlx`. Native
  records its binary32 Adams measurements under the like-with-like rule; if phase 5
  admits Adams under mixed precision by that rule, applying the same rule to single
  precision on `mlx` and native is proposed on the roadmap as one small follow-up.
- **Q4 — Noise criteria. Decided:** $N = 64$; the rounding margin of at least 20 dB as
  the primary criterion; the mean spectrum within the larger of 1 % and twice the
  standard error; $\lvert g_{12}\rvert$ within the larger of 0.02 and twice its
  standard error; a refusal only if the study finds a predicate computable from the
  problem, otherwise a documented limit; the rounding-floor estimate recorded in each
  noise-seeded mixed run's metadata (§5).
- **Q5 — The wheels. Decided:** the distributed CPU wheels carry all three modes, unless
  phase 2 measures more than 60 % growth or a size past PyPI's per-file limit (§6.3).
- **Q6 — `auto` under single precision on an Apple-silicon Mac. Decided:** `mlx`, as
  now, falling back to native's CPU build. *Open:* whether phase 4's benchmarks justify a
  size threshold, recorded for the roadmap.
- **Q7 — The mixed propagator on reduced-FP64 GPUs. Decided:** always binary64, with
  §6.4's remedies above a 15 % share of the step (phase 4).
- **Q8 — A fused batch axis. Decided:** the rule of §6.6, measured in samples times
  steps per second on a noise ensemble of small fibre runs (phase 4).
- **Q9 — AMD GPUs. Decided:** qualified in this SEP, not deferred: both modes on the
  HIP build on a rented CDNA 3 GPU, with class R tests, the confirmation cases, the
  probes, benchmarks and per-pass profiles (phase 4), within a budget agreed before the
  phase starts.
- **Q10 — Continuous integration. Decided:** CI runs both modes on the CPU build; the A40
  lane runs locally and the AMD lane on its rental, each part of the definition of done
  of the phases that touch device code, with results recorded in Part II.

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
5. NVIDIA, *cuBLAS library documentation*, "cublasSetMathMode" and
   "NVIDIA_TF32_OVERRIDE", https://docs.nvidia.com/cuda/cublas/ (accessed 2026-10-25).
6. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135–1184 (2006). doi:10.1103/RevModPhys.78.1135
7. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301

## Part II — Implementation record

*One report per phase, appended as each phase is completed (SEP 0 §8).*

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
