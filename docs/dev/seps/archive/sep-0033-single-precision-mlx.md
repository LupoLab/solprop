# SEP 33 — Single-precision propagation, and an MLX backend for Apple GPUs

| | |
|---|---|
| **SEP** | 33 |
| **Title** | Single-precision propagation, and an MLX backend for Apple GPUs |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-23); Opus 5.5 (2026-10-23, revision after review) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-23 |
| **Resolution** | Accepted 2026-10-24 by John Travers after his human review, the GPT-6 AI review and the revision answering it, recorded in Part III. Final 2026-10-25 with phase 4, its open questions moved to the roadmap (Part II, phase 4 report); merging its pull request is the maintainer's approval. |
| **Requires** | [SEP 25](sep-0025-rounding-level-reproducibility.md) (the floating-point policy and the bound classes this extends to binary32) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | John Travers with Opus 5.5 (AI); phases 1–4 on `feature/sep-0033-single-precision`, one pull request |
| **Manual pages** | [Single precision](../../../guide/performance/single-precision.md); [Choosing a backend](../../../guide/performance/backends.md#the-apple-gpu-backend); the worked example [Single precision on an Apple GPU](../../../auto_examples/performance/plot_01_single_precision.rst) |

## Part I — Proposal

*Revised on 2026-10-23 after the human and AI reviews, with the maintainer's decisions on
the review recorded in Part III. The response to the review (Part III) says what changed
for each finding.*

### 1. Abstract

Add a run-level arithmetic setting, `precision="single"`, under which the propagated
state and every per-step pass are held in IEEE binary32, and add a backend, `mlx`, that
executes such runs on Apple-silicon GPUs through Apple's MLX array framework. `numpy_ref`
gains the same single-precision path first and is its oracle, so the proposal also
answers, by measurement against criteria fixed in advance, which of solprop's problems
survive single precision and which do not. The MLX backend covers three routes, in the
order of their use and their measured gain: capillary fibre runs (mode-averaged and
multimode), radial free space and Cartesian free space. The precision setting, its
contract (what stays in binary64, how every buffer is kept in range), the binary32
bounds and the survival criteria are written so that a later single-precision or
mixed-precision mode of `native` or `numba_cpu` reuses them unchanged. Success is judged
by agreement with `numpy_ref`'s single-precision path within binary32 rounding bounds,
by the published survival record against binary64 references, and by recorded
speed-ups over the native CPU backend on the same machine.

### 2. Motivation and scope

**Apple-silicon GPUs are idle.** solprop runs on a Mac's CPU only: the native backend's
device builds target CUDA and HIP, and Kokkos has no Metal backend
([roadmap](../../roadmap.md), "Floating-point options and reduced precision"). An
Apple-silicon Mac has a GPU with several times the CPU's single-precision throughput,
sharing the same memory, and no hardware binary64 arithmetic at all. Using it needs a
backend outside Kokkos, and that backend can only compute in binary32.

**The indicative gain is large on the large problems.** A synthetic measurement
(2026-10-23, an Apple-silicon laptop: 18 CPU cores, a 40-core GPU, MLX 0.32.4) timed one
interaction-picture Kerr source evaluation: the stage combination with three earlier
stages, an inverse transform, $\gamma |E|^2 E$, a forward transform and the finish
$i\,w\,F$, with a real 256 × 256 transverse-transform GEMM each way on the radial shape
and a three-dimensional transform on the Cartesian one. Six evaluations were chained
per synchronisation, as in one DOPRI5 step. The GPU arm compiled the element-wise
passes with `mx.compile` and ran with `MLX_ENABLE_TF32=0`, so every reported row,
including the radial one, used float32 matrix products. The CPU arm used the FFTW
3.3.11 build the native backend links, planned with `FFTW_MEASURE` at the fastest of 1,
4 and 18 threads per shape, and fused parallel loops. The arms alternated within each of
five rounds in one process on a machine under other load (load average 3–9); the table
gives medians, which were within 9 % of the minima.

| shape | CPU binary64, µs | MLX GPU binary32, µs | ratio |
|---|---|---|---|
| one line, $2^{14}$ samples | 161 | 80 | 2.0 |
| one line, $2^{16}$ | 614 | 89 | 6.9 |
| 204 × 2048 (multimode) | 922 | 109 | 8.5 |
| 64 × $2^{14}$ (a batch) | 1773 | 320 | 5.5 |
| radial, 256 × 4096 | 6968 | 512 | 13.6 |
| Cartesian, 384 × 384 × 256 | 96 213 | 21 435 | 4.5 |

The GPU results agreed with binary64 to 6–9 × 10⁻⁷ relative. These figures are
indicative only: the real native backend spends more per evaluation than the synthetic
CPU arm (395 µs at $2^{14}$, from differencing fixed-schedule DOPRI5 runs), a real step
does more work than six bare evaluations on either side, and a tuned binary32 CPU build
would narrow the large-shape ratios by up to the bandwidth factor of two. The phases'
own short-step measurements against `native` replace them (§7). The small single-mode
case gains little: the GPU's floor of about 80 µs per evaluation is launch and
synchronisation latency.

**Single precision needs proving, not assuming.** Whether a supercontinuum, a filament
or a dispersive-wave run survives binary32 depends on the problem: coherent dynamics
with moderate dynamic range should; noise-seeded dynamics at high pulse energy, and
anything that differences nearly equal large quantities, may not (§4). A
single-precision oracle on `numpy_ref`, with survival criteria fixed before any
comparison (§7.3), makes that a measurement with a published record, and the record is
what a later native single-precision mode would need too.

**In scope.**
- The precision setting: `SimulationSettings.precision` and the keyword interfaces,
  `KernelProblem.precision`, the shared single-precision admission (defaults and
  refusals), the recorded configuration and ADR-054 (§9).
- The single-precision contract (§4.2): what is binary32, what is formed in binary64,
  how every narrowed buffer is kept in range (§4.3), and the tolerance floor.
- `numpy_ref`'s single-precision path on the capabilities of §5.1, as the oracle.
- `tests/bounds.py` with its formulas parameterised by the unit roundoff, and the
  binary32 bounds with their limits of use (§7.1).
- The survival study and its manual page, against the criteria of §7.3.
- The `mlx` backend (an optional, macOS-arm64-only extra) on the capabilities and in
  the phases of §5.1, with fixed-schedule lockstep batches through the existing
  `select_batch` interface.

**Not in scope.**
- The general modal route, the hybrid planar kind, coupled flows, the moving solid
  guide, TDSE responses, the dispersive electronic response, the far-field statistics
  and `"cf4"` sections: refused under single precision by every backend, by name.
- Noise models under single precision: refused by every backend, a known limitation
  recorded on the manual page (§4.5). Admitting them is later work.
- A fused batch axis on the GPU (new batch infrastructure), and adaptive or spatial
  batches, which no backend runs.
- Single-precision or mixed-precision modes of `native` and `numba_cpu`. The contract,
  bounds and criteria here are built for them; the modes themselves are a later SEP.
- Storing outputs in binary32 (§9, Q6).
- A Kokkos Metal backend, or SYCL through AdaptiveCpp (§10).

### 3. Context

- **The seam.** A backend implements operations on the propagated state
  (`solprop.backends.protocol.Backend`); the shared step loop
  (`solprop.backends.driver.StepDriver`) owns the march, the controller, landing,
  history and noise, and a backend supplies `_trial_step`, `_apply_mask`, `stats` and,
  optionally, `_finish` and the Adams and low-storage hooks
  ([backends.md §1](../../backends.md), [architecture.md §9](../../architecture.md)).
  Fixed-schedule batches run through `solprop.backends.batch.LockstepRunner`, whose
  `_trial_step` calls each member's trial in turn and needs each member's host norm at
  once.
- **Device-ready stage terms.** A stage's z-dependent arrays cross the seam as fixed
  terms uploaded once and a bundle of host scalars per stage
  (`solprop.translate.stage`); the propagator exponent is
  $\Delta\Phi = \sum_t s_t \Phi_t$, with the weight differences $s_t$ formed on the host
  in binary64. A problem that took the z table is not `device_ready`
  ([kernel-contract.md §3](../../kernel-contract.md)).
- **The responses as they are.** Raman kernels are per particle, with `dt` folded into
  the spectrum (`solprop.translate.raman.kernel_spectrum`), and multiplied by the
  density in the pass (`backends.numpy_ref.passes.raman_polarisation`): a molecular
  kernel's spectrum can be of order $10^{-50}$. The field-grid plasma forms the ionised
  fraction as $f = 1 - (1 - p)\exp(-X)$ and the loss current from $1 - f$
  (`backends.numpy_ref.passes.plasma_scan`, `plasma_current`); the envelope-grid plasma
  keeps the neutral survival separately and evaluates its cycle moments with a
  per-sample singular-value decomposition and adaptive angular quadrature
  (`backends.numpy_ref.envelope_plasma`, `translate.envelope_plasma.cycle_moments`).
- **The floating-point policy.** [SEP 25](sep-0025-rounding-level-reproducibility.md)
  compares independent computations by numerical error, with five comparison classes
  and the bound helpers of `tests/bounds.py`, all written in binary64's `EPS` and `u`
  ([testing.md §3](../../testing.md)).
- **Precision today.** Everything is binary64, and two normative pages say so:
  [architecture.md §1](../../architecture.md) ("Double precision throughout") and
  [conventions.md §6](../../conventions.md) (a complex128 kernel state). The translation
  layer builds its tables in binary64; `numpy_ref` computes with NumPy and `scipy.fft`,
  both of which preserve complex64 inputs; `numba_cpu` and `native` are binary64
  throughout.
- **Earlier decisions.** ADR-002 evaluated JAX as a CPU backend and did not adopt it;
  ADR-020 orders the priorities as accuracy and reliability, then CPU performance, then
  GPU, and builds GPU paths only for what the CPU backends have validated; ADR-051
  keeps no compatibility with earlier configurations or output schemas
  ([decisions.md](../../decisions.md)). §9 and §10 say how this proposal relates to each.
- **The roadmap item** "Floating-point options and reduced precision" lists reduced
  precision as the one floating-point option with a large expected gain; this SEP takes
  up its MLX and entirely-single-precision bullets for the routes in scope.

### 4. Theory

#### 4.1 Binary32 rounding in an interaction-picture step

Binary32 has unit roundoff $u_{32} = 2^{-24} \approx 6.0 \times 10^{-8}$, smallest
normal number $2^{-126} \approx 1.2 \times 10^{-38}$ and largest
$\approx 3.4 \times 10^{38}$; binary64 has $u_{64} = 2^{-53}$. The bounds of
`tests/bounds.py` are not all linear in $u$: Higham's $\gamma_k = k u / (1 - k u)$ has
a denominator that matters at binary32, so each formula is re-evaluated at $u_{32}$
rather than scaled from its binary64 value (§7.1).

Write a stage of the interaction-picture step as
$k = \mathcal{N}(e^{\Delta\Phi}(A + \Delta z\,\textstyle\sum_j a_j k_j))$, where
$\mathcal{N}$ is the source with its transforms. Four rounding contributions matter;
two are random in sign from step to step and two are coherent.

1. **The source** (random). A transform pair on $n$ samples has relative normwise error
   of order $u \log_2 n$ [3], and the pointwise response adds a few $u$. The error
   enters the state scaled by $\Delta z$: per step, relative to the state, about
   $c\,u\,\log_2 n \cdot \phi_{\mathrm{NL}}$, where
   $\phi_{\mathrm{NL}} \approx \Delta z\,\lVert\mathcal{N}\rVert / \lVert A \rVert$ is the
   step's nonlinear phase.
2. **The state** (random). Holding the state in binary32 rounds it by up to $u$
   relative at every accepted step. *If* these roundings are independent and unbiased
   and the dynamics do not amplify them, they grow like $u\sqrt{M}$ over $M$ steps;
   this is a heuristic, not a bound. Amplifying dynamics (modulation instability,
   soliton fission) multiply it by the propagation's sensitivity (the growth factor of
   `propagation_growth`), which then dominates.
3. **The linear propagator** (coherent). For a constant operator and step, the
   propagator $e^{\Delta\Phi}$ is the same complex number at every step, so its
   rounding error is applied with the same sign $M$ times and grows *linearly*. If
   $\Delta\Phi$ itself is rounded to binary32, the propagator's relative error is about
   $u\,|\Delta\Phi|$, and at the band edges $|\Delta\Phi|$ per step reaches
   $10^{2}$–$10^{4}$ rad: a scalar witness with $\Delta\Phi = -1234.56789\,i$ gives a
   relative error of $1.9 \times 10^{-5}$ after one multiplication and
   $1.9 \times 10^{-1}$ after $10^4$, against binary64. If instead $\Delta\Phi$ is
   formed, range-reduced and exponentiated to binary64 accuracy and only the propagator
   is rounded, the error is about $u$ per step: $1.9 \times 10^{-8}$ after one
   multiplication, $1.9 \times 10^{-4}$ after $10^4$. The contract takes the second
   form (§4.2); its remaining drift, about $c\,M\,u_{32}$ per occupied bin, is a
   budgeted error of the method (§7.3).
4. **The coefficients** (coherent). Every table is rounded once at upload, so each
   coefficient (a Kerr prefactor, a Raman kernel sample, a rate) carries a fixed
   relative error of up to $u$. Its effect is a systematic error of the order of $u$
   times the accumulated effect of that term, for example $u\,\Phi_{\mathrm{NL}}$ for a
   total nonlinear phase $\Phi_{\mathrm{NL}}$: small, but coherent, and budgeted with
   item 3.

**The controller sees only part of this.** An embedded estimate is
$\Delta z \sum_i (b_i - \hat b_i) k_i$. It measures the method's truncation error and
items 1–2 roughly, at about $u\,\phi_{\mathrm{NL}}$ relative to the state, so the
controller does not stall. It is blind to items 3–4: with no nonlinear source the
estimate is zero whatever the propagator's error. Those are therefore bounded by the
contract and measured by dedicated tests (§7.2), not by the tolerance.

**The floor is on the useful tolerance.** A local error target below the per-step
rounding of items 1–2 buys steps without accuracy. Taking the per-step rounding as
$c\,u_{32}$ with $c$ of order 10 gives a floor of $\mathrm{rtol} \ge 10^{-6}$ (Q1). The
study may raise it to $10^{-5}$ by the rule of §7.3.

#### 4.2 The single-precision contract

Under `precision="single"`:

- **Binary32:** the propagated state, the stage values, every transform, the responses
  (Kerr, Raman, plasma), the transverse transforms (the Hankel GEMM, the Cartesian
  transforms), the stage combinations, the error estimate's array and the statistics'
  pointwise work.
- **Binary64 on the host:** $z$, $\Delta z$, the controller and its norms' final
  scalar, the stage weight differences $s_t$ and every other bundle scalar, landing,
  schedules and save planes, and every combination of scalar coefficients and scale
  exponents (§4.3) before it is rounded once to binary32.
- **Binary64-accurate, then rounded:** the stage propagator $e^{\Delta\Phi}$: the
  combination $\sum_t s_t\Phi_t$, its range reduction and its exponential are evaluated
  to binary64 accuracy, and only the propagator is rounded (§4.1, item 3). On
  `numpy_ref` this is binary64 arithmetic on binary64 terms; on `mlx`, which has no
  binary64, the terms and weights are held as double-float pairs and the combination
  and range reduction are evaluated in double-float arithmetic in one small kernel,
  leaving binary32 `exp`, `cos` and `sin` of reduced arguments.
- **Built once in binary64 and narrowed at upload:** every table of the translation
  layer (operator terms, prefactors, masks, Raman kernels, rate tables, quadrature and
  transform matrices), each narrowed with its own exponent (§4.3). The translation
  layer does not change.
- **Stable forms:** the field-grid plasma's fraction as
  $f = p + (1 - p)\,[-\mathrm{expm1}(-X)]$ and its neutral survival as its own array,
  $\sigma = (1 - p)\,e^{-X}$, with the loss current formed from $\sigma$, never from
  $1 - f$. In binary32 at $p = 0$, $X = 10^{-8}$, the present subtraction gives $f = 0$
  and the stable form $10^{-8}$; at $X = 20$, $1 - f$ rounds to zero while
  $\sigma = 2.06 \times 10^{-9}$. These forms are part of the single-precision path
  only; the binary64 path keeps its present arithmetic and results (§9).
- **Accumulated, not summed naively:** every long reduction or running integral (the
  norms, the statistics, the plasma's running integrals over time, the field grid's
  band-exact running sums of
  [SEP 18](sep-0018-band-exact-plasma-integrals.md)) uses a pairwise or blocked
  order with the bound of its tree, or binary32 compensated summation, and states that
  bound: a sequential binary32 sum of $10^4$ terms could lose
  $10^4 u_{32} \approx 6 \times 10^{-4}$ of its value. The orders are the oracle's
  (§6.2), so the other backends have a defined target (Q5).
- **Scaled:** the state at an exact power-of-two scale, every narrowed buffer with its
  own exponent, and every sum of squares scaled before squaring (§4.3).
- **Never:** flush-to-zero imposed by solprop, approximate transcendental functions,
  reduced-precision matrix products (TF32 or lower), or any fast-math flag, as
  SEP 25 §5 requires. Subnormals a GPU flushes in hardware are accepted within the
  absolute budget of §4.3.

This is the contract `numpy_ref` implements as the oracle and `mlx` reproduces; how
`mlx` establishes each item on its runtime is §6.3. A later mixed mode (binary32
source, binary64 state) differs in the first two items only and needs a separate
precision value, not a different contract (§6.1).

#### 4.3 Range: every buffer, every intermediate

**Why the state alone is not enough.** Pointwise work on the field fits in SI units: a
field of $10^{12}$ V/m has $E^3 = 10^{36}$, and $E^3$ overflows only above
$7 \times 10^{12}$ V/m (about $6 \times 10^{18}$ W/cm²). But other quantities do not:

- *Reductions.* The transforms are unnormalised, so on the Cartesian route
  ($n \approx 4 \times 10^7$ samples, fields near $10^{11}$ V/m) a spectral sample
  reaches about $4 \times 10^{18}$, its square $10^{37}$, and an energy sum over the
  grid $10^{44}$: an overflow.
- *Coefficient buffers.* A molecular Raman kernel's spectrum (nitrogen at 293 K,
  $2.5 \times 10^{25}$ m⁻³, a 2 ps field grid) has a largest magnitude of about
  $5 \times 10^{-50}$: every bin becomes zero in binary32, and no later prefactor can
  recover it.

So the rule is per buffer and per intermediate, not per state.

**1. Block exponents for narrowed buffers.** Every array narrowed to binary32 at upload
is stored as $\tilde B = B \cdot 2^{-e_B}$, with the integer $e_B$ chosen so that
$\max |\tilde B| \in [1, 2)$ (zero for an all-zero buffer). Multiplying by a power of
two is exact, so $\tilde B$ is $B$ rounded once, with no range loss for entries within
$2^{-126}$ of the buffer's largest. Each exponent is carried as a host integer.

**2. A power-of-two state scale.** The state is held as $\tilde a = a / s$, with
$s = 2^k$ chosen from the *launch* (`KernelProblem.initial`) so that its peak temporal
amplitude over $s$ lies in $[1, 2)$, and $s = 1$ for an identically zero launch. Being
a function of the problem, not of the state at `set_initial`, the scale is the same for
a run, its restarts and its resumes (§5.3).

**3. Coefficients combined on the host.** Each pass multiplies narrowed buffers and
state by scalar coefficients: the density, the Raman amplitude, a prefactor's
constant, the state scale's powers, and the buffers' exponents. These are combined into
one coefficient per buffer product in binary64 on the host, as a binary64 mantissa and
a separate integer exponent, and the combination is applied at the point in the pass
where the intermediate it multiplies is largest, so that no intermediate leaves
binary32's normal range. The Raman pass, for example, forms
$\tilde K \cdot \widetilde{|E|^2}$ in the spectral domain with both factors near one,
transforms back, and applies the combined coefficient
$\rho\,\cdot 2^{e_K} \cdot s^2 \cdot (\text{constant})$ once to the product with
$\tilde E$. Each pass declares, in a comment, the largest and smallest magnitude of its
intermediates for the admitted inputs.

**4. The scaled equations.** With $E = s\tilde E$ the physical field, each response
term carries its homogeneity factor, folded into its combined coefficient:

| term | dependence on $E$ | factor on $\tilde a$'s equation |
|---|---|---|
| linear operator, masks, absorbers, collars | linear | 1 |
| Kerr (constant), third-harmonic term on envelope grids | cubic | $s^2$ |
| scalar and tensor Raman | cubic | $s^2$ |
| plasma rate $W$, fraction $f$, survival $\sigma$ | of $\lvert E\rvert$ | evaluated at $s\lvert\tilde E\rvert$ (one exact multiplication); unscaled |
| free current $\propto \int f E\,dt$ (field and envelope) | linear at a given population | 1 |
| loss current $I_p W \sigma E / \lvert E\rvert^2$ (field and envelope) | inverse at a given rate | $s^{-2}$ |
| held-current streams (SEP 27) | as the stream it holds | the free or loss factor, per stream |

The free and loss streams thus take different factors, and assembly sums them only
after each has its own; the loss term's division is masked where the rate is exactly
zero, as now, so no tiny $|E|^2$ is divided.

**5. Statistics, tolerances and observation.** Each statistic declares its homogeneity
degree $k_s$ in the state, and the host multiplies its binary32 result by $s^{k_s}$ in
binary64:

| statistics | $k_s$ |
|---|---|
| field observations, on-axis fields | 1 |
| power, energy, intensity, fluence, spectral energy density, photon number | 2 |
| widths, durations, centroids, bandwidths, positions, ratios, ionised fraction, electron density | 0 |

The controller's absolute tolerance is divided by $s$; `observe` and snapshots multiply
the state and every stored row by $s$ (§5.3).

**6. Scaled reductions.** Every sum of squares divides by a power of two chosen from
the largest magnitude before squaring, as LAPACK's `nrm2` does [2], and the host
multiplies back in binary64.

**7. Underflow budget.** An entry flushed to zero, by narrowing or by the GPU's
hardware, has absolute error at most $2^{-126} \cdot 2^{e}$ in a buffer of exponent
$e$, that is at most $2^{-126} \approx 10^{-38}$ relative to the buffer's largest
entry, and a normwise contribution of at most $\sqrt{n}\,2^{-126}$ for $n$ entries:
negligible beside $u_{32}$ wherever the buffer enters a product or a sum. It is not
negligible where a quantity is *divided* by a flushed value or where a vanishing
response is the result; the first is excluded by the masked divisions above, the second
is tested (§7.2). Rates below the flush threshold of the rate buffer (about
$10^{-22}$ s⁻¹ for a buffer whose largest rate is $10^{16}$ s⁻¹) ionise less than
$10^{-34}$ of the gas in a picosecond.

**Exactness and its limits.** Multiplying by a power of two commutes with rounding when
nothing overflows or becomes subnormal, so a pass whose operations are the same and
differ only by power-of-two factors gives identical results scaled or unscaled. Where
the combined coefficients regroup a product of non-power-of-two scalars (a density
times a prefactor, formed in binary64 and rounded once, against the unscaled path's two
binary32 roundings), the rounding sequence differs, and scaled against unscaled is a
class R comparison, not class P (§7.2).

#### 4.4 Plasma populations in binary32

The stable forms of §4.2 follow from writing the neutral survival
$\sigma = (1 - p)\,e^{-X}$, with $X = \int W\,dt$ the exposure, and the fraction
$f = 1 - \sigma = p + (1 - p)\,(1 - e^{-X})$; $1 - e^{-X} = -\mathrm{expm1}(-X)$ keeps
relative accuracy for small $X$, and $\sigma$ keeps relative accuracy for large $X$,
where $f$ rounds to one. $df/dt = W\sigma$, so the loss current
$J_{\mathrm{loss}} = I_p W \sigma E/|E|^2$ and the ionisation work
$\int E\cdot J_{\mathrm{loss}}\,dt = I_p (f_{\mathrm{end}} - p)$ are both formed from
$\sigma$. The exposure $X$ itself is a running integral under the rules of §4.2. The
envelope grid's path already holds the survival separately; its cycle moments are
evaluated in binary64 on the host by `numpy_ref` (§6.2) and, on `mlx`, by the
formulation of phase 4 (§6.3).

#### 4.5 Noise

A one-photon-per-mode seed (`noise`) has a spectral amplitude relative to the pulse's
peak spectral amplitude of roughly $\sqrt{\hbar\omega_0\,n_{\mathrm{occ}}/E_p}$, with
$n_{\mathrm{occ}}$ the occupied bins: about $1.6 \times 10^{-5}$ for a 1 µJ pulse over
1000 bins, $5 \times 10^{-7}$ at 1 mJ. On the heuristic of §4.1, accumulated binary32
rounding reaches $u_{32}\sqrt{M} \approx 6 \times 10^{-6}$ after $10^4$ steps.
Noise-seeded statistics might survive at low energy and short lengths, and would fail
at high energy. This SEP does not qualify that boundary: a noise model under single
precision is refused by every backend, and the manual page records it as a known
limitation (Q3).

### 5. Proposed interface

```text
from solprop.driver import prop_capillary

out = prop_capillary(
    ...,                       # the physics, unchanged
    precision="single",        # "double" (default) or "single"
    backend="mlx",             # or "numpy_ref"; "auto" resolves to "mlx" when installed
    rtol=1e-5,                 # at or above the floor, 1e-6
    noise=None,                # noise models are refused under single precision
)                              # the integrator defaults to RungeKutta() here
# the configuration records precision="single"; out.fields() returns binary64 arrays
```

- **`precision`**: `Literal["double", "single"]` on `SimulationSettings`, `Simulation`
  and every keyword interface that takes `backend`. The default is `"double"`, and
  nothing about a double-precision run's numbers changes.
- **`backend="mlx"`** requires `precision="single"` and refuses `"double"` by name,
  saying that Apple GPUs have no binary64 arithmetic. It refuses any problem outside
  its capabilities (§5.1), naming the backends that run it.
- **`backend="auto"`** with `precision="single"` resolves to `"mlx"` when MLX and a
  Metal device are present, and otherwise refuses, naming `numpy_ref` (the oracle, not
  a production backend) and the `mlx` extra. It never resolves to a backend that would
  run in binary64, and never to `numpy_ref`.
- **`numba_cpu` and `native`** refuse `precision="single"` by name until a later SEP.
- **Tolerances**: an explicit `rtol` below the single-precision floor, $10^{-6}$ (Q1),
  is refused with the floor in the message. Under single precision the default `rtol`
  of every interface is the floor, stated in its docstring; no explicit value is
  raised silently.
- **Integrator**: under single precision the default integrator is `RungeKutta()`
  (DOPRI5), stated in the docstrings, until the Adams integrator is admitted (Q2); an
  explicit `Multistep` is refused until then. The low-storage march is admitted.
- **Noise**: a noise model under single precision is refused by name, as a known
  limitation (Q3).
- **Outputs**: `observe` returns binary64 host arrays, so files, processing and
  plotting are unchanged (§5.3).
- **Memory**: `estimate_memory` and the memory ledger count binary32 bytes and, on
  `mlx`, its graph, cache and compiled workspaces (§6.3); `available_device_memory` on
  `mlx` reports the Metal device's recommended working set less MLX's active memory.
- **Batches**: `backend="mlx"` implements `select_batch` with member-owned stages; a
  member agrees with its solo run within a binary32 matched-discrete bound.
- **Failure**: a non-finite norm, state or statistic stops every run — adaptive, fixed
  schedule and low-storage — with `ConvergenceError`, as in binary64.

#### 5.1 Capabilities by backend and phase

"Phase $n$" is the phase in which the capability is admitted; "refused" means refused by
name for the whole of this SEP.

| capability | `numpy_ref` single | `mlx` |
|---|---|---|
| fibre route, mode-averaged and multimode: gas capillaries, solid-core and GNLSE runs (`prop_gnlse`, `prop_capillary`, `prop_waveguide`), field and envelope grids | phase 1 | phase 2 |
| constant Kerr, the envelope grid's third-harmonic term | phase 1 | phase 2 |
| scalar and tensor Raman, gas Raman leaves and mixtures, the silica bundle | phase 1 | phase 2 |
| field-grid plasma, including the held current | phase 1 | phase 2 |
| envelope-grid plasma | phase 1 | phase 4; refused until then |
| loss, both absorber schemes, the polarisation taper | phase 1 | phase 2 |
| device-ready gradients (constant and expansion representations) | phase 1 | phase 2 |
| the z table (not `device_ready`) | phase 1: evaluated in binary64, rounded | refused, naming the table's size, as on the native device builds |
| radial free space, its collars and statistics | phase 1 | phase 3 |
| Cartesian free space, including a line ($n_y = 1$) | phase 1 | phase 3 |
| Runge–Kutta, the low-storage march | phase 1 | phase 2 |
| Adams | study only (§7.3) | phase 4, if admitted |
| fixed-schedule lockstep batches (fibre) | phase 1 | phase 2 |
| default statistics, periodic statistics, on-axis statistics | phase 1 | phases 2–3, with their routes |
| noise; dispersive electronic response; general modal; planar; coupled flows; moving guide; TDSE; far-field statistics; `"cf4"` | refused | refused |

#### 5.2 Admission

One shared admission function applies the single-precision defaults and refusals of
§5 before any backend is constructed, so every interface and backend agrees.
`KernelProblem.validate` refuses the routes and responses that no backend runs under
single precision; each backend's own admission then refuses what §5.1 does not yet
admit on it. The study (§7.3) reaches capabilities that public admission refuses, such
as the Adams integrator, through a private test-only argument of the admission
function, never through a public interface.

#### 5.3 Configuration, resume and snapshots

- **The configuration.** The precision is a field of the run's hashed configuration,
  present for every run, double included, as ADR-051 requires of every serialised
  field; the output schema version changes in phase 1. A file of an earlier output
  schema is never resumed (ADR-051), so no meaning is assigned to its absence.
- **Resume.** A resume in the other precision is refused, through the configuration's
  hash and with a message naming both precisions. A resume in the same precision on
  another backend (`numpy_ref` single to `mlx`, or back) is admitted and is a class T
  comparison.
- **The scale across a resume.** The scale is a function of the launch (§4.3), so a
  resumed run recomputes the same $s$ from the same problem, whatever the peak of the
  state it resumes from.
- **Snapshots.** A `MarchSnapshot` stores the state, the stage and source rows, the
  history and compensation rows, all homogeneous of degree one in the state. Each is
  stored unscaled and widened to binary64 on the way out (multiplied by $s$, exact) and
  divided by $s$ and narrowed on the way in (exact for values that were binary32), so a
  snapshot round trip is class P. A row of another homogeneity, if a later phase adds
  one, declares its degree as the statistics do. The Adams history's layout is
  revisited when Adams is admitted (phase 4).

### 6. Implementation strategies

#### 6.1 General support (outside the backends)

- **The setting.** `SimulationSettings.precision` and the keyword interfaces pass a
  `Precision` record into `assemble`, stored as `KernelProblem.precision`: a frozen
  record with named fields (`state`, `source`), each `"double"` or `"single"`, the
  public strings mapping to equal fields. A later mixed mode is then a third public
  value, not a schema redesign. `SCHEMA_VERSION` changes in phase 1.
- **Admission and validation** as §5.2.
- **The translation layer is unchanged.** Tables stay binary64; narrowing with block
  exponents is each backend's upload, through one shared helper. This keeps the change
  out of the physics and translation layers and is what a native single mode would also
  do.
- **Range helpers.** The block exponent of a buffer, the state scale of a launch, the
  combination of coefficients and exponents, and the scaled sum of squares (§4.3) are
  small shared helpers in the translation layer's backend-facing utilities, so
  `numpy_ref`, `mlx` and a later native mode compute the same exponents.
- **Bounds** as §7.1.
- **Configuration, metadata and resume** as §5.3.
- **Memory.** `estimate_memory` takes the element sizes from the precision, and a
  backend may add its runtime's own terms (§6.3).
- **ADR-054** (§9), and the normative pages it amends (§8).

#### 6.2 `numpy_ref`

The oracle keeps one function per pass and no fusion. Under single precision it:

- narrows uploaded terms and tables to float32/complex64 with their block exponents,
  and combines coefficients as §4.3 prescribes;
- holds the state at the launch's power-of-two scale;
- forms the stage propagator from binary64 terms in binary64 and rounds it (§4.2), and
  evaluates a z table in binary64 and rounds its result;
- uses the stable population and survival forms of §4.4 on the field grid;
- evaluates the envelope grid's cycle moments in binary64 on the host from the
  binary32 field, rounding the results, with the existing decomposition and adaptive
  quadrature;
- evaluates reductions pairwise and running integrals (including the band-exact
  running sums) as a blocked, compensated scan, each with its stated bound, rather than
  with `np.sum`'s and `np.cumsum`'s orders: the oracle's orders are the target the other
  backends are compared against (Q5);
- relies on NumPy's and `scipy.fft`'s preservation of complex64, and on NumPy 2's rule
  that Python scalars do not promote, for the remaining arithmetic;
- asserts the dtype of the state and of every stage value after each pass, so that a
  binary64 array that slips into a binary32 pass fails a test instead of silently
  computing in binary64;
- has a private study switch that also evaluates the propagator from a binary32
  $\Delta\Phi$, for the record of §7.2 only.

Most of the package's explicit binary64 dtypes (about 740) are in the physics and
translation layers, which do not change; `numpy_ref`'s files carry about fifty, and the
shared driver a handful.

#### 6.3 The `mlx` backend

**Layer and package.** A new backend package, `solprop.backends.mlx_gpu` (not `mlx`,
which would shadow the library inside the package), registered as `"mlx"` with the
extra `solprop[mlx]`. Nothing outside it imports MLX; availability is probed with
`importlib.util.find_spec`, as for every backend.

**Structure.** It subclasses the shared `StepDriver` like `numpy_ref`, so the march,
controller, landing, history and failure handling are the shared ones. Its passes follow
`numpy_ref`'s in MLX, which is NumPy-like:

- stage combination, responses and finish as `mx.compile`d functions, which fuse the
  element-wise work into one kernel per pass;
- the stage propagator from double-float terms in one custom kernel (§4.2);
- transforms by `mx.fft` (complex64 only; real transforms for field grids);
- the radial route's Hankel transforms as real GEMMs on the real and imaginary parts;
- the Cartesian route's two-dimensional transverse transforms by `mx.fft.fftn`;
- the block exponents, state scale and combined coefficients of §4.3, shared with the
  oracle through one helper;
- the error norm and the statistics reduced on the device to a few scalars, scaled
  before squaring and copied to the host once per trial or plane;
- compensated scans for the plasma's running integrals, and the stable population
  forms, as custom kernels where MLX's own operations do not give the oracle's bound.

**The arithmetic policy.** Each protected operation of §4.2 has a stated mechanism and
evidence, separate from the TF32 admission:

| operation | mechanism | evidence |
|---|---|---|
| matrix products | `MLX_ENABLE_TF32=0` (below) | a construction probe on each GEMM dispatch family the run uses (its shapes), against the discriminating bound of §7.1 |
| stage propagator | custom kernel, double-float combination and range reduction, MLX's documented `"safe"` math mode | constant-operator and occupied-band tests over several steps (§7.2) |
| `expm1`, survival, compensated sums and scans | custom kernels in `"safe"` mode | adversarial tests: small and saturated exposures, a compensation that must retain a known lost increment (which fails under reassociation), against the oracle |
| finite-value detection | MLX's `isfinite` on the norm and the state | injected NaN and infinity in each pass, in fixed, low-storage and adaptive runs |
| transforms | MLX's FFT, complex64 | a characterisation of the error of each transform length and kind the routes use, against `fft_bound` at $u_{32}$, in the test suite, and a construction probe at the run's own lengths |
| other element-wise work in compiled functions | `mx.compile` exposes no math-mode setting | class R against the oracle on every pass, and the special-value and tiny-response tests of §7.2 |

If a probe fails, or a protected operation cannot be established on the installed MLX
and device, the backend refuses to run, naming the operation and what to do.

**TF32 and an initialised MLX.** MLX 0.32.4 on that laptop's GPU computes float32 matrix
products in TF32 by default (a relative error of $7.7 \times 10^{-4}$ was measured on
256 × 256 products, against $2.9 \times 10^{-7}$ in float32), and its documentation says
`MLX_ENABLE_TF32=0` keeps them in float32 and that many settings are read when a
subsystem is first initialised. The backend sets the variable when its package is
imported, if it is unset. If it is set to another value, or if MLX was already
initialised in the process with TF32 enabled, the GEMM probe fails and the backend
refuses, telling the user to set `MLX_ENABLE_TF32=0` before MLX is first used (in the
shell, or before any `import mlx`).

**Evaluation, memory and lifetimes.** MLX is lazy: a stage builds a graph and nothing
runs until `mx.eval`. A graph holding a whole step can retain intermediates an eager
pass would release, beside MLX's allocator cache and compiled workspaces. The backend
therefore:

- evaluates at least once per trial step (when the controller needs the norm), and
  also at stage or node-chunk boundaries when the step's estimated live bytes exceed a
  budget, so that lifetimes are bounded by a stage or a chunk;
- sets MLX's cache limit from the admission estimate and counts the cache and compiled
  workspaces in its memory ledger;
- measures its peak memory (MLX's peak-memory counter) on the benchmark cases and
  checks it against `estimate_memory` in the tests: the estimate must not be below the
  measured peak.

**Batches.** The existing `LockstepRunner` asks each member for its trial and its host
norm in turn, so a synchronising member trial cannot queue all members first. `mlx`
supplies its own batch runner, a subclass that keeps the public `select_batch`
interface, the members' independent states, ledgers and statistics, and splits each
member's trial into building its graph and reading its norm: it builds every member's
trial, evaluates all norms together, then hands each member its norm. The gain over
solo runs is measured in phase 2; the batch is kept for parity whatever it is, and the
record says so (Q10).

**Envelope-grid plasma.** The oracle's per-sample decomposition and adaptive quadrature
do not map onto a compiled graph without host decisions. Phase 4 qualifies a device
formulation: a closed-form decomposition of each sample's two-component field into its
polarisation ellipse (batched, stable for circular, nearly linear and degenerate
fields), a fixed quadrature with an error estimate per sample, and a device status flag
that reports a refinement failure or a cutoff crossing to the host, which refuses the
step as the oracle refuses. Because its discretisation differs from the adaptive
reference, it is qualified as a discretisation (class T, against the oracle's adaptive
moments), not as rounding. Until then `mlx` refuses envelope-grid plasma by name.

**Cost.** Binary32 halves the bytes per resident element against `numpy_ref`; the peak
is the measured one above. The GPU and CPU share memory, so a host copy for `observe` or
`stats` is a copy, not a transfer. Kernel launch latency sets a floor of tens of
microseconds per stage, which is why the small single-mode case gains little.

**Testing without a GPU.** MLX also runs its graphs on the CPU. The backend accepts an
internal device choice, so in continuous integration its tests run on MLX's CPU device,
which checks the graph code; the custom kernels' tests are GPU-only. A local GPU lane,
the whole `mlx` test suite on the Metal device, is part of each phase's definition of
done, as the device lanes are for the native builds. Phase 2 checks once whether a
hosted macOS runner exposes a usable Metal device and records the result (Q7).

#### 6.4 Paving the way for `native` and `numba_cpu`

What this SEP builds and a later native or numba single mode would reuse unchanged: the
public setting, the `Precision` record and its schema field, the admission function,
the contract of §4.2 with its range rules and stable forms, the binary32 bounds, the
oracle path, the survival criteria and record, and the tolerance floor. What it would
add on its own: templating or recompiling its kernels on the real type, Kokkos-FFT and
Kokkos Kernels in single precision (both support it), binary64 propagators (which a CPU
or NVIDIA build can compute natively), and its own protected blocks. A mixed mode adds a
third precision value.

#### 6.5 Dependencies

`mlx` (MIT licence, compatible with GPL-3.0-or-later), as an optional extra with the
environment marker `sys_platform == "darwin" and platform_machine == "arm64"`, pinned
to a tested minor version (`mlx>=0.32.4,<0.33` at drafting) and added to
[dependencies.md](../../dependencies.md), with the macOS floor taken from MLX's wheels.
The maintainer has agreed to it as an optional extra. A pin is raised only in a pull
request that runs the GPU lane and the arithmetic probes (Q9).

### 7. Verification and validation

#### 7.1 Bounds at binary32

- **Formulas, not factors.** `tests/bounds.py` takes the unit roundoff as a parameter
  (default binary64) in every formula — `gamma`, `higham_gamma`, `fft_bound`,
  `scan_bound`, `prefix_bound`, `stencil_bound`, `step_budget`, `propagation_growth` —
  and adds `UNIT_ROUNDOFF_32`. Existing call sites are unchanged. Each formula states
  its domain ($k u < 1$ for $\gamma_k$) and raises outside it.
- **What the bounds include.** The narrowing of every input table (one rounding per
  entry, entering as one more operation per use); signed sums bounded through the sum
  of magnitudes, so that a cancelling sum's relative bound carries its condition
  number; and the propagator's per-step error of §4.1.
- **Where a bound stops being useful.** At binary32 the established `fft_bound` margin
  is already $8.5 \times 10^{-4}$ at $2^{14}$ samples, before source counts and growth.
  A class R bound larger than a tenth of the case's class T envelope (§7.3) is
  uninformative: that comparison runs on a shorter window or fewer steps until its
  bound is below a tenth, and the test says so. Backend agreement within a class R
  bound is never a survival verdict.
- **The TF32 probe's bound.** The probe multiplies Gaussian random factors of each
  dispatch family's shapes and requires a normwise relative error below $2^{-17}$
  ($7.6 \times 10^{-6}$): float32 products of inner dimension up to $10^3$ are expected
  near $\sqrt{n}\,u_{32} \le 2 \times 10^{-6}$ (measured $2.9 \times 10^{-7}$ at 256),
  TF32 near its own unit roundoff $2^{-11}$ (measured $7.7 \times 10^{-4}$), so the
  threshold separates the two by more than a factor of four on each side.

#### 7.2 Class R, class P and the targeted tests

- **`mlx` against `numpy_ref`**, both single precision, on fixed schedules with a
  fixed-order method, every capability of §5.1 in its phase, within the binary32
  bounds of §7.1.
- **Batch members** against their solo runs, and repeated runs against themselves,
  within the same bounds.
- **The oracle path.** `numpy_ref` single against `numpy_ref` double for one step,
  within a bound in $u_{32}$.
- **Scaling.** Scaled against unscaled responses and propagation, separately: class P
  where the operation sequences differ only by power-of-two factors (§4.3), class R
  where coefficients are regrouped; a Cartesian case whose unscaled energy sum would
  overflow binary32 runs and reports its energy within its bound; the zero launch; a
  resume whose state peak differs from the launch's keeps the launch's scale.
- **Vanished responses.** A molecular Raman case (nitrogen at 293 K and
  $2.5 \times 10^{25}$ m⁻³ on a 2 ps field grid, whose kernel spectrum peaks near
  $5 \times 10^{-50}$) gives a nonzero Raman polarisation agreeing with binary64 within
  its bound, on `numpy_ref` and on the GPU; every pass reports its intermediates'
  largest and smallest magnitudes in the survival record.
- **Plasma populations.** Small exposure ($X = 10^{-8}$), nonzero pre-ionisation and
  saturation ($X = 20$) against independent witnesses: the free-electron density
  $\rho f$, the current, and the ionisation work $I_p\rho(f_{\mathrm{end}} - p)$
  against $\int E\cdot J_{\mathrm{loss}}\,dt$.
- **The propagator.** Constant-operator runs with no source and an occupied band, over
  several step sizes and $10^2$–$10^4$ steps, against the exact $e^{M\Delta\Phi}$: the
  error grows no faster than $c\,M\,u_{32}$ with the $c$ the record states; the study
  switch's binary32-$\Delta\Phi$ variant is recorded beside it.
- **Arithmetic probes.** The GEMM probe on each dispatch family; the adversarial tests
  of §6.3's table; the dtype guards of `numpy_ref`; a test that no float64 array
  reaches an `mlx` pass; a pre-initialised MLX with TF32 on is refused.
- **Resume and configuration.** Same-precision resume across `numpy_ref` and `mlx`
  (class T); a snapshot round trip (class P); a resume in the other precision refused;
  the precision present in every configuration's hash.
- **Failure.** A non-finite value injected into each pass stops fixed, low-storage and
  adaptive runs with `ConvergenceError`.
- **Memory.** The measured peak on the benchmark cases does not exceed the estimate.

#### 7.3 Class T: the survival study

**Criteria, fixed before any comparison.** Phase 1 records these in Part II before its
first single-precision comparison; they are the maintainer's (2026-10-23) and are not
fitted to results.

- **References.** For each case, a binary64 run on the *same* grid, discretisation,
  plasma rule and quadrature, at $\mathrm{rtol} = 10^{-10}$, so that grid and
  discretisation errors are common to both sides and the comparison isolates
  precision. Its own accuracy is established by a second binary64 run at
  $\mathrm{rtol} = 10^{-9}$: their difference must be below a tenth of every envelope
  below, or the reference is tightened until it is.
- **Metrics and envelopes.** Against the reference, at the end of the window:

  | metric | envelope |
  |---|---|
  | field, normwise relative | $10^{-3}$ |
  | spectral energy in each of 32 equal bands of the active band, for bands above −40 dB of the largest | 1 % |
  | total energy | $10^{-4}$ relative |
  | peak power (fibre) or peak intensity (spatial) | $10^{-3}$ relative |
  | final ionised fraction, where above $10^{-6}$ | $10^{-3}$ relative |
  | beam width (radial, Cartesian) | $10^{-3}$ relative |

- **Verdicts.** *Survives*: every metric within its envelope at the floor and at
  $\mathrm{rtol} = 10^{-5}$. *Survives with a limit*: within its envelopes only over a
  stated shorter length, a stated lower energy or a stated looser tolerance, recorded
  as such. *Fails*: otherwise. A failure becomes a refusal where the failing condition
  can be detected from the problem before the run, and a documented limitation where it
  cannot.
- **The floor (Q1).** The floor rises from $10^{-6}$ to $10^{-5}$ if, on every case,
  the error at $\mathrm{rtol} = 10^{-6}$ is not smaller than at $10^{-5}$ by more than
  25 % in any metric.
- **Adams (Q2).** Through the study's private admission (§5.2), the Adams integrator is
  admitted in phase 4 only if, on every case, its error at the same `rtol` is at most
  twice Runge–Kutta's in every metric and within every envelope.
- **The propagator drift.** The constant-operator record (§7.2) states $c$ and the step
  count at which $c\,M\,u_{32}$ reaches the field envelope; the manual page gives that
  count as a limit.

**Cases.** Reduced windows in tests; the full-length comparison is a worked example,
run by hand.

| case | route | expected |
|---|---|---|
| soliton and SPM, dispersion-free analytic phase | fibre, envelope | survives |
| reduced resonant dispersive-wave emission in helium, two modes, gradient | fibre, field | survives |
| reduced comb-to-continuum, noise-free | fibre, field | survives with a limit |
| argon plasma blue shift | fibre, field and envelope | to measure |
| molecular Raman in nitrogen | fibre, field | to measure |
| radial Kerr beam and a short plasma filament | radial | to measure |
| Cartesian Gaussian diffraction and a Kerr beam | Cartesian | survives |
| the dispersive-wave and plasma cases under the Adams integrator | fibre | to measure (Q2) |
| a noise model | any | refused: a known limitation (§4.5) |
| a tight-tolerance run below the floor | any | refused |

**Analytic checks in single precision.** Dispersion-free SPM's phase
$P\int\gamma\,dz$; linear propagation of a chirped Gaussian; Gaussian-beam diffraction
on the radial and Cartesian routes; photon-number conservation in lossless Kerr
propagation to the binary32 growth bound.

#### 7.4 Benchmarks

Each phase records, on short windows, the time per accepted step of `mlx` against
`native` (binary64, CPU) on the same Mac, at the shapes of §2 and the benchmark suite's
corresponding cases, under the benchmark rules
([benchmarks.md](../../benchmarks.md#rules)), with the MLX version, its precision settings
and the measured peak memory. These replace §2's indicative figures in the user guide;
they inform its advice and are not merge gates.

### 8. Documentation

- A user guide page, *Single precision and Apple GPUs*, under the performance guide:
  when to use it, the capability table, what is refused, the survival record with its
  criteria, the tolerance floor, the propagator's step-count limit and the known
  limitations.
- The backends guide ([Choosing a backend](../../../guide/performance/backends.md)): the
  `mlx` entry, its installation and `MLX_ENABLE_TF32`.
- The normative pages, amended when phase 1 changes their contracts:
  [architecture.md §1](../../architecture.md) ("Double precision throughout" becomes
  double precision by default, single precision opt-in under ADR-054) and §9 (the `mlx`
  backend); [conventions.md §6](../../conventions.md) (the kernel state is complex128, or
  complex64 under single precision, held at the run's scale); and
  [kernel-contract.md](../../kernel-contract.md) (the precision field, and that table
  dtypes are binary64 at the seam and narrowed by backends).
- [backends.md](../../backends.md): a section on `mlx`, a column in §9 *What runs where*,
  and §11's limits; [testing.md](../../testing.md) §3: the binary32 bounds and their limits
  of use; [decisions.md](../../decisions.md): ADR-054.
- Docstrings for `precision` on every interface that takes it.
- A worked example comparing single and double precision on one capillary and one
  radial case, run by hand with `poe docs-examples`.

### 9. Compatibility and impact

- **Defaults unchanged.** `precision="double"` is the default; double-precision results
  do not change. The stable population forms, block exponents and binary64 propagators
  belong to the single-precision path; the binary64 path keeps its arithmetic.
- **Schemas.** `SCHEMA_VERSION` changes in phase 1 for `KernelProblem.precision`, and
  the output schema for the configuration's precision field; under ADR-051 every
  configuration's hash changes with it, and files of the earlier output schema are not
  resumed.
- **Outputs.** No change to the saved arrays. Storing fields in binary32 would halve
  file sizes and is deferred (Q6).
- **ADR-054 — Single precision as an opt-in arithmetic, validated against its oracle.**
  Written at acceptance. It amends [architecture.md §1](../../architecture.md)'s "Double
  precision throughout": binary64 stays the default and the only arithmetic of
  `numba_cpu` and `native`; binary32 is an explicit per-run choice under SEP 33's
  contract. It records the exception to ADR-020 for the `mlx` route: a GPU path whose
  results are validated against `numpy_ref`'s single-precision path, the CPU oracle,
  rather than against a compiled CPU backend in the same precision, because no compiled
  CPU backend computes in binary32 and Apple GPUs cannot compute in binary64. ADR-020's
  first clause stands: no validated binary64 result moves.
- **Luna.jl** has no single-precision mode; nothing to record on the differences page.

### 10. Alternatives considered

- **A Kokkos Metal backend.** Kokkos compiles device lambdas with a single-source
  device compiler; no such compiler targets Metal, Apple GPUs have no binary64, and
  Metal has few 64-bit atomics. The runtime part of a backend is modest (each existing
  GPU backend is 5–10 thousand lines), but the compiler is a research project and
  would have to be maintained against every Kokkos release.
- **SYCL through AdaptiveCpp.** AdaptiveCpp's generic compiler has an experimental
  Metal target (no binary64, emulated 64-bit atomics, no `printf`, per-operation command
  buffers) [5]. Kokkos's SYCL backend depends on oneAPI extensions and has no
  AdaptiveCpp configuration, so this means porting that backend or rewriting the
  native kernels in SYCL, and still needs a Metal FFT. Worth revisiting if Kokkos gains
  an AdaptiveCpp configuration upstream.
- **JAX on Metal, PyTorch's MPS device.** JAX's Metal plugin is experimental; ADR-002's
  measurements stand against JAX as a CPU backend. PyTorch's MPS device covers complex
  arithmetic and FFTs only partially. MLX is Apple's own, supports complex64 transforms
  and custom Metal kernels, and is used here only for Apple GPUs, where no other
  backend runs, so ADR-002's CPU comparison does not apply; ADR-054 records how the
  route relates to ADR-020.
- **A mixed mode first** (binary32 source, binary64 state) on `native`. Better accuracy
  for the same transform cost on CPUs and reduced-binary64 NVIDIA GPUs, but impossible
  on Apple GPUs, which have no binary64 for the state; it remains open on the roadmap
  and reuses this SEP's setting (§6.4).
- **MLX without a `numpy_ref` path**, comparing against double-precision `numpy_ref`
  with tolerances sized for binary32. Less work, but every comparison would mix
  rounding with method and leave no oracle for a later native mode; not chosen.
- **A binary32 phase increment, measured first** (the draft's Q4). Simpler, but its
  error is coherent and grows as $M\,u\,|\Delta\Phi|$, invisible to the controller
  (§4.1): a binary64-accurate propagator costs one small element-wise kernel and removes
  the $|\Delta\Phi|$ factor; not chosen.
- **Double-float emulation of binary64 for the whole state.** Correctly rounded
  emulation costs of the order of ten binary32 operations per binary64 one; for the
  whole state it would erase the gain. Used only for the propagator's argument.
- **Normalised transforms or a different unit system** instead of block exponents. Every
  pass and convention would change, and the binary64 path with them; block exponents
  and one state scale keep SI units at the seam and touch only the narrowed path.

### 11. Phases

All phases are made on one branch and reach `main` in one pull request, opened when the
SEP is set Final; continuous integration runs its full gate. Each phase is qualified on
its own, with its local GPU lane where `mlx` is touched, and reports in Part II.

**Phase 1 — The precision setting, the oracle path and the survival study.**
- The setting, the `Precision` record, `KernelProblem.precision`, the admission function
  with its defaults and refusals, validation; the configuration field, the resume
  refusal and the output schema; the memory estimate; `SCHEMA_VERSION`.
- The shared range helpers (§4.3).
- `numpy_ref`'s single-precision path on the capabilities of §5.1: block exponents and
  combined coefficients, the state scale, binary64 propagators, the stable plasma forms,
  the envelope-plasma moments in binary64, the pairwise reductions and compensated scans,
  the dtype guards, and the study switch.
- `tests/bounds.py` parameterised by $u$, with domains and limits of use; the class R,
  class P and targeted tests of §7.2 on `numpy_ref`.
- The survival criteria recorded in Part II, then the survival study on reduced cases,
  the Adams measurement among them; its manual page with the known limitations; the
  floor, Adams and propagator outcomes recorded in Part II.
- The amended normative pages (§8).
- Verifies: no double-precision result changes (the existing suite), the tests of §7.2
  and the study of §7.3, the refusals.

**Phase 2 — `mlx` on capillary fibre runs.**
- The backend package, registry entry, extra and dependency record; the arithmetic
  policy and its probes; the propagator kernel; the CPU-device test lane in continuous
  integration and the local GPU lane.
- The fibre capabilities of §5.1 for phase 2, including field-grid plasma with the
  stable forms and compensated scans; Runge–Kutta and the low-storage march;
  statistics; `observe`; snapshots.
- The batch runner (§6.3), with its gain over solo runs measured.
- Memory: bounded lifetimes, the cache limit and the measured peak against the estimate.
- Verifies: class R against `numpy_ref` single on every response; batch members; the
  scaling, vanished-response, population, propagator, failure and resume tests;
  benchmarks against `native` at $2^{14}$, $2^{16}$ and the multimode shape.

**Phase 3 — `mlx` on radial and Cartesian free space.**
- The radial route (Hankel GEMMs, Kerr, field plasma and Raman on nodes) and the
  Cartesian route (transverse transforms), with their collars and statistics.
- Verifies: class R against `numpy_ref` single; Gaussian-beam diffraction; the
  Cartesian energy case that would overflow unscaled; the GEMM probe on the radial
  shapes; benchmarks against `native` on the radial and Cartesian shapes and the
  suite's filament and Kerr-beam cases.

**Phase 4 — Envelope plasma on the device, admissions, measured kernels.**
- Envelope-grid plasma on `mlx` through the device formulation of §6.3, qualified as a
  discretisation.
- The Adams integrator under single precision, if the study admits it (Q2), with the
  default integrator then restored to `Multistep` under single precision and the
  snapshot layout revisited.
- Custom kernels for the passes the phase 2–3 records show to be bound by launch count
  or memory traffic.
- The worked example and the final records; the SEP becomes Final when every backend
  that admits single precision agrees with the oracle within its bounds.

### 12. Open questions

The maintainer decided Q1–Q11 on 2026-10-23, and revised Q4 and decided Q12–Q14 on the
review the same day (Part III). What remains open is what the phase 1 study and the
phase 2–4 measurements settle within those decisions.

- **Q1 — The tolerance floor. Decided:** $\mathrm{rtol} \ge 10^{-6}$; an explicit value
  below it is refused, and every interface's single-precision default is the floor.
  *Open:* whether the rule of §7.3 raises it to $10^{-5}$ (phase 1).
- **Q2 — Adams under single precision. Decided:** measured by the study through its
  private admission; until admitted the single-precision default integrator is
  `RungeKutta()` and an explicit `Multistep` is refused. *Open:* admission by the rule
  of §7.3 (phase 4).
- **Q3 — Noise. Decided:** refused under single precision by every backend, a known
  limitation recorded on the manual page; admitting noise is outside this SEP.
- **Q4 — The phase increment. Decided, revised after the review:** the propagator is
  formed, range-reduced and exponentiated to binary64 accuracy and then rounded, on
  every backend that admits single precision; on `mlx` by a double-float kernel from
  phase 2. *Open:* the drift constant $c$ and the step-count limit (phase 1).
- **Q5 — Reductions. Decided:** the oracle defines the orders (§6.2); every `mlx` choice
  is a class R test against it. *Open:* the per-pass forms on `mlx` (phases 2–3).
- **Q6 — Binary32 outputs. Decided:** deferred; outputs stay binary64.
- **Q7 — Continuous integration. Decided:** CI runs the `mlx` tests on MLX's CPU device;
  a local GPU lane is part of each phase's definition of done. *Open:* whether a hosted
  macOS runner exposes a usable Metal device, checked once in phase 2.
- **Q8 — `auto` under single precision. Decided:** `mlx` when it and a Metal device are
  present, otherwise a refusal naming the extra and `numpy_ref`; never `numpy_ref` by
  resolution.
- **Q9 — The MLX pin. Decided:** `mlx>=0.32.4,<0.33`, the macOS floor from MLX's wheels,
  raised only in a pull request that runs the GPU lane and the arithmetic probes.
- **Q10 — Batches. Decided:** member-owned batches through `select_batch` and the `mlx`
  batch runner in phase 2, kept for parity whatever their measured gain; no fused batch
  axis.
- **Q11 — Range. Decided, extended after the review:** block exponents for every
  narrowed buffer, the launch's power-of-two state scale, coefficients combined on the
  host, per-term homogeneity factors and scaled reductions (§4.3), shared through one
  set of helpers.
- **Q12 — Survival criteria. Decided:** the references, envelopes, verdicts and
  admission rules of §7.3, recorded before any comparison.
- **Q13 — Envelope plasma on `mlx`. Decided:** refused in phases 2–3; a device
  formulation qualified as a discretisation in phase 4.
- **Q14 — ADR-054. Decided:** written at acceptance (§9).

### 13. References

1. IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (2019).
   doi:10.1109/IEEESTD.2019.8766229
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   doi:10.1137/1.9780898718027
3. J. C. Schatzman, "Accuracy of the discrete Fourier transform and the fast Fourier
   transform," *SIAM J. Sci. Comput.* **17**, 1150–1166 (1996). doi:10.1137/S1064827593247023
4. A. Hannun, J. Digani, A. Katharopoulos and R. Collobert, *MLX: Efficient and
   flexible machine learning on Apple silicon*, software (2023); documentation,
   including "Environment Variables", "Numerical Precision", `mlx.core.compile` and
   `mlx.core.fast.metal_kernel`, at https://ml-explore.github.io/mlx/ (accessed
   2026-10-23).
5. AdaptiveCpp documentation, "Installation instructions for Metal (macOS),"
   https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ (accessed 2026-10-23).

## Part II — Implementation record

### Phase 1 — survival criteria (recorded 2026-10-24, before the study's first comparison)

The criteria are §7.3's, as the maintainer decided them on 2026-10-23 (Q12); this
section fixes the cases and the way each metric is computed before any study comparison
runs. Development checks of the oracle against binary64 at equal tolerance were made
while the path was built; none of them is a study comparison, and none set a criterion.

**Cases.** Every case runs on `numpy_ref` with `RungeKutta()` unless stated, its
binary64 reference at `rtol = 1e-10` on the same grid, discretisation, plasma rule and
quadrature, checked against a binary64 run at `rtol = 1e-9`; candidates in single
precision at `rtol = 1e-6` (the floor) and `1e-5`, and a binary64 run at `1e-6` for
context. Windows are reduced; the full-length comparison is the worked example.

| id | route | definition | expected |
|---|---|---|---|
| `spm` | fibre, envelope | GNLSE, no dispersion, γ = 0.11 W⁻¹m⁻¹, 1 kW sech of 50 fs FWHM at 835 nm, length for a nonlinear phase of 10π (0.2856 m), 400 nm–3 µm, 1 ps | survives |
| `soliton` | fibre, envelope | GNLSE N = 1 soliton, β₂ = −1.183×10⁻²⁶ s²/m, γ = 0.11, T₀ = 28.4 fs, ten soliton periods, no Raman or shock | survives |
| `rdw` | fibre, field | SEP 30's `rdw` in two modes: 125 µm He capillary, 0.8 → 0 bar gradient, 1.5 m, 7.5 fs 275 µJ at 800 nm, 90 nm–4 µm, 400 fs | survives |
| `ctc` | fibre, field | the reduced comb-to-continuum case as SEP 21 corrected it: 16 µm H₂ capillary at 40 bar, one mode, 0.1 m, 400 fs 10 µJ at 515 nm, Raman, 294 K, 250 nm–2 µm, 2 ps, noise-free | survives with a limit |
| `argon` | fibre, field | SEP 30's `argon_blueshift`: 13 µm Ar capillary at 1.7 bar, one mode, 0.4 m, 30 fs 3 µJ at 800 nm, no loss, 300 nm–4 µm, 400 fs | to measure |
| `argon_envelope` | fibre, envelope | `argon` on an envelope grid with envelope-grid plasma | to measure |
| `nitrogen` | fibre, field | 50 µm N₂ capillary at 2 bar, one mode, 0.3 m, 30 fs 50 µJ at 800 nm, molecular Raman, 200 nm–3 µm, 1 ps | to measure |
| `radial_kerr` | radial | 64 nodes in 2 mm, vacuum with χ⁽³⁾ = 10⁻²⁵ m²/V², 300 µJ 50 fs Gaussian beam of 300 µm at 800 nm, 0.3 m, envelope 600 nm–1.2 µm, 300 fs | to measure |
| `radial_filament` | radial | SEP 30's `radial_filament_256`: 1 bar Ar, Kerr and optmat plasma, 3 critical powers, 100 µm waist, 30 fs at 800 nm, 3 cm, 256 nodes in 1 mm, field grid 350 nm–2 µm, 250 fs | to measure |
| `cartesian_diffraction` | Cartesian | 64 × 64 in 2 mm × 2 mm, vacuum, 300 µm waist, 50 fs at 800 nm, 0.3 m, envelope 600 nm–1.2 µm, 300 fs | survives |
| `cartesian_kerr` | Cartesian | `cartesian_diffraction` with χ⁽³⁾ = 10⁻²⁵ m²/V² and 300 µJ | survives |
| `rdw` and `argon` under Adams | fibre | the two cases with `Multistep()` through the study's private admission (Q2) | to measure |
| noise model | any | refused under single precision (§4.5) | refused |
| below the floor | any | refused under single precision (Q1) | refused |

**Metrics**, each against the reference at the last plane:

| metric | computed as | envelope |
|---|---|---|
| field | `‖a − a_ref‖ / ‖a_ref‖` over the whole saved state, binary64 | 10⁻³ |
| spectral bands | the active band cut into 32 equal frequency bands; a band's energy is `Σ |a|²` over its bins and every mode, polarisation and transverse column; bands above −40 dB of the largest compared, relative | 1 % |
| total energy | the run's own `energy` statistic | 10⁻⁴ relative |
| peak power or intensity | `peak_power` (fibre) or `peak_intensity` (spatial) | 10⁻³ relative |
| ionised fraction | `electron_density` over the gas density at the last plane, where the reference's exceeds 10⁻⁶ | 10⁻³ relative |
| beam width | `fwhm_r` (radial) or the flux-weighted second-moment width (Cartesian) | 10⁻³ relative |

**Verdicts and rules** as §7.3: *survives* within every envelope at `rtol = 1e-6` and
`1e-5`; *survives with a limit* within them only over a stated shorter length, lower
energy or looser tolerance; *fails* otherwise. The floor rises to 10⁻⁵ if on every case
no metric's error at 10⁻⁶ is smaller than at 10⁻⁵ by more than 25 %. Adams is admitted
in phase 4 only if on both its cases its error at each tolerance is at most twice
Runge–Kutta's in every metric and within every envelope. The propagator record states
the drift constant `c` of the constant-operator tests (§7.2) and the step count at which
`c M u₃₂` reaches the field envelope.

**Where the record lives.** The harness and raw results are kept with the project's
development records (SEP 30's rule); the verdicts, the constants and the limits are
recorded here and on the manual page.

### Phase 1 — report (2026-10-24)

**Delivered.**

- `solprop.translate.precision`: the `Precision` record (`DOUBLE`, `SINGLE`), the shared
  admission `admit_precision` with its private study route `study_admission`, the floor
  `SINGLE_RTOL_FLOOR`, and the range and reduction helpers (`block_exponent`, `narrow`,
  `launch_peak`, `state_scale`, `Coefficient`/`combine`, `pairwise_sum`, `prefix_sum`,
  `scaled_square_sum`).
- `KernelProblem.precision` (kernel schema 36) with the route refusals in `validate`;
  `SimulationSettings.precision`, the integrator resolved by precision, the `auto`
  refusal; `precision=` and the floor default on `prop_gnlse`, `prop_capillary` and
  `prop_waveguide`; the precision in every hashed configuration (output schema 25).
  `numba_cpu` and `native` (schema 36) refuse single precision by name.
- `numpy_ref`'s single-precision path, `backends/numpy_ref/single.py`
  (`NumpySingleBackend`), reached through hooks of the reference that leave its binary64
  arithmetic unchanged (`SourceTables`, `_propagator`, `_rounded`, `_error_norm`,
  `_envelope_sources`, `StepDriver._norm_controller`); dtype-following buffers in
  `passes.to_time`, the radial plasma spectrum and the host Adams store; the memory rows
  (`single_precision_rows`).
- `tests/bounds.py` parameterised by the roundoff (`UNIT_ROUNDOFF_32`, `EPS_32`);
  `tests/translate/test_precision.py` and `tests/backends/test_single_precision.py`
  (class R on every route and response family, class P scaling and reload, the review's
  counterexamples, refusals, guards, failure on every march, memory).
- ADR-054; architecture.md §1, conventions.md §6, kernel-contract.md, backends.md (§2,
  §9, §11), testing.md §3; the manual page *Single precision*; the changelog.

**How the contract is implemented on the oracle** (implementation choices within §4.2,
recorded here as the target `mlx` is compared against):

- *Block exponents.* Every narrowed table has its exponent computed and recorded
  (`block_exponents`). Where narrowing the table directly keeps every entry the
  block-exponent form keeps — its largest magnitude at least one, or no entry below
  binary32's smallest normal number — the exponent is applied at upload, exactly;
  otherwise the table is refused unless its pass carries the exponent, as the Raman
  kernel's does. The same rule governs every host-evaluated array rounded per stage.
- *Host evaluation.* Every z-dependent array a pass multiplies by — the propagator, the
  prefactor, a z-table value, the spatial collar's rate, the Adams flow and its
  moments — is evaluated in binary64 and rounded once, as is the ionisation rate's table
  lookup (whose binary32 logarithm would add `|ln E| u` to its argument), and the
  envelope-grid plasma's sources (moments, populations, currents). The oracle's
  statistics are binary64 functions of the exactly widened binary32 state.
- *Homogeneity.* Kerr and Raman take `s²` in their combined coefficient; the plasma is
  evaluated at the physical field `s E` (exact) and its current divided by `s` (exact),
  which is §4.3's table of factors.
- *Orders.* Running integrals use the doubling scan `prefix_sum` (each prefix of depth at
  most `ceil(log2 n)`); norms are scaled sums of squares reduced by `pairwise_sum`.

**Survival study** (cases and criteria above; references all passed their `1e-9` check;
the harness's metrics are relative errors at the last plane):

| case | single `1e-6`: field | largest other | single `1e-5`: field | verdict |
|---|---|---|---|---|
| `spm` | 7.9e-6 | 9.6e-5 bands | 9.3e-5 | survives |
| `soliton` | 2.7e-5 | 6.5e-6 peak | 2.4e-5 | survives |
| `rdw` | 2.6e-6 | 4.9e-5 bands | 1.7e-5 | survives |
| `ctc` | 2.6e-3 | 3.4e-4 peak | 1.3e-3 | survives with a limit: 0.05 m (field 4.3e-5 there; 0.025 m 6.6e-6) |
| `argon` | 1.8e-4 | 2.1e-4 bands | 5.2e-4 | survives |
| `argon_envelope`, over 0.1 m | 2.7e-6 | 4.2e-5 bands | 1.9e-5 | survives |
| `argon_envelope`, 0.4 m (completed after this report, recorded in phase 4) | 1.3e-5 | 2.1e-5 ionised | 3.5e-5 | survives |
| `nitrogen` | 1.1e-6 | 2.7e-5 bands | 3.1e-6 | survives |
| `radial_kerr` | 8.5e-6 | 1.3e-4 bands | 1.3e-4 | survives |
| `radial_filament` | 2.3e-6 | 1.2e-5 peak | 1.3e-5 | survives |
| `cartesian_diffraction` | 2.4e-7 | 2.4e-7 bands | 2.4e-7 | survives |
| `cartesian_kerr` | 2.7e-6 | 2.0e-5 bands | 7.4e-5 | survives |

- *The CtC limit is the physics.* A binary64 run whose launch carries an additive seed of
  `2^-24` of its peak spectral amplitude in every active bin departs from the reference
  by 5.1e-2 at 0.1 m and 2.3e-3 at 0.05 m (soliton 5.8e-7, rdw 9.9e-7, argon 4.6e-6):
  noise-free stimulated
  Raman scattering amplifies any rounding-sized Stokes seed, and binary32's rounding is
  one. A multiplicative perturbation of the same size, which seeds no empty bin, moves
  the result by 3.4e-7 only.
- *The floor* stays at `1e-6` (Q1): the floor rule does not hold (`spm`'s field error
  falls by 92 % from `1e-5` to `1e-6`).
- *Adams* (Q2) is **not admitted** by the rule: on `argon` at `1e-5` its field error is
  1.5e-3 (outside the envelope, three times Runge–Kutta's), and on `rdw` at `1e-6` its
  energy error is 1.6e-7 against Runge–Kutta's 6.9e-8. The binary64 Adams runs at the
  same tolerances have the same errors (argon at `1e-5`: field 1.5e-3; rdw at `1e-6`:
  field 2.7e-6 against 2.6e-6 in binary32): the failure is Adams's truncation error at
  those tolerances, not single precision. The rule as fixed compares with Runge–Kutta in
  binary32, so the default stays `RungeKutta()` and `Multistep` stays refused; whether a
  rule comparing Adams in binary32 with Adams in binary64 should decide phase 4 is the
  maintainer's (open).
- *The propagator drift* (`harness/drift.py`; constant operator, no source, uniform
  steps): on a He capillary's field-grid operator the error is 4.5e-7, 2.9e-6, 3.1e-5
  and 3.1e-4 after 10, 100, 1000 and 10 000 steps, `c` = 0.49–0.53 for `M >= 100`
  (0.76 at 10, where the launch's rounding dominates); on the GNLSE operator `c` falls
  from 0.32 to 0.085, rounding of the state averaging out. The field envelope 10⁻³ is
  reached at `M ≈ 3.2e4` steps at `c = 0.53`; the manual page states it. The
  binary32-exponent variant gives the same errors to 2 % for `M >= 100` (2.5 against
  0.76 at 10): in these frames the occupied bins' `|dPhi|` per step is small, so the
  review's witness (`|dPhi| ~ 10^3`) does not arise on these routes; the
  binary64-accurate propagator costs nothing on the oracle and is kept.

**Findings for later phases.**

- *The inverse propagator's range.* In binary32 `exp(-dPhi)` overflows when a step's
  damping exceeds about 88 nepers (709 in binary64). No study case reached it; an
  adaptive run would reject such a step as non-finite and shrink it, a replayed one stops
  with `ConvergenceError`. `mlx` (phase 2) records whether a strong collar on a long step
  reaches it.
- *Envelope plasma on the oracle is slow*: the per-sample decomposition and adaptive
  quadrature run in Python: one binary32 run of the envelope case takes 52–80 min over
  0.1 m on an idle workstation core and hours over 0.4 m. Its binary64 references (`1e-10`,
  `1e-9`, `1e-6`) were therefore computed on `native`, whose envelope plasma is the same
  discretisation and quadrature, qualified against `numpy_ref` to rounding — orders of
  magnitude inside these envelopes; the reference check passed (6.4e-9). The candidates
  ran on `numpy_ref`, the only single-precision path. The case ran over 0.1 m for its
  verdict; the 0.4 m candidates are recorded when they finish. Phase 4's device
  formulation is the remedy.
- *Adams snapshots under single precision* (`HostHistory.load_rows`) still restore a
  binary64 store; phase 4 revisits the layout, as §5.3 says.

**Deviations.** None from Part I. The study ran on reduced windows as §7.3 says; the two
CtC lengths and the perturbation witness were added to state the CtC limit, and are not
new criteria. The envelope case's binary64 references ran on `native` and its verdict
on 0.1 m, for run time (above). Two-component (tensor) Raman is implemented but has no test
or study case yet; phase 2 adds its class R test.

**Verification.** The new test modules pass (39 tests); the reference's existing tests
pass unchanged (746 in the `numpy_ref`, memory, batch, held-plasma and registry modules;
4302 in `tests/translate`, `tests/output`, `tests/processing` and `tests/driver` after the
expected record pins: schema-21 projections leave out the precision as they leave out
SEP 34's measure); native's backend tests pass with schema 36. CI runs the full gate on
the SEP's pull request.

### Phase 2 — report (2026-10-25)

**Delivered.**

- `solprop.backends.mlx_gpu` (`arithmetic.py`, `passes.py`, `backend.py`, `batch.py`),
  registered as `"mlx"`; the `mlx` extra (`mlx>=0.32.4,<0.33`, macOS arm64) written by
  `hatch_build.py`; `registry.available(precision=)` lists binary64 backends by
  default and `mlx` under `"single"`; `backend="auto"` under single precision resolves
  to `mlx` where MLX and Metal are present (Q8); `registry.metal_available`.
- `MlxBackend` on the fibre route: mode-averaged and multimode, field and envelope
  grids, Kerr and the harmonic, scalar and tensor Raman, mixtures, the field-grid plasma
  with the held current, loss, both absorber schemes, the polarisation taper,
  device-ready gradients; Runge–Kutta, the low-storage march (`MlxLowStorage`), lockstep
  batches (`MlxBatchRunner`), statistics and `observe` through the oracle's host side.
  Refused by name: radial and Cartesian free space (phase 3), the envelope-grid plasma
  (phase 4), a z table, Adams, interpolated planes.
- The protected operations as custom kernels in `"safe"` mode (propagator and term
  combination in double-float, populations, the doubling scan, the rate table) and the
  GEMM probe; `memory.mlx_rows`, `MLX_GRAPH_BUDGET` and the device ledger rows.
- `tests/backends/test_mlx.py` (29 tests, the kernels', the FFT characterisation's and
  the special values' among them); the macOS CI job installs the extra and
  prints whether the runner exposes Metal; backends guide, dependencies, the manual
  page's backend column, backends.md §4a, the changelog.

**Findings about MLX 0.32.4 on an Apple-silicon GPU** (each with its test):

- `expm1` errs by up to 1.0e-4 relative at small arguments (1,700 ulp); `exp`, `log`,
  `log1p`, `sin` and `cos` are within 3 ulp. The populations use Kahan's `expm1` in a
  kernel (4 u).
- The FFT is about three times less accurate than NumPy's float32 transform (forward
  3.4e-7 against 1.1e-7 at 2^12) and its round trip shrinks by 1.2–2.4e-7, a coherent
  bias. Only the split-step collar puts the state through a round trip; it is applied as
  a correction, which removed a 5.6e-6 drift over 16 steps. Characterised against
  `fft_bound` at binary32 for real and complex transforms of 1024 to 2^18 samples and
  the smooth lengths 3072 and 17920: the worst forward or round-trip error is 8.5e-7,
  over a thousand times inside the bound; `fft_probe` repeats the measurement at each
  run's own lengths when the backend is built.
- Matrix products default to TF32 (7.3e-4 relative), as §6.3 says; the probe separates
  them.
- Float32 subnormals are flushed on the GPU (accepted, §4.3 rule 7).
- The propagator kernel errs by at most 5 u (median 0.5 u), not the one rounding of the
  oracle: binary32 `exp`, `cos` and `sin` of reduced arguments. It does not grow with
  the phase. The device's drift constant is `c <= 0.85` (oracle 0.17–0.32);
  double-float `sincos` is the refinement if a record needs it.

**Agreement with the oracle** (class R, same schedule; bounds as the oracle tests'):
soliton with Raman 2.0e-7 (10 steps) and 4.1e-7 with the kernel; envelope capillary
8.7e-7; argon capillary 2.8e-6 and nitrogen 3.0e-6 (16 steps), whose free current
integrates an oscillating field to a small net value, a one-ulp difference in the
population moving it by 7e-4 — the conditioning §7.1's signed-sum bounds carry; four
modes 1.0e-5 over 256 steps; tensor Raman on two components; low-storage 2.0e-7; batch
members equal to their solo runs bitwise; a resume across the oracle and `mlx` within
its bound.

**Indicative timings** (one Apple-silicon laptop under heavy load, same-process A/B
against `native` in binary64, four alternating rounds, fixed 20-step schedules, median
time per step; not a benchmark, which §7.4 assigns to a quiet machine):

| case | `native` / `mlx` time |
|---|---|
| GNLSE, one line, 2^14 samples | 1.5 |
| GNLSE, one line, 2^16 | 2.7 |
| GNLSE, one line, 2^18 | 5.4 |
| Ar capillary with plasma, one mode, 300 fs (n_to 4096, 33 nodes) | 0.34 |
| the same, four modes | 0.51 |
| the same, eight modes on 1 ps (n_to 16384, 65 nodes) | 1.21 |

Small runs are bound by the GPU's per-operation latency (a 33 × 4096 stage costs about
0.8 ms in some sixty launches); the kernels for the scan and the rate table halved the
plasma's share. Fusing more passes is phase 4's (§11). A first cut evaluated every
propagator on the host (a bug in `DoubleFloatTerms.upload`, found by these timings and
fixed); the A/B ratios above are after the fix.

**Not yet done in this phase** (carried into the PR's remaining work, before Final):
the hosted macOS runner's Metal result (Q7), which the first CI run on the pull request
records; the benchmark on a quiet machine (§7.4).

**Deviations.** None from Part I.

### Phase 3 — report (2026-10-25)

**Delivered.**

- `MlxBackend` on the radial route: the Hankel synthesis and projection as the fibre
  route's float32 products (`gemm_probe` on the radial shapes), Kerr, Raman and the
  field-grid plasma on the nodes, the radial plasma spectrum formed on the nodes and
  projected once (`_device_radial_plasma`, the oracle's `_radial_plasma_spectrum`), and
  the radial absorber matrix in the stage.
- On the Cartesian route and its line (`_device_free`): the separable operator uploaded
  once, at unit distance, as one double-float term the stage's `dz` weights (the oracle
  forms `separable_values(...) * dz` in binary64 and rounds it, the same quantity), the
  per-column prefactor narrowed once, MLX's `fftn`/`ifftn` over the transverse axes,
  the responses per column chunk on the fine grid, and the window's in-stage rate with
  the temporal collar. `fft_probe` adds the transverse lengths; `gemm_probe` is skipped,
  the route having no quadrature products.
- `memory._mlx_per_stage` with `_MLX_TRANSIENT_COLUMNS` (18 fine-grid columns per
  column on the Cartesian route, 6 on the radial one): MLX's transverse transforms copy
  their axes and the reference's in-place passes are separate arrays on the device.
  Measured peak against the estimate: 0.72–0.84 on the Kerr beams with both collars,
  0.18–0.78 across 16 × 16 to 32 × 32 Cartesian grids with 128–512 samples, 0.30–0.51
  on the filaments, each graph policy (whole step or stage by stage) measured.
- Tests in `tests/backends/test_mlx.py` (42 on the Metal device; 36 and 6 kernel
  skips on MLX's CPU device): class R against `numpy_ref` single for Kerr beams on the
  radial, Cartesian and line routes, each with and without its collars; argon
  filaments on the radial and Cartesian routes and a nitrogen one on the radial route
  with its Raman response; Gaussian-beam diffraction; the Cartesian launch whose sum of
  squares exceeds binary32's range; the memory estimate on both routes. The case
  builders are shared with the oracle's tests in `test_single_precision.py`.

**Agreement with the oracle** (ten steps, same schedule, normwise relative): Kerr
beams 5.7e-7 (radial), 6.4e-7 (Cartesian) and 7.4e-7 (line); with the collars 5.8e-7,
6.4e-7 and 7.6e-7, against the collars' own effect of 9e-5 and 1e-4 on the radial and
Cartesian runs; the argon filament 5.7e-7. A full run with its output agrees on the saved field and every statistic within
2e-6 (the species' peak rate and ionisation energy the largest).

**Gaussian-beam diffraction** on a 128 × 128 Cartesian grid over one Rayleigh range of
a 100 µm waist: width, axial amplitude and Gouy phase within 1.6e-6, 1e-8 and 1.3e-8
of paraxial optics, against the exact operator's non-paraxial bound of 6.5e-6; the
oracle's errors are the same to two digits.

**Indicative timings** (the same loaded laptop, same-process A/B against `native` in
binary64, the best of two rounds, ten fixed steps after a warm step; not a benchmark):

| case | `native` / `mlx` time |
|---|---|
| Kerr beam, 64 radial nodes, 512 samples | 1.15 |
| Kerr beam, 256 radial nodes, 1024 samples | 3.8 |
| Kerr beam, 64 × 64 Cartesian, 256 samples | 2.2 |
| Kerr beam, 128 × 128 Cartesian, 512 samples | 2.0 |
| argon filament, 64 radial nodes, field grid (513 bins) | 1.5 |
| argon filament, 128 radial nodes | 2.3 |

The Cartesian route gains less than the radial one at the same state size: its two
transverse transform pairs per stage (four with the collar) are MLX's `fftn`, whose
axis copies are the transients the memory rows count. A fused transverse kernel is a
candidate for phase 4's measured kernels (§11).

**Not yet done in this phase**: the benchmark suite's filament and Kerr-beam cases on a
quiet machine (§7.4), with phase 2's; Q7 at the first CI run.

**Deviations.** None from Part I.

### Phase 4 — report (2026-10-25)

**Delivered.**

- The envelope-grid plasma on `mlx` (`mlx_gpu/envelope.py`), the device formulation of
  §6.3: one kernel thread per sample computes the polarisation ellipse in closed form
  (native's `envelope_axes`), the cycle moments on the accepted 64–512-point rules split
  at the table's cut angles with each level's sums compensated, refined to the
  representation's `2e-6`, the rate from the table's spline (the phase 2 rate kernel's
  arithmetic), the loss current per unit work, and a failure flag. The trial's failures
  are read with its norm, in the same synchronisation, and raise the oracle's
  `ConvergenceError`; the low-storage march reads them after its step. The exposure is
  the trapezoid's doubling scan, the populations the stable kernel, and the drive and
  loss are projected with the polarisation. The species statistics' node scan uses the
  same kernel through a new hook on the reference backend (`_envelope_populations`).
- The phasor of the propagator kernel is renormalised (below).
- Translation refuses, under single precision, a replayed step or an adaptive floor
  whose propagator exponent `|Re dPhi|` passes `log(FLT_MAX) = 88.72`, on every kind,
  and warns within a tenth of it (`KernelProblem._check_single_exponent`).
- The modal routes project their streams (polarisation, gated and edge currents,
  envelope drive and loss) in one matrix product per chunk (`passes.project_many`).
- On MLX's CPU device the populations are formed in binary64 and rounded (MLX's CPU
  `exp` errs by 3.4e-6 near `exp(-80)`), and the envelope plasma is the oracle's host
  evaluation.
- The memory estimate counts the device's envelope arrays; the measured peak is 0.63–0.69
  of it on three envelope cases.
- Tests (`tests/backends/test_mlx.py`, 47 on the Metal device; 38 and 9 skips on MLX's
  CPU device): the cycle kernel against the oracle's SVD and adaptive moments on 4,000
  linear, circular, elliptical, near-linear and zero fields, one and two components; a
  refinement failure stopping the march; the species populations against the oracle's;
  the envelope-plasma capillary against the oracle (class T); the memory estimate. In
  `test_single_precision.py`, the binary32 exponent refusal.
- The worked example *Single precision on an Apple GPU*
  (`docs/examples/performance/plot_01_single_precision.py`), run by `poe docs-examples`.

**Findings.**

- *A cut angle rounded across a cutoff.* The first device kernel evaluated the rate at
  every node of every panel, as native does in binary64. In binary32 the computed cut
  angle (an arccosine near one) and the rate's own cutoff test disagree for a field
  within a few parts in 10⁶ of the table's lower edge: nodes inside the panel saw a
  field rounded below the cutoff and a zero rate, successive levels jittered, and the
  refinement failed by order 512 at a rate of 2e-33 s⁻¹ in the survival study's
  envelope case, where the oracle converges. The kernel now honours the split itself —
  no contribution past the low cut, the ceiling rate before the high cut, the field held
  inside the table's range between them — which is what the oracle's binary64 split
  means. Within 0.1 % of the cutoff a moment is still conditioned by the angle; the
  tests compare against the representation's `1e-30` s⁻¹ floor, as its own refinement
  does. Loss currents per unit work below binary32's normal range (rates near the floor)
  are flushed by the GPU, as §4.3 rule 7 accepts.
- *The propagator's modulus bias.* The survival runs of the envelope case on `mlx`
  were inside their envelopes but ten times the oracle's error at `rtol = 1e-6`, and no
  better than at `1e-5`; the energy drifted by 6.2e-6 over 258 steps without the plasma.
  The phase 2 kernel's binary32 `cos` and `sin` leave `c² + s²` off one by a few u with
  a bias (an energy-weighted 1–3e-8 per propagator) that a long march accumulates. One
  Newton step on the double-float squared modulus removes it — applied as `c + c δ` by a
  fused multiply-add, because a factor `1 + δ` rounded to binary32 is itself biased, its
  spacing being u below one and 2u above (a first version applied the factor and moved
  the field-grid argon case from 2.2e-4 to 9.5e-4, deterministically, under every nudge
  of the first step). With the fix the device's drift constant is `c` 0.15–0.32 and its
  energy drift over 1,000 and 3,000 linear steps equals the oracle's to three digits
  (7.80e-6 and 2.26e-5): what remains is binary32 storage's own rounding, which phase 1
  records.
- *A binary32 range limit on the step.* An envelope-plasma test on a 1 mm capillary
  with 0.25 mm steps diverged on both single-precision backends and ran in binary64:
  the spectral collar gave `Re dPhi = -150`, and the interaction picture's inverse
  propagator `exp(150)` overflows binary32 (`exp(88.72)`) but not binary64 (`exp(709.78)`).
  Phase 1's range analysis covered the tables and the state, not the propagator's
  exponent; translation now refuses such a step (above).

**Survival (class T) on `mlx`**, the §7.3 criteria and harness of phase 1, against the
same binary64 references:

| case | `mlx` single `1e-6`: field | largest other | `mlx` single `1e-5`: field | verdict |
|---|---|---|---|---|
| `argon_envelope`, 0.4 m | 4.8e-5 | 5.3e-5 ionised | 6.9e-5 | survives |
| `argon_envelope`, 0.1 m | 8.6e-6 | 3.4e-5 bands | 1.8e-5 | survives |
| `argon` (field grid), 0.4 m | 2.2e-4 | 2.8e-4 bands | 3.4e-4 | survives |

The oracle's 0.4 m envelope runs, on the workstation, give field 1.3e-5 and ionised
fraction 2.1e-5 at `1e-6` (5.9 h; binary64 at the same tolerance: 1.3e-5 and 2.0e-5)
and 3.5e-5 and 2.8e-5 at `1e-5` (3.5 h): the case survives on the oracle too. The
device's runs take 5–7 s; its larger errors are its discretisation of the moments
(class T), inside the envelopes by a factor of twenty.

**Agreement with the oracle** (fixed schedule): the envelope-plasma capillary 4.4e-7
after eight steps, against a plasma effect of order one on the same field; the device's
envelope sources against the oracle's on a mid-run field, drive identical and loss
4.2e-7; the device rate's energy-weighted bias 1e-7 relative.

**Indicative timings** (one laptop under load, same process, median of three, ten fixed
steps after two):

| case | `native` / `mlx` time |
|---|---|
| envelope plasma capillary, one mode, 300 fs | 7.8 |
| the same, four modes | 7.6 |
| the same, eight modes on 1 ps | 4.6 |
| batched projection against per-stream, field plasma, one mode | 1.18 |
| the same, four and eight modes | 1.05, 1.01 |

**On a quiet machine.** The same-process A/B of every phase's cases, repeated on the
Apple-silicon laptop with nothing else running (six alternating rounds, the first dropped,
median per fixed step; `native` at its default thread budget; the spread of `native`
under 2 %, of `mlx` up to 27 % on the smallest, launch-bound cases):

| case | `native` ms | `mlx` ms | `native` / `mlx` |
|---|---|---|---|
| GNLSE with Raman, 2^14 | 4.34 | 2.53 | 1.7 |
| GNLSE with Raman, 2^16 | 12.0 | 3.51 | 3.4 |
| GNLSE with Raman, 2^18 | 41.0 | 7.79 | 5.3 |
| Ar capillary, field plasma, one mode, 300 fs | 2.17 | 4.26 | 0.51 |
| the same, four modes | 3.09 | 4.49 | 0.69 |
| the same, eight modes on 1 ps | 16.2 | 14.4 | 1.12 |
| Ar capillary, envelope plasma, one mode, 300 fs | 33.9 | 4.20 | 8.1 |
| the same, four modes | 34.5 | 4.28 | 8.1 |
| the same, eight modes on 1 ps | 16.0 | 4.58 | 3.5 |
| radial Kerr beam, 64 nodes | 2.56 | 2.45 | 1.05 |
| radial Kerr beam, 256 nodes | 14.9 | 5.13 | 2.9 |
| Cartesian Kerr beam, 64 × 64 | 26.2 | 13.5 | 1.9 |
| Cartesian Kerr beam, 128 × 128 | 243 | 127 | 1.9 |
| radial argon filament, 64 nodes | 7.34 | 4.15 | 1.8 |
| the same, 128 nodes | 11.6 | 4.65 | 2.5 |
| the same, 256 nodes | 28.5 | 8.84 | 3.2 |

They agree with the loaded machine's indicative ratios to within their spread. The
standard suite's harness (SEP 30) has no `mlx` cases yet; adding them is on the
roadmap.

**Measured kernels (§11).** Phases 2–3 showed the small plasma capillaries bound by
launches. The projections, a third of such a stage, are batched (above); the rest of
the stage is a few dozen elementwise passes and two scans per channel. A fused stage
kernel is the next step and is on the roadmap; the envelope plasma, the one pass whose
work justified a kernel, is one.

**Adams (Q2).** Phase 1's study did not admit Adams by §7.3's rule (on `argon` at
`1e-5` its field error is three times Runge–Kutta's and outside the envelope), and
binary64 Adams has the same errors at the same tolerances, so the excess is the method's
truncation, not the precision. Adams stays refused under single precision and
`RungeKutta()` stays the default; whether a rule comparing binary32 Adams with binary64
Adams should decide it is the maintainer's open decision, on the roadmap.

**Q7 answered.** The pull request's first CI run recorded that the hosted macOS
runner (macos-15, arm64) exposes Metal to MLX, so the `mlx` tests run on a GPU in CI.
Its GPU computes float32 products in float32 even with TF32 enabled, so the TF32
refusal test skips there.

**Open, moved to the roadmap**: `mlx` cases in the standard benchmark suite (SEP 30);
the Adams rule; a fused stage kernel; single-precision and mixed modes of `native` and
`numba_cpu`.

**Deviations.** None from Part I. The binary32 exponent refusal is new; it applies §4.3's
range rule to the propagator.

## Part III — History

- 2026-10-23: Drafted, after an indicative synthetic MLX measurement on an Apple-silicon laptop.
  The maintainer chose the routes (capillary fibre runs, radial and Cartesian free
  space; not the general modal route), batches only through the existing interface,
  MLX as an optional extra, and a `numpy_ref` single-precision oracle.
- 2026-10-23: Revised with the maintainer's decisions on Q1–Q11: the `rtol` floor and
  defaults, Runge–Kutta by default until Adams is admitted, noise refused as a known
  limitation, the oracle's reduction orders, the CI and GPU lanes, the MLX pin, and a
  new §4.3 on range: reductions of unnormalised spectra overflow binary32 on the
  Cartesian route, so the state is held at an exact power-of-two scale and sums of
  squares are scaled.
- 2026-10-23: John Travers confirmed that he has read the proposal and requested
  that his human review be recorded. GPT-6 completed the AI review of Part I below.
  Review status is `Reviewed`; proposal status remains `Draft`. The review records
  questions and recommendations, not acceptance or maintainer dispositions.
- 2026-10-23: Revised after the reviews (Opus 5.5): the maintainer's decisions on
  R3, R6, R8 and R11 and the response to every point are below; Part I rewrites
  §4.3 and §7 and adds §4.4, §5.1–5.3, the arithmetic policy of §6.3 and Q12–Q14.
- 2026-10-24: Accepted by John Travers and set In progress. §11 now follows the
  project's rule of one branch and one pull request per SEP, opened at Final, in place
  of a merge per phase. ADR-054 is written (§9). Single precision on `numba_cpu` and
  `native` is out of scope (§2), so Final needs agreement with the oracle on every
  backend that admits single precision, not parity across all three backends.
- 2026-10-24 to 2026-10-25: Phases 1–4 implemented and reported in Part II (Opus 5.5,
  with John Travers reviewing after phases 1, 2 and 3).
- 2026-10-25: Set Final with phase 4; the open questions are on the roadmap. Its
  decision record, written as ADR-052, is renumbered ADR-054: ADR-052 and ADR-053 were
  taken by the thread budget and the CI lanes, which reached `main` first.

### AI review — 2026-10-23 — GPT-6

Reviewed all of Part I against the current implementation, the architecture and
conventions, SEP 25 and SEP 0's acceptance criteria. **Recommendation: revise before
acceptance.** The separate precision setting, reference path and measured survival
study are a sound direction, but the range and arithmetic contracts need the
clarifications below before they can define a reliable implementation.

Only the review header and Part III have been changed. P1 findings concern numerical
correctness or an unproven execution contract; P2 findings concern compatibility or
implementation details that need a stated resolution. Maintainer decisions on these
points are pending; the earlier decisions on Q1–Q11 have not been reinterpreted.

1. **R1 (P1), §§4.2–4.3, 6.2–6.3 — State scaling does not protect every response
   intermediate.** The current Raman representation is per particle, and
   `translate.raman.kernel_spectrum` folds `dt` into the spectrum before
   `backends.numpy_ref.passes.raman_polarisation` multiplies by density. A probe using
   `FieldGrid.from_limits(0.01, 800e-9, (200e-9, 4e-6), 2e-12)`, nitrogen at
   293.15 K and $2.5\times10^{25}$ m⁻³, and
   `MaterialSelection(carrier_wavelength=800e-9)` gave a maximum kernel-spectrum
   magnitude of $5.21\times10^{-50}$: all 32 769 bins became zero on conversion to
   complex64. Multiplying a later prefactor cannot recover them. Normalising the
   propagated state and scaling sums of squares therefore do not establish §4.3's
   range claim. **Recommended change/question:** specify a range-safe representation
   for each response buffer, including Raman spectra and scalar coefficients, with
   coefficient combination or separate exact scales applied before narrowing. Add
   molecular Raman tests that detect a vanished response, including the device's
   subnormal behaviour; report intermediate ranges as well as the state's.

2. **R2 (P1), §§4.3, 7 — Derive scaling for the complete plasma source and each
   statistic.** The statement that the plasma current is linear in the field covers
   the free current at a given population, but not ionisation loss. For
   $E=s\tilde E$ and the population evaluated from the physical field,
   $J_{\rm loss}=I_p W(s|\tilde E|)(1-f)E/|E|^2$ gives
   $J_{\rm loss}/s=I_p W(s|\tilde E|)(1-f)\tilde E/(s^2|\tilde E|^2)$.
   The free and loss streams thus need different scale factors. Likewise, field
   observations scale by $s$, energy and power by $s^2$, and a fraction or width
   does not acquire either factor; the blanket statement that `stats` multiplies by
   $s$ is insufficient. **Recommended change/question:** give the scaled equations
   for both field and envelope plasma, held-current assembly and all statistics,
   including absolute tolerances and the zero launch. Qualify scaled versus unscaled
   responses and propagation separately. Retain class P only where the actual
   operation sequence is proved to commute with scaling; use class R when coefficient
   regrouping changes the rounding sequence.

3. **R3 (P1), §§4.1, 6.2, 7, Q4 — Linear-phase rounding can accumulate coherently
   and escape the controller.** A constant increment is rounded in the same direction
   on every step; re-anchoring avoids cumulative-phase subtraction but does not make
   this error independent. As a scalar witness, repeatedly multiplying complex64 by
   $\exp[-i\,\mathrm{float32}(1234.56789)]$ gives relative errors
   $1.89\times10^{-5}$ after one multiplication and $1.89\times10^{-4}$ after ten,
   against binary64. With no nonlinear source the embedded error is zero, so the
   `rtol` floor cannot detect this. **Recommended change/question:** describe
   $u\sqrt M$ as a heuristic under stated assumptions, and budget coherent phase
   and coefficient errors as well. Add constant-operator, occupied-band tests over
   several step sizes. The study needs to distinguish forming the increment in
   binary64 and then rounding it from evaluating its exponential in binary64 and
   then rounding the propagator: only the latter also isolates argument rounding.
   What measured error triggers Q4's device remedy or a named refusal?

4. **R4 (P1), §§4.2, 6.2, 7 — The field-plasma population is not already in the
   stated stable form.** `backends.numpy_ref.passes.plasma_scan` currently computes
   $f=1-(1-p)\exp(-X)$, whereas §4.2 says it keeps an `expm1` form. In binary32 at
   $p=0$, $X=10^{-8}$ gives zero with the current subtraction and approximately
   $10^{-8}$ with `-expm1(-X)`. At $X=20$, the rounded fraction is one while the
   neutral survival is still $2.06\times10^{-9}$; reconstructing survival as $1-f$
   also erases a nonzero loss current. The envelope path already retains survival
   independently. **Recommended change/question:** explicitly include stable
   population and survival evaluation in the single-precision field path, and test
   small exposure, nonzero pre-ionisation and saturation against independent
   density/current and ionisation-work witnesses. Keep the binary64 default's
   compatibility promise explicit rather than assuming its implementation already
   supplies this arithmetic.

5. **R5 (P1), §§4.1, 6.1, 7 — Recompute bounds at binary32 and check that they
   discriminate.** Not every helper is linear in $u$:
   `tests.bounds.higham_gamma` is $ku/(1-ku)$, and `gamma` has the analogous
   denominator in $\epsilon$. They cannot in general be obtained by multiplying
   binary64 bounds by $2^{29}$; their domain restrictions matter at binary32.
   Separately, the quoted FFT margin becomes $8.54\times10^{-4}$ at $2^{14}$
   samples, before source counts and propagation growth, so a large ladder can pass
   an error that the survival study would reject. **Recommended change/question:**
   parameterise the formulas themselves, include table/input quantisation and
   conditioned signed sums, and state where a bound becomes unusable. Define a
   separate, discriminating GEMM bound for the TF32 probe. Fix useful class T
   field, band and observable envelopes independently of these rounding margins;
   backend agreement alone must not constitute a survival verdict.

6. **R6 (P1), §§7, 11–12, Q1–Q2, Q4 — The study's decision rules are not yet
   reproducible.** “Survives”, “visible in observables”, “no accuracy gain” and
   Adams staying within a “fixed factor” have no stated thresholds. A tighter local
   tolerance alone also does not establish the reference's global accuracy or
   control its grid and plasma-integration errors. This matters because these
   outcomes can change the public tolerance floor and integrator default.
   **Recommended change/question:** before measuring candidates, specify each
   canonical case, the reference-convergence evidence and uncertainty, the asserted
   bands and minimum signal, the error metrics and application envelopes, and the
   factors used for the three admission decisions. Which accuracy losses are
   acceptable to the maintainer? Phase 1 can establish and record these criteria
   before its comparisons; they need not be invented by the implementer or fitted
   to a result. State how the study exercises Adams while public admission refuses
   it, and record a failed case as a limitation or refusal.

7. **R7 (P1), §§4.2, 6.3, 11 — `safe` custom-kernel mode is not a complete MLX
   arithmetic policy.** The documented
   [`metal_kernel` option](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)
   controls custom kernels and promises special-value behaviour; the documented
   [`mx.compile` interface](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html)
   exposes no corresponding math-mode parameter. Neither statement alone establishes
   the contract for compiled array operations, vendor transforms, compensated scans
   or small-argument functions. The
   [environment-variable documentation](https://ml-explore.github.io/mlx/build/html/usage/environment_variables.html)
   also says precision settings can be latched at subsystem initialisation, so a
   user who has already used MLX needs a defined outcome. **Recommended
   change/question:** identify the supported mechanism and evidence for each
   protected operation, separately from TF32 admission. Specify pre-initialised
   process handling and an actionable refusal if safe arithmetic cannot be
   established. Test the probe on the actual GEMM dispatch families and test
   compensation, finite-value detection and tiny responses with adversarial inputs.
   Accepting hardware flushing also needs an absolute underflow budget, beyond the
   relative $u$ model.

8. **R8 (P1), §§6.2–6.3, 11 phase 2 — Envelope plasma needs an explicit device
   strategy.** Its current reference pass is more than pointwise NumPy arithmetic:
   `backends.numpy_ref.envelope_plasma._cycle_quantities` performs an SVD at every
   temporal sample, then `translate.envelope_plasma.cycle_moments` splits angular
   intervals and refines 64–512-point quadrature with data-dependent convergence
   checks. Translating this literally does not establish the promised compiled
   response graph without intermediate host decisions. **Recommended change/question:**
   choose and qualify a device formulation before promising phase 2 admission:
   for example, a stable batched ellipse decomposition and bounded quadrature with
   device status reporting, or a separately error-controlled representation.
   Preserve scalar, circular, nearly linear and degenerate vector limits, cutoff
   crossings and refinement failure. Any different discrete quadrature needs its
   own accuracy qualification, rather than being classified solely as rounding
   against the adaptive reference. If feasibility is deferred, name the phase and
   the temporary refusal explicitly.

9. **R9 (P2), §§4.3, 5–6, 9, 11 phase 4 — Specify resume and saved-schema
   compatibility with the private scale.** A restart's peak can differ from the
   launch's, so choosing $s$ in each `set_initial` call can change it. A
   `MarchSnapshot` contains state, source/history rows and compensated rows, not
   only an observed field. The current `output.schema.RunMetadata` also preserves
   historical configuration hashes through explicit schema guards, while backend
   provenance is outside the hash. **Recommended change/question:** state whether
   the scale is retained or recomputed and how every restored row is converted;
   distinguish exact storage round trips from class T resumed accuracy. Choose
   whether precision lives in the existing hashed configuration or a versioned new
   metadata field, and specify the meaning of legacy files without it. Add
   same-precision cross-backend resume, changed-scale restart, wrong-precision
   refusal and historical-hash tests. Revisit snapshot layout when Adams is admitted.

10. **R10 (P2), §§5–6.3, 11 phases 2–3 — Budget lazy-graph memory and define the
    batch synchronisation hook.** Binary32 halves the bytes per resident element;
    it does not prove that peak resident memory is half the reference's. A graph
    holding all stages and node chunks until evaluation can retain intermediates
    that an eager pass would release, alongside MLX's allocator cache and compiled
    workspaces. The existing `LockstepRunner._trial_step` calls each member's trial
    and requires its host norm immediately, so using that loop with a synchronising
    trial cannot queue all members before one evaluation. **Recommended
    change/question:** specify bounded graph/chunk lifetimes, cache/workspace
    accounting and a measured peak-memory check against the admission estimate.
    Describe the MLX batch override that builds all member trials before evaluating
    their norms, retaining the existing public interface and independent ledgers.
    Require finite-value/failure detection for fixed and low-storage runs as well as
    adaptive runs.

11. **R11 (P2), §§2, 5, 8–9, 11 — Finish the capability and documentation
    contracts.** “Capillary fibre runs” does not unambiguously say whether ordinary
    GNLSE and solid mode-averaged runs, dispersive Kerr, scalar/tensor Raman mixtures,
    or non-device-ready z tables are admitted on `numpy_ref` and `mlx` in each phase.
    §8 also omits the normative architecture and conventions pages that currently
    require double precision and a complex128 kernel state. **Recommended
    change/question:** provide the phase/backend capability matrix and named
    refusals, including statistics and device-ready requirements, and update those
    normative pages when implementation changes their contracts. Record the
    synthetic benchmark's reproducible setup and arithmetic settings, particularly
    whether TF32 was disabled for the reported radial and multimode gains; retain
    its indicative status until the actual short-step measurements are available.

**Evidence and limits.** Review probes used the current NumPy/SciPy reference
representation and scalar binary32 arithmetic; they did not implement or qualify a
single-precision propagation path. The Raman probe and ten-multiplication phase
witness above are short, independent counterexamples, not full propagation runs.
The population probes checked both small exposure and neutral survival. The MLX
0.32.4 documentation for precision settings, compilation and custom Metal kernels
was checked; the proposal's synthetic timings, hardware-specific TF32 errors, Metal
arithmetic and GPU execution were not independently reproduced. No physical
coefficient or material model was transcribed or changed by this review.

### Maintainer decisions on the AI review — 2026-10-23

John Travers decided the four points that needed a maintainer's choice; the other seven
were resolved technically in the response below.

- **R3:** the binary64-accurate propagator is mandatory, on `mlx` by a double-float
  kernel from phase 2. This revises the earlier decision on Q4 (binary32 first, then
  measure).
- **R6:** the survival criteria proposed in the response are accepted: the envelopes,
  verdicts, floor rule and Adams rule of §7.3.
- **R8:** `mlx` refuses envelope-grid plasma until phase 4, which qualifies a device
  formulation; field-grid plasma stays in phase 2.
- **R11 (the normative pages):** a new ADR-054, written at acceptance, makes single
  precision an opt-in exception to "Double precision throughout" and records the
  `mlx` route's relation to ADR-020.

### Response to review — 2026-10-23

Every point is taken. The revision of Part I on 2026-10-23 rewrites §4.3 (range) and
§7 (verification), adds §4.4 (plasma populations), §5.1–5.3 (capabilities, admission,
configuration) and the arithmetic policy of §6.3, and adds Q12–Q14. The binary64 path
is unchanged throughout: every new form belongs to the single-precision path.

- **R1 (range of response buffers).** Taken; confirmed against the code: Raman kernels
  are per particle with `dt` folded in, so a molecular kernel's spectrum is of order
  $10^{-50}$. §4.3 now applies the range rule per buffer and per intermediate: block
  exponents for every narrowed buffer, coefficients (density, amplitudes, prefactor
  constants, scale powers, exponents) combined on the host in binary64 and applied where
  the intermediate they multiply is largest, and a declared magnitude range per pass.
  §7.2 adds the nitrogen case of the review as a vanished-response test on `numpy_ref`
  and the GPU, and the survival record reports intermediate ranges.
- **R2 (scaling of plasma and statistics).** Taken; the reviewer's derivation is right.
  §4.3 gives the scaled equations as a table of homogeneity factors: cubic terms take
  $s^2$, the free current 1 and the loss current $s^{-2}$, each held-current stream its
  own, with rates, fractions and survival evaluated at the physical field. Statistics
  declare a homogeneity degree (1, 2 or 0, tabulated); the absolute tolerance is divided
  by $s$; a zero launch takes $s = 1$. §7.2 tests scaled against unscaled responses and
  propagation separately, class P only where the operation sequences differ by powers of
  two, class R where coefficients are regrouped (§4.3, "Exactness and its limits").
- **R3 (coherent phase rounding).** Taken, with the maintainer's decision above;
  confirmed by reproducing the witness (binary32 argument: $1.89 \times 10^{-5}$ after
  one multiplication, $1.89 \times 10^{-1}$ after $10^4$; binary64 exponential then
  rounded: $1.89 \times 10^{-8}$ and $1.85 \times 10^{-4}$). §4.1 now separates random
  from coherent contributions, calls $u\sqrt M$ a heuristic under stated assumptions,
  budgets the propagator's drift and the coefficients' rounding linearly, and says the
  controller cannot see either. §4.2 makes the propagator binary64-accurate (formed,
  range-reduced and exponentiated, then rounded), so the study's two variants are those
  the reviewer distinguishes: the switch evaluates the binary32-$\Delta\Phi$ form only
  for the record. §7.2 adds constant-operator, occupied-band tests over several step
  sizes; §7.3 turns the measured drift into a documented step-count limit.
- **R4 (field-plasma population).** Taken; the draft's claim was wrong: the field path
  computes $1 - (1 - p)e^{-X}$ and the loss from $1 - f$. §4.2 and the new §4.4 specify
  the stable fraction and a separate survival array for the single-precision path, with
  the loss current and ionisation work formed from the survival; the binary64 path keeps
  its arithmetic (§9). §7.2 tests small exposure, pre-ionisation and saturation against
  density, current and ionisation-work witnesses.
- **R5 (bounds at binary32).** Taken. §4.1 no longer says the bounds scale by $2^{29}$;
  §7.1 parameterises the formulas themselves, with their domains, includes input
  narrowing and conditioned signed sums, defines when a class R bound is uninformative
  (above a tenth of the class T envelope, so the window shortens), states that backend
  agreement is never a survival verdict, and gives the TF32 probe a discriminating
  bound ($2^{-17}$ normwise).
- **R6 (reproducible decision rules).** Taken, with the maintainer's decision above.
  §7.3 fixes, before any comparison: same-grid binary64 references at
  $\mathrm{rtol} = 10^{-10}$ with their own convergence check, the metrics and
  envelopes, the minimum signal (−40 dB bands), the three verdicts and what a failure
  becomes, the floor rule (a 25 % gain), the Adams rule (at most twice Runge–Kutta's
  error and within the envelopes) and the propagator limit. §5.2 says how the study
  reaches Adams while public admission refuses it.
- **R7 (MLX arithmetic policy).** Taken; checked against MLX 0.32.4's documentation:
  `mx.compile` has no math-mode setting, `metal_kernel` has, and environment settings
  can be latched at initialisation. §6.3 now has a per-operation table of mechanism and
  evidence: protected operations in custom kernels in `"safe"` mode, each with
  adversarial tests (including a compensation that fails under reassociation); compiled
  element-wise work established by class R and special-value tests; transforms
  characterised against `fft_bound`; the GEMM probe on every dispatch family. A process
  where MLX was initialised with TF32 on is refused with an actionable message. §4.3
  adds an absolute underflow budget and says where it does not hold.
- **R8 (envelope plasma on the device).** Taken, with the maintainer's decision above;
  confirmed: `_cycle_quantities` performs a per-sample SVD and `cycle_moments` refines
  adaptively. §6.3 describes the phase 4 formulation (closed-form ellipse decomposition,
  fixed quadrature with a per-sample error estimate, a device status flag) and its
  qualification as a discretisation; `mlx` refuses envelope-grid plasma until then
  (§5.1). `numpy_ref`'s single path evaluates the moments in binary64 on the host.
- **R9 (resume and schema).** Taken, decided under ADR-051, which keeps no
  compatibility with earlier configurations or output schemas: the precision is a field
  of every run's hashed configuration, double included; the output schema changes, so
  every hash changes and earlier files are not resumed, which removes the question of a
  legacy meaning and of historical-hash tests. §5.3 fixes the scale from the launch, so
  a resume or restart recomputes it; snapshot rows are degree-one in the state and are
  stored unscaled in binary64 (class P round trip); a same-precision cross-backend
  resume is class T; the other precision is refused. The Adams snapshot layout is
  revisited in phase 4.
- **R10 (lazy-graph memory and batch synchronisation).** Taken; confirmed:
  `LockstepRunner._trial_step` needs each member's host norm at once. §6.3 bounds graph
  lifetimes (evaluation at stage or chunk boundaries above a live-bytes budget), counts
  MLX's cache and workspaces, and checks the measured peak against the estimate; it
  defines an `mlx` batch runner that builds all members' trials before evaluating their
  norms, behind the same `select_batch` interface with independent ledgers. §5 and §7.2
  require failure detection in fixed, low-storage and adaptive runs.
- **R11 (capability and documentation contracts).** Taken. §5.1 is a capability table
  by backend and phase with its refusals (GNLSE and solid-core runs admitted with the
  fibre route; the dispersive electronic response refused; the z table on `numpy_ref`
  only). §8 names the normative pages to amend (architecture.md §1 and §9, conventions.md
  §6, kernel-contract.md), and §9 drafts ADR-054. §2 now records the synthetic
  benchmark's setup, including that every reported row ran with `MLX_ENABLE_TF32=0`, and
  keeps it indicative until §7.4's measurements replace it.
