# SEP 33 — Single-precision propagation, and an MLX backend for Apple GPUs

| | |
|---|---|
| **SEP** | 33 |
| **Title** | Single-precision propagation, and an MLX backend for Apple GPUs |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-23) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-10-23 |
| **Resolution** | — |
| **Requires** | [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (the floating-point policy and the bound classes this extends to binary32) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add a run-level arithmetic setting, `precision="single"`, under which the propagated
state and every per-step pass are held in IEEE binary32, and add a backend, `mlx`, that
executes such runs on Apple-silicon GPUs through Apple's MLX array framework. `numpy_ref`
gains the same single-precision path first and is its oracle, so the proposal also
answers, by measurement, which of solprop's problems survive single precision and
which do not. The MLX backend covers three routes, in the order of their use and their
measured gain: capillary fibre runs (mode-averaged and multimode, field and envelope
grids), radial free space and Cartesian free space. The precision setting, its contract
(what stays in binary64) and the binary32 bounds are written so that a later
single-precision or mixed-precision mode of `native` or `numba_cpu` reuses them
unchanged. Success is judged by agreement with `numpy_ref`'s single-precision path
within binary32 rounding bounds, by the published survival record against
double-precision references, and by recorded speed-ups over the native CPU backend on
the same machine.

### 2. Motivation and scope

**Apple-silicon GPUs are idle.** solprop runs on a Mac's CPU only: the native backend's
device builds target CUDA and HIP, and Kokkos has no Metal backend
([roadmap](../roadmap.md), "Floating-point options and reduced precision"). An
Apple-silicon Mac has a GPU with several times the CPU's single-precision throughput,
sharing the same memory, and no hardware binary64 arithmetic at all. Using it needs a
backend outside Kokkos, and that backend can only compute in binary32.

**The indicative gain is large on the large problems.** A synthetic measurement
(2026-10-23, an Apple M5 Max: 18 CPU cores, a 40-core GPU, MLX 0.32.4) timed one
interaction-picture Kerr source evaluation — the stage combination, an inverse
transform, $\gamma |E|^2 E$, a forward transform and the finish, with a real
transverse-transform GEMM each way on the radial shape — as six chained evaluations per
synchronisation. The CPU arm used the FFTW build the native backend links and fused
parallel loops; all arms alternated in one process.

| shape | CPU binary64, µs | MLX GPU binary32, µs | ratio |
|---|---|---|---|
| one line, $2^{14}$ samples | 161 | 80 | 2.0 |
| one line, $2^{16}$ | 614 | 89 | 6.9 |
| 204 × 2048 (multimode) | 922 | 109 | 8.5 |
| 64 × $2^{14}$ (a batch) | 1773 | 320 | 5.5 |
| radial, 256 × 4096 | 6968 | 512 | 13.6 |
| Cartesian, 384 × 384 × 256 | 96 213 | 21 435 | 4.5 |

The GPU results agreed with binary64 to 6–9 × 10⁻⁷ relative. The real native backend
spends more per evaluation than the synthetic CPU arm (395 µs at $2^{14}$, from
differencing fixed-schedule DOPRI5 runs), so both arms are lower bounds on a real
step; and a tuned binary32 CPU build would narrow the large-shape ratios by up to the
bandwidth factor of two. The small single-mode case gains little: the GPU's floor of
about 80 µs per evaluation is launch and synchronisation latency.

**Single precision needs proving, not assuming.** Whether a supercontinuum, a filament
or a dispersive-wave run survives binary32 depends on the problem: coherent dynamics
with moderate dynamic range should; noise-seeded dynamics at high pulse energy, and
anything that differences nearly equal large quantities, may not (§4). A
single-precision oracle on `numpy_ref` makes that a measurement with a published
record, and the record is what a later native single-precision mode would need too.

**In scope.**
- The precision setting: `SimulationSettings.precision` and the keyword interfaces,
  `KernelProblem.precision`, refusals on every backend that does not run it, and the
  recorded metadata.
- The single-precision contract (§4.2): what is binary32, what is evaluated in binary64
  on the host, and the tolerance floor.
- `numpy_ref`'s single-precision path on the routes below, as the oracle.
- `tests/bounds.py` parametrised by the unit roundoff, and the binary32 bound ladder.
- Exact power-of-two scaling of the state and scaled reductions, so that binary32's
  range holds on every route in scope (§4.3).
- The survival study: a manual page recording, per canonical case, the binary32 error
  against binary64 and what is refused.
- The `mlx` backend (an optional, macOS-arm64-only extra) on capillary fibre runs,
  radial free space and Cartesian free space; fixed-schedule lockstep batches through
  the existing `select_batch` interface where its member-owned form fits.

**Not in scope.**
- The general modal route, the hybrid planar kind, coupled flows, the moving solid
  guide, TDSE responses and the far-field statistics: refused under single precision by
  every backend, by name.
- Noise models under single precision: refused by every backend, a known limitation
  recorded on the manual page (§4.4). Admitting them is later work.
- A fused batch axis on the GPU (new batch infrastructure), and adaptive or spatial
  batches, which no backend runs.
- Single-precision or mixed-precision modes of `native` and `numba_cpu`. The contract
  and bounds here are built for them; the modes themselves are a later SEP.
- Storing outputs in binary32 (§9, Q6).
- A Kokkos Metal backend, or SYCL through AdaptiveCpp (§10).

### 3. Context

- **The seam.** A backend implements operations on the propagated state
  (`solprop.backends.protocol.Backend`); the shared step loop
  (`solprop.backends.driver.StepDriver`) owns the march, the controller, landing,
  history and noise, and a backend supplies `_trial_step`, `_apply_mask`, `stats` and,
  optionally, `_finish` and the Adams and low-storage hooks
  ([backends.md §1](../backends.md), [architecture.md §9](../architecture.md)).
- **Device-ready stage terms.** A stage's z-dependent arrays cross the seam as fixed
  terms uploaded once and a bundle of host scalars per stage
  (`solprop.translate.stage`); the propagator exponent is
  $\Delta\Phi = \sum_t s_t \Phi_t$, with the weight differences $s_t$ formed on the host
  in binary64. A problem that took the z table is not `device_ready`
  ([kernel-contract.md §3](../kernel-contract.md)).
- **The floating-point policy.** [SEP 25](archive/sep-0025-rounding-level-reproducibility.md)
  compares independent computations by numerical error, with five comparison classes
  and the bound helpers of `tests/bounds.py`, all written in binary64's `EPS` and `u`
  ([testing.md §3](../testing.md)).
- **Precision today.** Everything is binary64. The translation layer builds its tables
  in binary64; `numpy_ref` computes with NumPy and `scipy.fft`, both of which preserve
  complex64 inputs; `numba_cpu` and `native` are binary64 throughout.
- **Earlier decisions.** ADR-002 evaluated JAX as a CPU backend and did not adopt it;
  ADR-020 orders the project's priorities as accuracy and reliability, then CPU
  performance, then GPU ([decisions.md](../decisions.md)). §10 explains why this
  proposal is consistent with both.
- **The roadmap item** "Floating-point options and reduced precision" lists reduced
  precision as the one floating-point option with a large expected gain; this SEP takes
  up its MLX and entirely-single-precision bullets for the routes in scope.

### 4. Theory

#### 4.1 Binary32 rounding in an interaction-picture step

Binary32 has unit roundoff $u_{32} = 2^{-24} \approx 6.0 \times 10^{-8}$ and range
$\approx 1.2 \times 10^{-38}$ to $3.4 \times 10^{38}$ (normal numbers); binary64 has
$u_{64} = 2^{-53}$. Every bound of `tests/bounds.py` is linear in $u$, so the binary32
ladder is the binary64 one scaled by $2^{29}$, with the same operation counts
([accuracy.md §7](../accuracy.md)).

Write a stage of the interaction-picture step as
$k = \mathcal{N}(e^{\Delta\Phi}(A + \Delta z\,\textstyle\sum_j a_j k_j))$, where
$\mathcal{N}$ is the source with its transforms. Three rounding contributions matter:

1. **The source.** A transform pair on $n$ samples has relative normwise error of order
   $u \log_2 n$ (the established `fft_bound` takes $512\,\epsilon \lceil \log_2 n\rceil$
   as its margin), and the pointwise response adds a few $u$. The error enters the
   state scaled by $\Delta z$: per step, relative to the state, about
   $c\,u\,\log_2 n \cdot \phi_{\mathrm{NL}}$, where
   $\phi_{\mathrm{NL}} \approx \Delta z\,\lVert\mathcal{N}\rVert / \lVert A \rVert$ is the
   step's nonlinear phase.
2. **The state.** Holding the state in binary32 rounds it by up to $u$ relative at
   every accepted step. Over $M$ steps independent roundings grow like
   $u\sqrt{M}$, multiplied by the propagation's sensitivity (the growth factor of
   `propagation_growth`); for strongly amplifying dynamics (modulation instability,
   soliton fission) the sensitivity, not $\sqrt{M}$, dominates.
3. **The linear phase.** $\Delta\Phi$ is formed from binary64 weight differences and
   binary32 terms: its error is about $n_t\,u\,\max_t |s_t \Phi_t|$ radians for
   $n_t$ terms, a single rounding for a constant operator. At the band edges,
   $|\Delta\Phi|$ per step can reach $10^{2}$–$10^{4}$ rad, so the phase error there
   is $10^{-5}$–$10^{-3}$ rad per step, accumulating as $\sqrt{M}$ — never as the
   cumulative phase, which the interaction picture never forms. Whether this matters
   depends on the spectral amplitude at those edges, largest on gradients whose
   expansions combine many terms; §4.2 keeps $\Delta\Phi$ in binary32, and the
   phase 1 study measures it against a binary64 increment (Q4).

**The error estimate does not floor at $u$.** An embedded estimate is
$\Delta z \sum_i (b_i - \hat b_i) k_i$; its rounding is about $u\,\phi_{\mathrm{NL}}$
relative to the state, far below $u$ when the step's nonlinear phase is small. The
controller therefore does not stall. The real floor is on the *useful* tolerance: a
local error target below the per-step rounding of contributions 1–2 buys steps without
accuracy. Taking the per-step rounding as $c\,u_{32}$ with $c$ of order 10 gives a
floor of $\mathrm{rtol} \ge 10^{-6}$ (Q1); the study may raise it to $10^{-5}$ if
$10^{-6}$ shows no accuracy gain over $10^{-5}$ on the canonical cases.

#### 4.2 The single-precision contract

Under `precision="single"`:

- **Binary32:** the propagated state, the stage values, every transform, the
  responses (Kerr, Raman, plasma), the transverse transforms (the Hankel GEMM, the
  Cartesian transforms), the stage combinations, the error estimate's array and the
  statistics' pointwise work.
- **Binary64, on the host:** $z$, $\Delta z$, the controller and its norms' final
  scalar, the stage weight differences $s_t$ and every other bundle scalar, landing,
  schedules and save planes. These are already host scalars.
- **Binary64, built once and rounded at upload:** every table of the translation layer
  (operator terms, prefactors, masks, Raman kernels, rate tables, quadrature and
  transform matrices). The translation layer does not change; a backend rounds the
  arrays it uploads.
- **Accumulated, not summed naively:** every long reduction or running integral (the
  norms, the statistics, the plasma's running integrals over time) uses a pairwise or
  blocked order with the bound of its tree, or binary32 compensated summation, and
  states that bound (§7): a sequential binary32 sum of $10^4$ terms could lose
  $10^4 u_{32} \approx 6 \times 10^{-4}$ of its value. The orders are the oracle's
  (§6.2), so the other backends have a defined target (Q5). The ionised fraction keeps
  its `expm1` form, which holds relative accuracy for a small fraction.
- **Scaled:** the state is held as $\tilde a = a / s$ with an exact power of two $s$
  (§4.3), and every sum of squares scales before squaring.
- **Never:** flush-to-zero imposed by solprop, approximate transcendental functions,
  reduced-precision matrix products (TF32 or lower), or any fast-math flag, as
  SEP 25 §5 requires. Subnormals a GPU flushes in hardware are accepted and recorded.

This is the contract `numpy_ref` implements as the oracle and `mlx` reproduces.
A later mixed mode (binary32 source, binary64 state) differs only in its second item
and needs a separate precision value, not a different contract (§6.1).

#### 4.3 Range and scaling

**Pointwise work fits in SI units.** A field of $10^{12}$ V/m has $E^3 = 10^{36}$;
$E^3$ above $3.4 \times 10^{38}$ needs $|E| > 7 \times 10^{12}$ V/m, an intensity of
about $6 \times 10^{18}$ W/cm², far beyond the routes' strong-field regime. The
electron density is below $10^{28}$ m⁻³. Rates such as $\exp(-a/|E|)$ underflow to
zero, or to a flushed subnormal, at small fields; the binary64 path already reaches
zero there for the smallest fields, and the difference is below every rate table's
stated accuracy.

**Reductions do not.** The transforms are unnormalised, so a spectral sample is the
temporal amplitude times up to the transform length. On the Cartesian route
($n \approx 4 \times 10^7$ samples, fields near $10^{11}$ V/m) a spectral sample
reaches about $4 \times 10^{18}$, its square about $10^{37}$, and an energy or fluence
sum over the grid about $10^{44}$: an overflow. Fibre states (true-power amplitudes in
$\sqrt{\mathrm{W}}$, conventions.md §4) have less extreme values but the same
structure.

**Scaling.** Two measures, both exact or bound-preserving:

1. **A power-of-two state scale.** At `set_initial` the backend chooses
   $s = 2^k$ with the launch's peak temporal amplitude $\max_t |a(t)| / s \in [1, 2)$,
   and holds $\tilde a = a / s$. Because $s$ is a power of two, $a \mapsto a/s$ and its
   inverse are exact in binary floating point (absent overflow and underflow), so the
   scaling changes no rounding, only the headroom. The equation for $\tilde a$ is the
   original with each response's homogeneous degree folded into its prefactor: a cubic
   response (Kerr, Raman) takes $s^2$; the plasma's rates take the physical field
   $s\,|\tilde a|$ (one exact multiplication) and its current is linear in the field;
   the linear operator is unchanged. The controller's absolute tolerance is divided by
   $s$; `observe`, `stats` and snapshots multiply by $s$ on the way out, and a resume
   divides on the way in. With the peak near one, cubes and long sums have about thirty
   orders of magnitude of headroom, and spectral wings $10^{-38}$ below the peak remain
   representable, far below the binary32 rounding floor.
2. **Scaled reductions.** Every sum of squares divides by a power-of-two scale chosen
   from the largest magnitude before squaring, as LAPACK's `nrm2` does, and multiplies
   back in binary64 on the host.

The scale is private to the backend and the oracle: the physics and translation layers
keep SI units, and no saved quantity changes. The survival study reports the largest
and smallest magnitudes met in each case, after scaling.

#### 4.4 Noise

A one-photon-per-mode seed (`noise`) has a spectral amplitude relative to the pulse's
peak spectral amplitude of roughly $\sqrt{\hbar\omega_0\,n_{\mathrm{occ}}/E_p}$, with
$n_{\mathrm{occ}}$ the occupied bins: about $1.6 \times 10^{-5}$ for a 1 µJ pulse over
1000 bins, $5 \times 10^{-7}$ at 1 mJ. Accumulated binary32 rounding reaches
$u_{32}\sqrt{M} \approx 6 \times 10^{-6}$ after $10^4$ steps. Noise-seeded
statistics might survive at low energy and short lengths, and would fail at high
energy. This SEP does not qualify that boundary: a noise model under single precision
is refused by every backend, and the manual page records it as a known limitation
(Q3).

### 5. Proposed interface

```text
from solprop.driver import prop_capillary
from solprop.stepper import RungeKutta

out = prop_capillary(
    ...,                       # the physics, unchanged
    precision="single",        # "double" (default) or "single"
    backend="mlx",             # or "numpy_ref"; "auto" resolves to "mlx" when installed
    rtol=1e-5,                 # at or above the floor, 1e-6
    noise=None,                # noise models are refused under single precision
)                              # the integrator defaults to RungeKutta() here
# out.metadata records precision="single"; out.fields() returns binary64 arrays
```

- **`precision`**: `Literal["double", "single"]` on `SimulationSettings`, `Simulation`
  and every keyword interface that takes `backend`. The default is `"double"`, and
  nothing about a double-precision run changes.
- **`backend="mlx"`** requires `precision="single"` and refuses `"double"` by name,
  saying that Apple GPUs have no binary64 arithmetic. It refuses any problem outside
  its routes (§2) naming the backends that run it.
- **`backend="auto"`** with `precision="single"` resolves to `"mlx"` when MLX and a
  Metal device are present, and otherwise refuses, naming `numpy_ref` (the oracle, not
  a production backend) and the `mlx` extra. It never resolves to a backend that would
  run in binary64.
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
- **Outputs**: `observe` returns binary64 host arrays (an exact widening), so files,
  processing and plotting are unchanged; the run's metadata records the precision, and
  the precision is part of the resume key, so a run cannot resume in the other
  precision.
- **Memory**: `estimate_memory` and the memory ledger count binary32 bytes for the
  backend's resident arrays; `available_device_memory` on `mlx` reports the Metal
  device's recommended working set less MLX's active memory.
- **Batches**: `backend="mlx"` implements `select_batch` with member-owned stages; a
  member agrees with its solo run within a binary32 matched-discrete bound.

### 6. Implementation strategies

#### 6.1 General support (outside the backends)

- **The setting.** `SimulationSettings.precision` and the keyword interfaces pass a
  `Precision` record into `assemble`, stored as `KernelProblem.precision`.
  Recommended: a frozen record with named fields (`state`, `source`), each `"double"`
  or `"single"`, and the public strings `"double"` and `"single"` mapping to equal
  fields. A later mixed mode is then a third public value, not a schema redesign.
  `SCHEMA_VERSION` changes in phase 1.
- **Validation.** `KernelProblem.validate` refuses single precision on the routes out
  of scope, naming the route, so every backend refuses alike; each backend's admission
  then refuses what it does not run.
- **The translation layer is unchanged.** Tables stay binary64; rounding to binary32 is
  each backend's upload. This keeps the change out of the physics and translation
  layers and is what a native single mode would also do.
- **Bounds.** `tests/bounds.py` gains the unit roundoff as a parameter (default
  binary64) and a `UNIT_ROUNDOFF_32`; `gamma`, `fft_bound`, `scan_bound` and the
  ladder take it. Existing call sites are unchanged.
- **Metadata and resume.** The precision enters `RunMetadata` and the resume key.
- **Memory.** `estimate_memory` takes the element sizes from the precision.
- **Defaults.** Under single precision the interfaces take the floor as their default
  `rtol` and `RungeKutta()` as their default integrator, and refuse an explicit
  `Multistep` and any noise model, in one shared admission function so that every
  interface and backend agrees.
- **Scaling helpers.** The choice of the power-of-two scale and the scaled sum of
  squares (§4.3) are small shared helpers, so `numpy_ref`, `mlx` and a later native
  mode compute the same scale.

#### 6.2 `numpy_ref`

The oracle keeps one function per pass and no fusion. Under single precision it:

- rounds uploaded terms and tables to float32/complex64 at construction;
- relies on NumPy's and `scipy.fft`'s preservation of complex64, and on NumPy 2's
  rule that Python scalars do not promote, for the arithmetic;
- asserts the dtype of the state and of every stage value after each pass, so that a
  binary64 array that slips into a binary32 pass (a float64 table not rounded, an
  intermediate created as complex128) fails a test instead of silently computing in
  binary64;
- holds the state at the power-of-two scale of §4.3 and folds it into the prefactors;
- evaluates reductions pairwise and the running integrals over time (including the
  field grid's band-exact running sums) as a blocked, compensated scan, each with its
  stated bound, rather than with `np.sum`'s and `np.cumsum`'s orders: the oracle's
  orders are the target the other backends are compared against (Q5);
- has a private study switch that forms $\Delta\Phi$ in binary64 and rounds it, for
  the phase increment's measurement (Q4).

Most of the package's explicit binary64 dtypes (about 740) are in the physics and
translation layers, which do not change; `numpy_ref`'s files carry about fifty, and the
shared driver a handful.

#### 6.3 The `mlx` backend

**Layer and package.** A new backend package, `solprop.backends.mlx_gpu` (not `mlx`,
which would shadow the library inside the package), registered as `"mlx"` with the
extra `solprop[mlx]`. Nothing outside it imports MLX; availability is probed with
`importlib.util.find_spec`, as for every backend.

**Structure.** It subclasses the shared `StepDriver` like `numpy_ref`, so the march,
controller, landing, history and noise are the shared ones. Its passes follow
`numpy_ref`'s in MLX, which is NumPy-like:

- stage combination, responses and finish as `mx.compile`d functions, which fuse the
  element-wise work into one kernel per pass;
- transforms by `mx.fft` (complex64 only; real transforms for field grids);
- the radial route's Hankel transforms as real GEMMs on the real and imaginary parts;
- the Cartesian route's two-dimensional transverse transforms by `mx.fft.fftn`;
- the state at the shared power-of-two scale (§4.3), with the scale folded into the
  prefactors as on the oracle;
- the error norm and the statistics reduced on the device to a few scalars, scaled
  before squaring and copied to the host once per trial or plane;
- custom Metal kernels (`mx.fast.metal_kernel`) only where a measured pass needs one,
  for example a compensated scan for the plasma's running integrals.

**Evaluation and synchronisation.** MLX is lazy: a stage builds a graph and nothing
runs until `mx.eval`. The backend evaluates once per trial step, when the controller
needs the norm, so six stages queue without host round trips; this is the pattern the
indicative measurement timed.

**Arithmetic settings.** MLX 0.32.4 on the M5 Max's GPU computes float32 matrix
products in TF32 by default (a relative error of 7.7 × 10⁻⁴ was measured on 256 × 256
products, against 2.9 × 10⁻⁷ in float32), and honours `MLX_ENABLE_TF32=0`, which
restores float32 at about a quarter more GEMM time. TF32 is
outside the contract (§4.2): the backend sets the variable before MLX is imported in
its process, and at construction runs a probe product and refuses to run if its error
exceeds the float32 bound, so a changed default in a later MLX cannot pass silently.
Compiled functions and custom kernels use MLX's `"safe"` math mode, never `"fast"`.

**Batches.** `select_batch` with member-owned stages, as on the other backends: each
member's graph is queued and the batch is evaluated together, which shares the
synchronisation without a fused batch axis. The gain over solo runs is measured in
phase 2; if it is small, as on the native device builds, the batch is still provided
for parity and the record says so (Q10).

**Cost.** Binary32 halves resident memory against `numpy_ref`. The GPU and CPU share
memory, so a host copy for `observe` or `stats` is a copy, not a transfer. Kernel
launch latency sets a floor of tens of microseconds per stage, which is why the small
single-mode case gains little.

**Testing without a GPU.** MLX also runs its graphs on the CPU. The backend accepts an
internal device choice, so in continuous integration its tests run on MLX's CPU
device, which checks the graph code and skips the custom Metal kernels. A local GPU
lane, the whole `mlx` test suite on the Metal device, is part of each phase's
definition of done, as the device lanes are for the native builds. Phase 2 checks once
whether a hosted macOS runner exposes a usable Metal device and records the result
(Q7).

#### 6.4 Paving the way for `native` and `numba_cpu`

What this SEP builds and a later native or numba single mode would reuse unchanged: the
public setting, the `Precision` record and its schema field, the route refusals, the
contract of §4.2, the binary32 bounds, the oracle path, the survival record and the
tolerance floor. What it would add on its own: templating or recompiling its kernels
on the real type, Kokkos-FFT and Kokkos Kernels in single precision (both support
it), and its own protected blocks. A mixed mode adds a third precision value.

#### 6.5 Dependencies

`mlx` (MIT licence, compatible with GPL-3.0-or-later), as an optional extra with the
environment marker `sys_platform == "darwin" and platform_machine == "arm64"`, pinned
to a tested minor version (`mlx>=0.32.4,<0.33` at drafting) and added to
[dependencies.md](../dependencies.md), with the macOS floor taken from MLX's wheels.
The maintainer has agreed to it as an optional extra. A pin is raised only in a pull
request that runs the GPU lane and the TF32 probe (Q9).

### 7. Verification and validation

**Class R, matched discrete computation.**
- `mlx` against `numpy_ref`, both single precision, on fixed schedules with a
  fixed-order method, every route and response in scope, within the binary32 ladder
  (the binary64 ladder's operation counts at $u_{32}$, with the same margins).
- A batch member against its solo run, and a repeated run against itself, within the
  same bounds.
- `numpy_ref` single precision against `numpy_ref` double precision for one step,
  within a bound in $u_{32}$, which tests the oracle path itself.

**Class T, accuracy: the survival study.** Each canonical case below runs in double
precision at a tight tolerance (the reference) and in single precision at the floor
and at a looser tolerance. The record gives the field, spectrum and observable errors
(energy, peak power, spectral energy per band, ionised fraction, beam width) and a
verdict: survives, survives with a stated limit, or fails. Cases run on reduced
windows in tests; the full-length comparison is a worked example, run by hand.

| case | route | expected |
|---|---|---|
| soliton and SPM, dispersion-free analytic phase | fibre, envelope | survives |
| reduced resonant dispersive-wave emission in helium, two modes, gradient | fibre, field | survives |
| reduced comb-to-continuum, noise-free | fibre, field | survives with a limit |
| argon plasma blue shift | fibre, field and envelope | to measure |
| radial Kerr beam and a short plasma filament | radial | to measure |
| Cartesian Gaussian diffraction and a Kerr beam | Cartesian | survives |
| the dispersive-wave and plasma cases under the Adams integrator | fibre | to measure (Q2) |
| a gradient case with $\Delta\Phi$ in binary32 and in binary64 | fibre, field | to measure (Q4) |
| a noise model | any | refused: a known limitation (§4.4) |
| a tight-tolerance run below the floor | any | refused |

**Analytic checks in single precision.** Dispersion-free SPM's phase
$P\int\gamma\,dz$; linear propagation of a chirped Gaussian; Gaussian-beam diffraction
on the radial and Cartesian routes; photon-number conservation in lossless Kerr
propagation to the binary32 growth bound.

**Arithmetic probes.** The TF32 probe (§6.3); the dtype guards of `numpy_ref`; a test
that no float64 array reaches an `mlx` pass.

**Scaling.** On `numpy_ref`, a run at its chosen power-of-two scale against the same
run forced to scale one, over a few steps of a case far from the range limits: a
class P comparison, exact because scaling by a power of two commutes with rounding
when nothing overflows or becomes subnormal. On `mlx`, the same pair within the class R
bound. A Cartesian case whose unscaled energy sum would overflow binary32 runs and
reports its energy to the binary32 bound.

**Benchmarks.** Each phase records, on short windows, the time per accepted step of
`mlx` against `native` (binary64, CPU) on the same Mac, at the shapes of §2 and the
benchmark suite's corresponding cases, under the benchmark rules
([benchmarks.md](../benchmarks.md#rules)). The records inform the user guide's advice;
they are not merge gates.

### 8. Documentation

- A user guide page, *Single precision and Apple GPUs*, under the performance guide:
  when to use it, what is refused, the survival record, and the tolerance floor.
- The backends guide ([Choosing a backend](../../guide/performance/backends.md)): the
  `mlx` entry and its installation.
- [backends.md](../backends.md): a section on `mlx`, a column in §9 *What runs where*,
  and §11's limits; [kernel-contract.md](../kernel-contract.md): the precision field;
  [testing.md](../testing.md) §3: the binary32 bounds.
- Docstrings for `precision` on every interface that takes it.
- A worked example comparing single and double precision on one capillary and one
  radial case, run by hand with `poe docs-examples`.

### 9. Compatibility and impact

- **Defaults unchanged.** `precision="double"` is the default; double-precision results
  do not change.
- **Schema.** `SCHEMA_VERSION` changes in phase 1 for `KernelProblem.precision`.
- **Outputs.** No change to the output schema's arrays; the metadata gains the
  precision. Storing fields in binary32 would halve file sizes and is deferred (Q6).
- **Resume.** The precision is in the resume key.
- **Luna.jl** has no single-precision mode; nothing to record on the differences page.

### 10. Alternatives considered

- **A Kokkos Metal backend.** Kokkos compiles device lambdas with a single-source
  device compiler; no such compiler targets Metal, Apple GPUs have no binary64, and
  Metal has few 64-bit atomics. The runtime part of a backend is modest (each existing
  GPU backend is 5–10 thousand lines), but the compiler is a research project and
  would have to be maintained against every Kokkos release.
- **SYCL through AdaptiveCpp.** AdaptiveCpp's generic compiler has an experimental
  Metal target (no binary64, emulated 64-bit atomics, no `printf`, per-operation command
  buffers). Kokkos's SYCL backend depends on oneAPI extensions and has no
  AdaptiveCpp configuration, so this means porting that backend or rewriting the
  native kernels in SYCL, and still needs a Metal FFT. Worth revisiting if Kokkos gains
  an AdaptiveCpp configuration upstream.
- **JAX on Metal, PyTorch's MPS device.** JAX's Metal plugin is experimental; ADR-002's
  measurements stand against JAX as a CPU backend. PyTorch's MPS device covers complex
  arithmetic and FFTs only partially. MLX is Apple's own, supports complex64 transforms
  and custom Metal kernels, and is used here only for the Mac GPU, where no other
  backend runs — so ADR-002's CPU comparison does not apply, and ADR-020's order is
  respected: the proposal adds a GPU route without touching the CPU backends.
- **A mixed mode first** (binary32 source, binary64 state) on `native`. Better accuracy
  for the same transform cost on CPUs and reduced-binary64 NVIDIA GPUs, but impossible
  on Apple GPUs, which have no binary64 for the state; it remains open on the roadmap
  and reuses this SEP's setting (§6.4).
- **MLX without a `numpy_ref` path**, comparing against double-precision `numpy_ref`
  with tolerances sized for binary32. Less work, but every comparison would mix
  rounding with method and leave no oracle for a later native mode; not chosen.
- **Double-float emulation of binary64 on the GPU.** Correctly rounded emulation costs
  of the order of ten binary32 operations per binary64 one; for the whole state it
  would erase the gain. It remains an option for the phase increments alone (Q4).

### 11. Phases

Each phase merges on its own with a green gate.

**Phase 1 — The precision setting, the oracle path and the survival study.**
- The setting, the `Precision` record, `KernelProblem.precision`, validation and
  refusals on every backend; the single-precision defaults (the `rtol` floor,
  `RungeKutta()`) and refusals (`Multistep`, noise) in one admission function;
  metadata, resume key, memory estimate; `SCHEMA_VERSION`.
- The shared scaling helpers (§4.3).
- `numpy_ref`'s single-precision path on the fibre, radial and Cartesian routes: the
  dtype guards, the power-of-two scale, the pairwise reductions and compensated scans,
  and the study switch for $\Delta\Phi$.
- `tests/bounds.py` parametrised by $u$; the class R test of `numpy_ref` single against
  double over one step; the scaling tests.
- The survival study on reduced cases, the Adams and $\Delta\Phi$ measurements among
  them, and its manual page with the known limitations; the floor, Adams and
  $\Delta\Phi$ outcomes recorded in Part II.
- Verifies: no double-precision result changes (the existing suite), the R and T tests
  above, the refusals.

**Phase 2 — `mlx` on capillary fibre runs.**
- The backend package, registry entry, extra and dependency record; the TF32 probe;
  the CPU-device test lane in continuous integration and the local GPU lane.
- Mode-averaged and multimode fibre runs on field and envelope grids: Kerr, scalar and
  tensor Raman, field and envelope plasma, loss, both absorber schemes, device-ready
  gradients; Runge–Kutta and the low-storage march; statistics; `observe`.
- Fixed-schedule lockstep batches through `select_batch`, with their gain over solo
  runs measured.
- Verifies: class R against `numpy_ref` single on every response; batch members;
  the scaling pair; benchmarks against `native` at $2^{14}$, $2^{16}$ and the
  multimode shape.

**Phase 3 — `mlx` on radial and Cartesian free space.**
- The radial route (Hankel GEMMs, Kerr, plasma and Raman on nodes) and the Cartesian
  route (transverse transforms), with their collars and statistics.
- Verifies: class R against `numpy_ref` single; Gaussian-beam diffraction; the
  Cartesian energy case that would overflow unscaled; benchmarks
  against `native` on the radial and Cartesian shapes and the suite's filament and
  Kerr-beam cases.

**Phase 4 — Admissions the study allows, and measured kernels.**
- The Adams integrator under single precision, if the phase 1 study admits it (Q2),
  with the default integrator then restored to `Multistep` under single precision.
- A double-float phase increment on the device, if the study showed $\Delta\Phi$ in
  binary32 to be visible in observables (Q4).
- Custom Metal kernels for the passes the phase 2–3 records show to be bound by launch
  count or memory traffic.
- The worked example and the final records; the SEP becomes Final when every backend
  that admits single precision agrees with the oracle within its bounds.

### 12. Open questions

The maintainer decided Q1–Q11 on 2026-10-23 (Part III). What remains open is what the
phase 1 study and the phase 2–3 measurements settle within those decisions.

- **Q1 — The tolerance floor. Decided:** $\mathrm{rtol} \ge 10^{-6}$; an explicit value
  below it is refused, and every interface's single-precision default is the floor.
  *Open:* the study may raise the floor to $10^{-5}$ if $10^{-6}$ gains no accuracy
  over it (phase 1).
- **Q2 — Adams under single precision. Decided:** the phase 1 study measures it; until
  it is admitted the single-precision default integrator is `RungeKutta()` and an
  explicit `Multistep` is refused. *Open:* admission, in phase 4, if its accuracy
  envelope stays within a fixed factor of Runge–Kutta's on the canonical cases.
- **Q3 — Noise. Decided:** refused under single precision by every backend, a known
  limitation recorded on the manual page; admitting noise is outside this SEP.
- **Q4 — The phase increment. Decided:** $\Delta\Phi$ in binary32, measured in phase 1
  against a binary64 increment through `numpy_ref`'s study switch on the gradient
  cases. *Open:* a double-float increment on the device in phase 4, only if the
  difference is visible in observables.
- **Q5 — Reductions. Decided:** the oracle defines the orders: pairwise reductions and
  a blocked, compensated scan for running integrals (the field grid's band-exact
  running sums of [SEP 18](archive/sep-0018-band-exact-plasma-integrals.md) among
  them), each with its bound; every `mlx` choice is a class R test against it.
  *Open:* the per-pass forms on `mlx` (phases 2–3).
- **Q6 — Binary32 outputs. Decided:** deferred; outputs stay binary64.
- **Q7 — Continuous integration. Decided:** CI runs the `mlx` tests on MLX's CPU device;
  a local GPU lane is part of each phase's definition of done. *Open:* whether a hosted
  macOS runner exposes a usable Metal device, checked once in phase 2.
- **Q8 — `auto` under single precision. Decided:** `mlx` when it and a Metal device are
  present, otherwise a refusal naming the extra and `numpy_ref`; never `numpy_ref` by
  resolution.
- **Q9 — The MLX pin. Decided:** `mlx>=0.32.4,<0.33`, the macOS floor from MLX's wheels,
  raised only in a pull request that runs the GPU lane and the TF32 probe.
- **Q10 — Batches. Decided:** member-owned batches through `select_batch` in phase 2,
  kept for parity whatever their measured gain, which the record states; no fused
  batch axis.
- **Q11 — Range. Decided:** the exact power-of-two state scale and scaled reductions of
  §4.3, private to the backends and shared through one helper.

### 13. References

1. IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (2019).
   doi:10.1109/IEEESTD.2019.8766229
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   doi:10.1137/1.9780898718027
3. J. C. Schatzman, "Accuracy of the discrete Fourier transform and the fast Fourier
   transform," *SIAM J. Sci. Comput.* **17**, 1150–1166 (1996). doi:10.1137/S1064827593247023
4. A. Hannun, J. Digani, A. Katharopoulos and R. Collobert, *MLX: Efficient and
   flexible machine learning on Apple silicon*, software (2023);
   documentation at https://ml-explore.github.io/mlx/ (accessed 2026-10-23).
5. AdaptiveCpp documentation, "Installation instructions for Metal (macOS),"
   https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ (accessed 2026-10-23).

## Part II — Implementation record

## Part III — History

- 2026-10-23: Drafted, after an indicative synthetic MLX measurement on an Apple M5 Max.
  The maintainer chose the routes (capillary fibre runs, radial and Cartesian free
  space; not the general modal route), batches only through the existing interface,
  MLX as an optional extra, and a `numpy_ref` single-precision oracle.
- 2026-10-23: Revised with the maintainer's decisions on Q1–Q11: the `rtol` floor and
  defaults, Runge–Kutta by default until Adams is admitted, noise refused as a known
  limitation, the oracle's reduction orders, the CI and GPU lanes, the MLX pin, and a
  new §4.3 on range: reductions of unnormalised spectra overflow binary32 on the
  Cartesian route, so the state is held at an exact power-of-two scale and sums of
  squares are scaled.
