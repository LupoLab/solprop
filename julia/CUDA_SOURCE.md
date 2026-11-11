# Binary64 CUDA modal source composition

This internal slice applies the existing scalar modal source graph on CUDA. It
does not change public device admission. The first milestone covers field and
envelope Kerr, including the analytic-field and envelope harmonic forms, constant
and represented prefactors, independent host material profiles, embedded modal
projection, and in-stage temporal absorption. The response attachment adds scalar
Raman and field/envelope plasma with the same independent profiles and represented
prefactors. Binary32 device sources refuse before source preparation allocates.

The arithmetic is the same scalar contraction, Fourier convention and response
formula as the CPU path. No GEMM, reordered modal sum, vendor reduction or new
material convention is introduced. CPU signatures keep their defaults and
recurrence order. CUDA qualification requires actual device tests with scalar
indexing disabled; the host orchestration model does not qualify device code.

## Preparation and ownership

`ModalSourceGrid.crop_index` and `.omega` are owned host metadata. Source
preparation builds a separate execution-owned `SpectralPadding` from the index
map. Windows, masks, carriers and pointwise numerical buffers belong to the
selected execution. `ModalSourceMap` retains complete dense numerical maps;
`synthesize!` and `project!` accept `first_node=1` and an active local count. Each
chunk uses those full maps directly, without a noncontiguous array view or a
second copy of the map. Inactive local columns are overwritten with exact zeros.

The map's `host_synthesis`, `host_projection` and `host_coarse_projection` remain
actual host preparation copies. Binary64 CPU records alias their numerical maps;
CUDA records have separate device and host owners. Both are included in the
ordinary memory traversal, with repeated owners counted once.

`SourceProfile` tables remain host-owned scalar data. Their position evaluation
does not read the evolving field. Numerical prefactor tables use
`SourcePrefactor(execution, table, shape; arithmetic)`; this copies coefficients
and breakpoints under the selected execution. The existing two-argument table
constructor keeps its CPU behavior. Expansion terms and their weight table also
belong to the selected execution. Constant prefactors and all work arrays are
uploaded or allocated through the execution interface.

All device fills and copies use the retained execution, including successful
embedded-source snapshots. Finite checking writes one first-failing row per
column and copies only that bounded integer vector to its persistent host mirror.
The first column, then first row, determines `NonfiniteSourceError`, exactly as
on CPU. No evolving time or spectral field is downloaded by a source call.

Caller output and state must be disjoint from each other and every retained
source numerical owner. The check includes the numerical arrays exposed by
`_retained_storage` for both directions of every FFT plan. It does not traverse
opaque vendor handles. The ordinary ledger retains the plan's queried native
workspace separately from those arrays. Device arrays must be dense, live and
owned in the selected context; wrapped views are not silently staged.

## Numerical witnesses and bounds

The focused fixtures have 16 fine time samples, eight coarse time samples, two
modes and three nodes. They exercise chunks of one, two and three nodes, all
prefactor forms and both kinds of Kerr field reconstruction. The independent
reference performs direct 256-bit DFTs, complete modal matrix sums and the
explicit cubic response. It does not call the production FFT, contraction or
response helpers. Constant inputs, maps, material coefficients and profiles are
dyadic; complex harmonic carriers are treated as their supplied binary64 values.

Let `R` bound the input maximum component modulus, `S` and `P` be the maximum
absolute row sums of the synthesis and projection, `m` the stored coarse-bin
count, and `n=16`. The conservative inverse gain `a=2mS/n` covers the real
Hermitian and analytic representations. On the tested position interval, density
is at most `9/8`, the Kerr coefficient is `1/8`, and the sum of the relevant
analytic cubic factors is at most one. All windows have magnitude at most one.
The represented prefactor's scalar multiplier is at most `5/4`. Thus, with
`F=max(abs(prefactor))*5/4`,

```text
C(R) = F n P (9/64) (a R)^3,
L(R) = F n P 3(9/64) a^3 R^2
```

bound the source magnitude and its maximum-norm Lipschitz constant. These
absolute envelopes are formed before any cancellation in inverse transforms,
maps, cubic terms, projection or the forward transform.

For the finite transform fixtures, the qualification model assumes vendor
twiddle components within `2u`, correctly rounded ordinary scalar operations,
and gradual underflow. Butterfly counts alone do not establish that twiddle
hypothesis. The actual CUDA tests check the resulting outputs against the direct
MPFR references; no general cuFFT accuracy theorem is claimed. A conservative
8,192 component operations covers both 16-point transforms (including real
splitting), two-mode and three-node sums, cubic operations and the represented
degree-one scalar weights. The local source allowance is
`delta = gamma_8192*C(R) + 8192*eta`, with `u=2^-53`, `eta=2^-1074` and
`gamma_k=ku/(1-ku)`. The product cross terms are included by this single gamma
bound, rather than adding first-order estimates after cancellation.

The one-step and eight-step witnesses use the actual two-stage Heun RK leaf and
the common fixed-step session, at `h=1/128 m` with exact zero linear flow. A
second MPFR reference applies precisely that discrete RK method, so no global
accuracy tolerance conceals a changed stage history. All exact and perturbed
stages stay strictly within `R=1`, checked with their propagated allowance.
For `L=L(1)`, the discrete map's Lipschitz constant is
`1+hL+(hL)^2/2 <= exp(2hL)`. The error recurrence is

```text
E_0 = 0,
E_next = exp(2hL) E + gamma_256 (1+h C(1)) + h (2+hL) delta.
```

The represented Heun state/stage updates use fewer than 64 scalar operations
per cell, with exact zero linear flow. The checked `hL <= 1/4` limits the
second-stage amplification of those local update errors to less than two;
`gamma_256` conservatively combines those update and component contributions.
The last term includes the second evaluation's sensitivity to the first source
error. The MPFR computation verifies the second-stage unit-ball margin using
`(1+hL)E + h delta + gamma_256(1+hC(1))`, and the next accepted field using its
updated `E_next`. Hard-mask zeros, unchanged inputs, failure-row ordering and successful
call counts are exact structural contracts, not tolerance comparisons.

The host model additionally checks compact transfer counts, hidden FFT-buffer
aliases, no device allocation for precision refusal, and warmed zero CPU
allocations. Actual CUDA tests separately report device allocation, task-local
context/stream restoration, real plan storage and the one/eight-step session
results. Host orchestration allocations and vendor caches are not inferred from
the device allocation counter.


## Raman and plasma preparation

`ModalPlasmaChannel(execution::KernelExecution, channel, density, preionisation;
quadrature, arithmetic)` accepts the existing host binary64 channel and optional
cycle rule. It first calls the ordinary host preparation constructor on current
array values. A record validated earlier is insufficient: its array storage is
mutable. Only after fresh validation succeeds are its owned coefficient, node and
weight arrays uploaded. Three private clone hooks check the exact copied shapes
and element types and retain scalar metadata without inspecting device values.
The field integral taps follow the same validate-then-upload sequence. Public
rate, integral and cycle constructors retain their complete value validation.

`ModalSourceData(execution::KernelExecution, grid, map, prefactor; ...)` validates
the selected numerical owners. For envelope plasma it copies only immutable mask,
Drude and loss coefficients to temporary host arrays and applies the existing
inactive-band rule. These copies are preparation work, not retained state or an
advance-time field transfer. The old data constructor retains CPU behavior and
refuses device plasma without the execution policy. The source workspace similarly
forms its held-current spectral coefficients from explicit preparation copies.
No default plasma cutoff, material coefficient, cycle tolerance, refinement rule,
window or transform normalization changes.

Each Raman component retains a host coefficient execution. CPU sources keep their
original execution; device sources use one CPU thread and a one-thread host FFTW
plan for bounded kernel reconstruction. Scalar decay/density profiles and undamped
kernel rows stay on the host. Each source evaluation rebuilds the coefficient-only
spectrum and copies it into the same persistent device spectrum. The numerical
drive, convolution, polarisation, projection and both convolution FFTs remain on
the selected device. There is no transfer of an evolving source field. A device
planning context is not reused as a CPU coefficient-planning context.

All added storage is explicit in the ordinary owner-deduplicating ledger:

| Retained owner | Shape and type |
|---|---|
| Raman host kernel and transform input | `(2*n_time,n_terms)` and `(2*n_time,)` binary64 |
| Raman host spectrum | `(n_time+1,)` complex binary64 per component |
| Raman device spectrum | `(n_time+1,)` complex binary64 per component |
| Raman host decay and fraction arrays | two `(n_terms,)` binary64 arrays per component |
| Plasma device spline, integral or cycle arrays | exact translated shapes and binary64 types |
| Plasma compact failure scratch and host mirror | two `(2,node_chunk)` machine-integer arrays |

The existing Raman drive/convolution buffers, plasma pointwise work and projected
streams keep their previous shapes; their allocations now explicitly use the
selected execution. FFT-owned arrays and queried native workspaces retain their
separate ledger rules. CPU binary64 Raman aliases its host/device spectrum as
before, and ordinary CPU plasma needs no new status array.

## Plasma failure reporting

Each node's device thread scans its completed status column in increasing sample
order and writes `(first_sample, first_code)` or `(0,0)`. The synchronous readback
contains exactly two integers per node. The host then scans nodes in increasing
order. Thus each chunk/channel retains the original first-node/first-sample
failure, including the global node offset. Failure raises before projection or
source publication; the next channel does not consume the failed result. Inactive
columns are overwritten by the existing plasma kernels and yield zero status.
Readback execution, shapes, writable storage and overlap are checked before the
compact scratch is written. No full status matrix or evolving field is downloaded.

## Response qualification model

The new fixtures retain the existing source test's independent direct DFT and
modal sums, explicit causal Raman convolution and analytic instantaneous rate
`W(E)=k*E²`. The scalar cycle mean is `k*abs(A)²/2` and its loss moment is `k*A`.
The running integral reference is the stated trapezoidal recurrence, independently
implemented. Two channels, independent piecewise-affine profiles, preionisation,
taper, embedded projections and temporal absorption are tested separately and
combined. Field analytic/real Raman drives use their existing factors. Device
status decisions and unchanged inputs are exact structural assertions.

For these finite, well-conditioned inputs, the single-source comparison retains
`gamma_8192` times the pre-cancellation absolute contribution envelope from the
independent source reference. Here the tests conservatively use `eps(Float64)`
in that gamma instead of unit roundoff. This also covers the reference's finite
direct-DFT evaluation. It is conditional on the finite FFT/twiddle hypothesis
above and on binary64 elementary-function readings within two ulps at the tested
arguments. The log-spline coordinates have magnitude below 40 and the rate is far
below its ceiling. The reference checks every nodal magnitude, with its propagated
field allowance, strictly between both clipping thresholds. In the envelope cycle,
the omitted angular tail has mean rate at most `delta_rate=k*field_min²` and loss
moment at most `2*delta_rate/min(abs(A))`. Survival/fraction change by at most
`T*delta_rate`; explicit material and spectral gains propagate those terms to an
absolute source allowance. Each fixture checks that this allowance is below
one-sixteenth of its positive source budget. No relative agreement is asserted
for a cancelled result. The cycle
refinement budget is not enlarged. These fixture checks are not a proof of vendor
transcendental accuracy throughout an arbitrary material table.

The one/eight-step response witnesses use the same actual Heun leaf, `h=1/128`,
zero linear generator and fixed positions inside the original profile spans.
They compare with an independently applied identical recurrence. Let `a` bound
the inverse-transform/synthesis/taper gain, `M=a*R`, `T` be the sample duration,
and `b` the forward-transform/projection/prefactor gain. A cubic term contributes
`3*c*M²` to the nodal Lipschitz bound. A Raman term contributes
`3*c*d*dt*sum(abs(kernel))*M²`, where `d` is its declared drive factor. For each
quadratic plasma rate, both survival and ionisation fraction derivatives are at
most `2*k*T*M` in the maximum norm. Their products with the field are therefore
bounded by `1+2*k*T*M²`. Multiply this by the explicit density, current-integral
length, material coefficients, windows and held-current/Drude/loss factors, sum
channels, then multiply by `a*b`. The separate temporal source is linear and its
operator bound is added. All profile extrema here occur at represented affine
endpoints; prefactor norms of these affine fixtures are convex. The test helper
adds a positive 512-operation outward margin to this preparation-only bound.

The reference checks every accepted and intermediate stage inside `R=1`, including
its propagated error allowance, and checks `h*L <= 1/4`. For both local source
errors use twice the source allowance (candidate plus independent reference), and
propagate them with the same Heun Lipschitz multiplier as above. The same positive component
derivation bounds the absolute source contributions by `C=L(R)*R`; the test
checks this bound at every reference stage and uses it in both local source
allowances. State formation has a separate `gamma_256*(R+h*C)` allowance.
No fitted floor or observed
difference sets the budget. Actual CUDA results, allocation counters and scoped
context/stream preservation must pass the supplied device suite before hardware
qualification; a serial host model is evidence only for ownership/orchestration.
