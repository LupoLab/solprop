# Binary64 CUDA modal source composition

This internal slice applies the existing scalar modal source graph on CUDA. It
does not change public device admission. The first milestone covers field and
envelope Kerr, including the analytic-field and envelope harmonic forms, constant
and represented prefactors, independent host material profiles, embedded modal
projection, and in-stage temporal absorption. Device Raman and plasma have named
refusals pending their separate attachment. Binary32 device sources refuse before
source preparation allocates.

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
