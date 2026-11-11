# Source arithmetic and physical scales

This implementation note complements [the shared precision contract](PRECISION.md)
and [SEP 33](../docs/dev/seps/archive/sep-0033-single-precision-mlx.md).
It describes numerical leaves, not public binary32 propagation admission or device
qualification.

`SourceArithmetic{T}` separates stored spectral state `a / 2^es` from stored nodal
field `E / 2^ef`. A synthesis map therefore absorbs `2^(es-ef)` and its matching
projection absorbs `2^(ef-es)`. Profiles and stage coordinates remain binary64.
The cubic coefficient absorbs `2^(2ef)`; the loss-current coefficient absorbs
`2^(-2ef)`. Material factors and densities are combined before one response-precision
rounding. No raw `Ip / 2^(2ef)` intermediate is formed.

`source_narrow` and its host-buffer counterpart `source_narrow!` return a named
`SourceRange`. They reproduce the NumPy single-precision oracle's direct-fold
admission: the largest magnitude is strictly below the binary32 maximum, and if
that largest magnitude is below one, every nonzero magnitude is at least the
binary32 normal minimum. At or above one, tiny entries may disappear within the
normwise underflow allowance of SEP 33 §4.3. This does not promise relative
accuracy for each entry, and does not license dividing by a discarded entry.
The full scan precedes destination mutation. Partial overlap, including separately
wrapped foreign storage, is refused. Exact in-place scaling is supported.

With `normalize=true`, the stored maximum is in `[1,2)` and the returned block
exponent remains explicit. The conversion does not first form a potentially
overflowing or vanishing physical power. In particular a tiny Raman kernel can be
retained and its exponent combined with density and material factors later.
`largest` and `smallest_nonzero` describe the binary64 buffer immediately before
rounding, after normalization when selected. An all-zero buffer has zero block
exponent and retains signed zeros.

`source_coefficient` follows the shared precision translator's mantissa/exponent
construction. For `n` finite nonzero binary64 factors, each mantissa product stays
in `[1/4,1)`, so no intermediate product overflows or underflows. Exact powers are
carried as checked integers. Under round-to-nearest, the relative product error is
bounded by `gamma_n(binary64)` before the final rounding; a normal binary32 result
adds its one rounding error. Nonzero final binary32 subnormals and zeros are
refused, matching `Coefficient.binary32`. A structural zero preserves the product
sign, and every supplied factor is still checked for finiteness.

These preparation facts do not bound FFT roundoff, response conditioning, future
fields or cumulative integrals. Those obligations remain attached to the numerical
pass and its independently derived tests. In particular, an accepted source buffer
does not establish the finite-range hypotheses of DF32 exponent assembly.

## Protected population and current arithmetic

Rate-table coefficients use the response precision. Their physical clipping bounds
and logarithmic grid remain immutable host binary64 metadata. At evaluation the
host shifts the log origin by `ef*log(2)` and scales the clipping coordinates by
`2^-ef`, then rounds those scalar parameters once. The pointwise table lookup therefore
uses the stored field directly. For real representable magnitudes, clipping bounds
are rounded upwards so comparison retains the side of the exact scaled threshold.
Collapsed, overflowing or subnormal scaled coordinate ranges are refused before
kernel writes. This does not promise bitwise agreement with binary64 rate lookup:
rate-table coefficient, coordinate and elementary-function errors remain part of a
conditioned source comparison.

Binary32 field and envelope exposure/current integrals use a doubling prefix scan.
At level `k`, row `i` adds the previous level's row `i-2^k`. Traversing backwards in
each owned column realizes that graph in place and needs no additional scratch.
Every prefix has at most `d=ceil(log2(n_time))` rounded additions, so its error relative
to the represented increments is bounded by `gamma_d * sum(abs(increments))` when
all intermediates remain finite. Additive gradual-underflow errors can be bounded
by one half-minimum-subnormal per addition edge. FIR and trapezoid construction
errors are separate inputs to that bound. No column shares storage with another;
the CPU launcher waits for every column. This is the shared NumPy oracle's protected
Hillis–Steele graph, but comparisons use rounding bounds, not a bitwise requirement.
Binary64 retains its sequential prefix graph.

`keep=Float32(1-preionisation)` is formed from the host value independently of
`initial=Float32(preionisation)`. Thus a small neutral fraction survives even when
the initial ionisation rounds to one. The stable population pair is
`initial + keep*(-expm1(-exposure))` and `keep*exp(-exposure)`, never a subtraction
from the rounded ionised fraction. In binary32 current evaluation, density is folded
into the charge and loss coefficients before the pass. Its current work therefore
already includes that stage density; binary64 retains its original per-atom work.

The envelope uses the translated cycle-refinement tolerance and negligible-rate
floor, with both positive directional moments tested. No tolerance is enlarged to
make binary32 converge. Exhaustion remains `PlasmaQuadratureFailure`. The committed
analytic fixture tests the default `2e-6` rule; it does not establish arbitrary
ionisation-table conditioning or future GPU transcendental accuracy. Nonfinite raw
exposure is a failure even if saturated populations leave finite currents (#701).

The cycle panel's existing magnitude-ordered compensated sum has the Sum2 graph:
each residual is an error-free addition residual, accumulated separately before
the final addition. Proposition 4.5, equation (4.8), of
[Ogita, Rump and Oishi (2005)](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)
([DOI](https://doi.org/10.1137/030601818)) bounds it by
`u*abs(sum) + gamma_(n-1)^2*sumabs` for `n*u<1`, including gradual underflow and
excluding overflow. This controls accumulation of the represented panel terms;
node, weight, rate, trigonometric and cutoff-conditioning errors remain separate.
