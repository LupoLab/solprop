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
