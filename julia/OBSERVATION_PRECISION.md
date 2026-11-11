# Observation arithmetic and physical scaling

The binary32 observation path follows SEP 33's separation of stored numerical work
from physical host results. State and reconstruction arrays, per-time power, per-node
fluence and reduction work use binary32. Coordinates, represented host coefficient
evaluation, final packed statistics and monitor/control/continuation records use
binary64. A physical field is a stored field multiplied by its explicit power of two;
that scale is never formed inside a state-sized product or square.

This document specifies the reduction primitives. Integration of each observation and
monitor family is qualified separately; the primitives do not establish FFT accuracy,
model accuracy, device compatibility or complete binary32 session support.

## Protected sum and its admission

`_Float32Sum()` is an immutable pair of binary32 numbers. `_float32_add(sum, value)`
uses the protected error-free `TwoSum` already used by DF32, followed by an ordinary
binary32 correction sum. `_float32_value(sum)` rounds the sum and its correction.
There is no reassociation, contraction or flush to zero in this computation.

This is Sum2s, Algorithm 4.1 of Ogita, Rump and Oishi, *Accurate Sum and Dot Product*,
SIAM Journal on Scientific Computing 26(6), 1955–1988 (2005),
[doi:10.1137/030601818](https://doi.org/10.1137/030601818). The authors' primary
[paper](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf), Proposition 4.5, gives

\[
 |\widehat s-s|\le u|s|+\gamma_{n-1}^{\,2}\sum_i |x_i|,
 \qquad \gamma_k=\frac{ku}{1-ku},\quad u=2^{-24},
\]

for the exact sum of the supplied binary32 terms, provided `n*u < 1` and all
intermediates are finite. Its addition analysis includes gradual underflow. This is
not a bound for unrounded products supplied to the sum: product, coefficient,
normalization and transform errors must be included separately.

`_float32_sum_count(n)` enforces `0 <= n < 2^24` before numerical work. A matrix-wide
bucket or strip reduction counts the whole pass, including entries filtered by a
bucket/bin predicate, so the admission does not depend on field values. Column and
node passes use their actual lengths. Boundary tests exercise `2^24-1` and `2^24`
without allocating giant arrays. No hierarchical reduction is currently implied.

The accumulator is an isbits value usable by the shared synchronous launcher. A
caller merging rounded partial sums must propagate their individual errors as well
as the final merge error; it cannot apply the exact-input theorem only to the merge.
Final small host merges can use binary64 with a separately accounted host bound.

## Normalization before products

For a real or complex buffer, let `m` be the maximum magnitude of any real or imaginary
component. When `m > 0`, choose `e` from `frexp(m)` so `m*2^-e` lies in `[1/2,1)`.
An all-zero buffer uses `e=0`. Component maxima avoid overflowing the modulus of a
finite complex value. The normalized components lie in `[-1,1]`; squaring and paired
Hermitian products therefore do not overflow at finite input extremes.

For independent buffers `a` and `b`, evaluate

\[
 q_i=\operatorname{Re}\{\overline{(2^{-e_a}a_i)}(2^{-e_b}b_i)\}
\]

in binary32 and retain `e_a+e_b`. Squared norms use `2*e_a`. Numerical weights receive
their own exponent before multiplication. Independent per-column exponents preserve
weak occupied columns; a total aligns their normalized partial sums to a common
exponent before adding them. Alignment can lose a term below the working absolute
rounding scale and is accounted for in a normwise, rather than elementwise-relative,
error bound.

Scaling an existing normal value by an exactly representable power of two is exact
until the subnormal region is reached. Every inexact binary32 normalization or
product then has absolute underflow error at most `eta/2`, where `eta=2^-149` is the
smallest positive binary32 subnormal. The product analysis includes these additive
terms; it must not assume every normalized component remains normal. Cancellation
uses an absolute-product bound, not a relative bound on a potentially zero result.

`_observation_restore` and `_observation_restore_ratio` combine binary64 mantissas and
integer exponents at the final host result boundary. They do not first form `2^e`,
a potentially overflowing product of physical coefficients, or its reciprocal.
Nonfinite inputs and unrepresentable finite-range results raise named errors before
publication; exact zero remains exact zero. Gradual underflow in the final binary64
physical result is retained.

## Homogeneity and conditioning

Field amplitudes have degree one in the physical field scale. Power, energy,
intensity, fluence, raw spectral occupancy and split removed energy have degree two.
Widths, centroids, normalized occupancies and population fractions have degree zero;
their physical units come from the coordinate/profile/rate inputs. An on-axis map
converts stored state to stored physical field, so state and field exponents are not
interchangeable. Plasma evaluates rates using the physical nodal field exponent.

A width is not uniformly well conditioned. Its discrete interpolation error includes
both ordinate/peak errors and division by the local crossing slope. Qualification
uses fixtures with an explicit nonzero slope or an independent conditioned interval;
an operation count alone does not bound a nearly flat half-maximum crossing. Existing
undefined-width NaNs remain catalogue results. They do not excuse nonfinite fields.

FFT and synthesis error models are hypotheses of each composed test. Neither Sum2s
nor a count of floating-point operations proves the accuracy of generated twiddle
factors. Tests against an independently evaluated DFT or tighter numerical reference
remain required for the actual transform and route under qualification.
