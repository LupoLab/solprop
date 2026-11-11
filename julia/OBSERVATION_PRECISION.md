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

Weighted products use the actual participating factors to choose a scale. Independent
whole-buffer maxima are insufficient: a tiny sample paired with a large coefficient
can dominate the physical result even when that sample disappears after an unrelated
normalization. For a term with `k` nonzero real factors, write each represented input
exactly as `x_j = m_j * 2^e_j` using `frexp`, with `1/2 <= |m_j| < 1`. Round a wide
mantissa once to binary32, multiply the binary32 mantissas, and align the product to

\[
 E=\max_i \sum_{j=1}^{k_i} e_{ij},
\]

where the maximum includes only nonzero products in this reduction. Zero products
cannot select the scale. The implementation admits at most eight factors per scalar
product. All mantissa intermediates are therefore normal, even when the original
factors are subnormal or their unscaled product would overflow. Only the final
power-of-two alignment can underflow. No binary64 sample product is formed.

For binary64 inputs, at most `k` mantissa conversions and `k-1` products contribute
rounding, so before alignment

\[
 |\widehat p-p| \le \gamma_{2k-1}|p|.
\]

A binary32 input's mantissa conversion is exact and reduces this count. With gradual
underflow, alignment adds at most `eta/2`, where `eta=2^-149`, in the chosen normalized
units. Each Hermitian pair adds its real and imaginary products once; include that
addition's error and both absolute products before applying the Sum2s bound. The
largest nonzero normalized product has magnitude at least `2^-8` before rounding,
so this absolute-underflow allowance cannot hide loss of a dominant positive term.
Signed cancellation still requires an absolute-product bound; no uniform relative
accuracy near a zero signed sum is claimed.

Spectral numerator and denominator, population mixtures and inventories, temporal
pairs, and each final monitor region use their own actual-product exponents. Monitor
region, split and transverse weighted sums start from original samples or spectra;
they do not amplify a previously normalized scratch value. A per-column or per-node
partial carries its own exponent until all amplifying factors have been applied.
Zero partials do not choose a later merge scale. Merges include both the partial's
prior error and the protected merge error.

Spatial fluence retains one exponent per node and a separate common scale for the
absolute-magnitude error diagnostic. Moment reductions combine each node exponent
with its actual quadrature/coordinate products. Only the final unweighted profile
used for a half-height crossing or angular interpolation aligns nodes to a common
peak scale. At that point the normalized peak lies in `[1/2,1]`; alignment loses at
most `eta/2` per ordinate, and interpolation uses nonnegative weights summing to one.
There is no later large physical weight to amplify that loss. Its width effect must
still be divided by the bracket's crossing slope, as described below. This treatment
does not add a normal-range refusal for a weak spatial node.

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

## Numerical families and owned scratch

The standalone statistics and monitor leaves now use these rules for binary32.
This does not open public session admission or qualify device execution.

- `StatisticsWorkspace` retains binary32 per-time power, total power, per-column
  energy/moment, per-node total/fluence/magnitude and radial-average buffers. One
  Int exponent per column and per node, plus small scalar family exponents, accompany
  them. The node metadata adds exactly `sizeof(Int)*n_nodes` retained array bytes.
- Spectral column reductions normalize actual weighted products; centroid numerator
  and denominator have independent exponents. Per-bin spectral flux gives each bin
  its own exponent. Temporal pairs preserve signed Hermitian products. Species
  inventories scale actual node-weight/fraction products before multiplication.
  Density, potential and physical transverse measure enter only at the final host
  result boundary. Default identity admission is conservative when a weight exponent
  is nonzero; explicit identity metadata must come from the original wide values.
- Transverse node energies, temporal power and strip/current products remain
  binary32. Their final reading arrays, occupancy vectors and control scalars stay
  binary64, with the same physical thresholds and continuation representation.
- `PlasmaDiagnosticWorkspace` owns six binary32 parts per active-node capacity and
  three Int exponents per node. Its six global maxima are host binary64. Raw
  exposure is normalized by its column envelope before differencing. If the exact
  normalized increment is `d`, endpoint alignment and the subtraction give an
  absolute error at most `u*abs(d) + (3/2+u)*eta`; taking each dip/rise maximum does
  not amplify that absolute error. A tiny dip beside a large envelope may disappear
  within this bound; per-extremum relative preservation is not claimed. The resulting
  ratio also retains its denominator conditioning. Paid-work defect and fraction/field
  integrands have separate actual-product scales. Inactive rates and zero fractions
  cannot determine them.
  The freed-fraction denominator remains unscaled and cannot disappear when the paid
  work is large. All chunks and channels share one field scale, which
  cancels from the boundary-level ratio. The paid-work sum uses `n_time-1` terms.
- `NormalizedObservationWeight` retains the immutable original binary64 direct
  table/expansion for varying weights, or an independently owned normalized binary32
  constant. `ObservationWeightWorkspace` retains only one binary64 coefficient per
  expansion term and scalar range/identity facts. Each varying element is evaluated
  in binary64, normalized and rounded once; correlated cancellation is not moved
  into binary32 Clenshaw arithmetic. Exact identity classification precedes narrowing.

## Base scales and derived normalization metadata

The base state and nodal-field powers remain in `[-1074,1023]`, exactly as in
`SourceArithmetic`; this admission is unchanged. A reconstructed closure or transverse
strip/stencil may instead carry a *derived* integer exponent. For example, a base
field exponent `100` and a normalized admittance exponent `1000` give closure exponent
`1100`. The latter is never materialized as `2^1100`. Two stored components `2^-100`
then give finite physical paired flux `2^1000` when the other field exponent is `100`.
Refusing the derived exponent as though it were a base scale would reject this finite
expression before its factorized restoration.

Derived closure/strip/stencil exponents have magnitude at most `4*1074`. The four
blocks cover one base field scale, up to two finite binary64 preparation factors,
and the component normalization of the stored binary32 field. This is a bound on
integer metadata, not a new range promise for a materialized field or coefficient.
Preparation must account for each constituent explicitly. Each finite nonzero
binary64 factor has a `frexp` exponent with magnitude at most `1074`; binary32 has
the smaller bound `149`. All-zero products keep the existing sentinel and never
add it to another exponent.

The integer audit uses `B=1074` and the actual fixed-depth paths below. The generic
product helper admits at most eight factors; its exponent sum therefore has magnitude
at most `8B`. A finite reduced value contributes at most another `B` when decomposed
at the final host boundary. Counting that value directly also covers any increase
from a protected sum; no unbounded running exponent accumulator exists.

- Spatial paired results combine at most `B+4B` physical scale, `8B` product scale,
  then `B+B` for the reduced value and final factor: at most `15B`.
- Strip energy combines at most `2B` sample-product scale and `2*(4B)` amplitude
  metadata, then `B+2B+B` for the value, two factors and one divisor: at most `14B`.
- Rectified onset current combines at most `2B` sample-product scale and `2*(4B)`
  amplitude metadata, then `B+B+2B` for the value, one factor and two divisors:
  at most `14B`.
- Spatial width intermediates combine the three-factor node partial with a finite
  partial and at most three coordinate/quadrature factors: at most `7B`. Differences
  of these exponents give at most `14B`; final restoration remains below `16B`.

The other spectral, population and temporal paths retain their original base-scale
admission. Their actual-product sums and fixed-depth partial merges remain below the
conservative common bound `32B=34368`, as do the rows above. Multiplication by two,
addition, subtraction, negation and the radial exponent split therefore cannot wrap
even a signed 32-bit `Int`. The widened checks run before any use of supplied metadata;
`typemin(Int)`, `typemax(Int)` and values just outside the derived range are refused
before output mutation. This reasoning applies to these bounded call graphs, not to
arbitrary repeated calls of the internal exponent helpers with unchecked integers.

Final physical underflow is gradual. `_observation_restore` and its ratio counterpart
combine mantissas before one final `ldexp`; nonfinite or overflowing physical results
still raise before publication. Tests cover finite closure exponent `1100`, finite
strip/current restoration with the same exponent, unchanged base-scale refusal,
integer-boundary rejection, final physical overflow and zero warmed allocations.

The binary64 branches retain their existing arithmetic order and array element
types. The new optional metadata is absent on those paths. Composition into modal,
spatial and transverse observation workspaces is a separate qualification milestone.

## Wide-coefficient conversion is a separate error source

The reduction contract starts from the supplied represented binary32 numerical
arrays. `source_narrow` retains the oracle's direct-fold admission. SEP 33 section 4.3
bounds the block's well-scaled entries (within `2^-126` of its largest entry); it does
not promise uniform relative accuracy for every subnormal tail. A positive normalized
power weight may remain subnormal after its one wide-to-binary32 rounding. Its error
is an additive `eta/2` in normalized weight units, and must be multiplied by the
actual squared field or other participating factors in the composed error budget.

For example, normalizing wide weights `(1, 3*eta/4)` rounds the second to `eta`.
A field supported only on that second entry can have a small rounding error relative
to the represented-weight reduction and a substantial relative difference from the
original wide-weight expression. The committed witness asserts these two bounds
separately. Product-aware reduction prevents additional avoidable loss; it cannot
recover information already discarded by the required coefficient conversion.
Existing preparation admission is preserved, and no uniform relative guarantee for
such original tiny weights is inferred.
