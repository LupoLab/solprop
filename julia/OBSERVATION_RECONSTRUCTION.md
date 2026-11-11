# Precision and physical observation reconstruction

This layer reconstructs the supplied observed field without reading mutable source
scratch or advancing propagation work. Its admitted scalar modal, radial, Cartesian
and line routes use the same prepared `SourceArithmetic` as their source. It does not
open public binary32 admission or qualify GPU execution. The reduction contract and
its error model are in [OBSERVATION_PRECISION.md](OBSERVATION_PRECISION.md).

## State, field and coefficient powers

Let the stored canonical spectrum be `s=a*2^-es`, where `es` is the state exponent.
If the immutable physical synthesis is `B`, preparation stores one rounded matrix
`Bs=round_T(B*2^(es-ef))`. Its output represents electric field divided by `2^ef`.
The same fold applies to each independent coarse/fine on-axis matrix. Radial coarse
synthesis first removes the already translated temporal padding factor; it never
reuses the fine population map as the coarse observation map. Cartesian and line
inverse transforms use their existing normalized raw DFT and one prepared scalar
state-to-field fold. Original binary64 matrices remain immutable and reachable for
provenance, later preparation and the storage ledger.

Constant modal power weights are normalized and converted once at preparation;
their normalized binary32 array and range fact are retained. Varying weights retain
their original binary64 table or expansion. Each varying element is evaluated at
the observed coordinate in binary64, normalized, and rounded once. If its normalized
weight has exponent `ew`, a weighted temporal
pair and its spectral energy restore `2^(2es+ew)` only into their host result.
This does not move represented polynomial evaluation into binary32.

Spatial flux follows the existing translated admittance, including the represented
low words and active-band branch. Evaluate `Y=c*sqrt(q)/omega` in host binary64 for
each coefficient at the actual observation coordinate. Normalize its complex
coefficient by a common exponent `ey`, then form the closure in binary32:

\[
 C_s=\operatorname{round}_{32}(\overline{Y}\,2^{-e_y})s.
\]

After the same spatial synthesis, electric and closure arrays represent physical
values divided by `2^ef` and `2^(ef+ey)`. Their signed Hermitian products restore
`2^(2ef+ey)`. Spectral power coefficients include their separately translated measure
and intensity factor, have their own normalization exponent, and use `2^(2es+ew)`.
There is no full-volume mutable binary64 admittance, closure or weight array. A
varying material retains only the existing one-complex-coefficient-per-frequency
host row. The finite derived exponent `ef+ey` need not itself be materializable as a
binary64 power; the bounded metadata/restoration proof is in the reduction document.

Wide coefficient evaluation uses the existing binary64 operation order and finite
coefficient admission. Conversion error, including subnormal coefficient tails, is
separate from the subsequent represented-array reduction error. This implementation
does not promise uniform relative error for an original tiny coefficient. The
binary64 reconstruction branches retain their existing arithmetic graph.

This coefficient-tail limitation can dominate a physical result. For example, an
unused admittance of `1` sets the global normalization, while an occupied bin has
admittance `1e-50` and stored amplitude `2^100`. The latter coefficient rounds to
zero. With an eight-sample transform and Parseval weight `1/8`, its wide-coefficient
energy is about `2.01e9`, while the represented-coefficient energy is zero. The
initial conversion's absolute `eta32/2` error, where `eta32=2^-149`, propagates to
`(eta32/2)*2^200/8`, about `1.41e14`; it covers that discrepancy but supplies no useful
relative accuracy. The reproducible test checks this separate conversion budget and
the exact represented zero. It does not enlarge the later product/reduction budget,
change physical admission, or claim that such ill-conditioned coefficient/field
combinations are accurately resolved in binary32.

## Populations and monitors

Fresh field/envelope populations use the independent fine temporal grid and map.
The rate helpers receive `field_exponent=ef`, so lookup sees the physical nodal field
without materializing an overflowing binary32 physical array. Fractions and per-node
peak rates remain binary32; species densities, physical measures, profiles and final
packed inventory are host binary64. Passive species retain exact zero columns. The
fine on-axis field route reconstructs both admitted real components; envelope rows
must satisfy the existing exact proportionality admission before scalar collapse.
No current-only coefficient enters population evaluation.

Temporal monitors reconstruct the canonical coarse field and use `es`. Transverse
monitors reconstruct physical nodes using `ef`, but their canonical-column occupancy
uses `es`. Their physical reading, threshold, continuation and decision records stay
binary64. No implementation normalization exponent enters portable continuation.

For a strip or onset stencil, determine the normalization exponent from each actual
nonzero component product of field, sample scale and (for a strip) window. Only then
form the binary32 sample buffer. This avoids losing a weak sample before an amplifying
weight. Each strip chunk uses the existing twice-padded raw forward DFT. All inactive
traces/padding are overwritten, and the complete derived amplitude exponent is passed
to the protected strip/current reduction. Small host chunk merges include prior
reduction error and their own binary64 rounding. Input noise remains binary64-only;
all existing binary64 noise/floor paths remain intact.

## Storage ownership

Let `r=sizeof(T)`, `c=2r`, `i=sizeof(Int)`, `F` be coarse frequencies, `N` coarse time,
`Q` fine frequencies, `U` fine time, `M` state columns, `X` physical nodes, `K` the
active node capacity, `S` complete species, `W` windows and `A` axis components.
These are retained numerical array data, not RSS or a bound on opaque FFT plans.

- Common statistics scratch is `(N*M+N+2M+3X+R)*r`, where `R` is the prepared radial
  average capacity. Binary32 adds `(M+X)*i` column/node exponents. Packed values and
  scalar host decisions remain binary64.
- Modal coarse spectrum and power weight use `F*M*(c+r)`. An analytic pair has
  `4NMc` bytes; window pairs have `4NMWc`; on-axis analytic input/output use `2NAc`.
- Modal/radial fine populations retain `(Q*c+U*s)*M` modal storage and `U*s*K` nodal
  samples, where `s=r` for a field and `s=c` for an envelope. Population recurrence
  uses `(5r+1)*U*K`; fractions/peaks use `2X*S*r`, and densities use `8S`.
- Fine on-axis populations retain `(Q*c+U*s)*M`, `U*(A+1)*s`, one-node recurrence,
  `2S*r` fractions/peaks and `8S` densities. Diagnostic work adds `6K*r+3K*i` and
  six binary64 aggregate maxima. Its dimensionless exposure ratios are unchanged.
- Spatial coarse spectrum uses `F*M*c`, optional normalized weights `F*M*r`, and
  optional closure `F*M*c`. Physical electric/closure pairs use `2F*X*c`, plus
  `F*X*c` Cartesian band storage when needed. Their analytic pair adds `4N*X*c`.
  Axis pairs use `2F*c` and, when selected, `4N*c` analytic work.
- Cartesian/line fine populations retain a full `F*X*c` band plus chunk-dependent
  `(Q+F)*K*c+U*K*s` and the same recurrence/species/diagnostic storage above.
- A temporal monitor retains `F*M*c+N*M*s+N*r`, plus two binary64 bucket reading
  vectors. A transverse monitor retains its `F*X*c` band, `X*r` node scratch and
  the existing small binary64 column/readout vectors. Each resolved line retains
  `c*(2*n_strip+3*n_onset)*min(trace_chunk,n_traces)` bytes, with `n_onset` zero or one.

Original and narrowed binary32 map/axis coefficient arrays are both owned; the
binary64 path aliases its original and executable copies. Immutable geometry,
profiles and wide coefficient vectors are also retained and discoverable through
preparation. `SourceRange`, arithmetic and line normalization facts are fixed scalar
records. Reshape views keep their original owner, so they add no duplicate array data.
All these owners remain discoverable to the shared ledger; no global mutable workspace
or borrowed observed field is retained.

## Qualification model

The tests use independent high-precision DFT sums, analytic quadratic ionisation and
explicit physical reference expressions. Transform comparisons assume the separately
qualified finite FFT/twiddle error model; an operation count alone cannot prove the
library's twiddle generation. The stated budgets combine field/coefficient rounding,
absolute pre-cancellation envelopes, products, protected reductions and final host
restoration. Width fixtures propagate ordinate and centroid errors through an
explicitly bounded crossing slope; dimensionless results are checked at non-unit
state and field scales. These are bounded numerical witnesses, not a claim of uniform
accuracy at arbitrary ill-conditioned zero flux or flat half-height crossings.

Bounds-enabled tests also exercise reused/poisoned scratch, passive species, periodic
selection, noise refusal, exponent and arithmetic mismatch, retained input ownership,
complete inactive writes, both CPU task widths and zero warmed allocations. Package
Aqua/JET and concrete reconstructed calls are required before integration. Session
construction, method services, Python transport and public admission remain separate
integration gates.
