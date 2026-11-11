# Embedded-source quadrature diagnostic

`quadrature_error!` scores the existing discrete ratio
`norm(coarse - fine) / norm(fine)` after identical source prefactors, crop and window
operations, before the common temporal absorber. It does not evaluate a new source,
change the controller, or certify the continuum quadrature error. The caller retains
accepted-step cadence and warning policy. Two unavailable inputs or zero fine source
return NaN; identical nonzero inputs return exact zero.

`QuadratureWorkspace(execution, n_columns; arithmetic=SourceArithmetic(Float64))`
owns only `maxima` with shape `(3,n_columns)` and `sums` with shape `(2,n_columns)`.
Both use source precision: exactly `5*n_columns*sizeof(T)` array bytes. The third
maximum row stores an exact zero/one difference-exponent flag. No field-sized scratch
or extra scale vector is retained. The physical source scale is common to both inputs
and cancels, so neither state nor field exponents enter this ratio.

## Protected binary32 computation

Find independent maximum real/imaginary component magnitudes of the fine source and
its difference. Do not first normalize both sources by the fine maximum: that can
erase every small changed component. A finite-input subtraction that would overflow
is replaced by subtracting its exact large-operand halves. The difference maximum
then denotes half the physical difference; a Boolean exponent carries the factor two.
Align earlier maxima when this flag first appears. All pointwise arithmetic, maxima,
normalized squares and retained partial sums remain binary32. The binary64 path keeps
its original arithmetic order; typed zero, one and half constants have identical values.

Use `_Float32Sum` for each column's nonnegative normalized squared terms and again
for the rounded column partials. This is the existing protected Sum2s implementation:
Ogita, Rump and Oishi, *Accurate Sum and Dot Product* (2005),
[Algorithm 4.1 and Proposition 4.5](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf),
[doi:10.1137/030601818](https://doi.org/10.1137/030601818).
For `n*u < 1`, its error on the supplied floating-point terms is bounded by
`u*abs(sum) + gamma_(n-1)^2*sum(abs(terms))`, including gradual underflow.
Here `u=2^-24` and `gamma_k=k*u/(1-k*u)`. Each column supplies exactly `n_rows`
terms, each merging a real and imaginary square; the final pass supplies exactly
`n_columns` terms. Refuse either count at or above `2^24` before scratch writes.

Only the final two maxima and two sums widen to binary64. Their `frexp` fractions,
square-root sum ratio and combined integer exponent form the result without an
intermediate potentially overflowing scale ratio. Finite binary32 inputs with these
count limits have fewer than `2^49` real components. The exact nonzero norm ratio lies
between `2^-303` and `2^303`; its final host arithmetic is normal and finite. Normalized
components are bounded by two, and even conservative bounds for their accumulated
partials and corrections remain below `2^52`. The tested subtraction may overflow
to select its finite half-difference fallback; all subsequent arithmetic is finite.
There is no reassociation, fast math or flush to zero. Device qualification is separate.

## Matched-input rounding bound

The reference for this leaf is the exact ratio of the supplied represented complex
inputs. Source assembly, narrowed coefficient, transform and model errors belong to
the surrounding composition and are not covered by the following bound.

Let `r=n_rows`, `c=n_columns`, `N=2*r*c`, and `eta=2^-149`. For either norm, let `S`
be its exact squared sum in the independently selected normalized units. Maxima are
selected from represented components or rounded safe differences. Therefore `S>=1/2`
when that norm is nonzero. An ordinary finite difference has relative error at most
`u`; a subnormal difference of two binary32 inputs is exact. Overflowing differences
use exact halves followed by one rounding. Normalization contributes one division
and, on the doubled path, at most one gradual-underflow rounding of a half.

For a normalized component `x`, these operations give absolute error no larger than
`gamma_3*abs(x)+eta`, with `abs(x)<2`. Squaring and adding the two nonnegative components
then gives the conservative aggregate bound `gamma_12*S+16*N*eta`. This includes
normalization, square and pair-addition underflow. Thus set

```text
p = gamma_12 + 32*N*eta
s_r = u + gamma_(r-1)^2
s_c = u + gamma_(c-1)^2
b = (1+p)*(1+s_r)*(1+s_c) - 1.
```

The first factor accounts for pointwise errors, the second for each column's Sum2s
error, and the third for merging its already rounded partials. Positivity lets these
bounds sum across columns without cancellation. Applying the final theorem only to
the merge would omit the first two factors.

When `b<1`, both squared norms have relative error at most `b`. With
`v=2^-53` and `g=8*v/(1-8*v)`, the final diagnostic error is bounded by

```text
abs(computed - reference) <= reference * (sqrt((1+b)/(1-b))*(1+g) - 1).
```

The host allowance covers both divisions, the square root and multiplication;
`frexp` and normal power-of-two restoration are exact. The bound may be uninformative
near the admitted count limit; it is not an accuracy tolerance or a new physical
admission rule. Committed rational/MPFR tests evaluate it for their actual dimensions.
They include finite extremes, subnormal inputs, tiny independent differences, mixed
safe-subtraction branches, and a long positive sum whose small terms ordinary binary32
accumulation loses. The binary64 regression retains its prior independent bound.

## Ownership and integration

Source inputs may overlap each other because both are read-only. They must be disjoint
from both work arrays, including offset foreign storage and gather-index dependencies.
The work arrays must also be mutually disjoint and writable. Validate layout, precision,
counts, aliases and finiteness before any numerical write. A failed quotient does not
modify either source. Every reuse overwrites all scratch subsequently read; unavailable
and zero-norm early returns make no claim about unused scratch values.

Session construction must supply the prepared arithmetic, and the memory planner must
charge `5*n_columns*sizeof(T)`. This leaf does not enable public single precision or
change observation delivery, source readiness, alarm thresholds or noise policy.
