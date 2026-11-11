# Protected wide operator assembly

This leaf implements a **CPU, per-operation certificate**, not public DF32 propagation
or device admission. It binds owned copies of the actual binary64 constant generator,
cumulative Chebyshev coefficients, or expansion terms and cumulative real weights.
`PHASE_ADMISSION.md` provides useful geometry and operand summaries, but those summaries
are not certificates and are not trusted in place of the arrays. The prepared record
keeps the host geometry/error data separate from the DF32 numerical buffers.

The certificate evaluates every cell's scalar graph, carrying outward absolute error
radii and checking intermediate ranges **before each primitive**. It allocates nothing
after compilation. A subsequent write repeats that graph in DF32, without host position
arithmetic or certificate arrays in its kernel captures. This deliberately costs a full
host pass plus a numerical pass. There is no throughput claim, reusable run-wide proof,
or device-to-host copy hidden in advancement. A future device path needs either a reusable
uniform certificate or a separately explicit device-safe strategy.

## 1. Exact targets and scalar geometry

All mathematical targets treat the supplied binary64 arrays and coordinates as exact
real numbers. Units and signs are unchanged: cumulative Phi and exponents are
dimensionless, rates are inverse metres, and all coordinates/steps are metres. For the
actual supplied endpoints `z0,z1` and separately supplied positive step `h`, define

\[
 Q=\frac{\Phi(z_1)-\Phi(z_0)}h,\qquad L=hQ.
\]

Never substitute `h` for `z1-z0`; landing and replay arithmetic can make them different.
The same-interval algorithm below preserves this distinction. For an actual retained
sample coordinate `s`, the transported exponent and compensated rate are

\[
 T=(h-(s-z_0))Q,
 \qquad R=D_s-Q.                                             \tag{1}
\]

The first identity is exactly `(1-(s-z0)/h)*L`, but does not form a potentially enormous
normalized-history multiplier. A dense destination `p` uses `T=(p-s)Q`, the exact
represented-coordinate identity for `theta=(p-z0)/h`. These are different floating-point
evaluation orders from the binary64 reference, not bitwise emulation of its divisions.
Their error is measured against the same defining polynomial/coordinate expressions.

The fixed `WideGeometry` admits the actual stage-coordinate span, retained-history span,
actual maximum step and optional positive minimum. The caller can lower them from
`PhaseSpans`; it must include actual replay lengths even when larger than `dz_max`.
An absent minimum permits arbitrarily small positive forced landings. The preparation
and each operation check these limits; they do not invent a lower bound from `dz_min`.

Interval selection calls the existing exact binary64 `_table_interval`: an interior
knot belongs to the right interval by default, and derivative `before=true` uses the
left interval without perturbing its coordinate. Admitted outer slack extrapolates the
endpoint interval. No coefficient continuity or physical derivative is presumed.

## 2. Correlated divided-difference Clenshaw

In an interval `[a,b]`, write `x(z)=(2z-a-b)/(b-a)`. The host computes the algebraically
equivalent stable map `((z-a)+(z-b))/(b-a)`, retaining every rounding error. Let

\[
 x_0=x(z_0),\quad x_1=x(z_1),\quad
 \beta=\frac{x_1-x_0}{h}=\frac{2((z_1-z_0)/h)}{b-a}.
\]

For the existing two-register Clenshaw convention, the base recurrence at `x0` is

\[
 P'=c_k-C,\qquad C'=P+2x_0C.
\]

Initialize its top two coefficients as usual. Alongside it, initialize divided
differences `U=V=0` and update using the **old** registers:

\[
 U'=-V,\qquad V'=U+2x_1V+2\beta C.                           \tag{2}
\]

Subtracting the recurrence at `x0` from that at `x1` proves (2) by induction. The final
divided difference is

\[
 Q=U+x_1V+\beta C.                                          \tag{3}
\]

Here `C` is the final base current register. Its last update is required even though
the final base `P` is unused. The constant coefficient cancels exactly; it is never
loaded by this graph. A degree-zero polynomial gives exact zero. Degree one uses
`beta*c1` directly. Putting `x1=x0` and `beta=2/(b-a)` yields the differentiated
Clenshaw recurrence and thus the represented rate, including exact knot-side selection.

Crucially the base-register errors in (2) are multiplied by `beta`, not `1/h` times
unrelated endpoint constants. When `z1-z0` scales with `h`, the rate error remains
rate-scaled as `h` shrinks. The exponent multiplies this error by `h`. This addresses
ordinary intra-interval forced landings without assuming a nonzero minimum step.

For different intervals, the leaf evaluates the two actual polynomials separately,
subtracts them and multiplies by a host-prepared reciprocal step. Represented jumps
and cumulative offsets remain in the error bound. Cancellation followed by `1/h` can
exceed the certificate budget, and a huge reciprocal can exceed the intermediate
domain. This is an explicit named refusal, not a switch to binary64 or a discarded
history. Cross-knot uniform certification remains narrower than the function-leaf
domain. A constant operator instead uses `Q=((z1-z0)/h)*D`.

An expansion applies (2)--(3), or the cross-interval graph, to each real cumulative
weight. It then contracts the fixed complex terms in their supplied order, with an
accurate real sum and imaginary sum. All term-conversion and weight-evaluation errors
are propagated through the products and sums. Cancellation in the final contraction
does not erase errors already incurred in its operands.

## 3. Host rounding and conversion

Host coordinates and bundle scalars remain RN-even binary64 as in SEP 33. A host scalar is
stored internally as `(value,error)`, enclosing the exact real expression in a symmetric
absolute interval. For a finite rounded result `r`, use

\[
 |r-\operatorname{exact}|\le
 \frac{u_{64}}{1-u_{64}}|r|+2^{-1074},\quad u_{64}=2^{-53}.    \tag{4}
\]

The additive term is deliberately twice the half-subnormal allowance. Error sums and
products are propagated with triangle inequalities. For division `x/y`, let the stored
values be `a,b`, radii `e,f`, rounded quotient `q` and its local allowance `r_q` from (4).
Require `d=|b|-f>0`; then use

\[
 e_{x/y}\le r_q+\frac{e+(|q|+r_q)f}{d}.                      \tag{5}
\]

The extra `r_q*f` term is retained. The denominator is rounded downward; positive
numerators, sums, products and quotients are rounded outward with `nextfloat`. A product
or quotient that underflows to zero still returns the least positive binary64 bound;
only a mathematically exact zero bound remains zero.

To split a host value `v`, set `hi=RN32(v)`, `r=v-Float64(hi)`, `lo=RN32(r)`, followed
by Fast2Sum normalization. Provided `hi` is finite, both differences
`v-Float64(hi)` and `r-Float64(lo)` are exact by Sterbenz, or because the rounded word
is zero. Therefore the **actual** conversion error is `abs(r-Float64(lo))`, not a
nominal precision assumption. Fast2Sum preserves the pair's exact real sum. The host
expression radius is added outward to this conversion error. This local split permits
subnormal words and lost residuals under the model below; it does not change the
stricter public `prepare_df32` contract. No general DF32 division theorem is assumed:
all divisions in this assembly are explicitly bounded host scalar divisions.

## 4. DF32 primitive propagation and finite intermediates

Assume binary32 round-to-nearest ties-to-even, correctly rounded FMA, gradual underflow,
and no reassociation or contraction except the primitive's explicit FMA. This is the
same conditional model as `DF32_BOUNDS.md`; CPU evidence does not establish a vendor
device's model. Write `u=2^-24`, `eta=2^-150`,

\[
 A=\frac{3u^2}{1-4u},\qquad B=7u^2.
\]

The accurate addition and multiplication graphs satisfy

\[
 |\operatorname{add}(x,y)-(x+y)|\le A|x+y|,
 \quad |\operatorname{mul}(x,y)-xy|\le B|xy|+5\eta.            \tag{6}
\]

These are the finite-underflow extensions derived in `DF32_BOUNDS.md` §§1.1--1.2,
not claims that the unlimited-exponent paper hypotheses hold after a normal leading
product. Addition/subtraction of word lattice values remains exact when subnormal;
the product's lost FMA residual is covered by the additive term. No nonzero residual
floor is assumed. The underlying graphs are Joldes, Muller and Popescu (2017),
Algorithms 6 and 12 ([DOI](https://doi.org/10.1145/3121432)); the multiplication proof
correction is Muller and Rideau (2022), Theorem 2.8
([DOI](https://doi.org/10.1145/3484514)). The finite-underflow extensions and this
whole-expression derivation are additional arguments, not attributed to those papers.

Let each certificate scalar contain an actual normalized DF32 value `x` and radius
`e_x` about its target. For an actual computed result `c`, (6) gives local bounds

\[
 a_c=\frac{A|c|}{1-A},\qquad
 m_c=\frac{B|c|+5\eta}{1-B}.                                \tag{7}
\]

Use an outward upper bound `abs(c.hi)+abs(c.lo)` for `|c|`. Denominators in (7) are
rounded downward. Propagated radii are

\[
 e_{x+y}=e_x+e_y+a_c,
\quad e_{xy}=|x|e_y+|y|e_x+e_xe_y+m_c.                       \tag{8}
\]

Unary negation is exact. Every sum/product in (7)--(8) is outward. Induction over the
actual scalar graph proves the cell's final radius. Exact cancellation in an addition
can make its *local* error zero, while inherited errors remain. Complex operations
are explicit componentwise accurate sums and real scalar products, so no unproved
generic complex arithmetic is used.

Before a primitive executes, the certificate requires its word-magnitude operands and
their sum or product bound at most `F/32`, `F=floatmax(Float32)`. This is a sufficient
intermediate margin, independent of the final exponent. For addition, the direct
TwoSum register bounds are below eight times the operand-magnitude sum; after those
finite operations its exact residual invariant bounds the two later Fast2Sum
corrections by `O(u)` of that sum. For multiplication, normalized low words are at most
`u` times their high words. The high product, low product, two cross FMAs, correction
sum and final Fast2Sum registers are below `8M+64eta`, where `M` is the word-magnitude
product. Thus the `F/32` margin also leaves room for all additive underflow allowances.
The low-product/subnormal cases use the explicit split in `DF32_BOUNDS.md` §1.2 to
retain normalization. A failed margin produces `:intermediate_range` without executing
the offending primitive and without writing an output.

## 5. Certificate and provenance boundary

An exponent or transport certificate requires each stored component to satisfy the
**exact two-word** function domain `|Re|<=16`, `|Im|<=2^20`, and the sum of the maximum
real and imaginary absolute errors to be at most `2^-27`. The check uses `_df32_abs_leq`,
not a rounded binary64 sum that could hide an endpoint's tiny positive low word. This
certifies the input to the reviewed function leaves; their function/truncation/output
errors are additional, as documented in `DF32_BOUNDS.md`. A compensated-rate output
instead retains inverse-metre errors without borrowing this dimensionless budget.

`WideRate` owns actual stored rates and componentwise error provenance. A represented
refresh first certifies all cells, then fills values/errors and advances a monotonic
generation. An operation records that generation; stale reuse refuses before writes.
A restored rate must supply explicit finite nonnegative prior-error radii, which are
copied with the actual values and augmented by conversion error. External provenance
must bound those actual stored binary64 values relative to their intended rates, including
every earlier export or precision conversion. An unadjusted certificate for a pre-export
DF32 value is insufficient. This is an explicit caller precondition, not independently
verified by the restore helper;
compatibility, history keys and sample selection remain shared integration obligations.
No stored rate is silently replaced by a newly sampled derivative.

Owned preparations and sampled-rate arrays must not be mutated through raw references.
Each operation retains the supplied endpoint, step, sample and optional dense-destination
coordinates, plus the combined host-formation/split radius of every scalar in its graph.
Writers check kind, certificate, generation, shape, writable layout and all retained
storage overlap before writing. A failed certification or stale operation preserves
the destination. A failed rate refresh preserves values, errors, position and generation.
Exact zero intervals produce exact structural zero exponents and transports.

The current graph may refuse well-conditioned arguments that a sharper proof would
admit, including sufficiently large cancelled intermediate contributions, cross-knot
offset cancellation divided by a tiny step, or arguments whose assembly error consumes
more than the function leaf's separate input budget. Such refusals are explicit numerical
domain limits. They neither silently select another precision nor enlarge the accepted
SEP's public domain. Public admission, device arithmetic, performance and whole-run
error propagation remain outside this leaf.
