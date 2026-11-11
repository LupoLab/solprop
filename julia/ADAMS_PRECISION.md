# Adams precision implementation boundary

This numerical leaf implements binary32 state with binary64-wide factors. It is not
public precision, session, physical-scale, device or DF32 admission. Sessions, bridge
records, portable history and public selection remain separate. The existing
binary64 numerical graph is preserved.

## Storage and scalar contract

`AdamsHistory`, `AdamsWork` and `AdamsDense` accept `AbstractMatrix{Complex{T}}`,
with `T` equal to `Float32` or `Float64`. State, source, predictor, candidate, defects
and retained sampled states use `Complex{T}`. Exponents, moments and sampled rates
always use `ComplexF64`. Geometry, polynomial coefficients and controller inputs
remain host binary64. Binary32 seeding, trials and dense reads convert their scalar
coordinates and steps to binary64 before validation, coefficient generation or
callbacks. The exponent callback receives the actual rounded pair `(z, z+h)`. The
numerical step remains `h`. These inputs are not identified after endpoint rounding.

Copies refuse different state precisions before mutation. The binary32 norm adds
one Float32 mantissa and one integer exponent per row, and four Float32 reduction
entries per column. Binary64 keeps its existing two Float64 entries per column and
has no denominator scratch. All arrays belong to the existing recursive ledger.

Binary32 does not keep unscaled transported sources: `carried` has zero planes.
`endpoint_compensation` retains its existing shape but is zero and unused on that
path. The source/state/rate history remains the actual sampled history. No generator
is narrowed or replaced by a derivative at another position.

## Complete factors and protected defects

Let `x` be the represented cumulative exponent over the actual endpoint pair,
`h` the supplied step and `c[j,k]` the prepared binary64 polynomial coefficients.
For each source sample, the binary32 graph first forms the complete wide factor

```math
w_j = h\sum_{k=0}^{q} c_{jk} k!\varphi_{k+1}(x).
```

It rounds `w_j` once to complex binary32, multiplies the state-precision source,
and accumulates in complex binary32. The homogeneous factor is likewise evaluated
wide and rounded before multiplying state. There is no implicit binary64 state
accumulator. Binary64 retains the original monomial-first graph; these are different
floating-point evaluation orders of the same represented polynomial integral.

A defect keeps the divided difference factored. For weights `a_j`, choose integer
`e` from `frexp(max(abs(a_j)))`. Form the protected state-precision difference
`sum(Float32(2^-e*a_j)*F_j)` in the original sample order. Multiply it by the once
narrowed complete factor `h*2^e*sum(b_k*k!*phi_(k+1)(x))`. Do not distribute that
factor across the subtractive difference. Weight underflow, when present, belongs
to the additive absolute rounding budget; it is not treated as exact arithmetic.

For compensation, the mathematical identity is

```math
h b_j e^{s_j x}(D_j-x/h)Y_j
 = [b_j e^{s_j x}(hD_j-x)]Y_j.
```

`D_j` and `Y_j` are the actual stored rate and sampled state. Prediction and
correction use `s_j=1-theta_j`; an endpoint sample has `s_j=0`. The graph keeps the
wide multiplication `h*D_j` and subtraction of `x` separate, combines the complete
coefficient including `b_j` and transport, then narrows before its state product.
It recomputes that coefficient for predictor, corrector and defect terms. This
avoids an artificial range limit on an unscaled carried source.

The ordinary binary64 product/subtraction can round to zero when the exact real
values of its supplied inputs have a nonzero residual. That result is a rounded
arithmetic zero, not a proof of an exact-real zero. In particular, it cannot waive
transport-domain validation. Exact zero weight, or both zero rate and zero phase,
are structural zero factors and need no transport. No FMA residual refinement or
DF32 claim is implicit in this graph.

## Exponent-separated binary64 arithmetic

Internal `_AdamsWideFactor` records a complex binary64 mantissa and an integer
power of two. Addition aligns exponents, then normalizes; multiplication operates
on normalized mantissas and adds exponents. It does not restore an intermediate
value to the binary64 exponent range. Consequently finite scalar coefficients such
as `h`, rates and polynomial weights may rescue underflowing or overflowing
intermediate products before final narrowing. The mantissa is ordinary binary64,
not an exact expansion. Alignment can discard tiny components, and cancellation
still carries an absolute error based on the pre-cancellation contributions.

For transport, write `a=Re(s*x)` and choose
`n=round(a/ln2_hi)` (round to nearest, ties to even). For `|a|<=2^20`, compute

```math
r = \operatorname{fl}(\operatorname{fma}(-n,\ell_h,a)
                     -\operatorname{fl}(n\ell_l)),\qquad
 e^{s x}=2^n e^{r+i\,\operatorname{Im}(sx)}.
```

The constants are derived by rounding `log(BigFloat(2))` at 512-bit precision:

- `ln2_hi = 0x1.62e42fefa39efp-1 = RN64(log(2))`;
- `ln2_lo = 0x1.abc9e3b39803fp-56 = RN64(log(2)-ln2_hi)`;
- the remaining absolute residual is at most `2^-109`.

These are a derivation from the defining logarithm, not copied empirical tuning
constants. Tests independently regenerate both words and bound the residual. The
reduction is the exact identity `exp(a)=2^n*exp(a-n*log(2))`; it does not require
`n` to be the exact nearest quotient. Here `|n|<2^21`, exactly representable as
binary64, and the residual remains below one in magnitude. The checked interval
makes the integer conversion safe before `unsafe_trunc` is used.

Writing `u=2^-53`, `eta=2^-1074`, the reduction error is bounded componentwise by

```math
|\widehat r+n\log 2-a|
 \le u|a-n\ell_h|+u|n\ell_l|+
       u(|a-n\ell_h|+|n\ell_l|)(1+u)
       +|n|2^{-109}+3\eta/2.
```

The first term is the single rounded FMA result, the second the low-word product,
and the third the final subtraction. Exponential sensitivity multiplies this
absolute argument error by `exp(a)*expm1(error)/error`. Formation of `s*x` has its
own ordinary component multiplication error. Accuracy of the binary64 complex
exponential and phi implementation remains an explicit function-error hypothesis,
qualified by their independent witnesses; operation counts alone do not prove libm.

Only the compensation graph uses this reduction. For `a < -2^20`, its remaining
finite factors have magnitude less than `2^3100`: `|hD-x|<2^2050`; a
prepared weight is less than `2^1024`, and a dense integral sums at most seven
terms with `0<xi<1`, hence is less than `2^1028` including rounding. The complete result is
far below half the smallest binary32 subnormal; zero is a valid final narrowing.
For `a > 2^20`, every nonzero computed normalized difference and nonzero finite
weight has product magnitude greater than `2^-32768`. A deliberately loose
computed-arithmetic bound suffices: each nonzero binary64 coefficient is at least
`2^-1074`, an inverse integer is at least `1/7`, and there are at most seven
multiplications by `xi>=2^-1074`. Allowing another 1074 exponent bits at each of
seven normalized Horner additions gives a dense-weight lower bound greater than
`2^-16400`. (An underflowed smaller aligned addend cannot cancel a nonzero dominant
one.) The product `hD` and subtraction of `x`, with another full 1074-bit allowance
for normalized complex cancellation, give a nonzero difference greater than
`2^-3300`. One further 1074-bit allowance for the final complex product still
leaves ample room above `2^-32768`. Thus the complete factor
cannot fit binary32. This side refuses before considering a cancellation-rounded
zero difference. Infinite positive transport, or a nonfinite imaginary transport,
also refuses. A negative infinite real transport selects the same underflow tail.
Phi factors do not occur in this transport-tail branch; their separate moment
sums stay exponent-separated until narrowing. These tail bounds use this finite
graph only, not arbitrary products or cancellation between different sample
contributions. Exact structural zero factors are exempt.

## Error accounting and domain limits

For either precision, write `gamma_n=n*u/(1-n*u)`. For binary32 let
`u32=2^-24`, `eta32=2^-149`. Each rounded real operation has absolute error at most
`u*abs(exact_result)+eta/2`, provided its computed result is finite. For complex
multiplication, sum the two absolute real products before cancellation in each
component. Powers of two are exact except for underflow; after alignment or
normalization the additive error is multiplied by the stored exponent. Propagate
these local bounds through the graph using absolute products and sums.

In particular a compensation bound must retain `|hD|+|x|`, multiplied by the
absolute transport and weight, even when `hD-x` nearly cancels. Source factors
retain the sum of absolute polynomial-moment contributions. The defect bound
retains `sum|a_j F_j| * sum|b_k M_k|`, not the magnitude of the small resulting
defect. Errors in host coefficient generation, nodes, endpoint coordinates and
function values are separate inputs to that propagation. The existing coefficient
and phi tests qualify those inputs for the matched fixtures; they are not assumed
exact in a claim about an ideal continuous method.

The committed matched-integral fixtures use at most seven samples, small finite
phases, independently evaluated 256-bit defining-series moments, and the actual
represented host coefficients. Their predeclared `gamma_(32*(q+1))` state budget
(`gamma_(64*(q+1))` for normalized defects) plus `gamma_8192` wide budget is applied
to those absolute contributions. The fixed eight-step witness propagates its
per-step allowance through the absolute two-register PEC matrix. These are
matched-history rounding tests, not global adaptive-error promises. Ill-conditioned
compensation may have a large absolute bound despite a small answer.

A complete factor that cannot narrow finitely, a nonfinite callback result, or a
nonfinite state product/sum causes a named refusal before `work.ready=true`.
Scratch may contain an incomplete trial; accepted state and history are unchanged.
The interface does not promise closure merely because a mathematical final sum is
finite: a required state-precision product can overflow before later cancellation.
Coefficients still obey the existing finite binary64 preparation contract. There
is no new normal-range restriction on individual polynomial coefficients or steps.

## Dense output, norms and integration obligations

Dense output combines `h*k!*xi^(k+1)*phi_(k+1)(xi*x)` before narrowing. Powers of
`xi` also use exponent-separated wide arithmetic. The C32 ordinary compensation
integrals use a normalized binary64 Horner recurrence too; they are never restored
to an underflowing scalar before multiplication by the transported rate difference.
Its unused scalar `weights` scratch is cleared, without evaluating that unscaled
integral a second time.
The binary64 path retains its original ordinary-integral values. Exact endpoints
copy committed
state, including signed zeros. The existing strong-damping rule for compensated
bins with `Re(x)<-1` uses homogeneous flow before reading irrelevant source/rate
arrays. Dense storage and copying preserve wide rate precision.

Norms use only the old state's masked magnitude, while checking all old and
candidate entries for finiteness, including excluded rows. The binary32 reduction
uses max-normalized components and Sum2s from Ogita, Rump and Oishi (2005),
Algorithm 4.1 and Proposition 4.5, DOI
[10.1137/030601818](https://doi.org/10.1137/030601818), through the existing reviewed
`march_precision.jl` helpers. The denominator uses a Float32 mantissa and integer
exponent; the small final merge is host binary64. Inactive score outputs are zero.

The master must attach physical scaling, portable history, session metadata,
controller policy and public admission separately. No input-noise, all-binary32
wide arithmetic, GPU or DF32 path is enabled here. A future DF32 attachment must
use actual `WideOperation` certificates for every transported and compensated
expression, including restored sampled rates and replay lengths; narrowing an
already assembled binary64 phase is not that implementation.
