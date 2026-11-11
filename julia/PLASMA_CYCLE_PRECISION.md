# Reduced logarithmic plasma coordinates

This note records the binary32 numerical correction for [issue
#738](https://github.com/LupoLab/solprop/issues/738). It supplements
[SOURCE_PRECISION.md](SOURCE_PRECISION.md), whose physical scaling, clipping,
population and current contracts remain in force. Binary64 keeps its original
arithmetic graph. Neither the physical spline coefficients nor the cycle rules,
`moment_rtol`, negligible-rate floor or failure policy change.

## Target and cancellation

Let `a` be the positive stored binary32 field magnitude and `ef` its exact physical
power, so `E = a 2^ef`. A supplied spline interval `j` uses

\[
 r_j=\log(a)+e_f\log2-L_0-jh,\qquad
 P_j(r)=c_{3j}r^3+c_{2j}r^2+c_{1j}r+c_{0j}.
\]

Here `L0` and `h > 0` are the original binary64 translated metadata, and the
coefficients are already narrowed binary32 inputs. The represented rate is
`exp(Pj(rj))`, subject to the unchanged physical cutoff and ceiling. Conversion
error of the coefficients is a separate input to a physical-model comparison.

Computing `log(a)` and the shifted origin as single words before subtracting them
loses a small local offset through cancellation. In the reported argon case the
absolute logarithm is about 13; one binary32 logarithm ulp is then much larger
than the local-coordinate error allowed by the steep spline. Positive panel sums
can therefore fail the unchanged `2e-6` refinement test even though the same
represented spline and rules evaluated in binary64 converge. Increasing the
quadrature order does not repair that local arithmetic error.

## Arithmetic graph and retained storage

Preparation chooses integer `b = exponent(field_min)` by `frexp` of the finite
binary64 lower clipping field. Set `l = RN64(log(2))` and prepare normalized
two-word splits of `l`, `h`, and `fma(-b,l,L0)`. The private scalar record retains
these three `DF32` pairs and integer `ef-b`. Existing array records and public
constructors are unchanged. There is no additional retained array or mutable
binary64 field buffer. The existing six rate parameters remain available; the
binary32 path appends this scalar record.

For each live field, exact `frexp` and, when necessary, one exact multiplication
by two give `a=m 2^e`, with `m` near `[sqrt(1/2),sqrt(2)]`. The implemented switch
is the represented binary32 `sqrt(1/2)`, so those endpoints have its rounding
allowance. Both `m` and one lie within a factor of two: `m-1` is exact by Sterbenz.
The kernel calls the existing binary32 `log1p(m-1)` and evaluates

\[
 C=\operatorname{DWAdd}(\log1p(m-1),
          \operatorname{DWMul}(e+e_f-b,l))-\operatorname{DW}(L_0-bl),
 \qquad R_j=C-\operatorname{DWMul}(j,h).
\]

All products/additions here are the reviewed explicit-FMA algorithms in
`double_float.jl`; this change implements no logarithm approximation. The exponent
sum is a small exactly representable integer under the existing finite field-scale
admission. Knot integers are split rather than silently rounded once. The kernel
then uses `Rj.hi` in the original three-FMA binary32 cubic and binary32 `exp`.

Interval selection keeps both words of `C/h`. A power-of-two scaling first puts
the denominator high word in `[8,16)` and renormalizes both pairs. This uses the
restricted reciprocal proof already derived in [DF32_BOUNDS.md §1.2](DF32_BOUNDS.md),
instead of extending the unlimited-exponent division theorem to an arbitrary tiny
spacing. The integer floor uses the high-word integral part and the remaining
fraction plus low word separately; rounding a two-word quotient to one word before
flooring would lose the side of an integer boundary. The selected interval is
clamped to the existing endpoint intervals.

## Hypotheses and coordinate allowance

The arithmetic model is round-to-nearest ties-to-even binary32, correctly rounded
basic operations/FMA, gradual underflow and finite intermediates. Put
`u=2^-24`, `eta=2^-150`, and `A=3u²/(1-4u)`. Metal's distinct subnormal/rounding
rules are not qualified by these CPU witnesses. The host preparation checks
finite splits both before and after renormalization; a last-word tie at overflow
must not escape as an infinite pair.

For a finite binary64 scalar `v` split by high word and rounded residual, the
absolute representation error is at most `u² abs(v)+eta`, including a tiny
residual that disappears. This source-local preparation deliberately permits that
gradual underflow; it does not weaken the separate `prepare_df32` API. The host
constant and reduced-origin errors are separate. For example, with exact `ln2`,

\[
 E_l\le |\widehat l-\ln2|+u^2|\widehat l|+\eta,
 \quad
 E_o\le |b|\,|\widehat l-\ln2|
       +u_{64}|L_0-b\widehat l|+\eta_{64}
       +u^2|\operatorname{fma}(-b,\widehat l,L_0)|+\eta.
\]

No independence of those correlated errors is assumed. An exact MPFR comparison
of these prepared inputs may replace the conservative right sides. It is input
representation error, not a fitted rate tolerance.

For the local elementary function the finite CPU qualification checks

\[
 |\widehat{\log1p(m-1)}-\ln m|\le2u|\ln m|+\eta.
\]

This is an explicit elementary-function hypothesis, tested against MPFR at the
qualified inputs; disabling fast math is not a proof that every vendor meets it.
The [pinned Julia implementation][julia-log] accepts and returns Float32 but its
CPU `log1p` helpers use binary64 internally; the existing population `expm1`
primitive does too. [Issue #751](https://github.com/LupoLab/solprop/issues/751)
records direct-callee LLVM evidence. The accepted source-precision boundary is the
Float32 elementary API and its separately qualified returned-value error, together
with binary32 response arrays and owned arithmetic. This fix adds no explicit
binary64 numerical fallback or binary64 response buffer. Its caller LLVM check
stops at those elementary APIs and must not be described as transitive evidence
that every CPU instruction is binary32. The strict all-binary32 graph still holds
for the protected two-word arithmetic helpers. The improvement comes from local
reduction and retained low words, not a new logarithm approximation.

A vendor specialization must compile without a host/Float64 fallback and satisfy
its own elementary-function and finite-range hypotheses before admission. The
CPU library implementation proves neither a device's arithmetic width nor its
accuracy. This explicit primitive boundary changes no physics, default or budget.

The following recurrences specify the assembly bound completely. If `xhat` and
`yhat` are exact sums of the computed pairs with errors `Ex`, `Ey` relative to
their targets, use

\[
 \begin{aligned}
 E_{+}&=E_x+E_y+A|\widehat x+\widehat y|,\\
 E_{\times}&=|\widehat x|E_y+|\widehat y|E_x+E_xE_y
                  +7u^2|\widehat x\widehat y|+5\eta.
 \end{aligned}
\]

Apply these in the displayed graph order, starting with the metadata and local-log
errors above. The integer index split contributes its exact representation error
if its value is too large for two words. The final high-word selection adds
`u abs(Rj)+eta`. The addition and product bounds, including the additive underflow
term, are the reviewed derivations in DF32_BOUNDS §1; normal input words alone do
not establish the intermediate-range hypotheses.

Interval selection has its own conditioning. Exact binary scaling can lose at
most `2eta` per pair, before exact renormalization. For scaled pair values `x,y`,
the restricted division adds

\[
 16u^2|x/y|+(6|x|(1+1/|y|)+6)\eta.
\]

Input errors add `(Ex+abs(x/y)Ey)/(abs(y)-Ey)` when `Ey<abs(y)`.
An exact interval decision follows only when this enclosure, also including the
cheap endpoint quotient test, stays on one side of the relevant integer. Near a
knot, use the coordinate/rate bound for every possible selected polynomial and
include any represented jump between them. A narrowed spline is not assumed
perfectly continuous. The committed pointwise bounds explicitly use the selected
polynomial, and also check the selected interval away from knots. These statements
do not promise accurate evaluation for every arbitrarily ill-conditioned table.

## From coordinate error to rate and panel error

Suppose the selected offset has bound `Er`, and let `R` bound both its exact and
computed magnitudes. For coefficients `(c3,c2,c1,c0)`, define

\[
 D=3|c_3|R^2+2|c_2|R+|c_1|,\quad
 S=|c_3|R^3+|c_2|R^2+|c_1|R+|c_0|.
\]

The mean-value theorem and three FMA roundings give

\[
 E_P\le D E_r+\gamma_3 S+(1+u)^2\eta(1+R+R^2).
\]

Under the independently checked finite-fixture `exp` allowance `2u exp(P)+eta`,
the absolute rate allowance is

\[
 E_W\le \exp(P)\operatorname{expm1}(E_P)
              +2u\exp(P+E_P)+\eta.
\]

The ceiling is 1-Lipschitz; lower/upper clipping comparisons retain their existing
rounded-up physical thresholds. Neither the estimate above nor the cycle
refinement criterion is a general relative-error promise at a discontinuity.

Panel terms are positive. For fixed represented scalar geometry, propagate `EW`
through the weight product, and through `(sin/cos)^2` for the directional moment.
One division used twice and two products contribute `gamma_4` to that directional
factor. Additive underflow is retained separately. Then apply the unchanged Sum2
bound `u abs(sum)+gamma_(n-1)^2 sumabs`, followed by the final width multiplication.
This is [Ogita, Rump and Oishi (2005), Proposition 4.5,
equation (4.8)](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf), under its finite
range and `n u < 1` hypotheses. The full primary text was checked for the existing
accumulator derivation; no accumulator algorithm is changed here.

The MPFR panel witnesses deliberately fix the existing represented nodes,
weights, angles and sine/cosine inputs, then independently evaluate the target
spline, products and sum. Their bounds therefore qualify this lookup/accumulation
change. They do not turn a finite quadrature difference into a proof of continuous
integration accuracy, or requalify every vendor `sincos` or node generator.

## Regression and limits

The self-contained steep fixture is the clipped power law
`W(E)=(E/2^32)^32/8`, represented on 128 logarithmic intervals of width `1/32`
starting at `E=2^27`. Each interval's linear log-polynomial is prepared from that
formula and narrowed once. The supplied rules remain 64/128/256/512 and the
budget remains `2e-6`. The 513 adjacent binary32 magnitudes around `466245.6f0`
include failures of the old absolute-logarithm graph. The new lookup must complete
every case; no accepted failure is hidden by a looser comparison. Separate MPFR
checks cover the local coordinate, selected spline and all four panel orders,
with strictly positive operation-derived bounds.

The reported physical argon table is retained in the issue's end-to-end
qualification case. Its exact failing major, original rules and 513-neighbour
control are also checked during the fix's qualification. Field-scale covariance
uses exponents `-40,0,13,40,80`; powers change representation only. Caller inputs,
inactive columns, pre-write refusal, binary64 graph, zero warmed allocations and
the owned binary32 arithmetic boundary remain tested. Full short propagation, all physical
tables and GPU elementary functions remain separate qualification gates.

The two-word primitive sources are [Joldes, Muller and Popescu
(2017)](https://doi.org/10.1145/3121432), Algorithms 6, 12 and corrected 18, with
the multiplication analysis of [Muller and Rideau
(2022)](https://doi.org/10.1145/3484514). The finite-exponent extensions used here
are explicitly the separate derivations in DF32_BOUNDS, not claims attributed to
those unlimited-exponent theorems.

[julia-log]: https://github.com/JuliaLang/julia/blob/v1.13.3/base/special/log.jl
