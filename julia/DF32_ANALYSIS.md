# DF32 phase and transcendental error analysis

This is a preparatory analysis for [SEP 48](../docs/dev/seps/sep-0048-julia-backend.md),
not an implementation or a qualification report. It does not advance the session gate
or establish that any GPU satisfies the arithmetic hypotheses below. The derivations
are independent of the experimental implementation.

The principal conclusions are:

- A three-word binary32 expansion of `2π`, exact integer turns, exact product
  transforms and accurate double-word subtraction give a small **absolute** reduction
  error. For the construction below it is less than `38u²` radians when the selected
  remainder is bounded by `22/7` and the turn count is at most `2²⁴`.
- Exponent assembly, not that reduction, can dominate the error at large phase. A
  split of one already assembled binary64 value and a product of two independently
  split values have different bounds. Cancellation in an expansion needs its own
  condition measure.
- A sine/cosine pair followed by one Newton normalization does not remove its
  first-order tangential error. A device library's accuracy must enter the budget.
  An alternative full-wide exponential has a useful conditional bound derived below;
  it still needs implementation, finite-range handling and device qualification.
- A relative bound for a `φ` function near a zero needs a conditioning factor.
  Binary32's smallest normal is a range floor, not an estimate of cancellation error.
  The error of the complete field increment must be bounded separately.
- The admissible phase ceiling is an output of these bounds and the chosen algorithm.
  Neither the experimental `8u` propagator witness nor its sampled `φ` errors are a
  universal implementation tolerance.

## 1. Sources, notation and proof boundary

Let `RN` denote binary32 round-to-nearest, ties-to-even, and let

$$u=2^{-24},\qquad \eta=2^{-150},\qquad f_{\min}=2^{-126}.$$

Here `η` is half the smallest binary32 subnormal. A normalized double word represents
the **exact real sum** `x = xh + xl` and satisfies `xh = RN(xh + xl)`. The magnitude
condition on the low word alone does not replace this definition at ties.

The following scalar bounds are the reusable inputs, not bounds for whole complex
expressions. They assume normalized inputs, the stated rounding order, correctly
rounded FMA and no exceptional exponent-range event:

| Operation | Analysed algorithm | Relative error bound |
|---|---|---|
| DW + DW | AccurateDWPlusDW | `a = 3u²/(1−4u)` |
| DW + FP | DWPlusFP | `2u²` |
| DW × DW | DWTimesDW3, including the low-by-low product | `b < 4u²` |
| DW × FP | DWTimesFP3 | `2u²` |
| DW ÷ FP | DWDivFP3, nonzero divisor | `d = 3u²` |
| DW ÷ DW | DWDivDW3, nonzero divisor | `9.8u²` |

Addition and division are from Joldes, Muller and Popescu, Algorithms 4, 6, 9, 15
and 18 and their theorems [J17]. Multiplication uses Muller and Rideau's corrected
and improved Theorem 2.8 [MR22]. For DWDivDW3, the final product is DWTimesDW3:
the authors' formal supplement uses `DWTimesDW12`, whose definition includes the
low-by-low product. This resolves the conflicting algorithm name in J17 Algorithm 18.
The retrieved supplement requires `p ≥ 14`; binary32's `p = 24` satisfies it.

Focused algorithm/theorem text and the corresponding supplement definitions were
checked. The Coq toolchain was **not** rerun. J17/MR22 establish scalar arithmetic
over an unlimited exponent range; they do not prove these transcendental functions,
GPU lowering, a phase ceiling or propagation accuracy.

### Finite binary32 arithmetic

For `TwoProd(a,b) = (RN(a*b), FMA(a,b,−RN(a*b)))`, a normal final product is not
enough to establish exactness of the residual. A sufficient condition from J17,
using leading-bit exponents `ea, eb`, is

$$e_a+e_b\ge e_{\min}+p-1=-103,$$

with no overflow. Exact representability of the residual can also be established
directly for special operands. Zero factors are exact special cases. This condition
must be checked for the **actual product transforms**, not only the final function.

With gradual underflow, a single RN operation admits an absolute model of the form
`|RN(t)−t| ≤ u|t| + η`. That does not extend a complete unlimited-exponent DW theorem
by adding one `η` at the end: every affected word operation and later amplification
must be accounted for. In particular:

- Low-by-low products can underflow even when the high result is normal.
- A normalized pair loses its extra precision near the bottom of the binary32 range.
  It cannot be treated as a 48-bit format with binary32's full exponent range.
- Multiplication by a power of two is exact only when its shifted result is
  representable. Halving a tiny low word is not automatically exact.
- Division, complex squared norms and reciprocals need their own scaling/exception
  policy. A naive `x²+y²` is not safe over the entire binary32 range.
- Exact zero, identity and masked-zero paths should bypass unnecessary arithmetic.
  A deliberately omitted tiny term needs a derived absolute tail bound.

The production choice must be either a proved safe range for each use, exact
power-of-two scaling with a proved reconstruction, or a word-level absolute-error
analysis of the exceptional range. Silent FTZ is not permitted by SEP 25.

Apple's Metal Shading Language 4.1 specification permits subnormal flushing and
nearest-even or toward-zero rounding. Its precise single-precision `sin`, `cos`
and `exp` guarantee is only 4 ulp; fast math has different guarantees [MSL]. Thus
the language specification alone does not establish the DW hypotheses. Each
qualified compiler/device route needs the exactness, rounding and range probes
required by SEP 48; observed errors smaller than the specification are witnesses.
Passing probes alone does not strengthen a platform's guarantee over untested inputs.
The proof must identify its arithmetic assumptions and the ranges in which the
qualified route establishes them.

## 2. Input representation and exponent assembly

There are three different reference values: the mathematical operator, the
translated binary64 data, and the exact sum of the supplied DF32 words. Tests must
name their reference. The bounds below start from exact operations on the translated
data, with any translation/approximation error added separately.

### One host assembly followed by one split

Suppose a finite binary64 value `x` is converted using

$$x_h=\operatorname{RN}_{32}(x),\qquad
  x_l=\operatorname{RN}_{32}(x-x_h).$$

When the first rounding has the normal relative bound, the subtraction is exact in
binary64 and the residual rounding does not encounter the finite-range caveat,

$$|(x_h+x_l)-x|\le u^2|x|.$$

If a binary64 assembly of a desired value `x*` has error `E64`, the bound becomes
`E64 + u²|x|`. The host assembly still needs its own cancellation bound; doing it
on the host does not erase that error.

Normalize the pair after the split, for example with exact TwoSum. The raw two
conversions can fail the strong normalization condition at a tie. A binary64
witness is `x = 1+3u−2⁻⁵²`: the first word is `1+2u`, the second rounds to `u`,
and RN of their exact sum is `1+4u`, not the first word. An exact normalization
preserves the represented sum and the bound above. The witness was checked with
binary32 conversions as well as by its binary exponent construction.

### Device assembly

For two split values representing `x*` and `y*`, with errors `Ex, Ey`, a DW product
has the conservative bound

$$E_{xy}\le |x^*|E_y+|y^*|E_x+E_xE_y
       +b(|x^*|+E_x)(|y^*|+E_y).$$

If both splits have the bound `u²` and the product uses DWTimesDW3, its leading
relative coefficient is `1 + 1 + 4 = 6`, not one. A truly exact binary32 factor
instead uses the DWTimesFP3 bound. Do not narrow an arbitrary binary64 step or
weight to binary32 and call that an exact factor.

For a sum of `m` products `θ = Σ tj`, let `Ej` bound each product's absolute error.
Sequential accurate DW additions give the conservative bound

$$E_\theta\le (1+a)^{m-1}\sum_j E_j
 +\big((1+a)^{m-1}-1\big)\sum_j|t_j|.$$

This is useful even when the result nearly cancels. Expressing it as a relative
bound introduces `κ = Σ|tj|/|θ|`. Accurate addition prevents the sloppy-add
catastrophe; it cannot undo errors already present in its operands.

Consequently, `H sup|Im L|` alone is not a bound on assembly error for an arbitrarily
cancelling representation of `L`. Admission also needs `Eθ`, or a conservative
term-magnitude bound that controls it. Constant and carrier-shifted combinations
should be assembled in binary64 and split once where the execution model permits.

At the proposed candidate ceiling `Θ = 2²⁴·2π`, even a single split contributes as
much as `2πu` of absolute phase error. The corresponding two-split product bound
has leading contribution `12πu`. This explains why the same phase ceiling cannot
be assigned to all assembly paths without further analysis.

## 3. Three-word `2π` reduction

Write `τ = 2π` and generate successive RN32 residuals from a certified interval for
the mathematical constant, rather than splitting a binary64 approximation to it:

$$t_1=\operatorname{RN}_{32}(\tau),\quad
 t_2=\operatorname{RN}_{32}(\tau-t_1),\quad
 t_3=\operatorname{RN}_{32}(\tau-t_1-t_2).$$

The resulting exact binary32 words are:

| Word | Binary32 bit pattern | Exact hexadecimal value |
|---|---|---|
| `t1` | `0x40c90fdb` | `0x1.921fb6p+2` |
| `t2` | `0xb43bbd2e` | `−0x1.777a5cp−23` |
| `t3` | `0xa7f72ced` | `−0x1.ee59dap−48` |

For `ρ = τ−(t1+t2+t3)`, exact rational interval arithmetic gives

$$|\rho|<2^{-74};$$

its magnitude is approximately `4.2251996267949711e−23`.

The calculation is reproducible without a floating-point value of π. Use
`τ = 32 atan(1/5) − 8 atan(1/239)`, with each arctangent enclosed by two successive
partial sums of its alternating rational series. Sixty-four terms give ample
separation from each binary32 rounding midpoint. After choosing a candidate word,
prove that the **entire** residual interval lies between its neighboring midpoints,
subtract that exact word from the interval, and repeat. The tangent addition formula
and the positive principal-angle interval establish the Machin identity used here.
The words, rounding intervals and strict `2⁻⁷⁴` bound were checked this way.

### Integer selection is a separate obligation

The turn count `n` must be an exact integer. For `|n| ≤ 2²⁴`, conversion to binary32
is exact, so `TwoProd(Float32(n), tj)` supplies each exact two-word product. Their
nonzero operand exponents satisfy the sufficient residual condition above.

A binary32 reciprocal/product followed by `round` is only a candidate turn count:
it need not select the mathematically nearest turn near a half-integer. Nor is
every integer just beyond `2²⁴` exactly representable in binary32. An implementation
must certify both the integer conversion and the remainder bound used below.

One viable design uses an initial approximate integer, forms the accurate remainder,
then adjusts the integer by a few turns and recomputes the remainder from the original
phase. Accept either neighboring integer at an uncertain half-turn; periodicity
does not require a unique nearest integer. Certify `|r| ≤ 22/7` with the reduction
error included. This generous interval is just wider than `π`. Avoid a circular
test that assumes the reduction has already been exact. The candidate selection,
boundary handling and integer-overflow proof remain implementation work.

### Reduction bound for a certified integer

Let `x` be the exact supplied DF32 phase, `r = x−nτ`, `K ≥ |n|`, and `R ≥ |r|`.
Compute exact product expansions `Pj = n tj`, then

$$r_1=\operatorname{DWAdd}(x,-P_1),\quad
 r_2=\operatorname{DWAdd}(r_1,-P_2),\quad
 \widehat r=\operatorname{DWAdd}(r_2,-P_3).$$

The exact partial remainders are bounded by

$$R_1=R+K(|t_2|+|t_3|+|\rho|),\quad
 R_2=R+K(|t_3|+|\rho|),\quad R_3=R+K|\rho|.$$

Propagating each accurate-add error yields

$$|\widehat r-r|\le
 a\big((1+a)^2R_1+(1+a)R_2+R_3\big)+K|\rho|. \tag{1}$$

For `K = 2²⁴`, `R = 22/7` and `|ρ| < 2⁻⁷⁴`, exact rational evaluation of this
upper bound is below `38u²` radians, approximately `1.327e−13`. This is a bound
relative to the supplied word pair, **not** to the phase before its split/assembly.
Add `Eθ` from section 2 for that comparison. The bound is not a proof that any
unimplemented reducer, integer-selection scheme or device realizes this sequence.

## 4. The propagator pair

Write the desired exponent as `z = a0+iθ`. Bound real assembly error by `Ea`,
phase assembly error by `Eθ`, and reduction error by `Er` from (1).

### Binary32 sine/cosine and Newton normalization

If the reduced pair is collapsed to one binary32 angle `t`, it adds at most
`u|rhat|+η` radians, subject to the usual range condition. Let
`q = (cos(t), sin(t))`, and let the library return `q+e` with `||e||₂ ≤ ε`.
The ideal Newton normalization is

$$F(q+e)=(q+e)\frac{3-\|q+e\|_2^2}{2}.$$

Since `||q||₂=1`, expansion gives

$$F(q+e)-q=e-q(q\cdot e)-e(q\cdot e)
 -\tfrac12q\|e\|^2-\tfrac12e\|e\|^2,$$

and therefore

$$\|F(q+e)-q\|_2\le\varepsilon+\tfrac32\varepsilon^2
 +\tfrac12\varepsilon^3. \tag{2}$$

The first-order radial error is removed; the tangential component remains. An
accurate norm and Newton step cannot turn a weak sine/cosine accuracy guarantee
into a strong phase guarantee.

For a concrete implementation, let `EN` bound error in the computed squared norm,
and `D` bound error in the scalar correction `(1−||q+e||²)/2`. The correction error
adds at most `(1+ε)D` to (2); final FMA rounding adds its componentwise RN bound.
With exact square products and accurate DW addition, one may take
`EN ≤ a(1+ε)²`. If the subtraction of one is exact and the final scalar sum is RN,

$$D\le\tfrac12\left[E_N+u(2\varepsilon+\varepsilon^2+E_N)+\eta\right].$$

These are explicit obligations on the implementation, including its use of FMA.
For `ε = κu`, the leading unit-vector budget is approximately

$$E_\theta+E_r+(R+\kappa+1)u,$$

before magnitude evaluation and final magnitude-times-vector rounding. A safe
conversion of a `ν`-ulp componentwise library guarantee gives `κ ≤ 2ν` for normal
outputs. Apple's portable precise `ν = 4` guarantee alone leaves insufficient room
to prove the experimental `8u` witness by this bound. That is a limitation of the
guarantee and bound, not a claim that a measured device necessarily fails.

The magnitude shortcut `exp(ah)·(1+al)` has separate errors: the library exponential,
rounding `1+al`, multiplication, and the Taylor defect

$$|e^{a_l}-(1+a_l)|\le\tfrac12 a_l^2 e^{|a_l|}.$$

Both signs of the real exponent must be covered. Forming the backward value as the
reciprocal of the forward result adds a different error budget; it is not equivalent
to separately evaluating the two magnitudes. Likewise, a forward/backward product
or a long repeated-propagator drift test needs its own accumulated bound.

### A conditional bound for a full-wide alternative

A possible alternative evaluates the complete complex exponential in DF32 and
rounds the result once. This derivation does not select that implementation over the
candidate sine/cosine path; performance and the final proof must decide.

Use the complex one-norm `|re|+|im|` for arithmetic estimates. Ordinary four-product
complex multiplication, with the verified real DW operations, has the bound

$$\|\widehat{vw}-vw\|_1\le c\|v\|_1\|w\|_1,
 \qquad c=b+a(1+b). \tag{3}$$

For complex squaring, its relative Euclidean error is at most `β = 2c`, because
`||v||₁² ≤ 2|v|²` and `|v²| = |v|²`. No cancellation factor is hidden here.

Scale `w = a0+i rhat` exactly to `y = w/2ˢ` with `|y|₂ ≤ 1/2`. Select `s` using a
safe upper bound on the **whole word pairs**, not a potentially underestimated
binary32 norm of their high parts. The degree-18 Taylor polynomial, evaluated as
`Pm = 1 + y P(m+1)/m`, `m = 18,...,1`, `P19 = 1`, has analytic tail

$$T\le e^{1/2}\frac{(1/2)^{19}}{19!}<2^{-75}.$$

For a conservative arithmetic bound use `r1 = 3/4 ≥ ||y||₁` and
`||Pm||₁ ≤ exp(r1) < 17/8`. Set

$$Q=(1+a)(1+d)(1+c),\qquad E_{19}=0,$$

$$E_m\le Q\frac{r_1}{m}E_{m+1}
 +(Q-1)\frac{r_1}{m}\frac{17}{8}+a. \tag{4}$$

This includes each multiplication, division by the exactly representable integer
`m`, and accurate addition of one. Exact rational evaluation at `u = 2⁻²⁴`, with
`b = 4u²`, gives `E1 < 37.209u²`. Since `|exp(y)| ≥ exp(−1/2) > 3/5`, take
`δ0 = (5/3)(E1 + 2⁻⁷⁵)` as the relative starting bound.

After `s` squarings the relative bound is

$$1+\delta_s\le(1+\delta_0)^{2^s}(1+\beta)^{2^s-1}. \tag{5}$$

For `s ≤ 6`, (5) is below `1.724e−11`; for `s ≤ 8`, below `6.909e−11`.
The latter is about `0.001160u`. These outward-rounded numerical bounds were
checked by rational evaluation of (4)–(5), not fitted to exponential samples.

These numbers exclude argument assembly/reduction, exceptional ranges and the final
binary32 conversion. Argument perturbation `Δz` contributes at most
`exp(|Δz|)−1` in relative complex error; for a purely imaginary perturbation the
sharper bound is `|Δθ|`. A final componentwise binary32 conversion contributes
`u|value|₂+sqrt(2)η` as an absolute Euclidean error.

For example, `|Re w| ≤ 87` and `|Im w| ≤ 22/7` need no more than eight halvings.
This is an illustrative conditional range, not an admitted range chosen by fiat.
The true real-part bounds must preserve both `exp(a0)` and `exp(−a0)` and every
intermediate. Even in this moderate range, tiny individual polynomial components
can invalidate the unlimited-exponent hypotheses; a finite-range implementation
must handle them before claiming (4)–(5).

## 5. `φ` functions: tails, recurrence and conditioning

For integer `k ≥ 1`,

$$\varphi_k(z)=\sum_{m=0}^{\infty}\frac{z^m}{(m+k)!}
 =\frac1{(k-1)!}\int_0^1 e^{(1-t)z}t^{k-1}\,dt,$$

with `φ0(z)=exp(z)` and `φk(0)=1/k!`. These definitions directly give

$$\varphi_{k+1}(z)=\frac{\varphi_k(z)-1/k!}{z},\qquad
 \varphi_k(z)=z\varphi_{k+1}(z)+1/k!,$$

and, by separating the integral or comparing power-series coefficients,

$$\varphi_k(2z)=2^{-k}\left[
 e^z\varphi_k(z)+\sum_{j=1}^{k}\frac{\varphi_j(z)}{(k-j)!}\right]. \tag{6}$$

Only the exponential is periodic. Reducing the argument in the denominator of
the recurrence or evaluating `φk` at the reduced phase would compute the wrong
function.

### Small-argument series

For `|y|₂ ≤ r ≤ 1/2`, truncation after degree `M` satisfies

$$|T_M|\le\frac{r^{M+1}}{(M+k+1)!}
 \frac1{1-r/(M+k+2)}. \tag{7}$$

This follows from bounding the ratio of every later term. A small final term alone
is not a tail bound without such a ratio, and it says nothing about rounding already
accumulated in preceding terms.

The integral supplies a nonzero lower bound in this disk:

$$|\varphi_k(y)|\ge\frac{e^{-r}\cos r}{k!}.$$

Thus a relative bound is meaningful here. The ratio of the positive majorant
`exp(r)/k!` to this lower bound is at most `exp(2r)/cos(r)`.

A production series should propagate the term error and the sum error separately.
Use (3), the DWDivFP3 bound for exact integer denominators, and the accurate-add
bound. Include errors in the initial `1/k!` coefficient; a coefficient formed in
binary64 and split has both its binary64 approximation error and its split error.
Multiplication by a pre-rounded reciprocal needs a different count than division
by an exact integer. Factorials should only be treated as exact binary32 factors
where representability has actually been established.

Downward recurrence and each doubling (6) propagate **absolute** errors through
the sum of magnitudes of their terms, using the same multiplication/addition bounds.
An error budget for `φ0` is not automatically an error budget for every `φk`.

### Upward recurrence

Suppose `Eprev` bounds the absolute error of `φ(k−1)`, `C` bounds the error of
`c = 1/(k−1)!`, and `V` bounds error in a computed reciprocal `v = 1/z`.
Here `Eprev`, `C`, `V`, `Et` and `Ek` are absolute complex one-norm error bounds.
For exact `t = φ(k−1)−c`, the numerator error is bounded in the one-norm by

$$E_t\le(1+a)(E_{\mathrm{prev}}+C)+a\|t\|_1.$$

After multiplication by the reciprocal, a valid bound is

$$E_k\le\|v\|_1E_t+\|t\|_1V+E_tV
 +c_{\mathrm{mul}}(\|v\|_1+V)(\|t\|_1+E_t), \tag{8}$$

where `c_mul` is (3), with no identification of the coefficient `c` and
`c_mul`. For `|z|₂ > 3`, `||1/z||₁ ≤ sqrt(2)/3`; this damps absolute error.
It does not guarantee small relative error at a zero of a resulting function.

A reciprocal formed as `conj(z)/(re(z)²+im(z)²)` needs a positive-denominator
bound. Ignoring the finite-range exceptions, its denominator has relative error
at most `(1+b)(1+a)−1`; applying DWDivDW3 gives a reciprocal relative bound
`(1+9.8u²)/(1−[(1+b)(1+a)−1])−1`. This is approximately `16.8u²` and is
separate from the scalar division theorem. A scaled reciprocal is required outside
a range where those squares and residuals are representable.

### Zeros and branch boundaries

`φ1(2πin)=0` for nonzero integer `n`. More generally the relative input condition
number is `|z φk'(z)|/|φk(z)|`, which is unbounded near a zero. Accurate DW
subtraction cannot remove the error already present in a computed exponential.

The integral gives a usable absolute input-error bound:

$$|\varphi_k(z+\Delta z)-\varphi_k(z)|
 \le |\Delta z|\frac{e^{\max(\operatorname{Re}z,0)+|\Delta z|}}{(k+1)!}. \tag{9}$$

For a concrete warning about an unconditioned relative test, choose the exact DF32
input `x = t1+t2` from section 3 and consider `φ1(ix)`. Its exact phase distance
from `2π` is about `6.860497998e−15`. Replacing `2π` by the three-word constant
changes this tiny distance by approximately `4.2252e−23`, a relative change of
about `6.16e−9`, even before evaluating a polynomial. The experimental witness
coefficient `2⁻³⁶ max(1,|Im z|)` is only about `9.14e−11` there. This illustrates
why that witness cannot be promoted to a universal relative requirement for this
three-word algorithm. It is not a measured failure of a production function.
Indeed, even with exact exponential evaluation after the approximate reduction, the
relative error in this `φ1` evaluation is

$$\frac{|\sin(\rho/2)|}{|\sin((x-\tau)/2)|}>6\times10^{-9}.$$

The strict inequality follows from the certified constant intervals and
`t−t³/6 ≤ sin(t) ≤ t` for these positive small arguments. The true `φ1` magnitude
is about `1.09e−15`, far above binary32's smallest normal, so that range floor
does not address the conditioning in this example.

A branch at magnitude `B < π` has a useful mathematical property: for every
`|z|₂ ≤ B`, rotate the integral by `exp(−i Im(z)/2)` to obtain

$$|\varphi_k(z)|\ge
 \frac{e^{\min(\operatorname{Re}z,0)}\cos(|\operatorname{Im}z|/2)}{k!}>0.$$

This explains a possible series/recurrence split below `π`; it does not by itself
choose `B=3` or establish its best cost/error tradeoff. At a selected boundary,
qualify both algorithms and their overlapping bounds on neighboring representable
arguments. Their outputs need not be bitwise equal, but both must fit the same
function and downstream increment budget.

The current public Adams limit is six predictor nodes; the corrector may need the
next moment. Qualification must use the maximum moment index actually requested by
the finalized predictor, corrector and defect tables, not infer it from the four
orders in an old witness. Tests at extra orders are useful arithmetic stress tests
but do not expand admission.

### From scalar errors to a field increment

For a representative increment

$$I=h\sum_{j=0}^{N-1}j!\varphi_{j+1}(z)G_j,$$

the contribution from errors in the functions is bounded absolutely by

$$|\Delta I|\le |h|\sum_{j=0}^{N-1}j!|G_j|E_{j+1},$$

before errors in `h`, `Gj`, coefficient multiplication, accumulation and state
rounding. Their budgets also use sums of magnitudes. A tiny `φj` does not make
its **absolute** contribution inaccurate; a small difference of large weighted
contributions can nevertheless be ill-conditioned. This is the distinction required
by SEP 33 §7.1 and SEP 25. A scalar relative-error problem and a protected
field-increment mixed bound must not be conflated.

## 6. Phase-domain construction for real sessions

For every exponential argument formed by the method, bound the relevant integral
of the actual represented operator, including history transport, stage frame changes,
carrier shifts and both signs of a propagator pair. For scalar transport,

$$\left|\operatorname{Im}\int_a^b L(z)\,dz\right|
 \le |b-a|\sup_{z\in[a,b]}|\operatorname{Im}L(z)|.$$

The following conservative span implements SEP 48's stated model. Its validity still
depends on verifying the finalized session's argument call sites and history capacity:

- For a fixed schedule, `hmax` is the largest actual prescribed step that may be used,
  not the adaptive request's `dz_max`. Landing/clipping may reduce it.
- For a fresh one-step session, use `H ≤ hmax`, capped by the remaining run length.
  Verify that every admitted tableau's stage/frame differences lie in that span;
  an outlying stage coefficient requires a corresponding factor.
- For fresh Adams history, use `(qmax+1)hmax`, capped by `zend−zstart`. Include
  the neighboring-order/corrector work, not just the currently selected predictor.
- For a history resume, let `A = zstart−zoldest` for **every retained row that can
  still be used**. A conservative bound is
  `H = min(A+(qmax+1)hmax, zend−zoldest)`. A partially filled history can retain
  samples from much larger old steps; replacing `A` with the new `dz_max` is wrong.
- Invalid, future, non-monotone or out-of-profile restored positions refuse before
  arithmetic. Field-only restart has no retained history age.

The supremum must cover frequencies/modes actually represented and all intersected
position intervals, including the restored past. For a polynomial interval, endpoints
and certified interior derivative extrema suffice. A maximum of interpolation samples
or unverified floating-point root estimates does not establish a bound. A convenient
conservative alternative is an outward-bounded Bernstein representation: its scalar
polynomial lies between its coefficient extrema on the interval. Account for rounding
when constructing those coefficients. Other representations need their own enclosure
or a named refusal.

Track assembly error alongside the phase bound. A shifted operator can have a small
phase and a badly conditioned sum; an external frame-restoration exponential may use
the unshifted phase. Each is a distinct bound obligation.

A practical admission decision therefore needs all of:

1. A certified span and operator bound for every argument family.
2. A bound on input/split/assembly error, including cancellation.
3. The reducer's integer and remainder certificate.
4. Real-part and intermediate representability bounds for both exponent signs.
5. The chosen transcendental algorithm's bound, propagated into the increment budget.
6. A qualified compiler/device arithmetic mode satisfying that algorithm's hypotheses.

Where a useful bound can be expressed as `c1 u + c2 |θ|u²`, choose the phase
ceiling from the allocated tolerance, the derived coefficients and the integer limit.
For a target `Etarget`, the formal inequality is

$$|\theta|\le\frac{E_{\mathrm{target}}-c_1u}{c_2u^2},$$

only if its numerator is positive and all other conditions hold. Conditioning may
require an absolute assembly test instead of a single global `c2`. A refusal off
Metal should name `wide="binary64"`; Metal must refuse rather than silently degrade.

## 7. What is established, and what remains

Established by source reading or the derivations here:

- The specific scalar arithmetic bounds and their unlimited-exponent hypotheses.
- Certified three-word constant bits, its strict residual bound and conditional (1).
- The effect of separate splits/products and cancellation on exponent assembly.
- Newton normalization's retained tangential error, and the vendor-bound limitation.
- The analytic Taylor tail, arithmetic recurrences (3)–(5), `φ` tail (7), recurrence
  error form (8), input sensitivity (9), and the near-zero conditioning distinction.
- The need to include actual replay steps, retained history and interior extrema.

Not established, and required before phase 3 can be considered complete:

- An implemented/certified turn-selection routine, finite-range primitive policy,
  selected `φ` branch/order strategy, or final phase and real-part admission constants.
- A compiler proof or device test of nearest-even, correctly rounded FMA/division,
  preservation of required subnormal behavior, or absence of accidental binary64.
- A production transcendental implementation satisfying these exact operation counts.
- End-to-end field/statistic parity, bounded drift, survival, performance, or GPU
  qualification. No random sample or high-precision comparison replaces the bounds.

Implementation tests should include exact rational primitive references; BigFloat
function references with precision/convergence checks; integer-turn boundaries;
points on both sides of multiples of `2π`; tiny and cancelling low words; both sides
of the `φ` branch; largest admitted real/imaginary arguments; actual replay steps
exceeding `dz_max`; and restored old history. Exact structural zeros remain exact.
Use separately derived absolute/mixed budgets around roots, and retain the full
field-increment conditioning and propagation sensitivity in session tests.

## References

- [J17] M. Joldes, J.-M. Muller and V. Popescu, “Tight and Rigorous Error Bounds
  for Basic Building Blocks of Double-Word Arithmetic,” *ACM TOMS* 44(2), article 15
  (2017), [doi:10.1145/3121432](https://doi.org/10.1145/3121432).
  [Author manuscript](https://hal.science/hal-01351529v3).
- [MR22] J.-M. Muller and L. Rideau, “Formalization of Double-Word Arithmetic,
  and Comments on ‘Tight and Rigorous Error Bounds for Basic Building Blocks of
  Double-Word Arithmetic’,” *ACM TOMS* 48(1), article 9 (2022),
  [doi:10.1145/3484514](https://doi.org/10.1145/3484514).
  [Author manuscript](https://hal.science/hal-02972245v2),
  [authors' formal supplement](https://www-sop.inria.fr/members/Laurence.Rideau/DW_arithmetic-submitted_paper_release.zip).
  The checked manuscript is dated August 31, 2021; its pagination differs from the
  publisher's typesetting.
- [MSL] Apple, *Metal Shading Language Specification*, version 4.1, June 4, 2026,
  §§8.1–8.4 and Table 8.1, printed pp. 368–370,
  [official specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
- W. J. Cody and W. Waite, *Software Manual for the Elementary Functions*
  (Prentice-Hall, 1980), is the historical reduction reference named by SEP 48.
  Its full text was not checked for this analysis; (1) is a derivation for the
  explicitly stated three-word algorithm, not a theorem attributed to that book.
