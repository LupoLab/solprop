# DF32 function bounds and initial domain

This is the derivation for the initial SEP 48 phase 3 function leaves. It does not
admit a runtime, qualify a device, or replace SEP 33's buffer-range rules. The scalar
primitives in `src/double_float.jl` are independently reviewed; the separate leaves
are in `src/double_float_functions.jl`. Bounds were fixed analytically before
numerical comparisons; their additional finite-underflow proofs need independent
review as well as numerical witnesses.

The initial domain is deliberately smaller than the nominal SEP 48 phase target:

\[
 |\Re z|\le16,\qquad |\Im z|\le2^{20},\qquad
 |\delta\Re z|+|\delta\Im z|\le u/8,\quad u=2^{-24}.
\]

Here `z` is the exact real sum of the stored word pairs, and `delta z` is the
difference from the argument whose mathematical function is requested. The last
condition includes representation and assembly; a bound on the net phase alone
does not establish it. All exponents are dimensionless.

## 1. Arithmetic model and source boundary

The current derivation assumes binary32 round-to-nearest ties-to-even, correctly
rounded division and FMA, gradual underflow, and no overflow. No contraction or
reassociation is permitted except the explicit FMA operations. Set

\[
 \eta=2^{-150},\qquad A=\frac{3u^2}{1-4u},\qquad \alpha=4u^2.
\]

`eta` is half the least positive binary32 subnormal. A correctly rounded ordinary
product, division or FMA has absolute error at most `u*abs(exact_result)+eta`.
This is an error bound, not permission to enable flush-to-zero.

The normalized-pair condition is `hi = RN(hi+lo)`, not just a magnitude inequality.
It implies `abs(lo) <= u*abs(hi)`; a subnormal high word necessarily has zero low
word. Internal results may contain subnormal words even though checked host
preparation deliberately refuses them.

The six selected algorithms and their tighter bounds when all their original
hypotheses hold are from Joldes, Muller and Popescu [J17], with the multiplication
correction of Muller and Rideau [M22]. Algorithm 18's final multiplication is
Algorithm 12, `DWTimesDW3`: the printed name `DWTimesDW2` conflicts with its
algorithm number/proof, and the authors' formal supplement confirms Algorithm 12.
The formal division theorem uses `p >= 14`; binary32 satisfies it.

The bounds below are additional conservative derivations for these exact graphs
with gradual underflow. They are not attributed to the papers' unlimited-exponent
theorems. Metal 4.1 permits toward-zero rounding and subnormal flushing [MSL];
neither its language specification nor disabled fast math establishes this model.
A Metal path needs a separately established model and propagated allowance before
using these bounds. CPU evidence does not establish CUDA or ROCm behavior either.

### 1.1 Addition and product transforms

Every binary32 word is an integer multiple of `2^-149`. If the exact sum or
difference of two words is subnormal, it is already representable. Otherwise its
rounding is the same as in the unlimited-exponent format. Consequently, without
overflow, every addition/subtraction in TwoSum and Fast2Sum agrees with the
unlimited-exponent operation. Their exactness and the addition bounds carry over;
Fast2Sum still needs its exponent condition.

TwoProd instead satisfies

\[
 p+e=ab+\epsilon_p,\qquad |\epsilon_p|\le\eta.                 \tag{1}
\]

The ideal residual is either representable or is rounded in the subnormal range.
A normal `p` alone is not sufficient for exactness: for
`a=b=(1+2^-23)*2^-63`, the normal rounded product loses residual `2^-172`.

### 1.2 Conservative scalar bounds used by the functions

Write `x`, `y` for exact pair values and `f` for a binary32 scalar. Under the model
above, with finite intermediate results:

| Operation | Function-use absolute bound |
|---|---|
| DW + FP | `2u^2 * abs(x+f)` |
| DW + DW | `A * abs(x+y)` |
| DW x FP | `3u^2 * abs(x*f) + 3eta` |
| DW x DW | `7u^2 * abs(x*y) + 5eta` |
| DW / exact integer `m`, `1 <= m <= 32` | `8u^2 * abs(x/m) + 4eta` |
| DW / DW, positive `y`, `8 <= y.hi <= 2^42` | `16u^2 * abs(x/y) + (6abs(x)*(1+1/abs(y))+6)eta` |

These constants deliberately exceed the no-underflow theorem constants. The last
two rows have restricted divisors; they are not new general division guarantees.

For DW x FP, (1), the FMA error and exact final Fast2Sum give

\[
 |E|\le2u^2|x_hf|+(2+2u)\eta
      \le3u^2|xf|+3\eta.                                    \tag{2}
\]

For DW x DW put `P=abs(x.hi*y.hi)`. Expanding the rounded low product, two FMAs
and final low-word addition gives

\[
 |E|\le(6u^2+8u^3+7u^4+4u^5+u^6)P
       +(4+8u+4u^2+u^3)\eta.                                 \tag{3}
\]

Use `abs(x*y) >= (1-u)^2 P` to obtain the table entry. If `P` is normal, the
leading product dominates the final correction, so Fast2Sum is exact. If `P` is
subnormal, the high-product residual rounds to zero and the cross terms are at
most `u*P <= eta`; their nearest-even rounding is zero. The result is the leading
product and its lost low contribution is covered by the absolute allowance.
The analogous split establishes (2) when its leading product is subnormal.

For integer division, allow rounding errors in both residual subtractions rather
than borrowing their unlimited-exponent exactness. With `R=abs(x.hi/m)`, direct
substitution in Algorithm 15 bounds its relative-size part by

\[
 (7u^2+11u^3+10u^4+5u^5+u^6)R.
\]

The absolute-underflow coefficient is below 4 for `1 <= m <= 32`. Since
`abs(x/m) >= (1-u)R`, the stated `8u^2` follows. If the leading quotient is
subnormal, multiplication by the exact integer has a representable residual;
the correction is at most one subnormal spacing. Fast2Sum therefore still has a
leading exponent at least as large as its correction, or has a zero first operand.

For the restricted DW divisor, its first reciprocal is normal and the FMA
`1-y.hi*reciprocal` is exact: its nonzero residual is on a lattice no finer than
`2^-47`. The low-product rounding contributes at most
`u^2*(1+u)+eta`. If that product underflows, its magnitude is far smaller than a
nonzero high residual; the zero-high case is also valid for Fast2Sum. Otherwise
the authors' original Fast2Sum argument applies. Algorithm 18's corrected
reciprocal `v` consequently satisfies

\[
 |yv-1|\le8u^2+4(1+|y|)\eta.                                \tag{4}
\]

One way to check the constant is to set
`d=y*reciprocal-1`, with `abs(d)<=2u+u^2`. The exact Newton error is `-d^2/y`.
Add the low-product error, (2) for the correction, and DWPlusFP's `2u^2` bound.
The relative part is below `8u^2`; multiplying by `x` with (3) yields the final
table row. This proof uses the corrected Algorithm 12 final product.

### 1.3 Complex operations

Use componentwise accurate sums, and the four scalar products of the usual complex
product. Since

\[
 \sqrt{(|ac|+|bd|)^2+(|ad|+|bc|)^2}
 \le\sqrt2\,|a+ib|\,|c+id|,
\]

the scalar bounds imply these convenient constants:

\[
 \begin{aligned}
 |E_{+}|&\le\alpha|x+y|,\\
 |E_{\times}|&\le\mu|xy|+\nu\eta,
       &&\mu=15u^2,\quad\nu=15,\\
 |E_{/m}|&\le8u^2|x/m|+6\eta,&&1\le m\le32.
 \end{aligned}                                               \tag{5}
\]

The product constant uses the sharper `A` when deriving it:
`sqrt(2)*(7u^2 + A*(1+7u^2)) < 15u^2`; the two scalar product allowances give
`10sqrt(2)*(1+A)eta < 15eta`. Adding the real scalar one has relative bound
`2u^2` in the complex norm.

The computed squared modulus `d` has

\[
 |d-|x|^2|\le11u^2|x|^2+11\eta.                             \tag{6}
\]

For `2.99 <= abs(x) <= 2^21`, its positive high word is in the divisor interval
of (4). Forming `conj(x)/d` componentwise then gives

\[
 |\widehat{x^{-1}}-x^{-1}|
 \le32u^2/|x|+32(1+|x|)\eta.                                \tag{7}
\]

No unscaled complex division is used outside this interval.

## 2. Integer turn selection and three-word reduction

Let `T=2pi`. Generate three words by successive nearest-even splitting of an
interval enclosing the mathematical constant, rather than splitting a rounded
binary64 constant:

| Word | Float32 bits | Exact word lattice |
|---|---|---|
| `c1` | `0x40c90fdb` | integer x `2^-21` |
| `c2` | `0xb43bbd2e` | integer x `2^-46` |
| `c3` | `0xa7f72ced` | integer x `2^-71` |

The constant remainder is positive and below `2^-74`. Directed 256-bit MPFR
intervals certify that both interval endpoints choose these same words and that
remainder bound. This is a derivation of the three-word scheme, not a claim about
an unread theorem in the Cody--Waite book.

Use `r=RN32(1/T)`, `q=RN32(theta.hi*r)`, and nearest-even integer `n` to `q`.
For `abs(theta)<=2^20`,

\[
 |q-\theta/T|
 \le\frac{3u+u^2}{1-u}\frac{2^{20}}T+\eta < 1/32.             \tag{8}
\]

The absolute term covers a tiny quotient. Thus `n` is within one of a nearest
integer, `abs(n)<2^18`, and `abs(theta-n*T) < 17pi/16 < 27/8`. **No correction
branch is needed:** the exponential polynomial covers this slightly wider
remainder interval. Nearest-turn identity at half-turn boundaries is not required.
Conversion of this bounded integer to Float32 is exact.
The kernel first rounds `q` to an integral binary32 value, then uses a protected
`unsafe_trunc(Int32, ...)`: the proven bound makes this conversion exact and avoids
an unreachable allocating exception path inside the numerical kernel.

Form each `n*cj` with TwoProd and subtract the exact product pairs with
AccurateDWPlusDW. Each nonzero product residual is at least `2^-71`, so these
particular products satisfy exact TwoProd even in finite binary32. If
`rj=theta-n*(c1+...+cj)`, three applications of the addition bound give

\[
 E_R\le |n|\delta_T+
 A\big[(1+A)^2|r_1|+(1+A)|r_2|+|r_3|\big]<32u^2.              \tag{9}
\]

For a simple independent bound, use `abs(c2)<2^-22`, `abs(c3)<2^-47`,
`abs(n)<2^18` and `delta_T<2^-74`. They bound the three remainders by
`27/8+1/16+2^-29+2^-56`, `27/8+2^-29+2^-56` and `27/8+2^-56`.
The computed remainder has magnitude below `7/2`. When `n=0`, return the input
pair unchanged.

## 3. Polynomial complex exponential

After (9), the real part has magnitude at most 16 and the imaginary part less
than `7/2`. Choose a power-of-two scale so each component has magnitude at most
`1/4`; at most six halvings suffice. Exact comparison of a normalized pair with
these power-of-two thresholds avoids a rounded norm decision. Scale each word
once and normalize the pair; the complex scaling error is at most `3eta`.
The reduced argument has modulus below `1/2`.

Use the degree-18 exponential polynomial in nested form:

\[
 F_{19}=1,\qquad F_m=1+(y/m)F_{m+1},\quad m=18,\ldots,1.      \tag{10}
\]

Evaluate multiplication, division by the exact integer, then addition of real one
in that order. This avoids rounded reciprocal-factorial coefficients in the
exponential. Its truncation error is

\[
 |e^y-F_1|\le e^{1/2}(1/2)^{19}/19!<2.586\times10^{-23}.
\]

From (5), one multiplication/division pair has relative part at most `24u^2`
and absolute part at most `22eta`. Let `r=abs(y)<=1/2`. Propagating its error
through (10), with `abs(F_m)<=exp(r)`, bounds the polynomial error by

\[
 e^r u^2\left[
 2e^{(1+27u^2)r}+
 \frac{25}{1+27u^2}\big(e^{(1+27u^2)r}-1\big)\right]
 +23\eta e^{(1+27u^2)r}.
\]

Dividing by `abs(exp(y))>=exp(-r)` and including truncation gives the simple
relative bound `64u^2+64eta`.

For each of at most six complex squarings, the relative error satisfies

\[
 e_{j+1}\le(1+15u^2)(1+e_j)^2-1+2^{28}\eta.                  \tag{11}
\]

The absolute term uses `abs(exp(argument))>2^-24` throughout the admitted real
range. Without underflow terms, (11) gives
`(1+64u^2)^64*(1+15u^2)^63-1 < 5042u^2`. Propagating its absolute terms, the
initial scaling error and (9) leaves the following deliberately rounded-up bound:

\[
 \frac{|\widehat{e^z}-e^z|}{|e^z|}
 \le B_E=8192u^2+2^{40}\eta.                                \tag{12}
\]

This compares against the exact stored argument. Its difference from the requested
argument contributes at most `exp(abs(delta z))-1` separately. All high-word
products in this exponential stay far below binary32 overflow: the final norm is
less than `2^24`, and the unreduced phase never enters the polynomial.

## 4. Corrected phasor and rounded propagator pair

Evaluate (12) at `i*theta` and narrow its components once to binary32. Their
normwise error from the unit phasor is below `1.001u`. Compute their squared norm
using TwoProd on each component and accurate DW addition. Then form

\[
 d=-\tfrac12\operatorname{RN}_{32}
       ((N_h-1)+N_\ell),\qquad
 (c',s')=(\operatorname{FMA}(c,d,c),\operatorname{FMA}(s,d,s)).
\]

`N.hi-1` is an exact Sterbenz subtraction here. Equations (1) and the addition
bound imply the correction error is at most `4u^2+4eta`. If `v=q+e`, exact
radial correction leaves

\[
 |v(1-(|v|^2-1)/2)-q|
 \le |e|+\tfrac32|e|^2+\tfrac12|e|^3.
\]

Including correction and final FMA rounding gives phasor error below `17u/8`
and squared-norm defect below `3u`. The correction removes radial first-order
error; no claim is made that it corrects phase error. No vendor sin/cos accuracy
assumption is used: both components come from the polynomial (10).

Evaluate the positive magnitudes `exp(+Re z)` and `exp(-Re z)` using (12), then
narrow them to binary32. Their relative error is below `9u/8`. Multiplying each
by the corrected phasor, with conjugation for the backward pair, adds at most one
binary32 normwise rounding. Both magnitudes are normal in the chosen real range.
Component underflow is covered by the propagated absolute allowances.

The pair error relative to the exact function of the stored argument is below
`9u/2`. Adding an assembly/input bound at most `u/8` gives **less than `5u`**
relative to each requested propagator. This leaves margin inside the accepted
`8u` target on this smaller domain. For a split-once constant, one convenient
form is `(9/2)u + 2u^2*(abs(Re z)+abs(Im z)) + 2B_host`, where `B_host` is the
host assembly's absolute error bound.
It is not a certificate for the nominal `2^24*2pi` phase domain.

## 5. Phi functions: absolute bounds, including the join

Use the defining series and integral, for `k >= 1`:

\[
 \varphi_k(x)=\sum_{m\ge0}\frac{x^m}{(m+k)!}
 =\frac1{(k-1)!}\int_0^1e^{(1-t)x}t^{k-1}\,dt.               \tag{13}
\]

They give `abs(phi_k(x)) <= exp(max(Re x,0))/k!` and
`phi_k'=phi_k-k*phi_(k+1)`. An argument perturbation of magnitude `b` contributes
at most `b*exp(max(Re x+b,0))/(k+1)!`; use `b*exp(Re x+b)` for order zero.
Near zeros these are absolute bounds. In particular, phi1 vanishes at nonzero
integer multiples of `2pi*i`; a relative-only tolerance is insufficient there.

Provide orders `0:8`. Order zero alone uses (12). At exact zero return the stored
reciprocal-factorial tuple. Each nonexact table entry has representation bound
`2u^2/k!`, including binary64 preparation and its split.

For the branch decision compute (6) and compare its high word with 9. The small
branch is valid for every argument of modulus at most 3.01, and the large branch
for every argument of modulus at least 2.99. Equation (6) and the high-word
rounding ensure the actual decision falls within these overlapping domains.
Thus both sides of the rounded join have independently valid bounds.

### 5.1 Small branch

Choose at most three halvings. Compare the high word of (6) with
`(1-4u)*4^s/4` to determine the scale. This downward margin dominates its squared
norm error and high-word rounding, ensuring `abs(y)<=1/2`. Scale words once as in
section 3; the equivalent full-argument perturbation is at most `24eta`.

For highest requested order `N>=1`, use exactly 24 terms, `m=0:23`, with
`term_m=(term_(m-1)*y)/(m+N)`. The integer divisor is at most 31. Accurate complex
addition accumulates the terms. No data-dependent early stopping is assumed in
this first proof. If `S_N(r)=sum(r^m/(m+N)!,m>=0)`,

\[
 B_N\le128u^2 S_N(r)+1024\eta+
 \frac{r^{24}}{(24+N)!\,[1-r/(25+N)]}.                        \tag{14}
\]

To see the constants, term error is bounded by
`((1+2u^2)*(1+24u^2)^m-1)*abs(term_m)+30eta` for `m>0`.
Use `sum(m*r^m/(m+N)!) <= r*S_N(r)` and 23 additions with factor at most
`(1+4u^2)^23`. Their total is below (14). The tail is below `3.919e-33` at `N=1`.

Obtain lower positive orders by `phi_k=1/k!+y*phi_(k+1)`. For rigorous error
propagation define a product bound using exact-magnitude bounds `U,V` and
absolute input-error bounds `B,C`:

\[
 P(U,B,V,C)=(1+\mu)(UC+VB+BC)+\mu UV+\nu\eta.                \tag{15}
\]

Set `U_k=exp(max(Re y,0))/k!`, `c_k=1/k!`, `C_k=2u^2*c_k`.
The downward bound is

\[
 B_k=(1+\alpha)[P(|y|,0,U_{k+1},B_{k+1})+C_k]+\alpha U_k.
\]

Compute phi0 directly with (10), whose absolute error is bounded by
`(64u^2+64eta)*abs(exp(y))`. Raise the tuple through the exact identity

\[
 2^k\varphi_k(2y)=e^y\varphi_k(y)+
                  \sum_{j=1}^k\frac{\varphi_j(y)}{(k-j)!}.    \tag{16}
\]

Apply (15) to each product. For every sum, with exact summand magnitude bounds
`U,V` and errors `B,C`, use `(1+alpha)*(B+C)+alpha*(U+V)`.
Multiply the resulting bound by `2^-k` and add `3eta` for the final exact-power
word scaling. Phi0's squaring uses (15) directly. Update the analytic `U_k` from
(13) after each of at most three doublings. Finally add `1024eta` for the initial
argument-scaling perturbation. This recurrence fixes each order's bound without
assuming a nonzero result or fitting a constant to tests.

### 5.2 Large branch

Set `U_0=exp(Re x)`, `B_0=B_E*U_0`, `V=1/abs(x)` and take the inverse bound `C`
from (7). For each order use the actual upward recurrence
`phi_k=(phi_(k-1)-1/(k-1)!)/x`. With

\[
 W=U_{k-1}+c_{k-1},\quad
 D=(1+\alpha)(B_{k-1}+C_{k-1})+\alpha W,
\]

set `B_k=P(W,D,V,C)` and
`U_k=min(exp(max(Re x,0))/k!,W*V)`. This includes the cancellation in the
numerator. After either branch, add the argument-error bound following (13).
The difference of two branch evaluations at the same join argument is bounded
by the sum of their separately derived absolute bounds.

The phase is reduced only inside `exp(x)`. Phi functions and their inverse
argument are not periodic and retain the original argument.

## 6. Whole-run admission inputs owned by the session

The master admission code must establish all of these before selecting this path:

1. Bound every exponent argument's real part and phase by the domain above,
   including the backward propagator and Adams transport of retained history.
   Replay uses its actual longest prescribed step, not `dz_max`. An adaptive run
   uses its admitted maximum. A fresh Adams span is at most `(q_max+1)*h_max`;
   a resume adds the actual retained sample age and caps by the represented
   distance from the oldest sample to the run end.
2. Bound the represented operator on complete intervals, including interior
   extrema. Chebyshev absolute-coefficient bounds are conservative certificates;
   a maximum of sampled values is not one.
3. Independently bound input/assembly error in the real-plus-imaginary norm by
   `u/8`. Constant exponents assembled on the host and split once have split
   error at most `u^2*(abs(real)+abs(imag))`, with host assembly error additional.
4. For a device sum of `n` real weights times complex terms, let
   `S=sum(abs(weight)*(abs(real(term))+abs(imag(term))))`. Assuming each input
   split has relative error at most `u^2`, a conservative assembled-word bound is

   `((1+10u^2)*(1+A)^(n-1)-1)*S + 10n*(1+A)^(n-1)*eta`,

   plus the independently propagated host/table-evaluation error. Cancellation
   of the net exponent does not remove this contribution bound. A profile
   evaluated by a recurrence needs that recurrence's own condition bound too.
5. Establish the arithmetic model on the selected device and preserve the stored
   source identity, noise policy and same-precision history compatibility.
   A portable history's actual sample coordinates, not snapped landing planes,
   determine its retained age and coefficient arguments.

Failure names the real-range, phase-range, assembly-budget or arithmetic-model
condition. Where available, it can name `wide="binary64"` as the alternative.
No single-precision noise or changed physics default follows from this document.

## 7. Leaf API and qualification

Host preparation:

```julia
check_df32_function_argument(z::Complex{DF32}; input_error::Float64=0.0)::Nothing
```

It checks finite normalized words, the exact stored-component bounds and the
supplied finite nonnegative input-error bound. It does not discover that bound,
qualify hardware, or certify a future adaptive trajectory. Construction remains
through the existing `prepare_df32` scalar functions; session admission owns the
whole-run certificate.

Protected allocation-free kernels, under that certificate:

```julia
_df32_reduce_phase(theta::DF32)::DF32
_df32_exp(z::Complex{DF32})::Complex{DF32}
_df32_sincos(theta::DF32)::Tuple{Float32,Float32} # sine, cosine
_df32_exp_pair(z::Complex{DF32})::Tuple{ComplexF32,ComplexF32}
_df32_phi_values(z::Complex{DF32}, ::Val{N}) # tuple, orders 0:N
```

No Base transcendental overload or runtime admission is added. New helpers remain
separate from the reviewed scalar algorithms. Tests must check directed constant
splits, turn boundaries and the selector allowance, exact zeros, both join paths,
phi zeros, cancellation, range refusals, subnormal intermediate witnesses,
operation-derived bounds against MPFR and zero warmed allocations. CPU generated
code must retain Float32 operations and explicit FMA without unsafe FP flags.
Numerical tests are witnesses to these bounds, not their derivation; device
qualification and complete propagation budgets remain separate requirements.

## Primary sources

- [J17] M. Joldes, J.-M. Muller and V. Popescu, *Tight and Rigorous Error Bounds
  for Basic Building Blocks of Double-Word Arithmetic*, ACM TOMS 44(2), article 15
  (2017), [DOI](https://doi.org/10.1145/3121432),
  [complete author text](https://hal.science/hal-01351529/document).
- [M22] J.-M. Muller and L. Rideau, *Formalization of Double-Word Arithmetic, and
  Comments on "Tight and Rigorous Error Bounds for Basic Building Blocks of
  Double-Word Arithmetic"*, ACM TOMS 48(1), article 9 (2022),
  [DOI](https://doi.org/10.1145/3484514),
  [complete author manuscript](https://hal.science/hal-02972245/document),
  [formal supplement](https://www-sop.inria.fr/members/Laurence.Rideau/DW_arithmetic-submitted_paper_release.zip).
- [MSL] Apple, *Metal Shading Language Specification* 4.1, dated 2026-06-04,
  sections 8.1--8.5 and Table 8.1,
  [complete specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Both complete arithmetic papers and the relevant supplement definitions were
checked. The supplement was not recompiled. The finite-underflow extensions and
function bounds in this document are new derivations requiring independent review.
