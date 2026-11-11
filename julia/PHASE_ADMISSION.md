# Represented phase operands before DF32 admission

`solprop.backends.julia._phase_bounds` supplies conservative preparation-time operand
analysis. It does **not** admit DF32, choose a wide arithmetic implementation, initialize
Julia, or certify a propagator error. The current production operator assembly remains
binary64. A finite result from this analyzer is not permission to narrow that assembly.

This distinction follows [SEP 48](../docs/dev/seps/sep-0048-julia-backend.md) and the
[protected-function derivation](DF32_BOUNDS.md). The latter proves conditional scalar
function bounds for already-prepared arguments. It does not prove the rounding error
of coordinates, represented coefficients, cumulative subtraction, expansion contraction,
or transported Adams compensation.

## Inputs, ownership and output

`phase_spans(run, start, history=...)` reads an already-resolved `RunTransfer` and the
actual accepted anchor. `operator_envelope(operator, spans)` reads a constant generator,
a full cumulative Chebyshev representation, or fixed complex terms with real cumulative
weights. `phase_operands(...)` combines those results and records missing proof inputs.
All positions and distances are metres; generator and sampled-rate bounds are inverse
metres; exponents and cumulative operator bounds are dimensionless.

The records contain independent absolute real/imaginary bounds over **every computed
cell**, including inactive bins. They retain maximum polynomial degree and term count,
full represented values and derivatives, represented jumps, endpoint differences, and
uncancelled endpoint/expansion contributions. They deliberately contain no `approved`
field. An unavailable transported-argument bound is `None` with an explicit reason.

The caller must prevent mutation during analysis. These immutable scalar summaries do
not retain, hash or freeze their input arrays. Any future certificate must bind the
actual prepared arrays, precision/scaling choices, history and arithmetic graph to its
result. Reusing a summary after changing those inputs is unsupported.

An optional snapshot means **the already-compatible history that will be restored**.
The analyzer requires the shared history key, actual accepted anchor, valid current
order/rejection metadata, positive proposal, strictly increasing oldest-first actual
sample coordinates, unique in-range slots and accepted/predicted sample kinds. It uses
the shared newest-sample proximity rule without snapping a sample to the accepted
anchor. Full source stores have shape `(order_max+2, 1, n_modes, n_pol, n_omega)`;
compensated histories also require paired full state/rate stores of that shape. All
rows, including inactive slots, must be finite. Unknown row names refuse. State and
operator cell counts must agree. It reads the actual selected stored rates rather
than replacing them with derivatives of the current representation.

These checks supplement, rather than replace, the shared schema and restore policy.
Precision compatibility, scaling, source identity, history-key construction, caller
choice of field-only restart, cumulative ledgers, controller compatibility and input
noise remain outside this analyzer. No history is silently discarded. Binary32 states
and binary64 stored rates can be analyzed without claiming that this combination is
admitted. Noise is never applied by the analyzer.

## Actual march geometry

Let `c = max(1, max(abs(tableau.nodes)))`. The current embedded RK graph only flows
between a step's anchor and each stage, and between that anchor and its endpoint; it
does not flow directly between arbitrary pairs of stages. Thus `c*h` bounds its exact
stage distances. No fixed stage count or restriction of nodes to `[0,1]` is assumed.
The four forward flows of the low-storage method fit the same conservative bound.

For replay, the analyzer uses the **remaining supplied lengths**, not `dz_max`. It
checks authoritative accumulated edges under the shared replay allowance, locates the
actual resumed anchor, and bounds each binary64 multiply-then-add stage coordinate
using exact rational products and sums followed by directed rounding. Both the raw
`z+h` and the recorded destination are retained in the coordinate envelope. A resumed
anchor within the shared allowance is used as supplied for its first step. A replay
that does not cover the start/end or crosses the requested end refuses.

For adaptive execution, the shared landing rule caps even a stretched step by
`dz_max`. A forced landing may be arbitrarily shorter than `dz_min`, so no positive
actual minimum step is asserted. The analyzer bounds endpoint arithmetic using

\[
 |\operatorname{RN}_{64}(x)-x|\le u_{64}|x|+\eta_{64},\qquad
 u_{64}=2^{-53},\quad\eta_{64}=2^{-1075}.
\]

For a start `a` and end `b`, subtraction contributes at most
`e_gap = u64*(b-a)+eta64`. A raw endpoint is bounded above by
`b+e_gap+u64*abs(b+e_gap)+eta64`. For nodes outside `[0,1]`, the separate product and
sum errors are added to the exact extrema over anchors in `[a,b]` and steps no larger
than the ceiling. Nodes inside `[0,1]` are bounded by monotonicity between the anchor
and the raw endpoint. All these computations are rational before their outward
binary64 conversion. Overflow of a required finite bound refuses.

Let `s` include the shared landing/replay allowance, `A` be the actual oldest retained
sample's age at start, and `q` the configured maximum Adams order. A conservative
history distance is

\[
 H=\min\{z_{\max}-z_{\min},\;A+(q+1)(c h_{\max}+2s)\}.
\]

For a method without retained Adams history, use one step instead of `q+1`. The
complete coordinate envelope includes old sample coordinates and endpoint rounding;
it may extend beyond the requested run. It must remain inside each table's explicitly
admitted span, including that table's own outer-edge allowance. The `q+1` factor covers
the retained extra oldest sample and subsequent accepted samples before it is retired.
Rejections do not advance this history. Hard-kink resets can only shorten its age.

When replay supplies a positive `h_min`, the exact normalized-history multiplier obeys

\[
 |1-(z_j-z)/h|\le 1+H/h_{\min}.
\]

This is also sufficient for dense transport `|theta-(z_j-z)/h|`, `0<=theta<=1`.
The rounding error of forming that multiplier is a separate obligation. Adaptive
execution reports no multiplier bound because its actual `h_min` is unknown.

## Complete represented intervals

For one component on interval `[a_i,b_i]`, write

\[
 P_i(z)=\sum_{k=0}^{d}p_{ik}T_k(x),\qquad
 x=(2z-a_i-b_i)/(b_i-a_i).
\]

The exact represented polynomial is the target here, not a rounded Clenshaw result.
For the part of this interval that can be read, set `X=max(1, max(abs(x)))`, including
any permitted outer-edge extrapolation. The identities
`T_k(cos(theta))=cos(k*theta)`, the corresponding second-kind representation, and
`T'_k=k U_(k-1)` give

\[
 |P_i|\le \sum_k |p_{ik}|T_k(X),\qquad
 |P'_i|\le\sum_{k\ge1}|p_{ik}|\frac{2kU_{k-1}(X)}{b_i-a_i}.
\]

For `|x|<=1`, the factors are `1` and `2*k*k/(b_i-a_i)`. For `X>=1`, the hyperbolic
representations make the positive factors monotone. The analyzer evaluates their
three-term recurrences with exact rational arithmetic, then rounds upward. These
bounds cover interior extrema; endpoint sampling would miss, for example, `1-T_2`.
The identities and recurrences were checked against NIST DLMF
[equations 18.5.1–2](https://dlmf.nist.gov/18.5#i),
[Table 18.9.1](https://dlmf.nist.gov/18.9#T1), and
[equation 18.9.21](https://dlmf.nist.gov/18.9.E21).

At a represented boundary the two polynomial limits need not coincide exactly.
For each cell/component, compute the actual coefficient-defined jump

\[
 J_i=\left|\sum_k p_{ik}-\sum_k(-1)^k p_{i+1,k}\right|
\]

using exact rational cancellation, and let `J` sum all jumps in the required span.
If `V` and `R` are the maximum value and derivative bounds across that span, any two
represented endpoints separated by at most `H` satisfy

\[
 |P(z_1)-P(z_0)|\le \min(2V,\;H R+J).
\]

This is the piecewise fundamental theorem of calculus plus all possible intervening
jumps. Both limits of a boundary are included. Physical continuity does not justify
setting `J=0`; exact represented continuity does. The arithmetic subtraction still
has operands bounded by `2V`, even when `H R+J` is very small. For a stored constant
generator `D`, there are no cumulative operands: the exact target is `(z1-z0)*D`.

For an expansion `Phi(z)=sum_r T_r*w_r(z)` with real weights, apply the table bounds
to every `w_r` before combining. For each real/imaginary component, the value, rate,
jump and difference bounds are sums of `abs(component(T_r))` times the respective
weight bounds. The uncancelled assembly operand is
`sum_r abs(component(T_r)) * 2*V_r`, even when terms cancel in the final phase.
The retained coefficient magnitude is that of the fixed terms; the future conversion
proof must also inspect the original scalar weight coefficients. No complex-weight
extension is implied.

## Outward calculation versus assembly error

Scalar geometry, jumps and Chebyshev growth factors are exact rational calculations
on the supplied binary values before directed binary64 conversion. Positive array
sums/products use round-to-nearest binary64 followed by one `nextafter(...,+Inf)`
after **each** operation. This encloses the exact positive result, including a product
that rounds to zero under gradual underflow. Exact structural zero products/sums
remain zero. Maxima and minima of already-valid bounds preserve validity. Nonfinite
bounds refuse; no warning, infinity sentinel or FTZ policy substitutes for a bound.

These are bounds on exact represented operands. They are not estimates of how
accurately a future DF32 kernel evaluates them. In particular, cumulative Clenshaw
needs an intermediate-register bound, coordinate-map and coefficient-conversion
errors, explicit FMA/contraction choices, an underflow allowance and endpoint
subtraction error. Expansion assembly additionally needs term conversion and the
specified contraction order. A small net phase supplies none of those proofs.

## Transported compensation and remaining certificate inputs

The implemented Adams numerical formula uses

\[
 L=\Phi(z+h)-\Phi(z),\quad t_j=(z_j-z)/h,\quad
 e^{(1-t_j)L},\quad D_{\rm ref}=L/h,\quad
 D_j-D_{\rm ref}.
\]

It does **not** generally replace `(1-t_j)*L` by `Phi(z+h)-Phi(z_j)` for a varying
operator. A history-distance-times-derivative envelope therefore cannot certify this
arithmetic graph. A represented jump contributes `J/h`; cumulative evaluation error
also becomes `error(L)/h`. Either can grow without bound as a landing step shrinks.
The returned transported argument is the step-argument bound multiplied by the known
replay multiplier, or `None` for adaptive compensation. Constant operators and a
structurally absent source do not use compensated history. Stored selected rates
are reported independently so a later proof cannot silently substitute a new value.

Before enabling a DF32 route, a separate assembly certificate must establish:

1. The immutable input/scaling identity and exact graph for every exponent, derivative,
   normalized node, dense fraction and compensation operation. The existing binary64
   graph is context, not the promised wide graph.
2. Every intermediate range, supported conversion and gradual-underflow term for that
   graph. A constant-distance multiplication may need a different proof from a pair
   of cumulative Clenshaw evaluations and a difference of scalar weights.
3. Absolute argument error propagated through all operations, including cancellation,
   represented jumps, actual replay lengths, old retained samples and stored-rate
   provenance. For arbitrary short adaptive landings, a stable difference/integral
   graph or a justified additional condition is necessary; `dz_min` is insufficient.
4. The protected functions' actual domains with margin for the propagated error.
   The currently proved leaf requirements are `abs(Re)<=16`, `abs(Im)<=2^20`, and a
   separate argument error no larger than `u/8`, `u=2^-24`, under the hypotheses in
   [DF32_BOUNDS.md](DF32_BOUNDS.md). These are downstream requirements, not a pass
   condition implemented by this analyzer. Phi orders and joining branches need the
   same complete assembled-argument proof.
5. The selected device's rounding, explicit FMA/division and gradual-underflow behavior,
   plus emitted-code checks. CPU evidence does not qualify Metal, CUDA or ROCm.

The targeted tests use independent exact defining-polynomial recurrences, rational
endpoint/jump and directed-rounding witnesses, cancellation-heavy expansions, actual
replay schedules, resumed sample coordinates and malformed full stores. They test
these operand inequalities and refusals. They do not assert a whole-expression error
budget or public single-precision propagation qualification.
