# Certified DF32 Adams execution

This internal CPU package implements landed exponential Adams for binary32 state
and DF32 complete factors. It does not admit public DF32 execution, devices,
separable spatial operators, tabulation or dense output. The numerical assumptions
are those of [WIDE_ASSEMBLY.md](WIDE_ASSEMBLY.md) and
[DF32_BOUNDS.md](DF32_BOUNDS.md); state scaling follows
[PRECISION.md](PRECISION.md). Binary64-wide Adams retains its existing graph.

## Frozen interfaces

```julia
DF32AdamsFlow(execution, operator, state::AbstractMatrix{ComplexF32},
    spec::WideStartSpec, policy::AdamsControlPolicy;
    history::Union{Nothing,WideStartHistory}=nothing, tabulate=false)

DF32AdamsSeed(slot::Int, position::Float64)

prepare_df32_adams_trial!(flow, history::AdamsHistory, stencil::AdamsStencil,
    site::DF32TrialSite; seed::Union{Nothing,DF32AdamsSeed}=nothing)

adams_trial!(work, state, position, step, history, stencil, source!,
    trial::DF32AdamsTrial, rate!, execution; source_absent=false)
```

The token binds one owner, history, active generation, actual supplied step, raw
origin and canonical destination, actual replay cursor, selected orders and every
sample coordinate/kind/slot/generation. A virtual accepted seed lets startup and
restart certify the complete first trial before invoking its source. The same
policy selects landing and stencil; no Python code or alternate controller does so.

## Targets and transaction

Write `z1=fl64(z0+h)`, `x=Phi(z1)-Phi(z0)` and `Q=x/h`. The supplied `h`, the
represented endpoint distance and canonical landing stay distinct. For actual
sample coordinate `zj`, the source weight is
`h*sum(c[j,k]*k!*phi[k+1](x))`. Transported compensation is the complete factor
`b[j]*exp((h-(zj-z0))*Q)*(h*D[j]-x)`, multiplied once by its binary32 sampled state.
The stored rate `D[j]` keeps its explicit prior-error radius and exact knot side.
Neither a small final phase nor a cancellation-rounded zero proves intermediate
range. Every actual exponential/phi argument retains the reviewed exact word and
assembly-error limits.

Two coefficient/operation banks stage preparation. Failure changes only inactive
scratch, preserving the active token, numerical work, source counters, accepted
field/history and live rate provenance. A successful precommit check reserves all
generation updates before the nonthrowing provenance publication. The existing
portable source/state/rate schema is unchanged and remains raw scaled storage.

All host certificate data and all per-slot raw/rate/error matrices are CPU-owned
and counted, including adopted restored records. DF32 exponent/moment buffers
replace the same-size binary64 work buffers; this does not remove the additional
history-provenance cost. No complete-factor array cache or session-lifetime global
cache is introduced. Warmed advancement must allocate zero bytes.

The complete-factor derivation, explicit arithmetic bounds, storage inventory and
qualification evidence are recorded below as the implementation is qualified.

## Complete-factor graph and outward error recurrence

All statements here are conditional on IEEE round-to-nearest binary32/binary64,
finite protected intermediate operations and gradual underflow. They reuse the
primitive proofs and the six primary algorithms cited in `DF32_BOUNDS.md`; they do
not extend those proofs to a device with a different underflow model. Set
`u=2^-24`, `eta=2^-150`. The per-cell host certificate is deliberately expensive:
it traverses the represented graph, every required phi value and every complete
factor. It is an explicit CPU range/error guard, not a performance qualification.

### Represented coordinates, rates and coefficients

The paired divided-difference Clenshaw graph from `WIDE_ASSEMBLY.md` evaluates
`Q=(Phi(z1)-Phi(z0))/h`. Its scalar bundle retains the rounding of `z1`, actual
endpoint separation, supplied `h`, normalized coordinates and cross-knot jumps.
It is used directly; no assembled binary64 exponent is converted to DF32.
The exact target of the linear factor is the represented cumulative difference
at the *supplied binary64 endpoints*. Canonical delivery coordinates do not replace
those endpoints. The transport scalar retains the error of both subtractions in
`h-(zj-z0)`. Old sample positions, including a slightly future newest sample or
an old restored sample, are checked against the actual start geometry.

The host coefficient recurrence is the existing direct Lagrange product graph.
It propagates a radius through actual-coordinate subtraction, division by `h`,
node differences, polynomial products, product denominators and integration.
For division of `(a,ra)` by `(b,rb)`, require a strictly positive downward bound
`d=abs(b)-rb`; the inherited radius includes
`(ra+(abs(fl(a/b))+rounding)*rb)/d`. `WIDE_ASSEMBLY.md` derives the host rounding
term, including binary64 subnormals. A denominator interval containing zero or
an unrepresentable host coefficient/radius refuses the trial by name. Integration
uses the exact positive integer divisors 1 through 7. All host radii are retained
when splitting coefficients; no caller-supplied coefficient is treated as exact.

For a restored rate, `reconstruct_wide_rate` first encloses the represented
one-sided derivative and adds the discrepancy of the *actual saved binary64
value*. The DF32 conversion adds its own error. The numerical complete factor
uses that saved value, not a replacement derivative. The live record retains its
position, accepted/predicted side, origin-wide provenance and generation. New
seed/endpoint records similarly retain their actual binary64 stored value and its
conversion/evaluation radius. The normalized rate representation below is built
from that raw value and the full inherited radius.

### Normalized products and gradual underflow

A complete scalar is `(m,e)` with complex DF32 mantissa `m` and integer exponent
`e`, representing `2^e*m`. The supplied binary64 `h`, every coefficient and each
actual saved rate are normalized *before* splitting into DF32 words. This matters:
forming an unscaled tiny DF32 step or rate first can erase a contribution later
rescued by another factor. Each complex rate uses one common exponent; component
split residuals are carried independently.

After each complex product/addition the high-word maximum is normalized with
`frexp`. For a component scaling by `2^k`, round each word and renormalize with the
protected Fast2Sum. Binary scaling of a normal result is exact. A subnormal word
loses at most `eta`; the pair contributes at most `2eta`, and the implementation
allows `3eta` to include renormalization. Therefore

```
r_scaled <= upward(2^k*r_original) + 3eta.
```

Every positive binary64 radius scaling is itself rounded upward, including when
it underflows. Relative exponent alignment in a sum uses the same formula; a lost
small addend is covered rather than assumed zero. Normalized complex operands have
component magnitudes at most one; component sums/products stay below four before
renormalization. The primitive finite-range hypotheses are therefore checked
locally. Integer exponents cannot approach the machine limit: the order is at most
six, a host operand contributes at most 1074 binary exponents, and this fixed graph
contains fewer than 256 successive scaling/product operations. Even the loose
`256*(1074+150)` bound fits a 32-bit signed integer. No dynamic expression tree or
arbitrary number of factors is admitted.

For checked arithmetic, the existing `_WideBall` additions/products carry all
input radii and the primitive local errors. Numerical closures execute the same
DF32 graph without the host error records. A radius that cannot be represented in
binary64 is a named refusal even if a more sophisticated scaled-radius proof
might certify the result. This is a declared conservative domain limitation.

This factor representation does not widen the prerequisite `WideStartBinding` and
`WideRate` domains. Original operator inputs, the represented divided difference
and raw-rate words still obey their finite intermediate bound, `floatmax(Float32)/32`.
For example, the complete-factor helper can represent a supplied `1e-300` step
times a `1e300` rate, but a restored session with that rate refuses in the existing
rate-binding preparation. A source call cannot occur before that refusal. No
binary64 numerical fallback expands this first package's domain.

Final state coefficients are narrowed only after the complete product/sum. Dropping
the low word and scaling the high word has component error bounded by

```
2^e*r + u*2^e*(abs(m.hi)+abs(m.lo)) + 2^-148.
```

The absolute term covers the final subnormal rounding and a possible double-rounding
boundary. A nonfinite ComplexF32 result refuses before any state/source write.
Errors are absolute; cancellation can make the relative error arbitrarily large.

### Absolute phi and exponential errors

The coefficient calculation needs orders zero through seven. `_da_phi` evaluates
the same protected functions as `double_float_functions.jl`, and implements the
absolute recurrences of `DF32_BOUNDS.md` section 5. For an exact zero argument the
reciprocal-factorial error is at most `2u^2/k!`. In the small branch the reduced
argument has magnitude at most one half, the 24-term tail is below `2^-106`, and
at most three doubling recurrences are required. The certificate uses the
conservative analytic `exp(1/2)<2`, followed by the declared product/addition
recurrences; it includes the additive underflow allowances at every doubling.
For the large branch a certified component lower bound, at least 2.98, bounds
`1/abs(x)` without assuming a binary64 square-root accuracy bound. The restricted
inverse error is `32u^2/abs(x)+32*(1+2^21)*eta`. Forward phi recurrence retains
absolute errors through subtraction, including near zeros and the overlapping join.

If argument error is `b`, the defining integral gives, for k>=1,

```
abs(phi_k(x+delta)-phi_k(x)) <= b*exp(max(0,Re(x)+b))/(k+1)!.
```

For k=0 use the same numerator without a factorial. The exact-word range and
`b<=2^-27` function admission remain mandatory. The exponential local relative
allowance is `8192u^2+2^40*eta`, as already proved in the function package. To bound
positive exponentials in the host certificate, use
`exp(t)<=2^ceil(3t/2)` for t>0 (since `log(2)>2/3`), and one otherwise. This avoids
borrowing an unproved host libm accuracy bound. All non-exact positive sums,
products and quotients in these bounds round outward.

### Complete source, compensation and defect factors

With exact Lagrange coefficients `c[j,k]`, the complete source coefficient is
`h*sum(c[j,k]*k!*phi_(k+1)(x))`. Coefficients and `h` retain their separate exponents
through the entire sum/product. The corresponding absolute factor certificate
includes coefficient formation, represented phase, phi recurrence, scaled sums
and final binary32 narrowing. The corrector's endpoint source is sampled at the
actual raw endpoint and at the *predictor* state.

Compensation evaluates

```
b[j] * exp((h-(zj-z0))*Q) * (h * (D[j]-Q)).
```

This is exactly the declared target `b[j]*exp(...)*(h*D[j]-x)` since `x=h*Q`;
it changes the floating-point evaluation graph relative to binary64-wide Adams.
The normalized saved rate and Q are subtracted before multiplication by normalized
h, retaining their full radii. The bound therefore retains the uncancelled rate
contributions even if the computed difference is zero. Every actual transport
argument is certified even when a weight/difference rounds to zero; a zero cannot
bypass function-domain admission.

Factored defects use the same divided-difference decomposition as binary32-wide
Adams. Write `w_j=2^e*wn_j`; narrow `wn_j` to binary32, sum `wn_j*N_j` in state
precision, and multiply once by the complete DF32 integral with the retained `2^e`.
This avoids unrepresentable individual divided-difference weights. The certificate
separately reports the largest normalized-weight error and complete-factor error.
For a state-error bound those quantities must be multiplied by the actual absolute
sample contributions and combined with the state summation error. A factor
certificate alone is not a global propagation error estimate or adaptive accuracy
claim.

## Ownership, publication and retained storage

A flow owns its original immutable operator plus one `WideStartBinding`. The latter
owns copied original operator/error/DF32 arrays; it is not a problem-wide cache.
Each live compensated slot retains three complex matrices of 16 bytes per cell:
raw binary64 values, DF32 words and binary64 component radii. It additionally
retains a normalized complex DF32/integer-exponent matrix: 24 bytes per cell on a
64-bit host. Thus each slot costs 72 bytes per cell, apart from scalar metadata.
Adopted restored records share their three original matrices with the binding;
they are counted once, not copied again. A concrete tuple makes these nested
matrices visible to the existing retained-storage traversal.

Both banks each retain separate candidate seed and endpoint rate records of the
same four-matrix layout. These are required to leave live provenance untouched
on a failed or rejected trial. For maximum order q, each coefficient record has
three `(q+1)^2` matrices, three `3*(q+1)` matrices and five `(q+1)` vectors:
`3*(q+1)^2+14*(q+1)` elements total. Each bank retains host values/radii,
scaled checked values/radii and numerical DF32/exponents separately, plus a bounded
stencil, positions/kinds/generations, q+1 full scalar bundles and their numerical
projections, normalized defect weights and exponents. Actual element sizes are
reported by `sizeof`; tests bind every retained matrix to the session ledger.
The work exponent and moment arrays use 16-byte complex DF32 cells, replacing
same-size binary64 buffers. State/source/defect/norm buffers retain their existing
state precision. There is no field-sized complete-factor cache. Completed flows
retain the original immutable operator reference but allocate no wide-operator,
rate or coefficient banks and execute no fabricated trial.

Preparation checks shapes, complete retained-owner aliases, actual history and
rate generations before numerical writes. Only the inactive bank changes while
certifying. Successful publication binds the token to the history owner, site,
selected slots/kinds/coordinates and generation. A refused plan preserves the last
active token. A successful later plan or acceptance invalidates an old token.
The seed's live generation capacity is checked before its source/accounting; an
acceptance precommit checks destination capacity and payload before field/history
writes. The subsequent owned-array copies and generation publication are
nonthrowing under the synchronous owner contract. A source failure retains actual
call accounting and the existing poisoned-session semantics.

Restored portable history remains raw: source/sample-state arrays are ComplexF32,
accepted state transport widens raw values exactly, and rates stay ComplexF64.
No physical rescaling or schema change occurs in this package. Dense retain on a
DF32 workspace refuses, and dense/spatial/tabulate/device requests refuse during
preparation. Those are explicit first-package limits, not completed Phase 3 work.

## Python session and history attachment

The internal Julia owner selects `DF32AdamsFlow` for binary32 landed Adams with
`wide="df32"`. The PythonCall factory supplies the same resolved policy object to the
flow and session, plus the original operator, launch scales and actual start. Python
neither forms complete factors nor drives trials. Public binary32 selection stays
closed pending the full qualification matrix.

Before phase binding, Python detaches candidate portable arrays and Julia applies
the same scalar, layout and numerical compatibility checks used by restoration.
The shared validator preserves the existing restore order: layout, retained alias,
then numerical values. An incompatible candidate produces the ordinary field-reset
reason and contributes no coordinates or rates to the phase domain. An unexpected
bridge failure raises; it cannot masquerade as a field reset.

The binding owns another immutable snapshot of the selected data. The actual restore
uses that selection, rather than rereading caller-owned arrays. A disagreement between
preflight and restoration is an error. Stored zero-based slot indices, accepted/predicted
sample identities and every raw rate row retain their original meaning. The portable
schema carries no previous wide type, so the adapter records that provenance as unknown
for every import; it never establishes zero prior error.

The history coordinate enclosure includes accepted landing slack and actual raw
predictor coordinates, even when one is slightly beyond the requested endpoint.
A completed start validates and restores raw history without preparing an operator,
rate certificates or factor banks. It performs no new numerical trial or source call.
The active-start rate-provenance requirements remain unchanged.

Managed witnesses in `test_cpu_df32_adams.py` cover landed orders 1, 3 and 6,
adaptive runs, plain/compensated history, binary64-wide and NumPy binary32 continuation,
lossless raw stores, completed starts and invalid-history resets. Ordinary replay and
cross-wide comparisons use SEP 33 accuracy envelopes with independent binary64
step-size refinement; transfer bytes and logical source charges remain structural.

## Qualification scope

The focused tests use exact-rational Vandermonde bases and independent 256-bit
defining series/integrals for all six Adams orders. Each complete-factor allowance
is fixed before comparison. An alternating-source history makes every scored
high-order defect nonzero; its contribution/error derivation must certify a bound
smaller than one quarter of the defect. The predictor/corrector tests propagate
source sensitivity separately from complete-factor errors. Rescued large/tiny
factor operands are distinguished from the narrower admitted binding domain.

Actual eight-step plain/compensated and short adaptive sessions test work counts,
rejections, restarts, raw/canonical replay coordinates, sampled-rate discrepancies,
generation exhaustion, stale tokens and offset-buffer aliases. A scaled four-point
FFT/Kerr example uses an independent MPFR PEC recurrence and an explicitly stated
finite FFT/twiddle error hypothesis; it does not infer transform accuracy from an
operation count. The unchanged binary64 and binary32-wide64 Adams, dense and service
tests exercise the default hooks. Warmed advancement, factor preparation and
publication retain zero allocation on the tested CPU task budgets.

Emitted numerical helper code is checked for binary32 arithmetic without binary64
extension, unsafe floating-point flags, vendor transcendental calls or GC pool
allocation. This is evidence for those CPU helper specializations, not a device or
whole-call-graph claim. The CPU certificate still evaluates every retained cell and
factor before every trial; no throughput, public-admission or full-run accuracy
qualification follows from these short numerical witnesses.
