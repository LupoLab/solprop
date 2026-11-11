# Certified CPU DF32 dense observations

The RK leaf retains the existing continuous extension for binary32 state and
certifies every represented phase in DF32. It is an internal CPU implementation,
not public single-precision, spatial, Adams-dense or device admission. The numerical
hypotheses and primitive/function bounds are those of
[WIDE_ASSEMBLY.md](WIDE_ASSEMBLY.md), [DF32_BOUNDS.md](DF32_BOUNDS.md) and
[DF32_EXECUTION.md](DF32_EXECUTION.md): RN-even arithmetic, explicit correctly
rounded FMA, finite protected intermediates and gradual underflow. No CPU evidence
establishes those hypotheses on a vendor device.

## Construction and retained ownership

```julia
RungeKuttaObservationWork(execution, state, tableau, dense_weights,
    flow::DF32LinearFlow; arithmetic)
```

The ordinary work record and its coefficient algorithm are shared with the
binary64-wide path. `state` is already scaled `ComplexF32`, `arithmetic` retains
its original launch exponents, and `tableau` must equal the immutable translated
method bound by the flow. The dense weights remain translated input; Julia does
not select them from a stage count. Replay plus interpolation, per-step split
collars, other methods and unqualified operators keep their named refusals.

The work's phase context borrows the exact immutable `WideStartBinding` created
for that flow. It holds two independent scalar operation records and a retention
generation. It borrows neither the mutable trial banks nor a trial token. Reusing
those banks cannot invalidate the retained accepted window. Direct mutation of
prepared operator, binding or work arrays is outside the ownership contract.

The bounded numerical arrays are unchanged: five `ComplexF32` coefficient fields,
the exact finished endpoint, the observed field, one `ComplexF32` factor field,
one unpacked stiff mask, empty expansion-weight storage and one finite-check
integer per column. There is no additional full-field wide phase array. Each
scalar operation record retains its DF32 scalar bundle, host formation radii,
aggregate component errors, intermediate bound, coordinates and generation.
`runge_kutta_observation_memory` lists the owned arrays; shared binding arrays
remain counted once by the containing session's storage traversal. Scalar
metadata and object headers are separate from numerical array payload.

The shared constructor hooks are `_rk_dense_admit(flow,state)` and
`_rk_dense_prepare(execution,state,tableau,dense_weights,flow; arithmetic)`.
Their `LinearFlow` defaults preserve the existing binary64-wide path. The
scheduler still owns admission of plane policy, endpoint commit, delivery,
warning continuation, snapshot priority and portable-history eligibility.

## Actual intervals and certificate transaction

For retained anchor `z`, supplied positive step `h` and raw endpoint
`r=fl64(z+h)`, the full-step branch uses

\[
 X=\Phi(r)-\Phi(z).
\]

For an actual interior requested coordinate `p`, the flow uses

\[
 X_p=\Phi(p)-\Phi(z).
\]

All supplied binary64 coordinates and operator data are exact operands of these
mathematical targets. The canonical landing is retained separately; it never
replaces `r` in the numerical expression. A query calls the existing paired
Clenshaw/expansion certificate with its actual `z,p` endpoints and the retained
`h`. It does not rescale the full-step phase, assume monotonicity or use an
endpoint maximum in place of the represented interior value. The correlated
same-interval graph and the cross-interval cancellation limits therefore remain
exactly those derived in `WIDE_ASSEMBLY.md`.

Every queried argument retains the existing exact-word limits
`abs(Re)<=16`, `abs(Im)<=2^20` and summed assembly/input error at most `2^-27`.
An interior extremum may fail despite a certified full step. That is a named
observation failure before any factor or observed field write; the accepted
session endpoint, source work and history remain unchanged.

Retention checks full-step phase and branch admission before changing readiness,
arrays or the previously retained scalar records. Only after success does it
publish the replacement scalar record and enter the ordinary coefficient/copy
path. Its existing nonfinite-input/coefficient failures still leave the new
window unavailable. A query similarly certifies into local scalar storage before
publishing its query record and filling factors. Certificate refusal preserves
the previous observed field and successful query record. No query advances the
flow's trial generation. Retention generation exhaustion refuses before mutation.

## Certified stiff-bin decision

The existing predicate is `abs(real(X))>1`. Let the checked real phase be a
normalized pair `v=hi+lo` with outward absolute radius `e`. Summing its words in
host binary64 produces `(c,ec)` using the existing host-error recurrence, then

\[
 \ell=\operatorname{down}(c-(e+e_c)),\qquad
 u=\operatorname{up}(c+(e+e_c)).
\]

Thus the exact represented real phase lies in `[ell,u]`, including a low word
invisible to a rounded binary64 sum. The branch is certified nonstiff if
`-1<=ell` and `u<=1`, and stiff if `u<-1` or `ell>1`. Otherwise retention refuses
with `DF32 dense stiff-branch certificate straddles ±1`. It does not choose a
binary64 fallback or change the strict threshold. Exact threshold inputs and
sufficiently close neighbors can conservatively refuse; adjacent binary32
values in the qualified fixtures are separated enough to admit. The numerical
mask then compares the same checked DF32 words exactly against one.

This refusal is a conservative internal domain boundary, not a numerical-policy
change. If a required physical case reaches it, a sharper certificate or a
justified bound for both branch results is needed. No tolerance is fitted to
avoid it.

## Shared polynomial and factor error

The accepted stage derivatives remain binary32 values in the anchor's interaction
picture. The existing ordered contractions retain

\[
\begin{aligned}
 r_1&=y_0,&r_2&=h\sum_j b_jk_j,&r_3&=hk_1-r_2,\\
 r_4&=r_2-hk_s-r_3,&r_5&=h\sum_j d_jk_j.
\end{aligned}
\]

Interior nonstiff bins use
`P=r1+theta*(r2+(1-theta)*(r3+theta*(r4+(1-theta)*r5)))`, where
`theta=(p-z)/h`; stiff bins use the retained anchor. This is the existing
DOPRI5 continuous-extension form documented in `runge_kutta_observation.jl`,
with the same binary32 conversion, contraction and stage-sum order. The DF32
change is confined to its represented linear factor.

The factor kernel repeats the certified DF32 assembly and uses the corrected
forward propagator. `DF32_BOUNDS.md` establishes relative factor error below
`5u32`, including the separate `u32/8` assembly allowance. A computed exact zero
phase gives an exact unit factor after certification. The existing unit-factor
branch skips multiplication. No library binary64 exponential or binary64
numerical phase is evaluated by this kernel. Host certificate arithmetic stays
binary64 as explicitly designed.

Retention pays one complete phase-certificate traversal and a second checked
traversal for the branch, followed by the DF32 mask fill. Each interior query
pays its own complete host certificate and DF32 factor fill. This explicit CPU
cost is not a throughput claim or a reusable uniform certificate for devices.

For the normal finite polynomial fixtures, fewer than 64 binary32 roundings lie
on a scalar coefficient/evaluation path, including translated-weight and step
conversion. `gamma128` additionally covers the final complex multiplication and
its certified factor error. The test forms an absolute majorant before every
signed stage contraction. If `B2,...,B5` bound the coefficient contributions and
`T,S` bound `abs(theta),abs(1-theta)`, use

\[
 M=|y_0|_1+T(B_2+S(B_3+T(B_4+SB_5))).
\]

Coordinate error is bounded separately by partial derivatives of the bivariate
polynomial in theta and its complement. The comparison allowance is
`exp(Re Xp)*(gamma128*M+(1+5u32)*Ecoordinate)` plus an explicit additive
subnormal allowance. To include all cross terms, write `c=sqrt(2)*gamma3` for
complex-product rounding. Since the polynomial's total degree is at most four
and `T,S` include their coordinate allowances, `Ecoordinate<=4M`. The coefficient
of `M` in the complete product bound is at most

```text
(1+5u32)*(1+c)*gamma64 + 5u32 + 5c*(1+5u32) < gamma128.
```

The last term includes product rounding acting on both the polynomial and its
coordinate error. Thus the stated allowance does not drop either cross term.
Stiff bins use the anchor majorant. These bounds compare
independent represented arithmetic, not an adaptive global-error guarantee.
Short session witnesses separately bound recurrence amplification with absolute
stage sums and the matched scalar method, while keeping exact accounting and
endpoint copies structural. No bitwise agreement across task budgets or yields
is required.

## Evidence and remaining scope

Focused tests cover constant, cumulative Chebyshev and cancelling expansion
operators, independent MPFR polynomial/phase evaluation, actual queried interior
extrema, supplied-step versus endpoint subtraction, tiny positive landings,
threshold refusal/preservation, signed endpoint zeros and both raw/canonical
rounding directions. They also cover trial-bank reuse, retained foreign aliases,
invalid query/generation preservation, bounded inventory, adaptive dense delivery,
rejected and source-failed trials, and warmed allocations at one/two CPU threads.
Emitted IR checks are scoped to the owned numerical assembly/factor specialization;
its host certificate is deliberately outside that strict binary32 kernel boundary.

Adams dense complete factors and selected rate-provenance retention are a separate
milestone. Public bridge admission and vendor qualification remain separate work.
