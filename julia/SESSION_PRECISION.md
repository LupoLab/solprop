# Precision in RK and low-storage sessions

The internal CPU session boundary accepts `ComplexF32` and `ComplexF64` states.
The caller supplies `arithmetic=SourceArithmetic(T; state_exponent, field_exponent)`
to `RungeKuttaSession`, `LowStorageSession`, and `TrialProbe`; the default remains
`SourceArithmetic(Float64)`. The state already represents the physical internal
spectral amplitude divided by `2^state_exponent`. The absolute error tolerance is
already divided by the same scale. See [the common scale contract](PRECISION.md).
These constructors neither select scales nor convert a physical field.

Preparation checks the state type, the exact arithmetic record on a known modal or
spatial source, and `LinearFlow.state_type` before allocating session/probe storage.
Generic Julia source and flow callables remain responsible for the same scale
contract. `_prepare_march` preserves the supplied complex precision while copying
and masking the state; it retains the existing position, replay and delivery checks.
The arithmetic record remains on the session/probe for later transport validation.

## Arithmetic and ownership

The reviewed [march leaves](MARCH_PRECISION.md) own binary32 stage derivatives,
combinations, defects, low-storage registers and reduction scratch. Positions,
controller histories, tableau records, attempt rows and final norm scalars remain
binary64. Immutable represented operator coefficients and `LinearFlow.exponent`
are the existing wide phase operands; they are not physical field-state storage.
No additional mutable binary64 physical field array is introduced by this slice.

Actual step multipliers must be zero or normal finite binary32 after conversion,
as specified by the march leaves. Low-storage sessions check the step and its
sixth before marking the accepted field in flight. Invalid trial-probe multipliers
refuse before clearing an unread candidate or changing work counters. A numerical
callback failure after entry keeps the existing distinction: RK owns a separate
candidate, whereas low-storage arithmetic has entered its in-place contract.
Attempted and committed low-storage source evaluations remain separate counters.
No precision-dependent controller or landing policy is introduced.

The bounded common scheduler preserves launch, plane, snapshot, warning, stop and
acknowledgment ownership. The precision keyword does not enable observations whose
prepared storage still uses another precision. In particular, binary32 probes
refuse a binary64 temporal or Cartesian split workspace during preparation. A
matching split/service preparation is separate integration work. All binary32
input noise remains refused; the binary64 noise path is unchanged.

## RK continuous extension

The same translated continuous extension is used for both precisions. Its five
retained coefficient fields, finished endpoint and observed field use state
precision. For binary32, its eighth field-sized array stores `ComplexF32` linear
factors; no additional wide exponent field is allocated. Expansion-weight scratch
is a small binary64 vector, independent of the state size. Boolean stiff flags and
integer failure locations retain their original structural types. Every new array
is reported by `runge_kutta_observation_memory` and the shared retained-array walker.

For a constant operator, the scalar-wide phase is `D*(z1-z0)`. A direct represented
operator subtracts the two original binary64 Clenshaw evaluations. An expansion
first subtracts its scalar cumulative weights, then contracts the original complex
terms in their existing order. This is the same represented arithmetic as
`operator_exponent!`, including its cancellation and interval hypotheses. Each
scalar complex exponential is formed in binary64 and rounded once to `ComplexF32`
before multiplying the observed state. This is a CPU wide64 path: no DF32 or device
phase bound is inferred.

The extension's solution/dense weights and step multiplier round to state precision
before the ordered stage contractions. The coordinate `theta=(z-z0)/h` and
`1-theta` are computed in binary64 and independently rounded for the nested
polynomial. This reproduces the precision boundary of the shared NumPy
`extension_coefficients` and `extension_value` graphs. It does not require bitwise
agreement across implementations. The stiff-bin decision still uses the wide
full-step phase and its existing magnitude threshold. Such bins use the anchor's
linear flow. Finiteness is checked before applying inactive-row zeros.

Exact starts and canonical endpoints copy retained state. The already validated
positive gap between raw and canonical endpoints copies the finished endpoint,
without polynomial extrapolation. A zero phase yields an exact identity factor;
its application preserves the original storage. Binary64 uses its previous wide
exponent array and the previous arithmetic order throughout.

## Qualification and remaining boundaries

Focused tests compare one/eight-step commuting linear equations with MPFR discrete
polynomials and compare the prepared modal Kerr source with its independently
reduced cubic equation. The latter uses two distinct state/field scale pairs.
Fixed operation-count budgets precede comparisons, use absolute stage/source
contributions, and include bounded propagation sensitivity. FFT/libm accuracy is
an explicit qualified hypothesis, not a consequence of counting arithmetic alone.
Dense tests cover constant, direct represented and expanded operators, stiff bins,
endpoint copies and raw/canonical landing gaps. Lifecycle tests cover rejection,
floor failures, invalid ranges, source failures, single-use probe readback and
bounded delivery. All warmed CPU advancement/retention/readback checks require zero
allocations; the affected binary64 tests retain their previous bounds.

This slice does not enable public single precision, change host import/export,
attach binary32 observation services, or admit DF32/device execution. Those callers
must supply matching scales, scaled tolerances and precision-compatible services.
