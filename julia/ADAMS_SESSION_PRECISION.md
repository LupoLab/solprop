# Precision in Adams sessions and observations

`AdamsSession` and `AdamsObservationWork` accept stored `ComplexF32` or
`ComplexF64` matrices. The caller supplies
`arithmetic=SourceArithmetic(T; state_exponent, field_exponent)`; the default is
`SourceArithmetic(Float64)`. This matches the shared
[precision contract](PRECISION.md) and the
[Adams numerical arithmetic](ADAMS_PRECISION.md). This internal CPU attachment
adds no public precision or device admission.

The session checks the exact state type, a known modal or spatial source's full
arithmetic record, and `LinearFlow.state_type` before copying the launch or
allocating session work. An explicit binary32 record is required for a binary32
state. Both scale exponents must agree with the source. Generic Julia callbacks
remain responsible for the same precision and scale contract. The session and
its dense owner retain the record, and retention refuses a different record before
invalidating an existing observation.

The input launch and absolute tolerance already have their selected internal
scale. The constructor copies and hard-masks the stored launch in its supplied
precision. It performs no physical conversion. The source, sample-state,
predictor, candidate, defect and observed-field storage follows state precision.
Sampled operator rates and exponents remain binary64. Scalar coordinates,
coefficients, control readings, final norm scalars and attempt rows retain the
existing binary64 contract. The separate norm scaling scratch from the numerical
leaf remains visible to the common retained-array memory ledger.

## Numerical and delivery behavior

This attachment calls the existing predictor/evaluate/correct leaf with the same
selected history and policy. The binary32 leaf computes complete factors in
binary64 before narrowing, including represented-operator compensation; its range
and conditioning restrictions remain those in `ADAMS_PRECISION.md`. In particular,
this attachment neither removes a factor refusal nor infers DF32 closure from a
small final phase. The old-state norm and full-state finite checks are unchanged.

The same bounded scheduler owns adaptive rejection, fixed replay, landing,
launch/plane/snapshot receipts, stop and failure events. Source invocations are
counted immediately before callback entry. Logical startup and predictor charges
remain distinct from actual calls for a structurally absent source. Observation
and delivery perform no extra source evaluations. Arithmetic retention introduces
no new controller choice, restart, accepted sample or order-selection decision.

Dense windows retain the already accepted corrector polynomial and its source and
sample-state arrays in state precision, plus wide rates and exponents. Observation
uses an independently owned state-precision field and never replaces the committed
field. Exact starts and canonical endpoints copy retained stored arrays. A validated
positive gap above the raw endpoint also copies the finished endpoint without
extrapolation. Interior observations use the qualified Adams numerical leaf and
check finiteness before applying hard-mask zeros. The arithmetic record has no
array storage; each retained dense array is counted once by `session_memory`.

Portable history is a separate boundary. Its accepted state is the **raw scaled
state**, losslessly widened to complex128; source and sampled-state rows retain
complex64 for single precision, while rates retain complex128. No history array
receives a physical scale factor here. Physical field export applies the scale at
its own boundary. Callers must also prepare observation services with matching
precision and scales; this constructor does not convert service workspaces.

## Qualification

Focused tests use eight-step windows, task budgets one and two, and bounds checking.
They exercise fixed orders one and four, adaptive rejection, source and budget
failures, snapshots, bounded delivery, represented cumulative compensation and dense
planes. Exact assertions cover ownership, scales, dtype, counters, receipt validity,
raw copies and hard-mask zeros. Warmed advancement requires zero allocations.
The directly affected binary64 suites retain their original numerical bounds.

For a constant linear operator and constant source, the order-one solution is
compared with its independent MPFR closed form. The comparison bounds complete
old-state and source contributions before cancellation; the declared operation
budget includes wide factor evaluation and one binary32 narrowing. Adaptive dense
solutions use a fixed short-window accuracy envelope against analytic solutions
for `y'=a*y+b*z` and `y'=a*z*y+b*z`, not a claim that local `rtol` bounds global error.

A real four-point modal FFT/Kerr fixture reduces independently to
`N(A)=-i*|A|^2*A/64` for its sole occupied DC bin. Its MPFR reference carries the
actual order-one PEC source register across all eight steps. For `|A|<=3`, the
source Jacobian norm is at most `L=27/64`. In the error norm
`max(|delta A|, |delta sampled source|/L)`, one step has induced majorant
`1+h*L+(h*L)^2/2`: the two rows of the nonnegative error matrix sum to that value
and `1+h*L`. A fixed local absolute budget is propagated with this majorant and
includes the initial source evaluation. This bound assumes the qualified finite
four-point FFT component error of at most `8u` times the absolute input sum; operation
counting alone does not establish FFT or transcendental accuracy. Two distinct
state/field exponent pairs test the physical source scaling independently from
stored session ownership.

Public single-precision admission, portable-history conversion, Python transport,
physical export, binary32 service attachment and DF32/device execution remain
separate integration obligations. No timing or device qualification follows from
these CPU allocation and accuracy checks.
