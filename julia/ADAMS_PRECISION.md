# Adams precision implementation boundary

This is the phase 3 numerical-leaf assignment, not a public capability claim.
The master owns sessions, bridge records, physical scaling, portable history and
public admission. The Adams agent exclusively owns `src/adams.jl`, new focused
precision tests and this design record. Existing binary64 behavior stays qualified.

First qualify binary32 state with binary64-wide operator arithmetic. Constructors
accept `AbstractMatrix{Complex{T}}` for `T` in `Float32,Float64`. Source, sampled state,
predictor, candidate, carried state increments and defects use the state precision.
Exponents, sampled rates and wide function operands stay binary64. Separate their
type parameters; never narrow a retained sampled generator or silently replace it
with a derivative at a different coordinate. Keep the existing constructor keywords
and field names used by session/transfer code.

Adams polynomial coefficients, geometry and scalar controllers remain host binary64.
Derive each working-precision combination explicitly: evaluate exponential/phi and
protected differences in wide arithmetic, then narrow the complete numerical factor
once before its state multiplication. Avoid an intermediate tiny or overflowing
coefficient when its combined factor is representable. Refuse unsupported ranges by
name before publishing a trial. Keep zero/no-op and source-absent rules exact.
Factored defects preserve their protected cancellation order; no reassociation or
implicit binary64 promotion of state accumulations. Compare matched histories with
operation-derived bounds and short trajectories with independent converged references.

Binary32 norm reductions reuse the reviewed scaled `_Float32Sum` primitives in
`march_precision.jl`: old-state magnitude for Adams, masked denominators, finiteness
of every state/candidate entry, final small host binary64 merge. All warmed numerical
entry points must remain allocation-free. Dense retention uses state-typed source and
sampled-state arrays, wide exponent/rate arrays and explicit factor narrowing. Copies
and retained sampled identities stay exact within one precision.

Do not attach DF32 in this first slice. The subsequent attachment must use the
reviewed `WideOperation` certificates for actual step, transport and compensation
graphs, including restored sampled rates, actual replay steps and history age. A plain
conversion of an already assembled binary64 phase is not that implementation.
Any required field/layout change is reported to the master before session edits.
