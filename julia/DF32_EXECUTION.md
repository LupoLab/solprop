# Certified DF32 trial flows

This package connects the qualified CPU wide-assembly and function graphs to landed
binary32 RK and fixed low-storage RK4IP trials. It does not enable a public execution
option or a device. The accepted scope and arithmetic model remain SEP 48,
`DF32_BOUNDS.md`, `WIDE_ASSEMBLY.md` and `DF32_INTEGRATION.md`.

## Frozen interfaces

```julia
DF32LinearFlow(execution, operator, state::AbstractMatrix{ComplexF32},
               spec::WideStartSpec; tabulate=false)

DF32TrialSite(origin::Float64, step::Float64, destination::Float64,
              replay_index::Int)

prepare_df32_trial!(flow::DF32LinearFlow,
                    scheme::Union{EmbeddedTableau,RK4IPTableau},
                    site::DF32TrialSite)::DF32TrialFlow

(trial::DF32TrialFlow)(output, input, origin::Float64,
                       destination::Float64)::Nothing
```

The flow constructs exactly one `WideStartBinding` from its retained original
operator and spec; there is no independently supplied operator/binding pair. It
retains that binding unchanged. State and source arithmetic are binary32 with the
spec's original-launch scales. CPU constant, represented cumulative and real-weight
expansion operators are admitted. Dense output, Adams, tabulation and separable
spatial operators refuse during preparation. Other state precisions refuse by name.

The site carries four independent facts: the actual accepted origin, supplied
positive step, canonical landed destination and replay cursor (zero for adaptive).
The canonical destination also remains inside the actual requested run, allowing
only the existing replay landing slack. The raw endpoint remains the actual
noncontracted `origin + step`; it is not replaced
by the canonical destination. RK stage coordinates use the actual supplied tableau.
RK4IP retains both separately computed half intervals. The replay cursor must identify
the actual supplied length and edge; adaptive sites have no fictitious minimum step.

A prepared token binds the method, complete coordinate tuple, replay cursor and plan
generation. Both direct numerical trial entry and session advancement check it before
source calls or state/work writes. Direct callers first prepare a token and pass it
as the existing flow callback; passing an unprepared DF32 flow refuses immediately.
The session performs this preparation automatically after landing. Low-storage
preparation precedes its `in_flight` latch and first source invocation.

## Transaction and arithmetic

Each flow owns two fixed-capacity host banks of isbits operation projections. Each
retains the full DF32 scalar bundle, its scalar error bundle, exact endpoints/step and
numeric certificate radii/status;
the shared operator is retained only once.
The inactive bank receives a certificate for every anchor-to-stage/endpoint interval
(RK) or both half intervals (RK4IP). A refused certificate leaves the published bank
and generation unchanged. Publication follows all checks; a later successful plan
invalidates every earlier token before any output write. There is no per-session
history of plans, no field-sized factor cache and no allocation during warmed trials.

Forward and backward applications use the same certified forward phase and choose
the corresponding result of `_df32_exp_pair`. The callback evaluates the protected
DF32 representation graph, forms that corrected pair, then multiplies its selected
ComplexF32 factor by ComplexF32 state. It never casts an already-assembled binary64
phase or enters a binary64 transcendental. Only numerical DF32 arrays and isbits scalar
bundles enter the launched body; host certificate/range records stay outside it.
A structural zero interval copies the input exactly after admission and alias checks.
Each exactly zero assembled-phase lane also copies its input bits, including signed
zeros. This branch follows certification and never waives a range or error check.

The existing certificate allows real/imaginary stored arguments up to 16 and 2^20,
and real-plus-imaginary assembly error at most 2^-27. Under the documented gradual
underflow/RN-even/FMA model, the corrected pair's relative error is below 5u for
u=2^-24. State-product and full-trial tests add their own arithmetic and sensitivity
allowances; this scalar pair bound is not a global propagation tolerance. These
function-domain bounds are necessary, not sufficient admission conditions. The
whole-expression certificate may refuse within them: a unit-step constant imaginary
phase of magnitude 2^20 currently exceeds the conservative u/8 assembly allowance.
Such a refusal remains explicit; this package does not relax the certificate or
claim the entire nominal function rectangle for arbitrary assembled inputs.

Caller buffers must match the preparation's shape and ComplexF32 dtype. Output must
have an admitted writable layout; input/output must be disjoint or identical. Both
must be disjoint from the full flow ownership graph, including the original operator,
binding coefficient/error arrays, any retained saved-rate values, both operation
banks and failure scratch. Direct trial entry also checks every mutable state/workspace
array against that owner before any stage or source write. Checks precede all output
mutation. Finiteness is checked
outside launched bodies, preserving existing failure/poison accounting.

## Shared hooks and ownership

`source.jl` supplies default internal `_flow_shape`, `_inplace_flow`,
`_flow_arithmetic`, `_flow_start`, `_prepare_trial_flow` and `_check_trial_flow` hooks.
Existing LinearFlow and generic callback behavior remains unchanged. RK/low-storage
numerical entries check prepared tokens; their sessions prepare from actual landing facts.
No existing binary64 assembly, factor, stage or controller arithmetic changes.

The parent integration owns module inclusion after `wide_binding.jl`, factories,
bridge records, exports, memory evidence and final public/device admission. This
package owns only `df32_flow.jl`, its focused test, this note, and the narrow hooks in
`source.jl`, `runge_kutta.jl`, `low_storage.jl`, `march.jl` and
`low_storage_session.jl`.

## Storage and deferred work

The original immutable operator arrays are shared with preparation. One owned
WideStartBinding retains DF32 coefficient/term arrays, binary64 error arrays and
breakpoints. Each active flow adds two bounded vectors of operation records and one
Int failure vector per column. Each isbits operation record occupies 400 bytes on
this 64-bit Julia ABI: 80 bytes of DF32 scalar metadata, 248 bytes of scalar radii,
32 bytes of aggregate certificate radii/status, and 40 bytes of coordinates/generation.
With N RK stages, the two banks occupy 800(N+1) payload bytes; RK4IP uses 1600 bytes.
The failure vector adds sizeof(Int)*n_columns bytes. Completed starts have empty banks.
Array headers and enclosing objects are separate host-object overhead, not numerical
payload. Tokens retain only their flow, site and generation.
All are explicit lifetime-owned inventory; no global cache or per-trial storage grows.
The general session ledger sees both banks and all binding arrays and excludes the
shared original operator. In this RK/low-storage package the binding's history
vectors are empty; the alias guard nevertheless visits saved matrices explicitly.

Each preflight performs N+1 complete per-cell host certificate traversals for RK,
or two for RK4IP. Each actual flow application then repeats only the DF32 graph:
RK invokes 2N+2 flows (its zero-stage copies avoid graph evaluation), and RK4IP invokes
four. Polynomial degree and expansion-term count multiply that work. This CPU
implementation intentionally pays that certificate cost and makes no throughput or
device qualification claim. A reusable uniform certificate or explicit device-safe
strategy is still required before a device can use this boundary.

Complete Adams factors require a separate numerical package. It must certify the
actual nonuniform coefficient formation, phi absolute error, transported sampled-rate
provenance, complete source/compensation/defect factors and dense integrals before
narrowing. Existing binary64 exponent/moment callbacks cannot provide DF32 admission.
Adams startup currently precedes trial landing, so its later transaction design must
also place certificate checks before seed-source calls and history/ledger mutation.

Separable spatial operators also remain refused: the protected squared-wavevector
subtraction, branch at real(q)=0, complex square root, frame cancellation and loss
combination need their own bound. Host split-once preparation must retain an explicit
assembly radius; casting the current binary64 rate or phase supplies no such proof.
The current modal binding and function domains do not qualify that representation.


## Internal Python session attachment

The direct Julia execution owner accepts `wide="df32"` for noise-free binary32
modal landed RK and low-storage sessions. Public registry admission remains closed
until the complete phase 3 matrix qualifies. The owner copies the original operator
once; each start owns its exact run transfer and actual coordinate geometry. The
PythonCall factory receives that same transfer by identity, preserves original-launch
scales on field restart, and constructs `DF32LinearFlow` from the preparation's
original operator. Julia performs every actual-trial certificate before source work.
There is no Python numerical trial loop or binary64 phase fallback. Timing probes,
Adams, dense observations, spatial operators and tabulated flow still refuse by name.

Managed CPU bridge tests cover eight replay steps for both methods, including RK
steps larger than the adaptive `dz_max`, at work limits 1 and 256. Same-grid binary64
16/32-step refinement must settle at least 100 times inside SEP 33's field and energy
envelopes before comparison. Actual range failures preserve the accepted field and
perform no source work; completed starts invent neither a phase nor a trial. Field
restarts at the middle plane bind only the remaining replay and retain the original
amplitude scales. Every retained operator copy and operation bank is inventoried.
Two-member fixed-schedule group checks cover both wide types and both methods,
including exact delivery and preservation of a member whose phase certificate refuses.
Independent PythonCall witnesses also reject mismatched run identity and actual start
coordinates before consuming caller storage, then reuse the same preparation.
These checks establish the stated short internal CPU paths; full phase 3 admission,
portable Adams history, dense output and vendor execution remain additional work.
