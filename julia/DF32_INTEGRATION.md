# Binding DF32 arithmetic to an actual start

This package binds the existing CPU whole-expression assembly to an actual request
and compatible restored history. It is an internal preparation boundary, not public
DF32 admission. The public adapter continues to refuse `wide="df32"` until every
admitted method and operator has a qualified numerical path. In particular,
separable spatial operators, complete Adams factors and dense observation remain
integration obligations. No device arithmetic or performance claim follows from a
CPU certificate.

The numerical assumptions and existing leaf contracts are in
[WIDE_ASSEMBLY.md](WIDE_ASSEMBLY.md), [DF32_BOUNDS.md](DF32_BOUNDS.md) and
[PHASE_ADMISSION.md](PHASE_ADMISSION.md). The accepted execution scope is
[SEP 48](../docs/dev/seps/sep-0048-julia-backend.md); state arithmetic and scaling
follow [SEP 33](../docs/dev/seps/archive/sep-0033-single-precision-mlx.md).

## Two boundaries

Preparation binds the exact operator representation, method, actual accepted start,
remaining replay, configured maximum Adams order and actual retained coordinates.
It computes operand envelopes from those same inputs. A finite operand envelope is
not a whole-expression error certificate. Before a trial changes any numerical
field or calls a source, a later numerical integration must certify every exponent,
transport and complete compensated factor that the trial can evaluate. The checked
coordinates include the raw endpoint and actual stage positions; the supplied step
and the rounded endpoint distance remain distinct.

This division permits a conservative CPU implementation without assuming that an
adaptive step has a positive lower bound. The controller's `dz_min` is not such a
bound: a forced landing can be shorter. A represented jump divided by that step, or
a large stale history interval, can cause a named numerical-domain refusal before
the trial. No refusal changes precision, discards history or chooses another method.

## First package interfaces

The first package owns only `_wide.py`, its Python tests, `wide_binding.jl`, its
Julia tests and this document. It changes no shared preparation, session, bridge,
export, history schema or module includes.

The pure Python interface is:

```python
DF32OperatorBinding(operator: CoefficientTransfer)

bind_df32_start(
    operator: CoefficientTransfer | DF32OperatorBinding,
    run: RunTransfer,
    start: float,
    *,
    history: MarchSnapshot | None = None,
    origin_wide: Literal["binary64", "df32"] | None = None,
) -> DF32StartBinding
```

`DF32StartBinding` is a frozen record containing `operator: DF32OperatorBinding`,
`run`, `history`,
`operands: PhaseOperands` and `origin_wide`. The actual start is
`operands.spans.start`; the scientific precision/scales are `run.arithmetic`.
`origin_wide=None` means unknown provenance for old history and no provenance for
a fresh start. A known origin is execution metadata, not a scientific identity or
an assertion of zero numerical error. Binary64 scientific state refuses here;
this package binds only binary32 state with a DF32 wide type.

`DF32OperatorBinding` always copies the supplied coefficient/table arrays into owned
read-only storage. Its `data: CoefficientTransfer` may then be shared across starts
under the immutable numerical-input contract. A read-only flag on an arbitrary
caller array does not prove ownership. Passing a raw descriptor to `bind_df32_start`
constructs this wrapper once. Run and history arrays are copied for the actual start,
and nested control/run/history records are rebuilt around those copies. History row
mappings become read-only. The same bound full
history must subsequently be restored; analyzing one snapshot and restoring another
is outside the contract. Directly re-enabling writes to an immutable prepared array
invalidates its preparation, as for the existing descriptor records.

The Julia interfaces are:

```julia
WideStartSpec(start::Float64, tableau::Union{Nothing,EmbeddedTableau},
    schedule::MarchSchedule;
    geometry::Union{Nothing,WideGeometry}, arithmetic::SourceArithmetic,
    method::Symbol, order_max::Int, source_absent::Bool)

WideStartHistory(positions, identities, slots;
    rates=nothing, origin_wide::Symbol=:unknown)

prepare_wide_start(execution, operator, spec::WideStartSpec;
    history::Union{Nothing,WideStartHistory}=nothing) -> WideStartBinding

reconstruct_wide_rate(execution, prepared::WideOperator, position::Float64, values;
    before::Bool=false, origin_wide::Symbol=:unknown) -> WideBoundRate
```

`WideStartSpec` retains the immutable tableau and schedule, actual start, geometry,
`SourceArithmetic{Float32}`, method (`:rk`, `:low_storage`, `:adams`), configured
maximum order (zero outside Adams), and structural source-absence fact. RK requires
its actual tableau; Adams and fixed RK4IP require `tableau=nothing`. No scheme is
invented for binding. The schedule
is immutable numerical input after construction. The supplied geometry is the
`PhaseSpans` lowering, not an error certificate. Preparation checks that it contains
the actual start/end, actual replay lengths/stages and retained samples. Subsequent
operations still check their exact supplied coordinates and step against it.

The low-storage method requires fixed replay. A completed start at the exact final
coordinate uses `geometry=nothing`; its binding retains no numerical `WideOperator`
and cannot authorize an operation. It never invents a small positive step merely to
fit the active-run geometry constructor. The same binary64/strided storage contract
is checked on completed starts, without splitting values or imposing DF32 numerical
range on an operator that will not be used.

`WideStartHistory` is a borrowed preparation-call record. Its binary64 positions,
UInt8 identities and Int64 slots are oldest first. Identities are accepted `0` and
predicted `1`; slots are portable zero-based indices into the full ComplexF64 rate
store of shape `(n_omega,n_columns,order_max+2)`. Plain Adams history retains the
same coordinate/kind/slot metadata with `rates=nothing`; its older sample ages still
constrain the binding. Read-only input arrays may overlap
one another because preparation copies selected values before retaining them. The
caller must keep all foreign owners alive for the complete call. The binding retains
no borrowed pointer.

`WideStartBinding` retains the spec, an owned `WideOperator` or `nothing`, owned
history coordinate/kind/slot vectors and a concrete vector of `WideBoundRate`
records for selected restored rates. The latter
records preserve the saved binary64 matrix, its `WideRate` conversion/radii and the
origin-wide provenance. Position and knot side are carried by `WideRate`. Slot/kind
metadata remains in owned vectors on the binding. The actual stored rate is never
replaced by a fresh derivative.

## Reconstructing a conservative saved-rate radius

Let `r` be an actual saved binary64 component. Evaluate the bound representation's
derivative at the same sample coordinate and exact knot side to obtain a DF32 value
`d` with certified absolute error `e_d` relative to the exact represented derivative
`D`. Form, with outward arithmetic,

\[
 e_r = e_d + |r-d|.
\]

The triangle inequality gives `|r-D| <= e_r` without asserting how the old backend
computed `r`. The discrepancy includes all previous evaluation, storage and export
error, including an unrecorded wide type. The subtraction uses both DF32 words and
retains the binary64 subtraction/accumulation errors; it is not a rounded equality
test. Passing `e_r` to `restore_wide_rate` adds the new binary64-to-DF32 conversion
radius. A large discrepancy can make a later complete-factor certificate refuse.
It never licenses replacing `r`, resetting the history or claiming exact provenance.

Accepted samples use the right-hand derivative at an exact knot; predicted samples
use the left-hand derivative, matching the existing stored sample kinds. If a future
portable format cannot establish this side, it must supply a conservative union of
the possible derivative bounds or refuse; it must not infer a side from proximity.

Reconstruction allocates only during preparation. Failure does not mutate inputs or
an existing binding. Warmed `wide_operation` and writers retain their existing
zero-allocation contract. Rate refresh generations remain the authority for rejecting
stale operation certificates before writes.

## Storage and lifetime

There is no global binding cache. Each actual Start owns its binding. Existing
owned immutable operator wrappers may be shared, but the binding does not retain an
origin problem, factory or execution owner. The Python `transfer_memory` traversal
counts all retained arrays once and can exclude arrays already held by the parent
preparation. Additional copies are exactly the initial operator wrapper and each
start's run and full-history arrays described above; no historical field copies are
hidden in a scalar envelope.

The Python start transfer and its copied full history are preparation-local. The
bridge must release them after it has restored the same snapshot and published the
owned Julia session. It may retain detached scalar evidence and the immutable
operator wrapper in the prepared owner. It must not place an actual start binding
or restored history in a problem-wide or session-lifetime metadata cache.

The Julia inventory is explicit: the existing `WideOperator` coefficient/term
DF32 buffers, separate binary64 error buffers and edges; owned coordinates and slot/kind
vectors; and, per selected restored sample, one raw ComplexF64 matrix, one DF32
matrix and one ComplexF64 radius matrix. Derivative/discrepancy temporaries are
preparation-local and are released. All these arrays are host CPU storage. They
cannot be read implicitly from a future device advancement kernel.

Preparation converts every coefficient/term cell. Each selected restored rate
requires a complete certified derivative evaluation and discrepancy pass. The
existing operation certificate traverses every numerical cell before the writer
repeats the DF32 graph. These are explicit CPU costs, not a reusable uniform proof
or a device-safe admission strategy.

## Required numerical integrations after this package

1. A DF32 linear-flow implementation must assemble from the bound coefficients and
   original scalar operands, then call the qualified function leaves. It must use
   the corrected forward/backward pair where inverse cancellation is required.
   Casting an already-assembled binary64 phase does not satisfy this requirement.
2. A pretrial plan must check all actual RK anchor/stage/endpoint flows and both
   low-storage half flows before the first source call or in-place state mutation.
   It must bind the current method coefficients and actual replay cursor. Reuse is
   invalid after either the coordinate tuple or a retained-rate generation changes.
   Writable outputs must be disjoint from the complete binding, including retained
   original saved-rate matrices, not just one temporary `WideOperation` record.
3. Adams needs certificates for complete source and compensation factors, including
   phi function absolute errors, coefficient formation, `h*D_j-L`, transport and
   dense coordinate powers. Certifying the inverse-metre `D_j-Q` alone is
   insufficient. The C32 graph must not narrow a large intermediate whose complete
   factor is representable. Its state arrays remain C32 throughout.
4. RK and Adams dense observation must use the same bound arithmetic and retained
   trial facts. A separate F64 dense phase path cannot inherit DF32 admission.
5. Separable spatial generators need their own protected wide assembly and error
   provenance, including grazing cancellation. The initial modal binding refuses
   them by name. Public DF32 stays closed until this admitted route is covered.
6. The master integration owns start/restore atomicity, lifetime and memory ledgers,
   execution metadata, bridge conversion and the final public/device admission.
   Restart metadata records both the requested new wide type and any known previous
   wide type. Scientific and history hashes do not acquire an execution option.

## Regression obligations

The binding tests must retain the counterexamples established during design:

- replay lengths larger than `dz_max` determine the phase domain;
- full represented intervals catch an interior extremum with small endpoints;
- configured maximum order and older retained samples determine history reach;
- adaptive cross-knot jumps cannot use a fictitious positive minimum step;
- mutation of a caller's writable array cannot invalidate a retained summary;
- a completed replay has no positive future step;
- separately rounded binary64 endpoints can erase a nonzero represented difference;
- a tiny cross-knot interval can fail assembly error even when its exact jump is zero;
- saved rates require a derived radius and keep their actual values;
- stale sample generations and failed certificates preserve every destination.

These witnesses establish the necessity of the boundary. They are not a propagation
accuracy record. Whole-run tests, a conditional binary32 rounding budget and the
per-vendor arithmetic model remain later qualification obligations.
