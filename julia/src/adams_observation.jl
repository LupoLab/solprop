# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public AdamsObservationWork, retain_adams_observation!, observe_adams!

"""
    AdamsObservationWork(execution, state, max_order; compensated=false,
                         arithmetic=SourceArithmetic(Float64))

Prepare exclusively owned CPU storage in state precision for one Adams dense window.

`state` supplies the nonempty `(n_omega,n_columns)` shape and is not retained.
Its precision must match `arithmetic`, which is retained without changing physical
units. Source/sample-state arrays have that precision; exponents and rates stay binary64.
`max_order` is one through six, and `compensated` matches the session history.
The workspace owns `dense::AdamsDense`, the observed `field`, and finite-check scratch.
Its finite `destination` is the canonical accepted landing coordinate in metres once
`ready=true`. Internal arrays remain owned by this workspace and must not be replaced.

Retention and observation allocate nothing after warmup. The field is meaningful only
after `observe_adams!` returns successfully; retaining a newer window invalidates that
observation by clearing its field. This numerical owner has no delivery cursor, source
counter, controller, monitor or portable-history state. Throw `ArgumentError` for
unsupported execution, empty state or unsupported order.
"""
mutable struct AdamsObservationWork{D, A, P}
    dense::D
    field::A
    failures::Vector{Int}
    destination::Float64
    ready::Bool
    arithmetic::P

    function AdamsObservationWork(
            execution::KernelExecution, state::AbstractMatrix{Complex{T}},
            max_order::Integer; compensated::Bool = false,
            arithmetic::SourceArithmetic = SourceArithmetic(Float64),
        ) where {T <: Union{Float32, Float64}}
        T === source_type(arithmetic) ||
            throw(ArgumentError("Adams dense state differs from its source arithmetic"))
        oncpu(execution) ||
            throw(ArgumentError("Adams dense observation currently requires CPU"))
        dense = AdamsDense(execution, state, max_order; compensated)
        field = dzeros(execution, Complex{T}, size(state)...)
        return new{typeof(dense), typeof(field), typeof(arithmetic)}(
            dense, field, zeros(Int, size(state, 2)), 0.0, false, arithmetic,
        )
    end
end

@inline _adams_observation_arrays(dense::AdamsDense) = (
    dense.initial, dense.endpoint, dense.exponent, dense.source, dense.states,
    dense.rates, dense.coefficients, dense.nodes, dense.weights,
)

@inline _adams_observation_alias(array, ::Tuple{}) = false
@inline function _adams_observation_alias(array, owned::Tuple{T, Vararg}) where {T}
    return Base.mightalias(array, first(owned)) ||
        _adams_observation_alias(array, Base.tail(owned))
end

function _adams_observation_landing(session, destination)
    work = session.work
    work.ready || throw(ArgumentError("no completed Adams trial is available"))
    work.position == session.state.z ||
        throw(ArgumentError("retain the Adams dense window before committing its field"))
    raw_end = work.position + work.step
    isfinite(raw_end) && raw_end > work.position &&
        isfinite(destination) && destination > work.position ||
        throw(ArgumentError("the Adams dense landing interval is invalid"))
    if isempty(session.schedule.replay_steps)
        _march_reached(min(raw_end, destination), max(raw_end, destination)) ||
            throw(ArgumentError("the Adams dense endpoint is not this trial's landing"))
    else
        abs(raw_end - destination) <= _replay_slack(session.schedule, destination) ||
            throw(ArgumentError("the Adams dense endpoint is not this replay landing"))
    end
    return nothing
end

"""
    retain_adams_observation!(observation, session, destination)

Retain one finished accepted candidate before committing its field or history slots.

The caller has accepted the completed trial, applied its hard mask and candidate
finishing, and supplies its actual scheduler `destination` in metres. The session must
still hold the previous accepted field and selected history samples. Validate the
canonical/raw endpoint snap with the existing adaptive reached rule or this session's
full replay slack, then retain the already implemented Adams corrector polynomial,
sampled predictor states/rates, raw start and finished candidate.

Return `nothing` without mutating any session field, history, controller, service or
counter. A failed numerical retention leaves this observation unavailable. Throw
`ArgumentError`/`DimensionMismatch` for inconsistent trial or landing data,
and `NonfiniteSourceError` for a nonfinite retained endpoint. No split collar is applied
here; dense admission remains the shared request policy.
"""
function retain_adams_observation!(
        observation::AdamsObservationWork, session::AdamsSession, destination::Float64,
    )
    observation.arithmetic == session.arithmetic ||
        throw(ArgumentError("Adams dense and session arithmetic differ"))
    _adams_observation_landing(session, destination)
    observation.ready = false
    _source_finite!(
        observation.failures, session.field, session.execution, :dense_initial,
    )
    _source_finite!(
        observation.failures, session.work.candidate, session.execution, :dense_endpoint,
    )
    adams_retain_dense!(
        observation.dense, session.work, session.field, session.work.candidate,
        session.history, session.stencil, session.execution,
    )
    fill!(observation.field, 0)
    observation.destination = destination
    observation.ready = true
    return nothing
end

"""
    observe_adams!(observation, z, active, execution)

Evaluate one masked dense field without changing the committed session or its history.

`z` is a finite coordinate in metres between the retained start and canonical accepted
destination, inclusive. Exact canonical destinations copy the finished endpoint. If
rounding places the validated canonical landing above the raw `start+step`, points in
that tiny positive gap also copy the endpoint: the polynomial is never extrapolated.
All other interior points use the unchanged retained Adams polynomial and transported
compensation; an exact start copies its retained raw field.

`active` is a one-based strided Boolean row vector disjoint from observation storage.
Check all evaluated values
for finiteness before setting inactive rows to structural zero, so a hard mask cannot
hide failed arithmetic. Return the workspace's owned `field`, valid until its next
retention or observation. No source is evaluated, no split collar is applied, and no
session counter, history, controller, service or delivery cursor changes.

Throw `ArgumentError`/`DimensionMismatch` before evaluation for an unavailable window,
unsupported execution, invalid coordinate/mask or retained mask alias. Numerical
nonfiniteness throws `NonfiniteSourceError` outside the scalar kernel.
"""
function observe_adams!(
        observation::AdamsObservationWork, z::Float64, active,
        execution::KernelExecution,
    )
    oncpu(execution) ||
        throw(ArgumentError("Adams dense observation currently requires CPU"))
    dense = observation.dense
    observation.ready && dense.ready ||
        throw(ArgumentError("no accepted Adams dense window is available"))
    active isa StridedVector{Bool} ||
        throw(ArgumentError("the dense mask needs unpacked strided Boolean storage"))
    length(active) == size(observation.field, 1) &&
        axes(active, 1) == Base.OneTo(length(active)) ||
        throw(DimensionMismatch("the dense mask does not match the retained state"))
    arrays = _adams_observation_arrays(dense)
    (
        _adams_observation_alias(active, arrays) ||
            Base.mightalias(active, observation.field) ||
            Base.mightalias(active, observation.failures)
    ) &&
        throw(ArgumentError("the dense mask must not alias observation storage"))
    _adams_observation_alias(observation.field, arrays) &&
        throw(ArgumentError("the observed field must not alias its retained window"))
    isfinite(z) && dense.position <= z <= observation.destination ||
        throw(ArgumentError("the observation lies outside its accepted Adams window"))
    if z == observation.destination || z > dense.position + dense.step
        copyto!(observation.field, dense.endpoint)
    else
        adams_observe!(observation.field, dense, z, execution)
    end
    _source_finite!(
        observation.failures, observation.field, execution, :dense_observation,
    )
    _adams_session_mask!(observation.field, active, execution)
    return observation.field
end
