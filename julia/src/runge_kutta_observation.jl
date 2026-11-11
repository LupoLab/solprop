# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public RungeKuttaObservationWork, retain_runge_kutta_observation!, observe_runge_kutta!
public runge_kutta_observation_memory

"""
    RungeKuttaObservationWork(execution, state, tableau, dense_weights, operator)

Prepare owned binary64 CPU storage for the translated RK continuous extension.

`state` supplies a finite nonempty `(n_omega,n_columns)` shape in internal spectral
units. `tableau` and finite `dense_weights` supply one dimensionless weight per stage;
the first/last stage nodes are zero/one. The caller selects the compatible DOPRI5
extension through solprop's `continuous_extension`, never by counting stages here.
`operator` is a binary64 `DiagonalOperator`, `RepresentedOperator` or `ExpansionOperator`
of the same shape. Its immutable prepared data are borrowed.

Own five coefficient fields, a finished endpoint, the observed field, exponent and
expansion-weight scratch, unpacked stiff flags, and per-column finite-check scratch.
No starting array or mutable session is retained. Internal array storage must not be
replaced. `runge_kutta_observation_memory` lists newly owned bytes; borrowed operator
storage is accounted by its preparation. Retention and observation allocate nothing
once warmed. Throw `ArgumentError` for unsupported data/execution and
`DimensionMismatch` for incompatible extents.

The five-coefficient form is the established DOPRI5 fourth-order continuous extension
of Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, second
edition, §II.6, doi:10.1007/978-3-540-78862-1. Its midpoint constraints recover the
weights from Dormand and Prince, "Runge-Kutta triples" (1986), p.1014,
doi:10.1016/0898-1221(86)90025-8. Coefficients remain translated inputs.
"""
mutable struct RungeKuttaObservationWork{N, O, C, A, M, W}
    solution_weights::NTuple{N, Float64}
    dense_weights::NTuple{N, Float64}
    operator::O
    coefficients::C
    endpoint::A
    field::A
    exponent::A
    stiff::M
    weights::W
    failures::Vector{Int}
    position::Float64
    step::Float64
    destination::Float64
    ready::Bool

    function RungeKuttaObservationWork(
            solution_weights::NTuple{N, Float64}, dense_weights::NTuple{N, Float64},
            operator::O, coefficients::C, endpoint::A, field::A, exponent::A,
            stiff::M, weights::W, failures::Vector{Int}, position::Float64,
            step::Float64, destination::Float64, ready::Bool,
        ) where {N, O, C, A, M, W}
        return new{N, O, C, A, M, W}(
            solution_weights, dense_weights, operator, coefficients, endpoint, field,
            exponent, stiff, weights, failures, position, step, destination, ready,
        )
    end
end

function _rk_observation_shape(operator::DiagonalOperator)
    eltype(operator.rate) === ComplexF64 ||
        throw(ArgumentError("RK dense observation requires a binary64 operator"))
    return size(operator.rate), 0
end

function _rk_observation_shape(operator::RepresentedOperator)
    table = operator.cumulative
    eltype(table.coefficients) in (Float64, ComplexF64) &&
        eltype(table.breakpoints) === Float64 ||
        throw(ArgumentError("RK dense observation requires a binary64 cumulative table"))
    return (operator.n_frequency, operator.n_columns), 0
end

function _rk_observation_shape(operator::ExpansionOperator)
    table = operator.cumulative_weights
    eltype(operator.terms) === ComplexF64 && eltype(table.coefficients) === Float64 &&
        eltype(table.breakpoints) === Float64 ||
        throw(ArgumentError("RK dense observation requires a binary64 expansion"))
    return (size(operator.terms, 1), size(operator.terms, 2)), size(operator.terms, 3)
end

_rk_observation_shape(operator) =
    throw(ArgumentError("RK dense observation requires a supported diagonal operator"))

function RungeKuttaObservationWork(
        execution::KernelExecution, state, tableau::EmbeddedTableau{N}, dense_weights,
        operator,
    ) where {N}
    oncpu(execution) || throw(ArgumentError("RK dense observation currently requires CPU"))
    state isa AbstractMatrix{ComplexF64} && !isempty(state) && all(isfinite, state) ||
        throw(ArgumentError("RK dense observation needs a finite binary64 matrix"))
    Base.require_one_based_indexing(state)
    N >= 2 && first(tableau.nodes) == 0.0 && last(tableau.nodes) == 1.0 ||
        throw(ArgumentError("the RK continuous extension needs endpoint stages"))
    length(dense_weights) == N && all(isfinite, dense_weights) ||
        throw(ArgumentError("RK dense weights need one finite value per stage"))
    copied = ntuple(index -> Float64(dense_weights[index]), Val(N))
    all(isfinite, copied) ||
        throw(ArgumentError("RK dense weights must fit in binary64"))
    shape, count = _rk_observation_shape(operator)
    shape == size(state) ||
        throw(DimensionMismatch("the RK dense operator and state shapes disagree"))
    return RungeKuttaObservationWork(
        tableau.solution_weights, copied, operator,
        dzeros(execution, ComplexF64, shape..., 5),
        dzeros(execution, ComplexF64, shape...), dzeros(execution, ComplexF64, shape...),
        dzeros(execution, ComplexF64, shape...), dzeros(execution, Bool, shape...),
        dzeros(execution, Float64, count), zeros(Int, shape[2]), 0.0, 0.0, 0.0, false,
    )
end

function _rk_observation_exponent!(work, origin, destination, execution)
    operator_exponent!(
        work.exponent, work.operator, origin, destination, execution; work = work.weights,
    )
    _source_finite!(work.failures, work.exponent, execution, :dense_exponent)
    return nothing
end

function _rk_observation_coefficients_finite!(work, execution)
    coefficients, failures = work.coefficients, work.failures
    each_column!(execution, size(work.field, 2)) do column
        failed = 0
        for row in 1:size(coefficients, 1), term in 1:5
            if !isfinite(coefficients[row, column, term])
                failed = row
                break
            end
        end
        failures[column] = failed
    end
    for column in eachindex(failures)
        failures[column] == 0 ||
            throw(NonfiniteSourceError(:dense_coefficients, failures[column], column))
    end
    return nothing
end

"""
    retain_runge_kutta_observation!(work, initial, endpoint, stages, z, dz,
                                  destination, execution)

Retain a completed accepted RK trial without changing its inputs or session.

`initial` and finished/masked `endpoint` are strided complex binary64 matrices matching
`work.field`. The strided `stages` array adds a third extent equal to the translated
stage count; its derivatives are in the step anchor's interaction picture. The caller
has completed and accepted this trial but has not committed its endpoint yet.
`z`, positive `dz`, and canonical `destination` are finite metres; validate the raw
`z+dz` landing snap using the existing adaptive reached rule.

Contract the same ordered stage sums as the shared `extension_coefficients` into
`r1=y0`, `r2=h*sum(b*k)`, `r3=h*k1-r2`, `r4=r2-h*k_last-r3`, and `r5=h*sum(d*k)`.
The endpoint constraints determine the first four terms; the remaining term enters as
`theta^2*(1-theta)^2*r5`, preserving both endpoint values and derivatives. Store the
existing stiff-bin decision `abs(real(Phi(z+h)-Phi(z))) > 1` separately.

Return `nothing`. Invalid arguments/aliasing raise `ArgumentError` or
`DimensionMismatch` before writes. Numerical failure raises `NonfiniteSourceError`
with this window unavailable; input fields/stages stay unchanged. Every owned entry
read by a later observation is overwritten, and the previous observed field is cleared.
No source is called and no controller/history/transport counter changes.
"""
function retain_runge_kutta_observation!(
        work::RungeKuttaObservationWork{N}, initial, endpoint, stages, z::Float64,
        dz::Float64, destination::Float64, execution::KernelExecution,
    ) where {N}
    oncpu(execution) || throw(ArgumentError("RK dense observation currently requires CPU"))
    initial isa StridedMatrix{ComplexF64} && endpoint isa StridedMatrix{ComplexF64} &&
        stages isa StridedArray{ComplexF64, 3} ||
        throw(ArgumentError("RK dense retention needs strided binary64 arrays"))
    Base.require_one_based_indexing(initial, endpoint, stages)
    size(initial) == size(endpoint) == size(work.field) &&
        size(stages) == (size(initial)..., N) ||
        throw(DimensionMismatch("RK dense trial extents disagree"))
    raw_end = z + dz
    isfinite(z) && isfinite(dz) && dz > 0 && isfinite(raw_end) && raw_end > z &&
        isfinite(destination) && destination > z ||
        throw(ArgumentError("the RK dense landing interval is invalid"))
    _march_reached(min(raw_end, destination), max(raw_end, destination)) ||
        throw(ArgumentError("the RK dense endpoint is not this trial's landing"))
    (
        _retained_alias(initial, work) || _retained_alias(endpoint, work) ||
            _retained_alias(stages, work)
    ) && throw(ArgumentError("RK dense trial inputs must not alias retained storage"))
    work.ready = false
    _source_finite!(work.failures, initial, execution, :dense_initial)
    _source_finite!(work.failures, endpoint, execution, :dense_endpoint)
    all(isfinite, stages) ||
        throw(ArgumentError("RK dense stage derivatives must be finite"))
    _rk_observation_exponent!(work, z, raw_end, execution)
    coefficients, stiff, exponent = work.coefficients, work.stiff, work.exponent
    solution, dense = work.solution_weights, work.dense_weights
    each!(execution, size(initial)...) do row, column
        span = zero(eltype(initial))
        extension = zero(eltype(initial))
        for stage in 1:N
            slope = stages[row, column, stage]
            # Preserve the shared contraction's term order without reassociation.
            span += solution[stage] * slope
            extension += dense[stage] * slope
        end
        span *= dz
        first_term = dz * stages[row, column, 1] - span
        last_term = span - dz * stages[row, column, N] - first_term
        coefficients[row, column, 1] = initial[row, column]
        coefficients[row, column, 2] = span
        coefficients[row, column, 3] = first_term
        coefficients[row, column, 4] = last_term
        coefficients[row, column, 5] = dz * extension
        stiff[row, column] = abs(real(exponent[row, column])) > 1.0
    end
    _rk_observation_coefficients_finite!(work, execution)
    copyto!(work.endpoint, endpoint)
    fill!(work.field, 0)
    work.position, work.step, work.destination = z, dz, destination
    work.ready = true
    return nothing
end

"""
    observe_runge_kutta!(work, position, active, execution)

Evaluate the retained RK extension without replacing the committed session field.

`position` is finite metres inside the retained canonical interval; `active` is a
one-based strided unpacked Boolean row mask disjoint from retained storage. Exact
starts and canonical endpoints copy retained fields. A validated positive gap between
raw `z+dz` and the canonical destination also copies the endpoint, never extrapolates.
Interior points evaluate the established nested polynomial at `(position-z)/dz`,
then apply the represented linear flow from anchor to plane. Bins whose stored
full-step real exponent has magnitude above one use the anchor's linear flow instead,
matching the shared stiff-bin guard for both attenuation and gain.

Return the workspace's owned field, valid until its next observation/retention. Check
numerical finiteness before making inactive rows exact zeros. No source call, snapshot,
controller or history mutation occurs. Invalid data raise `ArgumentError` or
`DimensionMismatch` before evaluation; nonfinite arithmetic raises `NonfiniteSourceError`
and never publishes a successful session selection. Warmed CPU calls allocate nothing.
"""
function observe_runge_kutta!(
        work::RungeKuttaObservationWork, position::Float64, active,
        execution::KernelExecution,
    )
    oncpu(execution) || throw(ArgumentError("RK dense observation currently requires CPU"))
    work.ready || throw(ArgumentError("no accepted RK dense window is available"))
    active isa StridedVector{Bool} ||
        throw(ArgumentError("the RK dense mask needs strided unpacked Boolean storage"))
    length(active) == size(work.field, 1) &&
        axes(active, 1) == Base.OneTo(length(active)) ||
        throw(DimensionMismatch("the RK dense mask and retained field disagree"))
    _retained_alias(active, work) &&
        throw(ArgumentError("the RK dense mask must not alias retained storage"))
    isfinite(position) && work.position <= position <= work.destination ||
        throw(ArgumentError("the observation lies outside its accepted RK window"))
    field, coefficients = work.field, work.coefficients
    if position == work.destination || position > work.position + work.step
        copyto!(field, work.endpoint)
    elseif position == work.position
        each!(execution, size(field)...) do row, column
            field[row, column] = coefficients[row, column, 1]
        end
    else
        theta = (position - work.position) / work.step
        rest = 1.0 - theta
        _rk_observation_exponent!(work, work.position, position, execution)
        stiff = work.stiff
        each!(execution, size(field)...) do row, column
            initial = coefficients[row, column, 1]
            field[row, column] = stiff[row, column] ? initial : initial + theta * (
                    coefficients[row, column, 2] + rest * (
                        coefficients[row, column, 3] + theta * (
                            coefficients[row, column, 4] +
                            rest * coefficients[row, column, 5]
                        )
                    )
                )
        end
        propagate!(field, field, work.exponent, execution)
    end
    _source_finite!(work.failures, field, execution, :dense_observation)
    each!(execution, size(field)...) do row, column
        active[row] || (field[row, column] = zero(eltype(field)))
    end
    return field
end

"""
    runge_kutta_observation_memory(work)

Return named byte counts for every owned dense array, excluding the borrowed operator.

The scalar tuple covers `coefficients`, `endpoint`, `field`, `exponent`, `stiff`,
`weights`, and `failures`. Storage is bounded independently of trials and save planes.
"""
function runge_kutta_observation_memory(work::RungeKuttaObservationWork)
    return (
        coefficients = sizeof(work.coefficients), endpoint = sizeof(work.endpoint),
        field = sizeof(work.field), exponent = sizeof(work.exponent),
        stiff = sizeof(work.stiff), weights = sizeof(work.weights),
        failures = sizeof(work.failures),
    )
end

mutable struct _RungeKuttaSessionObservation{W}
    work::W
    z::Float64
    selected::Bool
end

_march_defer_periodic(session::RungeKuttaSession) = session.observation !== nothing

function _march_landing_target(session::RungeKuttaSession)
    return session.observation === nothing ? session.schedule.planes[session.state.plane] :
        last(session.schedule.planes)
end

function _march_pending_dense(session::RungeKuttaSession)
    return session.observation !== nothing &&
        session.state.plane <= length(session.schedule.planes) &&
        !_march_reached(session.schedule.planes[session.state.plane], session.state.z)
end

function _march_select_observation!(session::RungeKuttaSession, kind, plane)
    observation = session.observation
    observation === nothing && return nothing
    observation.selected = false
    kind === :plane || return nothing
    position = session.schedule.planes[plane + 1]
    _march_reached(position, session.state.z) && return nothing
    observe_runge_kutta!(observation.work, position, session.norm.active, session.execution)
    observation.z, observation.selected = position, true
    return nothing
end

function _march_observed_field(session::RungeKuttaSession)
    observation = session.observation
    if observation !== nothing && observation.selected && _march_plane_selected(session)
        return observation.work.field
    end
    return session.field
end

function _march_observed_position(session::RungeKuttaSession)
    observation = session.observation
    if observation !== nothing && observation.selected && _march_plane_selected(session)
        return observation.z
    end
    return session.state.z
end
