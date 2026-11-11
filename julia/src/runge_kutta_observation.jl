# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public RungeKuttaObservationWork, retain_runge_kutta_observation!, observe_runge_kutta!
public runge_kutta_observation_memory

# Session construction uses these late-bound hooks after all numerical files load.
# The default path retains its existing binary64-wide flow and storage checks.
function _rk_dense_admit(flow, state)
    flow isa LinearFlow ||
        throw(ArgumentError("Runge-Kutta dense observation requires LinearFlow"))
    size(flow.exponent) == size(state) ||
        throw(DimensionMismatch("Runge-Kutta linear flow and state shapes disagree"))
    return nothing
end

function _rk_dense_prepare(
        execution, state, tableau, dense_weights, flow::LinearFlow; arithmetic,
    )
    return RungeKuttaObservationWork(
        execution, state, tableau, dense_weights, flow.operator; arithmetic,
    )
end

"""
    RungeKuttaObservationWork(execution, state, tableau, dense_weights, operator;
                              arithmetic=SourceArithmetic(Float64))

Prepare owned storage in state precision for the translated RK continuous extension.

`state` supplies a finite nonempty `(n_omega,n_columns)` shape in internal spectral
units. `tableau` and finite `dense_weights` supply one dimensionless weight per stage;
the first/last stage nodes are zero/one. The caller selects the compatible DOPRI5
extension through solprop's `continuous_extension`, never by counting stages here.
`operator` is a binary64 `DiagonalOperator`, `RepresentedOperator`, `ExpansionOperator`
or scalar Cartesian/line `SeparableSpatialOperator` of the same shape. Its immutable
prepared data are borrowed. Separable coefficients remain vectors; dense observation
does not retain a spectral operator volume.

`state` and `arithmetic` agree; binary32 state has already been physically scaled.
Own five coefficient fields, a finished endpoint, the observed field, and either a
binary64 exponent or binary32 factor matrix. Binary32 evaluates each represented phase
and exponential in binary64 scalar arithmetic before rounding its factor once; no
additional wide field-sized buffer is allocated. Keep small binary64
expansion-weight scratch, unpacked stiff flags, and per-column finite-check scratch.
No starting array or mutable session is retained. Internal array storage must not be
replaced. `runge_kutta_observation_memory` lists newly owned bytes; borrowed operator
storage is accounted by its preparation. Retention and observation allocate nothing
once warmed. Throw `ArgumentError` for unsupported data/execution and
`DimensionMismatch` for incompatible extents.
Device execution admits binary64 state and owns the numerical arrays in its prepared
context. Each finite check downloads one integer per column, never the field or
coefficient volume. Device binary32 refuses before allocation.

The five-coefficient form is the established DOPRI5 fourth-order continuous extension
of Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, second
edition, §II.6, doi:10.1007/978-3-540-78862-1. Its midpoint constraints recover the
weights from Dormand and Prince, "Runge-Kutta triples" (1986), p.1014,
doi:10.1016/0898-1221(86)90025-8. Coefficients remain translated inputs.
"""
mutable struct RungeKuttaObservationWork{N, O, C, A, X, M, W, F, P, H}
    solution_weights::NTuple{N, Float64}
    dense_weights::NTuple{N, Float64}
    operator::O
    coefficients::C
    endpoint::A
    field::A
    exponent::X
    factor::F
    stiff::M
    weights::W
    failures::H
    position::Float64
    step::Float64
    destination::Float64
    ready::Bool
    arithmetic::P

    function RungeKuttaObservationWork(
            solution_weights::NTuple{N, Float64}, dense_weights::NTuple{N, Float64},
            operator::O, coefficients::C, endpoint::A, field::A, exponent::X, factor::F,
            stiff::M, weights::W, failures::H, position::Float64,
            step::Float64, destination::Float64, ready::Bool, arithmetic::P,
        ) where {N, O, C, A, X, M, W, F, P, H}
        return new{N, O, C, A, X, M, W, F, P, H}(
            solution_weights, dense_weights, operator, coefficients, endpoint, field,
            exponent, factor, stiff, weights, failures, position, step, destination, ready,
            arithmetic,
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

_rk_observation_shape(operator::SeparableSpatialOperator) =
    (_spatial_operator_shape(operator), 0)

_rk_observation_shape(operator) =
    throw(ArgumentError("RK dense observation requires a supported diagonal operator"))

function RungeKuttaObservationWork(
        execution::KernelExecution, state, tableau::EmbeddedTableau{N}, dense_weights,
        operator; arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    ) where {N}
    T = source_type(arithmetic)
    oncpu(execution) || T === Float64 ||
        throw(ArgumentError("device RK dense observation requires binary64 state"))
    state isa AbstractMatrix{Complex{T}} && !isempty(state) &&
        (!oncpu(execution) || all(isfinite, state)) ||
        throw(ArgumentError("RK dense observation needs a finite matching state matrix"))
    Base.require_one_based_indexing(state)
    N >= 2 && first(tableau.nodes) == 0.0 && last(tableau.nodes) == 1.0 ||
        throw(ArgumentError("the RK continuous extension needs endpoint stages"))
    length(dense_weights) == N && all(isfinite, dense_weights) ||
        throw(ArgumentError("RK dense weights need one finite value per stage"))
    copied = ntuple(index -> Float64(dense_weights[index]), Val(N))
    all(isfinite, copied) ||
        throw(ArgumentError("RK dense weights must fit in binary64"))
    for coefficient in (tableau.solution_weights..., copied...)
        _march_scalar(T, coefficient)
    end
    shape, count = _rk_observation_shape(operator)
    shape == size(state) ||
        throw(DimensionMismatch("the RK dense operator and state shapes disagree"))
    oncpu(execution) || _rk_observation_device_operator(operator, execution)
    failures = _rk_observation_failure_storage(execution, state)
    return RungeKuttaObservationWork(
        tableau.solution_weights, copied, operator,
        dzeros(execution, Complex{T}, shape..., 5),
        dzeros(execution, Complex{T}, shape...), dzeros(execution, Complex{T}, shape...),
        T === Float64 ? dzeros(execution, ComplexF64, shape...) : nothing,
        T === Float32 ? dzeros(execution, ComplexF32, shape...) : nothing,
        dzeros(execution, Bool, shape...), dzeros(execution, Float64, count),
        failures, 0.0, 0.0, 0.0, false, arithmetic,
    )
end

function _rk_observation_wide_prepare(
        operator::DiagonalOperator, z0, z1, weights, execution,
    )
    isfinite(z0) && isfinite(z1) && isfinite(z1 - z0) ||
        throw(ArgumentError("the RK dense operator interval must be finite"))
    return nothing
end
function _rk_observation_wide_prepare(
        operator::RepresentedOperator, z0, z1, weights, execution,
    )
    _check_table_position(operator.cumulative, z0)
    _check_table_position(operator.cumulative, z1)
    return nothing
end
function _rk_observation_wide_prepare(
        operator::ExpansionOperator, z0, z1, weights, execution,
    )
    difference_table!(weights, operator.cumulative_weights, z0, z1, execution)
    return nothing
end
function _rk_observation_wide_prepare(
        operator::SeparableSpatialOperator, z0, z1, weights, execution,
    )
    _spatial_operator_position(operator, z0)
    _spatial_operator_position(operator, z1)
    isfinite(Float64(z1) - Float64(z0)) ||
        throw(ArgumentError("the RK dense operator interval must be finite"))
    return nothing
end

function _rk_observation_phase_body(operator::DiagonalOperator, z0, z1, weights)
    rate, distance = operator.rate, z1 - z0
    return (row, column) -> iszero(distance) ? 0.0im : rate[row, column] * distance
end
function _rk_observation_phase_body(operator::RepresentedOperator, z0, z1, weights)
    coefficients, edges = operator.cumulative.coefficients, operator.cumulative.breakpoints
    rows = operator.n_frequency
    return function (row, column)
        z0 == z1 && return 0.0im
        index = row + rows * (column - 1)
        return _table_value(coefficients, edges, index, z1) -
            _table_value(coefficients, edges, index, z0)
    end
end
function _rk_observation_phase_body(operator::ExpansionOperator, z0, z1, weights)
    terms = operator.terms
    return function (row, column)
        total = 0.0im
        for term in axes(terms, 3)
            total += terms[row, column, term] * weights[term]
        end
        return total
    end
end
function _rk_observation_phase_body(operator::SeparableSpatialOperator, z0, z1, weights)
    tables, distance = _spatial_operator_tables(operator), Float64(z1) - Float64(z0)
    # Reuse the protected high/low root, branch and mask assembly from ordinary
    # propagation. Capture only the rooted tuple of arrays/scalars in worker bodies.
    return (row, column) -> iszero(distance) ? 0.0im :
        first(_spatial_element(tables, row, column)) * distance
end

_rk_observation_preflight(work, operator, origin, step, destination, execution) = nothing

function _rk_observation_narrow_factors!(work, origin, destination, execution, retain)
    return _rk_observation_narrow_factors!(
        work, work.operator, origin, destination, execution, retain,
    )
end

function _rk_observation_narrow_factors!(
        work, operator, origin, destination, execution, retain,
    )
    weights = work.weights
    _rk_observation_wide_prepare(operator, origin, destination, weights, execution)
    factors, stiff, failures = work.factor, work.stiff, work.failures
    phase_at = _rk_observation_phase_body(operator, origin, destination, weights)
    each_column!(execution, size(factors, 2)) do column
        failed = 0
        for row in axes(factors, 1)
            phase = phase_at(row, column)
            finite = isfinite(phase)
            if retain
                stiff[row, column] = abs(real(phase)) > 1.0
            elseif finite
                factor = iszero(phase) ? one(eltype(factors)) :
                    eltype(factors)(exp(phase))
                factors[row, column] = factor
                finite = isfinite(factor)
            else
                # Do not enter libm with an invalid phase inside a worker body.
                factors[row, column] = eltype(factors)(NaN)
            end
            !finite && failed == 0 && (failed = row)
        end
        failures[column] = failed
    end
    for column in eachindex(failures)
        failures[column] == 0 ||
            throw(NonfiniteSourceError(:dense_exponent, failures[column], column))
    end
    return nothing
end

function _rk_observation_exponent!(work, origin, destination, execution)
    operator_exponent!(
        work.exponent, work.operator, origin, destination, execution; work = work.weights,
    )
    _source_finite!(work.failures, work.exponent, execution, :dense_exponent)
    return nothing
end

function _rk_observation_coefficients_finite!(work, execution)
    coefficients = work.coefficients
    failures = _rk_observation_failure_array(work.failures)
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
    return _source_check_rows(
        _rk_observation_failure_host(work.failures, execution), size(coefficients, 2),
        :dense_coefficients,
    )
end

"""
    retain_runge_kutta_observation!(work, initial, endpoint, stages, z, dz,
                                  destination, execution)

Retain a completed accepted RK trial without changing its inputs or session.

`initial` and finished/masked `endpoint` are strided complex matrices matching
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
    _rk_observation_execution(work, execution)
    T = source_type(work.arithmetic)
    _rk_observation_input_types(initial, endpoint, stages, T, execution) ||
        throw(ArgumentError("RK dense retention needs strided arrays in state precision"))
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
    _rk_observation_device_inputs(execution, initial, endpoint, stages)
    _march_scalar(T, dz)
    _rk_observation_preflight(work, work.operator, z, dz, destination, execution)
    work.ready = false
    _source_finite!(work.failures, initial, execution, :dense_initial)
    _source_finite!(work.failures, endpoint, execution, :dense_endpoint)
    _rk_observation_stages_finite(work.failures, stages, execution) ||
        throw(ArgumentError("RK dense stage derivatives must be finite"))
    if work.exponent === nothing
        _rk_observation_narrow_factors!(work, z, raw_end, execution, true)
    else
        _rk_observation_exponent!(work, z, raw_end, execution)
    end
    coefficients, stiff, exponent = work.coefficients, work.stiff, work.exponent
    solution, dense = map(T, work.solution_weights), map(T, work.dense_weights)
    step = T(dz)
    each!(execution, size(initial)...) do row, column
        span = zero(eltype(initial))
        extension = zero(eltype(initial))
        for stage in 1:N
            slope = stages[row, column, stage]
            # Preserve the shared contraction's term order without reassociation.
            span += solution[stage] * slope
            extension += dense[stage] * slope
        end
        span *= step
        first_term = step * stages[row, column, 1] - span
        last_term = span - step * stages[row, column, N] - first_term
        coefficients[row, column, 1] = initial[row, column]
        coefficients[row, column, 2] = span
        coefficients[row, column, 3] = first_term
        coefficients[row, column, 4] = last_term
        coefficients[row, column, 5] = step * extension
        if exponent !== nothing
            stiff[row, column] = abs(real(exponent[row, column])) > 1.0
        end
    end
    _rk_observation_coefficients_finite!(work, execution)
    _source_copy!(work.endpoint, endpoint, execution)
    _source_zero!(work.field, execution)
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
    _rk_observation_execution(work, execution)
    work.ready || throw(ArgumentError("no accepted RK dense window is available"))
    (oncpu(execution) ? active isa StridedVector{Bool} : active isa AbstractVector{Bool}) ||
        throw(ArgumentError("the RK dense mask needs strided unpacked Boolean storage"))
    length(active) == size(work.field, 1) &&
        axes(active, 1) == Base.OneTo(length(active)) ||
        throw(DimensionMismatch("the RK dense mask and retained field disagree"))
    _retained_alias(active, work) &&
        throw(ArgumentError("the RK dense mask must not alias retained storage"))
    _rk_observation_device_inputs(execution, active)
    isfinite(position) && work.position <= position <= work.destination ||
        throw(ArgumentError("the observation lies outside its accepted RK window"))
    field, coefficients = work.field, work.coefficients
    if position == work.destination || position > work.position + work.step
        _source_copy!(field, work.endpoint, execution)
    elseif position == work.position
        each!(execution, size(field)...) do row, column
            field[row, column] = coefficients[row, column, 1]
        end
    else
        coordinate = (position - work.position) / work.step
        scalar = source_type(work.arithmetic)
        theta, rest = scalar(coordinate), scalar(1.0 - coordinate)
        if work.exponent === nothing
            _rk_observation_narrow_factors!(work, work.position, position, execution, false)
        else
            _rk_observation_exponent!(work, work.position, position, execution)
        end
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
        if work.exponent === nothing
            factors = work.factor
            each!(execution, size(field)...) do row, column
                factor = factors[row, column]
                isone(factor) || (field[row, column] *= factor)
            end
        else
            propagate!(field, field, work.exponent, execution)
        end
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
`factor`, `weights`, and `failures`. Device failures include both numerical storage
and its owned host mirror. Absent precision-specific arrays count as zero.
Storage is bounded independently of trials and save planes.
"""
function runge_kutta_observation_memory(work::RungeKuttaObservationWork)
    return (
        coefficients = sizeof(work.coefficients), endpoint = sizeof(work.endpoint),
        field = sizeof(work.field),
        exponent = work.exponent === nothing ? 0 : sizeof(work.exponent),
        factor = work.factor === nothing ? 0 : sizeof(work.factor),
        stiff = sizeof(work.stiff), weights = sizeof(work.weights),
        failures = _rk_observation_failure_bytes(work.failures),
    )
end

mutable struct _RungeKuttaSessionObservation{W}
    work::W
    z::Float64
    selected::Bool
end

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
