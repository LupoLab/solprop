# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public AdamsHistory, AdamsStencil, AdamsCoefficients, AdamsWork
public adams_stencil!, prepare_adams_coefficients!, adams_seed!, adams_trial!
public adams_accept!, adams_error_norms!
public AdamsDense, adams_retain_dense!, adams_observe!
public adams_copy_history!, adams_copy_dense!

@inline function _adams_capacity(order::Integer)
    1 <= order <= 6 || throw(ArgumentError("exponential Adams supports orders one to six"))
    return Int(order)
end

"""
    AdamsHistory(execution, state::AbstractMatrix{ComplexF64}, max_order;
                 compensated=false)

Prepare owned binary64 source history with `max_order+2` reusable slots.

Each numerical array has spectral shape `(n_omega,n_columns,n_slots)`. The source is
the untransported physical RHS evaluated at its actual predictor sample. Compensated
history additionally retains that sampled state and the derivative of cumulative Phi
there; it never replaces the predictor with a corrected accepted state. Uncompensated
state/rate arrays have zero slots. Positions are metres, identities are caller-owned
integers, and occupied flags are unpacked Booleans. Selection and eviction belong to
the session, not this storage record. Orders one through six are admitted.
"""
struct AdamsHistory{A}
    source::A
    states::A
    rates::A
    positions::Vector{Float64}
    identities::Vector{Int}
    occupied::Vector{Bool}
    max_order::Int
    compensated::Bool
end

function AdamsHistory(
        execution::KernelExecution, state::AbstractMatrix{ComplexF64}, max_order::Integer;
        compensated::Bool = false,
    )
    maximum_order = _adams_capacity(max_order)
    isempty(state) && throw(ArgumentError("Adams state must be nonempty"))
    rows, columns = size(state)
    capacity = maximum_order + 2
    stored = compensated ? capacity : 0
    return AdamsHistory(
        dzeros(execution, ComplexF64, rows, columns, capacity),
        dzeros(execution, ComplexF64, rows, columns, stored),
        dzeros(execution, ComplexF64, rows, columns, stored),
        zeros(capacity), zeros(Int, capacity), fill(false, capacity),
        maximum_order, compensated,
    )
end

"""
    AdamsStencil(max_order)

Prepare fixed-capacity storage for a session-selected exponential Adams stencil.

Use `adams_stencil!` to provide the predictor order, newest-first source slots and up
to three scored orders. `nodes` then stores the dimensionless old positions relative
to a trial, filled by `prepare_adams_coefficients!`. This record makes no controller
or history-selection decision. All inactive entries are zeroed on every update.
"""
mutable struct AdamsStencil
    slots::Vector{Int}
    orders::Vector{Int}
    nodes::Vector{Float64}
    order::Int
    count::Int
    n_orders::Int
    max_order::Int
end

function AdamsStencil(max_order::Integer)
    maximum_order = _adams_capacity(max_order)
    return AdamsStencil(
        zeros(Int, maximum_order), zeros(Int, 3), zeros(maximum_order),
        0, 0, 0, maximum_order,
    )
end

"""
    adams_stencil!(stencil, order, slots, scored_orders)

Copy explicit slot and order selections into prepared storage without choosing policy.

`slots` contains distinct positive integer slots in newest-first order. There must be
at least as many slots as the largest selected order. `scored_orders` has one through
three distinct orders, includes `order`, and is bounded by the prepared maximum.
Return `nothing`; invalid selections throw before changing the record.
"""
function adams_stencil!(stencil::AdamsStencil, order::Integer, slots, scored_orders)
    if (
            slots isa AbstractArray &&
                (
                Base.mightalias(slots, stencil.slots) ||
                    Base.mightalias(slots, stencil.orders) ||
                    Base.mightalias(slots, stencil.nodes)
            )
        ) ||
            (
            scored_orders isa AbstractArray &&
                (
                Base.mightalias(scored_orders, stencil.slots) ||
                    Base.mightalias(scored_orders, stencil.orders) ||
                    Base.mightalias(scored_orders, stencil.nodes)
            )
        )
        throw(ArgumentError("Adams selections must be disjoint from stencil storage"))
    end
    1 <= order <= stencil.max_order || throw(ArgumentError("invalid Adams trial order"))
    1 <= length(scored_orders) <= 3 ||
        throw(ArgumentError("an Adams trial scores one through three orders"))
    length(slots) <= stencil.max_order ||
        throw(DimensionMismatch("Adams slot selection exceeds prepared capacity"))
    largest = Int(order)
    found = false
    for index in eachindex(scored_orders)
        current = scored_orders[index]
        current isa Integer && 1 <= current <= stencil.max_order ||
            throw(ArgumentError("invalid Adams scored order"))
        for earlier in 1:(index - 1)
            current != scored_orders[earlier] ||
                throw(ArgumentError("Adams scored orders must be distinct"))
        end
        largest = max(largest, current)
        found |= current == order
    end
    found || throw(ArgumentError("Adams scored orders must include the trial order"))
    length(slots) >= largest || throw(DimensionMismatch("Adams history is too short"))
    for index in eachindex(slots)
        slots[index] isa Integer && slots[index] > 0 ||
            throw(ArgumentError("Adams slots must be positive integers"))
        for earlier in 1:(index - 1)
            slots[index] != slots[earlier] ||
                throw(ArgumentError("Adams slots must be distinct"))
        end
    end
    fill!(stencil.slots, 0)
    fill!(stencil.orders, 0)
    fill!(stencil.nodes, 0)
    for index in eachindex(slots)
        stencil.slots[index] = slots[index]
    end
    for index in eachindex(scored_orders)
        stencil.orders[index] = scored_orders[index]
    end
    stencil.order = order
    stencil.count = length(slots)
    stencil.n_orders = length(scored_orders)
    return nothing
end

"""
    AdamsCoefficients(max_order)

Prepare all Julia-owned coefficient and scalar polynomial work for one Adams trial.

Predictor/corrector matrices are `(max_order+1,max_order+1)`, with sample rows and
ascending monomial columns. Factored defects retain divided-difference sample weights
and the old-node product polynomial separately, for up to three scored orders.
Compensation coefficients are ordinary integrated Lagrange weights. Nodes contain
the endpoint one followed by the selected old nodes; all reused padding is overwritten.
"""
struct AdamsCoefficients
    predictor::Matrix{Float64}
    corrector::Matrix{Float64}
    defect_samples::Matrix{Float64}
    defect_polynomial::Matrix{Float64}
    compensation_predictor::Vector{Float64}
    compensation_corrector::Vector{Float64}
    compensation_defects::Matrix{Float64}
    nodes::Vector{Float64}
    basis::Matrix{Float64}
    polynomial::Vector{Float64}
    product::Vector{Float64}
end

function AdamsCoefficients(max_order::Integer)
    width = _adams_capacity(max_order) + 1
    return AdamsCoefficients(
        zeros(width, width), zeros(width, width), zeros(width, 3), zeros(width, 3),
        zeros(width), zeros(width), zeros(width, 3), zeros(width),
        zeros(width, width), zeros(width), zeros(width),
    )
end

function _adams_basis!(output, nodes, first, count, polynomial, product)
    fill!(output, 0)
    for sample in 1:count
        fill!(polynomial, 0)
        polynomial[1] = 1
        degree_count = 1
        divisor = 1.0
        point = nodes[first + sample - 1]
        # Form each basis directly from its factors. Deflating a common polynomial
        # loses small coefficients for remote nonuniform history nodes (SEP 16).
        for other in 1:count
            other == sample && continue
            root = nodes[first + other - 1]
            fill!(product, 0)
            for power in 1:degree_count
                product[power + 1] = polynomial[power]
            end
            for power in 1:degree_count
                product[power] -= root * polynomial[power]
            end
            degree_count += 1
            copyto!(polynomial, product)
            divisor *= point - root
        end
        for power in 1:count
            output[sample, power] = polynomial[power] / divisor
        end
    end
    return nothing
end

@inline function _adams_integral(basis, sample, count, upper)
    integral = 0.0
    for power in count:-1:1
        integral = (integral + basis[sample, power] / power) * upper
    end
    return integral
end

"""
    prepare_adams_coefficients!(coefficients, stencil, history, position, step)

Build nonuniform exponential and compensation weights entirely inside Julia.

The finite positive step and positions are in metres. Selected occupied slots have
strictly decreasing actual source-sampling positions. Normalized old nodes are
`(sample_position-position)/step`; the newest node need not be exactly zero when the
session snaps a landed endpoint. Landing/proximity admission stays with the session.
Invalid history or nonfinite weights throw; no history is mutated.
Return `nothing` after overwriting every coefficient and inactive cell.
"""
function prepare_adams_coefficients!(
        coefficients::AdamsCoefficients, stencil::AdamsStencil,
        history::AdamsHistory, position::Real, step::Real,
    )
    stencil.max_order == history.max_order == size(coefficients.predictor, 1) - 1 ||
        throw(DimensionMismatch("Adams coefficient capacities disagree"))
    stencil.order > 0 || throw(ArgumentError("the Adams stencil has no selection"))
    isfinite(position) && isfinite(step) && step > 0 && isfinite(position + step) &&
        position + step > position || throw(ArgumentError("invalid Adams trial interval"))
    for index in 1:stencil.count
        slot = stencil.slots[index]
        slot <= length(history.positions) && history.occupied[slot] ||
            throw(ArgumentError("Adams selected an unoccupied history slot"))
        point = history.positions[slot]
        isfinite(point) || throw(ArgumentError("Adams history positions must be finite"))
        if index > 1
            point < history.positions[stencil.slots[index - 1]] ||
                throw(ArgumentError("Adams history must run strictly newest first"))
        end
        node = (point - position) / step
        isfinite(node) || throw(ArgumentError("Adams normalized history must be finite"))
        stencil.nodes[index] = node
    end
    fill!(coefficients.nodes, 0)
    coefficients.nodes[1] = 1
    for index in 1:stencil.count
        coefficients.nodes[index + 1] = stencil.nodes[index]
    end
    nodes = coefficients.nodes
    polynomial, product, basis =
        coefficients.polynomial, coefficients.product, coefficients.basis
    order = stencil.order
    _adams_basis!(coefficients.predictor, nodes, 2, order, polynomial, product)
    _adams_basis!(coefficients.corrector, nodes, 1, order + 1, polynomial, product)
    fill!(coefficients.compensation_predictor, 0)
    fill!(coefficients.compensation_corrector, 0)
    for sample in 1:order
        coefficients.compensation_predictor[sample] =
            _adams_integral(coefficients.predictor, sample, order, 1.0)
    end
    for sample in 1:(order + 1)
        coefficients.compensation_corrector[sample] =
            _adams_integral(coefficients.corrector, sample, order + 1, 1.0)
    end
    fill!(coefficients.defect_samples, 0)
    fill!(coefficients.defect_polynomial, 0)
    fill!(coefficients.compensation_defects, 0)
    for score in 1:stencil.n_orders
        scored = stencil.orders[score]
        for sample in 1:(scored + 1)
            divisor = 1.0
            for other in 1:(scored + 1)
                sample == other && continue
                divisor *= nodes[sample] - nodes[other]
            end
            coefficients.defect_samples[sample, score] = 1 / divisor
        end
        fill!(polynomial, 0)
        polynomial[1] = 1
        for index in 1:scored
            fill!(product, 0)
            for power in 1:index
                product[power + 1] = polynomial[power]
            end
            for power in 1:index
                product[power] -= nodes[index + 1] * polynomial[power]
            end
            copyto!(polynomial, product)
        end
        for power in 1:(scored + 1)
            coefficients.defect_polynomial[power, score] = polynomial[power]
        end
        _adams_basis!(basis, nodes, 1, scored + 1, polynomial, product)
        for sample in 1:(scored + 1)
            coefficients.compensation_defects[sample, score] =
                _adams_integral(basis, sample, scored + 1, 1.0)
        end
        _adams_basis!(basis, nodes, 2, scored, polynomial, product)
        for sample in 1:scored
            coefficients.compensation_defects[sample + 1, score] -=
                _adams_integral(basis, sample, scored, 1.0)
        end
    end
    all(isfinite, coefficients.predictor) && all(isfinite, coefficients.corrector) &&
        all(isfinite, coefficients.defect_samples) &&
        all(isfinite, coefficients.defect_polynomial) &&
        all(isfinite, coefficients.compensation_predictor) &&
        all(isfinite, coefficients.compensation_corrector) &&
        all(isfinite, coefficients.compensation_defects) ||
        throw(ArgumentError("Adams coefficients are nonfinite"))
    return nothing
end

"""
    AdamsWork(execution, state::AbstractMatrix{ComplexF64}, max_order;
              compensated=false)

Allocate bounded binary64 buffers for exponential Adams prediction, correction and norms.

State-shaped matrices retain exponent, flow, predictor, newly sampled source, candidate
and endpoint rate/compensation. Moments and carried history have `max_order+1` and
`max_order` planes, and defects have three planes. Coefficients are generated in Julia.
This workspace does not contain a controller, accepted-state owner or Python callback.
"""
mutable struct AdamsWork{A, B, R}
    exponent::A
    flow::A
    predictor::A
    source::A
    candidate::A
    endpoint_rate::A
    endpoint_compensation::A
    moments::B
    carried::B
    defects::B
    norm_parts::R
    coefficients::AdamsCoefficients
    max_order::Int
    compensated::Bool
    position::Float64
    step::Float64
    order::Int
    n_orders::Int
    source_absent::Bool
    ready::Bool
end

function AdamsWork(
        execution::KernelExecution, state::AbstractMatrix{ComplexF64}, max_order::Integer;
        compensated::Bool = false,
    )
    maximum_order = _adams_capacity(max_order)
    isempty(state) && throw(ArgumentError("Adams state must be nonempty"))
    rows, columns = size(state)
    matrices = ntuple(unused -> dzeros(execution, ComplexF64, rows, columns), 7)
    return AdamsWork(
        matrices..., dzeros(execution, ComplexF64, rows, columns, maximum_order + 1),
        dzeros(execution, ComplexF64, rows, columns, compensated ? maximum_order : 0),
        dzeros(execution, ComplexF64, rows, columns, 3),
        dzeros(execution, Float64, 2, columns), AdamsCoefficients(maximum_order),
        maximum_order, compensated, 0.0, 0.0, 0, 0, false, false,
    )
end

@inline function _adams_callback(result)
    result === nothing || throw(ArgumentError("Adams numerical callbacks return nothing"))
    return nothing
end

@inline function _adams_slot(history, slot, state)
    1 <= slot <= length(history.positions) || throw(ArgumentError("invalid Adams slot"))
    size(state) == (size(history.source, 1), size(history.source, 2)) ||
        throw(DimensionMismatch("Adams history and state shapes disagree"))
    return nothing
end

"""
    adams_seed!(history, slot, state, position, source!, rate!, execution;
                identity=slot, before=false, source_absent=false)

Store one actual state/source sample at a caller-selected history slot.

The Julia callbacks `source!(out,state,z)` and `rate!(out,z,before)` return `nothing`
or throw outside a kernel. Rates are needed only for transported compensation.
`source_absent=true` is an explicit structural declaration and invokes neither
callback; compensated history refuses that combination. The slot becomes occupied
only after successful sampling and copying. Return the actual source-call count.
"""
function adams_seed!(
        history::AdamsHistory, slot::Integer, state, position::Real,
        source!, rate!, execution::KernelExecution;
        identity::Integer = slot, before::Bool = false, source_absent::Bool = false,
    )
    _adams_slot(history, slot, state)
    isfinite(position) || throw(ArgumentError("Adams sample position must be finite"))
    source_absent && history.compensated &&
        throw(ArgumentError("structurally absent sources do not use compensation"))
    history.occupied[slot] = false
    source = view(history.source, :, :, slot)
    if source_absent
        fill!(source, 0)
    else
        _adams_callback(source!(source, state, position))
    end
    if history.compensated
        rate = view(history.rates, :, :, slot)
        _adams_callback(rate!(rate, position, before))
        states = history.states
        each!(execution, size(state)...) do row, column
            states[row, column, slot] = state[row, column]
        end
    end
    history.positions[slot] = position
    history.identities[slot] = identity
    history.occupied[slot] = true
    return source_absent ? 0 : 1
end

function _adams_trial_shapes(work, state, history, stencil)
    size(state) == size(work.candidate) ==
        (size(history.source, 1), size(history.source, 2)) ||
        throw(DimensionMismatch("Adams trial state shapes disagree"))
    work.max_order == history.max_order == stencil.max_order ||
        throw(DimensionMismatch("Adams trial capacities disagree"))
    work.compensated == history.compensated ||
        throw(ArgumentError("Adams trial and history compensation policies disagree"))
    if Base.mightalias(state, work.exponent) || Base.mightalias(state, work.flow) ||
            Base.mightalias(state, work.predictor) || Base.mightalias(state, work.source) ||
            Base.mightalias(state, work.candidate) ||
            Base.mightalias(state, work.endpoint_rate) ||
            Base.mightalias(state, work.endpoint_compensation) ||
            Base.mightalias(state, work.moments) || Base.mightalias(state, work.carried) ||
            Base.mightalias(state, work.defects)
        throw(ArgumentError("Adams accepted state must not alias trial scratch"))
    end
    return nothing
end

function _adams_predict!(work, state, history, stencil, step, execution)
    exponent, flow, moments, predictor =
        work.exponent, work.flow, work.moments, work.predictor
    source, states, rates = history.source, history.states, history.rates
    carried, slots, nodes = work.carried, stencil.slots, stencil.nodes
    coefficients = work.coefficients.predictor
    compensation = work.coefficients.compensation_predictor
    order, count, compensated = stencil.order, stencil.count, work.compensated
    width = size(moments, 3)
    capacity = work.max_order
    each!(execution, size(state)...) do row, column
        phase = exponent[row, column]
        linear = iszero(phase) ? one(phase) : exp(phase)
        flow[row, column] = linear
        phi = _phi_values(phase, Val(8))
        factor = 1.0
        for power in 1:width
            moments[row, column, power] = factor * phi[power + 1]
            factor *= power
        end
        increment = zero(phase)
        for power in 1:order
            combined = zero(phase)
            for sample in 1:order
                combined += coefficients[sample, power] *
                    source[row, column, slots[sample]]
            end
            increment += moments[row, column, power] * combined
        end
        if compensated
            correction = zero(phase)
            reference = phase / step
            for sample in 1:capacity
                carried_value = zero(phase)
                if sample <= count
                    slot = slots[sample]
                    difference = rates[row, column, slot] - reference
                    # Subtract generators before multiplying the actual sampled
                    # state; only forward transport is needed for retained rows.
                    if !iszero(difference)
                        carried_value = exp((1 - nodes[sample]) * phase) *
                            (difference * states[row, column, slot])
                    end
                end
                carried[row, column, sample] = carried_value
                if sample <= order
                    correction += compensation[sample] * carried_value
                end
            end
            increment += correction
        end
        baseline = iszero(phase) ? state[row, column] : linear * state[row, column]
        predictor[row, column] = baseline + step * increment
    end
    return nothing
end

function _adams_correct!(work, state, history, stencil, step, execution)
    exponent, flow, moments, source, predictor, candidate =
        work.exponent, work.flow, work.moments, work.source, work.predictor, work.candidate
    store, slots, carried, defects =
        history.source, stencil.slots, work.carried, work.defects
    rate, endpoint = work.endpoint_rate, work.endpoint_compensation
    coefficients = work.coefficients.corrector
    sample_weights, polynomial =
        work.coefficients.defect_samples, work.coefficients.defect_polynomial
    compensation = work.coefficients.compensation_corrector
    compensation_defects = work.coefficients.compensation_defects
    orders, order, n_orders = stencil.orders, stencil.order, stencil.n_orders
    compensated = work.compensated
    each!(execution, size(state)...) do row, column
        phase = exponent[row, column]
        newest = source[row, column]
        endpoint_value = compensated ?
            (rate[row, column] - phase / step) * predictor[row, column] : zero(phase)
        endpoint[row, column] = endpoint_value
        increment = zero(phase)
        for power in 1:(order + 1)
            combined = coefficients[1, power] * newest
            for sample in 1:order
                combined += coefficients[sample + 1, power] *
                    store[row, column, slots[sample]]
            end
            increment += moments[row, column, power] * combined
        end
        if compensated
            correction = compensation[1] * endpoint_value
            for sample in 1:order
                correction += compensation[sample + 1] * carried[row, column, sample]
            end
            increment += correction
        end
        baseline = iszero(phase) ? state[row, column] :
            flow[row, column] * state[row, column]
        candidate[row, column] = baseline + step * increment
        for score in 1:3
            defect = zero(phase)
            if score <= n_orders
                scored = orders[score]
                difference = sample_weights[1, score] * newest
                for sample in 1:scored
                    difference += sample_weights[sample + 1, score] *
                        store[row, column, slots[sample]]
                end
                moment = zero(phase)
                for power in 1:(scored + 1)
                    moment += polynomial[power, score] * moments[row, column, power]
                end
                defect = difference * moment
                if compensated
                    correction = compensation_defects[1, score] * endpoint_value
                    for sample in 1:scored
                        correction += compensation_defects[sample + 1, score] *
                            carried[row, column, sample]
                    end
                    defect += correction
                end
                defect *= step
            end
            defects[row, column, score] = defect
        end
    end
    return nothing
end

function _adams_homogeneous!(work, state, execution)
    exponent, flow, predictor, candidate =
        work.exponent, work.flow, work.predictor, work.candidate
    each!(execution, size(state)...) do row, column
        phase = exponent[row, column]
        factor = iszero(phase) ? one(phase) : exp(phase)
        value = iszero(phase) ? state[row, column] : factor * state[row, column]
        flow[row, column] = factor
        predictor[row, column] = value
        candidate[row, column] = value
    end
    return nothing
end

"""
    adams_trial!(work, state, position, step, history, stencil,
                 source!, exponent!, rate!, execution; source_absent=false)

Form one production exponential Adams PEC trial without mutating accepted state/history.

The positive step and positions are metres. Callbacks `source!(out,state,z)`,
`exponent!(out,z0,z1)` and `rate!(out,z,before)` are Julia-only and return `nothing`
or throw. Coefficients and up to three factored defects are formed in owned storage.
The source is evaluated exactly once, at the predictor, and the endpoint rate uses
`before=true`. Transported compensation is
`exp((1-theta_j)*x)*(D_j-x/step)*sampled_state_j`, following SEP 16 section 4.2.
Rates differentiate the same represented cumulative operator as the step exponent.

`source_absent=true` explicitly selects exact homogeneous propagation, with zero source
calls/defects and no compensation. Numerical zeros never imply that structural policy.
Return the source-call count. A thrown callback leaves `work.ready=false`; the caller
records invocation/failure accounting before entering the callback. No controller,
mask, collar, frame shift or interaction-picture policy is implemented here.
"""
function adams_trial!(
        work::AdamsWork, state, position::Real, step::Real,
        history::AdamsHistory, stencil::AdamsStencil, source!, exponent!, rate!,
        execution::KernelExecution; source_absent::Bool = false,
    )
    work.ready = false
    _adams_trial_shapes(work, state, history, stencil)
    source_absent && work.compensated &&
        throw(ArgumentError("structurally absent sources do not use compensation"))
    prepare_adams_coefficients!(work.coefficients, stencil, history, position, step)
    _adams_callback(exponent!(work.exponent, position, position + step))
    all(value -> isfinite(value) && real(value) <= 700, work.exponent) ||
        throw(ArgumentError("Adams exponents must be finite with real part at most 700"))
    if source_absent
        _adams_homogeneous!(work, state, execution)
        fill!(work.source, 0)
        fill!(work.defects, 0)
        fill!(work.endpoint_rate, 0)
        fill!(work.endpoint_compensation, 0)
        fill!(work.moments, 0)
    else
        _adams_predict!(work, state, history, stencil, step, execution)
        _adams_callback(source!(work.source, work.predictor, position + step))
        if work.compensated
            _adams_callback(rate!(work.endpoint_rate, position + step, true))
        else
            fill!(work.endpoint_rate, 0)
        end
        _adams_correct!(work, state, history, stencil, step, execution)
    end
    work.position, work.step = position, step
    work.order, work.n_orders = stencil.order, stencil.n_orders
    work.source_absent = source_absent
    work.ready = true
    return source_absent ? 0 : 1
end

"""
    AdamsDense(execution, state::AbstractMatrix{ComplexF64}, max_order;
               compensated=false)

Prepare independently owned storage for one accepted exponential Adams dense step.

The record retains both committed endpoints, cumulative exponent, corrector polynomial
and actual source samples. Compensated storage additionally keeps the sampled states
and represented rates. Retain before reusing history slots; observing calls no source
and cannot change the march history. This numerical leaf does not admit additional
user-facing plane policies or apply a mask/collar.
"""
mutable struct AdamsDense{A, B}
    initial::A
    endpoint::A
    exponent::A
    source::B
    states::B
    rates::B
    coefficients::Matrix{Float64}
    nodes::Vector{Float64}
    weights::Vector{Float64}
    max_order::Int
    compensated::Bool
    source_absent::Bool
    position::Float64
    step::Float64
    order::Int
    ready::Bool
end

function AdamsDense(
        execution::KernelExecution, state::AbstractMatrix{ComplexF64}, max_order::Integer;
        compensated::Bool = false,
    )
    maximum_order = _adams_capacity(max_order)
    isempty(state) && throw(ArgumentError("Adams dense state must be nonempty"))
    rows, columns = size(state)
    width = maximum_order + 1
    stored = compensated ? width : 0
    return AdamsDense(
        dzeros(execution, ComplexF64, rows, columns),
        dzeros(execution, ComplexF64, rows, columns),
        dzeros(execution, ComplexF64, rows, columns),
        dzeros(execution, ComplexF64, rows, columns, width),
        dzeros(execution, ComplexF64, rows, columns, stored),
        dzeros(execution, ComplexF64, rows, columns, stored),
        zeros(width, width), zeros(width), zeros(width), maximum_order,
        compensated, false, 0.0, 0.0, 0, false,
    )
end

"""
    adams_retain_dense!(dense, work, old_state, accepted_state, history, stencil, execution)

Copy a completed accepted step before the session overwrites any selected history slot.

`accepted_state` is the actual committed endpoint supplied by the session, including
its admitted mask/collar decisions; `old_state` is the step's committed start. Interior
evaluation retains the trial's source polynomial and actual predictor/rate samples.
All buffers, including inactive sample planes, are overwritten. Return `nothing`.
"""
function adams_retain_dense!(
        dense::AdamsDense, work::AdamsWork, old_state, accepted_state,
        history::AdamsHistory, stencil::AdamsStencil, execution::KernelExecution,
    )
    work.ready || throw(ArgumentError("no completed Adams trial is available"))
    dense.max_order == work.max_order == history.max_order == stencil.max_order &&
        dense.compensated == work.compensated == history.compensated ||
        throw(ArgumentError("Adams dense storage does not match its trial"))
    size(dense.initial) == size(old_state) ==
        size(accepted_state) == size(work.candidate) ||
        throw(DimensionMismatch("Adams dense endpoint shapes disagree"))
    stencil.order == work.order || throw(ArgumentError("Adams dense stencil changed"))
    for index in 1:work.order
        slot = stencil.slots[index]
        history.occupied[slot] &&
            (history.positions[slot] - work.position) / work.step ==
            work.coefficients.nodes[index + 1] ||
            throw(ArgumentError("Adams dense history was overwritten before retention"))
    end
    dense.ready = false
    copyto!(dense.initial, old_state)
    copyto!(dense.endpoint, accepted_state)
    copyto!(dense.exponent, work.exponent)
    copyto!(dense.coefficients, work.coefficients.corrector)
    copyto!(dense.nodes, work.coefficients.nodes)
    fill!(dense.weights, 0)
    output, sample_states, sample_rates = dense.source, dense.states, dense.rates
    store, states, rates = history.source, history.states, history.rates
    newest, predictor, endpoint_rate = work.source, work.predictor, work.endpoint_rate
    slots, order, compensated = stencil.slots, work.order, work.compensated
    width = size(output, 3)
    each!(execution, size(old_state)...) do row, column
        for sample in 1:width
            value, sampled_state, sampled_rate = zero(eltype(output)),
                zero(eltype(output)), zero(eltype(output))
            if sample == 1
                value = newest[row, column]
                if compensated
                    sampled_state = predictor[row, column]
                    sampled_rate = endpoint_rate[row, column]
                end
            elseif sample <= order + 1
                slot = slots[sample - 1]
                value = store[row, column, slot]
                if compensated
                    sampled_state = states[row, column, slot]
                    sampled_rate = rates[row, column, slot]
                end
            end
            output[row, column, sample] = value
            if compensated
                sample_states[row, column, sample] = sampled_state
                sample_rates[row, column, sample] = sampled_rate
            end
        end
    end
    dense.position, dense.step, dense.order = work.position, work.step, work.order
    dense.source_absent = work.source_absent
    dense.ready = true
    return nothing
end

"""
    adams_observe!(output, dense, position, execution)

Evaluate the retained accepted corrector polynomial without sampling the RHS.

The position is in metres inside the retained step. Exact endpoints copy the supplied
committed states. Interior source moments are
`k! * xi^(k+1) * phi_(k+1)(xi*x)` with `xi=(position-start)/step`.
Transported compensation uses ordinary Lagrange integrals to `xi`. For compensated
bins with `real(x)<-1`, ADR-039 selects the anchor's homogeneous flow; this branch is
taken before forming an exponentially amplified inverse endpoint transport.
Return `output`, disjoint from retained numerical storage. No mask/collar is implicit.
"""
function adams_observe!(
        output, dense::AdamsDense, position::Real, execution::KernelExecution,
    )
    dense.ready || throw(ArgumentError("no retained Adams dense step is available"))
    size(output) == size(dense.initial) ||
        throw(DimensionMismatch("Adams dense output has the wrong shape"))
    isfinite(position) && dense.position <= position <= dense.position + dense.step ||
        throw(ArgumentError("Adams observation lies outside its retained step"))
    if Base.mightalias(output, dense.initial) || Base.mightalias(output, dense.endpoint) ||
            Base.mightalias(output, dense.exponent) ||
            Base.mightalias(output, dense.source) ||
            Base.mightalias(output, dense.states) || Base.mightalias(output, dense.rates)
        throw(ArgumentError("Adams dense output must not overwrite retained data"))
    end
    if position == dense.position
        copyto!(output, dense.initial)
        return output
    elseif position == dense.position + dense.step
        copyto!(output, dense.endpoint)
        return output
    end
    fraction = (position - dense.position) / dense.step
    count = dense.order + 1
    coefficients, weights, nodes = dense.coefficients, dense.weights, dense.nodes
    for sample in eachindex(weights)
        weights[sample] = sample <= count ?
            _adams_integral(coefficients, sample, count, fraction) : 0.0
    end
    initial, exponent, source = dense.initial, dense.exponent, dense.source
    states, rates = dense.states, dense.rates
    step, compensated, source_absent = dense.step, dense.compensated, dense.source_absent
    each!(execution, size(output)...) do row, column
        phase = exponent[row, column]
        scaled = fraction * phase
        baseline = iszero(scaled) ? initial[row, column] :
            exp(scaled) * initial[row, column]
        if source_absent || (compensated && real(phase) < -1)
            output[row, column] = baseline
        else
            phi = _phi_values(scaled, Val(8))
            increment = zero(phase)
            factor, power_of_fraction = 1.0, fraction
            for power in 1:count
                combined = zero(phase)
                for sample in 1:count
                    combined += coefficients[sample, power] * source[row, column, sample]
                end
                increment += factor * power_of_fraction * phi[power + 1] * combined
                factor *= power
                power_of_fraction *= fraction
            end
            if compensated
                reference = phase / step
                for sample in 1:count
                    difference = rates[row, column, sample] - reference
                    if !iszero(difference)
                        transported = exp((fraction - nodes[sample]) * phase)
                        increment += weights[sample] * transported *
                            (difference * states[row, column, sample])
                    end
                end
            end
            output[row, column] = baseline + step * increment
        end
    end
    return output
end

"""
    adams_accept!(history, slot, work, execution; identity=slot)

Store a successful trial's actual predictor source after the session accepts its step.

The slot retains the untransported endpoint source, predictor sample and its rate, not
the corrected candidate. Its position is `work.position+work.step`. The session must
retain any dense step before reusing a selected history slot. This function commits
no state, mask, collar, order or controller decision. Return `nothing`.
"""
function adams_accept!(
        history::AdamsHistory, slot::Integer, work::AdamsWork,
        execution::KernelExecution; identity::Integer = slot,
    )
    work.ready || throw(ArgumentError("no completed Adams trial is available"))
    _adams_slot(history, slot, work.source)
    work.max_order == history.max_order && work.compensated == history.compensated ||
        throw(ArgumentError("Adams acceptance storage does not match its trial"))
    store, states, rates = history.source, history.states, history.rates
    source, predictor, endpoint_rate = work.source, work.predictor, work.endpoint_rate
    compensated = history.compensated
    each!(execution, size(source)...) do row, column
        store[row, column, slot] = source[row, column]
        if compensated
            states[row, column, slot] = predictor[row, column]
            rates[row, column, slot] = endpoint_rate[row, column]
        end
    end
    history.positions[slot] = work.position + work.step
    history.identities[slot] = identity
    history.occupied[slot] = true
    return nothing
end

"""
    adams_error_norms!(output, work, state, mask, weights, rtol, atol, execution)

Score each factored defect against the old accepted state's global magnitude.

For each in-band row the divisor is `rtol*norm(state[mask,:])*weights[row]+atol`,
following exponential Adams' old-state scale rather than the Runge–Kutta candidate
maximum. Preparation supplies positive weights, a Boolean row mask and nonnegative
tolerances. The CPU reduction reuses protected scaled sum-of-squares primitives to
avoid squaring physical amplitudes. Nonfinite old/candidate states, including masked
entries, produce infinite scores. Zero numerator and zero divisor contribute zero.
`output` has at least `work.n_orders` entries; inactive entries are zero. Return `output`.
"""
function adams_error_norms!(
        output, work::AdamsWork, state, mask, weights, rtol, atol,
        execution::KernelExecution,
    )
    oncpu(execution) || throw(ArgumentError("Adams norm device reduction is not admitted"))
    work.ready || throw(ArgumentError("no completed Adams trial is available"))
    length(output) >= work.n_orders ||
        throw(DimensionMismatch("Adams norm output is short"))
    size(state) == size(work.candidate) &&
        length(mask) == length(weights) == size(state, 1) ||
        throw(DimensionMismatch("Adams norm state and row inputs disagree"))
    fill!(output, 0)
    parts, candidate, defects = work.norm_parts, work.candidate, work.defects
    rows, columns = size(state)
    each_column!(execution, columns) do column
        column_scale, column_sum = 0.0, 0.0
        for row in 1:rows
            if !isfinite(state[row, column]) || !isfinite(candidate[row, column])
                column_scale, column_sum = Inf, 1.0
            elseif mask[row]
                column_scale, column_sum =
                    _rk_sumsq(column_scale, column_sum, real(state[row, column]))
                column_scale, column_sum =
                    _rk_sumsq(column_scale, column_sum, imag(state[row, column]))
            end
        end
        parts[1, column], parts[2, column] = column_scale, column_sum
    end
    scale, total = 0.0, 0.0
    for column in 1:columns
        scale, total = _rk_merge_sumsq(scale, total, parts[1, column], parts[2, column])
    end
    magnitude = scale * sqrt(total)
    if !isfinite(magnitude)
        for score in 1:work.n_orders
            output[score] = Inf
        end
        return output
    end
    relative = rtol * magnitude
    for score in 1:work.n_orders
        each_column!(execution, columns) do column
            part_scale, part_sum = 0.0, 0.0
            for row in 1:rows
                if mask[row]
                    divisor = relative * weights[row] + atol
                    value = defects[row, column, score]
                    if !isfinite(divisor)
                        part_scale, part_sum = Inf, 1.0
                    elseif !(iszero(divisor) && iszero(value))
                        part_scale, part_sum = _rk_sumsq(
                            part_scale, part_sum, real(value) / divisor,
                        )
                        part_scale, part_sum = _rk_sumsq(
                            part_scale, part_sum, imag(value) / divisor,
                        )
                    end
                end
            end
            parts[1, column], parts[2, column] = part_scale, part_sum
        end
        error_scale, error_sum = 0.0, 0.0
        for column in 1:columns
            error_scale, error_sum = _rk_merge_sumsq(
                error_scale, error_sum, parts[1, column], parts[2, column],
            )
        end
        output[score] = error_scale * sqrt(error_sum)
    end
    return output
end

"""
    adams_copy_history!(destination, source)

Copy exact numerical history and slot identities into compatible prepared storage.

Source rows, actual predictor states/rates, positions, occupancy and identity integers
are copied without reinterpretation, reordering or source evaluation. The session owns
snapshot format and logical history policy. Return `nothing`; incompatible capacities
or compensation policies throw before mutation.
"""
function adams_copy_history!(destination::AdamsHistory, source::AdamsHistory)
    destination.max_order == source.max_order &&
        destination.compensated == source.compensated &&
        size(destination.source) == size(source.source) ||
        throw(ArgumentError("Adams history copy requires compatible prepared storage"))
    copyto!(destination.source, source.source)
    copyto!(destination.states, source.states)
    copyto!(destination.rates, source.rates)
    copyto!(destination.positions, source.positions)
    copyto!(destination.identities, source.identities)
    copyto!(destination.occupied, source.occupied)
    return nothing
end

"""
    adams_copy_dense!(destination, source)

Copy one exact retained numerical step into compatible independently owned storage.

This leaf supports session snapshots without evaluating the source or rebuilding any
coefficient. Endpoint, source, sampled-state/rate and polynomial arrays keep their exact
values, including inactive storage. Scalar validity and interval metadata are preserved.
The session owns snapshot format and admission. Return `nothing`.
"""
function adams_copy_dense!(destination::AdamsDense, source::AdamsDense)
    destination.max_order == source.max_order &&
        destination.compensated == source.compensated &&
        size(destination.initial) == size(source.initial) ||
        throw(ArgumentError("Adams dense copy requires compatible prepared storage"))
    copyto!(destination.initial, source.initial)
    copyto!(destination.endpoint, source.endpoint)
    copyto!(destination.exponent, source.exponent)
    copyto!(destination.source, source.source)
    copyto!(destination.states, source.states)
    copyto!(destination.rates, source.rates)
    copyto!(destination.coefficients, source.coefficients)
    copyto!(destination.nodes, source.nodes)
    copyto!(destination.weights, source.weights)
    destination.source_absent = source.source_absent
    destination.position, destination.step = source.position, source.step
    destination.order, destination.ready = source.order, source.ready
    return nothing
end
