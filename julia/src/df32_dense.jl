# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Dense queries borrow the immutable binding, never the march's mutable trial banks.
# Both bounded scalar records retain the complete formation/assembly radii. Array
# storage remains the existing binary32 continuous-extension workspace.
mutable struct _DF32RKDensePhase{B}
    binding::B
    shape::Tuple{Int, Int}
    retained::Union{Nothing, _DF32FlowOperation}
    query::Union{Nothing, _DF32FlowOperation}
    generation::UInt64

    function _DF32RKDensePhase(binding::B, shape::Tuple{Int, Int}) where
        {B <: WideStartBinding}
        return new{B}(binding, shape, nothing, nothing, zero(UInt64))
    end
end

function _rk_dense_admit(flow::DF32LinearFlow, state)
    flow.binding.spec.method === :rk ||
        throw(ArgumentError("DF32 dense observation requires its bound RK method"))
    isempty(flow.binding.spec.schedule.replay_steps) ||
        throw(ArgumentError("interpolated planes require adaptive RK; replay lands"))
    _flow_arithmetic(flow, state, flow.binding.spec.arithmetic)
    return nothing
end

function _rk_dense_prepare(
        execution, state, tableau, dense_weights, flow::DF32LinearFlow; arithmetic,
    )
    return RungeKuttaObservationWork(
        execution, state, tableau, dense_weights, flow; arithmetic,
    )
end

"""
    RungeKuttaObservationWork(execution, state, tableau, dense_weights,
                              flow::DF32LinearFlow; arithmetic)

Prepare the existing binary32 RK extension with independently certified DF32 phases.

`state` has the flow's `(n_omega,n_columns)` shape and original source scales.
`tableau` is the identical translated tableau bound by `flow`; `dense_weights` have
the ordinary continuous-extension contract. Borrow only the immutable start/operator
binding. Retention and each interior query certify their actual coordinates before
changing phase, factor or observed storage. Subsequent trial-bank reuse cannot make a
retained window stale. No source, history, controller or flow generation is changed.

The full-step stiff decision remains `abs(real(phase)) > 1`. An error enclosure
straddling either threshold is refused by name before replacing a retained window.
Exact endpoints and validated landing gaps use the ordinary lossless copy paths.
No additional field-sized array or binary64 numerical phase/factor path is retained.
Two bounded scalar operation slots include every certificate radius; they are host
metadata, not a reusable uniform phase proof. See `DF32_DENSE.md` for the graph.

Throw `ArgumentError` for a different method/arithmetic, replay, unsupported storage
or uncertified phase, and `DimensionMismatch` for a different shape. This internal CPU
leaf does not admit public DF32, spatial or vendor execution. Warmed calls allocate
nothing under the same ownership contract as `RungeKuttaObservationWork`.
"""
function RungeKuttaObservationWork(
        execution::KernelExecution, state, tableau::EmbeddedTableau, dense_weights,
        flow::DF32LinearFlow;
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _rk_dense_admit(flow, state)
    _flow_arithmetic(flow, state, arithmetic)
    tableau === flow.binding.spec.tableau ||
        throw(ArgumentError("DF32 dense weights must use the bound RK tableau"))
    phase = _DF32RKDensePhase(flow.binding, flow.shape)
    return RungeKuttaObservationWork(
        execution, state, tableau, dense_weights, phase; arithmetic,
    )
end

_rk_observation_shape(phase::_DF32RKDensePhase) = (phase.shape, 0)

function _df32_dense_prepared(phase::_DF32RKDensePhase)
    prepared = phase.binding.operator
    prepared === nothing &&
        throw(ArgumentError("a completed DF32 start has no dense trial window"))
    return prepared
end

function _df32_dense_branch(prepared, operation)
    bundle = operation.scalar_certificate
    for row in 1:prod(prepared.shape)
        divided = _wide_graph(
            prepared.coefficients, prepared.terms, prepared.coefficient_errors,
            prepared.term_errors, prepared.kind, row, bundle, Val(:checked),
        )
        component = real(_wide_result(divided, bundle, 0x00, divided))
        # Add the exact-word sum's binary64 rounding to the assembly radius.
        # The extra outward endpoints cover tiny lows invisible to Float64(hi+lo).
        center = _wide_host(Float64(component.value.hi)) +
            _wide_host(Float64(component.value.lo))
        radius = _wide_sum(center.error, component.error)
        lower, upper = prevfloat(center.value - radius), nextfloat(center.value + radius)
        quiet = -1.0 <= lower && upper <= 1.0
        stiff = upper < -1.0 || lower > 1.0
        quiet || stiff ||
            throw(ArgumentError("DF32 dense stiff-branch certificate straddles ±1"))
    end
    return nothing
end

function _rk_observation_preflight(
        work, phase::_DF32RKDensePhase, origin, step, destination, execution,
    )
    phase.generation < typemax(UInt64) ||
        throw(ArgumentError("DF32 dense retention generation capacity is exhausted"))
    site = DF32TrialSite(origin, step, destination, 0)
    prepared = _df32_flow_site(phase, site)
    operation = wide_operation(prepared, origin, origin + step; step)
    _wide_check_operation(operation, :exponent)
    _df32_dense_branch(prepared, operation)
    generation = phase.generation + one(UInt64)
    # Publication follows every fallible certificate/branch check. In particular,
    # a refused replacement leaves the previous ready window and query untouched.
    phase.retained = _df32_flow_operation(operation, generation)
    phase.query = nothing
    phase.generation = generation
    return nothing
end

function _df32_dense_retained(phase::_DF32RKDensePhase, origin, destination)
    operation = phase.retained
    operation === nothing &&
        throw(ArgumentError("DF32 dense retention has no matching phase certificate"))
    operation.generation == phase.generation &&
        iszero(operation.certificate.status) && operation.origin == origin &&
        operation.destination == destination ||
        throw(ArgumentError("DF32 dense retention has no matching phase certificate"))
    return operation
end

@inline function _df32_dense_argument(coefficients, terms, kind, bundle, index)
    divided = _wide_graph(
        coefficients, terms, nothing, nothing, kind, index, bundle, Val(:compute),
    )
    return _wide_result(divided, bundle, 0x00, divided)
end

function _rk_observation_narrow_factors!(
        work, phase::_DF32RKDensePhase, origin, destination, execution, retain,
    )
    prepared = _df32_dense_prepared(phase)
    operation = if retain
        _df32_dense_retained(phase, origin, destination)
    else
        full = _df32_dense_retained(phase, origin, origin + work.step)
        checked = wide_operation(prepared, origin, destination; step = full.step)
        _wide_check_operation(checked, :exponent)
        queried = _df32_flow_operation(checked, phase.generation)
        phase.query = queried
        queried
    end
    coefficients, terms, kind, bundle =
        prepared.coefficients, prepared.terms, prepared.kind, operation.scalars
    factors, stiff = work.factor, work.stiff
    rows = size(factors, 1)
    each!(execution, size(factors)...) do row, column
        argument = _df32_dense_argument(
            coefficients, terms, kind, bundle, row + (column - 1) * rows,
        )
        if retain
            stiff[row, column] = !_df32_abs_leq(real(argument), 1.0f0)
        else
            factors[row, column] = iszero(argument) ? one(ComplexF32) :
                first(_df32_exp_pair(argument))
        end
    end
    return nothing
end
