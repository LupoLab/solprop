# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _launch_flow(
        execution, operator, state, schedule, transfer, z, tableau, ::Nothing;
        adams_policy = nothing,
    )
    work = operator isa SolpropJulia.ExpansionOperator ?
        zeros(size(operator.terms, 3)) : nothing
    return SolpropJulia.LinearFlow(
        execution, operator, state; work,
        tabulate = pyconvert(Bool, transfer.tabulate_flow),
    )
end

function _wide_history(wide::Py)
    history = wide.history
    pyis(history, nothing) && return nothing
    positions = copy_preparation(history.positions)
    slots = copy_preparation(history.slots)
    identities = UInt8[
        kind == "predicted" for kind in pyconvert(Tuple{Vararg{String}}, history.identities)
    ]
    raw = history.rows.get("rates", nothing)
    rates = if pyis(raw, nothing)
        nothing
    else
        copied = copy_preparation(raw)
        ndims(copied) == 5 && size(copied, 2) == 1 ||
            throw(DimensionMismatch("DF32 history rates require one trajectory"))
        reshape(
            permutedims(copied, (5, 4, 3, 2, 1)),
            size(copied, 5), size(copied, 3) * size(copied, 4), size(copied, 1),
        )
    end
    origin = pyis(wide.origin_wide, nothing) ? :unknown :
        Symbol(pyconvert(String, wide.origin_wide))
    return SolpropJulia.WideStartHistory(
        positions, identities, slots; rates, origin_wide = origin
    )
end

function _launch_flow(
        execution, operator, state, schedule, transfer, z, tableau, wide::Py;
        adams_policy = nothing,
    )
    pyis(wide, nothing) &&
        return _launch_flow(
        execution, operator, state, schedule, transfer, z, tableau, nothing; adams_policy
    )
    pyis(wide.run, transfer) ||
        throw(ArgumentError("DF32 launch must use its exact owned run transfer"))
    adams = !pyis(transfer.adams, nothing)
    adams == (adams_policy isa SolpropJulia.AdamsControlPolicy) ||
        throw(ArgumentError("DF32 Adams launch needs its actual session policy"))
    adams || pyis(wide.history, nothing) ||
        throw(ArgumentError("this DF32 march keeps no Adams history"))
    pyis(transfer.dense_weights, nothing) ||
        throw(ArgumentError("DF32 dense observations are not qualified yet"))
    spans = wide.operands.spans
    isequal(z, pyconvert(Float64, spans.start)) ||
        throw(ArgumentError("DF32 launch does not match its actual bound start"))
    geometry = z == last(schedule.planes) ? nothing : SolpropJulia.WideGeometry(
            pyconvert(Float64, spans.position_lower),
            pyconvert(Float64, spans.position_upper);
            history_lower = pyconvert(Float64, spans.oldest),
            history_upper = pyconvert(Float64, spans.position_upper),
            step_max = pyconvert(Float64, spans.step_max),
            step_min = pyconvert(Union{Nothing, Float64}, spans.step_min),
        )
    low_storage = pyconvert(Bool, transfer.low_storage)
    spec = SolpropJulia.WideStartSpec(
        z, low_storage ? nothing : tableau, schedule;
        geometry, arithmetic = _arithmetic(transfer.arithmetic),
        method = adams ? :adams : low_storage ? :low_storage : :rk,
        order_max = adams ? adams_policy.order : 0,
        source_absent = pyconvert(Bool, transfer.source_absent),
    )
    if adams
        return SolpropJulia.DF32AdamsFlow(
            execution, operator, state, spec, adams_policy;
            history = _wide_history(wide),
            tabulate = pyconvert(Bool, transfer.tabulate_flow),
        )
    end
    return SolpropJulia.DF32LinearFlow(
        execution, operator, state, spec;
        tabulate = pyconvert(Bool, transfer.tabulate_flow),
    )
end
