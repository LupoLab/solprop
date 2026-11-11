# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _launch_flow(execution, operator, state, schedule, transfer, z, tableau, ::Nothing)
    work = operator isa SolpropJulia.ExpansionOperator ?
        zeros(size(operator.terms, 3)) : nothing
    return SolpropJulia.LinearFlow(
        execution, operator, state; work,
        tabulate = pyconvert(Bool, transfer.tabulate_flow),
    )
end

function _launch_flow(execution, operator, state, schedule, transfer, z, tableau, wide::Py)
    pyis(wide, nothing) &&
        return _launch_flow(execution, operator, state, schedule, transfer, z, tableau, nothing)
    pyis(wide.run, transfer) ||
        throw(ArgumentError("DF32 launch must use its exact owned run transfer"))
    pyis(wide.history, nothing) ||
        throw(ArgumentError("DF32 restored Adams execution is not qualified yet"))
    pyis(transfer.adams, nothing) ||
        throw(ArgumentError("DF32 Adams complete factors are not qualified yet"))
    pyis(transfer.dense_weights, nothing) ||
        throw(ArgumentError("DF32 dense observations are not qualified yet"))
    spans = wide.operands.spans
    isequal(z, pyconvert(Float64, spans.start)) ||
        throw(ArgumentError("DF32 launch does not match its actual bound start"))
    geometry = z == last(schedule.planes) ? nothing : SolpropJulia.WideGeometry(
            pyconvert(Float64, spans.position_lower),
            pyconvert(Float64, spans.position_upper);
            history_lower = pyconvert(Float64, spans.oldest),
            history_upper = pyconvert(Float64, spans.end),
            step_max = pyconvert(Float64, spans.step_max),
            step_min = pyconvert(Union{Nothing, Float64}, spans.step_min),
        )
    low_storage = pyconvert(Bool, transfer.low_storage)
    spec = SolpropJulia.WideStartSpec(
        z, low_storage ? nothing : tableau, schedule;
        geometry, arithmetic = _arithmetic(transfer.arithmetic),
        method = low_storage ? :low_storage : :rk, order_max = 0,
        source_absent = pyconvert(Bool, transfer.source_absent),
    )
    return SolpropJulia.DF32LinearFlow(
        execution, operator, state, spec;
        tabulate = pyconvert(Bool, transfer.tabulate_flow),
    )
end
