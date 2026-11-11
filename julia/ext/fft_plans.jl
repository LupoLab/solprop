# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function SolpropJulia.prepare_fft_planning(transfer::Py)
    offered = Dict(pyconvert(Tuple{Vararg{Tuple{String, String}}}, transfer.offered))
    return SolpropJulia.FFTPlanning(;
        offered, effort = Symbol(pyconvert(String, transfer.effort)),
    )
end

function _planning_preparation(preparation::SolpropJulia._ProblemPreparation, planning)
    original = preparation.execution
    SolpropJulia.oncpu(original) ||
        throw(ArgumentError("persisted FFTW planning applies only to CPU execution"))
    execution = SolpropJulia.KernelExecution(
        original.backend, original.tasks, original.check_body; planning,
    )
    constructor = preparation isa SolpropJulia.ModalPreparation ?
        SolpropJulia.ModalPreparation : SolpropJulia.SpatialPreparation
    return constructor(
        execution, preparation.data, preparation.operator,
        preparation.power_weight, preparation.node_chunk,
    )
end

function _planning_preparation(preparation::SolpropJulia.SessionPreparation, planning)
    return SolpropJulia.SessionPreparation(
        _planning_preparation(preparation.problem, planning), preparation.observations,
    )
end
