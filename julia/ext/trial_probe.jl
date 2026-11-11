# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function SolpropJulia.start_trial_probe(
        preparation::Union{
            SolpropJulia._ProblemPreparation, SolpropJulia.SessionPreparation,
        },
        transfer::Py, field, z::Float64, noise = nothing, resume::Bool = false,
        planning = nothing,
    )
    if planning !== nothing
        preparation = _planning_preparation(preparation, planning)
    end
    problem = SolpropJulia._problem_preparation(preparation)
    execution = problem.execution
    state, norm, schedule, source, flow = _launch(problem, transfer, field)
    tableau = if pyconvert(Bool, transfer.low_storage)
        SolpropJulia.RK4IPTableau()
    else
        # Adams timing probes use the request's embedded startup tableau; no Adams
        # policy, history or continuation is read or initialized by this operation.
        SolpropJulia.EmbeddedTableau(
            copy_preparation(transfer.nodes), copy_preparation(transfer.matrix),
            copy_preparation(transfer.solution), copy_preparation(transfer.error),
        )
    end
    split = if preparation isa SolpropJulia.SessionPreparation
        policy = preparation.observations
        policy.split_rate === nothing ? nothing :
            SolpropJulia._temporal_split_workspace(
                execution, policy.observation, policy.split_rate,
            )
    else
        nothing
    end
    return SolpropJulia.TrialProbe(
        execution, state, z, pyconvert(Float64, transfer.control.dz_init),
        tableau, norm, schedule, source, flow;
        split, noise = _launch_noise(problem, noise), resume,
        arithmetic = SolpropJulia._prepared_arithmetic(problem.data),
    )
end

function SolpropJulia.copy_trial_probe_field!(
        output::Union{Py, PyArray},
        probe::SolpropJulia.TrialProbe{<:SolpropJulia.EmbeddedTableau},
    )
    borrowed = _python_buffer(ComplexF64, Val(3), output)
    modes, components, frequencies = size(borrowed)
    components == 1 ||
        throw(DimensionMismatch("a timing-probe host output needs scalar components"))
    wanted = (components * frequencies, frequencies, 1)
    all(axis -> size(borrowed, axis) == 1 || stride(borrowed, axis) == wanted[axis], 1:3) ||
        throw(ArgumentError("a timing-probe host output must be C-contiguous"))
    GC.@preserve output borrowed begin
        address = Base.unsafe_convert(Ptr{ComplexF64}, borrowed)
        columns = unsafe_wrap(
            Array, address, (frequencies, components * modes); own = false,
        )
        SolpropJulia.copy_trial_probe_field!(columns, probe; physical = true)
    end
    return nothing
end
