# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _transverse_line(transfer::Py)
    return SolpropJulia.TransverseLineData(
        Symbol(pyconvert(String, transfer.axis)), copy_preparation(transfer.sample_scale);
        strip = copy_preparation(transfer.strip),
        window = copy_preparation(transfer.window),
        spacing = pyconvert(Float64, transfer.spacing),
        measure = pyconvert(Float64, transfer.measure),
        inward = pyconvert(NTuple{2, Int}, transfer.inward),
        onset = pyis(transfer.onset, nothing) ? nothing :
            pyconvert(NTuple{3, Int}, transfer.onset),
        onset_step = pyconvert(Float64, transfer.onset_step),
    )
end

function _observation_transverse(execution, prepared, transfer::Py)
    pyis(transfer, nothing) && return nothing
    prepared isa SolpropJulia.SpatialObservationPreparation ||
        throw(ArgumentError("transverse monitoring requires scalar spatial observations"))
    value = transfer.policy
    policy = SolpropJulia.TransversePolicy(;
        warn = pyconvert(Float64, value.warn), stop = pyconvert(Float64, value.stop),
        backstop_start = pyconvert(Float64, value.backstop_start),
        backstop_max = pyconvert(Float64, value.backstop_max),
        reach_growth = pyconvert(Float64, value.reach_growth),
        guard_numerator = pyconvert(Float64, value.guard_numerator),
        k_min = pyconvert(Float64, value.k_min),
        k_floor_fraction = pyconvert(Float64, value.k_floor_fraction),
        due_tolerance = pyconvert(Float64, value.due_tolerance),
        noise_factor = pyconvert(Float64, value.noise_factor),
        strip_floor_margin = pyconvert(Float64, value.strip_floor_margin),
        strip_leakage = pyconvert(Float64, value.strip_leakage),
        has_design = pyconvert(Bool, value.has_design),
        has_collar = pyconvert(Bool, value.has_collar),
    )
    tables = SolpropJulia.TransverseTables(
        copy_preparation(transfer.faces), copy_preparation(transfer.clip),
        copy_preparation(transfer.totals), copy_preparation(transfer.column_weights),
        copy_preparation(transfer.kappa), copy_preparation(transfer.order),
        copy_preparation(transfer.ramp_rates);
        face_ranks = pyconvert(Tuple{Vararg{Int}}, transfer.face_ranks),
        noise_face_share = pyconvert(Float64, transfer.noise_face_share),
        noise_modes = pyconvert(Int, transfer.noise_modes),
    )
    return SolpropJulia.TransversePreparation(
        execution, prepared.map, policy, tables;
        n_frequency = length(prepared.grid.omega), rows = copy_preparation(transfer.rows),
        lines = Tuple(_transverse_line(line) for line in transfer.lines),
        declared = pyconvert(Bool, transfer.declared),
    )
end

_transverse_continuation(::Nothing) = nothing
function _transverse_continuation(transfer::Py)
    pyis(transfer, nothing) && return nothing
    return SolpropJulia.TransverseContinuation(
        pyconvert(Float64, transfer.launch),
        pyconvert(Tuple{Vararg{Float64}}, transfer.baseline),
        pyconvert(Tuple{Vararg{Float64}}, transfer.peak),
        pyconvert(Tuple{Vararg{Float64}}, transfer.peak_raw),
        pyconvert(Float64, transfer.allowance),
        pyconvert(Float64, transfer.ramp_raw), pyconvert(Float64, transfer.ramp_noise),
        pyconvert(Float64, transfer.noise_share_in_ramp),
        pyconvert(Float64, transfer.declared_share),
        pyconvert(Float64, transfer.spatial_removed), pyconvert(Float64, transfer.current),
        pyconvert(Float64, transfer.current_baseline),
        pyconvert(Float64, transfer.strip_baseline),
        pyconvert(Float64, transfer.strip_floor), pyconvert(Float64, transfer.strip_peak),
        pyconvert(Float64, transfer.returned_peak), pyconvert(Int, transfer.reads),
        pyconvert(Float64, transfer.largest_gap), pyconvert(Float64, transfer.warned_at),
        SolpropJulia.TransverseStatus(pyconvert(Int, transfer.status)),
    )
end
