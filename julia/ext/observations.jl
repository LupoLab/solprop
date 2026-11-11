# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _observation_profile(execution, transfer, name)
    pyis(transfer, nothing) && return nothing
    return _profile(execution, transfer.value, name)
end

function _observation_populations(modal, transfer)
    pyis(transfer, nothing) && return nothing
    profiles = Tuple(
        _profile(modal.execution, value, :species_density)
            for value in transfer.species_densities
    )
    data = modal.data
    if data isa SolpropJulia.CartesianSourceData
        return SolpropJulia.SpatialObservationPopulations(
            data.plasma, data.grid; species_densities = profiles,
            channel_species = pyconvert(Tuple{Vararg{Int}}, transfer.channel_species),
            peak_species = pyconvert(Int, transfer.peak_species),
            pad_scale = data.pad_scale,
        )
    end
    data = data isa SolpropJulia.RadialSourceData ? data.modal : data
    return SolpropJulia.ObservationPopulations(
        data.plasma, data.grid, data.map; species_densities = profiles,
        channel_species = pyconvert(Tuple{Vararg{Int}}, transfer.channel_species),
        peak_species = pyconvert(Int, transfer.peak_species),
    )
end

function _observation_temporal(transfer)
    pyis(transfer, nothing) && return nothing
    policy = transfer.policy
    lowered = SolpropJulia.TemporalMonitorPolicy(;
        warn = pyconvert(Float64, policy.warn), stop = pyconvert(Float64, policy.stop),
        activity = pyconvert(Float64, policy.activity),
        quiet_share = pyconvert(Float64, policy.quiet_share),
        quiet_checks = pyconvert(Int, policy.quiet_checks),
        quiet_collar = pyconvert(Bool, policy.quiet_collar),
        cadence = pyconvert(Int, policy.cadence),
        interval = pyconvert(Float64, policy.interval),
        launch_floor_ratio = pyconvert(Float64, policy.launch_floor_ratio),
        floor_margin = pyconvert(Float64, policy.floor_margin),
        split = pyconvert(Bool, policy.split),
        crossing_share = pyconvert(Float64, policy.crossing_share),
        bucket_limits = copy_preparation(policy.bucket_limits),
    )
    weights = SolpropJulia.TemporalMonitorWeights(
        copy_preparation(transfer.regions), copy_preparation(transfer.columns);
        buckets = _optional_copy(transfer.buckets),
    )
    return SolpropJulia.TemporalObservationData(lowered, weights)
end

function prepare_observations(
        original::SolpropJulia._ProblemPreparation, transfer::Py,
        node_chunk::Int = original.node_chunk,
    )
    modal = SolpropJulia.with_node_chunk(original, node_chunk)
    execution = modal.execution
    grid = SolpropJulia.StatisticsGrid(
        copy_preparation(transfer.grid.omega), copy_preparation(transfer.grid.time),
        copy_preparation(transfer.grid.parseval);
        windows = copy_preparation(transfer.grid.windows),
        spectral_density = pyis(transfer.grid.spectral_density, nothing) ? Float64[] :
            copy_preparation(transfer.grid.spectral_density),
        d_omega = pyconvert(Float64, transfer.grid.d_omega),
    )
    slots = Tuple(
        SolpropJulia.StatisticSlot(
            Symbol(pyconvert(String, slot.name)), pyconvert(Int, slot.offset),
            pyconvert(Int, slot.count),
        ) for slot in transfer.slots
    )
    selection = SolpropJulia.StatisticsSelection(
        slots, pyconvert(Tuple{Vararg{Int}}, transfer.periodic_indices),
    )
    profiles = SolpropJulia.ObservationProfiles(;
        density = _observation_profile(execution, transfer.profiles.density, :density),
        pressure = _observation_profile(execution, transfer.profiles.pressure, :pressure),
        core_radius = _observation_profile(
            execution, transfer.profiles.core_radius, :radius,
        ),
    )
    n_modes = pyconvert(Int, transfer.n_modes)
    prepared = if !pyis(transfer.spatial, nothing)
        _spatial_observation_preparation(modal, transfer, grid, selection, profiles)
    else
        SolpropJulia.ObservationPreparation(
            execution, grid, selection;
            kind = Symbol(pyconvert(String, transfer.kind)), n_modes,
            power_weight = _power_weight(
                execution, transfer.power_weight, (length(grid.omega), n_modes),
            ),
            profiles, axis_synthesis = _optional_copy(transfer.axis_synthesis),
            taper = _profile(execution, transfer.taper, :taper),
            intensity_factor = pyconvert(Float64, transfer.intensity_factor),
            populations = _observation_populations(modal, transfer.populations),
            temporal = _observation_temporal(transfer.temporal),
        )
    end
    policy = SolpropJulia.SessionObservationPolicy(
        prepared; period = pyconvert(Int, transfer.period),
        can_rest = pyconvert(Bool, transfer.can_rest),
        split_rate = _optional_copy(transfer.split_rate),
        transverse = _observation_transverse(execution, prepared, transfer.transverse),
        trace_chunk = pyis(transfer.transverse, nothing) ? 1 :
            pyconvert(Int, transfer.transverse.trace_chunk),
    )
    return SolpropJulia.SessionPreparation(modal, policy)
end

# PythonCall's PyArray has its own object identity, which does not establish
# ownership of a foreign interior view. PythonCall 0.9.37 retains byte strides;
# Base.strides refuses foreign steps not divisible by the item size.
# Keep bytes here so partial-item overlap is diagnosed before any destination write.
function SolpropJulia._transfer_storage_span(
        array::PyArray{T, N, M, L, T},
    ) where {T, N, M, L}
    isempty(array) && return nothing
    isbitstype(T) ||
        throw(ArgumentError("transfer storage requires fixed-size foreign elements"))
    GC.@preserve array begin
        address = UInt(Base.unsafe_convert(Ptr{T}, array))
        byte_strides = getfield(array, :strides)
        return SolpropJulia._transfer_strided_span(
            address, size(array), byte_strides, sizeof(T),
        )
    end
end


function SolpropJulia._transfer_writable_destination(
        array::PyArray{T, N, true, L, T},
    ) where {T, N, L}
    isempty(array) && return nothing
    isbitstype(T) ||
        throw(ArgumentError("transfer destinations require fixed-size foreign elements"))
    byte_strides = getfield(array, :strides)
    return SolpropJulia._transfer_writable_strides(size(array), byte_strides, sizeof(T))
end

function copy_session_observations!(
        positions, steps, values, plane, session, generation::Int,
    )
    borrowed_positions = _python_buffer(Float64, Val(1), positions)
    borrowed_steps = _python_buffer(Int64, Val(1), steps)
    borrowed_values = _python_buffer(Float64, Val(2), values)
    borrowed_plane = _python_buffer(Float64, Val(1), plane)
    owners = (borrowed_positions, borrowed_steps, borrowed_values, borrowed_plane)
    for owner in owners
        isempty(owner) && continue
        SolpropJulia._transfer_writable_destination(owner)
        expected = 1
        for axis in ndims(owner):-1:1
            size(owner, axis) == 1 || stride(owner, axis) == expected ||
                throw(ArgumentError("Python observation buffers must be C-contiguous"))
            expected = Base.checked_mul(expected, size(owner, axis))
        end
    end
    # The core owns all pairwise and retained-storage checks. Transposition only
    # changes the C matrix's view; the foreign allocation owners outlive every write.
    GC.@preserve owners positions steps values plane begin
        SolpropJulia.session_observations!(
            borrowed_positions, borrowed_steps, transpose(borrowed_values), borrowed_plane,
            session, generation,
        )
    end
    return nothing
end
