# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _separable(execution, transfer)
    return SolpropJulia.SeparableSpatialOperator(
        execution;
        k_squared = copy_preparation(transfer.k_squared),
        k_squared_low = copy_preparation(transfer.k_squared_low),
        kperp_sq = copy_preparation(transfer.kperp_sq),
        kperp_sq_low = copy_preparation(transfer.kperp_sq_low),
        frame = copy_preparation(transfer.frame),
        spectral_rate = copy_preparation(transfer.spectral_rate),
        prefactor_numerator = copy_preparation(transfer.prefactor_numerator),
        prefactor_clamped = copy_preparation(transfer.prefactor_clamped),
        active = copy_preparation(transfer.active),
        evanescent_rate = pyconvert(Float64, transfer.evanescent_rate),
        z_span = pyconvert(NTuple{2, Float64}, transfer.z_span),
        kperp_rate = _optional_copy(transfer.kperp_rate),
    )
end

function prepare_spatial(transfer::Py, threads::Int, node_chunk::Int)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), threads)
    kind = pyconvert(String, transfer.kind)
    if kind == "radial"
        modal = prepare_modal(transfer.modal, threads, node_chunk)
        data = SolpropJulia.RadialSourceData(
            modal.data, copy_preparation(transfer.spatial_rate),
            pyconvert(Float64, transfer.pad_scale),
        )
        return SolpropJulia.SpatialPreparation(
            execution, data, modal.operator, modal.power_weight, node_chunk,
        )
    end
    kind == "cartesian" || throw(ArgumentError("unknown scalar spatial layout"))
    source = transfer.source
    grid = _source_grid(execution, source)
    operator = _separable(execution, transfer.operator)
    shape = pyconvert(NTuple{2, Int}, transfer.shape)
    data = SolpropJulia.CartesianSourceData(
        grid, shape, operator, copy_preparation(transfer.spatial_rate),
        pyconvert(Float64, transfer.pad_scale);
        kerr = _kerr(execution, source.kerr, grid.arithmetic),
        raman = Tuple(
            _raman(execution, component, grid.dt, grid.arithmetic) for component in source.raman
        ),
        plasma = _plasma(execution, source.plasma, grid.dt, grid.arithmetic),
    )
    1 <= node_chunk <= prod(shape) ||
        throw(ArgumentError("node chunk must fit the prepared spatial grid"))
    return SolpropJulia.SpatialPreparation(
        execution, data, operator,
        nothing,
        node_chunk,
    )
end

function _spatial_observation_preparation(preparation, transfer, grid, selection, profiles)
    spatial, execution, data = transfer.spatial, preparation.execution, preparation.data
    map = if data isa SolpropJulia.RadialSourceData
        SolpropJulia.RadialObservationMap(
            data.modal.map.synthesis; pad_scale = pyconvert(Float64, spatial.pad_scale),
        )
    elseif data isa SolpropJulia.CartesianSourceData
        SolpropJulia.CartesianObservationMap(data.shape)
    else
        throw(ArgumentError("spatial observations require a matching spatial source"))
    end
    flux = SolpropJulia.SpatialFluxData(
        _separable(execution, spatial.flux);
        material = pyis(spatial.material, nothing) ? nothing :
            _table(execution, spatial.material),
        light_speed = pyconvert(Float64, spatial.light_speed),
    )
    width = spatial.widths
    widths = pyis(width, nothing) ? nothing : SolpropJulia.SpatialWidthGrid(
            Symbol(pyconvert(String, width.kind)), copy_preparation(width.x),
            copy_preparation(width.y), copy_preparation(width.weights),
        )
    return SolpropJulia.SpatialObservationPreparation(
        execution, grid, selection;
        kind = Symbol(pyconvert(String, transfer.kind)), map, flux,
        spectral_weights = copy_preparation(spatial.spectral_weights),
        on_axis = copy_preparation(spatial.on_axis),
        intensity_factor = pyconvert(Float64, transfer.intensity_factor), widths, profiles,
        populations = _observation_populations(preparation, transfer.populations),
        temporal = _observation_temporal(transfer.temporal),
    )
end
