# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SolpropJuliaPythonCallExt

import PythonCall: Py, PyArray, pyconvert, pyis, pyisinstance, pybuiltins
import SolpropJulia
import SolpropJulia: copy_preparation, prepare_modal, start_runge_kutta, start_adams
import SolpropJulia: start_low_storage
import SolpropJulia: export_session!, session_attempts!, session_orders!

"""
    copy_preparation(values::Py)

Copy a Python numerical array into independently owned Julia storage.

# Arguments
- `values`: A Python array or buffer with fixed-size numerical elements. Shape, element
  type and the translated input's units are preserved, including strided input arrays.

# Returns
A Julia `Array` that does not retain a Python array, buffer or owner.

# Throws
- `ArgumentError`: If the element type is not a fixed-size subtype of `Number`.
- `PythonCall.PyException`: If accessing the Python array's buffer raises an exception.
"""
copy_preparation(values::Py) = copy_preparation(PyArray(values; copy = false))

function _table(execution, transfer)
    return SolpropJulia.ChebyshevTable(
        execution, copy_preparation(transfer.edges),
        copy_preparation(transfer.coefficients);
        lower = pyconvert(Float64, transfer.lower),
        upper = pyconvert(Float64, transfer.upper),
    )
end

_optional_copy(value) = pyis(value, nothing) ? nothing : copy_preparation(value)

function _profile(execution, value, name)
    if pyisinstance(value, pybuiltins.float)
        return SolpropJulia.SourceProfile(pyconvert(Float64, value); name)
    end
    return SolpropJulia.SourceProfile(_table(execution, value); name)
end

function _prefactor(execution, transfer, shape)
    if !pyis(transfer.table, nothing)
        return SolpropJulia.SourcePrefactor(_table(execution, transfer.table), shape)
    elseif !pyis(transfer.weights, nothing)
        return SolpropJulia.SourcePrefactor(
            execution, copy_preparation(transfer.values),
            _table(execution, transfer.weights),
        )
    end
    return SolpropJulia.SourcePrefactor(execution, copy_preparation(transfer.values))
end

function _operator(execution, transfer, shape)
    if !pyis(transfer.table, nothing)
        return SolpropJulia.RepresentedOperator(
            execution, _table(execution, transfer.table), shape...,
        )
    elseif !pyis(transfer.weights, nothing)
        return SolpropJulia.ExpansionOperator(
            execution, copy_preparation(transfer.values),
            _table(execution, transfer.weights),
        )
    end
    return SolpropJulia.DiagonalOperator(execution, copy_preparation(transfer.values))
end

function _power_weight(execution, transfer, shape)
    if !pyis(transfer.table, nothing)
        return SolpropJulia.ObservationWeight(_table(execution, transfer.table), shape)
    elseif !pyis(transfer.weights, nothing)
        return SolpropJulia.ObservationWeight(
            execution, copy_preparation(transfer.values),
            _table(execution, transfer.weights),
        )
    end
    return SolpropJulia.ObservationWeight(execution, copy_preparation(transfer.values))
end

function _kerr(execution, transfer)
    pyis(transfer, nothing) && return nothing
    coefficients = SolpropJulia.KerrCoefficients(
        pyconvert(Float64, transfer.coefficient), Symbol(pyconvert(String, transfer.form));
        self_factor = pyconvert(Float64, transfer.self_factor),
        conjugate_factor = pyconvert(Float64, transfer.conjugate_factor),
        harmonic_factor = pyconvert(Float64, transfer.harmonic_factor),
        take_real = pyconvert(Bool, transfer.take_real),
    )
    return SolpropJulia.ModalKerr(
        coefficients, _profile(execution, transfer.density, :kerr),
    )
end

function _raman(execution, transfer, dt)
    kernel = SolpropJulia.RamanKernel(copy_preparation(transfer.undamped), dt)
    return SolpropJulia.ModalRaman(
        kernel, Tuple(_profile(execution, rate, :raman_decay) for rate in transfer.rates),
        _profile(execution, transfer.density, :raman_density);
        fraction = pyconvert(Float64, transfer.fraction),
        material_factor = pyconvert(Float64, transfer.material_factor),
        thg = pyconvert(Bool, transfer.thg),
    )
end

function _plasma_channel(execution, transfer)
    value = transfer.rate
    rate = SolpropJulia.IonisationRateTable(
        copy_preparation(value.coefficients), pyconvert(Float64, value.field_min),
        pyconvert(Float64, value.field_max), pyconvert(Float64, value.rate_max),
        pyconvert(Float64, value.log_field_min), pyconvert(Float64, value.d_log_field),
    )
    channel = SolpropJulia.PlasmaChannel(
        rate, pyconvert(Float64, transfer.charge_mass_ratio),
        pyconvert(Float64, transfer.ionisation_potential),
    )
    cycle = transfer.cycle
    quadrature = pyis(cycle, nothing) ? nothing : SolpropJulia.CycleQuadrature(
            copy_preparation(cycle.nodes), copy_preparation(cycle.weights),
            pyconvert(Tuple{Vararg{Int}}, cycle.ends);
            moment_rtol = pyconvert(Float64, cycle.moment_rtol),
            negligible_rate = pyconvert(Float64, cycle.negligible_rate),
        )
    return SolpropJulia.ModalPlasmaChannel(
        channel, _profile(execution, transfer.density, :plasma_density),
        _profile(execution, transfer.preionisation, :preionisation); quadrature,
    )
end

function _plasma(execution, transfer, dt)
    pyis(transfer, nothing) && return nothing
    channels = Tuple(_plasma_channel(execution, channel) for channel in transfer.channels)
    if !pyis(transfer.taps, nothing)
        rule = SolpropJulia.RunningIntegralRule(
            copy_preparation(transfer.taps), dt;
            floor_exposure = pyconvert(Bool, transfer.floor_exposure),
        )
        return SolpropJulia.ModalPlasma(execution, channels, rule)
    end
    return SolpropJulia.ModalPlasma(
        execution, channels; drude = copy_preparation(transfer.drude),
        loss = copy_preparation(transfer.loss),
    )
end

function prepare_modal(transfer::Py, threads::Int, node_chunk::Int)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), threads)
    dt = pyconvert(Float64, transfer.dt)
    grid = SolpropJulia.ModalSourceGrid(
        execution; kind = Symbol(pyconvert(String, transfer.kind)),
        n_time = pyconvert(Int, transfer.n_time), dt,
        crop_index = copy_preparation(transfer.crop_index),
        omega = copy_preparation(transfer.omega),
        active = copy_preparation(transfer.active),
        omega_window = copy_preparation(transfer.omega_window),
        source_window = copy_preparation(transfer.source_window),
        source_window_derivative = copy_preparation(transfer.source_window_derivative),
        temporal_rate = _optional_copy(transfer.temporal_rate),
        carrier = _optional_copy(transfer.carrier),
    )
    map = SolpropJulia.ModalSourceMap(
        execution; synthesis = copy_preparation(transfer.synthesis),
        projection = copy_preparation(transfer.projection),
        coarse_projection = copy_preparation(transfer.coarse_projection),
        taper = _profile(execution, transfer.taper, :taper),
    )
    shape = (length(grid.omega), size(map.synthesis, 2))
    data = SolpropJulia.ModalSourceData(
        grid, map, _prefactor(execution, transfer.prefactor, shape);
        kerr = _kerr(execution, transfer.kerr),
        raman = Tuple(_raman(execution, component, dt) for component in transfer.raman),
        plasma = _plasma(execution, transfer.plasma, dt),
    )
    1 <= node_chunk <= size(map.synthesis, 1) ||
        throw(ArgumentError("node chunk must fit the prepared modal quadrature"))
    return SolpropJulia.ModalPreparation(
        execution, data, _operator(execution, transfer.operator, shape),
        _power_weight(execution, transfer.power_weight, shape), node_chunk,
    )
end

function _control(transfer)
    control = transfer.control
    policy = control.controller
    return SolpropJulia.PIControl(;
        safety = pyconvert(Float64, policy.safety),
        max_factor = pyconvert(Float64, policy.max_factor),
        min_factor = pyconvert(Float64, policy.min_factor),
        beta1 = pyconvert(Float64, policy.beta1), beta2 = pyconvert(Float64, policy.beta2),
        lag = pyconvert(Int, policy.lag), reject_free = pyconvert(Bool, policy.reject_free),
        estimator_order = pyconvert(Int, policy.estimator_order),
        start_growth = pyconvert(Float64, transfer.start_growth),
        dz_min = pyconvert(Float64, control.dz_min),
        dz_max = pyconvert(Float64, control.dz_max),
        auto_init = pyconvert(Bool, control.auto_init),
    )
end

function _launch(preparation, transfer, field)
    execution = preparation.execution
    host = copy_preparation(field)
    ndims(host) == 3 && size(host, 2) == 1 ||
        throw(DimensionMismatch("the starting field needs mode/component/frequency axes"))
    state = reshape(permutedims(host, (3, 2, 1)), size(host, 3), size(host, 1))
    control = transfer.control
    norm = SolpropJulia.MarchNorm(
        execution, copy_preparation(transfer.active), copy_preparation(transfer.observed),
        copy_preparation(transfer.weights), pyconvert(Float64, control.controller.rtol),
        pyconvert(Float64, control.controller.atol),
    )
    schedule = SolpropJulia.MarchSchedule(;
        planes = copy_preparation(transfer.planes),
        breakpoints = copy_preparation(transfer.breakpoints),
        hard_kinks = copy_preparation(transfer.hard_kinks),
        stretch = pyconvert(Float64, control.stretch),
        step_planes = pyconvert(Int, transfer.step_planes),
        replay_steps = copy_preparation(transfer.steps),
        replay_edges = copy_preparation(transfer.edges),
    )
    source = SolpropJulia.ModalSource(
        execution, preparation.data; node_chunk = preparation.node_chunk,
    )
    operator = preparation.operator
    work = operator isa SolpropJulia.ExpansionOperator ?
        zeros(size(operator.terms, 3)) : nothing
    flow = SolpropJulia.LinearFlow(execution, operator, state; work)
    return state, norm, schedule, source, flow
end

function start_runge_kutta(
        preparation::SolpropJulia.ModalPreparation, transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
    )
    execution = preparation.execution
    tableau = SolpropJulia.EmbeddedTableau(
        copy_preparation(transfer.nodes), copy_preparation(transfer.matrix),
        copy_preparation(transfer.solution), copy_preparation(transfer.error),
    )
    policy = _control(transfer)
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    return SolpropJulia.RungeKuttaSession(
        execution, state, z, pyconvert(Float64, transfer.control.dz_init), tableau, policy,
        norm, schedule, source, flow; stop, plane_index = plane, plane_pending,
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight),
    )
end

function _adams_control(policy)
    return SolpropJulia.AdamsControlPolicy(;
        order = pyconvert(Int, policy.order),
        variable_order = pyconvert(Bool, policy.variable_order),
        safety = pyconvert(Float64, policy.safety),
        max_growth = pyconvert(Float64, policy.max_growth),
        min_factor = pyconvert(Float64, policy.min_factor),
        rejection_growth = pyconvert(Float64, policy.rejection_growth),
        hysteresis = pyconvert(Float64, policy.hysteresis),
        restart_after = pyconvert(Int, policy.restart_after),
        collar_stability = pyconvert(Float64, policy.collar_stability),
        real_intervals = pyconvert(Tuple{Vararg{Float64}}, policy.real_intervals),
        collar_rate = pyconvert(Float64, policy.collar_rate),
        carrier = pyconvert(Bool, policy.carrier),
        carrier_cap = pyconvert(Float64, policy.carrier_cap),
        max_calls = pyconvert(Union{Nothing, Int}, policy.max_calls),
        equal_approach = pyconvert(Bool, policy.equal_approach),
        interval_cap = pyconvert(Bool, policy.interval_cap),
        dz_min = pyconvert(Float64, policy.dz_min),
        dz_max = pyconvert(Float64, policy.dz_max),
        stretch = pyconvert(Float64, policy.stretch),
        z_end = pyconvert(Float64, policy.z_end),
        save_planes = pyconvert(Vector{Float64}, policy.save_planes),
        hard_kinks = pyconvert(Vector{Float64}, policy.hard_kinks),
        breakpoints = pyconvert(Vector{Float64}, policy.breakpoints),
        landing_ulps = pyconvert(Int, policy.landing_ulps),
        interpolate = pyconvert(Bool, policy.interpolate),
        quiet_interval = pyconvert(Union{Nothing, Float64}, policy.quiet_interval),
    )
end

function start_adams(
        preparation::SolpropJulia.ModalPreparation, transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
    )
    execution = preparation.execution
    policy = _adams_control(transfer.adams)
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    return SolpropJulia.AdamsSession(
        execution, state, z, pyconvert(Float64, transfer.control.dz_init),
        policy, norm, schedule, source, flow;
        stop, plane_index = plane, plane_pending,
        source_absent = pyconvert(Bool, transfer.source_absent),
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight),
    )
end

function start_low_storage(
        preparation::SolpropJulia.ModalPreparation, transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
    )
    execution = preparation.execution
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    return SolpropJulia.LowStorageSession(
        execution, state, z, SolpropJulia.RK4IPTableau(), norm, schedule, source, flow;
        stop, plane_index = plane, plane_pending,
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight),
    )
end

function _python_buffer(::Type{T}, ::Val{N}, value::Py) where {T, N}
    return PyArray{T, N, true}(value; copy = false)
end
function _python_buffer(::Type{T}, ::Val{N}, value::PyArray{T, N, true}) where {T, N}
    return value
end
function _python_buffer(::Type{T}, ::Val{N}, value) where {T, N}
    throw(ArgumentError("the Python destination needs writable $N-dimensional $T storage"))
end

function export_session!(
        output::Union{Py, PyArray}, session, generation::Int,
        preparation::SolpropJulia.ModalPreparation,
        power::Bool, optics::Bool,
    )
    borrowed = _python_buffer(ComplexF64, Val(3), output)
    modes, components, frequencies = size(borrowed)
    wanted = (components * frequencies, frequencies, 1)
    all(axis -> size(borrowed, axis) == 1 || stride(borrowed, axis) == wanted[axis], 1:3) ||
        throw(ArgumentError("a Python export destination must be C-contiguous"))
    GC.@preserve output borrowed begin
        address = Base.unsafe_convert(Ptr{ComplexF64}, borrowed)
        columns = unsafe_wrap(
            Array, address, (frequencies, components * modes); own = false,
        )
        export_session!(
            columns, session, generation, preparation, power, optics,
        )
    end
    return nothing
end

function session_attempts!(
        positions::Py, lengths::Union{Py, PyArray},
        norms::Union{Py, PyArray}, accepted::Union{Py, PyArray},
        elapsed::Union{Py, PyArray},
        session, generation::Int,
    )
    return session_attempts!(
        _python_buffer(Float64, Val(1), positions),
        _python_buffer(Float64, Val(1), lengths),
        _python_buffer(Float64, Val(1), norms),
        _python_buffer(Bool, Val(1), accepted),
        _python_buffer(Float64, Val(1), elapsed), session, generation,
    )
end

function session_orders!(output::Py, session::SolpropJulia.AdamsSession, generation::Int)
    return session_orders!(_python_buffer(Int64, Val(1), output), session, generation)
end

end
