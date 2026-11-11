# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SolpropJuliaPythonCallExt

import PythonCall: Py, PyArray, pyconvert, pyis, pyisinstance, pybuiltins
import SolpropJulia
import SolpropJulia: copy_preparation, prepare_modal, start_runge_kutta, start_adams
import SolpropJulia: start_low_storage, prepare_spatial
import SolpropJulia: copy_session_noise_phase!
import SolpropJulia: prepare_observations, copy_session_observations!
import SolpropJulia: copy_session_monitor_rows!, restore_session_monitor!
import SolpropJulia: export_session!, session_attempts!, session_orders!
import SolpropJulia: export_adams_history!, restore_adams_history!, restore_session_counts!

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

function _arithmetic(transfer)
    name = pyconvert(String, transfer.name)
    name in ("double", "single") || throw(ArgumentError("unknown state precision"))
    return SolpropJulia.SourceArithmetic(
        name == "single" ? Float32 : Float64;
        state_exponent = pyconvert(Int, transfer.state_exponent),
        field_exponent = pyconvert(Int, transfer.field_exponent),
    )
end

function _profile(execution, value, name)
    if pyisinstance(value, pybuiltins.float)
        return SolpropJulia.SourceProfile(pyconvert(Float64, value); name)
    end
    return SolpropJulia.SourceProfile(_table(execution, value); name)
end

function _prefactor(execution, transfer, shape, arithmetic)
    if !pyis(transfer.table, nothing)
        return SolpropJulia.SourcePrefactor(
            _table(execution, transfer.table), shape; arithmetic,
        )
    elseif !pyis(transfer.weights, nothing)
        return SolpropJulia.SourcePrefactor(
            execution, copy_preparation(transfer.values),
            _table(execution, transfer.weights); arithmetic,
        )
    end
    return SolpropJulia.SourcePrefactor(
        execution, copy_preparation(transfer.values); arithmetic,
    )
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
    pyis(transfer, nothing) && return nothing
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

function _kerr(execution, transfer, arithmetic)
    pyis(transfer, nothing) && return nothing
    coefficients = SolpropJulia.KerrCoefficients(
        pyconvert(Float64, transfer.coefficient), Symbol(pyconvert(String, transfer.form));
        self_factor = pyconvert(Float64, transfer.self_factor),
        conjugate_factor = pyconvert(Float64, transfer.conjugate_factor),
        harmonic_factor = pyconvert(Float64, transfer.harmonic_factor),
        take_real = pyconvert(Bool, transfer.take_real),
    )
    return SolpropJulia.ModalKerr(
        coefficients, _profile(execution, transfer.density, :kerr); arithmetic,
    )
end

function _raman(execution, transfer, dt, arithmetic)
    kernel = SolpropJulia.RamanKernel(copy_preparation(transfer.undamped), dt)
    return SolpropJulia.ModalRaman(
        kernel, Tuple(_profile(execution, rate, :raman_decay) for rate in transfer.rates),
        _profile(execution, transfer.density, :raman_density);
        fraction = pyconvert(Float64, transfer.fraction),
        material_factor = pyconvert(Float64, transfer.material_factor),
        thg = pyconvert(Bool, transfer.thg),
        arithmetic,
    )
end

function _plasma_channel(execution, transfer, arithmetic)
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
        _profile(execution, transfer.preionisation, :preionisation); quadrature, arithmetic,
    )
end

function _plasma(execution, transfer, dt, arithmetic)
    pyis(transfer, nothing) && return nothing
    channels = Tuple(
        _plasma_channel(execution, channel, arithmetic) for channel in transfer.channels
    )
    if !pyis(transfer.taps, nothing)
        rule = SolpropJulia.RunningIntegralRule(
            copy_preparation(transfer.taps), dt;
            floor_exposure = pyconvert(Bool, transfer.floor_exposure),
        )
        return SolpropJulia.ModalPlasma(execution, channels, rule; arithmetic)
    end
    return SolpropJulia.ModalPlasma(
        execution, channels; drude = copy_preparation(transfer.drude),
        loss = copy_preparation(transfer.loss),
        arithmetic,
    )
end

function _source_grid(execution, transfer)
    return SolpropJulia.ModalSourceGrid(
        execution; kind = Symbol(pyconvert(String, transfer.kind)),
        n_time = pyconvert(Int, transfer.n_time), dt = pyconvert(Float64, transfer.dt),
        crop_index = copy_preparation(transfer.crop_index),
        omega = copy_preparation(transfer.omega),
        active = copy_preparation(transfer.active),
        omega_window = copy_preparation(transfer.omega_window),
        source_window = copy_preparation(transfer.source_window),
        source_window_derivative = copy_preparation(transfer.source_window_derivative),
        temporal_rate = _optional_copy(transfer.temporal_rate),
        carrier = _optional_copy(transfer.carrier),
        arithmetic = _arithmetic(transfer.arithmetic),
    )
end

function _modal_data(execution, transfer)
    grid = _source_grid(execution, transfer)
    dt = grid.dt
    map = SolpropJulia.ModalSourceMap(
        execution; synthesis = copy_preparation(transfer.synthesis),
        projection = copy_preparation(transfer.projection),
        coarse_projection = _optional_copy(transfer.coarse_projection),
        taper = _profile(execution, transfer.taper, :taper),
        arithmetic = grid.arithmetic,
    )
    shape = (length(grid.omega), size(map.synthesis, 2))
    return SolpropJulia.ModalSourceData(
        grid, map, _prefactor(execution, transfer.prefactor, shape, grid.arithmetic);
        kerr = _kerr(execution, transfer.kerr, grid.arithmetic),
        raman = Tuple(
            _raman(execution, component, dt, grid.arithmetic)
                for component in transfer.raman
        ),
        plasma = _plasma(execution, transfer.plasma, dt, grid.arithmetic),
    )
end

function prepare_modal(transfer::Py, threads::Int, node_chunk::Int)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), threads)
    data = _modal_data(execution, transfer)
    shape = data.prefactor.shape
    1 <= node_chunk <= size(data.map.synthesis, 1) ||
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
    physical = reshape(permutedims(host, (3, 2, 1)), size(host, 3), size(host, 1))
    arithmetic = SolpropJulia._prepared_arithmetic(preparation.data)
    state = SolpropJulia._prepare_scaled_state(execution, physical, arithmetic)
    control = transfer.control
    norm = SolpropJulia.MarchNorm(
        execution, copy_preparation(transfer.active), copy_preparation(transfer.observed),
        copy_preparation(transfer.weights), pyconvert(Float64, control.controller.rtol),
        SolpropJulia._scaled_absolute_tolerance(
            pyconvert(Float64, control.controller.atol), arithmetic,
        ),
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
    source = SolpropJulia._source_workspace(
        execution, preparation.data, preparation.node_chunk,
    )
    operator = preparation.operator
    work = operator isa SolpropJulia.ExpansionOperator ?
        zeros(size(operator.terms, 3)) : nothing
    flow = SolpropJulia.LinearFlow(
        execution, operator, state; work,
        tabulate = pyconvert(Bool, transfer.tabulate_flow),
    )
    return state, norm, schedule, source, flow
end

_launch_noise(preparation, ::Nothing) = nothing
function _launch_noise(preparation, transfer::Py)
    pyis(transfer, nothing) && return nothing
    return SolpropJulia.InputNoise(
        preparation.execution, copy_preparation(transfer.spectrum),
        pyconvert(Float64, transfer.launch), preparation.operator,
    )
end

function start_runge_kutta(
        preparation::Union{
            SolpropJulia._ProblemPreparation, SolpropJulia.SessionPreparation,
        },
        transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
        noise = nothing, resume::Bool = false, transverse_continuation = nothing,
        planning = nothing,
    )
    planning === nothing || (preparation = _planning_preparation(preparation, planning))
    noise = _launch_noise(SolpropJulia._problem_preparation(preparation), noise)
    session_preparation = preparation
    transverse_continuation = _transverse_continuation(transverse_continuation)
    preparation = SolpropJulia._problem_preparation(preparation)
    execution = preparation.execution
    tableau = SolpropJulia.EmbeddedTableau(
        copy_preparation(transfer.nodes), copy_preparation(transfer.matrix),
        copy_preparation(transfer.solution), copy_preparation(transfer.error),
    )
    policy = _control(transfer)
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    services = SolpropJulia._launch_observation_services(
        session_preparation, state, noise, norm.active, z, resume;
        transverse_continuation,
    )
    return SolpropJulia.RungeKuttaSession(
        execution, state, z, pyconvert(Float64, transfer.control.dz_init), tableau, policy,
        norm, schedule, source, flow; stop, plane_index = plane, plane_pending, services,
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight; noise),
        dense_weights = pyis(transfer.dense_weights, nothing) ? nothing :
            copy_preparation(transfer.dense_weights),
        arithmetic = SolpropJulia._prepared_arithmetic(preparation.data),
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
        preparation::Union{
            SolpropJulia._ProblemPreparation, SolpropJulia.SessionPreparation,
        },
        transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
        noise = nothing, resume::Bool = false, transverse_continuation = nothing,
        planning = nothing,
    )
    planning === nothing || (preparation = _planning_preparation(preparation, planning))
    noise = _launch_noise(SolpropJulia._problem_preparation(preparation), noise)
    session_preparation = preparation
    transverse_continuation = _transverse_continuation(transverse_continuation)
    preparation = SolpropJulia._problem_preparation(preparation)
    execution = preparation.execution
    policy = _adams_control(transfer.adams)
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    services = SolpropJulia._launch_observation_services(
        session_preparation, state, noise, norm.active, z, resume;
        transverse_continuation,
    )
    return SolpropJulia.AdamsSession(
        execution, state, z, pyconvert(Float64, transfer.control.dz_init),
        policy, norm, schedule, source, flow;
        stop, plane_index = plane, plane_pending, services,
        source_absent = pyconvert(Bool, transfer.source_absent),
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight; noise),
        arithmetic = SolpropJulia._prepared_arithmetic(preparation.data),
    )
end

function start_low_storage(
        preparation::Union{
            SolpropJulia._ProblemPreparation, SolpropJulia.SessionPreparation,
        },
        transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
        noise = nothing, resume::Bool = false, transverse_continuation = nothing,
        planning = nothing,
    )
    planning === nothing || (preparation = _planning_preparation(preparation, planning))
    noise = _launch_noise(SolpropJulia._problem_preparation(preparation), noise)
    session_preparation = preparation
    transverse_continuation = _transverse_continuation(transverse_continuation)
    preparation = SolpropJulia._problem_preparation(preparation)
    execution = preparation.execution
    state, norm, schedule, source, flow = _launch(preparation, transfer, field)
    services = SolpropJulia._launch_observation_services(
        session_preparation, state, noise, norm.active, z, resume;
        transverse_continuation,
    )
    return SolpropJulia.LowStorageSession(
        execution, state, z, SolpropJulia.RK4IPTableau(), norm, schedule, source, flow;
        stop, plane_index = plane, plane_pending, services,
        arithmetic = SolpropJulia._prepared_arithmetic(preparation.data),
        exports = SolpropJulia._FieldExport(execution, preparation.power_weight; noise),
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
        preparation::Union{
            SolpropJulia._ProblemPreparation, SolpropJulia.SessionPreparation,
        },
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

include("spatial.jl")
include("fft_plans.jl")
include("transverse.jl")
include("observations.jl")
include("monitor.jl")
include("trial_probe.jl")

function copy_session_noise_phase!(
        output::Union{Py, PyArray}, session, generation::Int,
    )
    borrowed = _python_buffer(Float64, Val(3), output)
    modes, components, frequencies = size(borrowed)
    wanted = (components * frequencies, frequencies, 1)
    all(axis -> size(borrowed, axis) == 1 || stride(borrowed, axis) == wanted[axis], 1:3) ||
        throw(ArgumentError("a Python phase destination must be C-contiguous"))
    GC.@preserve output borrowed begin
        address = Base.unsafe_convert(Ptr{Float64}, borrowed)
        columns = unsafe_wrap(
            Array, address, (frequencies, components * modes); own = false,
        )
        copy_session_noise_phase!(columns, session, generation)
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

function restore_session_counts!(session::SolpropJulia.AdamsSession, counts::Py)
    return restore_session_counts!(session, pyconvert(NTuple{9, Int}, counts))
end

function _history_buffer(::Type{T}, ::Val{N}, value::Py) where {T, N}
    borrowed = _python_buffer(T, Val(N), value)
    expected = 1
    for axis in N:-1:1
        size(borrowed, axis) == 1 || stride(borrowed, axis) == expected ||
            throw(ArgumentError("Python history buffers must be C-contiguous"))
        expected *= size(borrowed, axis)
    end
    return borrowed
end

function _history_optional(value::Py, ::Type{T}) where {T}
    return pyis(value, pybuiltins.None) ? nothing :
        _history_buffer(T, Val(5), value)
end

function _history_owners(buffers::Py, ::Type{T}) where {T}
    owners = (
        _history_buffer(ComplexF64, Val(3), buffers.state),
        _history_buffer(Float64, Val(1), buffers.positions),
        _history_buffer(UInt8, Val(1), buffers.identities),
        _history_buffer(Int64, Val(1), buffers.slots),
        _history_buffer(T, Val(5), buffers.store),
        _history_optional(buffers.states, T), _history_optional(buffers.rates, ComplexF64),
    )
    # Nonowning Julia arrays do not carry a foreign view's common Python parent.
    # Compare byte ranges before wrapping, rather than relying on mightalias there.
    for first in eachindex(owners)
        left = owners[first]
        left === nothing && continue
        left_start = UInt(Base.unsafe_convert(Ptr{eltype(left)}, left))
        left_stop = left_start + sizeof(eltype(left)) * length(left)
        for second in (first + 1):length(owners)
            right = owners[second]
            right === nothing && continue
            right_start = UInt(Base.unsafe_convert(Ptr{eltype(right)}, right))
            right_stop = right_start + sizeof(eltype(right)) * length(right)
            left_start < right_stop && right_start < left_stop &&
                throw(ArgumentError("Python history buffers must be disjoint"))
        end
    end
    return owners
end

_history_view(::Nothing) = nothing
function _history_view(value::PyArray{T, 1}) where {T}
    return unsafe_wrap(Array, Base.unsafe_convert(Ptr{T}, value), size(value); own = false)
end
function _history_view(value::PyArray{ComplexF64, 3})
    modes, components, frequencies = size(value)
    return unsafe_wrap(
        Array, Base.unsafe_convert(Ptr{ComplexF64}, value),
        (frequencies, components * modes); own = false,
    )
end
function _history_view(value::PyArray{Complex{T}, 5}) where {T}
    slots, batch, modes, components, frequencies = size(value)
    batch == 1 || throw(ArgumentError("portable history rows require one trajectory"))
    return unsafe_wrap(
        Array, Base.unsafe_convert(Ptr{Complex{T}}, value),
        (frequencies, components * modes, slots); own = false,
    )
end

function export_adams_history!(
        buffers::Py, session::SolpropJulia.AdamsSession, generation::Int, z::Float64,
    )
    owners = _history_owners(buffers, eltype(session.field))
    GC.@preserve buffers owners begin
        borrowed = SolpropJulia.AdamsHistoryBuffers(map(_history_view, owners)...)
        export_adams_history!(borrowed, session, generation, z)
    end
    return nothing
end

function restore_adams_history!(
        session::SolpropJulia.AdamsSession, snapshot::Py, buffers::Py,
    )
    metadata = SolpropJulia.AdamsHistoryMetadata(
        pyconvert(Float64, snapshot.z), pyconvert(Float64, snapshot.proposal),
        pyconvert(Int, snapshot.order), pyconvert(Int, snapshot.rejections),
        pyconvert(Bool, snapshot.last_rejected), pyconvert(Bool, snapshot.collar_was_quiet),
    )
    owners = _history_owners(buffers, eltype(session.field))
    GC.@preserve buffers owners begin
        borrowed = SolpropJulia.AdamsHistoryBuffers(map(_history_view, owners)...)
        return restore_adams_history!(session, metadata, borrowed)
    end
end

end
