# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SolpropJuliaPythonCallExt

import PythonCall: Py, PyArray, pyconvert, pyis
import SolpropJulia
import SolpropJulia: copy_preparation, prepare_envelope, start_runge_kutta
import SolpropJulia: export_session!, session_attempts!

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

function prepare_envelope(transfer::Py, threads::Int)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), threads)
    kerr = if pyis(transfer.kerr, nothing)
        nothing
    else
        factors = pyconvert(NTuple{4, Float64}, transfer.kerr)
        coefficient, self, conjugate, harmonic = factors
        SolpropJulia.KerrCoefficients(
            coefficient, :analytic; self_factor = self, conjugate_factor = conjugate,
            harmonic_factor = harmonic,
        )
    end
    data = SolpropJulia.EnvelopeKerrData(
        execution;
        synthesis = copy_preparation(transfer.synthesis),
        projection = copy_preparation(transfer.projection),
        prefactor = copy_preparation(transfer.prefactor),
        omega_window = copy_preparation(transfer.omega_window),
        source_window = copy_preparation(transfer.source_window),
        kerr, density = pyconvert(Float64, transfer.density),
        carrier = pyis(transfer.carrier, nothing) ? nothing :
            copy_preparation(transfer.carrier),
    )
    operator = SolpropJulia.DiagonalOperator(execution, copy_preparation(transfer.operator))
    weight = copy_preparation(transfer.power_weight)
    all(value -> isfinite(value) && value > 0, weight) ||
        throw(ArgumentError("power export weights must be finite and positive"))
    return SolpropJulia.EnvelopePreparation(
        execution, data, operator, copy_preparation(transfer.crop_index), weight,
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

function start_runge_kutta(
        preparation::SolpropJulia.EnvelopePreparation, transfer::Py, field,
        z::Float64, plane::Int, plane_pending::Bool, stop::SolpropJulia.StopFlag,
    )
    execution = preparation.execution
    host = copy_preparation(field)
    ndims(host) == 3 && size(host, 2) == 1 ||
        throw(DimensionMismatch("the starting field needs mode/component/frequency axes"))
    state = reshape(permutedims(host, (3, 2, 1)), size(host, 3), size(host, 1))
    tableau = SolpropJulia.EmbeddedTableau(
        copy_preparation(transfer.nodes), copy_preparation(transfer.matrix),
        copy_preparation(transfer.solution), copy_preparation(transfer.error),
    )
    policy = _control(transfer)
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
    source = SolpropJulia.EnvelopeKerrSource(
        execution, preparation.data, preparation.crop_index
    )
    flow = SolpropJulia.LinearFlow(execution, preparation.operator, state)
    return SolpropJulia.RungeKuttaSession(
        execution, state, z, pyconvert(Float64, control.dz_init), tableau, policy,
        norm, schedule, source, flow; stop, plane_index = plane, plane_pending,
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
        preparation::SolpropJulia.EnvelopePreparation,
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
            columns, session, generation, preparation.power_weight, power, optics,
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

end
