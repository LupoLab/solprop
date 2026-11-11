# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia
import SolpropJulia as SP

include("device_rk_dense_fixture.jl")

# A serial transport model deliberately forbids host scalar indexing. It qualifies
# ownership, compact copying and orchestration; it is not evidence of GPU arithmetic.
mutable struct _RKDenseDeviceModel
    inside::Bool
    allocations::Int
    downloads::Int
    elements::Int
    fail_copy::Bool
end
const _RKDenseExecution = SP.KernelExecution{_RKDenseDeviceModel}

struct _RKDenseDeviceArray{T, N} <: AbstractArray{T, N}
    values::Array{T, N}
    execution::_RKDenseExecution
end
Base.size(array::_RKDenseDeviceArray) = size(array.values)
Base.sizeof(array::_RKDenseDeviceArray) = sizeof(array.values)
Base.IndexStyle(::Type{<:_RKDenseDeviceArray}) = IndexLinear()
function Base.getindex(array::_RKDenseDeviceArray, indices::Vararg{Int})
    array.execution.backend.inside || error("host scalar read of dense device storage")
    return array.values[indices...]
end
function Base.setindex!(array::_RKDenseDeviceArray, value, indices::Vararg{Int})
    array.execution.backend.inside || error("host scalar write of dense device storage")
    return array.values[indices...] = value
end
SP._transfer_storage_space(array::_RKDenseDeviceArray) = array.execution.backend
SP._transfer_storage_span(array::_RKDenseDeviceArray) = isempty(array) ? nothing :
    (UInt(pointer(array.values)), UInt(pointer(array.values)) + sizeof(array))
SP._transfer_writable_destination(::_RKDenseDeviceArray) = nothing
SP._memory_array_owner(array::_RKDenseDeviceArray) = (array.values, sizeof(array))

function SP.dzeros(execution::_RKDenseExecution, ::Type{T}, shape::Integer...) where {T}
    execution.backend.allocations += 1
    return _RKDenseDeviceArray(zeros(T, shape...), execution)
end
function SP.upload(execution::_RKDenseExecution, input::Array)
    execution.backend.allocations += 1
    return _RKDenseDeviceArray(copy(input), execution)
end
function SP._readback_storage(execution::_RKDenseExecution, ::Type{T}, shape) where {T}
    return SP.dzeros(execution, T, shape...), zeros(T, shape...)
end
function SP.copy_device!(
        destination::_RKDenseDeviceArray, source::_RKDenseDeviceArray,
        execution::_RKDenseExecution,
    )
    destination.execution === source.execution === execution ||
        throw(ArgumentError("dense device model execution mismatch"))
    size(destination) == size(source) && eltype(destination) === eltype(source) ||
        throw(ArgumentError("dense device model copy mismatch"))
    destination === source && return destination
    SP._transfer_arrays_alias(destination, source) &&
        throw(ArgumentError("dense device model partial copy overlap"))
    copyto!(destination.values, source.values)
    return destination
end
function SP.copy_host!(
        destination::Array, source::_RKDenseDeviceArray, execution::_RKDenseExecution,
    )
    source.execution === execution || throw(ArgumentError("wrong model execution"))
    execution.backend.downloads += 1
    execution.backend.elements += length(source)
    if execution.backend.fail_copy
        destination[1] = source.values[1]
        error("injected dense status download failure")
    end
    return copyto!(destination, source.values)
end
function SP._each!(body, execution::_RKDenseExecution, rows, columns, heavy)
    SP._check_kernel_body(typeof(body))
    execution.backend.inside = true
    try
        for column in 1:columns, row in 1:rows
            body(row, column)
        end
    finally
        execution.backend.inside = false
    end
    return nothing
end
function SP._each_column!(body, execution::_RKDenseExecution, columns)
    SP._check_kernel_body(typeof(body))
    execution.backend.inside = true
    try
        for column in 1:columns
            body(column)
        end
    finally
        execution.backend.inside = false
    end
    return nothing
end

_rk_dense_device_model() =
    SP.KernelExecution(_RKDenseDeviceModel(false, 0, 0, 0, false), 1, true)

function _rk_dense_model_fixture(; rate = 0.0im)
    cpu, reference, initial, endpoint, stages = rk_dense_fixture(; rate)
    device = _rk_dense_device_model()
    state = SP.upload(device, initial)
    finish = SP.upload(device, endpoint)
    slopes = SP.upload(device, stages)
    operator = SP.DiagonalOperator(device, fill(ComplexF64(rate), size(initial)))
    work = SP.RungeKuttaObservationWork(
        device, state, rk_dense_tableau(), rk_dense_weights(), operator,
    )
    return (; cpu, reference, device, work, state, finish, slopes)
end

@testset "Device RK dense keeps fields resident and finite facts compact" begin
    for rate in (0.0im, -0.25 + 0.5im, -6.0 + 0.25im, 6.0 + 0.25im)
        f = _rk_dense_model_fixture(; rate)
        active = SP.upload(f.device, Bool[true, false])
        @test_throws "host scalar read" all(isfinite, f.state)
        before = (copy(f.state.values), copy(f.finish.values), copy(f.slopes.values))
        downloads, elements = f.device.backend.downloads, f.device.backend.elements
        SP.retain_runge_kutta_observation!(
            f.work, f.state, f.finish, f.slopes, 0.0, 0.2, 0.2, f.device,
        )
        @test f.work.ready
        @test f.device.backend.downloads - downloads == 5
        @test f.device.backend.elements - elements == 5size(f.state, 2)
        for position in (0.0, 0.025, 0.1, prevfloat(0.2), 0.2)
            expected = SP.observe_runge_kutta!(
                f.reference, position, Bool[true, false], f.cpu,
            )
            output = SP.observe_runge_kutta!(f.work, position, active, f.device)
            budget = device_rk_dense_budget(
                f.state.values, f.slopes.values, f.work.solution_weights,
                f.work.dense_weights, 0.2, position,
            )
            @test all(>(0), budget)
            @test device_rk_dense_within(output.values, expected, budget)
            @test all(iszero, output.values[2, :])
            @test !f.device.backend.inside
        end
        @test (f.state.values, f.finish.values, f.slopes.values) == before
        bytes = SP.runge_kutta_observation_memory(f.work)
        @test bytes.failures == 2sizeof(Int) * size(f.state, 2)
        @test sum(bytes) == 8sizeof(f.state) + length(f.state) + bytes.failures
    end
end

@testset "Device RK dense validates execution and retained aliases before writes" begin
    f = _rk_dense_model_fixture()
    SP.retain_runge_kutta_observation!(
        f.work, f.state, f.finish, f.slopes, 0.0, 0.2, 0.2, f.device,
    )
    before = copy(f.work.coefficients.values)
    other = _rk_dense_device_model()
    @test_throws ArgumentError SP.retain_runge_kutta_observation!(
        f.work, f.state, f.finish, f.slopes, 0.0, 0.2, 0.2, other,
    )
    @test f.work.ready && f.work.coefficients.values == before
    @test other.backend.allocations == other.backend.downloads == 0
    for offset in (0, 1)
        owner = f.work.coefficients.values
        GC.@preserve owner begin
            header = unsafe_wrap(
                Array, pointer(owner) + offset * sizeof(ComplexF64),
                size(f.state); own = false
            )
            foreign = _RKDenseDeviceArray(header, f.device)
            @test_throws ArgumentError SP.retain_runge_kutta_observation!(
                f.work, foreign, f.finish, f.slopes, 0.0, 0.2, 0.2, f.device,
            )
            @test f.work.ready && f.work.coefficients.values == before
        end
    end
    count = f.device.backend.allocations
    @test_throws ArgumentError SP.RungeKuttaObservationWork(
        f.device, ComplexF32.(f.state.values), rk_dense_tableau(), rk_dense_weights(),
        f.work.operator; arithmetic = SP.SourceArithmetic(Float32),
    )
    @test f.device.backend.allocations == count
    @test_throws ArgumentError SP.RungeKuttaObservationWork(
        other, f.state, rk_dense_tableau(), rk_dense_weights(), f.work.operator,
    )
    @test other.backend.allocations == 0
end

@testset "Device RK dense failures invalidate retention and preserve trial inputs" begin
    f = _rk_dense_model_fixture()
    before = (copy(f.state.values), copy(f.finish.values))
    f.slopes.values[1, 2, 7] = ComplexF64(NaN)
    @test_throws ArgumentError SP.retain_runge_kutta_observation!(
        f.work, f.state, f.finish, f.slopes, 0.0, 0.2, 0.2, f.device,
    )
    @test !f.work.ready
    @test (f.state.values, f.finish.values) == before
    f.slopes.values[1, 2, 7] = 0.0im
    f.device.backend.fail_copy = true
    @test_throws "injected dense status" SP.retain_runge_kutta_observation!(
        f.work, f.state, f.finish, f.slopes, 0.0, 0.2, 0.2, f.device,
    )
    @test !f.work.ready
    f.device.backend.fail_copy = false
    SP.retain_runge_kutta_observation!(
        f.work, f.state, f.finish, f.slopes, 0.0, 0.2, nextfloat(nextfloat(0.2)), f.device,
    )
    active = SP.upload(f.device, fill(true, 2))
    @test SP.observe_runge_kutta!(f.work, nextfloat(0.2), active, f.device).values ==
        f.finish.values
    f.work.coefficients.values[2, 1, 1] = Inf
    active.values[2] = false
    @test_throws SP.NonfiniteSourceError SP.observe_runge_kutta!(
        f.work, 0.1, active, f.device,
    )
end

@testset "Device RK dense session delivers interior planes without extra source calls" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    reference = device_rk_dense_session(cpu)
    execution = _rk_dense_device_model()
    candidate = device_rk_dense_session(execution)
    cpu_trace, expected = device_rk_dense_drive(reference)
    device_trace, actual = device_rk_dense_drive(candidate)
    @test device_trace == cpu_trace
    @test candidate.state.attempts == reference.state.attempts == 2
    @test length(actual) == length(expected)
    u = eps(Float64) / 2
    gamma = 256u / (1 - 256u)
    @test gamma > 0
    for (first, second) in zip(actual, expected)
        # At most two full-step factors plus one dense factor, with phases below
        # 1/16. Four complex products and their elementary-function allowances fit
        # in this budget relative to the nonzero launch magnitude.
        @test all(abs.(first - second) .<= gamma .* abs.(expected[1]))
    end
end
