# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import LinearAlgebra
import SolpropJulia as SP

# This model qualifies orchestration and transfer volume, not device arithmetic.
mutable struct _LowStorageDevice
    allocations::Int
    downloads::Int
    bytes::Int
    largest::Int
end
const _LowStorageDeviceExecution = SP.KernelExecution{_LowStorageDevice}

function SP.dzeros(
        execution::_LowStorageDeviceExecution, ::Type{T}, shape::Integer...,
    ) where {T}
    execution.backend.allocations += 1
    return zeros(T, shape...)
end

function SP._readback_storage(
        execution::_LowStorageDeviceExecution, ::Type{T}, shape,
    ) where {T}
    return SP.dzeros(execution, T, shape...), zeros(T, shape...)
end

function SP.upload(execution::_LowStorageDeviceExecution, source::AbstractArray)
    return copyto!(SP.dzeros(execution, eltype(source), size(source)...), source)
end

function SP.copy_device!(destination::Array, source::Array, ::_LowStorageDeviceExecution)
    destination === source || copyto!(destination, source)
    return destination
end

function SP.copy_host!(
        destination::Array, source::Array, execution::_LowStorageDeviceExecution,
    )
    execution.backend.downloads += 1
    execution.backend.bytes += sizeof(source)
    execution.backend.largest = max(execution.backend.largest, sizeof(source))
    return copyto!(destination, source)
end

function SP._each!(body, ::_LowStorageDeviceExecution, rows, columns, heavy)
    for column in 1:columns, row in 1:rows
        body(row, column)
    end
    return nothing
end

function SP._each_column!(body, ::_LowStorageDeviceExecution, columns)
    for column in 1:columns
        body(column)
    end
    return nothing
end

low_storage_device_execution() = SP.KernelExecution(_LowStorageDevice(0, 0, 0, 0), 1)

mutable struct LowStorageDeviceSource{E}
    execution::E
    gain::ComplexF64
    calls::Int
    fail_on::Int
    nonfinite_on::Int
end

function (source::LowStorageDeviceSource)(output, input, position)
    source.calls += 1
    source.calls == source.fail_on && error("controlled device source failure")
    gain, nonfinite = source.gain, source.calls == source.nonfinite_on
    SP.each!(source.execution, size(input)...) do row, column
        output[row, column] = nonfinite && row == 2 ? ComplexF64(NaN) :
            gain * input[row, column]
    end
    return nothing
end

mutable struct LowStorageDeviceServices
    failure::Symbol
    accepted::Int
    finished::Int
end
SP._session_launch!(::LowStorageDeviceServices, session) = nothing
SP._session_observe!(::LowStorageDeviceServices, session, kind, plane) = nothing
SP._session_finish!(::LowStorageDeviceServices, session) = nothing
SP._session_barrier(::LowStorageDeviceServices, session) = :none
SP._session_continue!(::LowStorageDeviceServices, session) = :none
SP._session_acknowledge!(::LowStorageDeviceServices, session, generation) = nothing

function SP._session_finish_candidate!(
        services::LowStorageDeviceServices, session, candidate, dz, destination,
    )
    services.finished += 1
    nonfinite = services.failure === :nonfinite
    SP.each!(session.execution, size(candidate)...) do row, column
        if nonfinite && row == 2
            candidate[row, column] = ComplexF64(NaN)
        end
    end
    services.failure === :finish && error("controlled device finishing failure")
    return nothing
end

function SP._session_accepted!(services::LowStorageDeviceServices, session, dz)
    services.accepted += 1
    services.failure === :accepted && error("controlled device accepted failure")
    return nothing
end

function low_storage_device_fixture(
        execution; steps = 8, tabulate = false, fail_on = 0, nonfinite_on = 0,
        services = SP._NoSessionServices(), capacity = 16,
    )
    initial = ComplexF64[row + im * column for row in 1:17, column in 1:3] ./ 32
    active = fill(true, 17)
    active[2] = false
    observed = copy(active)
    state = SP.upload(execution, initial)
    operator = SP.DiagonalOperator(execution, fill(-0.25 + 0.5im, size(initial)))
    flow = SP.LinearFlow(execution, operator, state; tabulate)
    source = LowStorageDeviceSource(execution, 0.125im, 0, fail_on, nonfinite_on)
    norm = SP.MarchNorm(execution, active, observed, ones(17), 1.0e-4, 1.0e-10)
    finish, step = 1 / 32, 1 / (32steps)
    schedule = SP.MarchSchedule(
        planes = [finish], breakpoints = [0.0, finish], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = fill(step, steps),
        replay_edges = collect(0:step:finish),
    )
    session = SP.LowStorageSession(
        execution, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source, flow;
        services, capacity,
    )
    initial[2, :] .= 0
    return session, initial, step
end

function low_storage_device_drive(session; work_limit = 1)
    trace = Tuple{SP.SessionEvent, Float64}[]
    while !session.state.finished
        length(trace) < 64 || error("low-storage test exceeded its event allowance")
        report = SP.advance!(session; work_limit, deadline = 60.0)
        push!(trace, (report.kind, report.z))
        SP.acknowledge!(session, report.generation)
    end
    return trace
end

function low_storage_device_host(session)
    output = Matrix{ComplexF64}(undef, size(session.field))
    if SP.oncpu(session.execution)
        copyto!(output, session.field)
    else
        SP.copy_host!(output, session.field, session.execution)
    end
    return output
end

function low_storage_device_reference(initial, steps, step)
    return setprecision(BigFloat, 256) do
        h = BigFloat(step)
        x = h * big(0.125) * im
        polynomial = 1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24
        factor = exp(h * (-big(0.25) + big(0.5) * im)) * polynomial
        ComplexF64.(Complex{BigFloat}.(initial) .* factor^steps)
    end
end

function low_storage_device_bound(initial, steps, step)
    # CUDA_LOW_STORAGE.md counts <=256 scalar-rounding equivalents per step,
    # conditional on the separately checked exponential primitive allowance.
    # Twice that count covers two implementations and the rounded MPFR readout.
    # Entries are small dyadic components, so this one-norm sum is exact here.
    magnitude = sum(abs, real.(initial)) + sum(abs, imag.(initial))
    return setprecision(BigFloat, 256) do
        operations, unit = 512steps, BigFloat(2)^-53
        gamma = operations * unit / (1 - operations * unit)
        envelope = exp(BigFloat(0.875) * steps * BigFloat(step))
        # Outward binary64 conversion also dominates the much smaller MPFR
        # reference-rounding error at this bounded, nonzero fixture scale.
        nextfloat(Float64(gamma * BigFloat(magnitude) * envelope, RoundUp))
    end
end

@testset "Low-storage device work owns only compact control mirrors" begin
    execution = low_storage_device_execution()
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    field = reshape(ComplexF64.(1:51), 17, 3) ./ 64
    work = SP.LowStorageWork(execution, field, SP.RK4IPTableau())
    ordinary = SP.LowStorageWork(cpu, field, SP.RK4IPTableau())
    @test work.norm_readback.device === work.norm_parts
    @test work.finite_readback.device === work.finite_columns
    @test work.norm_readback.host !== work.norm_parts
    @test work.finite_readback.host !== work.finite_columns
    @test size(work.norm_readback.host) == (2, 3)
    @test size(work.finite_readback.host) == (3,)
    @test ordinary.norm_readback === ordinary.finite_readback === nothing
    legacy = SP.LowStorageWork(
        ordinary.accumulator, ordinary.stage, ordinary.source, ordinary.norm_mask,
        ordinary.norm_parts, ordinary.finite_columns,
    )
    @test legacy === ordinary
    for (owner, expected) in (
            (work, 3sizeof(field) + 17 + 34 * 3),
            (ordinary, 3sizeof(field) + 17 + 17 * 3),
        )
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "work", owner)
        @test sum(values(ledger)) == expected
    end
    magnitude, finite = SP._low_storage_norm!(work, field, execution)
    expected = setprecision(BigFloat, 256) do
        sqrt(sum(abs2, Complex{BigFloat}.(field)))
    end
    operations, unit = 32length(field) + 16size(field, 2), eps(Float64) / 2
    bound = operations * unit / (1 - operations * unit) * expected
    @test bound > 0
    @test abs(BigFloat(magnitude) - expected) <= bound
    @test finite
    @test execution.backend.downloads == 2
    @test execution.backend.bytes == 17 * 3
    @test execution.backend.largest == 2 * 3 * sizeof(Float64)
    work.norm_mask[2] = false
    field[2, 2] = NaN
    _, finite = SP._low_storage_norm!(work, field, execution)
    @test !finite
    @test work.finite_readback.host == [true, false, true]
    before = copy(work.norm_parts)
    @test_throws ArgumentError SP._low_storage_norm!(
        work, field, low_storage_device_execution(),
    )
    @test isequal(work.norm_parts, before)
    @test_throws ArgumentError SP._low_storage_norm!(ordinary, field, execution)
    before_allocations = execution.backend.allocations
    @test_throws ArgumentError SP.LowStorageWork(
        execution, ComplexF32.(field), SP.RK4IPTableau(),
    )
    @test execution.backend.allocations == before_allocations
end

@testset "Low-storage device sessions retain itinerary and failure ownership" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    for steps in (1, 8), tabulate in (false, true)
        execution = low_storage_device_execution()
        candidate, initial, step = low_storage_device_fixture(execution; steps, tabulate)
        reference, _, _ = low_storage_device_fixture(cpu; steps, tabulate)
        @test low_storage_device_drive(candidate) == low_storage_device_drive(reference)
        expected = low_storage_device_reference(initial, steps, step)
        bound = low_storage_device_bound(initial, steps, step)
        @test bound > 0
        @test LinearAlgebra.norm(candidate.field - expected) <= bound
        @test LinearAlgebra.norm(candidate.field - reference.field) <= bound
        @test candidate.counts.attempted_source_evaluations == 4steps
        @test candidate.counts.committed_source_evaluations == 4steps
        @test candidate.counts.norm_evaluations == steps
        @test candidate.state.accepted == steps && candidate.state.rejected == 0
        @test all(iszero, candidate.field[2, :])
        @test execution.backend.largest <= 2 * 3 * sizeof(Float64)
        @test !candidate.state.poisoned
        SP.close_session!(candidate)
        SP.close_session!(reference)
    end
    for failed_call in 1:4
        session, _, _ = low_storage_device_fixture(
            low_storage_device_execution(); fail_on = failed_call,
        )
        trace = low_storage_device_drive(session)
        @test last(trace)[1] == SP.SESSION_FAILED
        @test session.state.poisoned && session.state.in_flight
        @test session.source.calls == session.counts.attempted_source_evaluations ==
            failed_call
        @test session.counts.committed_source_evaluations == session.attempts.count == 0
        @test session.state.z == session.state.flight_z == 0.0
    end
    for failure in (:finish, :nonfinite, :accepted)
        services = LowStorageDeviceServices(failure, 0, 0)
        session, _, _ = low_storage_device_fixture(low_storage_device_execution(); services)
        trace = low_storage_device_drive(session)
        committed = failure === :accepted
        @test last(trace)[1] == SP.SESSION_FAILED
        @test session.state.poisoned == !committed
        @test session.state.in_flight == !committed
        @test session.counts.attempted_source_evaluations == 4
        @test session.counts.committed_source_evaluations == (committed ? 4 : 0)
        @test session.state.accepted == (committed ? 1 : 0)
        @test services.finished == 1
    end
    execution = low_storage_device_execution()
    session, _, _ = low_storage_device_fixture(execution)
    before_allocations = execution.backend.allocations
    @test_throws ArgumentError SP.LowStorageSession(
        execution, ComplexF32.(session.field), 0.0, SP.RK4IPTableau(), session.norm,
        session.schedule, session.source, session.flow;
        arithmetic = SP.SourceArithmetic(Float32),
    )
    @test execution.backend.allocations == before_allocations
    @test session.source.calls == 0
end

@testset "Low-storage refuses foreign scratch and repeated output rows" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    full = SP.LowStorageWork(execution, ones(ComplexF64, 9, 1), SP.RK4IPTableau())
    # A small valid workspace view retains the complete 9-row parent. Its foreign
    # offset caller header is wholly inside that parent but overlaps the work view.
    work = SP.LowStorageWork(
        view(full.accumulator, 1:8, :), view(full.stage, 1:8, :),
        view(full.source, 1:8, :), fill(true, 8), full.norm_parts,
        full.finite_columns, nothing, nothing,
    )
    source_calls = Ref(0)
    source = function (output, input, position)
        source_calls[] += 1
        copyto!(output, input)
        nothing
    end
    flow = (output, input, origin, destination) -> (copyto!(output, input); nothing)
    owner = full.stage
    before = copy(owner)
    GC.@preserve owner begin
        foreign = unsafe_wrap(Array, pointer(owner, 2), (8, 1); own = false)
        @test_throws ArgumentError SP.low_storage_trial!(
            work, foreign, 0.0, 0.125, SP.RK4IPTableau(), source, flow, execution,
        )
        @test owner == before && source_calls[] == 0
    end
    backing = ones(ComplexF64, 8, 1)
    repeated = view(backing, [1, 1, 3, 4, 5, 6, 7, 8], :)
    @test_throws ArgumentError SP.low_storage_trial!(
        work, repeated, 0.0, 0.125, SP.RK4IPTableau(), source, flow, execution,
    )
    @test backing == ones(ComplexF64, 8, 1) && source_calls[] == 0
end
