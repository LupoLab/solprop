# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct TransferOpaqueArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    storage::A
end
Base.size(array::TransferOpaqueArray) = size(array.storage)
Base.getindex(array::TransferOpaqueArray, indices...) = getindex(array.storage, indices...)
Base.setindex!(array::TransferOpaqueArray, value, indices...) =
    setindex!(array.storage, value, indices...)
# A storage family with an unknown parent contract must not trigger recursive walking.
Base.parent(array::TransferOpaqueArray) = array


# Public transport functions also accept the scripted session protocol used to
# qualify service hooks. Its external record still exposes owned engine services.
struct TransferScriptSession{S, T}
    services::S
    state::T
end

function transfer_alias_pending()
    session = monitor_transfer_case()
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    SolpropJulia.acknowledge!(session, launch.generation)
    report = SolpropJulia.advance!(session; deadline = 60.0)
    count, width, plane_width =
        SolpropJulia.session_observation_status(session, report.generation)
    destinations = (
        fill(-9.0, count), fill(Int64(-9), count), fill(-9.0, width, count),
        fill(-9.0, plane_width),
    )
    return session, report.generation, destinations
end

function transfer_alias_record(session)
    return (
        copy(session.field), copy(session.attempts.z), copy(session.attempts.dz),
        copy(session.services.rows.values), copy(session.services.plane_values),
        session.attempts.count, session.services.rows.count, session.state.generation,
        session.state.attempts, session.state.accepted, session.state.pending,
    )
end

@testset "Transfer byte spans cover interior owners and signed wrapper strides" begin
    storage = collect(1.0:64.0)
    original = copy(storage)
    GC.@preserve storage begin
        interior = unsafe_wrap(Array, pointer(storage, 3), (8,); own = false)
        @test !Base.mightalias(storage, interior)
        for wrapped in (
                interior, view(interior, 8:-1:1), reshape(interior, 2, 4),
                transpose(reshape(interior, 2, 4)),
                adjoint(reshape(interior, 2, 4)),
                PermutedDimsArray(reshape(interior, 2, 4), (2, 1)),
                reinterpret(ComplexF64, interior),
                reinterpret(UInt8, view(interior, 1:2:8)),
            )
            @test SolpropJulia._transfer_arrays_alias(wrapped, storage)
            @test SolpropJulia._transfer_arrays_alias(storage, wrapped)
            @test SolpropJulia._retained_alias(wrapped, (owned = (nothing, storage),))
        end
        left = unsafe_wrap(Array, pointer(storage), (8,); own = false)
        right = unsafe_wrap(Array, pointer(storage, 9), (8,); own = false)
        @test !SolpropJulia._transfer_arrays_alias(left, right)
        @test !SolpropJulia._transfer_arrays_alias(right, left)
        @test SolpropJulia._transfer_arrays_alias(view(storage, 1:8), view(storage, 9:16))
        @test !SolpropJulia._transfer_arrays_alias(view(storage, 1:0), interior)
    end
    @test storage == original
    @test !SolpropJulia._transfer_arrays_alias(zeros(0, 3), zeros(3))
    @test !SolpropJulia._retained_alias(zeros(0), (zeros(1),))
    @test_throws ArgumentError SolpropJulia._retained_alias(
        TransferOpaqueArray(zeros(2)), (zeros(2),),
    )
    @test_throws ArgumentError SolpropJulia._retained_alias(
        zeros(2), (TransferOpaqueArray(zeros(2)),),
    )
    @test !SolpropJulia._retained_alias(TransferOpaqueArray(zeros(0)), (zeros(2),))
end

@testset "Foreign stride spans include item width and reject address overflow" begin
    span = SolpropJulia._transfer_strided_span
    @test span(UInt(100), (3,), (8,), 8) == (UInt(100), UInt(124))
    @test span(UInt(116), (3,), (-8,), 8) == (UInt(100), UInt(124))
    @test span(UInt(116), (3, 2), (-8, 32), 8) == (UInt(100), UInt(156))
    @test span(UInt(100), (3,), (0,), 8) == (UInt(100), UInt(108))
    @test span(UInt(100), (), (), 8) == (UInt(100), UInt(108))
    @test span(UInt(100), (0, 3), (-8, 32), 8) === nothing
    for (address, dimensions, strides, item_size) in (
            (UInt(4), (2,), (-8,), 8),
            (typemax(UInt) - UInt(3), (1,), (0,), 8),
            (typemax(UInt) - UInt(12), (3,), (8,), 8),
            (UInt(100), (typemax(Int),), (8,), 8),
            (UInt(100), (-1,), (8,), 8),
            (UInt(100), (1,), (8,), 0),
        )
        @test_throws ArgumentError span(address, dimensions, strides, item_size)
    end
end

@testset "Observation overlap refusals precede every write and preserve retry" begin
    session, generation, destinations = transfer_alias_pending()
    original = transfer_alias_record(session)
    GC.@preserve session begin
        first = Ptr{UInt8}(pointer(session.field)) + sizeof(Float64)
        aliases = (
            unsafe_wrap(Array, Ptr{Float64}(first), size(destinations[1]); own = false),
            unsafe_wrap(Array, Ptr{Int64}(first), size(destinations[2]); own = false),
            unsafe_wrap(Array, Ptr{Float64}(first), size(destinations[3]); own = false),
            unsafe_wrap(Array, Ptr{Float64}(first), size(destinations[4]); own = false),
        )
        for index in 1:4
            candidate = ntuple(i -> i == index ? aliases[i] : destinations[i], 4)
            before = map(copy, candidate)
            @test_throws ArgumentError SolpropJulia.session_observations!(
                candidate..., session, generation,
            )
            @test isequal(map(copy, candidate), before)
            @test isequal(transfer_alias_record(session), original)
        end
    end
    count = length(destinations[1])
    backing = fill(-91.0, 4count)
    GC.@preserve backing begin
        positions = unsafe_wrap(Array, pointer(backing), (count,); own = false)
        steps = unsafe_wrap(Array, Ptr{Int64}(pointer(backing, 2)), (count,); own = false)
        @test !Base.mightalias(positions, steps)
        before = copy(backing)
        @test_throws ArgumentError SolpropJulia.session_observations!(
            positions, steps, destinations[3], destinations[4], session, generation,
        )
        @test backing == before
        @test all(==(-9.0), destinations[3]) && all(==(-9.0), destinations[4])
    end
    opaque = TransferOpaqueArray(destinations[3])
    @test_throws ArgumentError SolpropJulia.session_observations!(
        destinations[1], destinations[2], opaque, destinations[4], session, generation,
    )
    @test all(==(-9.0), opaque)
    for _ in 1:2
        SolpropJulia.session_observations!(destinations..., session, generation)
        @test destinations[1] == session.services.rows.z[1:count]
        @test destinations[2] == session.services.rows.step[1:count]
        @test destinations[3] == session.services.rows.values[:, 1:count]
        @test destinations[4] == session.services.plane_values
        @test isequal(transfer_alias_record(session), original)
    end
end


@testset "Scripted session adapters retain explicit service ownership admission" begin
    session, generation, destinations = transfer_alias_pending()
    scripted = TransferScriptSession(session.services, session.state)
    before = transfer_alias_record(session)
    @test_throws ArgumentError SolpropJulia.session_observations!(
        destinations[1], destinations[2], destinations[3],
        session.services.plane_values, scripted, generation,
    )
    @test isequal(transfer_alias_record(session), before)
    spectrum = session.services.work.spectrum
    retained = copy(spectrum)
    count = SolpropJulia.session_monitor_payload(scripted, generation).row_count
    GC.@preserve spectrum begin
        rows = unsafe_wrap(
            Array, Ptr{Float64}(pointer(spectrum)) + sizeof(Float64),
            (5, count); own = false,
        )
        @test_throws ArgumentError SolpropJulia.session_monitor_rows!(
            rows, scripted, generation,
        )
    end
    @test spectrum == retained
    SolpropJulia.session_observations!(destinations..., scripted, generation)
    @test destinations[4] == session.services.plane_values
    SolpropJulia.session_monitor_rows!(zeros(5, count), scripted, generation)
    @test isequal(transfer_alias_record(session), before)
end

@testset "Field and attempt copies share retained and different-type overlap guards" begin
    session, generation, _ = transfer_alias_pending()
    original = transfer_alias_record(session)
    count = session.attempts.count
    destinations = (
        fill(-9.0, count), fill(-9.0, count), fill(-9.0, count),
        fill(false, count), fill(-9.0, count),
    )
    GC.@preserve session begin
        output = unsafe_wrap(
            Array, pointer(session.field), size(session.field); own = false,
        )
        @test_throws ArgumentError SolpropJulia.export_session!(
            output, session, generation, ones(size(output)), false, false,
        )
        positions = unsafe_wrap(
            Array, Ptr{Float64}(pointer(session.field)) + sizeof(Float64),
            (count,); own = false,
        )
        @test_throws ArgumentError SolpropJulia.session_attempts!(
            positions, destinations[2:5]..., session, generation,
        )
    end
    backing = fill(-19.0, count + 1)
    GC.@preserve backing begin
        positions = unsafe_wrap(Array, pointer(backing), (count,); own = false)
        accepted = unsafe_wrap(Array, Ptr{Bool}(pointer(backing, 2)), (count,); own = false)
        @test_throws ArgumentError SolpropJulia.session_attempts!(
            positions, destinations[2], destinations[3], accepted, destinations[5],
            session, generation,
        )
        @test all(==(-19.0), backing)
    end
    external_weights = fill(2.0, 32)
    GC.@preserve external_weights begin
        weights = reshape(view(external_weights, 2:9), size(session.field))
        output = unsafe_wrap(
            Array, Ptr{ComplexF64}(pointer(external_weights, 3)), size(session.field);
            own = false,
        )
        @test !Base.mightalias(output, weights)
        @test_throws ArgumentError SolpropJulia.export_session!(
            output, session, generation, weights, true, false,
        )
        @test all(==(2.0), external_weights)
    end
    @test isequal(transfer_alias_record(session), original)
    SolpropJulia.session_attempts!(destinations..., session, generation)
    @test destinations[1] == session.attempts.z[1:count]
    @test destinations[4] == session.attempts.accepted[1:count]
    output = zeros(ComplexF64, size(session.field))
    SolpropJulia.export_session!(
        output, session, generation, ones(size(output)), false, false,
    )
    @test output == session.field
    @test isequal(transfer_alias_record(session), original)
end


@testset "Adams history refuses foreign retained and mixed-type destination aliases" begin
    session, metadata, buffers = adams_transfer_saved()
    before = adams_transfer_record(session)
    count = length(buffers.positions)
    GC.@preserve session begin
        positions = unsafe_wrap(
            Array, Ptr{Float64}(pointer(session.field)) + sizeof(Float64),
            (count,); own = false,
        )
        candidate = adams_transfer_rebuild(buffers; positions)
        @test_throws ArgumentError SolpropJulia.export_adams_history!(
            candidate, session, 7, 0.2,
        )
        @test isequal(adams_transfer_record(session), before)
    end
    backing = fill(-99.0, 2count + 1)
    GC.@preserve backing begin
        positions = unsafe_wrap(Array, pointer(backing), (count,); own = false)
        slots = unsafe_wrap(Array, Ptr{Int64}(pointer(backing, 2)), (count,); own = false)
        @test !Base.mightalias(positions, slots)
        candidate = adams_transfer_rebuild(buffers; positions, slots)
        @test_throws ArgumentError SolpropJulia.export_adams_history!(
            candidate, session, 7, 0.2,
        )
        @test all(==(-99.0), backing)
        @test isequal(adams_transfer_record(session), before)
    end
    fresh = adams_transfer_fixture(; z = 0.2)
    original = adams_transfer_record(fresh)
    GC.@preserve fresh begin
        positions = unsafe_wrap(
            Array, Ptr{Float64}(pointer(fresh.field)) + sizeof(Float64),
            (count,); own = false,
        )
        candidate = adams_transfer_rebuild(buffers; positions)
        reason = SolpropJulia.restore_adams_history!(fresh, metadata, candidate)
        @test reason isa String && occursin("alias", reason)
        @test isequal(adams_transfer_record(fresh), original)
    end
    SolpropJulia.export_adams_history!(buffers, session, 7, 0.2)
    @test buffers.positions == session.positions[1:count]
    @test buffers.slots == session.slots[1:count] .- 1
    @test isequal(adams_transfer_record(session), before)
end

function transfer_alias_allocations(session, generation, destinations, field, weights)
    SolpropJulia.session_observations!(destinations..., session, generation)
    SolpropJulia.export_session!(field, session, generation, weights, false, false)
    return (
        @allocated(
            SolpropJulia.session_observations!(destinations..., session, generation)
        ),
        @allocated(
            SolpropJulia.export_session!(field, session, generation, weights, false, false)
        ),
        @allocated(SolpropJulia._retained_alias(field, session)),
    )
end

@testset "Shared transfer ownership admission allocates nothing after warmup" begin
    session, generation, destinations = transfer_alias_pending()
    field, weights = similar(session.field), ones(size(session.field))
    @test transfer_alias_allocations(session, generation, destinations, field, weights) ==
        (0, 0, 0)
end
