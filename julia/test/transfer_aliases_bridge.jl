# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import PythonCall: PyArray, pyconvert, pyexec, pyimport, pyeval
import SolpropJulia

include("adams_transfer.jl")
include("monitor_transfer.jl")
include("transfer_aliases.jl")

function transfer_foreign_view(numpy, address, count; dtype = "float64")
    return pyeval(
        "numpy.ctypeslib.as_array(ctypes.cast(address, " *
            "ctypes.POINTER(ctypes.c_uint8)), shape=(count * 8,)).view(dtype)",
        Main, (numpy, ctypes = pyimport("ctypes"), address, count, dtype),
    )
end

function transfer_foreign_observation(numpy, session, generation)
    count, width, plane_width =
        SolpropJulia.session_observation_status(session, generation)
    return (
        numpy.full(count, -9.0; dtype = "float64"),
        numpy.full(count, -9; dtype = "int64"),
        numpy.full((count, width), -9.0; dtype = "float64"),
        numpy.full(plane_width, -9.0; dtype = "float64"),
    )
end

@testset "Python observation buffers reject retained interior and mixed-type overlap" begin
    @test Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing
    numpy = pyimport("numpy")
    session, generation, expected = transfer_alias_pending()
    SolpropJulia.session_observations!(expected..., session, generation)
    destinations = transfer_foreign_observation(numpy, session, generation)
    original = transfer_alias_record(session)
    count = length(expected[1])
    GC.@preserve session begin
        address = UInt(pointer(session.field)) + UInt(sizeof(Float64))
        for index in 1:4
            dtype = index == 2 ? "int64" : "float64"
            alias = transfer_foreign_view(numpy, address, length(expected[index]); dtype)
            index == 3 && (alias = alias.reshape(count, size(expected[3], 1)))
            buffers = ntuple(i -> i == index ? alias : destinations[i], 4)
            before = map(value -> pyconvert(Array, value.copy()), buffers)
            @test_throws ArgumentError SolpropJulia.copy_session_observations!(
                buffers..., session, generation,
            )
            @test isequal(map(value -> pyconvert(Array, value), buffers), before)
            @test isequal(transfer_alias_record(session), original)
        end
        matrix = transfer_foreign_view(
            numpy, address, length(expected[3]),
        ).reshape(count, 1)
        borrowed = PyArray{Float64, 2, true}(matrix; copy = false)
        for wrapper in (transpose(borrowed), PermutedDimsArray(borrowed, (2, 1)))
            @test_throws ArgumentError SolpropJulia.session_observations!(
                expected[1], expected[2], wrapper, expected[4], session, generation,
            )
            @test isequal(transfer_alias_record(session), original)
        end
    end
    backing = numpy.full(count + 1, -19.0; dtype = "float64")
    positions = pyeval("array[:-1]", Main, (array = backing,))
    steps = pyeval("array[1:].view('int64')", Main, (array = backing,))
    @test_throws ArgumentError SolpropJulia.copy_session_observations!(
        positions, steps, destinations[3], destinations[4], session, generation,
    )
    @test all(==(-19.0), pyconvert(Vector{Float64}, backing))
    @test all(==(-9.0), pyconvert(Matrix{Float64}, destinations[3]))
    @test all(==(-9.0), pyconvert(Vector{Float64}, destinations[4]))
    for _ in 1:2
        SolpropJulia.copy_session_observations!(destinations..., session, generation)
        @test pyconvert(Vector{Float64}, destinations[1]) == expected[1]
        @test pyconvert(Vector{Int64}, destinations[2]) == expected[2]
        @test pyconvert(Matrix{Float64}, destinations[3]) == permutedims(expected[3])
        @test pyconvert(Vector{Float64}, destinations[4]) == expected[4]
        @test isequal(transfer_alias_record(session), original)
    end
end

@testset "Python signed and holed strides retain exact touched byte bounds" begin
    numpy = pyimport("numpy")
    session, generation, _ = transfer_alias_pending()
    count = session.attempts.count
    original = transfer_alias_record(session)
    lengths = numpy.full(count, -9.0; dtype = "float64")
    norms = numpy.full(count, -9.0; dtype = "float64")
    accepted = numpy.zeros(count; dtype = "bool")
    elapsed = numpy.full(count, -9.0; dtype = "float64")
    GC.@preserve session begin
        address = UInt(pointer(session.field)) + UInt(sizeof(Float64))
        storage = transfer_foreign_view(numpy, address, 2count)
        for expression in ("array[1::-1]", "array[::2]", "array[2::-2]")
            positions = pyeval(expression, Main, (array = storage,))
            @test_throws ArgumentError SolpropJulia.session_attempts!(
                positions, lengths, norms, accepted, elapsed, session, generation,
            )
            @test all(==(-9.0), pyconvert(Vector{Float64}, lengths))
            @test all(==(-9.0), pyconvert(Vector{Float64}, norms))
            @test isequal(transfer_alias_record(session), original)
        end
    end
    for expression in ("array[::-1]", "array[::2]", "array[::-2]")
        size = expression == "array[::-1]" ? count : 2count
        backing = numpy.full(size, -19.0; dtype = "float64")
        positions = pyeval(expression, Main, (array = backing,))
        SolpropJulia.session_attempts!(
            positions, lengths, norms, accepted, elapsed, session, generation,
        )
        @test pyconvert(Vector{Float64}, positions) == session.attempts.z[1:count]
        @test pyconvert(Vector{Float64}, lengths) == session.attempts.dz[1:count]
        @test pyconvert(Vector{Bool}, accepted) == session.attempts.accepted[1:count]
        @test isequal(transfer_alias_record(session), original)
    end
    backing = numpy.full(2count, -29.0; dtype = "float64")
    positions = pyeval("array[1::-1]", Main, (array = backing,))
    # These Bool bytes overlap the second binary64 cell although the dtype and
    # first logical addresses differ. Pairwise validation precedes all copies.
    aliased_accepted = pyeval("array.view('bool')[8:10]", Main, (array = backing,))
    @test_throws ArgumentError SolpropJulia.session_attempts!(
        positions, lengths, norms, aliased_accepted, elapsed, session, generation,
    )
    @test all(==(-29.0), pyconvert(Vector{Float64}, backing))
end

@testset "Observation layout errors and empty copies preserve foreign owners" begin
    numpy = pyimport("numpy")
    session, generation, _ = transfer_alias_pending()
    destinations = transfer_foreign_observation(numpy, session, generation)
    before = map(value -> pyconvert(Array, value), destinations)
    for index in (1, 2, 3)
        reversed = pyeval("array[::-1]", Main, (array = destinations[index],))
        buffers = ntuple(i -> i == index ? reversed : destinations[i], 4)
        @test_throws ArgumentError SolpropJulia.copy_session_observations!(
            buffers..., session, generation,
        )
        @test isequal(map(value -> pyconvert(Array, value), destinations), before)
    end
    fresh = monitor_transfer_case()
    launch = SolpropJulia.advance!(fresh; deadline = 60.0)
    empty = transfer_foreign_observation(numpy, fresh, launch.generation)
    SolpropJulia.copy_session_observations!(empty..., fresh, launch.generation)
    @test length(pyconvert(Vector{Float64}, empty[1])) == 0
    @test size(pyconvert(Matrix{Float64}, empty[3])) == (0, 1)
    SolpropJulia.acknowledge!(fresh, launch.generation)
    @test_throws ArgumentError SolpropJulia.copy_session_observations!(
        empty..., fresh, launch.generation,
    )
end
