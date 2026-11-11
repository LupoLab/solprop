# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import PythonCall: Py, PyException, pyimport, pyeval, pyexec, pyconvert
import SolpropJulia

include("monitor_transfer.jl")

function monitor_bridge_record(; has_quiet_run = false)
    namespace = pyimport("types").SimpleNamespace
    return namespace(;
        launch = 8.0, launch_occupancy = (0.125, 0.25, 0.375),
        guard_max = (0.25, 0.375), guard_at = (0.1, 0.2), collar_max = 0.5,
        raw_removed = 0.5, floor_removed = 0.25, quiet_run = 3, quiet = true,
        raw_guard_max = 0.625, steps = 7, skipped = 5, left_out_checks = 2,
        last_plane_z = -0.1, has_quiet_run,
    )
end

@testset "Python monitor rows require writable independent C-order binary64 storage" begin
    @test Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing
    numpy = pyimport("numpy")
    session = monitor_transfer_case()
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    SolpropJulia.acknowledge!(session, launch.generation)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    count = SolpropJulia.session_monitor_payload(session, plane.generation).row_count
    @test count == 2
    expected = zeros(5, count)
    SolpropJulia.session_monitor_rows!(expected, session, plane.generation)
    destination = numpy.full((count, 5), -1.0; dtype = "float64")
    for _ in 1:2
        SolpropJulia.copy_session_monitor_rows!(destination, session, plane.generation)
        @test isequal(pyconvert(Matrix{Float64}, destination), permutedims(expected))
        @test session.services.monitor_count == count
    end
    pyexec("array[:] = -2", Main, (array = destination,))
    unchanged = zeros(5, count)
    SolpropJulia.session_monitor_rows!(unchanged, session, plane.generation)
    @test isequal(unchanged, expected)
    for (invalid, error) in (
            (numpy.full((count, 5), -1.0; dtype = "float32"), ErrorException),
            (numpy.full((count * 5,), -1.0; dtype = "float64"), ErrorException),
            (numpy.full((count, 5), -1.0; dtype = "float64", order = "F"), ArgumentError),
            (numpy.full((count, 4), -1.0; dtype = "float64"), DimensionMismatch),
        )
        before = pyconvert(Array, invalid.copy())
        @test_throws error SolpropJulia.copy_session_monitor_rows!(
            invalid, session, plane.generation,
        )
        @test isequal(pyconvert(Array, invalid), before)
    end
    readonly = numpy.full((count, 5), -1.0; dtype = "float64")
    pyexec("array.flags.writeable = False", Main, (array = readonly,))
    @test_throws ErrorException SolpropJulia.copy_session_monitor_rows!(
        readonly, session, plane.generation,
    )
    @test all(==(-1.0), pyconvert(Matrix{Float64}, readonly))
    backing = numpy.full((count, 10), -1.0; dtype = "float64")
    strided = pyeval("array[:, ::2]", Main, (array = backing,))
    @test_throws ArgumentError SolpropJulia.copy_session_monitor_rows!(
        strided, session, plane.generation,
    )
    @test all(==(-1.0), pyconvert(Matrix{Float64}, backing))
    original = copy(session.field)
    GC.@preserve session begin
        address = UInt(pointer(session.field)) + UInt(sizeof(Float64))
        alias = pyeval(
            "numpy.ctypeslib.as_array(ctypes.cast(address, " *
                "ctypes.POINTER(ctypes.c_double)), shape=(10,)).reshape(2, 5)",
            Main, (numpy, ctypes = pyimport("ctypes"), address),
        )
        @test_throws ArgumentError SolpropJulia.copy_session_monitor_rows!(
            alias, session, plane.generation,
        )
    end
    @test session.field == original
    SolpropJulia.copy_session_monitor_rows!(destination, session, plane.generation)
    @test isequal(pyconvert(Matrix{Float64}, destination), permutedims(expected))
    SolpropJulia.acknowledge!(session, plane.generation)
    @test_throws ArgumentError SolpropJulia.copy_session_monitor_rows!(
        destination, session, plane.generation,
    )
end

@testset "Python monitor continuation conversion completes before state mutation" begin
    session = monitor_transfer_case()
    source = monitor_bridge_record()
    SolpropJulia.restore_session_monitor!(session, source, true)
    record = SolpropJulia.session_monitor_payload(session, 0).record
    @test record.resuming && record.continuation.launch == 8.0
    @test !session.services.work.monitor.restore_quiet
    @test session.services.work.monitor.restored_quiet_run == 3
    pyexec(
        "record.raw_removed = 99; record.launch_occupancy = (99, 99, 99)",
        Main, (record = source,),
    )
    @test record.continuation.raw_removed ==
        session.services.work.monitor.raw_removed == 0.5
    @test session.services.work.monitor.launch_occupancy == (0.125, 0.25, 0.375)
    absent = monitor_transfer_case()
    SolpropJulia.restore_session_monitor!(absent, Py(nothing), false)
    @test SolpropJulia.session_monitor_payload(absent, 0).record.resuming
    for malformed in (:last_plane_z, :has_quiet_run, :guard_max)
        target = monitor_transfer_case()
        bad = monitor_bridge_record()
        pyexec("delattr(record, name)", Main, (record = bad, name = String(malformed)))
        before = SolpropJulia.session_monitor_payload(target, 0)
        @test_throws PyException SolpropJulia.restore_session_monitor!(target, bad, true)
        monitor_transfer_payload_equal(
            before, SolpropJulia.session_monitor_payload(target, 0),
        )
    end
end
