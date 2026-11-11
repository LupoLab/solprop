# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import PythonCall: pyconvert, pyimport, pyeval
import SolpropJulia

include("trial_probe.jl")

function trial_probe_bridge_records(kind; low_storage = false, adams = false)
    numpy, namespace = pyimport("numpy"), pyimport("types").SimpleNamespace
    execution = trial_probe_execution(2)
    data, initial = SolpropJulia._precompile_material_case(execution, Val(kind))
    preparation = SolpropJulia.ModalPreparation(
        execution, data,
        SolpropJulia.DiagonalOperator(execution, fill(-0.01 + 0.02im, size(initial))),
        SolpropJulia.ObservationWeight(execution, ones(size(initial))), 2,
    )
    tableau = SolpropJulia._precompile_rk_tableau()
    count = length(tableau.nodes)
    transfer = namespace(;
        nodes = numpy.array(collect(tableau.nodes)),
        matrix = numpy.array([tableau.matrix[i][j] for i in 1:count, j in 1:count]),
        solution = numpy.array(collect(tableau.solution_weights)),
        error = numpy.array(collect(tableau.error_weights)),
        active = numpy.array(data.grid.active), observed = numpy.array(data.grid.active),
        weights = numpy.ones(size(initial, 1)), planes = numpy.array([0.004]),
        breakpoints = numpy.array([0.0, 0.01]), hard_kinks = numpy.empty(0),
        steps = numpy.full(4, 0.001),
        edges = numpy.array([0.0, 0.001, 0.002, 0.003, 0.004]),
        step_planes = 0, low_storage,
        control = namespace(;
            dz_init = 0.001, stretch = 1.0,
            controller = namespace(; rtol = 1.0e-3, atol = 0.0),
        ),
        # Deliberately unusable continuation/monitor/controller records prove that
        # the probe only consumes the request tableau and numerical load policy.
        adams = adams ? namespace(; unusable = true) : nothing,
        history = namespace(; unusable = true), monitor = namespace(; unusable = true),
    )
    field = numpy.array(permutedims(initial); dtype = "complex128", order = "C").reshape(
        size(initial, 2), 1, size(initial, 1),
    )
    return preparation, transfer, field
end

@testset "Python probe construction copies state and ignores continuation services" begin
    @test Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing
    numpy = pyimport("numpy")
    for kind in (:field, :envelope), adams in (false, true), low_storage in (false, true)
        preparation, transfer, field = trial_probe_bridge_records(kind; adams, low_storage)
        original = pyconvert(Array, field.copy())
        probe = SolpropJulia.start_trial_probe(preparation, transfer, field, 0.0)
        @test probe.tableau isa (
            low_storage ? SolpropJulia.RK4IPTableau :
                SolpropJulia.EmbeddedTableau
        )
        @test !hasproperty(probe, :services) && !hasproperty(probe, :history)
        @test pyconvert(Array, field) == original
        SolpropJulia.trial_probe!(probe)
        output = numpy.empty_like(field)
        if low_storage
            @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
            @test probe.state.source_evaluations == 4
        else
            SolpropJulia.copy_trial_probe_field!(output, probe)
            @test all(isfinite, pyconvert(Array, output))
            @test probe.state.source_evaluations == 7
            held = pyconvert(Array, output.copy())
            SolpropJulia.trial_probe!(probe)
            @test pyconvert(Array, output) == held
        end
        @test pyconvert(Array, field) == original
        SolpropJulia.close_trial_probe!(probe)
    end
end

@testset "Python probe readback refuses invalid buffers before consuming readiness" begin
    numpy = pyimport("numpy")
    preparation, transfer, field = trial_probe_bridge_records(:envelope)
    probe = SolpropJulia.start_trial_probe(preparation, transfer, field, 0.0)
    SolpropJulia.trial_probe!(probe)
    before = copy(probe.work.candidate)
    readonly = numpy.empty_like(field)
    readonly.flags.writeable = false
    for output in (
            readonly, numpy.zeros(field.shape; dtype = "complex64"),
            numpy.zeros((1, size(probe.field, 1)); dtype = "complex128"),
            numpy.zeros((1, 2, size(probe.field, 1)); dtype = "complex128"),
            pyeval("array[..., ::-1]", Main, (array = numpy.empty_like(field),)),
        )
        @test_throws Union{ArgumentError, DimensionMismatch, ErrorException} (
            SolpropJulia.copy_trial_probe_field!(
                output, probe,
            )
        )
        @test probe.state.ready && probe.work.candidate == before
    end
    GC.@preserve probe begin
        # Foreign Python owner, shifted within the multi-stage retained allocation.
        address = UInt(pointer(probe.work.stages, 2))
        output = pyeval(
            "numpy.ctypeslib.as_array(ctypes.cast(address, " *
                "ctypes.POINTER(ctypes.c_double)), shape=(2*n,)).view('complex128')" *
                ".reshape(1, 1, n)",
            Main, (numpy, ctypes = pyimport("ctypes"), address, n = length(probe.field)),
        )
        retained = copy(probe.work.stages)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
        @test probe.work.stages == retained && probe.state.ready
    end
    output = numpy.empty_like(field)
    SolpropJulia.copy_trial_probe_field!(output, probe)
    @test all(isfinite, pyconvert(Array, output))
    @test !probe.state.ready
    @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
end

@testset "A session observation preparation supplies only its split policy" begin
    numpy = pyimport("numpy")
    for kind in (:field, :envelope)
        preparation, transfer, field = trial_probe_bridge_records(kind)
        services = SolpropJulia._precompile_services(
            preparation.execution, preparation.data,
        )
        observed = SolpropJulia.ObservationPreparation(
            preparation.execution, services.work.data.grid, services.work.data.selection;
            kind, n_modes = 1, power_weight = preparation.power_weight,
        )
        policy = SolpropJulia.SessionObservationPolicy(
            observed; split_rate = collect(1.0:8.0) ./ 8,
        )
        prepared = SolpropJulia.SessionPreparation(preparation, policy)
        before = SolpropJulia.observation_status(services)
        probe = SolpropJulia.start_trial_probe(prepared, transfer, field, 0.0)
        @test probe.split isa SolpropJulia.TemporalSplitWorkspace
        @test probe.split.rate == policy.split_rate
        @test !Base.mightalias(probe.split.rate, policy.split_rate)
        @test !hasproperty(probe, :services)
        SolpropJulia.trial_probe!(probe)
        SolpropJulia.copy_trial_probe_field!(numpy.empty_like(field), probe)
        @test SolpropJulia.observation_status(services) == before
    end
end
