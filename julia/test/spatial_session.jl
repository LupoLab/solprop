# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function spatial_split_allocation(candidate, work, execution)
    SolpropJulia.finish_temporal_split!(candidate, work, 0.2, execution)
    return @allocated SolpropJulia.finish_temporal_split!(candidate, work, 0.2, execution)
end

@testset "Cartesian temporal split preserves physical transform order" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for kind in (:field, :envelope), route in (:cartesian, :line)
            prepared, field = spatial_observation_case(execution, kind, route)
            rate = collect(range(0.1, 0.8; length = 8))
            work = SP._temporal_split_workspace(execution, prepared, rate)
            physical = spatial_observation_dft(field, prepared.map.shape; inverse = true)
            samples = composition_inverse(physical, 8; real_signal = kind == :field)
            expected = composition_dft(samples .* exp.(-0.1 .* rate))
            expected = spatial_observation_dft(
                expected[1:size(field, 1), :], prepared.map.shape,
            )
            saved, candidate = copy(field), copy(field)
            SP.finish_temporal_split!(candidate, work, 0.2, execution)
            # Four short transform passes plus one bounded exponential/product.
            # The absolute envelope precedes cancellation in the reference DFTs.
            bound = 4096eps(Float64) * sum(abs, field)
            @test maximum(abs.(candidate - expected)) <= bound
            @test field == saved
            @test spatial_split_allocation(candidate, work, execution) == 0
            zero_work = SP._temporal_split_workspace(execution, prepared, zeros(8))
            held = copy(candidate)
            SP.finish_temporal_split!(candidate, zero_work, 0.2, execution)
            @test candidate == held
            @test_throws ArgumentError SP.finish_temporal_split!(
                candidate, work, -0.1, execution,
            )
            @test candidate == held
            @test_throws ArgumentError SP.finish_temporal_split!(
                work.band.band, work, 0.2, execution,
            )
            @test_throws ArgumentError SP.finish_temporal_split!(
                work.temporal.spectrum, work, 0.2, execution,
            )
        end
    end
end

@testset "Spatial input-noise phase retains the represented constant operator" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    data, field = spatial_case(execution, :envelope)
    noise = SP.InputNoise(execution, field, 0.0, data.operator)
    rate = zeros(ComplexF64, size(field))
    SP.operator_rate!(rate, data.operator, 0.4, execution)
    actual = SP.input_noise_phase!(noise, 0.4, execution)
    # Both paths evaluate the identical represented rate and scalar product.
    @test actual == imag.(rate) .* 0.4
    @test (@allocated SP.input_noise_phase!(noise, 0.4, execution)) == 0
end
