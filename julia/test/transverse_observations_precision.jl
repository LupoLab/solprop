# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

@testset "Transverse reconstruction restores physical readings from binary32 storage" begin
    for route in (:radial, :cartesian, :line)
        _, reference_data, physical = transverse_observation_case(route; tasks = 2)
        reference = transverse_observation_reference(reference_data, physical)
        for (state_scale, field_scale) in ((0, 0), (40, -10), (-40, 20))
            arithmetic = SolpropJulia.SourceArithmetic(
                Float32; state_exponent = state_scale, field_exponent = field_scale,
            )
            execution, prepared, same = transverse_observation_case(
                route; tasks = 2, arithmetic,
            )
            @test same == physical
            field = ComplexF32.(
                ldexp.(real.(physical), -state_scale),
                ldexp.(imag.(physical), -state_scale)
            )
            work = SolpropJulia.TransverseWorkspace(execution, prepared; trace_chunk = 3)
            before = copy(field)
            @test eltype(work.band) === ComplexF32
            @test eltype(work.nodes) === Float32
            @test eltype(work.reading.columns) === Float64
            @test all(line -> eltype(line.spectrum) === ComplexF32, work.lines)
            # The independent256-bit DFT oracle supplies absolute envelopes before
            # cancellation. The conditional64n FFT model, <=8m synthesis operations,
            # two component products, compensated sums and <=four line merges fit
            # this operation count. It is not a theorem about FFTW twiddle generation.
            strip = maximum(2length(line.strip) for line in prepared.lines)
            count = 2 * (72prepared.n_columns + 64strip + 32) + 128
            unit = BigFloat(2)^-24
            gamma = count * unit / (1 - count * unit)
            @test gamma > 0
            for _ in 1:2
                SolpropJulia._transverse_form_reading!(
                    work, field; faces = true, full = true, current = true,
                )
                reading = work.reading
                @test all(
                    abs.(BigFloat.(reading.faces) .- reference.faces) .<=
                        gamma * reference.scales.nodes
                )
                @test all(
                    abs.(BigFloat.(reading.clip) .- reference.clip) .<=
                        gamma * reference.scales.nodes
                )
                @test all(
                    abs.(BigFloat.(reading.totals) .- reference.totals) .<=
                        gamma * reference.scales.nodes
                )
                @test all(
                    abs.(BigFloat.(reading.columns) .- reference.columns) .<=
                        gamma * reference.scales.columns
                )
                @test all(
                    abs.(BigFloat.(reading.strip) .- reference.strip) .<=
                        gamma * reference.scales.strip
                )
                @test abs(BigFloat(reading.current) - reference.current) <=
                    gamma * reference.scales.current
                @test field == before
            end
            @test transverse_reading_allocations(work, field) == 0
            @test_throws ArgumentError SolpropJulia.TransverseWorkspace(
                execution, prepared; trace_chunk = 3,
                noise_spectrum = zeros(ComplexF64, size(field)),
            )
        end
    end
end
