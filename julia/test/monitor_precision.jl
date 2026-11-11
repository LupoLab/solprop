# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia


function precision_monitor_allocations(
        reading, samples, weights, power, execution, spectrum, exponent
    )
    SolpropJulia.temporal_reading!(
        reading, samples, weights, power, execution;
        spectrum, amplitude_exponent = exponent
    )
    return @allocated SolpropJulia.temporal_reading!(
        reading, samples, weights, power, execution;
        spectrum, amplitude_exponent = exponent
    )
end

@testset "Monitor physical exponents refuse integer overflow before writes" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    samples = ones(ComplexF32, 4, 4)
    weights = SolpropJulia.TemporalMonitorWeights(ones(4, 4), ones(4))
    reading, power = SolpropJulia.TemporalReading(), fill(19.0f0, 4)
    for exponent in (typemin(Int), typemax(Int))
        @test_throws ArgumentError SolpropJulia.temporal_reading!(
            reading, samples, weights, power, execution; amplitude_exponent = exponent
        )
        @test_throws ArgumentError SolpropJulia.split_reading!(
            reading, samples, 0.25, weights, power, execution;
            amplitude_exponent = exponent
        )
        @test power == fill(19.0f0, 4) && reading.total == 0.0
    end
end

function precision_split_allocations(reading, samples, weights, power, execution, exponent)
    SolpropJulia.split_reading!(
        reading, samples, 0.125, weights, power, execution;
        amplitude_exponent = exponent
    )
    return @allocated SolpropJulia.split_reading!(
        reading, samples, 0.125, weights, power, execution;
        amplitude_exponent = exponent
    )
end

@testset "Binary32 temporal monitor restores physical readings" begin
    for tasks in (1, 2), kind in (:field, :envelope), exponent in (-100, 100)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        base = reshape(Float32.(1:12) ./ 16.0f0, 4, 3)
        samples = (kind == :field ? base : complex.(base, reverse(base; dims = 1))) .*
            2.0f0^exponent
        spectrum = ComplexF32.(base[1:3, :], reverse(base[1:3, :]; dims = 2)) .*
            2.0f0^exponent
        weights = SolpropJulia.TemporalMonitorWeights(
            [1.0 0.0 0.25 0.5; 0 0 0 0; 0 0 0 0; 0 1 0.5 2],
            [0.25, 1.0, 2.0]; buckets = [1 2 3; 2 3 1; 3 1 2],
        )
        reading, power = SolpropJulia.TemporalReading(3), zeros(Float32, 4)
        amplitude_exponent = -exponent + 20
        SolpropJulia.temporal_reading!(
            reading, samples, weights, power, execution;
            spectrum, amplitude_exponent
        )
        physical = Complex{BigFloat}.(samples) .* big(2.0)^amplitude_exponent
        expected_power = [
            sum(
                BigFloat(weights.columns[column]) * abs2(physical[row, column])
                    for column in 1:3
            ) for row in 1:4
        ]
        expected = [
            sum(expected_power);
            [sum(expected_power .* weights.regions[:, region]) for region in 1:4]
        ]
        # Three pointwise products and two positive Sum2s reductions; 32u includes
        # input weight conversions and both final binary64 scale restorations.
        bound = 32 * 2.0^-24
        @test bound > 0
        @test maximum(abs.(BigFloat.([reading.total; reading.regions...]) - expected)) <=
            bound * maximum(expected)
        physical_spectrum = Complex{BigFloat}.(spectrum) .* big(2.0)^amplitude_exponent
        for bucket in 1:3
            expected_bucket = sum(
                BigFloat(weights.columns[column]) * abs2(physical_spectrum[row, column])
                    for column in 1:3, row in 1:3 if weights.buckets[row, column] == bucket
            )
            @test abs(BigFloat(reading.occupancy[bucket]) - expected_bucket) <=
                bound * expected_bucket
        end
        @test precision_monitor_allocations(
            reading, samples, weights, power, execution, spectrum, amplitude_exponent
        ) == 0
        SolpropJulia.split_reading!(
            reading, samples, 0.125, weights, power, execution; amplitude_exponent
        )
        expected_removed = sum(
            expected_power .* -expm1.(-BigFloat.(weights.regions[:, 4]) ./ 8)
        )
        halfway = expected_power .* exp.(-BigFloat.(weights.regions[:, 4]) ./ 16)
        expected_found = [
            sum(halfway); [sum(halfway .* weights.regions[:, region]) for region in 1:3]
        ]
        # Wide host exp/expm1 then one Float32 rounding adds one product error to
        # the preceding positive-sum model. 48u bounds these conditioned fixtures.
        @test abs(BigFloat(reading.removed) - expected_removed) <=
            48 * 2.0^-24 * expected_removed
        @test maximum(abs.(BigFloat.(collect(reading.found)) - expected_found)) <=
            48 * 2.0^-24 * maximum(expected_found)
        @test precision_split_allocations(
            reading, samples, weights, power, execution, amplitude_exponent
        ) == 0
        old = reading.removed
        SolpropJulia.split_reading!(
            reading, samples, 0.125, weights, power, execution;
            amplitude_exponent, add = true
        )
        @test reading.removed == 2old
        @test eltype(power) === Float32 && eltype(reading.occupancy) === Float64
        snapshot = (reading.total, reading.regions, copy(reading.occupancy))
        @test_throws ArgumentError SolpropJulia.temporal_reading!(
            reading, samples, weights,
            power, execution; spectrum, amplitude_exponent = 1000
        )
        @test (reading.total, reading.regions, reading.occupancy) == snapshot
    end
end

function precision_transverse_tables()
    faces = [0.25 0 0 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0]
    totals = [0.25 0.125 0.5 0; 0.25 0.25 0.5 0; 0.25 0 0 0; 0.25 0 0 0]
    return SolpropJulia.TransverseTables(
        faces, copy(faces), totals, ones(4),
        [0.0, 1.0, 3.0, 6.0], [1, 2, 3, 4], zeros(4);
        face_ranks = (2, 1), noise_face_share = 0.25, noise_modes = 16
    )
end

function precision_transverse_allocations(
        reading, nodes, columns, tables, work, execution, node_exponent, column_exponent
    )
    SolpropJulia.transverse_reading!(
        reading, nodes, columns, tables, work, execution;
        node_exponent, column_exponent, full = true
    )
    return @allocated SolpropJulia.transverse_reading!(
        reading, nodes, columns, tables, work, execution;
        node_exponent, column_exponent, full = true
    )
end

function precision_strip_current_allocations(reading, strips, currents, execution, exponent)
    SolpropJulia.transverse_strip!(
        reading, strips, execution; amplitude_exponent = exponent
    )
    SolpropJulia.transverse_current!(reading, currents, 4.0; amplitude_exponent = exponent)
    return @allocated begin
        SolpropJulia.transverse_strip!(
            reading, strips, execution; amplitude_exponent = exponent
        )
        SolpropJulia.transverse_current!(
            reading, currents, 4.0; amplitude_exponent = exponent
        )
    end
end

@testset "Binary32 transverse reductions retain distinct state and field scales" begin
    for tasks in (1, 2), exponent in (-100, 100)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        base = reshape(ComplexF32.(1:16, reverse(1:16)), 4, 4) ./ 16.0f0
        nodes = base .* 2.0f0^exponent
        columns = reverse(base; dims = 2) .* 2.0f0^-exponent
        tables = precision_transverse_tables()
        reading, scratch = SolpropJulia.TransverseReading(4, 2), zeros(Float32, 4)
        node_exponent, column_exponent = -exponent + 10, exponent - 10
        SolpropJulia.transverse_reading!(
            reading, nodes, columns, tables, scratch, execution;
            node_exponent, column_exponent, full = true
        )
        node_norms = vec(sum(abs2.(Complex{BigFloat}.(nodes)); dims = 1)) .*
            big(2.0)^(2node_exponent)
        column_norms = vec(sum(abs2.(Complex{BigFloat}.(columns)); dims = 1)) .*
            big(2.0)^(2column_exponent)
        bound = 32 * 2.0^-24
        @test maximum(abs.(BigFloat.(reading.columns) - column_norms)) <=
            bound * maximum(column_norms)
        for (actual, weights) in (
                (reading.faces, tables.faces), (reading.clip, tables.clip),
                (reading.totals, tables.totals),
            )
            expected = [sum(node_norms .* weights[:, face]) for face in 1:4]
            @test maximum(abs.(BigFloat.(collect(actual)) - expected)) <=
                bound * maximum(expected)
        end
        @test reading.full && reading.has_faces &&
            !reading.has_strip && !reading.has_current
        @test eltype(scratch) === Float32
        @test precision_transverse_allocations(
            reading, nodes, columns, tables, scratch, execution,
            node_exponent, column_exponent
        ) == 0
        strips = (
            SolpropJulia.TransverseStripLine(nodes, 0.25, 2.0, 2, 3),
            SolpropJulia.TransverseStripLine(zeros(ComplexF32, 0, 1), 0.25, 2.0, 1, 0),
        )
        currents = (SolpropJulia.TransverseCurrentLine(nodes, 2, 1, 3, 0.5, 2.0),)
        SolpropJulia.transverse_strip!(
            reading, strips, execution; amplitude_exponent = node_exponent
        )
        SolpropJulia.transverse_current!(
            reading, currents, 4.0; amplitude_exponent = node_exponent
        )
        expected_strip = [
            sum(abs2, Complex{BigFloat}.(nodes[2:3, :])),
            sum(abs2, Complex{BigFloat}.(nodes)),
        ] .* big(2.0)^(2node_exponent) ./ 8
        @test maximum(abs.(BigFloat.(reading.strip[1, :]) - expected_strip)) <=
            bound * maximum(expected_strip)
        @test all(iszero, reading.strip[2, :])
        physical = Complex{BigFloat}.(nodes) .* big(2.0)^node_exponent
        expected_current = sum(
            max(
                imag(
                    conj(physical[2, column]) *
                        (physical[3, column] - physical[1, column])
                ), big(0.0)
            ) for column in 1:4
        )
        @test abs(BigFloat(reading.current) - expected_current) <=
            bound * max(expected_current, big(1.0))
        @test precision_strip_current_allocations(
            reading, strips, currents, execution, node_exponent
        ) == 0
    end
end

@testset "Monitor foreign scratch preserves every observed input" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    for T in (Float32, Float64)
        samples = reshape(T.(1:8), 4, 2)
        weights = SolpropJulia.TemporalMonitorWeights(ones(4, 4), ones(2))
        reading, original = SolpropJulia.TemporalReading(), copy(samples)
        GC.@preserve samples begin
            power = unsafe_wrap(Array, pointer(samples, 2), (4,); own = false)
            @test !Base.mightalias(power, samples)
            @test_throws ArgumentError SolpropJulia.temporal_reading!(
                reading, samples, weights, power, execution
            )
            @test samples == original && reading.total == 0.0
            @test_throws ArgumentError SolpropJulia.split_reading!(
                reading, samples, 0.125, weights, power, execution
            )
            @test samples == original && !reading.has_found
        end
        # Ordinary independent scratch remains accepted after the refused call.
        SolpropJulia.temporal_reading!(reading, samples, weights, zeros(T, 4), execution)
        @test reading.total == 204.0
    end
    samples = reshape(Float32.(1:16), 4, 4)
    tables = precision_transverse_tables()
    reading, original = SolpropJulia.TransverseReading(4, 2), copy(samples)
    GC.@preserve samples begin
        nodes = unsafe_wrap(Array, pointer(samples, 2), (4,); own = false)
        @test !Base.mightalias(nodes, samples)
        @test_throws ArgumentError SolpropJulia.transverse_reading!(
            reading, samples, copy(samples), tables, nodes, execution
        )
        @test samples == original && !reading.has_faces
    end
end

@testset "Monitor products scale before unrelated weights can erase them" begin
    # At most five factors, their normalization/conversion and product rounding,
    # then a positive Sum2s and host restoration fit64u on these short sums.
    # No tolerance is relative to a different, larger reading family.
    bound = 64big(2.0)^-24
    @test bound > 0
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        samples = reshape(Float32[2.0f0^-100, 2.0f0^100], 1, 2)
        weights = SolpropJulia.TemporalMonitorWeights(
            ones(1, 4), [
                2.0^100,
                2.0^-100,
            ]; buckets = [1 2]
        )
        reading, power = SolpropJulia.TemporalReading(2), zeros(Float32, 1)
        spectrum = ComplexF32.(samples)
        SolpropJulia.temporal_reading!(
            reading, samples, weights, power,
            execution; spectrum
        )
        expected = [
            BigFloat(weights.columns[column]) * abs2(
                BigFloat(
                    samples[
                        1,
                        column,
                    ]
                )
            ) for column in 1:2
        ]
        @test abs(BigFloat(reading.total) - sum(expected)) <= bound * sum(expected)
        @test all(abs.(BigFloat.(reading.occupancy) - expected) .<= bound .* expected)
        @test all(>(0), reading.occupancy)
        @test precision_monitor_allocations(
            reading, samples, weights, power,
            execution, spectrum, 0
        ) == 0

        # An apparently insignificant power row becomes dominant in a separately
        # weighted region. Its source must survive until that region is evaluated.
        samples = reshape(Float32[2.0f0^-100, 2.0f0^100], 2, 1)
        regions = [2.0^700 0.0 1.0 2.0^-100; 2.0^-700 0.0 1.0 2.0^-500]
        weights = SolpropJulia.TemporalMonitorWeights(regions, [1.0])
        reading, power = SolpropJulia.TemporalReading(), zeros(Float32, 2)
        SolpropJulia.temporal_reading!(reading, samples, weights, power, execution)
        expected_power = abs2.(BigFloat.(vec(samples)))
        expected_region = sum(expected_power .* BigFloat.(regions[:, 1]))
        @test abs(BigFloat(reading.regions[1]) - expected_region) <= bound * expected_region
        @test reading.regions[2] == 0.0
        SolpropJulia.split_reading!(reading, samples, 0.125, weights, power, execution)
        expected_removed = sum(expected_power .* -expm1.(-BigFloat.(regions[:, 4]) ./ 8))
        expected_found = sum(
            expected_power .* exp.(
                -BigFloat.(
                    regions[
                        :,
                        4,
                    ]
                ) ./ 16
            ) .* BigFloat.(regions[:, 1])
        )
        @test abs(BigFloat(reading.removed) - expected_removed) <= bound * expected_removed
        @test abs(BigFloat(reading.found[2]) - expected_found) <= bound * expected_found
        @test precision_split_allocations(
            reading, samples, weights, power,
            execution, 0
        ) == 0

        nodes = reshape(ComplexF32[2.0f0^-100, 2.0f0^100, 0, 0], 1, 4)
        face = zeros(4, 4)
        face[1:2, 1] .= [2.0^700, 2.0^-700]
        tables = SolpropJulia.TransverseTables(
            face, copy(face), copy(face), ones(4),
            [0.0, 1.0, 3.0, 6.0], [1, 2, 3, 4], zeros(4);
            face_ranks = (2, 1), noise_face_share = 0.25, noise_modes = 16
        )
        edge, scratch = SolpropJulia.TransverseReading(4, 1), zeros(Float32, 4)
        SolpropJulia.transverse_reading!(edge, nodes, nodes, tables, scratch, execution)
        @test abs(BigFloat(edge.faces[1]) - expected_region) <= bound * expected_region
        @test precision_transverse_allocations(
            edge, nodes, nodes, tables,
            scratch, execution, 0, 0
        ) == 0

        strips = (
            SolpropJulia.TransverseStripLine(
                reshape(vec(nodes), 4, 1), 1.0,
                1.0, 1, 1
            ),
        )
        SolpropJulia.transverse_strip!(edge, strips, execution)
        expected_inward = BigFloat(2.0)^-200 / 4
        @test abs(BigFloat(edge.strip[1, 1]) - expected_inward) <= bound * expected_inward
        @test edge.strip[1, 1] > 0
        # The large sample lies outside the actual three-point current stencil.
        field = reshape(ComplexF32[0, 2.0f0^-100, im * 2.0f0^-100, 2.0f0^100], 4, 1)
        current = (SolpropJulia.TransverseCurrentLine(field, 2, 1, 3, 1.0, 1.0),)
        SolpropJulia.transverse_current!(edge, current, 1.0)
        @test abs(BigFloat(edge.current) - BigFloat(2.0)^-200) <= bound * BigFloat(2.0)^-200
    end
end
