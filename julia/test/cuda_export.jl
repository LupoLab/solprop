# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import CUDA
import SolpropJulia as SP

@test CUDA.functional()
CUDA.allowscalar(false)

function _cuda_export_session(execution, field, weight)
    return (
        execution = execution, field = SP.upload(execution, field),
        exports = SP._FieldExport(execution, weight; shape = size(field)),
        state = (
            closed = false, pending = true, poisoned = false, generation = 1, z = 0.25,
        ),
    )
end

@testset "CUDA binary64 exports preserve the destination recipe and ownership" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    gpu = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    field = reshape(
        ComplexF64[
            complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.0, -0.0),
            complex(-0.0, -0.0), 0.5 + 0.25im, -0.25 + 0.5im,
            1.0e-200 - 2.0e-200im, -2.0e100 + 1.0e100im,
        ], 4, 2,
    )
    for unit in (true, false), power in (false, true), optics in (false, true)
        weights = ones(size(field))
        unit || (weights[4, 2] = 2.0)
        weight = SP.ObservationWeight(gpu, weights)
        candidate = _cuda_export_session(gpu, field, weight)
        reference = _cuda_export_session(cpu, field, SP.ObservationWeight(cpu, weights))
        output, expected = similar(field), similar(field)
        SP.export_session!(expected, reference, 1, weights, power, optics)
        SP.export_session!(output, candidate, 1, weight.data, power, optics)
        @test all(isfinite, output)
        # Same retained field and specified recipe; allow the correctly rounded
        # square root and at most eight products/additions per component. Zeros
        # remain exact structural contracts, including their signs.
        unit_roundoff = eps(Float64) / 2
        gamma = 16unit_roundoff / (1 - 16unit_roundoff)
        @test gamma > 0
        for index in eachindex(output)
            if iszero(field[index]) || !power
                @test isequal(output[index], expected[index])
            else
                @test abs(output[index] - expected[index]) <= gamma * abs(expected[index])
            end
        end
        @test candidate.state.pending && candidate.state.generation == 1
        retained = similar(field)
        SP.copy_host!(retained, candidate.field, gpu)
        @test isequal(retained, field)
        @test CUDA.context() == context && CUDA.stream() == stream
        @test_throws ArgumentError SP.export_session!(
            candidate.exports.transfer.field.host, candidate, 1, weight.data, power, optics,
        )
        device_output = SP.upload(gpu, fill(7.0 + 8.0im, size(field)))
        @test_throws ArgumentError SP.export_session!(
            device_output, candidate, 1, weight.data, power, optics,
        )
        SP.copy_host!(retained, device_output, gpu)
        @test all(==(7.0 + 8.0im), retained)
    end

    initial = fill(complex(ldexp(1.5, 1023), ldexp(0.5, 1023)), size(field))
    weight = SP.ObservationWeight(gpu, fill(4.0, size(field)))
    candidate = _cuda_export_session(gpu, initial, weight)
    output = fill(7.0 + 8.0im, size(field))
    @test_throws "physical field export exceeds binary64 range" SP.export_session!(
        output, candidate, 1, weight.data, true, true,
    )
    @test all(==(7.0 + 8.0im), output)
    @test candidate.state.pending
    SP.export_session!(output, candidate, 1, SP.upload(gpu, ones(size(field))), true, false)
    @test isequal(output, initial)
    other = merge(candidate, (; execution = SP.device_execution(Val(:cuda), 0, 1)))
    @test_throws ArgumentError SP.export_session!(
        output, other, 1, weight.data, false, false,
    )
    @test isequal(output, initial)
end

@testset "CUDA modal power-weight representations keep only scalar facts on the host" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    gpu = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    coefficients = reshape([2.0, 3.0, 4.0, 5.0, 0.125, -0.25, 0.5, -0.125], 4, 2, 1)
    table = SP.ChebyshevTable(cpu, [0.0, 1.0], coefficients)
    direct = SP.ObservationWeight(gpu, table, (2, 2))
    terms = reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)
    factors = SP.ChebyshevTable(cpu, [0.0, 1.0], reshape([2.0, 0.25], 1, 2, 1))
    expansion = SP.ObservationWeight(gpu, terms, factors)
    for (weight, count, expected) in (
            (direct, 0, reshape([1.9375, 3.125, 3.75, 5.0625], 2, 2)),
            (expansion, 1, reshape([1.875, 3.75, 5.625, 7.5], 2, 2)),
        )
        output = SP.dzeros(gpu, Float64, 2, 2)
        status = SP.DeviceReadback(gpu, Int, 2, 2)
        SP.evaluate_weight!(
            output, weight, 0.25, gpu; work = SP.dzeros(gpu, Float64, count), status,
        )
        host = zeros(2, 2)
        SP.copy_host!(host, output, gpu)
        # This degree-one dyadic example is exact, independently of contraction.
        @test host == expected
        @test status.host == [0 0; 1 1]
    end
    for value in (0.0, -1.0, Inf, NaN)
        values = SP.upload(gpu, fill(value, 2, 2))
        status = SP.DeviceReadback(gpu, Int, 2, 2)
        @test_throws ArgumentError SP._weight_summary!(status, values, gpu)
        @test status.host[1, :] == [1, 1]
    end
end
