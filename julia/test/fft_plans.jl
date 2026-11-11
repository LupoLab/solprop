# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import SolpropJulia as SJ
import LinearAlgebra

function fft_direct_reference(input, direction)
    n = length(input)
    return setprecision(256) do
        [
            sum(
                Complex{BigFloat}(input[j + 1]) *
                    cis(BigFloat(direction) * 2big(pi) * j * k / n)
                    for j in 0:(n - 1)
            ) for k in 0:(n - 1)
        ]
    end
end

function fft_rounding_bound(input)
    # Every tested size is <= 16. The qualification model assumes <=64n rounded
    # scalar operations per magnitude-propagated contribution, and <=n unit-modulus
    # twiddle factors with absolute error <=4u each. The latter is an explicit
    # hypothesis on the selected FFTW codelets/constant generation, not a consequence
    # of operation counting or a proof about every FFTW size. Product perturbations
    # give (1+gamma_64n)(1+gamma_4n)-1, with absolute-sum conditioning. Independent
    # 256-bit DFT rounding contributes <2^-200 times this scale. These bounded
    # witnesses qualify the tested small transforms under that stated model.
    T = typeof(real(zero(eltype(input))))
    u = BigFloat(eps(T)) / 2
    count = 64length(input)
    gamma = count * u / (1 - count * u)
    twiddle = 4length(input) * u / (1 - 4length(input) * u)
    return Float64(
        (gamma + twiddle + gamma * twiddle) *
            sum(abs, Complex{BigFloat}.(input))
    )
end

function fft_plan_allocations(output, plan, input)
    LinearAlgebra.mul!(output, plan, input)
    return @allocated LinearAlgebra.mul!(output, plan, input)
end

@testset "multiaxis complex transform and second-process wisdom reuse" begin
    input = reshape(ComplexF64[complex(j / 16, (7 - j) / 32) for j in 1:16], 4, 4)
    output = similar(input)
    plan = SJ.plan_owned_fft!(SJ.FFTPlanning(), input, output, (1, 2), :c2c_forward)
    expected = setprecision(256) do
        [
            sum(
                Complex{BigFloat}(input[j + 1, l + 1]) *
                    cis(-2big(pi) * (BigFloat(j * k) / 4 + BigFloat(l * m) / 4))
                    for j in 0:3, l in 0:3
            ) for k in 0:3, m in 0:3
        ]
    end
    LinearAlgebra.mul!(output, plan, input)
    @test maximum(abs.(Complex{BigFloat}.(output) .- expected)) <=
        fft_rounding_bound(vec(input))
    fft_plan_allocations(output, plan, input)
    @test fft_plan_allocations(output, plan, input) == 0

    x, y = ComplexF64.(1:8), zeros(ComplexF64, 8)
    context = SJ.FFTPlanning(; effort = :patient)
    SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
    key, wisdom = only(context.measured)
    mktempdir() do directory
        path = joinpath(directory, "wisdom.txt")
        write(path, wisdom)
        program = """
        import SolpropJulia as SJ
        import LinearAlgebra
        context = SJ.FFTPlanning(; offered=Dict(ARGS[1] => read(ARGS[2], String)))
        input, output = ComplexF64.(1:8), zeros(ComplexF64, 8)
        plan = SJ.plan_owned_fft!(context,input,output,(1,),:c2c_forward)
        @assert only(context.reused) == ARGS[1]
        @assert isempty(context.measured) && isempty(context.rejected)
        LinearAlgebra.mul!(output,plan,input)
        @assert output[1] == 36 # exactly representable DC sum
        print("reused")
        """
        project = dirname(Base.active_project())
        command = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes
            --project=$project -e $program $key $path`
        @test read(command, String) == "reused"
    end
end

fft_wisdom_lines(wisdom) = sort!(strip.(split(wisdom, '\n'; keepempty = false)))
function fft_global_state()
    return SJ.FFTW.@exclusive (
        fft_wisdom_lines(SJ._fft_export(Float64)),
        fft_wisdom_lines(SJ._fft_export(Float32)),
        SJ._fft_threads(Float64), SJ._fft_threads(Float32),
    )
end

@testset "owned complex FFT planning and independent DFT" begin
    for T in (Float32, Float64), n in (7, 8, 16), direction in (-1, 1)
        input = Complex{T}[complex((j - 3) / 8, (2 - j) / 16) for j in 1:n]
        output = fill(Complex{T}(5, -2), n)
        expected = fft_direct_reference(input, direction)
        before_input, before_output = copy(input), copy(output)
        kind = direction == -1 ? :c2c_forward : :c2c_backward
        context = SJ.FFTPlanning()
        plan = SJ.plan_owned_fft!(context, input, output, (1,), kind)
        @test isequal(input, before_input)
        @test isequal(output, before_output)
        @test length(context.measured) == 1
        @test isempty(context.reused)
        LinearAlgebra.mul!(output, plan, input)
        bound = fft_rounding_bound(before_input)
        @test bound > 0
        @test maximum(abs.(Complex{BigFloat}.(output) .- expected)) <= bound
        @test isequal(input, before_input)
        fft_plan_allocations(output, plan, input) # warm the measurement specialization too
        @test fft_plan_allocations(output, plan, input) == 0
        inplace = copy(input)
        inplace_plan = SJ.plan_owned_fft!(context, inplace, inplace, (1,), kind)
        @test isequal(inplace, before_input)
        LinearAlgebra.mul!(inplace, inplace_plan, inplace)
        @test maximum(abs.(Complex{BigFloat}.(inplace) .- expected)) <= bound
    end
end

@testset "owned real FFT planning and raw inverse" begin
    for T in (Float32, Float64), n in (7, 8, 16), axis in (1, 2)
        shape = axis == 1 ? (n, 2) : (2, n)
        half_shape = axis == 1 ? (fld(n, 2) + 1, 2) : (2, fld(n, 2) + 1)
        input = reshape(T[(j - 3) / 16 for j in 1:prod(shape)], shape)
        spectrum = fill(Complex{T}(2, 1), half_shape)
        output = fill(T(-3), shape)
        context = SJ.FFTPlanning()
        before_input, before_spectrum = copy(input), copy(spectrum)
        forward = SJ.plan_owned_fft!(context, input, spectrum, (axis,), :r2c)
        backward = SJ.plan_owned_fft!(context, spectrum, output, (axis,), :c2r)
        @test isequal(input, before_input)
        @test isequal(spectrum, before_spectrum)
        @test all(==(-3), output)
        LinearAlgebra.mul!(spectrum, forward, input)
        saved = copy(spectrum)
        LinearAlgebra.mul!(output, backward, spectrum)
        @test isequal(spectrum, saved)
        for column in 1:2
            lane = axis == 1 ? view(input, :, column) : view(input, column, :)
            actual = axis == 1 ? view(spectrum, :, column) : view(spectrum, column, :)
            expected = fft_direct_reference(lane, -1)
            bound = fft_rounding_bound(lane)
            @test bound > 0
            @test maximum(abs.(Complex{BigFloat}.(actual) .- expected[1:length(actual)])) <=
                bound
            reconstructed = axis == 1 ? view(output, :, column) : view(output, column, :)
            # Forward and backward each have the preceding absolute FFT budget;
            # the raw backward norm contributes n to the first transform's error.
            roundtrip = n * bound + fft_rounding_bound(Complex{T}.(expected))
            @test roundtrip > 0
            @test maximum(abs.(reconstructed .- n .* lane)) <= roundtrip
        end
        fft_plan_allocations(spectrum, forward, input)
        fft_plan_allocations(output, backward, spectrum)
        @test fft_plan_allocations(spectrum, forward, input) == 0
        @test fft_plan_allocations(output, backward, spectrum) == 0
    end
end

@testset "wisdom rigor, isolation, detached facts and provider identity" begin
    x, y = ComplexF64.(1:8), zeros(ComplexF64, 8)
    patient = SJ.FFTPlanning(; effort = :patient)
    SJ.plan_owned_fft!(patient, x, y, (1,), :c2c_forward)
    key = only(keys(patient.measured))
    offered = copy(patient.measured)
    context = SJ.FFTPlanning(; offered)
    empty!(offered)
    SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
    @test only(context.reused) == key
    @test isempty(context.measured)
    @test isempty(context.rejected)
    estimate = SJ.FFTPlanning(; offered = patient.measured, effort = :estimate)
    SJ.plan_owned_fft!(estimate, x, y, (1,), :c2c_forward)
    @test only(estimate.reused) == key
    @test isempty(estimate.measured)
    unknown = SJ.FFTPlanning(; effort = :estimate)
    SJ.plan_owned_fft!(unknown, x, y, (1,), :c2c_forward)
    @test isempty(unknown.measured) && isempty(unknown.reused)
    @test haskey(unknown.applied, key)
    discarded = SJ.FFTPlanning(; keep = false)
    SJ.plan_owned_fft!(discarded, x, y, (1,), :c2c_forward)
    @test isempty(discarded.measured)
    for bad in ("bad", "bad\0wisdom", replace(patient.measured[key], "fftw" => "fftwf"))
        context = SJ.FFTPlanning(; offered = Dict(key => bad))
        SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
        @test haskey(context.rejected, key)
        @test haskey(context.measured, key)
        @test isempty(context.reused)
    end
    # Valid wisdom for a distinct transform must not be mistaken for a known key.
    other = SJ.FFTPlanning()
    SJ.plan_owned_fft!(other, ComplexF64.(1:7), zeros(ComplexF64, 7), (1,), :c2c_forward)
    context = SJ.FFTPlanning(; offered = Dict(key => only(values(other.measured))))
    SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
    @test occursin("does not cover", context.rejected[key])
    @test haskey(context.measured, key)
    record = SJ.fft_planning_record(patient)
    empty!(record.measured)
    empty!(record.applied)
    push!(record.reused, "caller")
    record.rejected["caller"] = "caller"
    @test haskey(patient.measured, key) && haskey(patient.applied, key)
    @test isempty(patient.reused) && isempty(patient.rejected)
    identity = SJ.fft_runtime_identity()
    @test all(value -> value isa String && !isempty(value), identity)
    @test identity.provider == "fftw"
    @test identity.julia_version == string(VERSION)
    @test identity.fftw_package_version == string(Base.pkgversion(SJ.FFTW))
    @test isfile(identity.library_f64) && isfile(identity.library_f32)
end

@testset "actual layout/alignment keys and preserved backing storage" begin
    for T in (Float32, Float64), reverse in (false, true)
        input_backing = Complex{T}.(reshape(1:48, 24, 2))
        output_backing = fill(Complex{T}(11, -3), 24, 2)
        indices = reverse ? (20:-2:6) : (3:2:17)
        input, output = view(input_backing, indices, :), view(output_backing, indices, :)
        before_input, before_output = copy(input_backing), copy(output_backing)
        context = SJ.FFTPlanning()
        plan = SJ.plan_owned_fft!(context, input, output, (1,), :c2c_forward)
        @test isequal(input_backing, before_input)
        @test isequal(output_backing, before_output)
        @test SJ.FFTW.assert_applicable(plan, input, output) === nothing
        LinearAlgebra.mul!(output, plan, input)
        for column in 1:2
            reference = fft_direct_reference(view(input, :, column), -1)
            @test maximum(abs.(Complex{BigFloat}.(output[:, column]) .- reference)) <=
                fft_rounding_bound(view(input, :, column))
        end
        fft_plan_allocations(output, plan, input)
        @test fft_plan_allocations(output, plan, input) == 0
    end
    x, y = zeros(ComplexF64, 8, 2), zeros(ComplexF64, 8, 2)
    key(axes = (1,), kind = :c2c_forward, threads = 1, flags = SJ.FFTW.PRESERVE_INPUT) =
        SJ.FFTW.@exclusive SJ._fft_key(x, y, axes, kind, threads, flags)
    baseline = key()
    @test key((2,)) != baseline
    @test key((1, 2)) != baseline
    @test key((1,), :c2c_backward) != baseline
    @test key((1,), :c2c_forward, 2) != baseline
    @test key((1,), :c2c_forward, 1, UInt32(0)) != baseline
    @test SJ.FFTW.@exclusive(SJ._fft_key(x, x, (1,), :c2c_forward, 1, UInt32(0))) !=
        baseline
    shifted_backing = zeros(ComplexF64, 9, 2)
    shifted = view(shifted_backing, 2:9, :)
    shifted_key = SJ.FFTW.@exclusive SJ._fft_key(
        x, shifted, (1,), :c2c_forward, 1, SJ.FFTW.PRESERVE_INPUT,
    )
    @test shifted_key != baseline # output strides differ even if alignment does not
    # A one-real-element offset distinguishes alignment classes on SIMD providers;
    # the key must reflect the actual classes, even if a provider has only one.
    real_backing = zeros(Float64, 18)
    shifted_complex = reinterpret(ComplexF64, view(real_backing, 2:17))
    aligned_complex = reinterpret(ComplexF64, view(real_backing, 1:16))
    a, b = SJ.FFTW.@exclusive (
        SJ.FFTW.alignment_of(shifted_complex), SJ.FFTW.alignment_of(aligned_complex),
    )
    key_input = x[:, 1]
    first_key, second_key = SJ.FFTW.@exclusive (
        SJ._fft_key(key_input, shifted_complex, (1,), :c2c_forward, 1, UInt32(16)),
        SJ._fft_key(key_input, aligned_complex, (1,), :c2c_forward, 1, UInt32(16)),
    )
    @test (first_key != second_key) == (a != b)
end

@testset "transactional restoration on success, failure and concurrent preparations" begin
    original = fft_global_state()
    originals = SJ.FFTW.@exclusive (SJ._fft_export(Float64), SJ._fft_export(Float32))
    try
        SJ.FFTW.@exclusive begin
            SJ._fft_threads!(Float64, 3)
            SJ._fft_threads!(Float32, 4)
            SJ.FFTW.plan_fft(zeros(ComplexF64, 11); flags = SJ.FFTW.MEASURE)
            SJ.FFTW.plan_fft(zeros(ComplexF32, 13); flags = SJ.FFTW.MEASURE)
            SJ._fft_threads!(Float64, 1)
            SJ._fft_threads!(Float32, 1)
            SJ.FFTW.plan_fft(zeros(ComplexF64, 8); flags = SJ.FFTW.PATIENT)
            SJ.FFTW.plan_fft(zeros(ComplexF32, 8); flags = SJ.FFTW.PATIENT)
            SJ._fft_threads!(Float64, 3)
            SJ._fft_threads!(Float32, 4)
        end
        before = fft_global_state()
        for T in (Float32, Float64)
            # Exact storage, including signed zero and noncanonical NaN payloads.
            U = T === Float64 ? UInt64 : UInt32
            payload = T === Float64 ? U(0x7ff8000000000123) : U(0x7fc00123)
            x = fill(complex(reinterpret(T, payload), -zero(T)), 8)
            y = fill(Complex{T}(zero(T), -zero(T)), 8)
            bits_x, bits_y = copy(reinterpret(U, x)), copy(reinterpret(U, y))
            context = SJ.FFTPlanning()
            SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
            @test reinterpret(U, x) == bits_x
            @test reinterpret(U, y) == bits_y
            @test fft_global_state() == before
            failed = SJ.FFTPlanning()
            # Force a genuine C_NULL planner failure with WISDOM_ONLY and no seed.
            # This private transaction path is never a public policy option.
            @test_throws ErrorException SJ.FFTW.@exclusive SJ._fft_transaction!(
                failed, x, y, (1,), :c2c_forward, 1,
                SJ.FFTW.PRESERVE_INPUT | SJ.FFTW.WISDOM_ONLY,
            )
            @test reinterpret(U, x) == bits_x
            @test reinterpret(U, y) == bits_y
            @test isempty(failed.applied) && isempty(failed.measured)
            @test isempty(failed.reused) && isempty(failed.rejected)
            @test fft_global_state() == before
        end
        jobs = [
            Threads.@spawn begin
                context = SJ.FFTPlanning()
                x, y = ComplexF64.(1:n), zeros(ComplexF64, n)
                plan = SJ.plan_owned_fft!(context, x, y, (1,), :c2c_forward)
                LinearAlgebra.mul!(y, plan, x)
                (y, x, context)
            end for n in (7, 8)
        ]
        for job in jobs
            actual, input, context = fetch(job)
            expected, bound = fft_direct_reference(input, -1), fft_rounding_bound(input)
            @test maximum(abs.(Complex{BigFloat}.(actual) .- expected)) <= bound
            @test length(context.measured) == 1
        end
        @test fft_global_state() == before
    finally
        SJ.FFTW.@exclusive begin
            for (T, wisdom, threads) in (
                    (Float64, originals[1], original[3]),
                    (Float32, originals[2], original[4]),
                )
                SJ._fft_forget(T)
                @test SJ._fft_import(T, wisdom)
                SJ._fft_threads!(T, threads)
            end
        end
    end
    @test fft_global_state() == original
end

@testset "FFT preparation admission precedes writes and facts" begin
    @test_throws ArgumentError SJ.FFTPlanning(; effort = :exhaustive)
    @test_throws ArgumentError SJ.FFTPlanning(; provider = "mkl")
    @test_throws ArgumentError SJ.FFTPlanning(; offered = Dict("key" => 1))
    x, y = zeros(ComplexF64, 8, 2), fill(1.0 + 2.0im, 8, 2)
    context = SJ.FFTPlanning()
    for axes in ((), (0,), (3,), (1, 1), (1.0,))
        @test_throws ArgumentError SJ.plan_owned_fft!(context, x, y, axes, :c2c_forward)
    end
    for threads in (0, -1, Int(typemax(Cint)) + 1)
        @test_throws ArgumentError SJ.plan_owned_fft!(
            context, x, y, (1,), :c2c_forward; threads,
        )
    end
    for (input, output, kind, axes) in (
            (x, y, :unknown, (1,)), (x, zeros(ComplexF32, 8, 2), :c2c_forward, (1,)),
            (x, zeros(ComplexF64, 7, 2), :c2c_forward, (1,)),
            (x, zeros(ComplexF64, 16), :c2c_forward, (1,)),
            (zeros(ComplexF64, 0), zeros(ComplexF64, 0), :c2c_forward, (1,)),
            (zeros(Int, 8), zeros(ComplexF64, 5), :r2c, (1,)),
            (Any[0, 1], zeros(ComplexF64, 2), :c2c_forward, (1,)),
            (zeros(8, 2), zeros(ComplexF64, 4, 2), :r2c, (1,)),
            (zeros(8, 2), zeros(ComplexF64, 5, 2), :r2c, (1, 2)),
            (x, y, :r2c, (1,)), (zeros(8, 2), zeros(8, 2), :c2c_forward, (1,)),
            (view(x, [1, 1, 3, 4], :), zeros(ComplexF64, 4, 2), :c2c_forward, (1,)),
        )
        @test_throws ArgumentError SJ.plan_owned_fft!(context, input, output, axes, kind)
    end
    GC.@preserve x begin
        alias = unsafe_wrap(Array, pointer(x, 2), (7, 2); own = false)
        @test_throws ArgumentError SJ.plan_owned_fft!(
            context, view(x, 1:7, :), alias, (1,), :c2c_forward,
        )
    end
    @test all(iszero, x) && all(==(1 + 2im), y)
    @test isempty(context.applied) && isempty(context.measured)
end
