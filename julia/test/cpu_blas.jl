# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import LinearAlgebra
import Libdl
import SolpropJulia as SJ

function cpu_blas_product_reference(left, right)
    return setprecision(256) do
        [
            sum(
                Complex{BigFloat}(left[row, inner]) *
                    Complex{BigFloat}(right[inner, column])
                    for inner in axes(left, 2)
            ) for row in axes(left, 1), column in axes(right, 2)
        ]
    end
end

function cpu_blas_product_bound(left, right)
    # A complex multiply-add costs at most eight real operations per contribution;
    # real products need fewer. Inputs and every intermediate are finite normal or
    # exact zero. gamma_(8n+4) bounds any reduction order and final scaling here.
    # The independent 256-bit sum contributes less than 2^-200 of the same scale.
    return setprecision(256) do
        T = typeof(real(zero(eltype(left))))
        count, u = 8size(left, 2) + 4, BigFloat(eps(T)) / 2
        gamma = count * u / (1 - count * u) + big(2.0)^(-200)
        [
            gamma * sum(
                abs(Complex{BigFloat}(left[row, inner])) *
                    abs(Complex{BigFloat}(right[inner, column]))
                    for inner in axes(left, 2)
            ) for row in axes(left, 1), column in axes(right, 2)
        ]
    end
end

function cpu_blas_gemm!(output, left, right)
    T = eltype(output)
    LinearAlgebra.BLAS.gemm!('N', 'N', one(T), left, right, zero(T), output)
    return nothing
end

function cpu_blas_scoped(f, info)
    if info.provider == :accelerate
        api, reason = SJ._cpu_accelerate_api(info.library)
        isempty(reason) || error(reason)
        original = SJ._cpu_accelerate_mode(api)
        try
            SJ._cpu_accelerate_mode!(api, Cuint(1))
            return f()
        finally
            SJ._cpu_accelerate_mode!(api, original)
            Libdl.dlclose(api.handle)
        end
    end
    original = LinearAlgebra.BLAS.get_num_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1)
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(original)
    end
end

@testset "CPU provider capability checks precede forwarding" begin
    before = SJ._cpu_blas_owner("dgemm_", SJ._cpu_blas_interface())
    api, reason = SJ._cpu_accelerate_api(joinpath(tempdir(), "absent-accelerate-library"))
    @test api === nothing && reason == "accelerate-library-unavailable"
    api, reason = SJ._cpu_accelerate_api(before)
    @test api === nothing && reason == "accelerate-new-lapack-unavailable"
    @test SJ._cpu_blas_owner("dgemm_", SJ._cpu_blas_interface()) == before
end

cpu_blas_fake_mode()::Cuint = Cuint(0)
cpu_blas_fake_refusal(::Cuint)::Cint = Cint(-1)
cpu_blas_fake_single_refusal(mode::Cuint)::Cint = mode == 1 ? Cint(-1) : Cint(0)

@testset "Provider failure preserves the primary and restoration failures" begin
    fallback = SJ._cpu_blas_openblas("test")
    getter = @cfunction(cpu_blas_fake_mode, Cuint, ())
    both = @cfunction(cpu_blas_fake_refusal, Cint, (Cuint,))
    first = @cfunction(cpu_blas_fake_single_refusal, Cint, (Cuint,))
    api = SJ._AccelerateAPI(C_NULL, getter, both)
    failure = try
        SJ._cpu_blas_forward(api, fallback)
        nothing
    catch caught
        caught
    end
    @test failure isa CompositeException
    @test length(failure.exceptions) == 2
    @test all(error -> error isa ErrorException, failure.exceptions)
    @test all(
        error -> occursin("refused its calling-thread mode", sprint(showerror, error)),
        failure.exceptions,
    )
    api = SJ._AccelerateAPI(C_NULL, getter, first)
    @test_throws ErrorException SJ._cpu_blas_forward(api, fallback)
    @test SJ._cpu_blas_owner("dgemm_", fallback.interface) == fallback.library
end

@testset "Configured CPU provider owns the actual BLAS/LAPACK ABI" begin
    api, _ = Sys.isapple() ? SJ._cpu_accelerate_api(SJ._CPU_ACCELERATE) : (nothing, "")
    before = api === nothing ? nothing : SJ._cpu_accelerate_mode(api)
    fallback = SJ._cpu_blas_owner("dgemmt_", SJ._cpu_blas_interface())
    info = SJ.configure_cpu_blas!()
    @test info === SJ.configure_cpu_blas!()
    @test info.interface == (sizeof(LinearAlgebra.BlasInt) == 8 ? :ilp64 : :lp64)
    @test !isempty(info.library)
    for symbol in SJ._CPU_BLAS_SYMBOLS
        @test SJ._cpu_blas_owner(symbol, info.interface) == info.library
    end
    if api === nothing
        @test info.provider == :openblas
        @test info.thread_scope == :process && isempty(info.fallback_library)
        @test !isempty(info.reason)
    else
        try
            @test info.provider == :accelerate && info.thread_scope == :calling_thread
            @test isempty(info.reason) && !isempty(info.fallback_library)
            @test SJ._cpu_accelerate_mode(api) == before
            for interface in (:lp64, :ilp64), symbol in SJ._CPU_BLAS_SYMBOLS
                @test SJ._cpu_blas_owner(symbol, interface) == info.library
            end
            # Accelerate does not supply the extension GEMMT in the qualified
            # platform. clear=false must retain its previous OpenBLAS forward.
            if Libdl.dlsym_e(api.handle, "dgemmt\$NEWLAPACK\$ILP64") == C_NULL
                @test SJ._cpu_blas_owner("dgemmt_", info.interface) == fallback
            end
        finally
            Libdl.dlclose(api.handle)
        end
    end
    changed = SJ.CPUBlasInfo(
        info.provider, "not-the-loaded-provider", info.interface,
        info.fallback_library, info.thread_scope, info.reason,
    )
    @test_throws ErrorException SJ._cpu_blas_validate(changed)
    if Threads.nthreads(:default) > 1
        outcomes = fill(false, Threads.maxthreadid())
        Threads.@threads :static for _ in 1:Threads.nthreads(:default)
            if Threads.threadid() != 1
                outcomes[Threads.threadid()] = try
                    SJ.configure_cpu_blas!()
                    false
                catch failure
                    failure isa ArgumentError
                end
            end
        end
        @test any(outcomes)
    end
end

@testset "Actual selected BLAS products and missing-symbol fallback" begin
    info = SJ.configure_cpu_blas!()
    cpu_blas_scoped(info) do
        for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 7, 9)
            left = [
                T <: Real ? T((3row - 2inner) / 32) :
                    T(complex((3row - 2inner) / 32, (row + inner) / 64))
                    for row in 1:5, inner in 1:n
            ]
            right = [
                T <: Real ? T((inner - 3column) / 16) :
                    T(complex((inner - 3column) / 16, (2inner - column) / 32))
                    for inner in 1:n, column in 1:4
            ]
            output = zeros(T, 5, 4)
            cpu_blas_gemm!(output, left, right)
            reference = cpu_blas_product_reference(left, right)
            bound = cpu_blas_product_bound(left, right)
            @test all(>(0), bound)
            @test all(abs.(Complex{BigFloat}.(output) .- reference) .<= bound)
            @test all(isfinite, output)
            @test (@allocated cpu_blas_gemm!(output, left, right)) == 0
        end
        left = [Float64(3row - inner) / 16 for row in 1:7, inner in 1:5]
        right = [Float64(inner - 2column) / 32 for inner in 1:5, column in 1:7]
        output = fill(-3.0, 7, 7)
        LinearAlgebra.BLAS.gemmt!('U', 'N', 'N', 1.0, left, right, 0.0, output)
        reference = cpu_blas_product_reference(left, right)
        bound = cpu_blas_product_bound(left, right)
        for column in 1:7, row in 1:7
            if row <= column
                @test abs(BigFloat(output[row, column]) - reference[row, column]) <=
                    bound[row, column]
            else
                @test output[row, column] == -3.0
            end
        end
    end
end

@testset "Selected LAPACK solves the known conditioned rank-one system" begin
    info = SJ.configure_cpu_blas!()
    cpu_blas_scoped(info) do
        for T in (Float64, ComplexF64)
            n = 7
            matrix = fill(T(1 / 16), n, n)
            for index in 1:n
                matrix[index, index] += one(T)
            end
            expected = T <: Real ? T.((1:n) ./ 8) :
                [T(complex(index / 8, (n - index) / 16)) for index in 1:n]
            # All products/sums defining b are dyadic and exactly representable.
            rhs = [sum(matrix[row, col] * expected[col] for col in 1:n) for row in 1:n]
            solution = copy(rhs)
            LinearAlgebra.LAPACK.gesv!(copy(matrix), solution)
            # A=I+11'/16 has inverse I-11'/(16+n), infinity condition <2, and
            # no-pivot LU factors with infinity norms <2. Standard LU/triangular
            # substitution bounds cost O(n) per row; gamma_(128n²) also covers
            # complex scalar operations and their propagation through both factors.
            # A factor8 covers both factor norms and ||inv(A)||, before comparison.
            u = eps(Float64) / 2
            count = 128n^2
            bound = 8count * u / (1 - count * u) * maximum(abs, expected)
            @test bound > 0
            @test maximum(abs.(solution .- expected)) <= bound
            @test all(isfinite, solution)
        end
    end
end
