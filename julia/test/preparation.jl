# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Preparation owns its numerical arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64, Int64)
        source = reshape(T.(1:24), 2, 3, 4)
        owned = SolpropJulia.copy_preparation(source)
        @test owned isa Array{T, 3}
        @test size(owned) == size(source)
        @test owned == source
        @test owned !== source
        source[1] = T(99)
        @test owned[1] == T(1)
        owned[2] = T(100)
        @test source[2] == T(2)

        strided = view(source, 2:-1:1, 1:2:3, :)
        copy = SolpropJulia.copy_preparation(strided)
        @test copy == strided
        @test copy isa Array{T, 3}
        @test !Base.mightalias(copy, source)
    end

    empty = zeros(Float64, 0, 2)
    @test size(SolpropJulia.copy_preparation(empty)) == (0, 2)
    scalar = fill(1.0)
    @test SolpropJulia.copy_preparation(scalar)[] == 1.0

    # Preparation transport preserves bits, including signed zeros and NaN payloads.
    bits = UInt64[0x8000000000000000, 0x7ff800000000002a, 0x7ff0000000000000]
    values = reinterpret(Float64, bits)
    @test reinterpret(UInt64, SolpropJulia.copy_preparation(values)) == bits
    @test_throws ArgumentError SolpropJulia.copy_preparation(["not numerical"])
    @test_throws ArgumentError SolpropJulia.copy_preparation(BigFloat[1])
end

@testset "The standalone engine does not load its weak dependencies" begin
    for name in ("PythonCall", "CUDA", "AMDGPU", "Metal")
        @test all(package.name != name for package in keys(Base.loaded_modules))
    end
end
