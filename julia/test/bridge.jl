# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import PythonCall: pyimport, pyeval, pyexec
import SolpropJulia

@testset "Python preparation copies retain no borrowed storage" begin
    @test Base.get_extension(SolpropJulia, :SolpropJuliaPythonCallExt) !== nothing
    numpy = pyimport("numpy")
    cases = (("float32", Float32), ("float64", Float64), ("complex128", ComplexF64))
    for (dtype, T) in cases
        source = numpy.arange(24; dtype).reshape(2, 3, 4)
        expected = [T((i - 1) * 12 + (j - 1) * 4 + k - 1) for i in 1:2, j in 1:3, k in 1:4]
        owned = SolpropJulia.copy_preparation(source)
        @test owned isa Array{T, 3}
        @test owned == expected
        strided = pyeval("array[::-1, :, ::2]", Main, (array = source,))
        @test SolpropJulia.copy_preparation(strided) == expected[2:-1:1, :, 1:2:4]
        pyexec("array[:] = 0", Main, (array = source,))
        @test owned == expected
        owned[1] = T(99)
        @test pyeval(Int, "array[0, 0, 0].real", Main, (array = source,)) == 0
    end
    objects = numpy.array(["not numerical"]; dtype = "object")
    @test_throws ArgumentError SolpropJulia.copy_preparation(objects)
end
