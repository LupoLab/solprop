# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import SolpropJulia as SP
import PythonCall: PyArray, pyimport, pyconvert

@testset "PythonCall numerical arrays preserve host storage-space admission" begin
    numpy = pyimport("numpy")
    source = numpy.arange(16; dtype = "float64")
    wrapped = PyArray(source)
    @test SP._transfer_storage_space(wrapped) == SP._HostStorageSpace()
    for values in (
            source, source.reshape(4, 4).T,
            numpy.flip(source), numpy.swapaxes(source.reshape(2, 2, 4), 0, 2),
        )
        array = PyArray(values)
        @test SP._transfer_storage_space(array) == SP._HostStorageSpace()
        @test SP._transfer_arrays_alias(wrapped, array)
        @test SP._transfer_arrays_alias(array, wrapped)
    end
    GC.@preserve wrapped begin
        address = Base.unsafe_convert(Ptr{Float64}, wrapped)
        interior = unsafe_wrap(Array, address + 3sizeof(Float64), (4,); own = false)
        @test SP._transfer_arrays_alias(interior, wrapped)
        @test SP._transfer_arrays_alias(wrapped, interior)
    end
    @test !SP._transfer_arrays_alias(zeros(16), wrapped)
    @test pyconvert(Vector{Float64}, source) == collect(0.0:15.0)
end
