# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function invoke_stop_leaf(flag, setter)
    GC.@preserve flag ccall(setter, Cvoid, (Ptr{UInt8},), pointer(flag.storage))
    return nothing
end

@testset "owned atomic stop flag" begin
    flag = SolpropJulia.StopFlag()
    other = SolpropJulia.StopFlag()
    setter = Ptr{Cvoid}(SolpropJulia.stop_setter_address())
    @test setter != C_NULL
    @test SolpropJulia.stop_address(flag) == UInt(pointer(flag.storage))
    @test !SolpropJulia.stop_requested(flag)
    @test !SolpropJulia.stop_requested(other)
    invoke_stop_leaf(flag, setter)
    @test SolpropJulia.stop_requested(flag)
    @test !SolpropJulia.stop_requested(other)
    GC.gc()
    invoke_stop_leaf(flag, setter)
    @test SolpropJulia.stop_requested(flag)
    @test (@allocated SolpropJulia.stop_requested(flag)) == 0
    @test (@allocated invoke_stop_leaf(flag, setter)) == 0
    @test_throws ArgumentError SolpropJulia.StopFlag(0)
    @test_throws ArgumentError SolpropJulia.StopFlag(-1)
    GC.@preserve other begin
        borrowed = SolpropJulia.StopFlag(SolpropJulia.stop_address(other))
        @test !SolpropJulia.stop_requested(borrowed)
        invoke_stop_leaf(other, setter)
        @test SolpropJulia.stop_requested(borrowed)
    end
end
