# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

group = get(ENV, "GROUP", "Core")
group in ("Core", "Quality") || throw(ArgumentError("unknown Julia test GROUP: $group"))

if group == "Core"
    include("preparation.jl")
    include("environments.jl")
    include("inventory.jl")
else
    include("quality.jl")
end
