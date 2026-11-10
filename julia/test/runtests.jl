# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws

group = get(ENV, "GROUP", "Core")
group in ("Core", "Quality") || throw(ArgumentError("unknown Julia test GROUP: $group"))

if group == "Quality"
    # JET must track definitions as the package loads. Importing it afterwards can
    # leave the direct runner with no signatures even though CI analyses them.
    get!(ENV, "JULIA_REVISE_POLL", "1")
    import JET
end
import SolpropJulia

if group == "Core"
    include("cancellation.jl")
    include("execution.jl")
    include("phi_functions.jl")
    include("transforms.jl")
    include("tables.jl")
    include("power_weights.jl")
    include("operators.jl")
    include("runge_kutta.jl")
    include("adams.jl")
    include("low_storage.jl")
    include("control.jl")
    include("adams_control.jl")
    include("responses.jl")
    include("raman.jl")
    include("plasma.jl")
    include("statistics.jl")
    include("monitor.jl")
    include("source.jl")
    include("composition.jl")
    include("observations.jl")
    include("march.jl")
    include("transfer.jl")
    include("preparation.jl")
    include("environments.jl")
    include("inventory.jl")
else
    include("quality.jl")
end
