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
    include("double_float.jl")
    include("double_float_functions.jl")
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
    include("monitor_restore.jl")
    include("source.jl")
    include("composition.jl")
    include("observations.jl")
    include("quadrature_alarm.jl")
    include("march.jl")
    include("group_boundaries.jl")
    include("march_services.jl")
    include("adams_session.jl")
    include("adams_observation.jl")
    include("adams_transfer.jl")
    include("observation_session.jl")
    include("adams_dense_session.jl")
    include("adams_dense_transfer.jl")
    include("observation_transfer.jl")
    include("monitor_transfer.jl")
    include("transfer_aliases.jl")
    include("input_noise.jl")
    include("input_noise_transfer.jl")
    include("low_storage_session.jl")
    include("group_session.jl")
    include("transfer.jl")
    include("preparation.jl")
    include("environments.jl")
    include("inventory.jl")
else
    include("quality.jl")
end
