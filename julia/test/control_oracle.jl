# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import TOML
using Test: @test, @testset
import SolpropJulia

function canonical_control(value)
    value isa Bool && return value ? "true" : "false"
    isnan(value) && return "nan"
    return string(reinterpret(UInt64, Float64(value)); base = 16, pad = 16)
end

@testset "Identical recorded Python readings" begin
    for case in TOML.parsefile(only(ARGS))["cases"]
        policy = SolpropJulia.PIControl(;
            (Symbol(key) => value for (key, value) in case["policy"])...
        )
        history = SolpropJulia.NormHistory(policy)
        for value in case["history"]
            SolpropJulia.record_norm!(history, value)
        end
        arguments = case["arguments"]
        expected_error = case["error"]
        retained, count = SolpropJulia.control_history(history), history.count
        try
            value = if case["kind"] == "decide"
                SolpropJulia.control_decide(policy, arguments...)
            elseif case["kind"] == "start"
                SolpropJulia.control_start(policy, arguments...)
            elseif case["kind"] == "reject"
                SolpropJulia.control_reject(policy, arguments...)
            else
                SolpropJulia.control_propose(policy, first(arguments), history)
            end
            @test isempty(expected_error)
            results = value isa Tuple ? value : (value,)
            @test canonical_control.(collect(results)) == case["expected"]
        catch error
            @test string(nameof(typeof(error))) == expected_error
            @test isequal(SolpropJulia.control_history(history), retained)
            @test history.count == count
        end
    end
end
