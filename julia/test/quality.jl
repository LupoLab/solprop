# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import Aqua

# JET loads Revise. Polling avoids allocating OS directory watchers for the whole
# dependency graph in this noninteractive process; all error analysis still runs.
ENV["JULIA_REVISE_POLL"] = "1"
import JET

@testset "Blocking package quality" begin
    @test JET.JET_AVAILABLE
    Aqua.test_all(SolpropJulia)
    JET.test_package(SolpropJulia)
end
