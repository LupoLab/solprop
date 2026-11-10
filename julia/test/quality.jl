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
    analysis_log = IOBuffer()
    JET.test_package(SolpropJulia; toplevel_logger = analysis_log)
    transcript = String(take!(analysis_log))
    print(transcript)
    # The pinned JET version exposes progress through this supported logger, not
    # structured signature counts. Fail closed if its summary changes or definition
    # tracking silently analyses nothing (for example, from the wrong import order).
    summary = match(
        Regex(
            raw"Analyzed all top-level definitions \(all: (\d+) " *
                raw"\| analyzed: (\d+) \| cached: (\d+)",
        ),
        transcript,
    )
    @test summary !== nothing
    if summary !== nothing
        total, analysed, cached = parse.(Int, summary.captures)
        @test total > 0
        @test analysed + cached > 0
    end
end
