# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Profile, SolpropJulia

function profile_window(engine, initial; windows=engine.config.free ? 3 : 100)
    Profile.clear()
    Profile.@profile for _ in 1:windows
        SolpropJulia.load!(engine,initial)
        for _ in 1:10
            SolpropJulia.trial!(engine,1e-5)
            SolpropJulia.commit!(engine,1e-5)
        end
    end
    buffer = IOBuffer()
    Profile.print(buffer; format=:flat, sortedby=:count, mincount=10, C=true)
    String(take!(buffer))
end
