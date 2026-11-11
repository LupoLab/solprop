# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function copy_session_monitor_rows!(rows, session, generation::Int)
    borrowed = _python_buffer(Float64, Val(2), rows)
    count, columns = size(borrowed)
    if !isempty(borrowed)
        (columns == 1 || stride(borrowed, 2) == 1) &&
            (count == 1 || stride(borrowed, 1) == columns) ||
            throw(ArgumentError("Python temporal monitor rows must be C-contiguous"))
    end
    GC.@preserve rows borrowed begin
        address = Base.unsafe_convert(Ptr{Float64}, borrowed)
        canonical = unsafe_wrap(Array, address, (columns, count); own = false)
        SolpropJulia.session_monitor_rows!(canonical, session, generation)
    end
    return nothing
end

function restore_session_monitor!(
        session, continuation::Py, plane::Bool, transverse = nothing,
    )
    record = if pyis(continuation, nothing)
        nothing
    else
        SolpropJulia.TemporalContinuation(
            pyconvert(Float64, continuation.launch),
            pyconvert(NTuple{3, Float64}, continuation.launch_occupancy),
            pyconvert(NTuple{2, Float64}, continuation.guard_max),
            pyconvert(NTuple{2, Float64}, continuation.guard_at),
            pyconvert(Float64, continuation.collar_max),
            pyconvert(Float64, continuation.raw_removed),
            pyconvert(Float64, continuation.floor_removed),
            pyconvert(Int, continuation.quiet_run),
            pyconvert(Bool, continuation.quiet),
            pyconvert(Float64, continuation.raw_guard_max),
            pyconvert(Int, continuation.steps),
            pyconvert(Int, continuation.skipped),
            pyconvert(Int, continuation.left_out_checks),
            pyconvert(Float64, continuation.last_plane_z),
            pyconvert(Bool, continuation.has_quiet_run),
        )
    end
    return SolpropJulia.restore_session_monitor!(
        session, record, plane, _transverse_continuation(transverse),
    )
end

function restore_session_monitor!(
        session, ::Nothing, plane::Bool, transverse::Py,
    )
    return SolpropJulia.restore_session_monitor!(
        session, nothing, plane, _transverse_continuation(transverse),
    )
end
