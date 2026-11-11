# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# CPU reporting retains its original node-major scan. Device reporting performs
# the same within-node scan in parallel and transfers only two integers per node.
function _source_plasma_failure!(
        ::Nothing, status::AbstractMatrix{UInt8}, execution::KernelExecution,
    )
    oncpu(execution) ||
        throw(ArgumentError("device plasma status requires prepared compact readback"))
    return plasma_failure(status)
end

function _source_plasma_failure!(
        work::DeviceReadback, status::AbstractMatrix{UInt8}, execution::KernelExecution,
    )
    work.execution === execution ||
        throw(ArgumentError("plasma status scratch belongs to another execution"))
    rows, columns = size(status)
    size(work.device) == size(work.host) == (2, columns) ||
        throw(DimensionMismatch("plasma status readback needs two integers per node"))
    eltype(work.device) === eltype(work.host) === Int ||
        throw(ArgumentError("plasma status readback requires integer storage"))
    Base.require_one_based_indexing(status)
    _transfer_admit_destinations((work.device,))
    (
        _transfer_arrays_alias(work.device, status) ||
            _transfer_arrays_alias(work.host, status)
    ) &&
        throw(ArgumentError("plasma status readback must not overlap its input"))
    oncpu(execution) || copy_device!(status, status, execution)
    compact = work.device
    each_column!(execution, columns) do column
        failed_sample, failed_code = 0, 0
        for row in 1:rows
            code = status[row, column]
            if !iszero(code)
                failed_sample, failed_code = row, Int(code)
                break
            end
        end
        compact[1, column] = failed_sample
        compact[2, column] = failed_code
    end
    host = readback!(work, execution)
    for column in 1:columns
        iszero(host[2, column]) || return PlasmaFailure(
            PlasmaCode(host[2, column]), column, host[1, column]
        )
    end
    return PlasmaFailure(PlasmaSuccess, 0, 0)
end
