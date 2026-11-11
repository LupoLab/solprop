# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# A device export prepares the complete physical recipe in session-owned scratch.
# Failed numerical work or a partial download cannot change the caller's destination.
struct _DeviceFieldExport{F, S}
    field::F
    status::S
end

function _device_field_export(execution, shape)
    oncpu(execution) && return nothing
    shape isa Tuple{Int, Int} && all(>(0), shape) ||
        throw(ArgumentError("device field export requires its positive state shape"))
    return _DeviceFieldExport(
        DeviceReadback(execution, ComplexF64, shape...),
        DeviceReadback(execution, Int, 2, shape[2]),
    )
end

function _device_export_work(output, session, field)
    eltype(field) === ComplexF64 ||
        throw(ArgumentError("device field export currently requires binary64 state"))
    _transfer_storage_space(output) isa _HostStorageSpace ||
        throw(ArgumentError("device field exports require a host destination"))
    storage = session.exports
    storage !== nothing && storage.transfer isa _DeviceFieldExport ||
        throw(ArgumentError("device field export requires prepared transfer storage"))
    work = storage.transfer
    execution = session.execution
    work.field.execution === execution && work.status.execution === execution ||
        throw(ArgumentError("device field export belongs to another execution"))
    size(work.field.device) == size(work.field.host) == size(field) &&
        eltype(work.field.device) === eltype(work.field.host) === ComplexF64 ||
        throw(DimensionMismatch("device field export scratch differs from its state"))
    _weight_summary_admit(work.status, work.field.device, execution)
    # A no-op copy validates the live selected context before numerical writes.
    copy_device!(field, field, execution)
    copy_device!(work.field.device, work.field.device, execution)
    copy_device!(work.status.device, work.status.device, execution)
    return work
end

function _device_export!(output, session, field, weight, power, optics)
    work = _device_export_work(output, session, field)
    execution = session.execution
    if power
        weight isa AbstractMatrix{Float64} ||
            throw(ArgumentError("device export power weights require binary64 values"))
        _retained_alias(weight, work) &&
            throw(ArgumentError("power weights must be disjoint from export scratch"))
        copy_device!(weight, weight, execution)
    end
    # Unit weights skip the product globally, preserving the existing signed-zero
    # recipe. A per-cell unit shortcut would change mixed-weight export semantics.
    scale = power && !_weight_summary!(work.status, weight, execution)
    staged = work.field.device
    if scale
        _device_export_scaled!(staged, field, weight, optics, execution)
    else
        _device_export_raw!(staged, field, optics, execution)
    end
    parts = work.status.device
    rows, columns = size(field)
    each_column!(execution, columns) do column
        invalid = 0
        for row in 1:rows
            if !isfinite(staged[row, column])
                invalid = row
                break
            end
        end
        parts[1, column] = invalid
    end
    facts = readback!(work.status, execution)
    for column in 1:columns
        iszero(facts[1, column]) ||
            throw(ArgumentError("physical field export exceeds binary64 range"))
    end
    host = readback!(work.field, execution)
    copyto!(output, host)
    return nothing
end

function _device_export_raw!(staged, field, optics, execution)
    arithmetic = SourceArithmetic(Float64)
    each!(execution, size(field)...) do row, column
        staged[row, column] = _export_field_value(
            field[row, column], arithmetic, 1.0, row, false, optics,
        )
    end
    return nothing
end

function _device_export_scaled!(staged, field, weight, optics, execution)
    arithmetic = SourceArithmetic(Float64)
    each!(execution, size(field)...) do row, column
        staged[row, column] = _export_field_value(
            field[row, column], arithmetic, weight[row, column], row, true, optics,
        )
    end
    return nothing
end
