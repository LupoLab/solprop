# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function _rk_observation_device_operator(operator::DiagonalOperator, execution)
    return _source_device_array(operator.rate, execution)
end
function _rk_observation_device_operator(operator::RepresentedOperator, execution)
    return _source_device_array(operator.cumulative, execution)
end
function _rk_observation_device_operator(operator::ExpansionOperator, execution)
    _source_device_array(operator.terms, execution)
    return _source_device_array(operator.cumulative_weights, execution)
end
function _rk_observation_device_operator(operator, execution)
    throw(ArgumentError("device RK dense observation requires a qualified modal operator"))
end

function _rk_observation_failure_storage(execution, state)
    oncpu(execution) && return zeros(Int, size(state, 2))
    copy_device!(state, state, execution)
    failures = DeviceReadback(execution, Int, size(state, 2))
    _source_failure_rows!(failures.device, state, execution)
    all(iszero, readback!(failures, execution)) ||
        throw(ArgumentError("RK dense observation needs a finite matching state matrix"))
    return failures
end

function _rk_observation_execution(work, execution)
    failures = work.failures
    if failures isa DeviceReadback
        failures.execution === execution ||
            throw(ArgumentError("RK dense workspace belongs to another execution"))
    else
        oncpu(execution) ||
            throw(ArgumentError("device RK dense work requires prepared readback"))
    end
    return nothing
end

function _rk_observation_input_types(
        initial, endpoint, stages, ::Type{T}, execution,
    ) where {T}
    if oncpu(execution)
        return initial isa StridedMatrix{Complex{T}} &&
            endpoint isa StridedMatrix{Complex{T}} && stages isa StridedArray{Complex{T}, 3}
    end
    return T === Float64 && initial isa AbstractMatrix{ComplexF64} &&
        endpoint isa AbstractMatrix{ComplexF64} && stages isa AbstractArray{ComplexF64, 3}
end

function _rk_observation_device_inputs(execution, arrays...)
    oncpu(execution) && return nothing
    for array in arrays
        copy_device!(array, array, execution)
    end
    return nothing
end

_rk_observation_failure_array(failures) = failures
_rk_observation_failure_array(failures::DeviceReadback) = failures.device
_rk_observation_failure_host(failures, execution) = failures
_rk_observation_failure_host(failures::DeviceReadback, execution) =
    readback!(failures, execution)
_rk_observation_failure_bytes(failures) = sizeof(failures)
_rk_observation_failure_bytes(failures::DeviceReadback) =
    sizeof(failures.device) + sizeof(failures.host)

_rk_observation_stages_finite(failures, stages, execution) = all(isfinite, stages)

function _rk_observation_stages_finite(failures::DeviceReadback, stages, execution)
    status = failures.device
    each_column!(execution, size(stages, 2)) do column
        failed = 0
        for stage in axes(stages, 3), row in axes(stages, 1)
            if !isfinite(stages[row, column, stage])
                failed = row
                break
            end
        end
        status[column] = failed
    end
    return all(iszero, readback!(failures, execution))
end
