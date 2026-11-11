# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

isdefined(@__MODULE__, :rk_dense_fixture) || include("runge_kutta_observation.jl")

function device_rk_dense_budget(initial, stages, solution, dense, step, position)
    u = BigFloat(eps(Float64)) / 2
    gamma = 512u / (1 - 512u)
    h = BigFloat(step)
    theta = BigFloat(position / step)
    rest = BigFloat(1.0 - position / step)
    norm1(value) = abs(real(value)) + abs(imag(value))
    return map(CartesianIndices(initial)) do index
        row, column = Tuple(index)
        slopes = Complex{BigFloat}.(stages[row, column, :])
        span = h * sum(abs.(BigFloat.(solution)) .* norm1.(slopes))
        first = h * norm1(slopes[1]) + span
        last = span + h * norm1(slopes[end]) + first
        extension = h * sum(abs.(BigFloat.(dense)) .* norm1.(slopes))
        envelope = norm1(Complex{BigFloat}(initial[index])) + abs(theta) * (
            span + abs(rest) * (first + abs(theta) * (last + abs(rest) * extension))
        )
        # Two independently rounded graphs each have fewer than 128 scalar
        # operations. The rest covers phase formation and the established finite
        # binary64 elementary-function hypothesis at these small arguments. The
        # largest tested real phase is1.2: its gain is bounded a priori by4.
        # The envelope includes absolute dense weights before cancellation.
        return 4gamma * envelope
    end
end

function device_rk_dense_within(actual, expected, budget)
    return all(eachindex(actual)) do index
        delta = Complex{BigFloat}(actual[index]) - Complex{BigFloat}(expected[index])
        abs(real(delta)) + abs(imag(delta)) <= budget[index]
    end
end

mutable struct DeviceRKDenseZeroSource{E}
    execution::E
    calls::Int
end
function (source::DeviceRKDenseZeroSource)(output, input, position)
    source.calls += 1
    SP.each!(source.execution, size(output)...) do row, column
        output[row, column] = zero(eltype(output))
    end
    return nothing
end

function device_rk_dense_session(execution)
    initial = ComplexF64[1.0 + 0.25im 0.5 - 0.125im; -0.25 + 0.5im 0.125 + 0.25im]
    operator = SP.DiagonalOperator(execution, fill(-0.25 + 0.5im, size(initial)))
    flow = SP.LinearFlow(execution, operator, SP.upload(execution, initial))
    source = DeviceRKDenseZeroSource(execution, 0)
    norm = SP.MarchNorm(execution, fill(true, 2), fill(true, 2), ones(2), 1.0e-6, 1.0e-12)
    finish, step = 1 / 16, 1 / 32
    schedule = SP.MarchSchedule(
        planes = collect(0:(step / 2):finish), breakpoints = [0.0, finish],
        hard_kinks = Float64[], stretch = 1.1, step_planes = 0,
        replay_steps = Float64[], replay_edges = Float64[],
    )
    control = SP.PIControl(
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.2, beta2 = -0.08,
        lag = 0, reject_free = false, estimator_order = 4, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = step, auto_init = false,
    )
    return SP.RungeKuttaSession(
        execution, initial, 0.0, step, rk_dense_tableau(), control, norm, schedule,
        source, flow; capacity = 8, dense_weights = rk_dense_weights(),
    )
end

function device_rk_dense_drive(session)
    trace = Tuple{SP.SessionEvent, Float64, Int, Int}[]
    fields = Matrix{ComplexF64}[]
    while !session.state.finished
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        push!(trace, (report.kind, report.z, session.state.attempts, session.source.calls))
        @test report.kind != SP.SESSION_FAILED
        if report.kind in (SP.SESSION_LAUNCH, SP.SESSION_PLANE, SP.SESSION_COMPLETED)
            field = SP._march_observed_field(session)
            host = Matrix{ComplexF64}(undef, size(field))
            SP.oncpu(session.execution) ? copyto!(host, field) :
                SP.copy_host!(host, field, session.execution)
            push!(fields, host)
        end
        SP.acknowledge!(session, report.generation)
    end
    return trace, fields
end
