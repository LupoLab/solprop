# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# These small, synthetic translated records exercise compilation, not a physical
# material model or an accuracy reference. They are local to each invocation. In
# particular, no FFT plan, source workspace, stop byte or session survives in a global.
function _precompile_cycle_rule()
    nodes, weights = Float64[], Float64[]
    ends = ntuple(4) do level
        count = 2^(level + 1)
        # The symmetric Jacobi matrix of the orthonormal Legendre recurrence gives
        # Gauss nodes; twice the first eigenvector component squared gives weights.
        # Golub & Welsch (1969), (2.2)/(2.6), doi:10.1090/S0025-5718-69-99647-1;
        # for Legendre, the recurrence gives beta_n=n/sqrt(4n^2-1), mu_0=2.
        off_diagonal = [index / sqrt(4index^2 - 1) for index in 1:(count - 1)]
        rule = LinearAlgebra.eigen(
            LinearAlgebra.SymTridiagonal(zeros(count), off_diagonal)
        )
        append!(nodes, rule.values)
        append!(weights, 2 .* rule.vectors[1, :] .^ 2)
        length(nodes)
    end
    return CycleQuadrature(
        nodes, weights, ends; moment_rtol = 1.0e-5, negligible_rate = 1.0e-30,
    )
end

function _precompile_material_case(execution::KernelExecution, ::Val{K}) where {K}
    kind = K
    kind in (:field, :envelope) || throw(ArgumentError("unknown precompile formulation"))
    n_time = 8
    n_frequency = kind == :field ? n_time ÷ 2 + 1 : n_time
    active = fill(true, n_frequency)
    active[1] = active[end] = false
    omega = collect(1.0:n_frequency)
    grid = ModalSourceGrid(
        execution; kind, n_time, dt = 0.125,
        crop_index = collect(0:(n_frequency - 1)), omega, active,
        omega_window = ones(n_frequency), source_window = ones(n_time),
        source_window_derivative = zeros(n_time),
    )
    mapping = ModalSourceMap(
        execution; synthesis = reshape([1.0, 0.75], 2, 1),
        projection = reshape([0.5, 0.25], 1, 2),
    )
    density = SourceProfile(
        ChebyshevTable(execution, [0.0, 0.01], reshape([1.0, 0.01], 1, 2, 1)),
    )
    kerr = ModalKerr(
        KerrCoefficients(0.01, :analytic; take_real = kind == :field), density,
    )
    undamped = zeros(2n_time, 1)
    for row in 1:n_time
        undamped[row, 1] = 0.01sin(0.25(row - 1))
    end
    raman = ModalRaman(
        RamanKernel(undamped, grid.dt), (SourceProfile(0.1),), SourceProfile(1.0);
        fraction = 0.2, material_factor = 0.1, thg = false,
    )
    lower, upper, scale = 1.0e-8, 10.0, 0.001
    rate = IonisationRateTable(
        reshape([0.0, 0.0, 2.0, log(scale) + 2log(lower)], 4, 1),
        lower, upper, scale * upper^2, log(lower), log(upper / lower),
    )
    channel = ModalPlasmaChannel(
        PlasmaChannel(rate, 1.0, 0.01), SourceProfile(0.1), SourceProfile(0.01);
        quadrature = kind == :field ? nothing : _precompile_cycle_rule(),
    )
    plasma = if kind == :field
        ModalPlasma(execution, (channel,), RunningIntegralRule([1.0], grid.dt))
    else
        ModalPlasma(
            execution, (channel,);
            drude = [-active[row] / omega[row]^2 for row in 1:n_frequency],
            loss = ComplexF64[-im * active[row] / omega[row] for row in 1:n_frequency],
        )
    end
    prefactor = reshape(ComplexF64[-0.001im * mask for mask in active], n_frequency, 1)
    data = ModalSourceData(
        grid, mapping, SourcePrefactor(execution, prefactor);
        kerr, raman = (raman,), plasma,
    )
    initial = zeros(ComplexF64, n_frequency, 1)
    initial[2] = 0.2 + 0.1im
    return data, initial
end

function _precompile_services(
        execution::KernelExecution, data::ModalSourceData{<:ModalSourceGrid},
    )
    source_grid = data.grid
    n_time, n_frequency = source_grid.n_time, length(source_grid.omega)
    grid = StatisticsGrid(
        copy(source_grid.omega), collect(0:(n_time - 1)) .* source_grid.dt,
        fill(source_grid.dt / n_time, n_frequency),
    )
    selection = StatisticsSelection(
        (StatisticSlot(:energy, 1, 1), StatisticSlot(:peak_power_total, 2, 1)), (1, 2),
    )
    regions = zeros(n_time, 4)
    regions[1, 1] = regions[end, 2] = 1.0
    regions[1, 3] = regions[end, 3] = 1.0
    monitor = TemporalObservationData(
        TemporalMonitorPolicy(;
            warn = 2.0, stop = 3.0, activity = 0.1, quiet_share = 0.0,
            quiet_checks = 2, quiet_collar = false, cadence = 1, interval = 0.01,
            launch_floor_ratio = 1.0e-8, floor_margin = 1.0, split = false,
        ),
        TemporalMonitorWeights(regions, [1.0]),
    )
    preparation = ObservationPreparation(
        execution, grid, selection; kind = source_grid.kind, n_modes = 1,
        power_weight = ObservationWeight(execution, ones(n_frequency, 1)),
        temporal = monitor,
    )
    return ObservationServices(
        execution, SessionObservationPolicy(preparation); attempt_capacity = 4,
        node_chunk = 2,
    )
end

function _precompile_rk_tableau()
    # The existing translated DOPRI5 pair, Dormand & Prince (1980), Table 2,
    # doi:10.1016/0771-050X(80)90013-3. This local workload data is not a default
    # selected by the engine. Seven stages compile the production tableau shape.
    nodes = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
    matrix = [
        0 0 0 0 0 0 0
        1 / 5 0 0 0 0 0 0
        3 / 40 9 / 40 0 0 0 0 0
        44 / 45 -56 / 15 32 / 9 0 0 0 0
        19372 / 6561 -25360 / 2187 64448 / 6561 -212 / 729 0 0 0
        9017 / 3168 -355 / 33 46732 / 5247 49 / 176 -5103 / 18656 0 0
        35 / 384 0 500 / 1113 125 / 192 -2187 / 6784 11 / 84 0
    ]
    solution = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
    embedded = [
        5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40,
    ]
    return EmbeddedTableau(nodes, matrix, solution, solution .- embedded)
end

function _precompile_adams_policy()
    return AdamsControlPolicy(;
        order = 4, variable_order = false, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = true, interval_cap = false, dz_min = 1.0e-10, dz_max = 0.001,
        stretch = 1.0, z_end = 0.004, save_planes = [0.002, 0.004],
        hard_kinks = Float64[], breakpoints = [0.0, 0.01], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
end

function _precompile_session(
        execution::KernelExecution, ::Val{M}, ::Val{K},
    ) where {M, K}
    method, kind = M, K
    method in (:rk, :adams, :low_storage) ||
        throw(ArgumentError("unknown precompile session method"))
    data, initial = _precompile_material_case(execution, Val(kind))
    source = ModalSource(execution, data; node_chunk = 2)
    services = _precompile_services(execution, data)
    operator = DiagonalOperator(execution, fill(-0.01 + 0.02im, size(initial)))
    flow = LinearFlow(execution, operator, initial)
    norm = MarchNorm(
        execution, data.grid.active, data.grid.active, ones(size(initial, 1)), 1.0e-3, 0.0,
    )
    replay = method != :rk
    schedule = MarchSchedule(;
        planes = [0.002, 0.004], breakpoints = [0.0, 0.01], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = replay ? fill(0.001, 4) : Float64[],
        replay_edges = replay ? [0.0, 0.001, 0.002, 0.003, 0.004] : Float64[],
    )
    if method == :rk
        control = PIControl(;
            safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.14, beta2 = -0.08,
            lag = 0, reject_free = false, estimator_order = 4, start_growth = 2.0,
            dz_min = 1.0e-10, dz_max = 0.001, auto_init = false,
        )
        return RungeKuttaSession(
            execution, initial, 0.0, 0.001, _precompile_rk_tableau(), control,
            norm, schedule, source, flow; capacity = 4, services,
        )
    elseif method == :adams
        return AdamsSession(
            execution, initial, 0.0, 0.001, _precompile_adams_policy(), norm,
            schedule, source, flow; capacity = 4, services,
        )
    end
    return LowStorageSession(
        execution, initial, 0.0, RK4IPTableau(), norm, schedule, source, flow;
        capacity = 4, services,
    )
end

function _precompile_drain!(session)
    rows, planes = 0, 0
    return try
        # Four short trials, launch, two planes and completion fit well inside this
        # explicit event bound. A broken lifecycle must fail cache creation, not spin.
        for _ in 1:16
            report = advance!(session; work_limit = 4, deadline = 60.0)
            report.kind in (
                SESSION_LAUNCH, SESSION_PLANE, SESSION_PROGRESS, SESSION_COMPLETED,
            ) || error("unexpected precompile session event $(report.kind)")
            rows += observation_rows(session.services).count
            observation = observation_planes(session.services)
            if observation.valid
                all(isfinite, observation.values) ||
                    error("nonfinite precompile observation")
                planes += 1
            end
            acknowledge!(session, report.generation)
            if report.kind == SESSION_COMPLETED
                all(isfinite, session.field) || error("nonfinite precompile field")
                return (
                    accepted = session.state.accepted,
                    calls = session.source.calls,
                    rows = rows, planes = planes,
                )
            end
        end
        error("precompile session exceeded its event bound")
    finally
        close_session!(session)
    end
end

function _precompile_cpu_workload()
    # Compile the actual-buffer planner used by managed preparations, without
    # measuring or publishing wisdom. All context and plan storage remains local.
    planning = FFTPlanning(; effort = :estimate, keep = false)
    execution = KernelExecution(KernelAbstractions.CPU(), 1; planning)
    spatial = zeros(ComplexF64, 2, 4, 4)
    spatial_plan = SpatialFFT(execution, spatial, (2, 3))
    spatial_forward!(spatial, spatial_plan)
    spatial_inverse!(spatial, spatial_plan, execution)
    _precompile_drain!(_precompile_session(execution, Val(:rk), Val(:field)))
    _precompile_drain!(_precompile_session(execution, Val(:adams), Val(:envelope)))
    _precompile_drain!(_precompile_session(execution, Val(:low_storage), Val(:envelope)))
    isempty(planning.measured) || error("precompile planning must not retain measurements")
    return nothing
end

# Both macros are deliberately at top level. PrecompileTools 1.3's documented
# generation guard keeps these calls out of normal package loading; the let scope
# inside @setup_workload prevents its local values from becoming module storage.
PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        _precompile_cpu_workload()
    end
end
