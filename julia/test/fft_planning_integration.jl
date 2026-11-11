# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import LinearAlgebra
import SolpropJulia as SP

function planned_dft(input, sign)
    return setprecision(256) do
        n = length(input)
        [
            sum(
                Complex{BigFloat}(input[j + 1]) * cis(sign * 2big(pi) * j * k / n)
                    for j in 0:(n - 1)
            ) for k in 0:(n - 1)
        ]
    end
end

function planned_fft_bound(input)
    # Conditional small-FFT qualification: <=64n rounded scalar operations per
    # magnitude-propagated contribution, and <=n twiddle factors of modulus one
    # with absolute error <=4u. Twiddle accuracy is a separate hypothesis on these
    # FFTW codelets, not a consequence of counting operations. Product perturbations
    # give (1+gamma_64n)(1+gamma_4n)-1; the 256-bit DFT adds <2^-200 of this scale.
    T = typeof(real(zero(eltype(input))))
    u = BigFloat(eps(T)) / 2
    n = length(input)
    arithmetic = 64n * u / (1 - 64n * u)
    twiddle = 4n * u / (1 - 4n * u)
    return (arithmetic + twiddle + arithmetic * twiddle + big(2.0)^(-200)) *
        sum(abs, Complex{BigFloat}.(input))
end

function planned_temporal_bytes(time, spectrum, plan, execution)
    SP.temporal_forward!(spectrum, plan, time)
    SP.temporal_inverse!(time, plan, spectrum, execution)
    return @allocated begin
        SP.temporal_forward!(spectrum, plan, time)
        SP.temporal_inverse!(time, plan, spectrum, execution)
    end
end

function planned_spatial_bytes(buffer, plan, execution)
    SP.spatial_forward!(buffer, plan)
    SP.spatial_inverse!(buffer, plan, execution)
    return @allocated begin
        SP.spatial_forward!(buffer, plan)
        SP.spatial_inverse!(buffer, plan, execution)
    end
end

function planned_process_state()
    return SP.FFTW.@exclusive (
        sort!(strip.(split(SP._fft_export(Float64), '\n'; keepempty = false))),
        sort!(strip.(split(SP._fft_export(Float32), '\n'; keepempty = false))),
        SP._fft_threads(Float64), SP._fft_threads(Float32),
    )
end

@testset "Actual temporal buffers use isolated planning and retain normalization" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))),
            T in (Float32, Float64), real_signal in (false, true)
        planning = SP.FFTPlanning()
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks; planning)
        plain = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks)
        @test plain.planning === nothing
        @test typeof(plain) === typeof(execution)
        @test execution.planning === planning
        n = 8
        time = real_signal ? reshape(T.((1:16) ./ 16), n, 2) :
            reshape(Complex{T}[complex(j / 16, (7 - j) / 32) for j in 1:16], n, 2)
        spectrum = fill(Complex{T}(3, -2), real_signal ? n ÷ 2 + 1 : n, 2)
        before_time, before_spectrum = copy(time), copy(spectrum)
        process = planned_process_state()
        plan = SP.TemporalFFT(execution, time, spectrum)
        @test isequal(time, before_time) && isequal(spectrum, before_spectrum)
        @test planned_process_state() == process
        @test plan.inverse_scale == inv(T(n))
        @test length(planning.applied) == length(planning.measured) == 2
        @test isempty(planning.rejected)
        forward_kind = real_signal ? :r2c : :c2c_forward
        inverse_kind = real_signal ? :c2r : :c2c_backward
        for (input, output, kind) in (
                (time, spectrum, forward_kind), (spectrum, time, inverse_kind),
            )
            key = SP._fft_key(input, output, (1,), kind, 1, SP.FFTW.PRESERVE_INPUT)
            @test haskey(planning.applied, key)
        end
        SP.temporal_forward!(spectrum, plan, time)
        @test isequal(time, before_time)
        for column in axes(time, 2)
            expected = planned_dft(before_time[:, column], -1)[axes(spectrum, 1)]
            bound = planned_fft_bound(before_time[:, column])
            @test bound > 0
            @test maximum(abs.(Complex{BigFloat}.(spectrum[:, column]) .- expected)) <=
                bound
        end
        before_inverse = copy(spectrum)
        SP.temporal_inverse!(time, plan, spectrum, execution)
        @test isequal(spectrum, before_inverse)
        for column in axes(time, 2)
            full = real_signal ?
                vcat(
                    before_inverse[:, column],
                    conj.(before_inverse[(end - 1):-1:2, column]),
                ) :
                before_inverse[:, column]
            expected = planned_dft(full, 1) ./ n
            bound = planned_fft_bound(full) / n
            @test bound > 0
            @test maximum(abs.(Complex{BigFloat}.(time[:, column]) .- expected)) <= bound
        end
        facts = SP.fft_planning_record(planning)
        planned_temporal_bytes(time, spectrum, plan, execution)
        @test planned_temporal_bytes(time, spectrum, plan, execution) == 0
        @test SP.fft_planning_record(planning) == facts

        # The same actual allocations ensure identical alignment classes. A new
        # allocation is allowed to need another key rather than falsely claim reuse.
        offered = SP.FFTPlanning(; offered = planning.measured)
        reused_execution = SP.KernelExecution(
            SP.KernelAbstractions.CPU(), tasks;
            planning = offered
        )
        before_time, before_spectrum = copy(time), copy(spectrum)
        SP.TemporalFFT(reused_execution, time, spectrum)
        @test isequal(time, before_time) && isequal(spectrum, before_spectrum)
        @test Set(keys(offered.applied)) == offered.reused == Set(keys(planning.measured))
        @test isempty(offered.measured) && isempty(offered.rejected)
        @test planned_process_state() == process
        for context in (
                SP.FFTPlanning(; effort = :estimate), SP.FFTPlanning(; keep = false),
            )
            SP.TemporalFFT(
                SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks; planning = context),
                time, spectrum,
            )
            @test length(context.applied) == 2
            @test isempty(context.measured) && isempty(context.reused)
        end
        # Standalone callers still prepare public estimate plans without a context.
        standalone = SP.TemporalFFT(plain, time, spectrum)
        planned_temporal_bytes(time, spectrum, standalone, plain)
        @test planned_temporal_bytes(time, spectrum, standalone, plain) == 0
    end
end

function planned_spatial_reference(input, sign)
    return setprecision(256) do
        [
            sum(
                Complex{BigFloat}(input[row, j + 1, l + 1]) *
                    cis(sign * 2big(pi) * (BigFloat(j * k) / 4 + BigFloat(l * m) / 4))
                    for j in 0:3, l in 0:3
            ) for row in axes(input, 1), k in 0:3, m in 0:3
        ]
    end
end

@testset "Actual strided spatial buffers preserve storage and reuse both plans" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        planning = SP.FFTPlanning()
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks; planning)
        parent = reshape(ComplexF64[complex(j / 64, (31 - j) / 128) for j in 1:64], 4, 4, 4)
        buffer = view(parent, 1:2:3, :, :)
        original, original_parent = copy(buffer), copy(parent)
        process = planned_process_state()
        plan = SP.SpatialFFT(execution, buffer, (2, 3))
        @test isequal(parent, original_parent)
        @test plan.inverse_scale == 1 / 16
        @test length(planning.measured) == 2
        SP.spatial_forward!(buffer, plan)
        expected = planned_spatial_reference(original, -1)
        for row in axes(buffer, 1)
            bound = planned_fft_bound(vec(original[row, :, :]))
            @test bound > 0
            @test maximum(
                abs.(Complex{BigFloat}.(buffer[row, :, :]) .- expected[row, :, :]),
            ) <= bound
        end
        spectrum = copy(buffer)
        expected = planned_spatial_reference(spectrum, 1) ./ 16
        SP.spatial_inverse!(buffer, plan, execution)
        for row in axes(buffer, 1)
            bound = planned_fft_bound(vec(spectrum[row, :, :])) / 16
            @test bound > 0
            @test maximum(
                abs.(Complex{BigFloat}.(buffer[row, :, :]) .- expected[row, :, :]),
            ) <= bound
        end
        @test parent[2:2:4, :, :] == original_parent[2:2:4, :, :]
        facts = SP.fft_planning_record(planning)
        planned_spatial_bytes(buffer, plan, execution)
        @test planned_spatial_bytes(buffer, plan, execution) == 0
        @test SP.fft_planning_record(planning) == facts
        offered = SP.FFTPlanning(; offered = planning.measured)
        before = copy(parent)
        SP.SpatialFFT(
            SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks; planning = offered),
            buffer, (2, 3),
        )
        @test isequal(parent, before)
        @test offered.reused == Set(keys(planning.measured))
        @test isempty(offered.measured) && isempty(offered.rejected)
        @test planned_process_state() == process
    end
end

function planned_source_bytes(source::S, output, initial) where {S}
    source(output, initial, 0.001)
    return @allocated source(output, initial, 0.001)
end

function planned_session_bytes(session::S) where {S}
    return @allocated SP._precompile_drain!(session)
end

function planned_spatial_session(execution, formulation)
    modal, initial_column = SP._precompile_material_case(execution, formulation)
    grid = modal.grid
    n_frequency = length(grid.omega)
    shape, columns = (4, 2), 8
    operator = SP.SeparableSpatialOperator(
        execution; k_squared = fill(4.0 + 0.0im, n_frequency),
        k_squared_low = zeros(ComplexF64, n_frequency), kperp_sq = zeros(columns),
        kperp_sq_low = zeros(columns), frame = fill(2.0, n_frequency),
        spectral_rate = fill(0.01, n_frequency),
        prefactor_numerator = fill(-0.002im, n_frequency),
        prefactor_clamped = fill(-0.001im, n_frequency), active = grid.active,
        evanescent_rate = 1.0, z_span = (0.0, 0.01),
    )
    data = SP.CartesianSourceData(
        grid, shape, operator, zeros(columns), 1.0;
        kerr = modal.kerr, raman = modal.raman, plasma = modal.plasma,
    )
    source = SP.CartesianSource(execution, data; node_chunk = 3)
    initial = repeat(initial_column, 1, columns)
    norm = SP.MarchNorm(execution, grid.active, grid.active, ones(n_frequency), 1.0e-3, 0.0)
    schedule = SP.MarchSchedule(;
        planes = [0.002, 0.004], breakpoints = [0.0, 0.01], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = fill(0.001, 4),
        replay_edges = [0.0, 0.001, 0.002, 0.003, 0.004],
    )
    return SP.LowStorageSession(
        execution, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source,
        SP.LinearFlow(execution, operator, initial); capacity = 4,
    )
end

function planned_spatial_drain!(session)
    for _ in 1:16
        report = SP.advance!(session; work_limit = 4, deadline = 60.0)
        report.kind in (
            SP.SESSION_LAUNCH, SP.SESSION_PLANE, SP.SESSION_PROGRESS, SP.SESSION_COMPLETED,
        ) || error("unexpected spatial planning fixture event")
        SP.acknowledge!(session, report.generation)
        report.kind == SP.SESSION_COMPLETED && return nothing
    end
    error("spatial planning fixture exceeded its event bound")
end

function planned_spatial_session_bytes(session::S) where {S}
    return @allocated planned_spatial_drain!(session)
end

@testset "Prepared material and observation sessions never plan during advancement" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        planning = SP.FFTPlanning()
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks; planning)
        process = planned_process_state()
        for (method, kind) in (
                (:rk, :field), (:adams, :envelope), (:low_storage, :envelope),
            )
            session = SP._precompile_session(execution, Val(method), Val(kind))
            source = session.source
            component = only(source.raman)
            key = SP._fft_key(
                component.time_kernel, component.spectrum, (1,), :r2c, 1,
                SP.FFTW.PRESERVE_INPUT,
            )
            @test haskey(planning.applied, key)
            initial, output = copy(session.field), similar(session.field)
            facts = SP.fft_planning_record(planning)
            planned_source_bytes(source, output, initial)
            @test planned_source_bytes(source, output, initial) == 0
            @test isequal(initial, session.field)
            @test all(isfinite, output) && any(!iszero, output)
            @test SP.fft_planning_record(planning) == facts
            SP.close_session!(session)
            # Warm the exact drain specialization before measuring a fresh session.
            warm = SP._precompile_session(execution, Val(method), Val(kind))
            planned_session_bytes(warm)
            measured = SP._precompile_session(execution, Val(method), Val(kind))
            facts = SP.fft_planning_record(planning)
            @test planned_session_bytes(measured) == 0
            @test measured.state.accepted == 4
            @test measured.source.calls > 0 && all(isfinite, measured.field)
            @test SP.fft_planning_record(planning) == facts
        end
        for kind in (:field, :envelope)
            warm = planned_spatial_session(execution, Val(kind))
            planned_spatial_session_bytes(warm)
            SP.close_session!(warm)
            measured = planned_spatial_session(execution, Val(kind))
            band = measured.source.band
            key = SP._fft_key(band.planes, band.planes, (2, 3), :c2c_forward, 1, UInt32(0))
            @test haskey(planning.applied, key)
            facts = SP.fft_planning_record(planning)
            @test planned_spatial_session_bytes(measured) == 0
            @test measured.state.accepted == 4 && measured.source.calls == 16
            @test all(isfinite, measured.field)
            @test SP.fft_planning_record(planning) == facts
            SP.close_session!(measured)
        end
        @test planned_process_state() == process
    end
end
