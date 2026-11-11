# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared fixtures for the host orchestration model and actual CUDA execution.
# Their MPFR reference is a direct DFT/cubic/DFT calculation, independent of the
# production transform, contraction and response implementations.
import SolpropJulia as SP

function device_source_fixture(
        execution, kind; form = :ordinary, prefactor = :constant, temporal = false,
    )
    host = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    n_time = 16
    crop = kind === :field ? collect(0:4) : [0, 1, 2, 3, 12, 13, 14, 15]
    frequency = length(crop)
    active = fill(true, frequency)
    active[1] = false
    kind === :field && (active[end] = false)
    window = [0.5 + mod(time, 4) / 8 for time in 0:(n_time - 1)]
    spectral_window = [0.5 + mod(row, 3) / 8 for row in 1:frequency]
    carrier = form === :harmonic && kind === :envelope ?
        ComplexF64[cispi(time / 4) for time in 0:(n_time - 1)] : nothing
    grid = SP.ModalSourceGrid(
        execution; kind, n_time, dt = 0.125, crop_index = crop,
        omega = collect(1.0:frequency), active, omega_window = spectral_window,
        source_window = window, source_window_derivative = zeros(n_time), carrier,
        temporal_rate = temporal ? fill(0.25, n_time) : nothing,
    )
    synthesis = [1.0 0.25; 0.5 -0.5; -0.25 0.75]
    projection = [0.5 0.25 -0.125; 0.125 -0.25 0.5]
    map = SP.ModalSourceMap(
        execution; synthesis, projection, coarse_projection = 0.875projection,
    )
    values = ComplexF64[
        active[row] ? -im * (row + column) / 128 : 0
            for row in 1:frequency, column in 1:2
    ]
    prepared = if prefactor === :table
        coefficients = cat(reshape(values, :, 1), reshape(values / 8, :, 1); dims = 2)
        table = SP.ChebyshevTable(
            execution, [0.0, 1.0], reshape(coefficients, length(values), 2, 1),
        )
        SP.SourcePrefactor(execution, table, size(values))
    elseif prefactor === :expansion
        weights = SP.ChebyshevTable(
            execution, [0.0, 1.0], reshape([1.0, 0.5, 0.125, -0.25], 2, 2, 1),
        )
        SP.SourcePrefactor(execution, cat(values, values / 4; dims = 3), weights)
    else
        SP.SourcePrefactor(execution, values)
    end
    # The material profile is host-owned even when numerical maps are on CUDA.
    profile = SP.SourceProfile(
        SP.ChebyshevTable(host, [0.0, 1.0], reshape([1.0, 0.125], 1, 2, 1)),
    )
    analytic = kind === :envelope || form === :analytic
    coefficients = SP.KerrCoefficients(
        0.125, analytic ? :analytic : :real;
        take_real = kind === :field && analytic,
        harmonic_factor = form === :harmonic ? 0.125 : 0.0,
    )
    data = SP.ModalSourceData(
        grid, map, prepared; kerr = SP.ModalKerr(coefficients, profile),
    )
    state = ComplexF64[
        active[row] ? complex(row - column, mod(row + column, 3) - 1) / 32 : 0
            for row in 1:frequency, column in 1:2
    ]
    reference = (;
        kind, form, prefactor, crop, active, window, spectral_window, carrier,
        synthesis, projection, values, n_time,
    )
    return data, state, reference
end

function device_source_exact(input, position, fixture)
    n_time, modes = fixture.n_time, size(input, 2)
    field = fixture.kind === :field
    analytic = field && fixture.form === :analytic
    full = zeros(Complex{BigFloat}, n_time, modes)
    for mode in 1:modes, row in axes(input, 1)
        full[fixture.crop[row] + 1, mode] = Complex{BigFloat}(input[row, mode])
    end
    if field
        full[1, :] .= real.(full[1, :])
        full[n_time ÷ 2 + 1, :] .= real.(full[n_time ÷ 2 + 1, :])
        for row in 2:(n_time ÷ 2)
            if analytic
                full[row, :] .*= 2
            else
                full[n_time - row + 2, :] .= conj.(full[row, :])
            end
        end
    end
    modal = Complex{BigFloat}[
        sum(
            full[row + 1, mode] * cis(2big(pi) * row * time / n_time)
                for row in 0:(n_time - 1)
        ) / n_time
            for time in 0:(n_time - 1), mode in 1:modes
    ]
    field && !analytic && (modal .= real.(modal))
    nodal = modal * transpose(BigFloat.(fixture.synthesis))
    density = 1 + (2BigFloat(position) - 1) / 8
    material = similar(nodal)
    for node in axes(nodal, 2), row in axes(nodal, 1)
        value = nodal[row, node]
        cubic = if field && !analytic
            value^3
        else
            harmonic = fixture.carrier === nothing ? zero(value) :
                Complex{BigFloat}(fixture.carrier[row]) * value^3 / 8
            3abs2(value) * value / 4 + harmonic
        end
        material[row, node] = density / 8 * fixture.window[row] *
            (field ? real(cubic) : cubic)
    end
    projected = material * transpose(BigFloat.(fixture.projection))
    result = zeros(Complex{BigFloat}, size(input))
    coordinate = 2BigFloat(position) - 1
    prefactor_scale = fixture.prefactor === :table ? 1 + coordinate / 8 :
        fixture.prefactor === :expansion ?
        1 + coordinate / 8 + (big(0.5) - coordinate / 4) / 4 : big(1)
    for mode in 1:modes, row in axes(input, 1)
        fixture.active[row] || continue
        frequency = fixture.crop[row]
        transformed = sum(
            projected[time + 1, mode] * cis(-2big(pi) * frequency * time / n_time)
                for time in 0:(n_time - 1)
        )
        result[row, mode] = transformed * fixture.values[row, mode] *
            prefactor_scale * fixture.spectral_window[row]
    end
    return result
end

function device_source_envelope(fixture; radius = big(1))
    # For all tested z in [0,1], the represented profile is <=9/8 and the
    # prefactor multiplier is <=5/4. Twice the number of stored coarse bins
    # bounds both Hermitian and analytic inverse spectral sums. These absolute
    # envelopes precede all modal/FFT cancellation.
    synthesis = maximum(sum(abs, fixture.synthesis; dims = 2))
    projection = maximum(sum(abs, fixture.projection; dims = 2))
    inverse = BigFloat(2length(fixture.crop)) / fixture.n_time
    alpha = inverse * synthesis
    coefficient = big(9) / 64 # (9/8 density) * (1/8 coefficient), cubic factor <=1.
    forward = maximum(abs, fixture.values) * big(5) / 4 * fixture.n_time * projection
    magnitude = forward * coefficient * (alpha * radius)^3
    lipschitz = forward * 3coefficient * alpha^3 * radius^2
    return magnitude, lipschitz
end

function device_source_budget(fixture; radius = big(1))
    magnitude, _ = device_source_envelope(fixture; radius)
    unit, count = BigFloat(eps(Float64)) / 2, 8192
    gamma = count * unit / (1 - count * unit)
    @assert 0 < gamma < big(1.0e-11)
    # n=16 direct/real-split FFTs, <=3-node and 2-mode sums, the cubic and all
    # scalar weights fit below 8192 component operations. The vendor FFT model
    # additionally assumes twiddles within 2u/component; the actual GPU suite
    # checks its finite fixtures against this independent MPFR transform graph.
    # This is conditional finite-fixture qualification, not a vendor FFT proof.
    return gamma * magnitude + count * BigFloat(nextfloat(0.0))
end

device_source_tableau() = SP.EmbeddedTableau(
    [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
)

function device_source_exact_steps(initial, fixture, steps)
    field = Complex{BigFloat}.(initial)
    h = big(1) / 128
    error = big(0)
    magnitude, lipschitz = device_source_envelope(fixture)
    source_error = device_source_budget(fixture)
    unit, count = BigFloat(eps(Float64)) / 2, 256
    gamma = count * unit / (1 - count * unit)
    @assert h * lipschitz <= big(0.25)
    for step in 0:(steps - 1)
        position = step * h
        first = device_source_exact(field, position, fixture)
        stage = field + h * first
        second = device_source_exact(stage, position + h, fixture)
        field += h / 2 * (first + second)
        # Include source/formation errors in the actual second-stage domain,
        # not just the accumulated accepted-state error from the previous step.
        stage_error = (1 + h * lipschitz) * error + h * source_error +
            gamma * (1 + h * magnitude)
        @assert maximum(abs, stage) + stage_error < 1
        # Heun's one-step map has Lipschitz bound 1+hL+(hL)^2/2 <= exp(2hL).
        # 256 component operations cover the represented state/stage updates;
        # the two source perturbations include the second-stage sensitivity.
        local_error = gamma * (1 + h * magnitude) +
            h * (2 + h * lipschitz) * source_error
        error = exp(2h * lipschitz) * error + local_error
        @assert maximum(abs, field) + error < 1
    end
    return field, error
end

function device_source_steps!(state, source, flow, work, tableau, execution, steps)
    h = 1 / 128
    for step in 0:(steps - 1)
        SP.rk_trial!(work, state, step * h, h, tableau, source, flow, execution)
        SP.rk_commit!(state, work, source.data.grid.active, execution)
    end
    return nothing
end
