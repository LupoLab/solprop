# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import LinearAlgebra
import SolpropJulia as SP

function composition_profile(execution, offset, slope; name = :test_profile, edge = 0.4)
    edges = [0.0, edge, 1.0]
    coefficients = zeros(Float64, 1, 2, 2)
    for interval in 1:2
        left, right = edges[interval], edges[interval + 1]
        coefficients[1, 1, interval] = offset + slope * (left + right) / 2
        coefficients[1, 2, interval] = slope * (right - left) / 2
    end
    return SP.SourceProfile(SP.ChebyshevTable(execution, edges, coefficients); name)
end

function composition_value(profile, position)
    profile.value isa Float64 && return profile.value
    return composition_table(profile.value, position)[1]
end

function composition_table(table, position)
    interval = min(
        searchsortedlast(table.breakpoints, position), length(table.breakpoints) - 1
    )
    interval = max(interval, 1)
    left, right = table.breakpoints[interval:(interval + 1)]
    coordinate = (2position - left - right) / (right - left)
    result = copy(table.coefficients[:, 1, interval])
    previous, current = 1.0, coordinate
    for degree in 2:size(table.coefficients, 2)
        result .+= table.coefficients[:, degree, interval] .* current
        previous, current = current, 2coordinate * current - previous
    end
    return result
end

function composition_prefactor(prepared, position)
    if prepared.weights !== nothing
        weights = composition_table(prepared.weights, position)
        return dropdims(sum(prepared.data .* reshape(weights, 1, 1, :); dims = 3); dims = 3)
    elseif prepared.data isa SP.ChebyshevTable
        return reshape(composition_table(prepared.data, position), prepared.shape)
    end
    return prepared.data
end

function composition_dft(samples)
    n_time, columns = size(samples)
    return ComplexF64[
        sum(
            samples[time + 1, column] * cis(-2pi * frequency * time / n_time)
                for time in 0:(n_time - 1)
        )
            for frequency in 0:(n_time - 1), column in 1:columns
    ]
end

function composition_inverse(spectrum, n_time; real_signal = false, analytic = false)
    full = zeros(ComplexF64, n_time, size(spectrum, 2))
    if real_signal
        full[1, :] .= real.(spectrum[1, :])
        full[n_time ÷ 2 + 1, :] .= real.(spectrum[end, :])
        for index in 2:(n_time ÷ 2)
            full[index, :] .= spectrum[index, :]
            full[n_time - index + 2, :] .= conj.(spectrum[index, :])
        end
        if analytic
            full[2:(n_time ÷ 2), :] .*= 2
            full[(n_time ÷ 2 + 2):end, :] .= 0
        end
    else
        full .= spectrum
    end
    result = ComplexF64[
        sum(
            full[frequency + 1, column] * cis(2pi * frequency * time / n_time)
                for frequency in 0:(n_time - 1)
        ) / n_time
            for time in 0:(n_time - 1), column in axes(spectrum, 2)
    ]
    return real_signal && !analytic ? real.(result) : result
end

function composition_cycle()
    # Gauss-Legendre from its Jacobi eigenproblem, independent of the plasma kernel.
    nodes, weights, ends = Float64[], Float64[], Int[]
    for count in (16, 32)
        off_diagonal = [index / sqrt(4index^2 - 1) for index in 1:(count - 1)]
        eigensystem = LinearAlgebra.eigen(
            LinearAlgebra.SymTridiagonal(zeros(count), off_diagonal)
        )
        append!(nodes, eigensystem.values)
        append!(weights, 2 .* eigensystem.vectors[1, :] .^ 2)
        push!(ends, length(nodes))
    end
    return SP.CycleQuadrature(
        nodes, weights, Tuple(ends); moment_rtol = 2.0e-8, negligible_rate = 1.0e-30
    )
end

function composition_channel(execution, scale, index, kind)
    lower, upper = 1.0e-12, 100.0
    coefficients = reshape([0.0, 0.0, 2.0, log(scale) + 2log(lower)], 4, 1)
    rate = SP.IonisationRateTable(
        coefficients, lower, upper, scale * upper^2, log(lower), log(upper / lower)
    )
    channel = SP.PlasmaChannel(rate, 1.1 + 0.2index, 0.3 + 0.1index)
    density = composition_profile(execution, 0.4 + 0.1index, 0.07index; edge = 0.31)
    initial = composition_profile(execution, 0.03index, 0.01index; edge = 0.62)
    return SP.ModalPlasmaChannel(
        channel, density, initial;
        quadrature = kind == :field ? nothing : composition_cycle(),
    )
end

function composition_case(
        execution, kind; kerr = true, raman = 0, plasma = false, thg = false,
        temporal = false, coarse = true, prefactor_kind = :constant,
    )
    n_time, coarse_time = 16, 8
    crop = kind == :field ? collect(0:4) : [0, 1, 2, 3, 12, 13, 14, 15]
    n_frequency = length(crop)
    omega = collect(1.0:n_frequency) .+ 3.0
    active = fill(true, n_frequency)
    active[1] = false
    kind == :field && (active[end] = false)
    window = [0.65 + 0.25cos(2pi * index / n_time) for index in 0:(n_time - 1)]
    derivative = [-0.25sin(2pi * index / n_time) for index in 0:(n_time - 1)]
    carrier = kind == :envelope && thg ?
        ComplexF64[cis(4pi * index / n_time) for index in 0:(n_time - 1)] : nothing
    grid = SP.ModalSourceGrid(
        execution; kind, n_time, dt = 0.04, crop_index = crop, omega, active,
        omega_window = collect(range(0.7, 0.95; length = n_frequency)),
        source_window = window, source_window_derivative = derivative,
        temporal_rate = temporal ? collect(range(0.1, 0.4; length = n_time)) : nothing,
        carrier,
    )
    synthesis = 2.0 .* [0.8 0.3; 0.5 -0.4; 0.2 0.7]
    projection = [0.4 0.2 -0.1; 0.1 -0.3 0.5]
    maps = SP.ModalSourceMap(
        execution; synthesis, projection,
        coarse_projection = coarse ? 0.93projection : nothing,
        taper = composition_profile(execution, 0.9, 0.15; name = :taper, edge = 0.72),
    )
    prefactor = ComplexF64[
        active[row] ? -im * (0.15 + 0.02row + 0.01column) / 2 : 0
            for row in 1:n_frequency, column in 1:2
    ]
    prepared_prefactor = if prefactor_kind == :constant
        SP.SourcePrefactor(execution, prefactor)
    elseif prefactor_kind == :table
        coefficients = zeros(ComplexF64, length(prefactor), 2, 1)
        coefficients[:, 1, 1] .= vec(prefactor) .* 1.1
        coefficients[:, 2, 1] .= vec(prefactor) .* 0.1
        SP.SourcePrefactor(
            SP.ChebyshevTable(execution, [0.0, 1.0], coefficients), size(prefactor)
        )
    else
        terms = cat(prefactor, 0.2prefactor; dims = 3)
        weights = SP.ChebyshevTable(
            execution, [0.0, 1.0], reshape([1.0, 0.5, 0.1, -0.2], 2, 2, 1)
        )
        SP.SourcePrefactor(execution, terms, weights)
    end
    kerr_data = if kerr
        form = kind == :field && thg ? :real : :analytic
        coefficients = SP.KerrCoefficients(
            0.7, form; take_real = kind == :field && !thg,
            harmonic_factor = kind == :envelope && thg ? 0.25 : 0.0,
        )
        SP.ModalKerr(coefficients, composition_profile(execution, 0.8, 0.1; edge = 0.2))
    else
        nothing
    end
    raman_data = ntuple(raman) do index
        undamped = zeros(Float64, 2n_time, 2)
        for time in 1:n_time
            undamped[time, 1] = 0.3sin(0.2(time - 1))
            undamped[time, 2] = 0.1sin(0.4(time - 1))
        end
        kernel = SP.RamanKernel(undamped, grid.dt)
        decays = (
            composition_profile(execution, 0.2index, 0.04; edge = 0.18),
            composition_profile(execution, 0.1index, 0.07; edge = 0.57),
        )
        density = composition_profile(execution, 0.3index, 0.09; edge = 0.86)
        SP.ModalRaman(kernel, decays, density; fraction = 0.25, material_factor = 0.4, thg)
    end
    plasma_data = if plasma
        channels = (
            composition_channel(execution, 0.12, 1, kind),
            composition_channel(execution, 0.07, 2, kind),
        )
        if kind == :field
            SP.ModalPlasma(execution, channels, SP.RunningIntegralRule([1.0], grid.dt))
        else
            SP.ModalPlasma(
                execution, channels;
                drude = [-active[row] / omega[row]^2 for row in 1:n_frequency],
                loss = [-im * active[row] / omega[row] for row in 1:n_frequency],
            )
        end
    else
        nothing
    end
    data = SP.ModalSourceData(
        grid, maps, prepared_prefactor;
        kerr = kerr_data, raman = raman_data, plasma = plasma_data,
    )
    state = ComplexF64[
        active[row] ? (0.3 + 0.1column) * cis(0.7row + 0.2column) : 0
            for row in 1:n_frequency, column in 1:2
    ]
    @test grid.n_time == 2coarse_time
    return data, state
end

function composition_scan(input, dt)
    result = zeros(size(input))
    for column in axes(input, 2), row in 2:size(input, 1)
        result[row, column] = result[row - 1, column] +
            dt * (input[row - 1, column] + input[row, column]) / 2
    end
    return result
end

function composition_reference(
        data, state, position; temporal_active = true, coarse = false,
    )
    grid, maps = data.grid, data.map
    field = grid.kind == :field
    fine_count = field ? grid.n_time ÷ 2 + 1 : grid.n_time
    padded = zeros(ComplexF64, fine_count, size(state, 2))
    padded[grid.crop_index .+ 1, :] .= state
    modal = composition_inverse(padded, grid.n_time; real_signal = field)
    analytic = composition_inverse(
        padded, grid.n_time; real_signal = field, analytic = true
    )
    taper = composition_value(maps.taper, position)
    nodes = (modal * transpose(maps.synthesis)) .* taper
    analytic_nodes = (analytic * transpose(maps.synthesis)) .* taper
    polarisation = zeros(eltype(nodes), size(nodes))
    contributions = zeros(Float64, size(nodes))
    if data.kerr !== nothing
        coefficients = data.kerr.coefficients
        signal = field && coefficients isa SP.KerrCoefficients{Float64, :analytic} ?
            analytic_nodes : nodes
        value = if coefficients isa SP.KerrCoefficients{Float64, :real}
            signal .^ 3
        else
            fundamental = coefficients.self_factor .* abs2.(signal) .* signal .+
                coefficients.conjugate_factor .* signal .^ 2 .* conj.(signal)
            if !iszero(coefficients.harmonic_factor)
                fundamental .+= coefficients.harmonic_factor .* grid.carrier .* signal .^ 3
            end
            coefficients.take_real ? real.(fundamental) : fundamental
        end
        value .*= coefficients.coefficient * composition_value(data.kerr.density, position)
        polarisation .+= value
        contributions .+= abs.(value)
    end
    for component in data.raman
        kernel = zeros(grid.n_time)
        for term in axes(component.kernel.undamped, 2), row in 1:grid.n_time
            rate = composition_value(component.decay_profiles[term], position)
            kernel[row] += component.fraction * component.kernel.undamped[row, term] *
                exp(-rate * (row - 1) * grid.dt)
        end
        drive = field && component.thg ? nodes .^ 2 :
            abs2.(field ? analytic_nodes : nodes) ./ 2
        convolution = [
            grid.dt * sum(kernel[row - index + 1] * drive[index, column] for index in 1:row)
                for row in 1:grid.n_time, column in axes(nodes, 2)
        ]
        factor = component.material_factor * composition_value(component.density, position)
        value = factor .* nodes .* convolution
        polarisation .+= value
        contributions .+= abs.(value)
    end
    projection = coarse ? maps.coarse_projection : maps.projection
    project(samples) = (samples * transpose(projection)) ./ taper
    spectrum(samples) = composition_dft(project(samples))[grid.crop_index .+ 1, :]
    # Absolute projection/DFT contributions, before cancellation, set the rounding scale.
    source_scale = sum(
        (contributions .* grid.source_window) * transpose(abs.(projection)) ./ taper;
        dims = 1,
    )
    result = spectrum(polarisation .* grid.source_window)
    bound = repeat(source_scale, length(grid.omega), 1)
    if data.plasma !== nothing
        first, second = zeros(eltype(nodes), size(nodes)), zeros(eltype(nodes), size(nodes))
        for component in data.plasma.channels
            channel = component.channel
            density = composition_value(component.density, position)
            initial = composition_value(component.preionisation, position)
            # The fixture rate is exactly k*E². Its scalar cycle mean is k*|A|²/2,
            # with loss envelope k*A. This independent analytic oracle uses no rate
            # spline or angular quadrature implementation from the tested leaves.
            scale = channel.rate.rate_max / channel.rate.field_max^2
            rate = scale .* abs2.(nodes) .* (field ? 1 : 0.5)
            exposure = composition_scan(rate, grid.dt)
            survival = (1 - initial) .* exp.(-exposure)
            fraction = initial .+ (1 - initial) .* (-expm1.(-exposure))
            if field
                current = density .* (
                    channel.charge_mass_ratio .*
                        composition_scan(fraction .* nodes, grid.dt) .+
                        channel.ionisation_potential * scale .* survival .* nodes
                )
                first .+= current .* grid.source_window
                second .+= current .* grid.source_window_derivative
            else
                first .+= channel.charge_mass_ratio * density .* fraction .* nodes .*
                    grid.source_window
                second .+= channel.ionisation_potential * density * scale .*
                    survival .* nodes .* grid.source_window
            end
        end
        factors = if field
            (
                [
                    grid.active[row] ? inv(im * grid.omega[row]) : 0
                        for row in eachindex(grid.omega)
                ],
                [
                    grid.active[row] ? -inv((im * grid.omega[row])^2) : 0
                        for row in eachindex(grid.omega)
                ],
            )
        else
            (data.plasma.drude, data.plasma.loss)
        end
        for (stream, factor) in zip((first, second), factors)
            result .+= factor .* spectrum(stream)
            bound .+= abs.(factor) .* sum(
                abs.(stream) * transpose(abs.(projection)) ./ taper; dims = 1
            )
        end
    end
    prefactor = composition_prefactor(data.prefactor, position) .* grid.omega_window
    result .*= prefactor
    bound .*= abs.(prefactor)
    if temporal_active && grid.temporal_rate !== nothing
        absorber = -0.5 .* grid.temporal_rate .* modal
        result .+= composition_dft(absorber)[grid.crop_index .+ 1, :] .* grid.active
        bound .+= sum(abs.(absorber); dims = 1) .* grid.active
    end
    result[.!grid.active, :] .= 0
    return result, maximum(bound)
end

function composition_cycle!(source::S, output, state, position) where {S}
    source(output, state, position)
    return nothing
end

function composition_allocations(source, output, state, position)
    composition_cycle!(source, output, state, position)
    composition_cycle!(source, output, state, position)
    return @allocated composition_cycle!(source, output, state, position)
end

@testset "modal scalar source composition" begin
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    # A single source has fewer than 8192 scalar/DFT recurrence operations along any
    # dependency path in these tiny fixtures. The absolute contribution scale retains
    # cancellation within maps, streams and FFTs; no relative-to-result tolerance is used.
    gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for (kerr, raman, plasma, thg, temporal, prefactor_kind) in (
                (true, 0, false, true, false, :constant),
                (true, 0, false, false, false, :table),
                (false, 1, false, true, false, :constant),
                (false, 2, false, false, false, :expansion),
                (false, 0, true, false, false, :table),
                (true, 2, true, false, true, :expansion),
            )
            data, state = composition_case(
                execution, kind; kerr, raman, plasma, thg, temporal, prefactor_kind
            )
            before = copy(state)
            reference, scale = composition_reference(data, state, 0.53)
            @test isfinite(scale) && scale > 0
            for chunk in (1, 2, 3)
                source = SP.ModalSource(execution, data; node_chunk = chunk)
                output = fill(ComplexF64(NaN), size(state))
                @test source(output, state, 0.53) === nothing
                @test source.calls == 1 && source.ready
                @test maximum(abs.(output - reference)) <= gamma * scale
                @test all(iszero, output[.!data.grid.active, :])
                @test isequal(state, before)
                coarse, coarse_scale = composition_reference(
                    data, state, 0.53; temporal_active = false, coarse = true
                )
                fine, fine_scale = composition_reference(
                    data, state, 0.53; temporal_active = false
                )
                @test maximum(abs.(source.quadrature_coarse - coarse)) <=
                    gamma * coarse_scale
                @test maximum(abs.(source.quadrature_fine - fine)) <= gamma * fine_scale
                @test composition_allocations(source, output, state, 0.53) == 0
                # An incomplete final block must clear every later-read scratch cell.
                fill!(source.nodes, NaN)
                fill!(source.polarisation, NaN)
                fill!(source.projected, NaN)
                fill!(source.projection_work, NaN)
                fill!(source.coarse_projected, NaN)
                source.analytic === nothing || fill!(source.analytic.nodes, NaN)
                if source.raman_buffers !== nothing
                    fill!(source.raman_buffers.drive, NaN)
                    fill!(source.raman_buffers.frequency, NaN)
                end
                if source.plasma !== nothing
                    fill!(source.plasma.first, NaN)
                    fill!(source.plasma.second, NaN)
                end
                source(output, state, 0.53)
                @test maximum(abs.(output - reference)) <= gamma * scale
                if chunk == 2
                    @test all(iszero, source.nodes[:, 2])
                    @test all(iszero, source.polarisation[:, 2])
                end
            end
        end
    end
end

@testset "source absence, temporal policy and failed calls" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    for kind in (:field, :envelope)
        data, state = composition_case(execution, kind; kerr = false, temporal = true)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        output = similar(state)
        reference, scale = composition_reference(data, state, 0.3)
        source(output, state, 0.3)
        @test maximum(abs.(output - reference)) <= 4096eps(Float64) * scale
        @test all(iszero, source.quadrature_fine)
        SP.set_temporal_source!(source, false)
        @test !source.ready && !source.temporal_active
        source(output, state, 0.3)
        @test all(iszero, output)
        SP.set_temporal_source!(source, true)
        @test composition_allocations(source, output, state, 0.3) == 0
        calls = source.calls
        @test_throws ArgumentError source(state, state, 0.3)
        @test source.calls == calls + 1 && !source.ready
        @test_throws ArgumentError source(output, state, NaN)
        @test_throws DimensionMismatch source(zeros(ComplexF64, 1, 1), state, 0.3)
        damaged = copy(state)
        damaged[1] = NaN
        @test_throws SP.NonfiniteSourceError source(output, damaged, 0.3)
        @test !source.ready
    end
    data, state = composition_case(execution, :envelope)
    source = SP.ModalSource(execution, data; node_chunk = 2)
    @test_throws ArgumentError SP.set_temporal_source!(source, true)
    @test_throws ArgumentError source(similar(state), state, 1.1)
    @test source.calls == 1 && !source.ready
    @test_throws ArgumentError SP.ModalSource(execution, data; node_chunk = 0)
    @test_throws ArgumentError SP.SourceProfile(Inf)
    @test_throws ArgumentError SP.SourceProfile(
        SP.ChebyshevTable(execution, [0.0, 1.0], zeros(2, 1, 1))
    )
    @test_throws ArgumentError SP.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1), n_state_components = 2
    )
    @test_throws ArgumentError SP.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1),
        n_physical_components = 2,
    )
    kernel = SP.RamanKernel(zeros(8, 1), 0.1)
    @test_throws ArgumentError SP.ModalRaman(
        kernel, (SP.SourceProfile(0.0),), SP.SourceProfile(1.0);
        fraction = 0.2, material_factor = 1.0, tensor = true,
    )
end

@testset "source owns numerical inputs and refuses incompatible preparation" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    table = SP.ChebyshevTable(execution, [0.0, 1.0], reshape([2.0], 1, 1, 1))
    profile = SP.SourceProfile(table; name = :owned)
    table.breakpoints[1] = -1.0
    table.coefficients[1] = 7.0
    @test profile.value.breakpoints == [0.0, 1.0]
    @test composition_value(profile, 0.4) == 2.0
    values = fill(1.0im, 3, 2)
    prefactor = SP.SourcePrefactor(execution, values)
    fill!(values, 0)
    @test all(==(1.0im), prefactor.data)
    @test_throws ArgumentError SP.SourcePrefactor(execution, fill(ComplexF64(Inf), 2, 2))
    @test_throws DimensionMismatch SP.SourcePrefactor(
        SP.ChebyshevTable(execution, [0.0, 1.0], zeros(ComplexF64, 1, 1, 1)), (2, 2)
    )
    data, state = composition_case(execution, :field)
    @test_throws DimensionMismatch SP.ModalSourceMap(
        execution; synthesis = ones(2, 1), projection = ones(2, 1)
    )
    @test_throws DimensionMismatch SP.ModalSourceData(
        data.grid, data.map, SP.SourcePrefactor(execution, zeros(ComplexF64, 2, 2))
    )
    envelope, _ = composition_case(execution, :envelope)
    @test_throws ArgumentError SP.ModalSourceData(
        envelope.grid, envelope.map, envelope.prefactor;
        kerr = SP.ModalKerr(SP.KerrCoefficients(1.0, :real), SP.SourceProfile(1.0)),
    )
    @test_throws ArgumentError SP.ModalSourceData(
        data.grid, data.map, data.prefactor;
        kerr = SP.ModalKerr(SP.KerrCoefficients(1.0, :analytic), SP.SourceProfile(1.0)),
    )
    @test_throws ArgumentError SP.ModalSourceData(
        envelope.grid, envelope.map, envelope.prefactor;
        kerr = SP.ModalKerr(
            SP.KerrCoefficients(1.0, :analytic; harmonic_factor = 0.25),
            SP.SourceProfile(1.0),
        ),
    )
    invalid_taper = SP.ModalSourceMap(
        execution; synthesis = data.map.synthesis, projection = data.map.projection,
        taper = SP.SourceProfile(-1.0),
    )
    source = SP.ModalSource(
        execution,
        SP.ModalSourceData(data.grid, invalid_taper, data.prefactor; kerr = data.kerr);
        node_chunk = 2,
    )
    @test_throws ArgumentError source(similar(state), state, 0.2)
    @test source.calls == 1 && !source.ready
    for kind in (:field, :envelope)
        prepared, _ = composition_case(
            execution, kind; kerr = false, raman = 1, plasma = true
        )
        grid, plasma = prepared.grid, prepared.plasma
        wrong_kind = kind == :field ? :envelope : :field
        wrong_channel = composition_channel(execution, 0.1, 1, wrong_kind)
        if kind == :field
            @test_throws ArgumentError SP.ModalPlasma(
                execution, (wrong_channel,), plasma.rule
            )
        else
            @test_throws ArgumentError SP.ModalPlasma(
                execution, (wrong_channel,); drude = plasma.drude, loss = plasma.loss
            )
        end
        component = prepared.raman[1]
        @test_throws DimensionMismatch SP.ModalRaman(
            component.kernel, (), component.density; fraction = 0.2, material_factor = 0.4
        )
        @test_throws ArgumentError SP.ModalRaman(
            component.kernel, component.decay_profiles, component.density;
            fraction = -0.2, material_factor = 0.4,
        )
        malformed = SP.RamanKernel(zeros(Float64, 8, 1), grid.dt)
        short = SP.ModalRaman(
            malformed, (SP.SourceProfile(0.0),), SP.SourceProfile(1.0);
            fraction = 0.2, material_factor = 0.4,
        )
        @test_throws DimensionMismatch SP.ModalSourceData(
            grid, prepared.map, prepared.prefactor; raman = (short,)
        )
    end
end

@testset "failed plasma channels do not reach spectral assembly" begin
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        data, state = composition_case(execution, kind; kerr = false, plasma = true)
        first, second = data.plasma.channels
        overflow = SP.ModalPlasmaChannel(
            SP.PlasmaChannel(second.channel.rate, floatmax(Float64), 1.0),
            SP.SourceProfile(floatmax(Float64)), second.preionisation;
            quadrature = second.quadrature,
        )
        bad_plasma = kind == :field ?
            SP.ModalPlasma(execution, (first, overflow), data.plasma.rule) :
            SP.ModalPlasma(
                execution, (first, overflow);
                drude = data.plasma.drude, loss = data.plasma.loss,
            )
        source = SP.ModalSource(
            execution,
            SP.ModalSourceData(data.grid, data.map, data.prefactor; plasma = bad_plasma);
            node_chunk = 2,
        )
        output = fill(ComplexF64(7), size(state))
        before = copy(state)
        error = try
            source(output, state, 0.4)
            nothing
        catch caught
            caught
        end
        @test error isa SP.SourcePlasmaError
        if error isa SP.SourcePlasmaError
            @test error.code == SP.PlasmaNonfiniteResult
            @test error.channel == 2 && 1 <= error.node <= 2 && error.sample > 0
            @test occursin("channel 2", sprint(showerror, error))
        end
        @test source.calls == 1 && !source.ready
        @test all(==(ComplexF64(7)), output)
        @test isequal(before, state)
    end
end

@testset "signed composite Kerr profile and nonfinite source arithmetic" begin
    # A translated mixture stores sum(chi3_s*rho_s) in the stage-weight slot,
    # which may be signed even though the underlying number densities are not (#635).
    for kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
        data, state = composition_case(execution, kind; coarse = false)
        signed = SP.ModalKerr(
            data.kerr.coefficients, composition_profile(execution, -0.4, 0.1)
        )
        prepared = SP.ModalSourceData(data.grid, data.map, data.prefactor; kerr = signed)
        source = SP.ModalSource(execution, prepared; node_chunk = 2)
        output = similar(state)
        reference, scale = composition_reference(prepared, state, 0.2)
        source(output, state, 0.2)
        @test maximum(abs.(output - reference)) <= 8192eps(Float64) * scale
        @test composition_allocations(source, output, state, 0.2) == 0
        before = copy(state)
        huge = copy(state)
        huge .*= 1.0e150
        @test_throws SP.NonfiniteSourceError source(output, huge, 0.2)
        @test !source.ready
        @test isequal(state, before)
    end
end
