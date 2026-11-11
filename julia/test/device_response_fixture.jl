# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The existing host fixture supplies independent direct-DFT, causal-convolution and
# analytic W(E)=kE² references. Include it once in standalone device qualification.
isdefined(@__MODULE__, :composition_case) || include("composition.jl")

function device_response_data(execution, host)
    grid, maps, prefactor = host.grid, host.map, host.prefactor
    owned_grid = SP.ModalSourceGrid(
        execution; kind = grid.kind, n_time = grid.n_time, dt = grid.dt,
        crop_index = grid.crop_index, omega = grid.omega, active = grid.active,
        omega_window = grid.omega_window, source_window = grid.source_window,
        source_window_derivative = grid.source_window_derivative,
        temporal_rate = grid.temporal_rate, carrier = grid.carrier,
    )
    owned_map = SP.ModalSourceMap(
        execution; synthesis = maps.synthesis, projection = maps.projection,
        coarse_projection = maps.coarse_projection, taper = maps.taper,
    )
    owned_prefactor = if prefactor.weights !== nothing
        SP.SourcePrefactor(execution, prefactor.data, prefactor.weights)
    elseif prefactor.data isa SP.ChebyshevTable
        SP.SourcePrefactor(execution, prefactor.data, prefactor.shape)
    else
        SP.SourcePrefactor(execution, prefactor.data)
    end
    raman = map(host.raman) do component
        SP.ModalRaman(
            component.kernel, component.decay_profiles, component.density;
            fraction = component.fraction, material_factor = component.material_factor,
            thg = component.thg,
        )
    end
    plasma = if host.plasma === nothing
        nothing
    else
        channels = map(host.plasma.channels) do channel
            SP.ModalPlasmaChannel(
                execution, channel.channel, channel.density, channel.preionisation;
                quadrature = channel.quadrature,
            )
        end
        grid.kind == :field ? SP.ModalPlasma(execution, channels, host.plasma.rule) :
            SP.ModalPlasma(
                execution, channels; drude = host.plasma.drude, loss = host.plasma.loss
            )
    end
    return SP.ModalSourceData(
        execution, owned_grid, owned_map, owned_prefactor; kerr = host.kerr, raman, plasma
    )
end

function device_response_host(array, execution)
    SP.oncpu(execution) && return copy(array)
    host = Array{eltype(array)}(undef, size(array))
    SP.copy_host!(host, array, execution)
    return host
end

function device_response_profile_bound(profile)
    profile.value isa Float64 && return abs(profile.value)
    table = profile.value
    # All fixture profiles are affine on each independent represented interval.
    @test size(table.coefficients, 2) <= 2
    return maximum(
        abs(composition_value(profile, position)) for position in table.breakpoints
    )
end

function device_response_lipschitz(data, radius)
    grid, maps = data.grid, data.map
    field = grid.kind == :field
    weights = field ? [1.0; fill(2.0, grid.n_time ÷ 2 - 1); 1.0] : ones(grid.n_time)
    inverse = sum(weights[grid.crop_index .+ 1]) / grid.n_time
    taper_max = device_response_profile_bound(maps.taper)
    taper_min = minimum(composition_value(maps.taper, z) for z in (0.0, 1.0))
    synthesis = maximum(sum(abs.(maps.synthesis); dims = 2))
    node_gain = inverse * synthesis * taper_max
    node_radius = node_gain * radius
    projection = maximum(sum(abs.(maps.projection); dims = 2)) / taper_min
    prefactor = maximum(
        maximum(abs, composition_prefactor(data.prefactor, z) .* grid.omega_window)
            for z in (0.0, 1.0)
    )
    duration = (grid.n_time - 1) * grid.dt
    response = 0.0
    if data.kerr !== nothing
        coefficients = data.kerr.coefficients
        algebra = coefficients isa SP.KerrCoefficients{Float64, :real} ? 1.0 :
            abs(coefficients.self_factor) + abs(coefficients.conjugate_factor) +
            abs(coefficients.harmonic_factor)
        response += 3 * abs(coefficients.coefficient) * algebra * node_radius^2 *
            device_response_profile_bound(data.kerr.density)
    end
    for component in data.raman
        kernel = grid.dt * component.fraction * sum(abs, component.kernel.undamped)
        drive = field && component.thg ? 1.0 : 0.5
        response += 3 * component.material_factor *
            device_response_profile_bound(component.density) *
            drive * kernel * node_radius^2
    end
    material = maximum(abs, grid.source_window) * response
    if data.plasma !== nothing
        for component in data.plasma.channels
            channel = component.channel
            rate = channel.rate.rate_max / channel.rate.field_max^2
            population_gain = 1 + 2rate * duration * node_radius^2
            density = device_response_profile_bound(component.density)
            if field
                current = density * population_gain * (
                    channel.charge_mass_ratio * duration +
                        channel.ionisation_potential * rate
                )
                material += current * (
                    maximum(abs, grid.source_window) * maximum(inv, grid.omega) +
                        maximum(abs, grid.source_window_derivative) *
                        maximum(value -> inv(value^2), grid.omega)
                )
            else
                material += density * population_gain * maximum(abs, grid.source_window) * (
                    channel.charge_mass_ratio * maximum(abs, data.plasma.drude) +
                        channel.ionisation_potential * rate * maximum(abs, data.plasma.loss)
                )
            end
        end
    end
    value = grid.n_time * projection * prefactor * material * node_gain
    if grid.temporal_rate !== nothing
        value += grid.n_time * maximum(grid.temporal_rate) / 2 * inverse
    end
    # Outward margin for the short positive preparation graph above (under 512 ops).
    return nextfloat(value * (1 + 512eps(Float64)))
end

function device_response_exact_steps(initial, data, steps)
    state = copy(initial)
    radius, h, error = 1.0, 1 / 128, 0.0
    lipschitz = device_response_lipschitz(data, radius)
    @test h * lipschitz <= 1 / 4
    magnitude = lipschitz * radius
    gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
    update_gamma = 256eps(Float64) / (1 - 256eps(Float64))
    for step in 0:(steps - 1)
        z = 0.53 + step * h
        @test maximum(abs, state) + error < radius
        first, first_scale = composition_reference(data, state, z)
        @test device_response_rate_domain(data, state, z, error) < gamma * first_scale / 16
        stage = state + h * first
        @test first_scale <= magnitude
        formation = update_gamma * (radius + h * magnitude)
        first_error = 2gamma * magnitude
        stage_error = (1 + h * lipschitz) * error + h * first_error + formation
        @test maximum(abs, stage) + stage_error < radius
        second, second_scale = composition_reference(data, stage, z + h)
        @test device_response_rate_domain(data, stage, z + h, stage_error) <
            gamma * second_scale / 16
        # Both source evaluations include their direct-DFT reference error. The
        # first error is transported into the second through the stated L bound.
        @test second_scale <= magnitude
        delta = 2gamma * magnitude
        state += (h / 2) * (first + second)
        error = exp(2h * lipschitz) * error + 2formation + h * (2 + h * lipschitz) * delta
        @test maximum(abs, state) + error < radius
    end
    @test 0 < error < 1.0e-7
    return state, error
end

function device_response_rate_domain(data, state, position, error = 0.0)
    data.plasma === nothing && return 0.0
    grid, maps = data.grid, data.map
    field = grid.kind == :field
    fine_count = field ? grid.n_time ÷ 2 + 1 : grid.n_time
    padded = zeros(ComplexF64, fine_count, size(state, 2))
    padded[grid.crop_index .+ 1, :] .= state
    modal = composition_inverse(padded, grid.n_time; real_signal = field)
    taper = composition_value(maps.taper, position)
    nodes = (modal * transpose(maps.synthesis)) .* taper
    gain = 2length(grid.crop_index) / grid.n_time *
        maximum(sum(abs.(maps.synthesis); dims = 2)) * taper
    margin = gain * (error + 16384eps(Float64))
    lower, upper = minimum(abs, nodes) - margin, maximum(abs, nodes) + margin
    cutoff = 0.0
    for component in data.plasma.channels
        channel = component.channel
        @test lower > channel.rate.field_min
        @test upper < channel.rate.field_max
        if !field
            rate_scale = channel.rate.rate_max / channel.rate.field_max^2
            lost_rate = rate_scale * channel.rate.field_min^2
            exposure = (grid.n_time - 1) * grid.dt * lost_rate
            density = composition_value(component.density, position)
            cutoff += density * (
                maximum(abs, data.plasma.drude) *
                    channel.charge_mass_ratio * upper * exposure +
                    maximum(abs, data.plasma.loss) * channel.ionisation_potential *
                    (2lost_rate / lower + rate_scale * upper * exposure)
            )
        end
    end
    return 2cutoff * grid.n_time * maximum(abs, grid.source_window) *
        maximum(sum(abs.(maps.projection); dims = 2)) / taper *
        maximum(abs, composition_prefactor(data.prefactor, position) .* grid.omega_window)
end

function device_response_steps!(state, source, flow, work, tableau, execution, steps)
    h = 1 / 128
    for step in 0:(steps - 1)
        SP.rk_trial!(work, state, 0.53 + step * h, h, tableau, source, flow, execution)
        SP.rk_commit!(state, work, source.data.grid.active, execution)
    end
    return nothing
end
