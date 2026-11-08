# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Longitudinal expansions. The translation owns the expansion and its scalar
# evaluation: only fixed term arrays and each stage's weights cross, never a pressure
# law or a material formula.

function path_workspace(c, host)
    host.has("phi_terms") || return nothing
    C = Complex{c.T}
    # Phi's terms stay in the wide type, as the operator does (modal.jl).
    phi, pre = widen_host(wide_eltype(c), host.array("phi_terms", ComplexF64)), host.array("prefactor_terms", C)
    stages = 7
    (phi=upload(c, phi), prefactor=upload(c, pre),
     delta=zeros(size(phi, 3), stages), weights=zeros(size(pre, 3), stages),
     delta_device=dzeros(c, wide_eltype(c), size(phi, 3), stages),
     weights_device=dzeros(c, c.T, size(pre, 3), stages),
     # Their host staging, kept: a converted copy a trial was two fresh arrays. (Pageable
     # on purpose: the driver copies it at once, so the next trial may rewrite it.)
     delta_staging=fill(c.wide, size(phi, 3), stages),
     weights_staging=zeros(c.T, size(pre, 3), stages),
     # Kerr, the first channel's density and preionisation, the taper, then each later
     # channel's density and preionisation.
     material=[zeros(3, stages); ones(1, stages);
               zeros(2 * ((host.has("plasma_count") ? host.int("plasma_count") : 1) - 1), stages)],
     tables=Dict{String,Any}(), constants=zeros(4),
     # A varying Raman kernel's decay rates (a term a row) and component densities.
     raman_rates=zeros(host.flag("raman_varying") ? length(host.array("raman_component", Int)) : 0, stages),
     raman_densities=zeros(host.flag("raman_varying") ? length(host.array("raman_fractions", Float64)) : 0, stages))
end

"""Take a trial's stage weights: Phi increments, prefactor weights and material scales."""
function configure_stages!(engine, delta, weights, material)
    p = engine.work.path
    copyto!(p.delta, Matrix{Float64}(delta))
    copyto!(p.weights, Matrix{Float64}(weights))
    copyto!(p.material, Matrix{Float64}(material))
    if p.delta_device isa Array
        copyto!(p.delta_device, widen_host(wide_eltype(engine.config), p.delta))
        copyto!(p.weights_device, real_type(engine).(p.weights))
    else
        tally!(:upload, 2)
        W, T = wide_eltype(engine.config), real_type(engine)
        p.delta_staging .= W.(p.delta)
        p.weights_staging .= T.(p.weights)
        copyto!(p.delta_device, p.delta_staging)
        copyto!(p.weights_device, p.weights_staging)
    end
    material_stage!(engine, 1)
    nothing
end

"""Make a stage's prefactor, Kerr factor and plasma density current."""
function material_stage!(engine, stage)
    c, w = engine.config, engine.work
    p = w.path
    (; prefactor) = c
    terms, weights = p.prefactor, p.weights_device
    each!(c, size(prefactor)...) do f, m
        value = zero(eltype(prefactor))
        @inbounds for term in axes(terms, 3)
            value += terms[f, m, term] * weights[term, stage]
        end
        @inbounds prefactor[f, m] = value
    end
    engine.factor = p.material[1, stage]
    engine.taper = p.material[4, stage]
    r = w.raman
    if r !== nothing && r.varying && !isempty(p.raman_rates)
        rebuild_raman!(r, view(p.raman_rates, :, stage), view(p.raman_densities, :, stage), c)
    end
    if w.plasma !== nothing
        w.plasma.parameters[8] = p.material[2, stage]
        w.plasma.parameters[9] = p.material[3, stage]
        for k in 2:length(w.plasma.channels)
            parameters = w.plasma.channels[k].parameters
            parameters[8], parameters[9] = p.material[3+2(k-1), stage], p.material[4+2(k-1), stage]
        end
    end
end

"""Every stage's propagator pair from the expansion: `exp(+-sum_k phi_k delta_k)`."""
function path_propagators!(engine)
    c, w = engine.config, engine.work
    p = w.path
    (; forward, inverse) = w
    phi, delta = p.phi, p.delta_device
    each!(c, size(w.y)...) do f, m
        @inbounds for stage in 1:7
            exponent = zero(eltype(phi))
            for term in axes(phi, 3)
                exponent += phi[f, m, term] * delta[term, stage]
            end
            forward[f, m, stage], inverse[f, m, stage] = wide_exp_pair(exponent)
        end
    end
end

# --- the stage scalars, read by the engine (SEP 39's staged tables) ------------------

"""Hold one of the stage's scalar tables as `PiecewiseChebyshev.packed` gives it:
`phi` and `prefactor` (the expansions' weights), `kerr`, `plasma`, `preionisation`
(densities). A name not bound is the constant case `configure_stages_at!` names."""
bind_stage_table!(engine, name, edges, lower, upper, c) =
    (engine.work.path.tables[String(name)] =
         (edges=Vector{Float64}(edges), lower=Float64(lower), upper=Float64(upper),
          c=Array{Float64,3}(c)); nothing)

"""Hold the stage's constants: the Kerr coefficient (0 for none), the state scale's
square, and a constant prefactor's span."""
bind_stage_constants!(engine, kerr, scale2, low, high) =
    (engine.work.path.constants .= (kerr, scale2, low, high); nothing)

"""A bound table's values at the positions, (n_positions, n_values)."""
function stage_values(table, positions)
    out = zeros(length(positions), size(table.c, 3))
    bad = clenshaw_points!(positions, table.edges, table.lower, table.upper, table.c, out)
    bad >= 0 && throw(ArgumentError(
        "z = $(positions[bad+1]) is outside the represented span " *
        "[$(table.edges[1]), $(table.edges[end])]"))
    out
end

"""A trial's stage scalars from the bound tables, then `configure_stages!`.

`translate.stage.scalars_at`'s values for the fields the engine reads, by the same
operations: Phi's increment from `z0` (the distance, for a constant operator), the
prefactor's weights (one, for a constant prefactor, inside its span), and the Kerr
factor, plasma density and preionisation of each position.
"""
function configure_stages_at!(engine, z0, positions)
    path = engine.work.path
    tables, (kerr, scale2, low, high) = path.tables, path.constants
    zs = Vector{Float64}(positions)
    n = length(zs)
    delta = if haskey(tables, "phi")
        here = stage_values(tables["phi"], zs)
        permutedims(here .- stage_values(tables["phi"], [Float64(z0)]))
    else
        reshape([z1 - z0 for z1 in zs], 1, n)
    end
    weights = if haskey(tables, "prefactor")
        permutedims(stage_values(tables["prefactor"], zs))
    else
        for z in zs
            low <= z <= high || throw(ArgumentError(
                "z = $z m lies outside the table, which covers [$low, $high] m"))
        end
        ones(1, n)
    end
    column(name) = haskey(tables, name) ? stage_values(tables[name], zs)[:, 1] : zeros(n)
    density = column("kerr")
    taper = haskey(tables, "taper") ? stage_values(tables["taper"], zs)[:, 1] : ones(n)
    later = [column(name * "_$k") for k in 2:(size(path.material, 1) - 2) ÷ 2
             for name in ("plasma", "preionisation")]
    material = permutedims(hcat([kerr * d * scale2 for d in density],
                                column("plasma"), column("preionisation"), taper, later...))
    # A varying Raman kernel's rates and densities, before the stages are configured
    # (whose first stage then rebuilds it).
    for k in axes(path.raman_rates, 1)
        path.raman_rates[k, 1:n] .= column("raman_rate_$k")
    end
    for j in axes(path.raman_densities, 1)
        path.raman_densities[j, 1:n] .= column("raman_density_$j")
    end
    configure_stages!(engine, delta, weights, material)
end

const bind_stage_table_b = bind_stage_table!
const bind_stage_constants_b = bind_stage_constants!
const configure_stages_at_b = configure_stages_at!
