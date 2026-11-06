# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Longitudinal expansions. The translation owns the expansion and its scalar
# evaluation: only fixed term arrays and each stage's weights cross, never a pressure
# law or a material formula.

function path_workspace(c, host)
    host.has("phi_terms") || return nothing
    C = Complex{c.T}
    phi, pre = host.array("phi_terms", C), host.array("prefactor_terms", C)
    stages = 7
    (phi=upload(c, phi), prefactor=upload(c, pre),
     delta=zeros(size(phi, 3), stages), weights=zeros(size(pre, 3), stages),
     delta_device=dzeros(c, c.T, size(phi, 3), stages),
     weights_device=dzeros(c, c.T, size(pre, 3), stages), material=zeros(3, stages))
end

"""Take a trial's stage weights: Phi increments, prefactor weights and material scales."""
function configure_stages!(engine, delta, weights, material)
    p = engine.work.path
    copyto!(p.delta, Matrix{Float64}(delta))
    copyto!(p.weights, Matrix{Float64}(weights))
    copyto!(p.material, Matrix{Float64}(material))
    copyto!(p.delta_device, real_type(engine).(p.delta))
    copyto!(p.weights_device, real_type(engine).(p.weights))
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
    if w.plasma !== nothing
        w.plasma.parameters[8] = p.material[2, stage]
        w.plasma.parameters[9] = p.material[3, stage]
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
            forward[f, m, stage] = exp(exponent)
            inverse[f, m, stage] = exp(-exponent)
        end
    end
end
