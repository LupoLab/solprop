# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The translation layer owns the expansion and its scalar evaluation. No pressure
# law or material formula crosses this boundary: only fixed arrays and weights.
function path_workspace(input,c)
    haskey(input,"phi_terms") || return nothing
    phi=Array{ComplexF64,3}(input["phi_terms"])
    pre=Array{ComplexF64,3}(input["prefactor_terms"])
    (phi=phi,prefactor=pre,delta=zeros(size(phi,3),length(c.c)),
     weights=zeros(size(pre,3),length(c.c)),material=zeros(3,length(c.c)))
end
function configure_stages!(engine,delta,weights,material)
    p=engine.work.path
    copyto!(p.delta,Matrix{Float64}(delta))
    copyto!(p.weights,Matrix{Float64}(weights))
    copyto!(p.material,Matrix{Float64}(material))
    material_stage!(engine,1)
    nothing
end
function material_stage!(engine,stage)
    c,w=engine.config,engine.work
    p=w.path
    foreach_sample(c.prefactor,c.tasks) do f,m,i
        value=0.0im
        @inbounds for term in axes(p.prefactor,3)
            value += p.prefactor[f,m,term]*p.weights[term,stage]
        end
        @inbounds c.prefactor[i]=value
    end
    engine.factor=p.material[1,stage]
    if !isnothing(w.plasma)
        w.plasma.parameters[8]=p.material[2,stage]
        w.plasma.parameters[9]=p.material[3,stage]
    end
end
function path_propagators!(engine)
    c,w=engine.config,engine.work
    p=w.path
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds for stage in eachindex(c.c)
            exponent=0.0im
            for term in axes(p.phi,3)
                exponent += p.phi[f,m,term]*p.delta[term,stage]
            end
            w.forward[f,m,stage]=exp(exponent)
            w.inverse[f,m,stage]=exp(-exponent)
        end
    end
end
