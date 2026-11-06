# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# SEP 16's resident exponential predictor-corrector. The Python march chooses steps
# and orders and supplies the scalar coefficient tables; the history and every
# state-sized operation stay here.

const RECIP_FACTORIAL = ntuple(k -> 1.0 / factorial(k - 1), 10)

"""`phi_0(x) .. phi_N(x)` by a shifted series and scaling and squaring.

Evaluated in binary64 at either precision: the flow and its moments are rounded once
to the state's precision (SEP 33).
"""
@inline function phi_values(x::ComplexF64, ::Val{N}) where {N}
    if abs(x) > 3.0
        out = ntuple(_ -> zero(ComplexF64), Val(N + 1))
        out = Base.setindex(out, exp(x), 1)
        for k in 1:N
            out = Base.setindex(out, (out[k] - RECIP_FACTORIAL[k]) / x, k + 1)
        end
        return out
    end
    levels = abs(x) <= 0.5 ? 0 : ceil(Int, log2(abs(x) / 0.5))
    y = x / 2.0^levels
    term = complex(RECIP_FACTORIAL[N+1])
    total = term
    for m in 1:23
        term = term * y / (m + N)
        total += term
    end
    out = ntuple(_ -> zero(ComplexF64), Val(N + 1))
    out = Base.setindex(out, total, N + 1)
    for k in N-1:-1:1
        out = Base.setindex(out, RECIP_FACTORIAL[k+1] + y * out[k+2], k + 1)
    end
    out = Base.setindex(out, exp(y), 1)
    for _ in 1:levels
        old = out
        out = Base.setindex(out, old[1] * old[1], 1)
        for k in 1:N
            value = old[1] * old[k+1]
            for j in 1:k
                value += old[j+1] * RECIP_FACTORIAL[k-j+1]
            end
            out = Base.setindex(out, value / 2.0^k, k + 1)
        end
    end
    out
end
phi_snapshot(x, order) = collect(phi_values(ComplexF64(x), Val(Int(order))))

function adams_workspace(c, host, nw, nm)
    order = host.has("adams_order") ? host.int("adams_order") : 0
    order == 0 && return nothing
    C = Complex{c.T}
    slots, width = order + 2, order + 1
    compensated = host.flag("adams_compensated")
    held(dims...) = compensated ? dzeros(c, C, dims...) : dzeros(c, C, ntuple(_ -> 0, length(dims))...)
    nphi = host.has("phi_terms") ? size(host.array("phi_terms", ComplexF64), 3) : 1
    # The trial's coefficient tables, filled on the host and copied over once a trial.
    tables = (predictor=zeros(width, width), corrector=zeros(width, width),
              defects=zeros(3, 2width), spans=zeros(width), cpredictor=zeros(width),
              ccorrector=zeros(width + 1), cdefects=zeros(3, width), rate=zeros(nphi),
              slots=zeros(Int, slots))
    (store=dzeros(c, C, nw, nm, slots), states=held(nw, nm, slots), rates=held(nw, nm, slots),
     moments=dzeros(c, C, nw, nm, width), carried=held(nw, nm, max(width - 1, 0)),
     endpoint=held(nw, nm), host=tables, device=map(a -> upload(c.backend, copy(a)), tables),
     sizes=zeros(Int, 3), compensated=compensated)
end

function configure_adams!(engine, predictor, corrector, defects, slots, spans, cpredictor,
                          ccorrector, cdefects, rate)
    a = engine.work.adams
    h = a.host
    p, q, d = Matrix{Float64}(predictor), Matrix{Float64}(corrector), Matrix{Float64}(defects)
    order, width, count = size(p, 1), size(q, 1), size(d, 1)
    a.sizes .= (order, width, count)
    h.predictor[1:order, 1:order] .= p
    h.corrector[1:width, 1:width] .= q
    h.defects[1:count, 1:2width] .= d
    incoming = Vector{Int}(slots)
    h.slots[1:length(incoming)] .= incoming .+ 1
    if a.compensated
        s, cp, cc = Vector{Float64}(spans), Vector{Float64}(cpredictor), Vector{Float64}(ccorrector)
        h.spans[1:length(s)] .= s
        h.cpredictor[1:length(cp)] .= cp
        h.ccorrector[1:length(cc)] .= cc
        h.cdefects[1:count, 1:width] .= Matrix{Float64}(cdefects)
        h.rate .= Vector{Float64}(rate)
    end
    foreach(copyto!, a.device, a.host)
    nothing
end
function configure_adams_rate!(engine, rate)
    a = engine.work.adams
    a.host.rate .= Vector{Float64}(rate)
    copyto!(a.device.rate, a.host.rate)
    nothing
end

"""`D(z)` at one element: the constant operator, or its expansion's weighted terms."""
@inline function adams_rate(linear, phi, weights, f, m)
    phi === nothing && return @inbounds linear[f, m]
    value = zero(eltype(phi))
    @inbounds for term in axes(phi, 3)
        value += phi[f, m, term] * weights[term]
    end
    value
end

path_terms(w) = w.path === nothing ? nothing : w.path.phi

function adams_start!(engine)
    c, w = engine.config, engine.work
    a = w.adams
    w.path === nothing || material_stage!(engine, 1)
    source!(engine, w.y)
    copyto!(@view(a.store[:, :, 1]), w.source)
    if a.compensated
        copyto!(@view(a.states[:, :, 1]), w.y)
        (; rates) = a
        linear, phi, weights = c.linear, path_terms(w), a.device.rate
        each!(c, size(w.y)...) do f, m
            @inbounds rates[f, m, 1] = adams_rate(linear, phi, weights, f, m)
        end
    end
    nothing
end

adams_trial!(engine, h) = adams_trial_width!(engine, h, Val(engine.work.adams.sizes[2]))

function adams_trial_width!(engine, h, ::Val{N}) where {N}
    c, w = engine.config, engine.work
    a = w.adams
    order, _, count = a.sizes
    T = real_type(engine)
    C = Complex{T}
    (; y, lab, error, forward, candidate, source) = w
    (; store, states, rates, moments, carried, endpoint) = a
    (; predictor, corrector, defects, spans, cpredictor, ccorrector, cdefects, slots) = a.device
    linear, phi, weights, compensated = c.linear, path_terms(w), a.device.rate, a.compensated
    delta = w.path === nothing ? nothing : w.path.delta_device
    step = Float64(h)
    mask = c.norm_mask
    # The predictor, the flow and its moments, and the state's norm for the scale.
    active = c.mask
    squared = @section :adams_predict reduce_each(+, zero(T), c, size(y)...; heavy=true) do f, m
        @inbounds begin
            # Outside the band the state, the history and the source are exactly zero,
            # and so is everything this element forms.
            if !active[f]
                lab[f, m] = 0
                compensated && (endpoint[f, m] = 0; error[f, m] = 0)
                return zero(T)
            end
            x = ComplexF64(step * linear[f, m])
            if phi !== nothing
                x = zero(ComplexF64)
                for term in axes(phi, 3)
                    x += phi[f, m, term] * delta[term, 7]
                end
            end
            values = phi_values(x, Val(N))
            forward[f, m, 1] = values[1]
            predicted = zero(C)
            for k in 1:N
                moment = C(values[k+1] / RECIP_FACTORIAL[k])
                moments[f, m, k] = moment
                if k <= order
                    weighted = zero(C)
                    for j in 1:order
                        weighted += predictor[j, k] * store[f, m, slots[j]]
                    end
                    predicted += moment * weighted
                end
            end
            if compensated
                correction = zero(C)
                for j in 1:N-1
                    slot = slots[j]
                    value = C(exp(spans[j] * x)) * (rates[f, m, slot] - C(x / step)) * states[f, m, slot]
                    carried[f, m, j] = value
                    j <= order && (correction += cpredictor[j] * value)
                end
                predicted += correction
                error[f, m] = adams_rate(linear, phi, weights, f, m)
            end
            value = C(values[1]) * y[f, m] + T(step) * predicted
            lab[f, m] = value
            compensated && (endpoint[f, m] = (error[f, m] - C(x / step)) * value)
            mask[f] ? abs2(y[f, m]) : zero(T)
        end
    end
    w.path === nothing || material_stage!(engine, 7)
    source!(engine, lab; quadrature=true)
    magnitude = sqrt(squared)
    relative, absolute, weight = T(c.rtol) * magnitude, T(c.atol), c.weights
    zero3 = (zero(T), zero(T), zero(T))
    norms = @section :adams_correct reduce_each(.+, zero3, c, size(y)...; heavy=true) do f, m
        @inbounds begin
            if !active[f]
                candidate[f, m] = 0
                return zero3
            end
            corrected = zero(C)
            for k in 1:N
                weighted = corrector[1, k] * source[f, m]
                for j in 2:N
                    weighted += corrector[j, k] * store[f, m, slots[j-1]]
                end
                corrected += moments[f, m, k] * weighted
            end
            if compensated
                correction = ccorrector[1] * endpoint[f, m]
                for j in 1:order
                    correction += ccorrector[j+1] * carried[f, m, j]
                end
                corrected += correction
            end
            candidate[f, m] = forward[f, m, 1] * y[f, m] + T(step) * corrected
            result = zero3
            if mask[f]
                scale = relative * weight[f] + absolute
                for row in 1:count
                    difference = defects[row, 1] * source[f, m]
                    for j in 2:N
                        difference += defects[row, j] * store[f, m, slots[j-1]]
                    end
                    polynomial = zero(C)
                    for k in 1:N
                        polynomial += defects[row, N+k] * moments[f, m, k]
                    end
                    defect = difference * polynomial
                    if compensated
                        correction = cdefects[row, 1] * endpoint[f, m]
                        for j in 1:N-1
                            correction += cdefects[row, j+1] * carried[f, m, j]
                        end
                        defect += correction
                    end
                    defect *= T(step)
                    score = scale == 0 && defect == 0 ? zero(T) : abs2(defect / scale)
                    result = Base.setindex(result, score, row)
                end
            end
            result
        end
    end
    [sqrt(Float64(norms[row])) for row in 1:count]
end

function adams_accept!(engine, slot)
    w = engine.work
    a = w.adams
    copyto!(@view(a.store[:, :, slot+1]), w.source)
    if a.compensated
        copyto!(@view(a.states[:, :, slot+1]), w.lab)
        copyto!(@view(a.rates[:, :, slot+1]), w.error)
    end
    nothing
end
adams_rows(engine, name) = Array{ComplexF64}(Array(getproperty(engine.work.adams, Symbol(name))))
function adams_load_rows!(engine, name, rows)
    target = getproperty(engine.work.adams, Symbol(name))
    copyto!(target, upload(engine.config.backend, Array{eltype(target),3}(rows)))
    nothing
end
const configure_adams_b = configure_adams!
const configure_adams_rate_b = configure_adams_rate!
const adams_start_b = adams_start!
const adams_trial_b = adams_trial!
const adams_accept_b = adams_accept!
const adams_load_rows_b = adams_load_rows!
