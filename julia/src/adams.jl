# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# SEP 16's resident exponential predictor-corrector. The Python march chooses steps
# and orders and supplies the scalar coefficient tables; the history and every
# state-sized operation stay here.

const RECIP_FACTORIAL = ntuple(k -> 1.0 / factorial(k - 1), 10)

const RECIP_INTEGER = ntuple(k -> 1.0 / k, 40)

# The tables in the wide type: binary64's are the engine's original constants, the
# double-float ones the same binary64 values split.
recip_factorial(::Type{Float64}, k) = @inbounds RECIP_FACTORIAL[k]
recip_integer(::Type{Float64}, k) = @inbounds RECIP_INTEGER[k]
const RECIP_FACTORIAL_DF = map(DF32, RECIP_FACTORIAL)
const RECIP_INTEGER_DF = map(DF32, RECIP_INTEGER)
recip_factorial(::Type{DF32}, k) = @inbounds RECIP_FACTORIAL_DF[k]
recip_integer(::Type{DF32}, k) = @inbounds RECIP_INTEGER_DF[k]

# The size that picks the branch and the halvings: binary64's `abs`, or the high
# parts' modulus in binary32 (the choice needs no more).
phi_size(x::ComplexF64) = abs(x)
phi_size(x::CDF32) = sqrt(real(x).hi * real(x).hi + imag(x).hi * imag(x).hi)
# The series' stop: binary64's squared test as it was; on pairs a one-norm test on the
# high parts, whose squares would be subnormal in binary32 for a small sum.
series_done(term::ComplexF64, total::ComplexF64, stop) = abs2(term) <= stop^2 * abs2(total)
series_done(term::CDF32, total::CDF32, stop) =
    abs(real(term).hi) + abs(imag(term).hi) <= stop * (abs(real(total).hi) + abs(imag(total).hi))
# `value 2^-k`, exactly: binary64's expression as it was, halvings on pairs.
scale_down(value::ComplexF64, k) = value * 0.5^k
scale_down(value::CDF32, k) = halve(value, k)
# k! phi_{k+1}: binary64 divides by the reciprocal, as it did; on pairs the factorial
# (exact in binary32 to 10!) multiplies.
moment_of(value::ComplexF64, k) = value / RECIP_FACTORIAL[k]
moment_of(value::CDF32, k) = value * Float32(factorial(k - 1))

"""`phi_0(x) .. phi_N(x)` by a shifted series and scaling and squaring.

Evaluated in the wide type at either precision (binary64, or a double-float where the
device lacks it): the flow and its moments are rounded once to the state's precision
(SEP 33). Native's `adams_phi` (adams_phi.hpp) is the model: past |x| = 3 the
recurrence from `exp(x)` with one reciprocal of x; below it the series of `phi_N(y)`,
`y = x / 2^s` with |y| <= 1/2, multiplied by the divisors' reciprocals and stopped
once a term is under a quarter ulp of the sum (each later one is smaller by
|y| / (m + N) <= 1/2), then raised back by `s` squarings. In binary64 the operations
are exactly the original ones.
"""
@inline function phi_values(x::Complex{W}, ::Val{N}) where {W,N}
    magnitude = phi_size(x)
    S = typeof(magnitude)
    out = ntuple(_ -> zero(Complex{W}), Val(N + 1))
    if magnitude > S(3)
        inverse = inv(x)
        out = Base.setindex(out, exp(x), 1)
        for k in 1:N
            out = Base.setindex(out, (out[k] - recip_factorial(W, k)) * inverse, k + 1)
        end
        return out
    end
    half = S(0.5)
    levels, scale = 0, one(S)
    reduced = magnitude
    while reduced > half
        levels += 1
        scale *= half
        reduced *= half
    end
    y = scale * x
    term = complex(recip_factorial(W, N + 1))
    total = term
    stop = S(0.25) * S(eps(W))
    for m in 1:23
        term = term * y * recip_integer(W, m + N)
        total += term
        series_done(term, total, stop) && break
    end
    out = Base.setindex(out, total, N + 1)
    for k in N-1:-1:1
        out = Base.setindex(out, muladd(y, out[k+2], recip_factorial(W, k + 1)), k + 1)
    end
    out = Base.setindex(out, exp(y), 1)
    for _ in 1:levels
        old = out
        out = Base.setindex(out, old[1] * old[1], 1)
        for k in 1:N
            value = old[1] * old[k+1]
            for j in 1:k
                value = muladd(old[j+1], recip_factorial(W, k - j + 1), value)
            end
            out = Base.setindex(out, scale_down(value, k), k + 1)
        end
    end
    out
end
phi_snapshot(x, order) = collect(phi_values(ComplexF64(x), Val(Int(order))))
# The double-float evaluation of the same, widened back for comparison (tests).
phi_snapshot_df(x, order) =
    [ComplexF64(Float64(real(v)), Float64(imag(v)))
     for v in phi_values(only(widen_host(DF32, [ComplexF64(x)])), Val(Int(order)))]

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
    tables_on_device = device_tables(c, tables)
    (store=dzeros(c, C, nw, nm, slots), states=held(nw, nm, slots), rates=held(nw, nm, slots),
     moments=dzeros(c, C, nw, nm, width), carried=held(nw, nm, max(width - 1, 0)),
     # The device's tables at the state's precision: a binary64 coefficient beside a
     # binary32 state would widen a trial's every product (and its norms' types).
     endpoint=held(nw, nm), host=tables,
     device=tables_on_device.device, packed=tables_on_device.packed,
     sizes=zeros(Int, 3), compensated=compensated)
end

"""The device's copies of the trial's tables: on a CPU an array each; on a GPU the real
tables views of one device vector (`packed`), refreshed by one copy a trial with the
integer slots beside it (`upload_tables!`), where nine small uploads a trial were nine
round trips through the copy engine."""
function device_tables(c, tables)
    if oncpu(c)
        device = map(a -> upload(c.backend, eltype(a) <: Integer ? copy(a) : c.T.(a)), tables)
        return (device=device, packed=nothing)
    end
    reals = Base.structdiff(tables, NamedTuple{(:slots,)})
    total = sum(length, values(reals))
    packed = dzeros(c, c.T, total)
    stops = cumsum(collect(map(length, values(reals))))
    views = ntuple(k -> reshape(view(packed, stops[k]-length(reals[k])+1:stops[k]), size(reals[k])),
                   length(reals))
    device = merge(NamedTuple{keys(reals)}(views), (slots=upload(c.backend, copy(tables.slots)),))
    (device=device, packed=(device=packed, host=zeros(c.T, total)))
end

"""Copy the host's trial tables to the device: each on a CPU, one packed copy and the
slots on a GPU."""
function upload_tables!(a)
    packed = hasproperty(a, :packed) ? a.packed : nothing
    if packed === nothing
        foreach(copyto!, a.device, a.host)
        return nothing
    end
    staging, offset = packed.host, 0
    for name in keys(a.host)
        name === :slots && continue
        table = getproperty(a.host, name)
        for k in eachindex(table)
            @inbounds staging[offset+k] = table[k]
        end
        offset += length(table)
    end
    tally!(:upload, 2)
    copyto!(packed.device, staging)
    copyto!(a.device.slots, a.host.slots)
    nothing
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
    upload_tables!(a)
    nothing
end
function configure_adams_rate!(engine, rate)
    a = engine.work.adams
    a.host.rate .= Vector{Float64}(rate)
    upload_tables!(a)
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
    pool_copy!(@view(a.store[:, :, 1]), w.source, c)
    if a.compensated
        pool_copy!(@view(a.states[:, :, 1]), w.y, c)
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
    # The step in the wide type, split on the host; a kernel body captures values only.
    step = wide_value(c, Float64(h))
    xzero = complex(zero(c.wide))
    mask = c.norm_mask
    # The predictor, the flow and its moments, and the state's norm for the scale.
    active = c.mask
    none, nil, h = zero(T), zero(C), T(step)
    # The state's squared norm, for the corrector's scale, stays where the corrector
    # reads it: on a GPU the device (`sum_device!`).
    squared = scratch(c, :adams_squared, T, 1)
    @section :adams_predict sum_device!(squared, c, size(y)...; heavy=true) do f, m
        @inbounds begin
            # Outside the band the state, the history and the source are exactly zero,
            # and so is everything this element forms.
            if !active[f]
                lab[f, m] = 0
                compensated && (endpoint[f, m] = 0; error[f, m] = 0)
                return none
            end
            x = step * linear[f, m]
            if phi !== nothing
                x = xzero
                for term in axes(phi, 3)
                    x += phi[f, m, term] * delta[term, 7]
                end
            end
            values = phi_values(x, Val(N))
            forward[f, m, 1] = values[1]
            predicted = nil
            for k in 1:N
                moment = oftype(nil, moment_of(values[k+1], k))
                moments[f, m, k] = moment
                if k <= order
                    weighted = nil
                    for j in 1:order
                        weighted = muladd(predictor[j, k], store[f, m, slots[j]], weighted)
                    end
                    predicted = muladd(moment, weighted, predicted)
                end
            end
            if compensated
                correction = nil
                for j in 1:N-1
                    slot = slots[j]
                    value = oftype(nil, exp(spans[j] * x)) * (rates[f, m, slot] - oftype(nil, x / step)) * states[f, m, slot]
                    carried[f, m, j] = value
                    j <= order && (correction = muladd(cpredictor[j], value, correction))
                end
                predicted += correction
                error[f, m] = adams_rate(linear, phi, weights, f, m)
            end
            value = muladd(h, predicted, oftype(nil, values[1]) * y[f, m])
            lab[f, m] = value
            compensated && (endpoint[f, m] = (error[f, m] - oftype(nil, x / step)) * value)
            mask[f] ? abs2(y[f, m]) : none
        end
    end
    w.path === nothing || material_stage!(engine, 7)
    source!(engine, lab; quadrature=true)
    # A GPU reads the quadrature alarm with the norms, for the commit to take.
    alarm = oncpu(c) ? nothing : alarm_arrays(engine)
    norms = @section :adams_correct corrector_norms(engine, Val(N), squared, h, alarm)
    alarm === nothing || hold_alarm!(engine, norms)
    [sqrt(Float64(norms[row])) for row in 1:count]
end

"""The corrected candidate and the defect norms' squares, one to three, and with an
`alarm` (`alarm_arrays`) its two sums after them, in one reduction: one value the host
waits for a trial."""
function corrector_norms(engine, ::Val{N}, squared, h, alarm) where {N}
    c, w = engine.config, engine.work
    a = w.adams
    order, _, count = a.sizes
    T = real_type(engine)
    C = Complex{T}
    (; y, forward, candidate, source) = w
    (; store, moments, carried, endpoint) = a
    (; corrector, defects, ccorrector, cdefects, slots) = a.device
    compensated = a.compensated
    active, mask, weight = c.mask, c.norm_mask, c.weights
    rtol, absolute = T(c.rtol), T(c.atol)
    none, nil = zero(T), zero(C)
    zero3 = (none, none, none)
    neutral = alarm === nothing ? zero3 : (zero3..., none, none)
    reduce_each(.+, neutral, c, size(y)...; heavy=true) do f, m
        @inbounds begin
            extra = alarm === nothing ? () : alarm_term(alarm, f, m)
            if !active[f]
                candidate[f, m] = 0
                return (zero3..., extra...)
            end
            corrected = nil
            for k in 1:N
                weighted = corrector[1, k] * source[f, m]
                for j in 2:N
                    weighted = muladd(corrector[j, k], store[f, m, slots[j-1]], weighted)
                end
                corrected = muladd(moments[f, m, k], weighted, corrected)
            end
            if compensated
                correction = ccorrector[1] * endpoint[f, m]
                for j in 1:order
                    correction = muladd(ccorrector[j+1], carried[f, m, j], correction)
                end
                corrected += correction
            end
            candidate[f, m] = muladd(h, corrected, forward[f, m, 1] * y[f, m])
            result = zero3
            if mask[f]
                # rtol |y|, formed as the host formed it from the predictor's sum.
                relative = rtol * sqrt(squared[1])
                scale = relative * weight[f] + absolute
                for row in 1:count
                    difference = defects[row, 1] * source[f, m]
                    for j in 2:N
                        difference += defects[row, j] * store[f, m, slots[j-1]]
                    end
                    polynomial = nil
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
                    defect *= h
                    score = scale == 0 && defect == 0 ? none : abs2(defect / scale)
                    result = Base.setindex(result, score, row)
                end
            end
            (result..., extra...)
        end
    end
end

function adams_accept!(engine, slot)
    w = engine.work
    a = w.adams
    pool_copy!(@view(a.store[:, :, slot+1]), w.source, engine.config)
    if a.compensated
        pool_copy!(@view(a.states[:, :, slot+1]), w.lab, engine.config)
        pool_copy!(@view(a.rates[:, :, slot+1]), w.error, engine.config)
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
adams_start_b(args...) = flushed(adams_start!, args...)
adams_trial_b(args...) = flushed(adams_trial!, args...)
adams_accept_b(args...) = flushed(adams_accept!, args...)
const adams_load_rows_b = adams_load_rows!
