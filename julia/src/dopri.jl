# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The interaction-picture DOPRI5 trial and the accepted step's commit.

"""Cache `exp(+-c_s dz L)` per stage: exact reuse while dz is unchanged."""
function propagators!(engine, dz)
    c, w = engine.config, engine.work
    c.free && return nothing
    if w.path !== nothing
        path_propagators!(engine)
        return nothing
    end
    c.cache_propagators && engine.propagator_dz == dz && return nothing
    (; linear) = c
    (; forward, inverse) = w
    # Formed on the host in binary64, then split where the wide type is a pair.
    steps = ntuple(s -> wide_value(c, dz * c.nodes_c[s]), Val(7))
    each!(c, size(linear)...) do f, m
        @inbounds for s in 1:7
            forward[f, m, s], inverse[f, m, s] = wide_exp_pair(steps[s] * linear[f, m])
        end
    end
    engine.propagator_dz = dz
    nothing
end

"""Form one IP-DOPRI5 candidate and its error; the accepted state is untouched.

Returns the scaled error norm. With a valid first-same-as-last sample the first stage
is the previous step's last source, which `commit!` left in stage one's slot.
"""
function trial!(engine, dz::Float64)
    c, w = engine.config, engine.work
    T = real_type(engine)
    (; y, lab, stages, forward, inverse, source) = w
    @section :propagators propagators!(engine, dz)
    for stage in 1:7
        stage == 1 && engine.fsal_valid && continue
        w.path !== nothing && material_stage!(engine, stage)
        if stage == 1
            # c_1 = 0: both propagators are exactly the identity.
            if !engine.fsal_valid
                source!(engine, y)
                pool_copy!(@view(stages[:, :, 1]), source, c)
            end
            continue
        end
        slot = c.free ? free_propagators!(engine, dz * c.nodes_c[stage]) : stage
        row = ntuple(p -> T(dz * c.a[stage, p]), Val(6))
        each!(c, size(y)...) do f, m
            @inbounds begin
                combination = zero(eltype(y))
                for p in 1:stage-1
                    combination = muladd(row[p], stages[f, m, p], combination)
                end
                lab[f, m] = forward[f, m, slot] * (y[f, m] + combination)
            end
        end
        @section :source source!(engine, lab; quadrature=stage == 7)
        each!(c, size(y)...) do f, m
            @inbounds stages[f, m, stage] = inverse[f, m, slot] * source[f, m]
        end
    end
    engine.fsal_valid = true
    @section :combine combine!(engine, dz, c.free ? 1 : 7)
end

"""The candidate and the error estimate, and the controller's norm of the error.

The norm is `‖e‖ / (rtol w max(‖y‖, ‖y_new‖) + atol)` over the in-band bins, as the
shared controller defines it; with uniform weights all three sums are one pass.
"""
function combine!(engine, dz, slot)
    c, w = engine.config, engine.work
    T = real_type(engine)
    (; y, candidate, error, stages, forward) = w
    mask = c.norm_mask
    b = ntuple(s -> T(dz * c.b[s]), Val(7))
    e = ntuple(s -> T(dz * c.e[s]), Val(7))
    # A GPU reads the quadrature alarm with the norms, for the commit to take.
    alarm = oncpu(c) ? nothing : alarm_arrays(engine)
    sums = combine_sums(y, candidate, error, stages, forward, mask, b, e, slot, alarm, c, T)
    alarm === nothing || hold_alarm!(engine, sums)
    magnitude = max(sqrt(sums[1]), sqrt(sums[2]))
    if c.uniform
        scale = T(c.rtol) * magnitude + T(c.atol)
        return scale == 0 ? (sums[3] == 0 ? 0.0 : Inf) : Float64(sqrt(sums[3]) / scale)
    end
    weighted_norm(engine, error, magnitude)
end

function combine_sums(y, candidate, error, stages, forward, mask, b, e, slot, alarm, c, T)
    zero3 = (zero(T), zero(T), zero(T))
    neutral = alarm === nothing ? zero3 : (zero3..., zero(T), zero(T))
    reduce_each(.+, neutral, c, size(y)...) do f, m
        @inbounds begin
            extra = alarm === nothing ? () : alarm_term(alarm, f, m)
            solution, defect = zero(eltype(y)), zero(eltype(y))
            for s in 1:7
                value = stages[f, m, s]
                solution = muladd(b[s], value, solution)
                defect = muladd(e[s], value, defect)
            end
            flow = forward[f, m, slot]
            new = flow * (y[f, m] + solution)
            estimate = flow * defect
            candidate[f, m] = new
            error[f, m] = estimate
            ((mask[f] ? (abs2(y[f, m]), abs2(new), abs2(estimate)) : zero3)..., extra...)
        end
    end
end

function weighted_norm(engine, error, magnitude)
    c = engine.config
    T = real_type(engine)
    (; weights, norm_mask) = c
    relative, absolute = T(c.rtol) * magnitude, T(c.atol)
    nothing_ = zero(T)
    residual = reduce_each(+, nothing_, c, size(error)...) do f, m
        @inbounds begin
            scale = relative * weights[f] + absolute
            value = error[f, m]
            norm_mask[f] && !(scale == 0 && value == 0) ? abs2(value / scale) : nothing_
        end
    end
    Float64(sqrt(residual))
end

"""Accept the candidate: the mask, the split-step collar, then the quadrature alarm."""
commit!(engine, dz::Float64, monitored::Bool=false) = @section :commit commit_step!(engine, dz, monitored)
function commit_step!(engine, dz, monitored)
    c, w = engine.config, engine.work
    mask!(engine, w.candidate)
    if c.split_active
        # With a split monitor bound, the power the split finds (`split_power`).
        c.free ? split_free!(engine, dz, monitored) : split_modal!(engine, dz, monitored)
        mask!(engine, w.candidate)
        engine.fsal_valid = false
    elseif !c.low_storage && w.adams === nothing
        # The last stage's source is the next trial's first.
        pool_copy!(@view(w.stages[:, :, 1]), w.source, c)
        engine.fsal_valid = true
    end
    pool_copy!(w.y, w.candidate, c)
    c.embedded ? quadrature_alarm(engine) : NaN
end

function split_modal!(engine, dz, monitored)
    c, w = engine.config, engine.work
    (; coarse) = w
    mul!(coarse, w.coarse_inverse, w.candidate)
    if monitored
        fill!(w.monitor.split, 0)
        add_power!(w.monitor.split, coarse, w.monitor.columns, 1, size(coarse, 2), c)
    end
    rate = c.coarse_rate
    T = real_type(engine)
    # The unnormalised inverse's 1/nc is folded into the collar's factor.
    half, inverse = T(-0.5 * dz), inv(T(c.nc))
    factor = collar_factor(rate, half, inverse, c)
    each!(c, size(coarse)...) do t, m
        @inbounds coarse[t, m] *= factor[t]
    end
    mul!(w.candidate, w.coarse_forward, coarse)
end

"""The collar's half-step factor a sample, `exp(half rate[t]) inverse`, formed once a
sample rather than once an element (384 modes each took the same exponential)."""
function collar_factor(rate, half, inverse, c)
    factor = oncpu(c) ? similar(rate) : scratch(c, :collar, eltype(rate), length(rate))
    each!(c, length(rate), 1) do t, _
        @inbounds factor[t] = exp(half * rate[t]) * inverse
    end
    factor
end

"""The quadrature alarm's arrays for a reduction body (`alarm_term`), its transforms done:
the exact and the coarse spectra, the crop (`nothing` on the band), the prefactor and
the window; `nothing` without an embedded quadrature."""
function alarm_arrays(engine)
    c, w = engine.config, engine.work
    c.embedded || return nothing
    c.spectral && return (w.band_source, w.band_coarse, nothing, c.prefactor, c.omega_window)
    # A GPU's packed streams: the quadrature source transformed and assembled them.
    w.pack === nothing || return (w.fine, w.coarse_spec, c.crop, c.prefactor, c.omega_window)
    if w.plasma === nothing
        mul!(w.coarse_spec, w.forward_plan, w.projected_coarse)
    else
        p = w.plasma
        transform_all!(w.forward_plan, (w.coarse_spec, p.spectrum, p.edge_spectrum),
                       (w.projected_coarse, p.coarse, p.coarse_edge), w.forward_stage, c)
        add_plasma_spectra!(w.coarse_spec, p, c)
    end
    (w.fine, w.coarse_spec, c.crop, c.prefactor, c.omega_window)
end

"""One element's two alarm sums: the exact source's power and its coarse error's."""
@inline function alarm_term(alarm, f, m)
    exact, coarse, crop, prefactor, window = alarm
    @inbounds begin
        k = crop === nothing ? f : crop[f]
        weight = prefactor[f, m] * window[f]
        value = weight * exact[k, m]
        (abs2(value), abs2(weight * coarse[k, m] - value))
    end
end
alarm_value(power, error) = power == 0 ? NaN : Float64(sqrt(error / power))

"""The relative error of the quadrature's coarse source against the exact one: the
GPU trial's own, read with its norms, when it left one (`alarm_ready`)."""
function quadrature_alarm(engine)
    if engine.alarm_ready
        engine.alarm_ready = false
        return engine.alarm
    end
    alarm = alarm_arrays(engine)
    T = real_type(engine)
    sums = alarm_sums(alarm, engine.config, T, size(engine.work.y))
    alarm_value(sums...)
end
alarm_sums(alarm, c, T, dims) =
    reduce_each((f, m) -> alarm_term(alarm, f, m), .+, (zero(T), zero(T)), c, dims...)
band_alarm(engine) = quadrature_alarm(engine)

"""Keep a GPU trial's alarm, read with its norms, for the commit."""
function hold_alarm!(engine, sums)
    engine.alarm = alarm_value(sums[end-1], sums[end])
    engine.alarm_ready = true
end
