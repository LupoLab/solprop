# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Cartesian and line free space: the state is (n_omega, n_x n_y) in the transverse
# spectral basis; the response is evaluated in physical space, chunk by chunk of
# columns, and the operator is formed per element from separable tables.

function create_free(c, host)
    c = common_tables(c, host)
    T, C = c.T, Complex{c.T}
    shape = Tuple(Int.(host.value("spatial_shape")))
    nw, nm, nt, nf = length(host.array("crop", Int)), prod(shape), c.nt, c.nf
    spatial_rate = host.array("spatial_rate", Float64)
    # An exactly absent additive loss omits its transforms; the accepted-step split
    # and the diagonal spectral loss are separate.
    in_stage = any(!iszero, spatial_rate) || (!c.split && any(!iszero, c.rate))
    nc = length(c.coarse_rate)
    # Both inverse transforms are unnormalised; their 1/(nt n_x n_y) goes into the
    # factor that forms the physical field from them.
    scale = inv(T(nt) * T(nm))
    # The separable operator stays binary64 at either precision: the longitudinal root's
    # compensated subtraction decides its branch near grazing (free_element).
    R64, C64 = Float64, ComplexF64
    op = (
        k_squared=upload(c, host.array("k_squared", C64)), k_squared_low=upload(c, host.array("k_squared_low", C64)),
        kperp_sq=upload(c, host.array("kperp_sq", R64)), kperp_sq_low=upload(c, host.array("kperp_sq_low", R64)),
        frame=upload(c, host.array("frame", R64)), spectral_rate=upload(c, host.array("spectral_rate", R64)),
        prefactor_numerator=upload(c, host.array("prefactor_numerator", C64)),
        prefactor_clamped=upload(c, host.array("prefactor_clamped", C64)),
        active=upload(c, host.array("active", Bool)), kperp_rate=upload(c, host.array("kperp_rate", R64)),
        evanescent_rate=host.real("evanescent_rate"),
    )
    c = merge(c, (
        spatial_shape=shape, op=op, field_scale=T(host.real("pad_scale")) * scale,
        absorber_scale=T(-0.5) * scale, spatial_rate=upload(c, T.(spatial_rate)),
        temporal_rate=upload(c, T.(c.split ? zero(c.rate) : c.rate)), in_stage=in_stage,
        coarse_rate=upload(c, T.(c.coarse_rate)), nc=nc, embedded=false,
    ))
    chunk = min(host.int("chunk"), nm)
    R = c.envelope ? C : T
    stages = c.low_storage ? 0 : 7
    physical = dzeros(c, C, nw, nm)
    spatial = reshape(physical, nw, shape...)
    fine, modal = dzeros(c, C, nf, chunk), dzeros(c, R, nt, chunk)
    analytic = dzeros(c, C, nt, chunk)
    coarse, coarse_spec = dzeros(c, R, nc, chunk), dzeros(c, C, nw, chunk)
    planes = (threads=c.transverse_fft_threads, batch=1)
    work = (
        adams=nothing, path=nothing, raman=nothing, monitor=monitor_workspace(c, nw, nm, nc), edge=Ref{Any}(nothing), march=Ref{Any}(nothing),
        plasma=plasma_workspace(c, host, nt, chunk, chunk, nm),
        y=dzeros(c, C, nw, nm), candidate=dzeros(c, C, nw, nm), lab=dzeros(c, C, nw, nm),
        source=dzeros(c, C, nw, nm), error=dzeros(c, C, nw, nm),
        stages=dzeros(c, C, nw, nm, stages),
        forward=dzeros(c, C, nw, nm, 1), inverse=dzeros(c, C, nw, nm, c.low_storage ? 0 : 1),
        physical=physical, loss=dzeros(c, C, in_stage ? nw : 0, in_stage ? nm : 0),
        fine=fine, modal=modal, projected=dzeros(c, R, nt, chunk), analytic=analytic,
        absorber=dzeros(c, R, nt, chunk), absorber_spec=similar(fine), coarse=coarse, coarse_spec=coarse_spec,
        spatial_inverse=plan(c, plan_bfft!, spatial, (2, 3); planes...),
        spatial_forward=plan(c, plan_fft!, spatial, (2, 3); planes...),
        inverse_plan=c.envelope ? plan(c, plan_bfft, fine, 1) : plan(c, plan_brfft, fine, nt, 1),
        forward_plan=c.envelope ? plan(c, plan_fft, modal, 1) : plan(c, plan_rfft, modal, 1),
        analytic_forward=plan(c, plan_fft!, analytic, 1), analytic_inverse=plan(c, plan_bfft!, analytic, 1),
        coarse_inverse=c.envelope ? plan(c, plan_bfft, coarse_spec, 1) : plan(c, plan_brfft, coarse_spec, nc, 1),
        coarse_forward=c.envelope ? plan(c, plan_fft, coarse, 1) : plan(c, plan_rfft, coarse, 1),
    )
    Engine(c, work, false, 0, NaN, c.factor)
end

# For an exactly real squared longitudinal wave number the principal root is sqrt(x)
# for x > 0 and i sqrt(-x) for x < 0, keeping the imaginary signed zero; a lossy or
# nonfinite input takes the library's complex root. No near-zero cutoff.
@inline function free_root(q)
    x, y = real(q), imag(q)
    if iszero(y) && isfinite(x)
        x > 0 && return complex(sqrt(x), y)
        x < 0 && return complex(zero(x), copysign(sqrt(-x), y))
        return complex(zero(x), y)
    end
    sqrt(q)
end

"""The element's linear rate and its prefactor, from the separable tables.

The compensated subtraction that forms `q` keeps translate.spatial.longitudinal_root's
order: near grazing the residual decides the branch, so no reassociation here.
"""
@inline function free_element(op, f, m)
    @inbounds begin
        q = (op.k_squared[f] - op.kperp_sq[m]) + (op.k_squared_low[f] - op.kperp_sq_low[m])
        root = free_root(q)
        propagating = real(q) > 0
        attenuation = propagating ? imag(root) : min(abs(imag(root)), op.evanescent_rate)
        live = op.active[f]
        rate = live ? complex(-attenuation - op.kperp_rate[m] / 2, -(real(root) - op.frame[f])) :
               zero(root)
        prefactor = propagating ? op.prefactor_numerator[f] / real(root) : op.prefactor_clamped[f]
        return rate - op.spectral_rate[f] / 2, live ? prefactor : zero(prefactor)
    end
end

"""Form the propagator pair of one distance in slot one; return the slot."""
function free_propagators!(engine, h)
    engine.propagator_dz == h && return 1
    c, w = engine.config, engine.work
    (; forward, inverse) = w
    if h == 0
        # The zero-distance stage is exactly the identity.
        fill!(forward, 1)
        fill!(inverse, 1)
    else
        op, both, distance = c.op, !c.low_storage, Float64(h)
        each!(c, size(forward, 1), size(forward, 2)) do f, m
            @inbounds begin
                exponent = distance * first(free_element(op, f, m))
                # Both signs share exact trigonometric inputs, as in the compiled seam.
                scale = exp(real(exponent))
                sine, cosine = sincos(imag(exponent))
                forward[f, m, 1] = scale * complex(cosine, sine)
                both && (inverse[f, m, 1] = complex(cosine, -sine) / scale)
            end
        end
    end
    engine.propagator_dz = h
    1
end

transverse!(a, plan, c) = (plan * reshape(a, size(a, 1), c.spatial_shape...); nothing)

"""Copy columns `first:first+count-1` of `physical` into the fine spectrum."""
function scatter_columns!(fine, physical, first, count, c)
    uncrop = c.uncrop
    each!(c, size(fine)...) do k, m
        @inbounds begin
            f = uncrop[k]
            fine[k, m] = f == 0 || m > count ? zero(eltype(fine)) : physical[f, first+m-1]
        end
    end
end

function source_free!(engine, lab)
    c, w = engine.config, engine.work
    (; physical, fine, modal, projected, analytic, loss) = w
    nm = size(lab, 2)
    chunk = size(modal, 2)
    copyto!(physical, lab)
    transverse!(physical, w.spatial_inverse, c)
    # Transverse inversion first: a transverse spectral column alone is not the
    # spectrum of a real temporal field.
    scale, window, a = c.field_scale, c.window, c.analytic_factor
    T = real_type(engine)
    factor = T(c.factor)
    for first in 1:chunk:nm
        count = min(chunk, nm - first + 1)
        scatter_columns!(fine, physical, first, count, c)
        mul!(modal, w.inverse_plan, fine)
        if w.plasma !== nothing
            each!(c, size(modal)...) do t, m
                @inbounds projected[t, m] = scale * modal[t, m]
            end
            plasma_current!(w.plasma, projected, c, count)
        end
        if c.envelope
            each!(c, size(modal)...) do t, m
                @inbounds begin
                    v = scale * modal[t, m]
                    projected[t, m] = factor * a * abs2(v) * v * window[t]
                end
            end
        elseif c.analytic
            each!(c, size(modal)...) do t, m
                @inbounds projected[t, m] = scale * modal[t, m]
            end
            analytic_signal!(analytic, projected, w.analytic_forward, w.analytic_inverse, c, chunk)
            nonlinear_nodes!(projected, analytic, c, factor, nothing, chunk)
        else
            each!(c, size(modal)...) do t, m
                @inbounds projected[t, m] = polarization(scale * modal[t, m], factor, a) * window[t]
            end
        end
        mul!(fine, w.forward_plan, projected)
        # The held-current polarisation joins before the transverse forward transform,
        # as the reference Cartesian source does.
        w.plasma === nothing || add_plasma!(fine, w.plasma, w.forward_plan, c; direct=true)
        c.in_stage && free_absorber!(w, c, first, count)
        gather_columns!(physical, fine, c.in_stage ? loss : nothing, w.absorber_spec, first, count, c)
    end
    transverse!(physical, w.spatial_forward, c)
    c.in_stage && transverse!(loss, w.spatial_forward, c)
    (; source) = w
    (; op, omega_window, mask) = c
    live = c.in_stage
    extra = live ? loss : physical
    each!(c, size(source)...) do f, m
        @inbounds begin
            prefactor = oftype(physical[f, m], last(free_element(op, f, m)))
            value = prefactor * omega_window[f] * physical[f, m]
            if live && mask[f]
                value += extra[f, m]
            end
            source[f, m] = value
        end
    end
    engine.calls += 1
    nothing
end

"""The in-stage loss of a chunk's columns: their spatial collar and, unsplit, time's."""
function free_absorber!(w, c, first, count)
    (; absorber, modal) = w
    spatial, temporal, scale = c.spatial_rate, c.temporal_rate, c.absorber_scale
    each!(c, size(absorber)...) do t, m
        @inbounds begin
            rate = (m <= count ? spatial[first+m-1] : zero(scale)) + temporal[t]
            absorber[t, m] = scale * rate * modal[t, m]
        end
    end
    mul!(w.absorber_spec, w.forward_plan, absorber)
end

function gather_columns!(physical, fine, loss, absorber_spec, first, count, c)
    crop = c.crop
    live = loss !== nothing
    target = live ? loss : physical
    each!(c, size(physical, 1), count) do f, m
        @inbounds begin
            k = crop[f]
            physical[f, first+m-1] = fine[k, m]
            live && (target[f, first+m-1] = absorber_spec[k, m])
        end
    end
end

function split_free!(engine, dz, monitored)
    c, w = engine.config, engine.work
    (; physical, coarse, coarse_spec) = w
    copyto!(physical, w.candidate)
    transverse!(physical, w.spatial_inverse, c)
    chunk, nm, nw = size(coarse, 2), size(physical, 2), size(physical, 1)
    T = real_type(engine)
    rate, half = c.coarse_rate, T(-0.5 * dz)
    # Both unnormalised inverses' 1/(nc n_x n_y) is folded into the collar's factor.
    inverse = inv(T(c.nc) * T(nm))
    monitored && fill!(w.monitor.split, 0)
    for first in 1:chunk:nm
        count = min(chunk, nm - first + 1)
        columns_into!(coarse_spec, physical, first, count, c)
        mul!(coarse, w.coarse_inverse, coarse_spec)
        monitored && add_power!(w.monitor.split, coarse, w.monitor.columns, first, count, c)
        each!(c, size(coarse)...) do t, m
            @inbounds coarse[t, m] *= exp(half * rate[t]) * inverse
        end
        mul!(coarse_spec, w.coarse_forward, coarse)
        each!(c, nw, count) do f, m
            @inbounds physical[f, first+m-1] = coarse_spec[f, m]
        end
    end
    transverse!(physical, w.spatial_forward, c)
    copyto!(w.candidate, physical)
    nothing
end

function population_free_snapshot(engine, input)
    c, w = engine.config, engine.work
    p = w.plasma
    (; physical, fine, modal, projected) = w
    copyto!(physical, scaled_input(engine, input))
    transverse!(physical, w.spatial_inverse, c)
    chunk, nm = size(modal, 2), size(physical, 2)
    scale = c.field_scale
    maxima = zeros(6)
    for first in 1:chunk:nm
        count = min(chunk, nm - first + 1)
        scatter_columns!(fine, physical, first, count, c)
        mul!(modal, w.inverse_plan, fine)
        each!(c, size(modal)...) do t, m
            @inbounds projected[t, m] = scale * modal[t, m]
        end
        maxima .= max.(maxima, population_parts!(p, projected, c, first, count))
    end
    (Vector{Float64}(Array(p.fractions)), Vector{Float64}(Array(p.peaks)), maxima)
end
