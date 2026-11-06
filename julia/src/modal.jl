# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Modal and radial routes: the state is (n_omega, n_modes); the response is evaluated
# at the transform's nodes, chunk by chunk, and projected back onto the modes.

"""Settings and buffers both routes share: band, windows, tableau, controller."""
function common_tables(c, host)
    T = c.T
    crop = host.array("crop", Int) .+ 1
    nt = length(host.array("window", Float64))
    nf = c.envelope ? nt : nt ÷ 2 + 1
    # Which band bin feeds each fine bin, zero for the padding: scattering a stage
    # into the fine spectrum is then one pass that also clears the padding.
    uncrop = zeros(Int, nf)
    uncrop[crop] .= eachindex(crop)
    merge(c, (
        crop=upload(c, crop), uncrop=upload(c, uncrop),
        mask=upload(c, host.array("mask", Bool)), norm_mask=upload(c, host.array("norm_mask", Bool)),
        weights=upload(c, host.array("weights", T)), uniform=all(==(1), host.array("weights", Float64)),
        window=upload(c, host.array("window", T)), omega_window=upload(c, host.array("omega_window", T)),
        rate=host.array("rate", Float64), coarse_rate=host.array("coarse_rate", Float64),
        a=host.array("a", Float64), b=host.array("b", Float64), e=host.array("e", Float64),
        nodes_c=host.array("c", Float64), nt=nt, nf=nf,
        split_active=c.split && any(!iszero, host.array("coarse_rate", Float64)),
    ))
end

function create_modal(c, host)
    c = common_tables(c, host)
    T, C = c.T, Complex{c.T}
    nt, nf = c.nt, c.nf
    synthesis = host.array("synthesis", Float64)
    spatial = host.has("spatial_absorber") ? host.array("spatial_absorber", Float64) : zeros(0, 0)
    # The translated rates are constant over a trial: an exactly absent loss needs no
    # additive temporal transform; the accepted-step split stays separate.
    in_stage = any(!iszero, spatial) || (!c.split && any(!iszero, c.rate))
    nw, nm = size(host.array("linear", ComplexF64))
    nn = size(synthesis, 2)
    chunk = min(host.int("chunk"), nn)
    R = c.envelope ? C : T
    # Every inverse transform is unnormalised: its 1/nt is folded into the tables that
    # read the time-domain field, the synthesis and the in-stage absorber.
    c = merge(c, (
        linear=upload(c, host.array("linear", C)), prefactor=upload(c, host.array("prefactor", C)),
        synthesis=upload(c, T.(synthesis ./ nt)), projection=upload(c, host.array("projection", T)),
        projection_coarse=upload(c, host.array("projection_coarse", T)),
        embedded=host.flag("embedded"), spatial_absorber=upload(c, T.(spatial ./ nt)),
        in_stage=in_stage, absorber_rate=upload(c, T.(-0.5 .* c.rate ./ nt)),
        coarse_rate=upload(c, T.(c.coarse_rate)), nc=length(c.coarse_rate),
    ))
    ns = length(c.nodes_c)
    fine = dzeros(c, C, nf, nm)
    modal = dzeros(c, R, nt, nm)
    nodes = dzeros(c, R, nt, chunk)
    analytic = dzeros(c, C, nt, chunk)
    coarse = dzeros(c, R, c.nc, nm)
    band = dzeros(c, C, nw, nm)
    work = (
        adams=adams_workspace(c, host, nw, nm), path=path_workspace(c, host),
        monitor=monitor_workspace(c, nw, nm, c.nc), edge=Ref{Any}(nothing),
        raman=raman_workspace(c, host, nt, chunk; modes=nm, nodes=nn, synthesis=synthesis ./ nt), plasma=plasma_workspace(c, host, nt, chunk, nm, nn),
        y=dzeros(c, C, nw, nm), candidate=dzeros(c, C, nw, nm), lab=dzeros(c, C, nw, nm),
        source=dzeros(c, C, nw, nm), fsal=dzeros(c, C, nw, nm), error=dzeros(c, C, nw, nm),
        stages=dzeros(c, C, nw, nm, ns), forward=dzeros(c, C, nw, nm, ns), inverse=dzeros(c, C, nw, nm, ns),
        fine=fine, modal=modal, nodes=nodes, analytic=analytic,
        projected=dzeros(c, R, nt, nm), projected_coarse=dzeros(c, R, nt, nm), coarse_spec=similar(fine),
        absorber=dzeros(c, R, nt, nm), absorber_spec=similar(fine), coarse=coarse,
        inverse_plan=c.envelope ? plan(c, plan_bfft, fine, 1) : plan(c, plan_brfft, fine, nt, 1),
        forward_plan=c.envelope ? plan(c, plan_fft, modal, 1) : plan(c, plan_rfft, modal, 1),
        analytic_forward=plan(c, plan_fft!, analytic, 1), analytic_inverse=plan(c, plan_bfft!, analytic, 1),
        coarse_inverse=c.envelope ? plan(c, plan_bfft, band, 1) : plan(c, plan_brfft, band, c.nc, 1),
        coarse_forward=c.envelope ? plan(c, plan_fft, coarse, 1) : plan(c, plan_rfft, coarse, 1),
    )
    Engine(c, work, false, 0, NaN, c.factor)
end

"""Place a band-limited state in the fine spectrum, clearing the padding."""
function scatter!(fine, lab, c)
    uncrop = c.uncrop
    each!(c, size(fine)...) do k, m
        @inbounds begin
            f = uncrop[k]
            fine[k, m] = f == 0 ? zero(eltype(fine)) : lab[f, m]
        end
    end
end

"""Evaluate the translated source of `lab` into `work.source`."""
function source!(engine, lab; quadrature=false)
    c, w = engine.config, engine.work
    c.free && return source_free!(engine, lab)
    nm = size(lab, 2)
    @section :scatter scatter!(w.fine, lab, c)
    @section :inverse_fft mul!(w.modal, w.inverse_plan, w.fine)
    fill!(w.projected, 0)
    quadrature && fill!(w.projected_coarse, 0)
    clear_plasma!(w.plasma)
    if engine.factor != 0 || w.plasma !== nothing || w.raman !== nothing
        products = w.raman !== nothing && w.raman.products
        products && @section :raman raman_products!(w.raman, w.modal, c)
        chunk, nn = size(w.nodes, 2), size(c.synthesis, 2)
        for first in 1:chunk:nn
            last = min(first + chunk - 1, nn)
            count = last - first + 1
            nodes = @view w.nodes[:, 1:count]
            @section :synthesis gemm!(nodes, w.modal, @view(c.synthesis[:, first:last]), true, false, c)
            if w.plasma !== nothing
                @section :plasma plasma_current!(w.plasma, w.nodes, c, count)
                @section :plasma_projection project_plasma!(w.plasma, c, first, last, quadrature)
            end
            products && @section :raman raman_nodal!(w.raman, first, last, c)
            node_response!(engine, count)
            @section :projection gemm!(w.projected, nodes, @view(c.projection[first:last, :]), true, true, c)
            quadrature && c.embedded &&
                gemm!(w.projected_coarse, nodes, @view(c.projection_coarse[first:last, :]), true, true, c)
        end
    end
    @section :forward_fft mul!(w.fine, w.forward_plan, w.projected)
    w.plasma === nothing || @section :plasma_spectrum add_plasma!(w.fine, w.plasma, w.forward_plan, c)
    if c.in_stage
        absorber!(w.absorber, w.modal, c)
        isempty(c.spatial_absorber) || gemm!(w.absorber, w.modal, c.spatial_absorber, true, true, c)
        mul!(w.absorber_spec, w.forward_plan, w.absorber)
    end
    @section :finish finish_source!(w.source, w.fine, c.in_stage ? w.absorber_spec : nothing, c)
    engine.calls += 1
    nothing
end

"""Kerr and Raman at a chunk's nodes, through the analytic signal where one is read."""
function node_response!(engine, count)
    c, w = engine.config, engine.work
    nodes, raman = w.nodes, w.raman
    if c.envelope
        raman === nothing || raman_drive!(raman, nodes, nodes, c, count)
        nonlinear_envelope!(nodes, c, engine.factor, raman, count)
    elseif c.analytic || (raman !== nothing && !raman.thg && !raman.products)
        @section :analytic analytic_signal!(w.analytic, nodes, w.analytic_forward, w.analytic_inverse, c, count)
        raman === nothing || raman.products || @section :raman raman_drive!(raman, nodes, w.analytic, c, count)
        @section :kerr nonlinear_nodes!(nodes, c.analytic ? w.analytic : nodes, c, engine.factor, raman, count)
    else
        raman === nothing || raman.products || @section :raman raman_drive!(raman, nodes, nodes, c, count)
        @section :kerr nonlinear_nodes!(nodes, nodes, c, engine.factor, raman, count)
    end
end

"""The in-stage temporal collar's loss, `-alpha(t) E / 2`, unless the step splits it."""
function absorber!(absorber, modal, c)
    if c.split
        fill!(absorber, 0)
    else
        rate = c.absorber_rate
        each!(c, size(absorber)...) do t, m
            @inbounds absorber[t, m] = rate[t] * modal[t, m]
        end
    end
end

"""Scale the fine spectrum back onto the band: the prefactor and the omega window."""
function finish_source!(source, fine, loss, c)
    (; prefactor, omega_window, crop, mask) = c
    live = loss !== nothing
    extra = live ? loss : fine
    each!(c, size(source)...) do f, m
        @inbounds begin
            k = crop[f]
            value = prefactor[f, m] * omega_window[f] * fine[k, m]
            if live && mask[f]
                value += extra[k, m]
            end
            source[f, m] = value
        end
    end
end

"""Observation populations of an accepted state: fractions, peaks and the six maxima."""
function population_snapshot(engine, input)
    engine.config.free && return population_free_snapshot(engine, input)
    c, w = engine.config, engine.work
    p = w.plasma
    scatter!(w.fine, device_matrix(engine, input), c)
    mul!(w.modal, w.inverse_plan, w.fine)
    chunk, nn = size(w.nodes, 2), size(c.synthesis, 2)
    maxima = zeros(6)
    for first in 1:chunk:nn
        last = min(first + chunk - 1, nn)
        count = last - first + 1
        gemm!(@view(w.nodes[:, 1:count]), w.modal, @view(c.synthesis[:, first:last]), true, false, c)
        maxima .= max.(maxima, population_parts!(p, w.nodes, c, first, count))
    end
    (Vector{Float64}(Array(p.fractions)), Vector{Float64}(Array(p.peaks)), maxima)
end
