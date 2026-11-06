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
    # Synthesis and projection in the frequency domain, on the band's rows (`source_spectral!`),
    # where the nodes are about as many as the modes, as radially: half the GEMM rows of
    # the fine time grid's, and one projection for the response and both plasma streams.
    # Not where the modes are few (a capillary's nodes would add transforms), nor beside
    # Raman or an unsplit temporal collar, which read the modes' time signals.
    spectral = !c.envelope && !host.flag("raman") && 2nm >= nn &&
               !(!c.split && any(!iszero, c.rate))
    # Every inverse transform is unnormalised: its 1/nt is folded into the tables that
    # read the time-domain field, the synthesis and the in-stage absorber.
    c = merge(c, (spectral=spectral, spectral_absorber=upload(c, T.(spatial)),
        # The operator stays binary64 at either precision: propagators and the Adams
        # flow are formed from it in binary64 and rounded once (SEP 33, 4.1 item 3).
        linear=upload(c, host.array("linear", ComplexF64)), prefactor=upload(c, host.array("prefactor", C)),
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
    # The spectral path's buffers: a chunk's node spectra on the band and the fine grid,
    # its polarisation's, and the band's projected source (and its coarse quadrature's).
    node_band = dzeros(c, C, spectral ? nw : 0, spectral ? chunk : 0)
    node_fine = dzeros(c, C, spectral ? nf : 0, spectral ? chunk : 0)
    work = (
        node_band=node_band, node_fine=node_fine, band_source=dzeros(c, C, spectral ? nw : 0, spectral ? nm : 0),
        band_coarse=dzeros(c, C, spectral ? nw : 0, spectral ? nm : 0),
        node_inverse=spectral ? plan(c, plan_brfft, node_fine, nt, 1) : nothing,
        node_forward=spectral ? plan(c, plan_rfft, nodes, 1) : nothing,
        absorber_spec_band=dzeros(c, C, spectral && in_stage ? nw : 0, spectral && in_stage ? nm : 0),
        adams=adams_workspace(c, host, nw, nm), path=path_workspace(c, host),
        monitor=monitor_workspace(c, nw, nm, c.nc), edge=Ref{Any}(nothing), statistics=Ref{Any}(nothing),
        raman=raman_workspace(c, host, nt, chunk; modes=nm, nodes=nn, synthesis=synthesis ./ nt), plasma=plasma_workspace(c, host, nt, chunk, nm, nn),
        y=dzeros(c, C, nw, nm), candidate=dzeros(c, C, nw, nm), lab=dzeros(c, C, nw, nm),
        source=dzeros(c, C, nw, nm), fsal=dzeros(c, C, nw, nm), error=dzeros(c, C, nw, nm),
        stages=dzeros(c, C, nw, nm, ns), forward=dzeros(c, C, nw, nm, ns), inverse=dzeros(c, C, nw, nm, ns),
        fine=fine, modal=modal, nodes=nodes, analytic=analytic,
        projected=dzeros(c, R, nt, spectral ? 0 : nm), projected_coarse=dzeros(c, R, nt, spectral ? 0 : nm),
        coarse_spec=similar(fine), absorber=dzeros(c, R, nt, spectral ? 0 : nm), absorber_spec=similar(fine),
        coarse=coarse,
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
    c.spectral && return source_spectral!(engine, lab, quadrature)
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
    if w.plasma === nothing
        @section :forward_fft mul!(w.fine, w.forward_plan, w.projected)
    else
        # The source's and the currents' transforms as one pool's work.
        p = w.plasma
        @section :forward_fft transform_all!(w.forward_plan, (w.fine, p.spectrum, p.edge_spectrum),
                                             (w.projected, p.projected, p.edge_projected))
        @section :plasma_spectrum add_plasma_spectra!(w.fine, p, c)
    end
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

"""The source synthesised and projected on the band, chunk by chunk of nodes.

The same arithmetic as `source!`, reordered: the synthesis and projection are linear
along the modes and the transforms along time, so each node's spectrum is the band's
modes times its synthesis row, and the source's band is the nodes' forward spectra
times the projection. A node's analytic signal is the inverse transform of its doubled
one-sided spectrum, whose real part is its field. The plasma's two streams join each
node's spectrum before the one projection, as in free space.
"""
function source_spectral!(engine, lab, quadrature)
    c, w = engine.config, engine.work
    (; node_band, node_fine, nodes, analytic, band_source, band_coarse) = w
    nt, chunk = size(nodes)
    nn = size(c.synthesis, 2)
    fill!(band_source, 0)
    quadrature && c.embedded && fill!(band_coarse, 0)
    (; uncrop, crop) = c
    T = real_type(engine)
    unit, two, half = one(T), T(2), nt ÷ 2
    for first in 1:chunk:nn
        last = min(first + chunk - 1, nn)
        count = last - first + 1
        band = @view node_band[:, 1:count]
        @section :synthesis gemm!(band, lab, @view(c.synthesis[:, first:last]), true, false, c)
        if c.analytic
            @section :analytic begin
                each!(c, nt, chunk) do k, m
                    @inbounds begin
                        f = k <= half + 1 && m <= count ? uncrop[k] : 0
                        weight = k == 1 || k == half + 1 ? unit : two
                        analytic[k, m] = f == 0 ? zero(eltype(analytic)) : weight * node_band[f, m]
                    end
                end
                w.analytic_inverse * analytic
                each!(c, nt, chunk) do t, m
                    @inbounds nodes[t, m] = real(analytic[t, m])
                end
            end
        else
            @section :inverse_fft begin
                each!(c, size(node_fine)...) do k, m
                    @inbounds begin
                        f = m <= count ? uncrop[k] : 0
                        node_fine[k, m] = f == 0 ? zero(eltype(node_fine)) : node_band[f, m]
                    end
                end
                mul!(nodes, w.node_inverse, node_fine)
            end
        end
        w.plasma === nothing || @section :plasma plasma_current!(w.plasma, nodes, c, count)
        @section :kerr nonlinear_nodes!(nodes, c.analytic ? analytic : nodes, c, engine.factor, nothing, count)
        @section :forward_fft mul!(node_fine, w.node_forward, nodes)
        @section :plasma_spectrum spectral_plasma!(node_band, node_fine, w.plasma, w.node_forward, c)
        @section :projection begin
            gemm!(band_source, band, @view(c.projection[first:last, :]), true, true, c)
            quadrature && c.embedded &&
                gemm!(band_coarse, band, @view(c.projection_coarse[first:last, :]), true, true, c)
        end
    end
    loss = nothing
    if c.in_stage
        # The spatial collar is linear and time-independent: its band is the state's
        # band times the matrix. The spectral path runs only without a temporal one.
        gemm!(w.absorber_spec_band, lab, c.spectral_absorber, true, false, c)
        loss = w.absorber_spec_band
    end
    @section :finish finish_band!(w.source, band_source, loss, c)
    engine.calls += 1
    nothing
end

"""A chunk's band from its nodes' fine spectra, with the plasma streams joined."""
function spectral_plasma!(band, fine, p, plan, c)
    crop = c.crop
    if p === nothing
        each!(c, size(band)...) do f, m
            @inbounds band[f, m] = fine[crop[f], m]
        end
        return nothing
    end
    mul!(p.spectrum, plan, p.gated)
    mul!(p.edge_spectrum, plan, p.edge)
    (; spectrum, edge_spectrum, inverse_omega) = p
    mask = c.mask
    each!(c, size(band)...) do f, m
        @inbounds begin
            k = crop[f]
            value = fine[k, m]
            if mask[f]
                inverse = inverse_omega[f]
                value += -im * inverse * spectrum[k, m] + inverse^2 * edge_spectrum[k, m]
            end
            band[f, m] = value
        end
    end
end

"""The band's source: the prefactor and the omega window, and an in-stage loss."""
function finish_band!(source, band, loss, c)
    (; prefactor, omega_window, mask) = c
    live = loss !== nothing
    extra = live ? loss : band
    each!(c, size(source)...) do f, m
        @inbounds begin
            value = prefactor[f, m] * omega_window[f] * band[f, m]
            if live && mask[f]
                value += extra[f, m]
            end
            source[f, m] = value
        end
    end
end

"""A state given to an observation, at the engine's scale; `nothing` is its own state."""
scaled_input(engine, ::Nothing) = engine.work.y
scaled_input(engine, input) =
    device_matrix(engine, input) .* real_type(engine)(inv(engine.config.scale))

"""Observation populations of an accepted state: fractions, peaks and the six maxima."""
function population_snapshot(engine, input)
    engine.config.free && return population_free_snapshot(engine, input)
    c, w = engine.config, engine.work
    p = w.plasma
    scatter!(w.fine, scaled_input(engine, input), c)
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
