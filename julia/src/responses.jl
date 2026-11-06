# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The translated scalar responses: Kerr, one field-plasma channel and one Raman term.
# Rate tables, constants, FIR taps and collar derivatives all come from the seam.

@inline polarization(value::Real, factor, analytic_factor) = factor * (value * value) * value
@inline polarization(value::Complex, factor, analytic_factor) =
    factor * analytic_factor * abs2(value) * real(value)

"""Write the Kerr and Raman polarisation of each node's field over it, windowed.

`values` is what the Kerr term reads: the field, or its analytic signal. Raman, when
present, multiplies the field by its convolved drive (`raman_drive!`).
"""
function nonlinear_nodes!(nodes, values, c, factor, raman, count)
    window, a = c.window, c.analytic_factor
    T = eltype(window)
    f = T(factor)
    if raman === nothing
        each!(c, size(nodes, 1), count) do t, m
            @inbounds nodes[t, m] = polarization(values[t, m], f, a) * window[t]
        end
    else
        drive, r = raman.drive, T(raman.factor)
        each!(c, size(nodes, 1), count) do t, m
            @inbounds begin
                v = nodes[t, m]
                nodes[t, m] = (polarization(values[t, m], f, a) + r * v * drive[t, m]) * window[t]
            end
        end
    end
end

function nonlinear_envelope!(nodes, c, factor, raman, count)
    window = c.window
    T = eltype(window)
    f = T(factor) * c.analytic_factor
    r = raman === nothing ? zero(T) : T(raman.factor)
    drive = raman === nothing ? nodes : raman.drive
    live = raman !== nothing
    each!(c, size(nodes, 1), count) do t, m
        @inbounds begin
            v = nodes[t, m]
            response = f * abs2(v) * v
            if live
                response += r * v * real(drive[t, m])
            end
            nodes[t, m] = response * window[t]
        end
    end
end

"""Replace `analytic` by the analytic signal of the real columns of `nodes`."""
function analytic_signal!(analytic, nodes, w, c, count)
    nt, chunk = size(analytic)
    each!(c, nt, chunk) do t, m
        @inbounds analytic[t, m] = m <= count ? nodes[t, m] : 0
    end
    w.analytic_forward * analytic
    # The unnormalised inverse's 1/nt is folded into the one-sided spectral factor.
    T = real(eltype(analytic))
    one, two, half = inv(T(nt)), 2 / T(nt), nt ÷ 2
    each!(c, nt, chunk) do f, m
        @inbounds analytic[f, m] *= (f == 1 || f == half + 1) ? one : f <= half ? two : zero(T)
    end
    w.analytic_inverse * analytic
    nothing
end

# --- plasma -------------------------------------------------------------------------

function plasma_workspace(c, host, nt, chunk, nm, nn)
    host.flag("plasma") || return nothing
    T = c.T
    columns = c.free ? 0 : nm
    (
        coefficients=upload(c, host.array("plasma_coefficients", T)),
        # Field range, rate cap, log-table origin and step, charge-mass ratio and
        # ionisation potential; density and preionisation are a stage's (stages.jl).
        parameters=host.array("plasma_parameters", Float64),
        taps=upload(c, host.array("plasma_taps", T)), floor=host.flag("plasma_floor"),
        dt=T(host.real("dt")),
        inverse_omega=upload(c, T.(inv.(host.array("omega", Float64)))),
        derivative=upload(c, host.array("window_derivative", T)),
        rate=dzeros(c, T, nt, chunk), fraction=dzeros(c, T, nt, chunk),
        integrand=dzeros(c, T, nt, chunk), current=dzeros(c, T, nt, chunk),
        gated=dzeros(c, T, nt, chunk), edge=dzeros(c, T, nt, chunk),
        projected=dzeros(c, T, nt, columns), edge_projected=dzeros(c, T, nt, columns),
        coarse=dzeros(c, T, nt, columns), coarse_edge=dzeros(c, T, nt, columns),
        spectrum=dzeros(c, Complex{T}, nt ÷ 2 + 1, c.free ? chunk : nm),
        edge_spectrum=dzeros(c, Complex{T}, nt ÷ 2 + 1, c.free ? chunk : nm),
        fractions=dzeros(c, T, nn), peaks=dzeros(c, T, nn), parts=dzeros(c, T, 6, chunk),
    )
end

"""The piecewise-cubic ionisation rate in log |E|, clamped to the table's range."""
@inline function rate_value(value, table, lower, upper, cap, origin, step)
    x = abs(value)
    x < lower && return zero(cap)
    x >= upper && return cap
    logarithm = log(x)
    j = clamp(unsafe_trunc(Int, floor((logarithm - origin) / step)) + 1, 1, size(table, 2))
    s = logarithm - (origin + (j - 1) * step)
    @inbounds value = ((table[1, j] * s + table[2, j]) * s + table[3, j]) * s + table[4, j]
    min(exp(value), cap)
end
@inline rate_arguments(p) = (p.coefficients, ntuple(k -> typeof(p.dt)(p.parameters[k]), 5)...)

# A symmetric FIR, reflected at the ends as RunningIntegral is, then a running
# trapezoid. The scan's order is strict: no reassociation.
@inline function filtered(values, t, node, taps)
    nt = size(values, 1)
    @inbounds value = taps[1] * values[t, node]
    @inbounds for k in 1:length(taps)-1
        left = t - k < 1 ? 2 * values[1, node] - values[1+k-t+1, node] : values[t-k, node]
        right = t + k > nt ? 2 * values[nt, node] - values[2nt-t-k, node] : values[t+k, node]
        value += taps[k+1] * (left + right)
    end
    value
end

"""Integrate one column of `values` into `out`, sequentially down the column.

Exact zero support is skipped: reflection cannot carry the live interval past the
filter's radius, since its endpoint is then live already. Interior samples apply the
taps one at a time over the whole interval, which a CPU vectorises without reordering
any one sample's additions.
"""
@inline function integrate_column!(out, values, node, taps, dt)
    nt = size(values, 1)
    half = length(taps) - 1
    first, last = 1, nt
    @inbounds while first <= nt && iszero(values[first, node])
        first += 1
    end
    if first > nt
        @inbounds for t in 1:nt
            out[t, node] = 0
        end
        return nothing
    end
    @inbounds while iszero(values[last, node])
        last -= 1
    end
    first, last = max(1, first - half), min(nt, last + half)
    @inbounds for t in 1:first-1
        out[t, node] = 0
    end
    interior_first, interior_last = max(first, half + 1), min(last, nt - half)
    @inbounds for t in first:last
        if t < interior_first || t > interior_last
            out[t, node] = filtered(values, t, node, taps)
        else
            out[t, node] = taps[1] * values[t, node]
        end
    end
    @inbounds for k in 1:half
        tap = taps[k+1]
        @simd for t in interior_first:interior_last
            out[t, node] += tap * (values[t-k, node] + values[t+k, node])
        end
    end
    @inbounds previous = out[1, node]
    total = zero(dt)
    @inbounds out[1, node] = 0
    stop = min(nt, last + 1)
    @inbounds for t in max(2, first):stop
        current = out[t, node]
        total += dt * (current + previous) / 2
        out[t, node] = total
        previous = current
    end
    @inbounds for t in stop+1:nt
        out[t, node] = total
    end
    nothing
end

"""One column's rates and exposure; return the peak rate."""
@inline function exposure_column!(rate, exposure, nodes, node, taps, dt, args)
    peak = zero(dt)
    @inbounds for t in axes(nodes, 1)
        value = rate_value(nodes[t, node], args...)
        rate[t, node] = value
        peak = max(peak, value)
    end
    integrate_column!(exposure, rate, node, taps, dt)
    peak
end

"""The plasma current of each node of a chunk, gated by the window and the collar."""
function plasma_current!(p, nodes, c, count)
    T = typeof(p.dt)
    (; rate, fraction, integrand, current, gated, edge, taps, dt, derivative) = p
    window, floor = c.window, p.floor
    args = rate_arguments(p)
    charge, potential = T(p.parameters[6]), T(p.parameters[7])
    density, initial = T(p.parameters[8]), T(p.parameters[9])
    nt = size(nodes, 1)
    each_column!(c, count) do node
        peak = exposure_column!(rate, fraction, nodes, node, taps, dt, args)
        # With exactly zero rates and no initial electrons both currents vanish.
        if iszero(peak) && iszero(initial)
            @inbounds for t in 1:nt
                gated[t, node] = 0
                edge[t, node] = 0
            end
            return nothing
        end
        previous, value = T(NaN), zero(T)
        @inbounds for t in 1:nt
            exposure = floor ? max(fraction[t, node], zero(T)) : fraction[t, node]
            # An unchanged prefix gives the same fraction: the zero-rate head and the
            # constant tail cost no exponential. No threshold, no approximation.
            if exposure != previous
                value = 1 - (1 - initial) * exp(-exposure)
                previous = exposure
            end
            fraction[t, node] = value
            integrand[t, node] = value * charge * nodes[t, node]
        end
        integrate_column!(current, integrand, node, taps, dt)
        @inbounds for t in 1:nt
            field = nodes[t, node]
            squared = field * field
            loss = rate[t, node] > 0 && squared > 0 ?
                   (potential * rate[t, node] * (1 - fraction[t, node]) / squared) * field : zero(T)
            total = density * (current[t, node] + loss)
            gated[t, node] = window[t] * total
            edge[t, node] = derivative[t] * total
        end
        nothing
    end
end

clear_plasma!(::Nothing) = nothing
clear_plasma!(p) = foreach(a -> fill!(a, 0), (p.projected, p.edge_projected, p.coarse, p.coarse_edge))

function project_plasma!(p, c, first, last, quadrature)
    count = last - first + 1
    for (values, full, coarse) in ((p.gated, p.projected, p.coarse), (p.edge, p.edge_projected, p.coarse_edge))
        block = @view values[:, 1:count]
        mul!(full, block, @view(c.projection[first:last, :]), true, true)
        quadrature && c.embedded &&
            mul!(coarse, block, @view(c.projection_coarse[first:last, :]), true, true)
    end
end

"""Add `-i J / omega + edge / omega^2` of the projected currents to a spectrum."""
function add_plasma!(destination, p, plan, c; coarse=false, direct=false)
    mul!(p.spectrum, plan, direct ? p.gated : coarse ? p.coarse : p.projected)
    mul!(p.edge_spectrum, plan, direct ? p.edge : coarse ? p.coarse_edge : p.edge_projected)
    (; spectrum, edge_spectrum, inverse_omega) = p
    crop, mask = c.crop, c.mask
    each!(c, length(crop), size(destination, 2)) do f, m
        @inbounds if mask[f]
            k = crop[f]
            inverse = inverse_omega[f]
            destination[k, m] += -im * inverse * spectrum[k, m] + inverse^2 * edge_spectrum[k, m]
        end
    end
end

"""Fresh accepted-field populations, for an observation: rates and exposure only."""
function population_parts!(p, nodes, c, first, count)
    T = typeof(p.dt)
    (; rate, fraction, taps, dt, fractions, peaks, parts) = p
    floor, args = p.floor, rate_arguments(p)
    initial = T(p.parameters[9])
    nt, reach = size(nodes, 1), 2 * (length(taps) - 1)
    each_column!(c, count) do node
        peak = exposure_column!(rate, fraction, nodes, node, taps, dt, args)
        dip = rise = paid = early = largest = zero(T)
        before = product_before = start = zero(T)
        @inbounds for t in 1:nt
            exposure = fraction[t, node]
            if t > 1
                dip = max(dip, before - exposure)
                rise = max(rise, exposure - before)
            end
            before = exposure
            value = 1 - (1 - initial) * exp(-(floor ? max(exposure, zero(T)) : exposure))
            fraction[t, node] = value
            product = rate[t, node] * (1 - value)
            if t == 1
                start = value
            else
                # The diagnostic's trapezoid is ordered and strict, like the prefix.
                paid += dt * (product + product_before) / 2
            end
            product_before = product
            integrand = abs(value * nodes[t, node])
            largest = max(largest, integrand)
            t <= reach && (early = max(early, integrand))
        end
        @inbounds begin
            freed = fraction[nt, node] - start
            fractions[first+node-1] = fraction[nt, node]
            peaks[first+node-1] = peak
            parts[1, node], parts[2, node], parts[3, node] = dip, rise, abs(paid - freed)
            parts[4, node], parts[5, node], parts[6, node] = freed, early, largest
        end
        nothing
    end
    vec(maximum(Array(parts)[:, 1:count]; dims=2, init=zero(T)))
end

# --- Raman ----------------------------------------------------------------------------

function raman_workspace(c, host, nt, chunk)
    host.flag("raman") || return nothing
    T = c.T
    rows, rates = host.array("raman_undamped", Float64), host.array("raman_rates", Float64)
    dt = host.real("dt")
    kernel = zeros(2nt)
    for term in eachindex(rates), t in 1:2nt
        kernel[t] += rows[t, term] * exp(-rates[term] * (t - 1) * dt)
    end
    kernel .*= host.real("raman_fraction")
    # The inverse transform is unnormalised: its 1/(2 nt) is folded into the kernel.
    spectrum = FFTW.rfft(kernel) .* (dt / (2nt))
    drive = dzeros(c, T, 2nt, chunk)
    frequency = dzeros(c, Complex{T}, nt + 1, chunk)
    (kernel=upload(c, Complex{T}.(spectrum)), factor=host.real("raman_factor"),
     thg=host.flag("raman_thg"), drive=drive, frequency=frequency,
     forward=plan(c, plan_rfft, drive, 1; threads=c.raman_fft_threads),
     inverse=plan(c, plan_brfft, frequency, 2nt, 1; threads=c.raman_fft_threads))
end

"""Convolve each node's drive, `|E|^2` (or half the analytic signal's), with the kernel."""
function raman_drive!(r, nodes, analytic, c, count)
    nt = size(nodes, 1)
    (; drive, frequency, kernel) = r
    thg = r.thg
    each!(c, 2nt, size(drive, 2)) do t, m
        @inbounds drive[t, m] = t > nt || m > count ? zero(eltype(drive)) :
                                thg ? abs2(nodes[t, m]) : abs2(analytic[t, m]) / 2
    end
    mul!(frequency, r.forward, drive)
    each!(c, nt + 1, size(drive, 2)) do f, m
        @inbounds frequency[f, m] *= kernel[f]
    end
    mul!(drive, r.inverse, frequency)
    nothing
end
