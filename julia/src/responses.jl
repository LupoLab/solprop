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
        drive, r = raman.products ? raman.nodal : raman.drive, T(raman.factor)
        each!(c, size(nodes, 1), count) do t, m
            @inbounds begin
                v = nodes[t, m]
                nodes[t, m] = muladd(r * v, drive[t, m], polarization(values[t, m], f, a)) * window[t]
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
    # The third harmonic's term, factor h exp(2 i w0 t) E^2 E, where one is compiled.
    h, carrier = T(factor) * c.harmonic, c.carrier
    harmonic = !iszero(h)
    each!(c, size(nodes, 1), count) do t, m
        @inbounds begin
            v = nodes[t, m]
            response = f * abs2(v) * v
            if harmonic
                response += h * carrier[t] * (v * v) * v
            end
            if live
                response = muladd(r * v, real(drive[t, m]), response)
            end
            nodes[t, m] = response * window[t]
        end
    end
end

"""Replace `analytic` by the analytic signal of the real columns of `nodes`."""
function analytic_signal!(analytic, nodes, forward, inverse, c, count)
    nt, chunk = size(analytic)
    each!(c, nt, chunk) do t, m
        @inbounds analytic[t, m] = m <= count ? nodes[t, m] : 0
    end
    forward * analytic
    # The unnormalised inverse's 1/nt is folded into the one-sided spectral factor.
    T = real(eltype(analytic))
    one, two, half, none = inv(T(nt)), 2 / T(nt), nt ÷ 2, zero(T)
    each!(c, nt, chunk) do f, m
        @inbounds analytic[f, m] *= (f == 1 || f == half + 1) ? one : f <= half ? two : none
    end
    inverse * analytic
    nothing
end

# --- plasma -------------------------------------------------------------------------

function plasma_workspace(c, host, nt, chunk, nm, nn)
    host.flag("plasma") || return nothing
    T = c.T
    # Columns: the chunk's where the streams are transformed node by node (free space,
    # the spectral modal path), the modes' where they are projected first.
    by_node = c.free || (haskey(c, :spectral) && c.spectral)
    columns = by_node ? 0 : nm
    # Several channels (a mixture's species): each its rate table and constants, the
    # first's also as the workspace's own fields, so that one channel runs as before.
    count = host.has("plasma_count") ? host.int("plasma_count") : 1
    suffix(k) = k == 1 ? "" : "_$k"
    channels = [(coefficients=upload(c, host.array("plasma_coefficients" * suffix(k), T)),
                 parameters=host.array("plasma_parameters" * suffix(k), Float64)) for k in 1:count]
    fractions_all, peaks_all = dzeros(c, T, nn, count), dzeros(c, T, nn, count)
    (
        channels=channels, fractions_all=fractions_all, peaks_all=peaks_all,
        coefficients=channels[1].coefficients,
        # Field range, rate cap, log-table origin and step, charge-mass ratio and
        # ionisation potential; density and preionisation are a stage's (stages.jl).
        parameters=channels[1].parameters,
        taps=upload(c, host.array("plasma_taps", T)), floor=host.flag("plasma_floor"),
        dt=T(host.real("dt")), field_scale=T(c.scale),
        inverse_omega=upload(c, T.(inv.(host.array("omega", Float64)))),
        derivative=upload(c, host.array("window_derivative", T)),
        rate=dzeros(c, T, nt, chunk), fraction=dzeros(c, T, nt, chunk),
        # A CPU column's live samples, by index, and their fields (`exposure_column!`).
        live=zeros(Int32, oncpu(c) ? nt : 0, oncpu(c) ? chunk : 0),
        scratch=zeros(T, oncpu(c) ? nt : 0, oncpu(c) ? chunk : 0),
        knots=zeros(Int32, oncpu(c) ? nt : 0, oncpu(c) ? chunk : 0),
        integrand=dzeros(c, T, nt, chunk), current=dzeros(c, T, nt, chunk),
        gated=dzeros(c, T, nt, chunk), edge=dzeros(c, T, nt, chunk),
        projected=dzeros(c, T, nt, columns), edge_projected=dzeros(c, T, nt, columns),
        coarse=dzeros(c, T, nt, columns), coarse_edge=dzeros(c, T, nt, columns),
        spectrum=dzeros(c, Complex{T}, nt ÷ 2 + 1, by_node ? chunk : nm),
        edge_spectrum=dzeros(c, Complex{T}, nt ÷ 2 + 1, by_node ? chunk : nm),
        fractions=view(fractions_all, :, 1), peaks=view(peaks_all, :, 1),
        parts=dzeros(c, T, 6, chunk),
    )
end

"""The piecewise-cubic ionisation rate in log |E|, clamped to the table's range."""
@inline function rate_value(value, table, lower, upper, cap, origin, step)
    x = abs(value)
    x < lower && return zero(cap)
    x >= upper && return cap
    table_rate(x, table, cap, origin, step)
end
# The vectorised passes are functions of their own: inside a column's closure the
# loops stayed scalar. The table's lookup is a pass between them, since a gather keeps
# LLVM from vectorising a loop: argon's live rates 39 ns a sample -> 10 ns.
@noinline function live_rates!(scratch, knots, node, count, table, cap, origin, step)
    last = Int32(size(table, 2))
    # The logarithm, then each sample's knot and its distance past it, vectorised; only
    # the table's four loads and the cubic stay scalar (`log_rate`'s cubic, contracted
    # the same way, so the two paths agree).
    @inbounds @simd for i in 1:count
        logarithm = vlog(scratch[i, node])
        j = clamp(unsafe_trunc(Int32, floor((logarithm - origin) / step)) + Int32(1),
                  Int32(1), last)
        knots[i, node] = j
        scratch[i, node] = logarithm - (origin + (j - 1) * step)
    end
    @inbounds for i in 1:count
        j, s = knots[i, node], scratch[i, node]
        scratch[i, node] = cubic(table, j, s)
    end
    @inbounds @simd for i in 1:count
        scratch[i, node] = min(vexp(scratch[i, node]), cap)
    end
end
"""Every sample's fraction and the free current's integrand, vectorised (`vexp`).

Before the column's support the exposure is exactly zero, and past it exactly its
last value: each run takes the one fraction its exposure gives, formed once, so only
the samples between take an exponential each (the values are the same bit for bit).
"""
@noinline function fraction_column!(fraction, integrand, nodes, node, initial, charge, floor)
    none = zero(initial)
    nt = size(nodes, 1)
    fraction_of(exposure) = 1 - (1 - initial) * vexp(-(floor ? max(exposure, none) : exposure))
    head = 0
    @inbounds while head < nt && iszero(fraction[head+1, node])
        head += 1
    end
    tail = nt + 1
    @inbounds last = fraction[nt, node]
    @inbounds while tail > head + 1 && fraction[tail-1, node] === last
        tail -= 1
    end
    before, after = fraction_of(none), fraction_of(last)
    @inbounds @simd for t in 1:head
        fraction[t, node] = before
        integrand[t, node] = before * charge * nodes[t, node]
    end
    @inbounds @simd for t in head+1:tail-1
        value = fraction_of(fraction[t, node])
        fraction[t, node] = value
        integrand[t, node] = value * charge * nodes[t, node]
    end
    @inbounds @simd for t in tail:nt
        fraction[t, node] = after
        integrand[t, node] = after * charge * nodes[t, node]
    end
end
"""The gated and edge currents of a column: the free current plus the loss current,
formed everywhere and kept where the rate and the field are live."""
@noinline function currents_column!(gated, edge, nodes, rate, fraction, current, window,
                                    derivative, node, potential, density, add=false)
    none = zero(potential)
    @inbounds @simd for t in axes(nodes, 1)
        field, value = nodes[t, node], rate[t, node]
        squared = field * field
        loss = ifelse((value > 0) & (squared > 0),
                      (potential * value * (1 - fraction[t, node]) / squared) * field, none)
        total = density * (current[t, node] + loss)
        if add
            gated[t, node] += window[t] * total
            edge[t, node] += derivative[t] * total
        else
            gated[t, node] = window[t] * total
            edge[t, node] = derivative[t] * total
        end
    end
end
"""The rate inside the table's span, `lower <= x < upper` (or NaN)."""
@inline table_rate(x, table, cap, origin, step) =
    min(exp(log_rate(log(x), table, origin, step)), cap)
"""The table's `log W` at `log |E|`: the knot's cubic in the distance past it."""
@inline function log_rate(logarithm, table, origin, step)
    j = clamp(unsafe_trunc(Int, floor((logarithm - origin) / step)) + 1, 1, size(table, 2))
    s = logarithm - (origin + (j - 1) * step)
    cubic(table, j, s)
end
"""Knot `j`'s cubic at `s` by Horner's rule, contracted (SEP 25 permits fused
multiply-adds in an ordinary kernel)."""
@inline cubic(table, j, s) =
    @inbounds muladd(muladd(muladd(table[1, j], s, table[2, j]), s, table[3, j]), s, table[4, j])
@inline rate_arguments(p) =
    (p.field_scale, p.coefficients, ntuple(k -> typeof(p.dt)(p.parameters[k]), 5)...)
"""The rate at a scaled field's physical value `s E`: exact, s a power of two."""
@inline scaled_rate(value, scale, args...) = rate_value(scale * value, args...)

# A symmetric FIR, reflected at the ends as RunningIntegral is, then a running
# trapezoid. The scan runs in order on a CPU; that is a choice, not policy: SEP 25 (5,
# item 5) permits a reassociated or segmented scan within its prefix's rounding bound,
# as the GPU's column scans are. The sums contract.
@inline function filtered(values, t, node, taps)
    nt = size(values, 1)
    @inbounds value = taps[1] * values[t, node]
    @inbounds for k in 1:length(taps)-1
        left = t - k < 1 ? 2 * values[1, node] - values[1+k-t+1, node] : values[t-k, node]
        right = t + k > nt ? 2 * values[nt, node] - values[2nt-t-k, node] : values[t+k, node]
        value = muladd(taps[k+1], left + right, value)
    end
    value
end

"""The interior samples' FIR, one pass: each sample's sum in the order the taps give it
(the centre, then each pair), contracted as `filtered` contracts it, with a fifth of
the tap-at-a-time passes' memory traffic. The taps are a tuple, so the sum unrolls and
the samples vectorise."""
@noinline function fir_interior!(out, values, node, first, last, taps::NTuple{N}) where {N}
    @inbounds @simd for t in first:last
        value = taps[1] * values[t, node]
        for k in 1:N-1
            value = muladd(taps[k+1], values[t-k, node] + values[t+k, node], value)
        end
        out[t, node] = value
    end
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
    @inbounds for t in first:interior_first-1
        out[t, node] = filtered(values, t, node, taps)
    end
    @inbounds for t in interior_last+1:last
        out[t, node] = filtered(values, t, node, taps)
    end
    fir_interior!(out, values, node, interior_first, interior_last, Tuple(taps))
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

"""One column's rates and exposure; return the peak rate.

`scaled_rate` at every sample, in three passes: the samples whose rate is zero or the
cap are written and the others gathered (their index into `live`, their field into
`scratch`); the gathered ones are evaluated in a loop the CPU vectorises, with
`vlog` and `vexp`; and scattered back. The carrier takes the field across the
table's lower edge twice a cycle, a branch no predictor follows, which kept the
logarithm and exponential scalar and discarded them in flight (argon's rates 2.1 ms
-> 0.7 ms a source on one thread, its plasma 4.7 ms -> 2.9 ms). A NaN field is live
and stays NaN.
"""
@inline function exposure_column!(rate, exposure, nodes, node, taps, dt, args, live, scratch, knots)
    scale, table, lower, upper, cap = args[1], args[2], args[3], args[4], args[5]
    origin, step = args[6], args[7]
    none = zero(dt)
    live_count = 0
    @inbounds for t in axes(nodes, 1)
        x = abs(scale * nodes[t, node])
        rate[t, node] = x < lower ? none : cap
        live[live_count+1, node] = t
        scratch[live_count+1, node] = x
        live_count += !(x < lower) & !(x >= upper)
    end
    live_rates!(scratch, knots, node, live_count, table, cap, origin, step)
    @inbounds for i in 1:live_count
        rate[live[i, node], node] = scratch[i, node]
    end
    peak = none
    @inbounds for t in axes(nodes, 1)
        peak = max(peak, rate[t, node])
    end
    integrate_column!(exposure, rate, node, taps, dt)
    peak
end

"""The plasma current of each node of a chunk, gated by the window and the collar.

On a CPU each node's column is one task, scanned in order with its exact zero support
skipped. On a GPU, where a column a thread leaves most of the device idle, every
pass is a sample a thread and the two running integrals are column-wise scans: their
sums are reassociated, which SEP 25 permits and the tests bound, as native's
two-level device scans do.
"""
plasma_current!(p, nodes, c, count) = each_channel(p) do q, add
    oncpu(c) && !SAMPLE_PLASMA[] ? plasma_columns!(q, nodes, c, count, add) :
    plasma_samples!(q, nodes, c, count, add)
end

"""Call `f(q, add)` for each plasma channel: `q` the workspace with the channel's rate
table, constants and population tables in place of the first's, `add` whether its
currents add to those the channels before it formed."""
function each_channel(f::F, p) where {F}
    for (k, channel) in enumerate(p.channels)
        q = k == 1 ? p :
            merge(p, (coefficients=channel.coefficients, parameters=channel.parameters,
                      fractions=view(p.fractions_all, :, k), peaks=view(p.peaks_all, :, k)))
        f(q, k > 1)
    end
end

"""The channels' population tables for the host: a vector for one channel, a
(channel, node) matrix for several."""
channel_rows(a) = size(a, 2) == 1 ? Vector{Float64}(Array(a)[:, 1]) :
                  Matrix{Float64}(permutedims(Array(a)))
# SOLPROP_JULIA_PLASMA=samples takes the GPU's path on a CPU too.
const SAMPLE_PLASMA = Ref(false)

"""`out` = the running trapezoid of the FIR-filtered `values`, every column at once."""
function integrate_samples!(out, values, taps, dt, c, count)
    nt = size(values, 1)
    half, none = dt / 2, zero(dt)
    # Each trapezoid increment filters its two samples itself: the same values as a
    # filtered pass and then an increments pass, one kernel and one memory pass fewer.
    each!(c, nt, count) do t, n
        @inbounds out[t, n] = t == 1 ? none :
                              half * (filtered(values, t, n, taps) + filtered(values, t - 1, n, taps))
    end
    tally!(:scan)
    AK.accumulate!(+, @view(out[:, 1:count]); dims=1, init=zero(dt), backend=c.backend)
    nothing
end

function plasma_samples!(p, nodes, c, count, add=false)
    T = typeof(p.dt)
    (; rate, fraction, integrand, current, gated, edge, taps, dt, derivative) = p
    window, floor = c.window, p.floor
    args = rate_arguments(p)
    charge, potential = T(p.parameters[6]), T(p.parameters[7])
    density, initial = T(p.parameters[8]), T(p.parameters[9])
    nt, none = size(nodes, 1), zero(T)
    each!(c, nt, count) do t, n
        @inbounds rate[t, n] = scaled_rate(nodes[t, n], args...)
    end
    integrate_samples!(fraction, rate, taps, dt, c, count)
    each!(c, nt, count) do t, n
        @inbounds begin
            exposure = floor ? max(fraction[t, n], none) : fraction[t, n]
            value = 1 - (1 - initial) * exp(-exposure)
            fraction[t, n] = value
            integrand[t, n] = value * charge * nodes[t, n]
        end
    end
    integrate_samples!(current, integrand, taps, dt, c, count)
    each!(c, nt, count) do t, n
        @inbounds begin
            field = nodes[t, n]
            squared = field * field
            loss = rate[t, n] > 0 && squared > 0 ?
                   (potential * rate[t, n] * (1 - fraction[t, n]) / squared) * field : none
            total = density * (current[t, n] + loss)
            if add
                gated[t, n] += window[t] * total
                edge[t, n] += derivative[t] * total
            else
                gated[t, n] = window[t] * total
                edge[t, n] = derivative[t] * total
            end
        end
    end
end

function plasma_columns!(p, nodes, c, count, add=false)
    T = typeof(p.dt)
    (; rate, fraction, integrand, current, gated, edge, taps, dt, derivative, live, scratch, knots) = p
    window, floor = c.window, p.floor
    args = rate_arguments(p)
    charge, potential = T(p.parameters[6]), T(p.parameters[7])
    density, initial = T(p.parameters[8]), T(p.parameters[9])
    nt = size(nodes, 1)
    none = zero(T)
    each_column!(c, count) do node
        peak = exposure_column!(rate, fraction, nodes, node, taps, dt, args, live, scratch, knots)
        # With exactly zero rates and no initial electrons both currents vanish.
        if iszero(peak) && iszero(initial)
            # (A later channel adds nothing.)
            add || @inbounds for t in 1:nt
                gated[t, node] = 0
                edge[t, node] = 0
            end
            return nothing
        end
        fraction_column!(fraction, integrand, nodes, node, initial, charge, floor)
        integrate_column!(current, integrand, node, taps, dt)
        currents_column!(gated, edge, nodes, rate, fraction, current, window, derivative,
                         node, potential, density, add)
        nothing
    end
end

clear_plasma!(::Nothing) = nothing
clear_plasma!(p) = foreach(a -> fill!(a, 0), (p.projected, p.edge_projected, p.coarse, p.coarse_edge))

function project_plasma!(p, c, first, last, quadrature, alpha=true)
    count = last - first + 1
    for (values, full, coarse) in ((p.gated, p.projected, p.coarse), (p.edge, p.edge_projected, p.coarse_edge))
        block = @view values[:, 1:count]
        gemm!(full, block, table_rows(c, c.projection, first, last), alpha, true, c)
        quadrature && c.embedded &&
            gemm!(coarse, block, table_rows(c, c.projection_coarse, first, last), alpha, true, c)
    end
end

"""Add `-i J / omega + edge / omega^2` of the projected currents to a spectrum."""
function add_plasma!(destination, p, plan, c; coarse=false, direct=false)
    mul!(p.spectrum, plan, direct ? p.gated : coarse ? p.coarse : p.projected)
    mul!(p.edge_spectrum, plan, direct ? p.edge : coarse ? p.coarse_edge : p.edge_projected)
    add_plasma_spectra!(destination, p, c)
end
"""Add `-i J / omega + edge / omega^2` of the currents' spectra, already transformed."""
function add_plasma_spectra!(destination, p, c; coarse=false)
    # The coarse quadrature's currents (a GPU's packed streams), or the source's.
    spectrum, edge_spectrum = coarse ? (p.coarse_spectrum, p.coarse_edge_spectrum) :
                                       (p.spectrum, p.edge_spectrum)
    inverse_omega = p.inverse_omega
    crop, mask = c.crop, c.mask
    each!(c, length(crop), size(destination, 2)) do f, m
        @inbounds if mask[f]
            k = crop[f]
            inverse = inverse_omega[f]
            destination[k, m] += -im * inverse * spectrum[k, m] + inverse^2 * edge_spectrum[k, m]
        end
    end
end

"""Fresh accepted-field populations, for an observation: rates and exposure only.

Return the six diagnostics' maxima over the chunk (`plasma_integration_statistics`),
and set its nodes' final fractions and peak rates. A GPU forms them a sample a thread,
with the column diagnostics as array reductions, as it forms the current.
"""
population_parts!(p, nodes, c, first, count) =
    oncpu(c) && !SAMPLE_PLASMA[] ? population_columns!(p, nodes, c, first, count) :
    population_samples!(p, nodes, c, first, count)

function population_samples!(p, nodes, c, first, count)
    T = typeof(p.dt)
    (; rate, fraction, integrand, current, gated, edge, taps, dt, fractions, peaks) = p
    floor, args = p.floor, rate_arguments(p)
    initial, none = T(p.parameters[9]), zero(T)
    nt, reach = size(nodes, 1), 2 * (length(taps) - 1)
    each!(c, nt, count) do t, n
        @inbounds rate[t, n] = scaled_rate(nodes[t, n], args...)
    end
    integrate_samples!(fraction, rate, taps, dt, c, count)
    columns = 1:count
    # The exposure's steps, then its fractions, the depletion they pay and the integrand.
    each!(c, nt, count) do t, n
        @inbounds integrand[t, n] = t == 1 ? none : fraction[t, n] - fraction[t-1, n]
    end
    each!(c, nt, count) do t, n
        @inbounds begin
            exposure = fraction[t, n]
            value = 1 - (1 - initial) * exp(-(floor ? max(exposure, none) : exposure))
            fraction[t, n] = value
            gated[t, n] = rate[t, n] * (1 - value)
            current[t, n] = abs(value * nodes[t, n])
        end
    end
    each!(c, nt, count) do t, n
        @inbounds edge[t, n] = t == 1 ? none : dt * (gated[t, n] + gated[t-1, n]) / 2
    end
    # The columns' work sums and peak rates as device reductions, each column's final
    # fraction, peak and defect by a small kernel, and the six maxima as one reduction
    # over every sample, read back once: the separate reductions and view copies were
    # seven waits and two slow copies a chunk (each about 0.2 ms on Metal). A thread a
    # column, looping down it, was slower still: a capillary has a few columns.
    paid = scratch(c, :population_paid, T, 1, count)
    peak = scratch(c, :population_peak, T, 1, count)
    sum!(paid, @view(edge[:, columns]))
    maximum!(peak, @view(rate[:, columns]))
    tally!(:launch, 2)
    defects = scratch(c, :population_defects, T, 2, count)
    each!(c, count, 1) do n, _
        @inbounds begin
            freed = fraction[nt, n] - fraction[1, n]
            fractions[first+n-1] = fraction[nt, n]
            peaks[first+n-1] = peak[1, n]
            defects[1, n], defects[2, n] = abs(paid[1, n] - freed), freed
        end
    end
    head, lowest = min(reach, nt), typemin(T)
    maxima = reduce_each(tuple_max, (none, none, none, lowest, none, none), c, nt, count) do t, n
        @inbounds begin
            step, value = integrand[t, n], current[t, n]
            first_row = t == 1
            (-step, step, first_row ? defects[1, n] : none, first_row ? defects[2, n] : lowest,
             t <= head ? value : none, value)
        end
    end
    Float64[max(0, maxima[1]), max(0, maxima[2]), maxima[3], maxima[4], maxima[5], maxima[6]]
end
"""The elementwise maximum of two tuples, for `reduce_each`."""
tuple_max(a, b) = map(max, a, b)

function population_columns!(p, nodes, c, first, count)
    T = typeof(p.dt)
    (; rate, fraction, taps, dt, fractions, peaks, parts, live, scratch, knots) = p
    floor, args = p.floor, rate_arguments(p)
    initial = T(p.parameters[9])
    nt, reach = size(nodes, 1), 2 * (length(taps) - 1)
    none = zero(T)
    each_column!(c, count) do node
        peak = exposure_column!(rate, fraction, nodes, node, taps, dt, args, live, scratch, knots)
        dip = rise = paid = early = largest = none
        before = product_before = start = none
        @inbounds for t in 1:nt
            exposure = fraction[t, node]
            if t > 1
                dip = max(dip, before - exposure)
                rise = max(rise, exposure - before)
            end
            before = exposure
            value = 1 - (1 - initial) * exp(-(floor ? max(exposure, none) : exposure))
            fraction[t, node] = value
            product = rate[t, node] * (1 - value)
            if t == 1
                start = value
            else
                # The diagnostic's trapezoid stays ordered and uncontracted: it is a
                # defect, a difference of nearly equal sums (strict by SEP 25).
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

"""The Raman workspace: the kernel's spectrum and the drives' transforms.

With fewer of the M (M + 1) / 2 modal products than nodes, each product of the modes'
time signals is convolved once and a node's convolved drive is their sum weighted by
its synthesis (native's ``raman_modal_products``): the convolution is linear, so this
agrees with convolving the node's own drive to the rounding of the drive's scale.
"""
function raman_workspace(c, host, nt, chunk; modes=0, nodes=0, synthesis=nothing)
    host.flag("raman") || return nothing
    T, C = c.T, Complex{c.T}
    rows, rates = host.array("raman_undamped", Float64), host.array("raman_rates", Float64)
    dt = host.real("dt")
    # Each term weighted by its component's density and fraction (several components,
    # summed into one kernel), or all by one if the adapter gives none.
    weights = host.has("raman_weights") ? host.array("raman_weights", Float64) : ones(length(rates))
    kernel = zeros(2nt)
    for term in eachindex(rates), t in 1:2nt
        kernel[t] += weights[term] * rows[t, term] * exp(-rates[term] * (t - 1) * dt)
    end
    kernel .*= host.real("raman_fraction")
    # The inverse transform is unnormalised: its 1/(2 nt) is folded into the kernel.
    # The spectrum is held as `spectrum * 2^-e`, its largest entry in [1, 2), and the
    # response's factor takes 2^e (SEP 33, rule 1: a block exponent): in binary32 the
    # kernel's small entries are otherwise subnormal, a 1.7 % error in an H2 window.
    spectrum = FFTW.rfft(kernel) .* (dt / (2nt))
    e = exponent(maximum(abs, spectrum))
    spectrum .*= exp2(-e)
    count = modes * (modes + 1) ÷ 2
    products = !c.envelope && synthesis !== nothing && nodes >= 2 && count < nodes && count <= chunk
    columns = products ? count : chunk
    drive = dzeros(c, T, 2nt, columns)
    frequency = dzeros(c, C, nt + 1, columns)
    thg = host.flag("raman_thg")
    extra = (;)
    if products
        pairs = [(m, n) for m in 1:modes for n in m:modes]
        weights = [synthesis[m, j] * synthesis[n, j] * (m == n ? 1 : 2) for (m, n) in pairs, j in 1:nodes]
        signal = dzeros(c, C, thg ? 0 : nt, thg ? 0 : modes)
        extra = (weights=upload(c, T.(weights)), nodal=dzeros(c, T, nt, chunk),
                 first=upload(c, first.(pairs)), second=upload(c, last.(pairs)), signal=signal,
                 signal_forward=thg ? nothing : plan(c, plan_fft!, signal, 1),
                 signal_inverse=thg ? nothing : plan(c, plan_bfft!, signal, 1))
    end
    (kernel=upload(c, C.(spectrum)), factor=host.real("raman_factor") * exp2(e), thg=thg,
     products=products, drive=drive, frequency=frequency, extra...,
     forward=plan(c, plan_rfft, drive, 1; threads=c.raman_fft_threads),
     inverse=plan(c, plan_brfft, frequency, 2nt, 1; threads=c.raman_fft_threads))
end

"""Convolve the drive's columns with the Raman kernel in place."""
function convolve!(r, c)
    (; drive, frequency, kernel) = r
    @section :raman_forward mul!(frequency, r.forward, drive)
    each!(c, size(frequency)...) do f, m
        @inbounds frequency[f, m] *= kernel[f]
    end
    @section :raman_inverse mul!(drive, r.inverse, frequency)
    nothing
end

"""Convolve each node's drive, `|E|^2` (or half the analytic signal's), with the kernel."""
function raman_drive!(r, nodes, analytic, c, count)
    nt = size(nodes, 1)
    drive, thg = r.drive, r.thg
    each!(c, 2nt, size(drive, 2)) do t, m
        @inbounds drive[t, m] = t > nt || m > count ? zero(eltype(drive)) :
                                thg ? abs2(nodes[t, m]) : abs2(analytic[t, m]) / 2
    end
    convolve!(r, c)
end

"""Convolve the modal products once a source: `a_m a_n`, or `Re(a_m conj a_n) / 2`."""
function raman_products!(r, modal, c; signal=nothing)
    nt, modes = size(modal)
    (; drive, first, second) = r
    if r.thg
        each!(c, 2nt, size(drive, 2)) do t, p
            @inbounds drive[t, p] = t > nt ? zero(eltype(drive)) : modal[t, first[p]] * modal[t, second[p]]
        end
    else
        # The modes' analytic signals: given (formed from the spectrum), or from `modal`.
        if signal === nothing
            signal = r.signal
            analytic_signal!(signal, modal, r.signal_forward, r.signal_inverse, c, modes)
        end
        each!(c, 2nt, size(drive, 2)) do t, p
            @inbounds begin
                a, b = signal[t, first[p]], signal[t, second[p]]
                drive[t, p] = t > nt ? zero(eltype(drive)) : (real(a) * real(b) + imag(a) * imag(b)) / 2
            end
        end
    end
    convolve!(r, c)
end

"""A chunk's convolved node drives, rebuilt from the convolved products."""
raman_nodal!(r, first, last, c, alpha=true) =
    gemm!(@view(r.nodal[:, 1:last-first+1]), @view(r.drive[1:size(r.nodal, 1), :]),
          @view(r.weights[:, first:last]), alpha, false, c)
