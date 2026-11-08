# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The envelope grid's plasma (translate/envelope_plasma.py; the oracle is
# numpy_ref/envelope_plasma.py). Each channel's cycle-averaged rate comes from the
# instantaneous table by angular quadrature over the field's polarisation ellipse;
# the population is the running trapezoid of that rate; two currents follow, the
# Drude drive and the loss, each windowed, projected and transformed as the field
# grid's two plasma streams are, and joined with -1/omega^2 and -i/omega.
#
# A scalar envelope's ellipse is a line: its major axis |E| and its minor/major ratio
# zero, so the cosine moment Mc is the rate and the loss current is 2 Mc E / |E|^2.

"""The envelope plasma's workspace: the field plasma's fields where the shared paths
read them (the channels, the population tables, the projected and transformed
streams), complex, and the cycle quadrature's rules and spectral factors."""
function envelope_plasma_workspace(c, host, nt, chunk, nm, nn)
    host.flag("envelope_plasma") || return nothing
    T, C = c.T, Complex{c.T}
    by_node = c.free
    columns = by_node ? 0 : nm
    count = host.int("plasma_count")
    suffix(k) = k == 1 ? "" : "_$k"
    channels = [precise_table(c, host.array("plasma_coefficients" * suffix(k), Float64),
                              host.array("plasma_parameters" * suffix(k), Float64)) for k in 1:count]
    fractions_all, peaks_all = dzeros(c, T, nn, count), dzeros(c, T, nn, count)
    # The rules of orders 64, 128, 256 and 512 end to end; `levels` each one's end.
    nodes, weights = host.array("envelope_rule_nodes", Float64), host.array("envelope_rule_weights", Float64)
    ends = Tuple(cumsum(Int.(host.array("envelope_rule_orders", Int))))
    nf = c.nf
    (
        envelope=Val(true), channels=channels, fractions_all=fractions_all, peaks_all=peaks_all,
        channel=1,
        coefficients=channels[1].coefficients, parameters=channels[1].parameters,
        constant_tail=channels[1].constant_tail, inverse_edges=channels[1].inverse_edges,
        inverse_tail=channels[1].inverse_tail,
        dt=T(host.real("dt")), field_scale=T(c.scale),
        rule_nodes=upload(c, T.(nodes)), rule_weights=upload(c, T.(weights)), levels=ends,
        drude=upload(c, T.(host.array("envelope_drude", Float64))),
        loss_factor=upload(c, C.(host.array("envelope_loss", ComplexF64))),
        rate=dzeros(c, T, nt, chunk), exposure=dzeros(c, T, nt, chunk),
        gated=dzeros(c, C, nt, chunk), edge=dzeros(c, C, nt, chunk),
        projected=dzeros(c, C, nt, columns), edge_projected=dzeros(c, C, nt, columns),
        coarse=dzeros(c, C, nt, columns), coarse_edge=dzeros(c, C, nt, columns),
        spectrum=dzeros(c, C, nf, by_node ? chunk : nm),
        edge_spectrum=dzeros(c, C, nf, by_node ? chunk : nm),
        fractions=view(fractions_all, :, 1), peaks=view(peaks_all, :, 1),
    )
end

"""A channel's table for `precise_rate`: the cubic at the state's precision, with its
constant term's tail, and each interval's inverse left edge as a head and a tail (the
mlx backend's DeviceRateTable)."""
function precise_table(c, coefficients, parameters)
    T = c.T
    n = size(coefficients, 2)
    edges = exp.(parameters[4] .+ (0:n-1) .* parameters[5])
    inverse = inv.(edges)
    (coefficients=upload(c, T.(coefficients)), parameters=parameters,
     constant_tail=upload(c, T.(coefficients[4, :] .- Float64.(T.(coefficients[4, :])))),
     inverse_edges=upload(c, T.(inverse)), inverse_tail=upload(c, T.(inverse .- Float64.(T.(inverse)))))
end

"""The rate at a field inside the table's span, accurate to the state's precision: the
local coordinate `log1p(E / E_i - 1)` against the interval's left edge (a binary32
`log E` would carry `|log E| u`, which the rate's slope multiplies), and the constant
term added by an exact two-sum and its tail, the rate `exp(hi) (1 + lo)` (the mlx
backend's `rate_at`). The cycle quadrature's refinement needs it: two rules agree to
2e-6, which a binary32 rate formed through `log E` misses by an order."""
@inline function precise_rate(e, table, tail, inverse, inverse_tail, cap, origin, step)
    k = clamp(unsafe_trunc(Int, floor((log(e) - origin) / step)) + 1, 1, size(table, 2))
    @inbounds begin
        p = e * inverse[k]
        error = fma(e, inverse[k], -p) + e * inverse_tail[k]
        x = (p - 1) + error
        u = 1 + x
        coordinate = u == 1 ? x : log(u) * x / (u - 1)
        constant = table[4, k]
        remainder = ((table[1, k] * coordinate + table[2, k]) * coordinate + table[3, k]) * coordinate
        head = constant + remainder
        shifted = head - constant
        low = (constant - (head - shifted)) + (remainder - shifted) + tail[k]
        min(exp(head) * (1 + low), cap)
    end
end

"""`(sin, cos)` of an angle in [0, pi/2] by Taylor polynomials on [0, pi/4] (the
other half by the complement): about one ulp on every device, where a GPU's library
gives two or three (Metal), which near the table's low cutoff, where the rate is
steepest in the field, is more than the refinement's 2e-6 can take. Binary64 keeps the
library's."""
@inline quadrant_sincos(theta::Float64) = sincos(theta)
@inline function quadrant_sincos(theta::T) where {T}
    complement = theta > T(pi) / 4
    x = complement ? T(pi) / 2 - theta : theta
    x2 = x * x
    sine = x * evalpoly(x2, (T(1), T(-1 / 6), T(1 / 120), T(-1 / 5040), T(1 / 362880),
                             T(-1 / 39916800)))
    cosine = evalpoly(x2, (T(1), T(-1 / 2), T(1 / 24), T(-1 / 720), T(1 / 40320),
                           T(-1 / 3628800), T(1 / 479001600)))
    complement ? (cosine, sine) : (sine, cosine)
end

is_envelope(p) = false
is_envelope(p::NamedTuple) = haskey(p, :envelope)

"""One panel's two directional moments, `(upper - lower)/pi` times the rule's sums of
`W(major sqrt(d)) / d` against cos^2 and sin^2, `d = cos^2 + ratio^2 sin^2` (zero
ratio here). A panel's kind replaces the table's cutoffs: before the high cut the
ceiling rate, between the cuts the field held inside the table's span (binary32's
rounding of a cut angle cannot put a node across a cutoff). The sums are compensated."""
@inline function panel_moments(major, lower, upper, ceiling, args, nodes, weights, first, last)
    T = typeof(major)
    half = (upper - lower) / 2
    s0, e0, s1, e1 = zero(T), zero(T), zero(T), zero(T)
    table, tail, inverse, inverse_tail, low, high, cap, origin, step = args
    @inbounds for i in first:last
        theta = muladd(nodes[i] + 1, half, lower)
        sine, cosine = quadrant_sincos(theta)
        c2, s2 = cosine * cosine, sine * sine
        rate = ceiling ? cap :
               precise_rate(clamp(major * sqrt(c2), low, high), table, tail, inverse, inverse_tail, cap, origin, step)
        r = rate / c2
        a, b = weights[i] * r * c2, weights[i] * r * s2
        # Neumaier's compensated sums.
        t0 = s0 + a
        e0 += abs(s0) >= abs(a) ? (s0 - t0) + a : (a - t0) + s0
        s0 = t0
        t1 = s1 + b
        e1 += abs(s1) >= abs(b) ? (s1 - t1) + b : (b - t1) + s1
        s1 = t1
    end
    width = (upper - lower) / T(pi)
    (width * (s0 + e0), width * (s1 + e1))
end

"""The rule of `level`'s moments: split at the angles where `major cos(theta)` crosses
the table's cutoffs; the panel past the low cut contributes nothing."""
@inline function rule_moments(major, args, nodes, weights, first, last)
    T = typeof(major)
    low, high = args[5], args[6]
    quarter = T(pi) / 2
    # Past the low cut's angle the field is below the table: zero.
    cut_low = low / major < 1 ? acos(low / major) : quarter
    cut_high = high / major < 1 ? acos(high / major) : zero(T)
    a0, a1 = panel_moments(major, zero(T), cut_high, true, args, nodes, weights, first, last)
    b0, b1 = panel_moments(major, cut_high, cut_low, false, args, nodes, weights, first, last)
    (a0 + b0, a1 + b1)
end

"""The cosine moment of a scalar envelope of physical amplitude `major`, refined until
two successive rules agree within 2e-6 (floor 1e-30 s^-1); NaN where the 512-point rule
does not converge (the oracle's ConvergenceError: the step's norm refuses it)."""
@inline function cycle_rate(major, args, nodes, weights, levels)
    T = typeof(major)
    low = args[5]
    major < low && return zero(T)
    budget, floor = T(2e-6), T(1e-30)
    previous = rule_moments(major, args, nodes, weights, 1, levels[1])
    for k in 2:length(levels)
        current = rule_moments(major, args, nodes, weights, levels[k-1] + 1, levels[k])
        # The cosine moment only: at a scalar envelope's zero ratio the sine moment
        # enters neither the rate nor the current, and its integrand (the rate times
        # tan^2) is the one binary32 misses the budget on near the low cut. (The oracle
        # asks both; a deliberate difference, recorded in julia/STATUS.md.)
        if isfinite(current[1]) && abs(current[1] - previous[1]) <= budget * max(current[1], floor)
            return current[1]
        end
        previous = current
    end
    T(NaN)
end

"""The chunk's cycle-averaged rates and their running trapezoid, per channel."""
function envelope_exposure!(q, nodes, c, count)
    (; rate, exposure, dt, rule_nodes, rule_weights, levels) = q
    T = typeof(dt)
    args = (q.coefficients, q.constant_tail, q.inverse_edges, q.inverse_tail,
            ntuple(k -> T(q.parameters[k]), 5)...)
    scale = q.field_scale
    nt = size(nodes, 1)
    each!(c, nt, count) do t, n
        @inbounds rate[t, n] = cycle_rate(scale * abs(nodes[t, n]), args, rule_nodes, rule_weights, levels)
    end
    half, none = dt / 2, zero(dt)
    each!(c, nt, count) do t, n
        @inbounds exposure[t, n] = t == 1 ? none : muladd(half, rate[t-1, n], half * rate[t, n])
    end
    tally!(:scan)
    scan_columns!(@view(exposure[:, 1:count]), c, none)
    nothing
end

"""The chunk's windowed drive and loss currents, every channel's summed."""
function envelope_current!(p, nodes, c, count)
    window = c.window
    each_channel(p) do q, add
        envelope_exposure!(q, nodes, c, count)
        (; rate, exposure, gated, edge) = q
        T = typeof(q.dt)
        charge, potential = T(q.parameters[6]), T(q.parameters[7])
        k = q.channel
        dens = stage_scalar(c, 2k, q.parameters[8])
        init = stage_scalar(c, 2k + 1, q.parameters[9])
        none = zero(eltype(gated))
        each!(c, size(nodes, 1), count) do t, n
            @inbounds begin
                density, initial = stage_value(dens), stage_value(init)
                v, e = nodes[t, n], exposure[t, n]
                survival = (1 - initial) * exp(-e)
                fraction = initial + (1 - initial) * -expm1(-e)
                squared = abs2(v)
                drive = (charge * density * fraction) * v
                loss = squared > 0 ? (potential * density * survival) * (2 * rate[t, n] / squared) * v : none
                if add
                    gated[t, n] += window[t] * drive
                    edge[t, n] += window[t] * loss
                else
                    gated[t, n] = window[t] * drive
                    edge[t, n] = window[t] * loss
                end
            end
        end
    end
    nothing
end

"""Add the drive's and the loss's spectra times -1/omega^2 and -i/omega (zero off the
supported band)."""
function add_envelope_spectra!(destination, p, c; coarse=false)
    spectrum, edge_spectrum = p.spectrum, p.edge_spectrum
    drude, loss, crop = p.drude, p.loss_factor, c.crop
    each!(c, length(crop), size(destination, 2)) do f, m
        @inbounds begin
            k = crop[f]
            destination[k, m] += drude[f] * spectrum[k, m] + loss[f] * edge_spectrum[k, m]
        end
    end
end

"""Fresh populations at an observation: each node's final fraction and peak rate."""
function envelope_populations!(q, nodes, c, first, count)
    envelope_exposure!(q, nodes, c, count)
    (; rate, exposure, fractions, peaks) = q
    T = typeof(q.dt)
    initial, none = T(q.parameters[9]), zero(q.dt)
    nt = size(nodes, 1)
    each!(c, count, 1) do n, _
        @inbounds begin
            peak = none
            for t in 1:nt
                peak = max(peak, rate[t, n])
            end
            e = exposure[nt, n]
            fractions[first+n-1] = initial + (1 - initial) * -expm1(-e)
            peaks[first+n-1] = peak
        end
    end
    zeros(6)
end

"""Whether the last source left a non-converged sample (NaN) in a channel's rates: the
adapter asks only after a nonfinite norm, to raise the oracle's ConvergenceError."""
function envelope_failed(engine)
    p = engine.work.plasma
    p === nothing || !is_envelope(p) ? false : any(isnan, Array(p.rate))
end
