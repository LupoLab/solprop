# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# What the driver reads of the state between steps: the state itself, and the boundary
# monitor's reductions, formed here so that only short vectors cross (SEP 21, ADR-057).

"""The accepted state, for the host: the engine's own array on a CPU in binary64."""
host_state(engine) = @section :observe host_view(engine.work.y)
host_candidate(engine) = host_view(engine.work.candidate)
host_view(a::Matrix{ComplexF64}) = a
host_view(a) = Matrix{ComplexF64}(Array(a))

"""A Python buffer's memory as a Julia array; the caller keeps the buffer alive."""
host_wrap(buffer, T) = (array = PyArray{T}(buffer); unsafe_wrap(Array, pointer(array), size(array)))

"""Copy a state between the engine and a host (n_omega, n_modes) column-major buffer.

On a CPU the pool's tasks copy, each touching its own pages: a single-threaded copy
of a 226 MB Cartesian state into fresh host memory took 75-220 ms.
"""
function copy_between!(target, source, c)
    if eltype(target) == eltype(source) && (oncpu(c) || !(target isa Array))
        pool_copy!(target, source, c)
    else
        copyto!(target, convert(Array{eltype(target)}, Array(source)))
    end
    target
end
function copy_state!(engine, buffer)
    @section :observe begin
        c, w = engine.config, engine.work
        target = host_wrap(buffer, ComplexF64)
        if oncpu(c)
            copy_between!(target, w.y, c)
        else
            # One copy from the device a state, however many readers ask (`version`).
            held = w.held_state
            if held[] === nothing || first(held[]) != engine.version
                held[] = (engine.version, Matrix{ComplexF64}(Array(w.y)))
            end
            copyto!(target, last(held[]))
        end
        scale = engine.config.scale
        scale == 1 || (target .*= scale)
    end
    nothing
end

"""Load a host state given as its (n_omega, n_modes) column-major view."""
function load!(engine, input)
    y = engine.work.y
    source = host_wrap(input, ComplexF64)
    inverse = inv(engine.config.scale)
    if oncpu(engine.config) && eltype(y) == ComplexF64 && inverse == 1
        copy_between!(y, source, engine.config)
    else
        copyto!(y, upload(engine.config, eltype(y).(source .* inverse)))
    end
    reset!(engine)
end

"""Hold the monitor's column weights and, for a split crossing, its rate buckets."""
function bind_monitor!(engine, columns, buckets)
    c, w = engine.config, engine.work
    T = real_type(engine)
    nm = size(w.y, 2)
    weights = columns === nothing ? ones(T, nm) : T.(Vector{Float64}(columns))
    copyto!(w.monitor.columns, weights)
    if buckets !== nothing
        # The (n_omega, n_modes) view of the crossing's table, converted by the pool.
        table = host_wrap(buckets, UInt8)
        target = oncpu(c) ? w.monitor.buckets : similar(table, Int32)
        each!(merge(c, (backend=KA.CPU(),)), size(table)...) do f, m
            @inbounds target[f, m] = Int32(table[f, m]) + Int32(1)
        end
        oncpu(c) || copyto!(w.monitor.buckets, target)
    end
    nothing
end
function monitor_workspace(c, nw, nm, nc)
    # Bucket one until a crossing binds its own: an index the histogram always holds.
    (columns=dzeros(c, c.T, nm), buckets=fill!(dzeros(c, Int32, nw, nm), Int32(1)), power=dzeros(c, c.T, nc),
     split=dzeros(c, c.T, nc), histogram=dzeros(c, c.T, 128, c.tasks))
end

"""Add `sum_m w_m |x(t, m)|^2 / n^2` of unnormalised inverse samples into `power`."""
function add_power!(power, samples, weights, first, count, c)
    T = eltype(power)
    scale = inv(T(size(samples, 1)))^2
    rowsum!(c, power, size(samples, 1), count) do t, m
        @inbounds abs2(samples[t, m]) * weights[first+m-1] * scale
    end
end

"""`P(t)` of the accepted state on the coarse time axis: the monitor's `power`."""
state_power(engine) = @section :monitor monitor_power(engine)
function monitor_power(engine)
    c, w = engine.config, engine.work
    (; power, columns) = w.monitor
    fill!(power, 0)
    if c.free
        chunk, nm = size(w.coarse, 2), size(w.y, 2)
        for first in 1:chunk:nm
            count = min(chunk, nm - first + 1)
            columns_into!(w.coarse_spec, w.y, first, count, c)
            mul!(w.coarse, w.coarse_inverse, w.coarse_spec)
            add_power!(power, w.coarse, columns, first, count, c)
        end
    else
        # The c2r transform overwrites its input: the scratch takes a copy.
        pool_copy!(w.lab, w.y, c)
        mul!(w.coarse, w.coarse_inverse, w.lab)
        add_power!(power, w.coarse, columns, 1, size(w.y, 2), c)
    end
    Vector{Float64}(Array(power)) .* engine.config.scale^2
end
split_power(engine) = Vector{Float64}(Array(engine.work.monitor.split)) .* engine.config.scale^2

function columns_into!(target, source, first, count, c)
    each!(c, size(target)...) do f, m
        @inbounds target[f, m] = m <= count ? source[f, first+m-1] : zero(eltype(target))
    end
end

"""The state's spectral energy in each of the split crossing's rate buckets."""
# A device's histogram a workgroup at a time in local memory, then added: every sample's
# atomic add on the few global buckets a crossing uses contended (747 us a radial
# occupancy on an AMD Instinct MI325X).
KA.@kernel function bucket_kernel!(histogram, @Const(y), @Const(buckets), @Const(columns), nw, n)
    i = KA.@index(Global, Linear)
    local_index = KA.@index(Local, Linear)
    groupsize = KA.@groupsize()[1]
    shared = KA.@localmem eltype(histogram) (128,)
    b = local_index
    while b <= 128
        @inbounds shared[b] = 0
        b += groupsize
    end
    KA.@synchronize
    if i <= n
        m = (i - 1) ÷ nw + 1
        f = i - (m - 1) * nw
        @inbounds Atomix.@atomic shared[buckets[f, m]] += abs2(y[f, m]) * columns[m]
    end
    KA.@synchronize
    b = local_index
    while b <= 128
        @inbounds value = shared[b]
        !iszero(value) && (@inbounds Atomix.@atomic histogram[b] += value)
        b += groupsize
    end
end

function occupancy(engine)
    c, w = engine.config, engine.work
    (; buckets, columns, histogram) = w.monitor
    y = w.y
    nw, nm = size(y)
    T = real_type(engine)
    fill!(histogram, 0)
    if oncpu(c)
        tasks = size(histogram, 2)
        @batch for k in 1:tasks
            for m in k:tasks:nm, f in 1:nw
                @inbounds histogram[buckets[f, m], k] += abs2(y[f, m]) * columns[m]
            end
        end
    else
        bucket_kernel!(c.backend, 256)(view(histogram, :, 1), y, buckets, columns, nw,
                                       nw * nm; ndrange=nw * nm)
    end
    vec(sum(Array{Float64}(Array(histogram)); dims=2)) .* engine.config.scale^2
end

# --- the transverse edge (SEP 44): per-node and per-column energies -----------------

"""Set `out[m]` to the sum over `t` of `f(t, m)`: one column's reduction each."""
function colsum!(f::F, c, out, nt::Integer, nm::Integer) where {F}
    if !oncpu(c)
        # A thread a column left most of the device idle: a parallel fill and a sum.
        scratch = reduce_terms(f, c, out, nt, nm)
        sum!(reshape(out, 1, :), scratch)
        return nothing
    end
    each_column!(c, nm) do m
        total = zero(eltype(out))
        @inbounds for t in 1:nt
            total += f(t, m)
        end
        @inbounds out[m] = total
        nothing
    end
end

"""Each spectral column's energy, `sum |E|^2` over frequency."""
function column_energy(engine)
    y = engine.work.y
    out = dzeros(engine.config, real_type(engine), size(y, 2))
    colsum!(engine.config, out, size(y)...) do f, m
        @inbounds abs2(y[f, m])
    end
    Vector{Float64}(Array(out)) .* engine.config.scale^2
end

"""Hold a radial read's synthesis rows (0-based `rows` of `inverse`) for `node_energy`."""
function bind_edge!(engine, inverse, rows)
    c, y = engine.config, engine.work.y
    T = real_type(engine)
    selected = Vector{Int}(rows) .+ 1
    matrix = c.free ? nothing : upload(c, T.(permutedims(Matrix{Float64}(inverse)[selected, :])))
    engine.work.edge[] = (selected=selected, matrix=matrix,
                          synthesised=c.free ? nothing : dzeros(c, Complex{T}, size(y, 1), length(selected)),
                          energy=dzeros(c, T, c.free ? size(y, 2) : length(selected)))
    nothing
end

"""Each physical node's energy at the bound rows (`bind_edge!`), zero elsewhere: the
radial synthesis's rows, or every node of a Cartesian grid through the transverse
inverse transform."""
node_energy(engine) = @section :edge edge_energy(engine)
function edge_energy(engine)
    c, w = engine.config, engine.work
    y, edge = w.y, w.edge[]
    T = real_type(engine)
    nodes = zeros(Float64, size(y, 2))
    out = edge.energy
    if c.free
        physical = w.physical
        pool_copy!(physical, y, c)
        transverse!(physical, w.spatial_inverse, c)
        scale = inv(T(size(y, 2)))^2
        colsum!(c, out, size(y)...) do f, m
            @inbounds abs2(physical[f, m]) * scale
        end
        all = Array(out)
        nodes[edge.selected] .= all[edge.selected]
        return nodes .* c.scale^2
    end
    synthesised = edge.synthesised
    gemm!(synthesised, y, edge.matrix, true, false, c)
    colsum!(c, out, size(synthesised)...) do f, m
        @inbounds abs2(synthesised[f, m])
    end
    nodes[edge.selected] .= Array(out)
    nodes .* c.scale^2
end

# --- the Poynting statistics' time-domain power ------------------------------------

"""Hold the Poynting statistics' buffers and their transform, and the power weight
(`weight`, (n_omega, n_modes)) and on-axis rows (`axis`, (2, n_modes), or `nothing`).
A varying guide's weight is bound again at each plane; nothing else changes."""
function bind_poynting!(engine, weight, axis)
    c, w = engine.config, engine.work
    T = real_type(engine)
    C = Complex{T}
    nw, nm = size(w.y)
    nt = c.nc
    s = w.statistics[]
    if s === nothing
        signal = dzeros(c, C, nt, nm)
        rows = axis === nothing ? nothing : upload(c, T.(permutedims(Matrix{Float64}(axis))))
        width = rows === nothing ? 0 : size(rows, 2)
        s = (signal=signal, weighted=dzeros(c, C, nt, nm), weight=dzeros(c, T, nw, nm),
             inverse=plan(c, plan_bfft!, signal, 1), rows=rows,
             electric=dzeros(c, C, nt, width), magnetic=dzeros(c, C, nt, width),
             power=dzeros(c, T, nt, nm), flux=dzeros(c, T, nt),
             host_power=zeros(T, nt, nm), host_flux=zeros(T, rows === nothing ? 0 : nt))
        w.statistics[] = s
    end
    rebind_weight!(engine, weight)
end

"""A new power weight (n_omega, n_modes) for the bound Poynting buffers: a varying
guide's, at each plane."""
function rebind_weight!(engine, weight)
    target = engine.work.statistics[].weight
    copyto!(target, eltype(target).(Matrix{Float64}(weight)))
    nothing
end

"""The state's analytic power per mode and its on-axis flux, for the statistics.

`stats._poynting_statistics`'s transforms: the analytic signals of the state and of
the state times its power weight, the cycle-averaged power `Re(a conj b)` per mode and
time, and with `flux` the on-axis flux through the bound rows. A field grid's analytic
signal keeps the zero-frequency and Nyquist bins once, as `fourier.analytic_signal`
does; an envelope's state already is one. Returns host arrays the engine reuses: the
power (n_t, n_modes) and the flux (n_t), in the state's physical units.
"""
function poynting!(engine, flux)
    c, w = engine.config, engine.work
    y, s = w.y, w.statistics[]
    T = real_type(engine)
    nw = size(y, 1)
    nt, nm = size(s.signal)
    (; signal, weighted, weight, power) = s
    envelope, nyquist = c.envelope, nt ÷ 2 + 1
    unit, two, none = one(T), T(2), zero(Complex{T})
    for (target, weighting) in ((signal, false), (weighted, true))
        each!(c, nt, nm) do k, m
            @inbounds begin
                value = k <= nw ? y[k, m] * (weighting ? weight[k, m] : unit) : none
                if !envelope
                    value = k == 1 || k == nyquist ? complex(real(value)) : k < nyquist ? two * value : none
                end
                target[k, m] = value
            end
        end
        s.inverse * target
    end
    # Each inverse is unnormalised: numpy's 1/n_t on each signal, and the state scale.
    factor = T((c.scale / nt)^2)
    each!(c, nt, nm) do t, m
        @inbounds power[t, m] = real(signal[t, m] * conj(weighted[t, m])) * factor
    end
    copyto!(s.host_power, power)
    if flux
        (; electric, magnetic, rows) = s
        gemm!(electric, signal, rows, true, false, c)
        gemm!(magnetic, weighted, rows, true, false, c)
        values, start = s.flux, zero(T)
        each!(c, nt, 1) do t, _
            @inbounds begin
                value = start
                for j in axes(electric, 2)
                    value += real(electric[t, j] * conj(magnetic[t, j]))
                end
                values[t] = value * factor
            end
        end
        copyto!(s.host_flux, values)
    end
    (s.host_power, s.host_flux)
end

const bind_poynting_b = bind_poynting!
const rebind_weight_b = rebind_weight!
const poynting_b = poynting!

# --- the spatial routes' local statistics --------------------------------------------

"""Hold the spatial local statistics' buffers and tables: the closure's factor, the
conjugate axial admittance (`closure`, (n_omega, n_modes)), and a modal transform's
synthesis over the pad scale (`synthesis`, (n_nodes, n_modes)); a free transform's
nodes are its inverse transverse FFT. `chunk` nodes are transformed at a time."""
function bind_spatial!(engine, closure, synthesis, chunk)
    c, w = engine.config, engine.work
    T = real_type(engine)
    C = Complex{T}
    nw, nm = size(w.y)
    s = w.spatial[]
    if s === nothing
        S = synthesis === nothing ? nothing :
            upload(c, T.(permutedims(Matrix{Float64}(synthesis))))
        nn = S === nothing ? nm : size(S, 2)
        width = min(Int(chunk), nn)
        signal = dzeros(c, C, c.nc, width)
        s = (closure=dzeros(c, C, nw, nm), S=S, electric=dzeros(c, C, nw, nn),
             magnetic=dzeros(c, C, nw, nn), weighted=dzeros(c, C, nw, nm), signal=signal,
             other=dzeros(c, C, c.nc, width), inverse=plan(c, plan_bfft!, signal, 1),
             fluence=dzeros(c, T, nn), magnitudes=dzeros(c, T, nn), peaks=dzeros(c, T, nn))
        w.spatial[] = s
    end
    copyto!(s.closure, C.(Matrix{ComplexF64}(closure)))
    nothing
end

"""The nodes' spectra of a modal-domain matrix `x` (n_omega, n_modes) into `nodes`."""
function node_spectra!(nodes, x, s, engine)
    c = engine.config
    if s.S === nothing
        copyto!(nodes, x)
        transverse!(nodes, engine.work.spatial_inverse, c)
    else
        gemm!(nodes, x, s.S, true, false, c)
    end
end

"""`backends.spatial._local_statistics`'s transforms and sums, by the engine.

Each node's field and closure (the field times the conjugate admittance) as analytic
signals, their cycle-averaged Poynting flux `I(t) = 0.5 eps0 c Re(E conj(H))`, and per
node its fluence `dt sum_t I`, the magnitude `0.5 eps0 c dt sum_t |E||H|` and the peak
of `I`. Returns the three per-node arrays (the reference's own forms the widths)."""
function spatial_local!(engine, dt, conversion)
    c, w = engine.config, engine.work
    s = w.spatial[]
    T = real_type(engine)
    y = w.y
    (; closure, weighted, electric, magnetic, signal, other, fluence, magnitudes, peaks) = s
    each!(c, size(y)...) do k, m
        @inbounds weighted[k, m] = y[k, m] * closure[k, m]
    end
    node_spectra!(electric, y, s, engine)
    node_spectra!(magnetic, weighted, s, engine)
    nw, nn = size(electric)
    nt, width = size(signal)
    # numpy's 1/n_t on each inverse, the state's scale, a free transform's 1/N.
    points = s.S === nothing ? prod(c.spatial_shape) : 1
    factor = T((c.scale / (nt * points))^2)
    flux, step = T(conversion), T(dt)
    envelope, nyquist = c.envelope, nt ÷ 2 + 1
    two, none, start, lowest = T(2), zero(Complex{T}), zero(T), T(-Inf)
    for first in 1:width:nn
        count = min(width, nn - first + 1)
        for (target, source) in ((signal, electric), (other, magnetic))
            each!(c, nt, width) do k, n
                @inbounds begin
                    value = k <= nw && n <= count ? source[k, first+n-1] : none
                    if !envelope
                        value = k == 1 || k == nyquist ? complex(real(value)) :
                                k < nyquist ? two * value : none
                    end
                    target[k, n] = value
                end
            end
            s.inverse * target
        end
        if oncpu(c)
            each_column!(c, count) do n
                total, magnitude, peak = start, start, lowest
                @inbounds for t in 1:nt
                    e, h = signal[t, n], other[t, n]
                    intensity = flux * real(e * conj(h)) * factor
                    total += intensity
                    magnitude += abs(e) * abs(h) * factor
                    peak = max(peak, intensity)
                end
                @inbounds begin
                    fluence[first+n-1] = step * total
                    magnitudes[first+n-1] = flux * step * magnitude
                    peaks[first+n-1] = peak
                end
                nothing
            end
        else
            # A device's reductions over a parallel fill, not a thread a node.
            fluxes = reduce_terms(c, fluence, nt, width) do t, n
                @inbounds flux * real(signal[t, n] * conj(other[t, n])) * factor
            end
            products = reduce_terms(c, fluence, nt, width; use=:products) do t, n
                @inbounds abs(signal[t, n]) * abs(other[t, n]) * factor
            end
            totals, sizes, highest = (scratch(c, use, T, 1, width) for use in (:totals, :sizes, :peaks))
            sum!(totals, fluxes)
            sum!(sizes, products)
            maximum!(highest, fluxes)
            each!(c, count, 1) do n, _
                @inbounds begin
                    fluence[first+n-1] = step * totals[1, n]
                    magnitudes[first+n-1] = flux * step * sizes[1, n]
                    peaks[first+n-1] = highest[1, n]
                end
            end
        end
    end
    (Vector{Float64}(Array(fluence)), Vector{Float64}(Array(magnitudes)),
     Vector{Float64}(Array(peaks)))
end

const bind_spatial_b = bind_spatial!
const spatial_local_b = spatial_local!
