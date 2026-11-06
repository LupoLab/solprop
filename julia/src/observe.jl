# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# What the driver reads of the state between steps: the state itself, and the boundary
# monitor's reductions, formed here so that only short vectors cross (SEP 21, ADR-057).

"""The accepted state, for the host: the engine's own array on a CPU in binary64."""
host_state(engine) = host_view(engine.work.y)
host_candidate(engine) = host_view(engine.work.candidate)
host_view(a::Matrix{ComplexF64}) = a
host_view(a) = Matrix{ComplexF64}(Array(a))

"""Load a host state given as its (n_omega, n_modes) column-major view."""
function load!(engine, input)
    y = engine.work.y
    if y isa Matrix{ComplexF64}
        copyto!(y, PyArray(input))
    else
        copyto!(y, Matrix{eltype(y)}(PyArray(input)))
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
        copyto!(w.monitor.buckets, Int32.(permutedims(Matrix{Int}(buckets))) .+ Int32(1))
    end
    nothing
end
function monitor_workspace(c, nw, nm, nc)
    (columns=dzeros(c, c.T, nm), buckets=dzeros(c, Int32, nw, nm), power=dzeros(c, c.T, nc),
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
function state_power(engine)
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
        copyto!(w.lab, w.y)
        mul!(w.coarse, w.coarse_inverse, w.lab)
        add_power!(power, w.coarse, columns, 1, size(w.y, 2), c)
    end
    Vector{Float64}(Array(power))
end
split_power(engine) = Vector{Float64}(Array(engine.work.monitor.split))

function columns_into!(target, source, first, count, c)
    each!(c, size(target)...) do f, m
        @inbounds target[f, m] = m <= count ? source[f, first+m-1] : zero(eltype(target))
    end
end

"""The state's spectral energy in each of the split crossing's rate buckets."""
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
        AK.foreachindex(1:nw*nm; backend=c.backend) do i
            m = (i - 1) ÷ nw + 1
            f = i - (m - 1) * nw
            @inbounds Atomix.@atomic histogram[buckets[f, m], 1] += abs2(y[f, m]) * columns[m]
        end
    end
    vec(sum(Array{Float64}(Array(histogram)); dims=2))
end

# --- the transverse edge (SEP 44): per-node and per-column energies -----------------

"""Set `out[m]` to the sum over `t` of `f(t, m)`: one column's reduction each."""
function colsum!(f::F, c, out, nt::Integer, nm::Integer) where {F}
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
    Vector{Float64}(Array(out))
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
function node_energy(engine)
    c, w = engine.config, engine.work
    y, edge = w.y, w.edge[]
    T = real_type(engine)
    nodes = zeros(Float64, size(y, 2))
    out = edge.energy
    if c.free
        physical = w.physical
        copyto!(physical, y)
        transverse!(physical, w.spatial_inverse, c)
        scale = inv(T(size(y, 2)))^2
        colsum!(c, out, size(y)...) do f, m
            @inbounds abs2(physical[f, m]) * scale
        end
        all = Array(out)
        nodes[edge.selected] .= all[edge.selected]
        return nodes
    end
    synthesised = edge.synthesised
    mul!(synthesised, y, edge.matrix)
    colsum!(c, out, size(synthesised)...) do f, m
        @inbounds abs2(synthesised[f, m])
    end
    nodes[edge.selected] .= Array(out)
    nodes
end
