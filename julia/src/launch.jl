# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The device layer: everything that differs between a CPU and a GPU is here.

"""Read the settings a Python dict carries, once, at construction."""
function HostInput(input)
    has(key) = haskey(input, key)
    (has=has, int=key -> Int(input[key]), real=key -> Float64(input[key]),
     flag=key -> has(key) && Bool(input[key]), string=key -> String(input[key]),
     array=(key, T) -> Array{T}(input[key]), value=key -> input[key])
end

device_backend(::Val{:cpu}) = KA.CPU()
device_backend(::Val{D}) where {D} =
    throw(ArgumentError("device $D needs its package loaded first (SOLPROP_JULIA_DEVICE)"))
device_label(::KA.CPU) = "cpu"
oncpu(c) = c.backend isa KA.CPU

"""Put a host array on the engine's device, owning it on the CPU too."""
upload(::KA.CPU, a::Array) = a
upload(backend, a::Array) = copyto!(KA.allocate(backend, eltype(a), size(a)), a)
upload(c::NamedTuple, a::Array) = upload(c.backend, a)
dzeros(c, T, dims...) = KA.zeros(c.backend, T, dims...)

# Below this many samples one pass is faster on one thread than split over tasks
# (julia/bench/launch.jl: a Kerr pass over 10240 x 4 takes 8.5 us on one thread of the
# Xeon, 10.5 us on four).
const PARALLEL_SAMPLES = 65536

"""Call `f(t, m)` for every sample of an `nt` x `nm` column-major array.

On a CPU each task takes whole columns, or pieces of one when there are fewer columns
than tasks, and runs a contiguous `@simd` loop over its rows: an index divide per
sample costs a factor of ten (julia/bench/launch.jl). On a GPU each sample is a thread.
"""
@inline each!(f::F, c, nt::Integer, nm::Integer) where {F} =
    each_on!(f, c.backend, c.tasks, Int(nt), Int(nm))

function each_on!(f::F, ::KA.CPU, tasks, nt, nm) where {F}
    pieces, rows = tiling(tasks, nt, nm)
    if pieces * nm == 1
        @inbounds @simd for t in 1:nt
            @inline f(t, 1)
        end
        return nothing
    end
    AK.foreachindex(1:pieces*nm; max_tasks=tasks, min_elems=1) do k
        m = (k - 1) ÷ pieces + 1
        first = (k - (m - 1) * pieces - 1) * rows + 1
        @inbounds @simd for t in first:min(first + rows - 1, nt)
            @inline f(t, m)
        end
    end
    nothing
end

"""Split `nt` x `nm` samples into column pieces: one task's worth each."""
function tiling(tasks, nt, nm)
    if tasks == 1 || nt * nm < PARALLEL_SAMPLES
        return 1, nt
    end
    nm >= tasks && return 1, nt
    pieces = min(cld(tasks, nm), cld(nt * nm, PARALLEL_SAMPLES ÷ 4))
    pieces, cld(nt, pieces)
end

KA.@kernel inbounds = true function each_kernel!(f)
    t, m = KA.@index(Global, NTuple)
    f(t, m)
end
function each_on!(f::F, backend, tasks, nt, nm) where {F}
    each_kernel!(backend, (256, 1))(f; ndrange=(nt, nm))
    nothing
end

"""Call `f(m)` for each of `nm` columns, whose body runs sequentially down one."""
function each_column!(f::F, c, nm::Integer) where {F}
    if oncpu(c)
        AK.foreachindex(1:Int(nm); max_tasks=c.tasks, min_elems=1) do m
            @inline f(m)
        end
    else
        AK.foreachindex(f, 1:Int(nm); backend=c.backend, block_size=64)
    end
    nothing
end

"""Return `op` folded over `f(t, m)` for every sample, starting from `neutral`.

`op` must be associative and commutative: the order is the tiling's (CPU) or the
reduction tree's (GPU), and differs with the width (SEP 25 bounds it, not bits).
"""
function reduce_each(f::F, op::O, neutral, c, nt::Integer, nm::Integer) where {F,O}
    nt, nm = Int(nt), Int(nm)
    if oncpu(c)
        pieces, rows = tiling(c.tasks, nt, nm)
        partials = fill(neutral, pieces * nm)
        AK.foreachindex(1:pieces*nm; max_tasks=c.tasks, min_elems=1) do k
            m = (k - 1) ÷ pieces + 1
            first = (k - (m - 1) * pieces - 1) * rows + 1
            value = neutral
            @inbounds for t in first:min(first + rows - 1, nt)
                value = op(value, f(t, m))
            end
            @inbounds partials[k] = value
        end
        return foldl(op, partials; init=neutral)
    end
    AK.mapreduce(op, 1:nt*nm; backend=c.backend, init=neutral, neutral=neutral) do i
        m = (i - 1) ÷ nt + 1
        f(i - (m - 1) * nt, m)
    end
end

# FFTW's planner: native's policy (native/include/solprop_native/fft_plan.hpp). Serial
# row transforms and every transverse plane are measured, threaded rows estimated (a
# measured threaded plan was slower there); SOLPROP_FFTW_PLAN overrides all of them.
const FLAGS = Dict("estimate" => FFTW.ESTIMATE, "measure" => FFTW.MEASURE,
                   "patient" => FFTW.PATIENT)
function plan_flags(shape::Symbol, threads)
    chosen = get(ENV, "SOLPROP_FFTW_PLAN", "")
    isempty(chosen) && return shape === :rows && threads > 1 ? FFTW.ESTIMATE : FFTW.MEASURE
    haskey(FLAGS, chosen) || throw(ArgumentError("SOLPROP_FFTW_PLAN must be estimate, measure or patient"))
    FLAGS[chosen]
end

"""Plan `kind(a, args...)` for the engine's device; inverses are left unnormalised.

On a CPU, measuring overwrites `a`: plan before anything is stored in it.
"""
plan(c, kind, a, args...; threads=c.fft_threads, shape=:rows) =
    oncpu(c) ? kind(a, args...; flags=plan_flags(shape, threads), num_threads=threads) :
    kind(a, args...)
