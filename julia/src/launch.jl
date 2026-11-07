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
upload(backend, a::Array) = (tally!(:upload); copyto!(KA.allocate(backend, eltype(a), size(a)), a))
upload(c::NamedTuple, a::Array) = upload(c.backend, a)
dzeros(c, T, dims...) = KA.zeros(c.backend, T, dims...)

# Development counters (SOLPROP_JULIA_COUNT=1 when the module loads): a device's kernel
# launches, the values the host waits for, uploads, products and scratch allocations,
# by kind. Off, `tally!` costs one branch.
const COUNT = Ref(false)
const COUNTS = Dict{Symbol,Int}()
@inline tally!(kind::Symbol, n=1) = (COUNT[] && (COUNTS[kind] = get(COUNTS, kind, 0) + n); nothing)
counts() = Dict(String(k) => v for (k, v) in COUNTS)
clear_counts() = (empty!(COUNTS); nothing)

# Below this many samples one pass is faster on one thread than split over tasks
# (julia/bench/launch.jl: a Kerr pass over 10240 x 4 takes 8.5 us on one thread of the
# Xeon, 10.5 us on four).
const PARALLEL_SAMPLES = 65536

"""Call `f(t, m)` for every sample of an `nt` x `nm` column-major array.

On a CPU each task takes whole columns, or pieces of one when there are fewer columns
than tasks, and runs a contiguous `@simd` loop over its rows: an index divide per
sample costs a factor of ten (julia/bench/launch.jl). The tasks are Polyester's,
whose threads wait spinning: a pass between serial work costs 2.2 us on four Xeon
cores where a spawned task costs 10 and `Threads.@threads` 110 (10240 x 4; 13, 49 and
147 us over 1728 x 128). On a GPU each sample is a thread.
"""
@inline function each!(f::F, c, nt::Integer, nm::Integer; heavy=false) where {F}
    CHECK[] && check_body(F)
    each_on!(f, c.backend, c.tasks, Int(nt), Int(nm), heavy)
end

# SOLPROP_JULIA_CHECK=1: refuse a body a GPU kernel could not take, on any device. A
# captured type, box or union is found on a CPU run instead of on the first GPU one.
const CHECK = Ref(false)
function check_body(::Type{F}) where {F}
    for T in fieldtypes(F)
        (T <: Type || T === Core.Box || T isa Union || !isconcretetype(T)) &&
            throw(ArgumentError("kernel body $F captures $T, which a GPU kernel cannot take"))
        (T <: Function || T <: Tuple || T <: NamedTuple) && check_body(T)
    end
end

function each_on!(f::F, ::KA.CPU, tasks, nt, nm, heavy) where {F}
    pieces, rows = tiling(tasks, nt, nm, heavy)
    if pieces * nm == 1
        @inbounds @simd for t in 1:nt
            @inline f(t, 1)
        end
        return nothing
    end
    @batch for k in 1:pieces*nm
        m = (k - 1) ÷ pieces + 1
        first = (k - (m - 1) * pieces - 1) * rows + 1
        @inbounds @simd for t in first:min(first + rows - 1, nt)
            @inline f(t, m)
        end
    end
    nothing
end

"""Split `nt` x `nm` samples into column pieces: one task's worth each.

A `heavy` body (tens of operations a sample, such as the phi functions) is worth
splitting from a few hundred samples.
"""
function tiling(tasks, nt, nm, heavy=false)
    # A heavy body (the Adams trial's phi functions, about 100 ns an element) pays for a
    # task from a few hundred elements: argon's 1601 ran serially on 16 threads.
    least = heavy ? 256 : PARALLEL_SAMPLES
    if tasks == 1 || nt * nm < least
        return 1, nt
    end
    nm >= tasks && return 1, nt
    pieces = min(cld(tasks, nm), cld(nt * nm, least ÷ 4))
    pieces, cld(nt, pieces)
end

KA.@kernel inbounds = true function each_kernel!(f)
    t, m = KA.@index(Global, NTuple)
    f(t, m)
end
function each_on!(f::F, backend, tasks, nt, nm, heavy) where {F}
    tally!(:launch)
    each_kernel!(backend, (256, 1))(f; ndrange=(nt, nm))
    nothing
end

"""A device scratch array the engine keeps (`c.scratch`), by its use, type and shape: a
GPU's pool allocations a trial were dozens."""
scratch(c, use::Symbol, T, dims...) =
    get!(() -> (tally!(:scratch_alloc); KA.allocate(c.backend, T, dims...)), c.scratch, (use, T, dims))

"""`f(t, m)` at every sample, into a device scratch matrix of `out`'s element type, for
a device reduction over its rows or columns."""
function reduce_terms(f::F, c, out, nt, nm; use=:terms) where {F}
    scratch = SolpropJulia.scratch(c, use, eltype(out), nt, nm)
    each!(c, nt, nm) do t, m
        @inbounds scratch[t, m] = f(t, m)
    end
    scratch
end

"""Call `f(m)` for each of `nm` columns, whose body runs sequentially down one."""
function each_column!(f::F, c, nm::Integer) where {F}
    CHECK[] && check_body(F)
    if oncpu(c)
        @batch for m in 1:Int(nm)
            @inline f(m)
        end
    else
        tally!(:launch)
        AK.foreachindex(f, 1:Int(nm); backend=c.backend, block_size=64)
    end
    nothing
end

"""Set `out[t]` to the sum over `m` of `f(t, m)`: a reduction across columns.

Each task (CPU) or thread (GPU) owns a block of rows, so no two write one sum.
"""
function rowsum!(f::F, c, out, nt::Integer, nm::Integer) where {F}
    nt, nm = Int(nt), Int(nm)
    CHECK[] && check_body(F)
    if oncpu(c)
        rows = max(256, cld(nt, c.tasks))
        @batch for b in 1:cld(nt, rows)
            r = (b-1)*rows+1:min(b * rows, nt)
            for m in 1:nm
                @inbounds @simd for t in r
                    out[t] += f(t, m)
                end
            end
        end
    else
        # A thread a row looped over every column (36864 for a Cartesian plane): the
        # terms fill a scratch matrix in parallel and the device's reduction sums it.
        scratch = reduce_terms(f, c, out, nt, nm)
        total = SolpropJulia.scratch(c, :rows, eltype(out), nt, 1)
        tally!(:launch, 2)
        sum!(total, scratch)
        out .+= vec(total)
    end
    nothing
end

"""Return `op` folded over `f(t, m)` for every sample, starting from `neutral`.

`op` must be associative and commutative: the order is the tiling's (CPU) or the
reduction tree's (GPU), and differs with the width (SEP 25 bounds it, not bits).
"""
function reduce_each(f::F, op::O, neutral, c, nt::Integer, nm::Integer; heavy=false) where {F,O}
    nt, nm = Int(nt), Int(nm)
    CHECK[] && check_body(F)
    if oncpu(c)
        pieces, rows = tiling(c.tasks, nt, nm, heavy)
        partials = fill(neutral, pieces * nm)
        @batch for k in 1:pieces*nm
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
    tally!(:readback)
    if heavy
        # A heavy body a thread: AK's reduction gives each thread several elements
        # (six blocks for rdw's 10244, its predictor 539 us a trial on an A40), so the
        # bodies run as an elementwise kernel into scratch and only the sums reduce.
        scratch = SolpropJulia.scratch(c, :heavy, typeof(neutral), nt * nm)
        each!(c, nt, nm) do t, m
            @inbounds scratch[t+(m-1)*nt] = f(t, m)
        end
        return AK.mapreduce(identity, op, scratch; backend=c.backend, init=neutral,
                            neutral=neutral)
    end
    AK.mapreduce(op, 1:nt*nm; backend=c.backend, init=neutral, neutral=neutral) do i
        m = (i - 1) ÷ nt + 1
        f(i - (m - 1) * nt, m)
    end
end

"""The sum of `f(t, m)` over every sample into `out[1]`, which the next kernel reads: on a
GPU it stays on the device (each value the host waits for cost about 0.2 ms on an M5
Max), on a CPU it is `reduce_each`'s, the same bits."""
function sum_device!(f::F, out, c, nt::Integer, nm::Integer; heavy=false) where {F}
    nt, nm = Int(nt), Int(nm)
    if oncpu(c)
        @inbounds out[1] = reduce_each(f, +, zero(eltype(out)), c, nt, nm; heavy=heavy)
        return out
    end
    terms = SolpropJulia.scratch(c, :sum_terms, eltype(out), 1, nt * nm)
    each!(c, nt, nm) do t, m
        @inbounds terms[1, t+(m-1)*nt] = f(t, m)
    end
    tally!(:launch)
    sum!(reshape(out, 1, 1), terms)
    out
end

# FFTW's planner measures every plan unless SOLPROP_FFTW_PLAN says otherwise. Unlike
# native's policy (rows estimated when threaded), a threaded estimate is far slower
# here: the Raman convolution's c2r of 73728 x 17 takes 6.6 ms estimated and 1.49 ms
# measured on four Xeon cores, 7.5 and 4.9 ms on one (planning is at construction).
const FLAGS = Dict("estimate" => FFTW.ESTIMATE, "measure" => FFTW.MEASURE,
                   "patient" => FFTW.PATIENT)
function plan_flags()
    chosen = get(ENV, "SOLPROP_FFTW_PLAN", "")
    isempty(chosen) && return FFTW.MEASURE
    haskey(FLAGS, chosen) || throw(ArgumentError("SOLPROP_FFTW_PLAN must be estimate, measure or patient"))
    FLAGS[chosen]
end

"""Plan `kind(a, args...)` for the engine's device; inverses are left unnormalised.

On a CPU with several tasks the transform is split into blocks of whole batches along
dimension `batch` (columns for row transforms, rows for transverse planes), one
single-threaded plan each, run by the pool that runs every other pass: FFTW's and
BLAS's own threads beside Polyester's spinning ones made a four-thread rdw run seven
times slower than one thread. Measuring overwrites `a`: plan before storing in it.
"""
function plan(c, kind, a, args...; threads=c.fft_threads, batch=2)
    oncpu(c) || return kind(a, args...)
    blocks = min(threads, size(a, batch))
    blocks <= 1 && return kind(a, args...; flags=plan_flags(), num_threads=1)
    ranges = [r for r in Iterators.partition(axes(a, batch), cld(size(a, batch), blocks))]
    # A block's view starts anywhere in the array: its plan may not assume alignment.
    plans = [kind(block_view(a, batch, r), args...; flags=plan_flags() | FFTW.UNALIGNED,
                  num_threads=1) for r in ranges]
    Blocked(plans, ranges, batch)
end

struct Blocked{P}
    plans::Vector{P}
    ranges::Vector{UnitRange{Int}}
    batch::Int
end
block_view(a, batch, r) = batch == 2 ? view(a, :, r) : view(a, r, ntuple(_ -> Colon(), ndims(a) - 1)...)
function LinearAlgebra.mul!(y, p::Blocked, x)
    @batch for b in eachindex(p.plans)
        mul!(block_view(y, p.batch, p.ranges[b]), p.plans[b], block_view(x, p.batch, p.ranges[b]))
    end
    y
end
"""`mul!(outs[k], p, ins[k])` for each pair: on a CPU one pool task for each block of
each, so that transforms over a few columns (a capillary's modes) fill the pool."""
function transform_all!(p::Blocked, outs::Tuple, ins::Tuple)
    blocks = length(p.plans)
    @batch for k in 1:blocks*length(outs)
        j, b = divrem(k - 1, blocks) .+ 1
        transform_block!(p, outs, ins, j, b)
    end
end
@inline function transform_block!(p, outs, ins, j, b)
    # A tuple's elements by a run-time index: each pair's arrays share one type here.
    r = p.ranges[b]
    mul!(block_view(outs[j], p.batch, r), p.plans[b], block_view(ins[j], p.batch, r))
end
# On a GPU each transform over a capillary's few modes is a few blocks (one a column,
# four for rdw, 96 us each on an A40): the group goes through one batched plan over a
# staging buffer instead, the copies a few microseconds each. The staging lives in the
# engine's workspace (`stage`, a `Ref`), not in a global table of plans.
function transform_all!(p, outs::Tuple, ins::Tuple, stage::Ref, c)
    x = first(ins)
    if x isa Array
        foreach((y, x) -> mul!(y, p, x), outs, ins)
        return nothing
    end
    n, columns = length(ins), size(x, 2)
    held = stage[]
    if held === nothing || size(held[1], 2) != n * columns
        staged = similar(x, size(x, 1), n * columns)
        result = similar(first(outs), size(first(outs), 1), n * columns)
        held = (staged, result, eltype(x) <: Real ? plan_rfft(staged, 1) : plan_fft(staged, 1))
        stage[] = held
    end
    staged, result, batched = held
    for j in 1:n
        device_copy!(view(staged, :, (j-1)*columns+1:j*columns), ins[j], c)
    end
    mul!(result, batched, staged)
    for j in 1:n
        device_copy!(outs[j], view(result, :, (j-1)*columns+1:j*columns), c)
    end
    nothing
end
transform_all!(p::Blocked, outs::Tuple, ins::Tuple, ::Ref, c) = transform_all!(p, outs, ins)

function Base.:*(p::Blocked, x)
    @batch for b in eachindex(p.plans)
        p.plans[b] * block_view(x, p.batch, p.ranges[b])
    end
    x
end

"""`copyto!(target, source)` for two state-sized arrays of the device: on a CPU as the
pool's tasks, since one thread's copy of a Cartesian state (226 MB) took 45 ms."""
function pool_copy!(target, source, c)
    each!(c, size(target)...) do f, m
        @inbounds target[f, m] = source[f, m]
    end
    target
end

"""`copyto!(target, source)` for two matrices of the device, views included, as one
kernel: Metal.jl's `copyto!` into or out of a view took about 200 us (a staged FFT batch
of three 8192 x 4 transforms 1.6 ms against 0.11 ms for three separate ones)."""
device_copy!(target, source, c) = pool_copy!(target, source, c)

"""`C = alpha A B + beta C`: on a CPU, blocks of rows of `C` and `A` as the pool's tasks.

A complex `C` and `A` against a real `B` are their real views, of twice the rows: one
real GEMM, BLAS's or cuBLAS's. Julia's own mixed-type product is generic, slow on a GPU,
and in binary32 not safe to run from several tasks (a 1.5 relative error on the Xeon,
LinearAlgebra in Julia 1.13.3).
"""
function gemm!(C, A, B, alpha, beta, c)
    if eltype(C) <: Complex && eltype(A) <: Complex && eltype(B) <: Real
        return gemm!(real_rows(C), real_rows(A), B, alpha, beta, c)
    end
    if !oncpu(c)
        tally!(:gemm)
        A, B = dense(c, A, :gemm_a), dense(c, B, :gemm_b)
    end
    if oncpu(c) && eltype(C) <: Real && (size(B, 2) <= THIN || size(B, 1) <= THIN)
        return thin_gemm!(C, A, B, alpha, beta, c)
    end
    blocks = oncpu(c) ? min(c.tasks, cld(length(C), PARALLEL_SAMPLES ÷ 4)) : 1
    blocks <= 1 && return mul!(C, A, B, alpha, beta)
    rows = cld(size(C, 1), blocks)
    @batch for b in 1:blocks
        r = (b-1)*rows+1:min(b * rows, size(C, 1))
        mul!(view(C, r, :), view(A, r, :), B, alpha, beta)
    end
    C
end

"""On a GPU, an operand that is a strided view (a range of a matrix's rows) copied into
scratch: Metal.jl's product of one takes a generic kernel, about 240 us for 16384 x 36
x 8 on the M5 Max against 65 to 140 us copied first."""
function dense(c, X::SubArray{<:Any,2}, use)
    strides(X) == (1, size(X, 1)) && return X
    tally!(:dense_copy)
    device_copy!(scratch(c, use, eltype(X), size(X)...), X, c)
end
dense(c, X, use) = X

"""The nodes (or columns) a source evaluates at once: the translation's chunk, sized for a
CPU's memory budget, on a CPU; on a GPU all of them, up to 2^26 samples a buffer, since
each chunk repeats the synthesis, the responses and the projections as launches
(`ctc_reduced`'s 33 nodes were three chunks of 11)."""
node_chunk(c, chunk, count, nt) =
    oncpu(c) ? min(chunk, count) : min(count, max(chunk, (1 << 26) ÷ max(nt, 1)))

"""Rows `first:last` of a constant table (a projection), for a product: a view on a CPU;
on a GPU a contiguous copy, made once and kept in `c.scratch`, since a product with a
strided operand there takes a generic kernel or a copy every call (`dense`)."""
function table_rows(c, table, first, last)
    oncpu(c) && return @view(table[first:last, :])
    first == 1 && last == size(table, 1) && return table
    get!(() -> (tally!(:scratch_alloc); copy(@view(table[first:last, :]))),
         c.scratch, (:table_rows, objectid(table), first, last))
end

"""A product's thin side, at most this many columns or inner terms: a loop, not BLAS."""
const THIN = 8

"""`C = alpha A B + beta C` for real matrices with a thin `B`: as the pool's tasks over
blocks of rows, each a column of `C` at a time, a contiguous axpy per inner term.
A capillary's one mode made each projection sixteen BLAS calls of a vector each,
whose overheads were most of argon's 0.25 ms a source."""
function thin_gemm!(C, A, B, alpha, beta, c)
    rows, inner = size(A)
    columns = size(B, 2)
    blocks = max(1, min(c.tasks, cld(rows * (inner + columns), PARALLEL_SAMPLES ÷ 4)))
    span = cld(rows, blocks)
    a, b = eltype(C)(alpha), eltype(C)(beta)
    @batch for block in 1:blocks
        r = (block-1)*span+1:min(block * span, rows)
        @inbounds for j in 1:columns
            if iszero(b)
                @simd for t in r
                    C[t, j] = 0
                end
            elseif !isone(b)
                @simd for t in r
                    C[t, j] *= b
                end
            end
            for p in 1:inner
                weight = a * B[p, j]
                @simd for t in r
                    C[t, j] += A[t, p] * weight
                end
            end
        end
    end
    C
end

"""A column-contiguous complex matrix as the real one of its interleaved parts."""
real_rows(X) = reshape(reinterpret(real(eltype(X)), X), 2size(X, 1), size(X, 2))

# Section timers for development: SOLPROP_JULIA_TIMING=1 when the module loads. Off,
# `@section` costs one branch.
const TIMING = Ref(false)
const TIMES = Dict{Symbol,Float64}()
# A device's timed sections wait for its queue at both ends (`synchronize_all`).
const SYNCHRONIZE = Ref{Any}(() -> nothing)
macro section(name, ex)
    quote
        if TIMING[]
            SYNCHRONIZE[]()
            local started = time_ns()
            local result = $(esc(ex))
            SYNCHRONIZE[]()
            TIMES[$name] = get(TIMES, $name, 0.0) + (time_ns() - started) * 1e-9
            result
        else
            $(esc(ex))
        end
    end
end
timings() = Dict(String(k) => v for (k, v) in TIMES)
clear_timings() = (empty!(TIMES); nothing)
timing_enabled() = TIMING[]
