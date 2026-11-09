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

# The body inlined at the call (as the CPU loops do): left to the inliner, a body of a
# few operations became a device call per sample, its arguments through local memory,
# 9 times slower in binary64 and 20 in binary32 on an A40 (a 1.08 M-sample cube, 296
# against 33 us; DMOG `scripts/each_cost3.jl`).
KA.@kernel inbounds = true function each_kernel!(f)
    t, m = KA.@index(Global, NTuple)
    @inline f(t, m)
end
function each_on!(f::F, backend, tasks, nt, nm, heavy) where {F}
    tally!(:launch)
    if KERNEL_TIMING[]
        SYNCHRONIZE[]()
        started = time_ns()
        each_kernel!(backend, (256, 1))(f; ndrange=(nt, nm))
        SYNCHRONIZE[]()
        time_kernel!(F, nt * nm, time_ns() - started)
        return nothing
    end
    each_kernel!(backend, (256, 1))(f; ndrange=(nt, nm))
    nothing
end

# SOLPROP_JULIA_KERNEL_TIMING=1: each GPU pass timed alone (synchronised around it),
# by the body's name and size: which closure a profiler's generic kernel is.
const KERNEL_TIMING = Ref(false)
const KERNEL_TIMES = Dict{String,Tuple{Int,Float64,Int}}()
function time_kernel!(::Type{F}, size, elapsed) where {F}
    name = replace(string(nameof(F)), r"#+" => "#")
    count, total, _ = get(KERNEL_TIMES, name, (0, 0.0, size))
    KERNEL_TIMES[name] = (count + 1, total + elapsed * 1e-9, size)
end
kernel_timings() = Dict(k => collect(v) for (k, v) in KERNEL_TIMES)
clear_kernel_timings() = empty!(KERNEL_TIMES)

"""`out = transpose(inp)` for two device matrices, through 32 x 32 tiles in local memory:
both the reads and the writes coalesced. GPUArrays' `permutedims!` took 2.65 ms to
transpose a 384 x 36864 binary64 complex state on an A40, three times the copy's."""
function transpose_device!(out, inp, c)
    n1, n2 = size(inp)
    tally!(:transpose)
    transpose_kernel!(c.backend, (32, 32))(out, inp; ndrange=(cld(n1, 32) * 32, cld(n2, 32) * 32))
    out
end
KA.@kernel inbounds = true function transpose_kernel!(out, @Const(inp))
    ti, tj = KA.@index(Local, NTuple)
    bi, bj = KA.@index(Group, NTuple)
    tile = KA.@localmem eltype(out) (33, 32)
    n1, n2 = size(inp)
    i, j = (bi - 1) * 32 + ti, (bj - 1) * 32 + tj
    if i <= n1 && j <= n2
        tile[ti, tj] = inp[i, j]
    end
    KA.@synchronize
    # The tile's transpose, to the block on the other side of the diagonal.
    p, q = (bj - 1) * 32 + ti, (bi - 1) * 32 + tj
    if p <= n2 && q <= n1
        out[p, q] = tile[tj, ti]
    end
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
        @inbounds scratch[t, m] = @inline f(t, m)
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
    # The bodies run as an elementwise kernel into scratch, inlined (`each_kernel!`), and
    # only the values reduce: inside AK's reduction the body was a device call per
    # element, and a heavy one got several elements a thread (six blocks for rdw's
    # 10244, its predictor 539 us a trial on an A40).
    scratch = SolpropJulia.scratch(c, heavy ? :heavy : :reduce, typeof(neutral), nt * nm)
    each!(c, nt, nm) do t, m
        @inbounds scratch[t+(m-1)*nt] = @inline f(t, m)
    end
    # The reduction's own scratch, kept (AK's `workspace`): a new partials buffer a call
    # was an allocation from the pool and its copies, dozens a step.
    workspace = get!(c.scratch, (:reduce_workspace, typeof(op), typeof(neutral), nt * nm)) do
        AK.workspace(AK.mapreduce, identity, op, scratch; backend=c.backend, init=neutral,
                     neutral=neutral)
    end
    AK.mapreduce(identity, op, scratch; backend=c.backend, init=neutral, neutral=neutral,
                 workspace=workspace)
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
        @inbounds terms[1, t+(m-1)*nt] = @inline f(t, m)
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
    oncpu(c) || return device_plan(c, kind, a, args...)
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
        held = (staged, result, eltype(x) <: Real ? plan(c, plan_rfft, staged, 1) : plan(c, plan_fft, staged, 1))
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
    # A scale (a taper's) at the arrays' precision: an Apple GPU takes no binary64.
    alpha isa Bool || (alpha = real(eltype(C))(alpha))
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

"""The running sums down each column of `v` (`AK.accumulate!`, `dims=1`), with the
scan's scratch kept on a GPU: a new one a call was a pool allocation, and inside a
captured source graph (`source!`) a freed buffer the replay would still read."""
function scan_columns!(v, c, init)
    oncpu(c) && return AK.accumulate!(+, v; dims=1, init=init, backend=c.backend)
    workspace = get!(c.scratch, (:scan_workspace, eltype(v), size(v))) do
        AK.workspace(AK.accumulate!, +, v; dims=1, init=init, backend=c.backend)
    end
    AK.accumulate!(+, v; dims=1, init=init, backend=c.backend, workspace=workspace)
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
# SOLPROP_JULIA_TIMING=host: the sections unsynchronised, so each measures the host's
# own time in it (enqueueing, host arithmetic, and any wait it makes itself).
const HOST_TIMING = Ref(false)
macro section(name, ex)
    quote
        if TIMING[]
            HOST_TIMING[] || SYNCHRONIZE[]()
            local started = time_ns()
            local result = $(esc(ex))
            HOST_TIMING[] || SYNCHRONIZE[]()
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

# --- Device graphs -------------------------------------------------------------------

"""`SOLPROP_JULIA_GRAPH=0` launches every source's passes one by one."""
const GRAPHS = Ref(true)
"""Whether the device's extension captures and replays graphs (CUDA's does)."""
graph_capable(backend) = false
"""Capture `f`'s device work on the task's stream into an executable graph, or return
`nothing` where the device cannot."""
graph_capture(backend, f) = nothing
graph_launch(exec) = throw(ArgumentError("no device graphs on this backend"))

# --- Stage scalars in device memory ---------------------------------------------------

"""A source's stage scalars on a device that replays graphs: the Kerr factor, then each
plasma channel's density and preionisation, in one device vector the kernels read
(`stage_value`), so that one captured graph serves every stage of a gradient. Slot 1 is
the factor; channel `k`'s density and preionisation are slots `2k` and `2k + 1`."""
function stage_slots(c, slots)
    get!(c.scratch, :stage_slots) do
        # (The staging is pageable on purpose: the driver copies it at once, so the
        # next stage may rewrite it; a pinned copy would only be queued.)
        (host=fill(NaN, slots), device=KA.allocate(c.backend, c.T, slots), staging=zeros(c.T, slots))
    end
end
"""A kernel's view of stage scalar `index` (times `scale`): the device vector's slot
where sources replay graphs, else the host's value."""
function stage_scalar(c, index, host_value, scale=one(c.T))
    T = c.T
    graph_values(c) || return T(host_value) * T(scale)
    (values=c.scratch[:stage_slots].device, index=index, scale=T(scale))
end
@inline stage_value(x::Number) = x
@inline stage_value(s::NamedTuple{(:values, :index, :scale)}) = @inbounds s.values[s.index] * s.scale

# --- Long binary64 transforms on a GPU, four-step -------------------------------------

"""Whether the device's library takes long binary64 transforms slowly enough to split
them (`LongFFT`): an extension's hook, false unless it says otherwise."""
long_transforms(backend) = get(ENV, "SOLPROP_JULIA_FOURSTEP", "") == "1"

"""Plan `kind(a, args...)` on a GPU: the library's plan, or a four-step one (`LongFFT`)
for a long binary64 transform along the first dimension where the device asks for it."""
function device_plan(c, kind, a, args...)
    split = long_transforms(c.backend) && get(ENV, "SOLPROP_JULIA_FOURSTEP", "1") != "0"
    split || return kind(a, args...)
    real_input = kind === plan_rfft
    output_real = kind === plan_brfft
    n = output_real ? Int(args[1]) : size(a, 1)
    region = last(args)
    shape = factor_length(n)
    (ndims(a) == 2 && region == 1 && real(eltype(a)) === Float64 && shape !== nothing &&
     (!real_input && !output_real ? true : iseven(n))) || return kind(a, args...)
    sign = kind in (plan_fft, plan_fft!, plan_rfft) ? -1 : 1
    LongFFT(c, a, n, shape..., sign, kind in (plan_fft!, plan_bfft!), real_input, output_real)
end

"""`n = n1 n2` with `n2` the smallest divisor that keeps `n1 <= 512`, for `n >= 2048`;
`nothing` otherwise (short transforms, or a large prime factor)."""
function factor_length(n)
    n < 2048 && return nothing
    for n2 in cld(n, 512):64
        n % n2 == 0 && return (n ÷ n2, n2)
    end
    nothing
end

"""A length-`n` transform along the first dimension as `n = n1 n2`: with `n = a + n1 j`
and `k = n2 p + q`, `X[n2 p + q] = sum_a w_n1^(a p) w_n^(a q) sum_j x[a + n1 j] w_n2^(j q)`.
The samples go to an `(n1, m, n2)` work array, the short transforms run along its last
dimension (one plan, strided), the twiddles, the long transforms along its first
(contiguous), and the result comes back in order. On an A40, cuFFT's own binary64
transform of 5120 or 10240 points is one block a transform, 96 or 138 us at any batch;
the split takes about 23 (DMOG, `scripts/fft_fourstep2.jl`). Real input is promoted to
complex, and a real output's Hermitian spectrum is completed first: both twice the
arithmetic of a real transform, still a quarter of the time."""
struct LongFFT{S,L,W,T,C}
    short::S
    long::L
    work::W
    twiddle::T
    device::C
    n::Int
    n1::Int
    n2::Int
    columns::Int
    inplace::Bool
    real_input::Bool
    output_real::Bool
    sign::Int
    # The same transform over other column counts, planned when first asked.
    others::Dict{Int,Any}
end
function LongFFT(c, a, n, n1, n2, sign, inplace, real_input, output_real)
    twiddle = upload(c, [cispi(2 * sign * (p - 1) * (j - 1) / n) for p in 1:n1, j in 1:n2])
    # What `each!` reads of the engine's configuration.
    device = (backend=c.backend, tasks=c.tasks)
    long_columns(a, size(a, 2), twiddle, device, n, n1, n2, sign, inplace, real_input, output_real)
end
function long_columns(a, m, twiddle, device, n, n1, n2, sign, inplace, real_input, output_real)
    work = similar(a, ComplexF64, n1, m, n2)
    short = sign < 0 ? plan_fft!(work, 3) : plan_bfft!(work, 3)
    long = sign < 0 ? plan_fft!(work, 1) : plan_bfft!(work, 1)
    LongFFT(short, long, work, twiddle, device, n, n1, n2, m, inplace, real_input, output_real,
            sign, Dict{Int,Any}())
end

function long_transform!(y, p::LongFFT, x)
    m = size(x, 2)
    if m != p.columns
        q = get!(p.others, m) do
            long_columns(p.work, m, p.twiddle, p.device, p.n, p.n1, p.n2, p.sign, p.inplace,
                         p.real_input, p.output_real)
        end
        return long_transform!(y, q, x)
    end
    long_transform!(y, p, x, Val(p.real_input), Val(p.output_real))
end
function long_transform!(y, p::LongFFT, x, ::Val{R}, ::Val{O}) where {R,O}
    (; work, twiddle, n, n1, n2, columns) = p
    c = p.device
    half = n ÷ 2
    tally!(:long_fft)
    # The samples in, `x[a + n1 j]` to `work[a, b, j]`; a half spectrum completed.
    each!(c, n, columns) do t, b
        @inbounds begin
            i = t - 1
            value = O ? (i <= half ? complex(x[t, b]) : conj(complex(x[n-i+1, b]))) :
                    complex(x[t, b])
            work[i % n1 + 1, b, i ÷ n1 + 1] = value
        end
    end
    p.short * work
    flat = reshape(work, n1, columns * n2)
    each!(c, n1, columns * n2) do a, r
        @inbounds flat[a, r] *= twiddle[a, (r - 1) ÷ columns + 1]
    end
    p.long * work
    # `work[p, b, q]` is `X[n2 p + q]`.
    rows = R ? half + 1 : n
    each!(c, rows, columns) do t, b
        @inbounds begin
            k = t - 1
            value = work[k ÷ n2 + 1, b, k % n2 + 1]
            y[t, b] = O ? real(value) : value
        end
    end
    y
end

LinearAlgebra.mul!(y, p::LongFFT, x) = long_transform!(y, p, x)
Base.:*(p::LongFFT, x) = p.inplace ? long_transform!(x, p, x) :
                         throw(ArgumentError("an out-of-place LongFFT takes mul!"))
