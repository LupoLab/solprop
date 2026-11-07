# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Time one pointwise body under each CPU launch strategy the engine could use.
# Run: julia -t N --project=julia/bench julia/bench/launch.jl
using BenchmarkTools, Polyester, OhMyThreads
import AcceleratedKernels as AK
import KernelAbstractions as KA

@inline kerr(v, f, w) = f * (v * v) * v * w

function serial!(out, x, w, f)
    @inbounds for m in axes(out, 2)
        @simd for t in axes(out, 1)
            out[t, m] = kerr(x[t, m], f, w[t])
        end
    end
end

# Flat blocks of contiguous samples: the same work split for any column count.
const BLOCK = 16384
@inline function block!(out, x, w, f, b)
    nt = size(out, 1)
    first, last = (b - 1) * BLOCK + 1, min(b * BLOCK, length(out))
    @inbounds @simd for i in first:last
        t = i - ((i - 1) ÷ nt) * nt
        out[i] = kerr(x[i], f, w[t])
    end
end
function ak!(out, x, w, f)
    AK.foreachindex(1:cld(length(out), BLOCK); min_elems=1) do b
        block!(out, x, w, f, b)
    end
end
function poly!(out, x, w, f)
    @batch for b in 1:cld(length(out), BLOCK)
        block!(out, x, w, f, b)
    end
end
function static!(out, x, w, f)
    Threads.@threads :static for b in 1:cld(length(out), BLOCK)
        block!(out, x, w, f, b)
    end
end
function omt!(out, x, w, f)
    tforeach(1:cld(length(out), BLOCK); scheduler=:static) do b
        block!(out, x, w, f, b)
    end
end
# Column-blocked with a contiguous inner loop (the prototype's host traversal).
function columns!(out, x, w, f)
    AK.foreachindex(axes(out, 2); min_elems=max(1, cld(32768, size(out, 1)))) do m
        @inbounds @simd for t in axes(out, 1)
            out[t, m] = kerr(x[t, m], f, w[t])
        end
    end
end
KA.@kernel function kerr_kernel!(out, @Const(x), @Const(w), f)
    t, m = KA.@index(Global, NTuple)
    @inbounds out[t, m] = kerr(x[t, m], f, w[t])
end
function ka!(out, x, w, f)
    kerr_kernel!(KA.CPU(), 1024)(out, x, w, f; ndrange=size(out))
    KA.synchronize(KA.CPU())
end
broadcast!(out, x, w, f) = (out .= kerr.(x, f, w); nothing)

for (nt, nm) in ((10240, 4), (20480, 64), (10240, 1024))
    x, w = randn(nt, nm), rand(nt)
    out = similar(x)
    println("size $nt x $nm, threads $(Threads.nthreads())")
    for (name, f!) in (("serial", serial!), ("broadcast", broadcast!), ("ak-flat", ak!),
                       ("polyester", poly!), ("threads", static!), ("ohmythreads", omt!),
                       ("ak-columns", columns!), ("ka-cpu", ka!))
        t = @belapsed $f!($out, $x, $w, 0.5)
        println(rpad("  $name", 16), round(t * 1e6; digits=1), " us  ",
                round(16e-9 * length(out) / t; digits=1), " GB/s")
    end
end
