# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Julia backend experiment's engine.

One source for every device and precision: the arrays are `Array`s or a GPU array type
(loaded by a package extension), of `Float64` or `Float32`, and every state-sized
operation is a scalar body launched by `each!`, `each_column!` or `reduce_each`
(`launch.jl`). FFTs go through AbstractFFTs plans and contractions through `mul!`, so
FFTW and BLAS on a CPU, cuFFT and cuBLAS on an NVIDIA GPU. A body must capture only
arrays and scalars, never a configuration or workspace: those hold plans and strings,
which a GPU kernel cannot take.
"""
module SolpropJulia
using LinearAlgebra
using FFTW
using PythonCall
using Polyester: @batch
import AcceleratedKernels as AK
import KernelAbstractions as KA
import Atomix

# The digest of the sources this module was loaded from, set by the Python loader.
const SOURCE_DIGEST = Ref("")
record_digest(digest) = (SOURCE_DIGEST[] = String(digest); nothing)
loaded_digest() = SOURCE_DIGEST[]

mutable struct Engine{C,W}
    config::C
    work::W
    fsal_valid::Bool
    calls::Int
    propagator_dz::Float64
    factor::Float64
end

include("launch.jl")
include("dfloat.jl")
include("vmath.jl")
include("responses.jl")
include("modal.jl")
include("free.jl")
include("dopri.jl")
include("rk4.jl")
include("adams.jl")
include("helpers.jl")
include("march.jl")
include("stages.jl")
include("observe.jl")

real_type(engine) = real(eltype(engine.work.y))

# Read at load, not at precompilation: a constant would be baked into the cache.
function __init__()
    TIMING[] = get(ENV, "SOLPROP_JULIA_TIMING", "0") == "1"
    CHECK[] = get(ENV, "SOLPROP_JULIA_CHECK", "0") == "1"
    SAMPLE_PLASMA[] = get(ENV, "SOLPROP_JULIA_PLASMA", "columns") == "samples"
    nothing
end

"""Read the shared settings and route to the modal or the free-space engine."""
function create(input)
    host = HostInput(input)
    T = host.has("precision") && host.string("precision") == "single" ? Float32 : Float64
    device = host.has("device") ? host.string("device") : "cpu"
    backend = device_backend(Val(Symbol(device)))
    # The wide type of the operator's exponents and the Adams flow: binary64, or a
    # double-float in a binary32 run that asks for it (an Apple GPU has no binary64).
    W = wide_type(T, host.has("wide") ? host.string("wide") : "binary64")
    device == "metal" && W === Float64 &&
        throw(ArgumentError("an Apple GPU has no binary64: metal runs single precision with wide = df32"))
    # SOLPROP_JULIA_FTZ=1: flush subnormals to zero in the engine's numerical work (an
    # experiment; solprop's policy forbids it, #561).
    flush_subnormals!(get(ENV, "SOLPROP_JULIA_FTZ", "0") == "1")
    # Apple's Accelerate for the products on macOS (SOLPROP_JULIA_BLAS=openblas keeps
    # OpenBLAS): radial's window on one M5 Max thread 1.02 s -> 0.60 s, no case slower.
    blas = get(ENV, "SOLPROP_JULIA_BLAS", Sys.isapple() ? "accelerate" : "openblas")
    blas in ("accelerate", "openblas") ||
        throw(ArgumentError("SOLPROP_JULIA_BLAS must be accelerate or openblas"))
    blas == "accelerate" && use_accelerate!()
    # One pool runs everything (launch.jl, `plan`): the libraries' own are single.
    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)
    c = (
        backend=backend, T=T, wide=zero(W), tasks=host.int("threads"), scratch=Dict{Any,Any}(),
        fft_threads=host.int("fft_threads"),
        raman_fft_threads=host.has("raman_fft_threads") ? host.int("raman_fft_threads") : host.int("fft_threads"),
        transverse_fft_threads=host.has("transverse_fft_threads") ? host.int("transverse_fft_threads") : host.int("fft_threads"),
        envelope=host.flag("envelope"), low_storage=host.flag("low_storage"),
        free=host.flag("free"), split=host.flag("split"), analytic=host.flag("analytic"),
        analytic_factor=T(host.real("analytic_factor")), factor=host.real("factor"),
        rtol=host.real("rtol"), atol=host.real("atol"),
        cache_propagators=host.has("cache_propagators") ? host.flag("cache_propagators") : true,
        # A binary32 state is held as a / s, s an exact power of two (SEP 33, rule 2).
        scale=host.has("state_scale") ? host.real("state_scale") : 1.0,
    )
    engine = c.free ? create_free(c, host) : create_modal(c, host)
    oncpu(c) || (SYNCHRONIZE[] = () -> KA.synchronize(backend))
    compile!(engine)
    warm!(engine)
    engine
end

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
"""Forward libblastrampoline's BLAS and LAPACK to Accelerate's new interfaces (macOS 13.4
and later), LP64 and ILP64, keeping OpenBLAS beneath for what Accelerate lacks, as
AppleAccelerate.jl does (the `\x1a` marks an exact suffix); then set Accelerate
single-threaded (macOS 15 and later), since the engine's pool already splits its
products into blocks of rows (`gemm!`). Return the configuration."""
function use_accelerate!()
    Sys.isapple() || throw(ArgumentError("SOLPROP_JULIA_BLAS=accelerate needs macOS"))
    BLAS.lbt_forward(ACCELERATE; suffix_hint="\x1a\$NEWLAPACK")
    BLAS.lbt_forward(ACCELERATE; suffix_hint="\x1a\$NEWLAPACK\$ILP64")
    # BLASSetThreading(BLAS_THREADING_SINGLE_THREADED = 1) returns 0 on success.
    ccall((:BLASSetThreading, ACCELERATE), Cint, (Cint,), 1) == 0 ||
        throw(ErrorException("Accelerate's BLASSetThreading failed"))
    BLAS.get_config()
end

"""Set or clear flush-to-zero (and denormals-are-zero) for the engine's numerical work.

A CPU's floating-point control register is per thread. Julia's other threads run only
pool tasks, so theirs is set for good; thread one is the Python interpreter's own,
whose NumPy and `math` must keep subnormals (a flushed one made a `math.log` of a
tiny positive value fail), so the entry points set it on entry and restore it on exit
(`flushed`). GPU kernels are unaffected."""
function flush_subnormals!(on::Bool)
    FTZ[] = on
    Threads.@threads :static for k in 1:Threads.nthreads()
        k == 1 || set_zero_subnormals(on)
    end
    nothing
end
const FTZ = Ref(false)
@inline function flushed(f::F, args...) where {F}
    FTZ[] || return f(args...)
    previous = get_zero_subnormals()
    set_zero_subnormals(true)
    try
        f(args...)
    finally
        set_zero_subnormals(previous)
    end
end

"""Run each path once on the zero state, then forget it: a GPU's kernels compile at
their first launch, which `compile!`'s inference does not reach."""
function warm!(engine)
    c, w = engine.config, engine.work
    a = w.adams
    if c.low_storage
        rk4!(engine, 1e-12, 1e-12)
    elseif a !== nothing
        # Valid history rows, so that no warm trial reads outside the store.
        fill!(a.host.slots, 1)
        copyto!(a.device.slots, a.host.slots)
        adams_start!(engine)
        for n in 1:size(a.moments, 3)
            a.sizes .= (min(n, size(a.moments, 3) - 1), n, 1)
            adams_trial_width!(engine, 1e-12, Val(n))
        end
        adams_accept!(engine, 1)
        fill!(a.sizes, 0)
    else
        trial!(engine, 1e-12)
    end
    commit!(engine, 1e-12, true)
    # The observers' paths too: the monitor, the edge (on a provisional binding), the
    # occupancy and the populations.
    state_power(engine)
    host_state(engine)
    occupancy(engine)
    nm = size(w.y, 2)
    bind_edge!(engine, c.free ? nothing : Matrix{Float64}(I, nm, nm), [0])
    node_energy(engine)
    column_energy(engine)
    w.edge[] = nothing
    w.plasma === nothing || population_snapshot(engine, nothing)
    reset_work!(engine)
end

"""Zero the state and every buffer a warm-up wrote, and the engine's counters."""
function reset_work!(engine)
    c, w = engine.config, engine.work
    a = w.adams
    buffers = (w.y, w.candidate, w.lab, w.source, w.error, w.stages, w.forward, w.inverse)
    a === nothing || (buffers = (buffers..., a.store, a.states, a.rates, a.moments, a.carried, a.endpoint))
    foreach(buffer -> fill!(buffer, 0), buffers)
    engine.fsal_valid, engine.calls, engine.propagator_dz = false, 0, NaN
    TIMING[] && empty!(TIMES)
    nothing
end

"""Compile every entry point the march calls for this engine's types, now.

Construction is timed apart from propagation; without this, a run's first step and
first boundary reading pay seconds of compilation inside the propagation's time.
"""
function compile!(engine)
    E = typeof(engine)
    for (f, arguments) in ((trial!, (E, Float64)), (commit!, (E, Float64, Bool)),
                           (rk4!, (E, Float64, Float64)), (state_power, (E,)),
                           (split_power, (E,)), (occupancy, (E,)), (host_state, (E,)),
                           (host_candidate, (E,)), (adams_start!, (E,)), (adams_accept!, (E, Int)),
                           (material_stage!, (E, Int)), (reset!, (E,)))
        precompile(f, arguments)
    end
    a = engine.work.adams
    if a !== nothing
        for n in 1:size(a.moments, 3)
            precompile(adams_trial_width!, (E, Float64, Val{n}))
        end
    end
    nothing
end

function reset!(engine)
    mask!(engine, engine.work.y)
    engine.fsal_valid = false
    engine.calls = 0
    nothing
end
device_matrix(engine, input) =
    upload(engine.config.backend, Matrix{Complex{real_type(engine)}}(input))

function mask!(engine, values)
    mask = engine.config.mask
    each!(engine.config, size(values)...) do f, m
        @inbounds if !mask[f]
            values[f, m] = 0
        end
    end
end

complex_host(a) = Matrix{ComplexF64}(Array(a))
function evaluate!(engine, input)
    copyto!(engine.work.lab, device_matrix(engine, input))
    source!(engine, engine.work.lab)
    complex_host(engine.work.source)
end

# Plans hold implementation-owned storage; the arrays the engine owns are counted.
function ledger(engine)
    result = Dict{String,Int}()
    function visit(prefix, object)
        for name in propertynames(object)
            value = getproperty(object, name)
            key = "$(prefix).$(name)"
            if value isa DenseArray && !isempty(value)
                result[key] = sizeof(value)
            elseif value isa NamedTuple
                visit(key, value)
            end
        end
    end
    visit("config", engine.config)
    visit("work", engine.work)
    result
end

device_name(engine) = device_label(engine.config.backend)
runtime() = (string(VERSION), Threads.nthreads(), FFTW.get_num_threads(),
             BLAS.get_num_threads(), string(BLAS.get_config()))
rk4_b(args...) = flushed(rk4!, args...)
const configure_stages_b = configure_stages!
const load_b = load!
const bind_monitor_b = bind_monitor!
const copy_state_b = copy_state!
const bind_edge_b = bind_edge!
trial_b(args...) = flushed(trial!, args...)
commit_b(args...) = flushed(commit!, args...)
evaluate_b(args...) = flushed(evaluate!, args...)
const reset_work_b = reset_work!


# The specialisations the benchmark windows compiled (every focus case, binary64, one
# and sixteen threads), replayed when the package is built so that their native code is
# in its image: `julia/src/precompile_trace.jl`, recorded with juliacall's
# `PYTHON_JULIACALL_TRACE_COMPILE`. A statement that no longer resolves is skipped.
import KernelAbstractions, AcceleratedKernels, FFTW, Polyester, Atomix
let file = joinpath(@__DIR__, "precompile_trace.jl")
    if ccall(:jl_generating_output, Cint, ()) == 1 && isfile(file)
        for line in eachline(file)
            try
                Base.eval(@__MODULE__, Meta.parse(line))
            catch
            end
        end
    end
end
end
