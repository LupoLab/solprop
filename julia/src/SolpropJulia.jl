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
include("responses.jl")
include("modal.jl")
include("free.jl")
include("dopri.jl")
include("rk4.jl")
include("adams.jl")
include("stages.jl")
include("observe.jl")

real_type(engine) = real(eltype(engine.work.y))

# Read at load, not at precompilation: a constant would be baked into the cache.
function __init__()
    TIMING[] = get(ENV, "SOLPROP_JULIA_TIMING", "0") == "1"
    CHECK[] = get(ENV, "SOLPROP_JULIA_CHECK", "0") == "1"
    nothing
end

"""Read the shared settings and route to the modal or the free-space engine."""
function create(input)
    host = HostInput(input)
    T = host.has("precision") && host.string("precision") == "single" ? Float32 : Float64
    backend = device_backend(Val(Symbol(host.has("device") ? host.string("device") : "cpu")))
    # One pool runs everything (launch.jl, `plan`): the libraries' own are single.
    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)
    c = (
        backend=backend, T=T, tasks=host.int("threads"),
        fft_threads=host.int("fft_threads"),
        raman_fft_threads=host.has("raman_fft_threads") ? host.int("raman_fft_threads") : host.int("fft_threads"),
        transverse_fft_threads=host.has("transverse_fft_threads") ? host.int("transverse_fft_threads") : host.int("fft_threads"),
        envelope=host.flag("envelope"), low_storage=host.flag("low_storage"),
        free=host.flag("free"), split=host.flag("split"), analytic=host.flag("analytic"),
        analytic_factor=T(host.real("analytic_factor")), factor=host.real("factor"),
        rtol=host.real("rtol"), atol=host.real("atol"),
        cache_propagators=host.has("cache_propagators") ? host.flag("cache_propagators") : true,
    )
    engine = c.free ? create_free(c, host) : create_modal(c, host)
    oncpu(c) || (SYNCHRONIZE[] = () -> KA.synchronize(backend))
    compile!(engine)
    engine
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
const rk4_b = rk4!
const configure_stages_b = configure_stages!
const load_b = load!
const bind_monitor_b = bind_monitor!
const bind_edge_b = bind_edge!
const trial_b = trial!
const commit_b = commit!
const evaluate_b = evaluate!

end
