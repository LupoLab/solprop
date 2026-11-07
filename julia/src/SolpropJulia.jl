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
        # A binary32 state is held as a / s, s an exact power of two (SEP 33, rule 2).
        scale=host.has("state_scale") ? host.real("state_scale") : 1.0,
    )
    engine = c.free ? create_free(c, host) : create_modal(c, host)
    oncpu(c) || (SYNCHRONIZE[] = () -> KA.synchronize(backend))
    compile!(engine)
    warm!(engine)
    engine
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
const rk4_b = rk4!
const configure_stages_b = configure_stages!
const load_b = load!
const bind_monitor_b = bind_monitor!
const copy_state_b = copy_state!
const bind_edge_b = bind_edge!
const trial_b = trial!
const commit_b = commit!
const evaluate_b = evaluate!
const reset_work_b = reset_work!

end
