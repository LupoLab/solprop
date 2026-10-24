# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Experimental schema-34 field/Kerr engine. All arrays behind the bridge are owned."""
module SolpropJulia
using LinearAlgebra
using FFTW
using PythonCall
import AcceleratedKernels as AK

const SCHEMA_VERSION = 35

mutable struct Engine{C,W}
    config::C
    work::W
    fsal_valid::Bool
    calls::Int
    propagator_dz::Float64
    factor::Float64
end

# These loops inline their body for one task, and use the same body on GPU arrays.
function foreach_element(f, a, tasks)
    AK.foreachindex(f, a; max_tasks=tasks, min_elems=4096)
end

# Host traversal removes the per-sample integer divide. Device arrays retain a
# flat index kernel with the same scalar body; threading needs enough work per task.
function foreach_sample(f, a, tasks)
    nr = size(a,1)
    foreach_element(a,tasks) do i
        f(mod1(i,nr),(i-1)÷nr+1,i)
    end
end
function foreach_sample(f, a::StridedMatrix, tasks)
    nr = size(a,1)
    AK.foreachindex(axes(a,2);max_tasks=tasks,min_elems=max(1,cld(32768,nr))) do m
        @inbounds @simd for row in axes(a,1)
            @inline f(row,m,row+(m-1)*nr)
        end
    end
end

@inline polarization(value::Real, factor, analytic_factor) = factor*(value*value)*value
@inline polarization(value::Complex, factor, analytic_factor) = factor*analytic_factor*abs2(value)*real(value)

function polarize!(nodes, values, c, factor=c.factor)
    nt = size(nodes,1)
    foreach_element(nodes,c.tasks) do i
        @inbounds nodes[i] = polarization(values[i],factor,c.analytic_factor)*c.window[mod1(i,nt)]
    end
end

# The first profile was dominated by indexing the pointwise Kerr product. This
# CPU-only traversal removes integer division and exposes independent time samples
# to SIMD. The scalar arithmetic is the same as the generic device-ready body.
const HostRealMatrix = Union{Matrix{Float64},SubArray{Float64,2,Matrix{Float64}}}
function polarize_column!(nodes, values, window, factor, analytic_factor, node)
    @inbounds @simd for t in axes(nodes,1)
        nodes[t,node] = polarization(values[t,node],factor,analytic_factor)*window[t]
    end
    nothing
end

function polarize!(nodes::HostRealMatrix, values, c, factor=c.factor)
    AK.foreachindex(axes(nodes,2); max_tasks=c.tasks,min_elems=max(1,cld(32768,size(nodes,1)))) do node
        polarize_column!(nodes,values,c.window,factor,c.analytic_factor,node)
    end
end

function polarize_envelope!(nodes, values, c, factor)
    foreach_sample(nodes,c.tasks) do t,m,i
        @inbounds nodes[i] = factor*c.analytic_factor*abs2(values[i])*values[i]*c.window[t]
    end
end

# AbstractFFTs' normalized inverse plan performs a serial lmul! after its FFT.
# Keep exactly that scale and multiplication order, distributing only independent
# samples when the measured operation is large enough to justify task launches.
fft_apply!(destination,plan,source,c)=mul!(destination,plan,source)
function fft_apply!(destination,plan::FFTW.ScaledPlan,source,c)
    if !c.parallel_fft_scale || c.tasks == 1 || length(destination) < 32768
        return mul!(destination,plan,source)
    end
    mul!(destination,plan.p,source)
    scale=plan.scale
    AK.foreachindex(1:cld(length(destination),32768);max_tasks=c.tasks,min_elems=1) do block
        first=32768(block-1)+1
        last=min(first+32767,length(destination))
        @inbounds @simd for i in first:last
            destination[i]=scale*destination[i]
        end
    end
    destination
end

# AppleAccelerate extends AbstractFFTs for dense Arrays. A strided view selects
# FFTW's planner explicitly, keeping this BLAS choice separate from FFT choice.
fft_storage(a) = view(a, :, :)

function create(input)
    matrix(name, T) = Matrix{T}(input[name])
    vector(name, T) = Vector{T}(input[name])
    scalar(name, T) = convert(T, input[name])
    tasks = scalar("threads", Int)
    FFTW.set_num_threads(scalar("fft_threads", Int))
    BLAS.set_num_threads(scalar("blas_threads", Int))
    plan_flag = lowercase(get(ENV,"SOLPROP_FFTW_PLAN","estimate"))
    fft_flags = get(Dict("estimate"=>FFTW.ESTIMATE,"measure"=>FFTW.MEASURE,"patient"=>FFTW.PATIENT),plan_flag,nothing)
    isnothing(fft_flags) && throw(ArgumentError("SOLPROP_FFTW_PLAN must be estimate, measure or patient"))
    config = (
        plan_flag=plan_flag, fft_flags=fft_flags,
        parallel_fft_scale=get(ENV,"SOLPROP_JULIA_PARALLEL_FFT_SCALE","1")=="1",
        real_root=get(ENV,"SOLPROP_JULIA_REAL_ROOT","1")=="1",
        skip_zero_plasma=get(ENV,"SOLPROP_JULIA_SKIP_ZERO_PLASMA","1")=="1",
        fft_threads=scalar("fft_threads",Int),
        blas_threads=scalar("blas_threads",Int),
        raman_fft_threads=haskey(input,"raman_fft_threads") ? scalar("raman_fft_threads",Int) : scalar("fft_threads",Int),
        transverse_fft_threads=haskey(input,"transverse_fft_threads") ? scalar("transverse_fft_threads",Int) : scalar("fft_threads",Int),
        envelope=haskey(input,"envelope") && scalar("envelope",Bool),
        low_storage=haskey(input,"low_storage") && scalar("low_storage",Bool),
        free=haskey(input,"free") && scalar("free",Bool),
        linear=matrix("linear", ComplexF64), prefactor=matrix("prefactor", ComplexF64),
        synthesis=matrix("synthesis", Float64), projection=matrix("projection", Float64),
        projection_coarse=matrix("projection_coarse", Float64), embedded=scalar("embedded", Bool),
        spatial_absorber=haskey(input,"spatial_absorber") ? matrix("spatial_absorber",Float64) : zeros(0,0),
        crop=vector("crop", Int) .+ 1, mask=vector("mask", Bool),
        norm_mask=vector("norm_mask", Bool), weights=vector("weights", Float64),
        window=vector("window", Float64), omega_window=vector("omega_window", Float64),
        rate=vector("rate", Float64), coarse_rate=vector("coarse_rate", Float64),
        a=matrix("a", Float64), b=vector("b", Float64), e=vector("e", Float64),
        c=vector("c", Float64), factor=scalar("factor", Float64),
        analytic=scalar("analytic", Bool), analytic_factor=scalar("analytic_factor", Float64),
        split=scalar("split", Bool), rtol=scalar("rtol", Float64), atol=scalar("atol", Float64),
        tasks=tasks, cache_propagators=haskey(input,"cache_propagators") ? scalar("cache_propagators",Bool) : true,
    )
    config.free && return create_free(config, input)
    # The translated rates are constant over a trial. An exactly absent loss
    # needs no additive temporal FFT; the accepted-step split remains separate.
    active_absorber = any(!iszero,config.spatial_absorber) ||
        (!config.split && any(!iszero,config.rate))
    config = merge(config, (modal_in_stage_absorber=(
        get(ENV,"SOLPROP_JULIA_SKIP_ZERO_MODAL_ABSORBER","1") == "1" ?
        active_absorber : (!config.split || !isempty(config.spatial_absorber))
    ),))
    nw, nm = size(config.linear)
    nt = length(config.window)
    chunk = min(scalar("chunk", Int), size(config.synthesis, 2))
    nf = config.envelope ? nt : nt ÷ 2 + 1
    T = config.envelope ? ComplexF64 : Float64
    fine = zeros(ComplexF64, nf, nm)
    modal = zeros(T, nt, nm)
    nodes = zeros(T, nt, chunk)
    analytic = zeros(ComplexF64, nt, chunk)
    coarse = zeros(T, length(config.coarse_rate), nm)
    work = (
        adams=adams_workspace(input,nw,nm),
        path=path_workspace(input,config),
        raman=raman_workspace(input,config,nt,chunk),
        plasma=plasma_workspace(input,config,nt,chunk,nm,size(config.synthesis,2)),
        y=zeros(ComplexF64, nw, nm), candidate=zeros(ComplexF64, nw, nm),
        lab=zeros(ComplexF64, nw, nm), source=zeros(ComplexF64, nw, nm),
        fsal=zeros(ComplexF64, nw, nm), error=zeros(ComplexF64, nw, nm),
        stages=zeros(ComplexF64, nw, nm, length(config.c)),
        forward=zeros(ComplexF64, nw, nm, length(config.c)), inverse=zeros(ComplexF64, nw, nm, length(config.c)),
        fine=fine, modal=modal, nodes=nodes, analytic=analytic,
        projected=zeros(T,nt, nm), projected_coarse=zeros(T,nt,nm), coarse_spec=similar(fine), absorber=zeros(T,nt, nm), absorber_spec=similar(fine),
        coarse=coarse, partials=zeros(3, nm),
        inverse_plan=config.envelope ? plan_ifft(fft_storage(fine),1;flags=config.fft_flags) : plan_irfft(fft_storage(fine), nt, 1; flags=config.fft_flags),
        forward_plan=config.envelope ? plan_fft(fft_storage(modal),1;flags=config.fft_flags) : plan_rfft(fft_storage(modal), 1; flags=config.fft_flags),
        analytic_forward=plan_fft!(fft_storage(analytic), 1; flags=config.fft_flags),
        analytic_inverse=plan_ifft!(fft_storage(analytic), 1; flags=config.fft_flags),
        coarse_inverse=config.envelope ? plan_ifft(fft_storage(zeros(ComplexF64,nw,nm)),1;flags=config.fft_flags) : plan_irfft(fft_storage(zeros(ComplexF64,nw,nm)), length(config.coarse_rate), 1; flags=config.fft_flags),
        coarse_forward=config.envelope ? plan_fft(fft_storage(coarse),1;flags=config.fft_flags) : plan_rfft(fft_storage(coarse), 1; flags=config.fft_flags),
    )
    Engine(config, work, false, 0, NaN,config.factor)
end

function load!(engine, input)
    copyto!(engine.work.y, Matrix{ComplexF64}(input))
    mask!(engine, engine.work.y)
    engine.fsal_valid = false
    engine.calls = 0
    nothing
end

function mask!(engine, values)
    mask = engine.config.mask
    foreach_sample(values, engine.config.tasks) do f,m,i
        @inbounds begin
            if !mask[f]
                values[i] = 0
            end
        end
    end
end

"""Evaluate the translated source, with padding/crop scales already in the seam."""
function source!(engine, lab; quadrature=false)
    c, w = engine.config, engine.work
    c.free && return source_free!(engine,lab)
    nt, nm = size(w.modal)
    nw = size(lab, 1)
    fill!(w.fine, 0)
    foreach_sample(lab, c.tasks) do f,m,i
        @inbounds begin
            w.fine[c.crop[f],m] = lab[i]
        end
    end
    fft_apply!(w.modal,w.inverse_plan,w.fine,c)
    fill!(w.projected, 0)
    quadrature && fill!(w.projected_coarse, 0)
    clear_plasma!(w.plasma)
    if engine.factor != 0 || !isnothing(w.plasma) || !isnothing(w.raman)
        chunk = size(w.nodes,2)
        for first in 1:chunk:size(c.synthesis,2)
            last = min(first+chunk-1, size(c.synthesis,2))
            count = last-first+1
            nodes = @view w.nodes[:,1:count]
            mul!(nodes, w.modal, @view(c.synthesis[:,first:last]))
            if !isnothing(w.plasma)
                plasma_current!(w.plasma,nodes,c,first)
                project_plasma!(w.plasma,c,first,last,quadrature)
            end
            if c.envelope
                isnothing(w.raman) || raman_response!(w.raman,nodes,nodes,c)
                polarize_envelope!(nodes,nodes,c,engine.factor)
            elseif c.analytic || (!isnothing(w.raman) && !w.raman.thg)
                fill!(w.analytic, 0)
                @views w.analytic[:,1:count] .= nodes
                w.analytic_forward * w.analytic
                foreach_sample(w.analytic, c.tasks) do f,m,i
                    @inbounds begin
                        w.analytic[i] *= (f == 1 || f == nt÷2+1) ? 1 : f <= nt÷2 ? 2 : 0
                    end
                end
                fft_apply!(w.analytic,w.analytic_inverse,w.analytic,c)
                isnothing(w.raman) || raman_response!(w.raman,nodes,w.analytic,c)
                polarize!(nodes,c.analytic ? w.analytic : nodes,c,engine.factor)
            else
                isnothing(w.raman) || raman_response!(w.raman,nodes,w.analytic,c)
                polarize!(nodes,nodes,c,engine.factor)
            end
            isnothing(w.raman) || add_raman!(w.raman,nodes,c)
            mul!(w.projected, nodes, @view(c.projection[first:last,:]), 1.0, 1.0)
            if quadrature && c.embedded
                mul!(w.projected_coarse, nodes, @view(c.projection_coarse[first:last,:]), 1.0, 1.0)
            end
        end
    end
    mul!(w.fine, w.forward_plan, w.projected)
    isnothing(w.plasma) || add_plasma!(w.fine,w.plasma,w.forward_plan,c)
    in_stage_absorber = c.modal_in_stage_absorber
    if in_stage_absorber
        if c.split
            fill!(w.absorber, 0)
        else
            foreach_sample(w.absorber, c.tasks) do t,m,i
                @inbounds begin
                    w.absorber[i] = -0.5 * c.rate[t] * w.modal[i]
                end
            end
        end
        # The shared radial map has no temporal pad scale. Its loss is additive,
        # independent of Kerr and its material prefactor, under either time scheme.
        if !isempty(c.spatial_absorber)
            mul!(w.absorber, w.modal, c.spatial_absorber, 1.0, 1.0)
        end
        mul!(w.absorber_spec, w.forward_plan, w.absorber)
    end
    foreach_sample(w.source, c.tasks) do f,m,i
        @inbounds begin
            value = c.prefactor[i] * c.omega_window[f] * w.fine[c.crop[f],m]
            if in_stage_absorber && c.mask[f]
                value += w.absorber_spec[c.crop[f],m]
            end
            w.source[i] = value
        end
    end
    engine.calls += 1
    nothing
end

# Constant linear tables make the propagators depend only on dz. Reuse is exact
# for equal step lengths; adaptive trials rebuild on every changed dz.
function propagators!(engine, dz)
    engine.config.free && return
    if !isnothing(engine.work.path)
        path_propagators!(engine)
        return
    end
    engine.config.cache_propagators && engine.propagator_dz == dz && return
    c, w = engine.config, engine.work
    n = length(w.y)
    foreach_element(w.y,c.tasks) do i
        @inbounds for stage in eachindex(c.c)
            j = i + (stage-1)*n
            exponent = (dz*c.c[stage])*c.linear[i]
            w.forward[j] = exp(exponent)
            w.inverse[j] = exp(-exponent)
        end
    end
    engine.propagator_dz = dz
    nothing
end

"""Form one IP-DOPRI5 candidate, preserving y even when the controller rejects it."""
function trial!(engine, dz::Float64)
    c, w = engine.config, engine.work
    nw, nm = size(w.y)
    ns = length(c.c)
    propagators!(engine,dz)
    for stage in 1:ns
        !c.free && !isnothing(w.path) && material_stage!(engine,stage)
        if c.free && stage == 1 && engine.fsal_valid
            # At c[1] == 0 the interaction-frame derivative is the FSAL source.
            copyto!(@view(w.stages[:,:,1]),w.fsal)
            continue
        end
        c.free && free_propagators!(engine,dz*c.c[stage])
        slot = c.free ? 1 : stage
        foreach_sample(w.y, c.tasks) do f,m,i
            @inbounds begin
                combination = zero(ComplexF64)
                for previous in 1:stage-1
                    combination += c.a[stage,previous] * w.stages[i+(previous-1)*length(w.y)]
                end
                w.lab[i] = w.forward[i+(slot-1)*length(w.y)] * (w.y[i] + dz*combination)
            end
        end
        if stage == 1 && engine.fsal_valid
            copyto!(w.source, w.fsal)
        else
            source!(engine, w.lab; quadrature=stage==ns)
            if stage == 1
                copyto!(w.fsal, w.source)
                engine.fsal_valid = true
            end
        end
        foreach_sample(w.source, c.tasks) do f,m,i
            @inbounds begin
                w.stages[i+(stage-1)*length(w.y)] = w.inverse[i+(slot-1)*length(w.y)] * w.source[i]
            end
        end
    end
    foreach_sample(w.y, c.tasks) do f,m,i
        @inbounds begin
            solution, defect = zero(ComplexF64), zero(ComplexF64)
            for stage in 1:ns
                value = w.stages[i+(stage-1)*length(w.y)]
                solution += c.b[stage]*value
                defect += c.e[stage]*value
            end
            flow = w.forward[i+(c.free ? 0 : ns-1)*length(w.y)]
            w.candidate[i] = flow*(w.y[i]+dz*solution)
            w.error[i] = flow*(dz*defect)
        end
    end
    # Column partials retain a clear summation bound and require no per-trial arrays.
    AK.foreachindex(1:nm; max_tasks=c.tasks, min_elems=max(1,cld(32768,nw))) do m
        @inbounds begin
            before, after = 0.0, 0.0
            for f in 1:nw
                if c.norm_mask[f]
                    before += abs2(w.y[f,m])
                    after += abs2(w.candidate[f,m])
                end
            end
            w.partials[1,m], w.partials[2,m] = before, after
        end
    end
    magnitude = max(sqrt(sum(@view w.partials[1,:])), sqrt(sum(@view w.partials[2,:])))
    AK.foreachindex(1:nm; max_tasks=c.tasks, min_elems=max(1,cld(32768,nw))) do m
        @inbounds begin
            residual = 0.0
            for f in 1:nw
                if c.norm_mask[f]
                    scale = c.rtol*c.weights[f]*magnitude+c.atol
                    value = w.error[f,m]
                    residual += scale == 0 && value == 0 ? 0.0 : abs2(value/scale)
                end
            end
            w.partials[3,m] = residual
        end
    end
    sqrt(sum(@view w.partials[3,:]))
end

function commit!(engine, dz::Float64)
    c, w = engine.config, engine.work
    mask!(engine, w.candidate)
    if c.split && any(!iszero, c.coarse_rate)
        if c.free
            split_free!(engine,dz)
        else
        fft_apply!(w.coarse,w.coarse_inverse,w.candidate,c)
        nt = size(w.coarse,1)
        foreach_sample(w.coarse, c.tasks) do t,m,i
            @inbounds begin
                w.coarse[i] *= exp(-0.5*dz*c.coarse_rate[t])
            end
        end
        mul!(w.candidate, w.coarse_forward, w.coarse)
        end
        mask!(engine, w.candidate)
        engine.fsal_valid = false
    else
        if !c.low_storage
            copyto!(w.fsal, w.source)
            engine.fsal_valid = true
        end
    end
    copyto!(w.y, w.candidate)
    if !c.embedded
        return NaN
    end
    mul!(w.coarse_spec, w.forward_plan, w.projected_coarse)
    isnothing(w.plasma) || add_plasma!(w.coarse_spec,w.plasma,w.forward_plan,c;coarse=true)
    scale, defect = 0.0, 0.0
    for m in axes(w.y,2), f in axes(w.y,1)
        weight = c.prefactor[f,m]*c.omega_window[f]
        fine = weight*w.fine[c.crop[f],m]
        coarse = weight*w.coarse_spec[c.crop[f],m]
        scale += abs2(fine)
        defect += abs2(coarse-fine)
    end
    scale == 0 ? NaN : sqrt(defect/scale)
end

snapshot(engine) = copy(engine.work.y)
candidate(engine) = copy(engine.work.candidate)
source_snapshot(engine) = copy(engine.work.source)
error_snapshot(engine) = copy(engine.work.error)
function evaluate!(engine, input)
    lab = Matrix{ComplexF64}(input)
    source!(engine, lab)
    source_snapshot(engine)
end

# Plans have implementation-owned storage; explicitly owned arrays are counted here.
function ledger(engine)
    result = Dict{String,Int}()
    function visit(prefix,object)
        for name in propertynames(object)
            value=getproperty(object,name)
            key="$(prefix).$(name)"
            if value isa Array && !isempty(value)
                result[key]=sizeof(value)
            elseif value isa NamedTuple
                visit(key,value)
            end
        end
    end
    visit("config",engine.config)
    visit("work",engine.work)
    result
end

include("adams.jl")
include("rk4.jl")
include("stages.jl")
include("responses.jl")
include("free.jl")

runtime() = (string(VERSION), Threads.nthreads(), FFTW.get_num_threads(), BLAS.get_num_threads(), string(BLAS.get_config()))
const rk4_b = rk4!
const configure_stages_b = configure_stages!
const load_b = load!
const trial_b = trial!
const commit_b = commit!
const evaluate_b = evaluate!

end
