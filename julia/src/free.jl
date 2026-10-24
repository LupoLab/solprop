# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Cartesian geometry keeps shared coefficient vectors and one current-stage flow.
# FFT/array planning is CPU-specific; the scalar loops and Kerr body are shared.
function create_free(common, input)
    vector(name,T) = Vector{T}(input[name])
    shape = Tuple(Int.(input["spatial_shape"]))
    c = merge(common, (
        parallel_copy=get(ENV,"SOLPROP_JULIA_PARALLEL_COPY","1")=="1",
        spatial_shape=shape, pad_scale=Float64(input["pad_scale"]),
        spatial_rate=vector("spatial_rate",Float64),
        k_squared=vector("k_squared",ComplexF64), k_squared_low=vector("k_squared_low",ComplexF64),
        kperp_sq=vector("kperp_sq",Float64), kperp_sq_low=vector("kperp_sq_low",Float64),
        frame=vector("frame",Float64), spectral_rate=vector("spectral_rate",Float64),
        prefactor_numerator=vector("prefactor_numerator",ComplexF64),
        prefactor_clamped=vector("prefactor_clamped",ComplexF64), active=vector("active",Bool),
        kperp_rate=vector("kperp_rate",Float64), evanescent_rate=Float64(input["evanescent_rate"]),
    ))
    # Only an exactly absent additive source can omit these transforms. The
    # accepted-step temporal split and the diagonal spectral loss are separate.
    active_absorber = any(!iszero,c.spatial_rate) || (!c.split && any(!iszero,c.rate))
    c = merge(c, (in_stage_absorber=(
        get(ENV,"SOLPROP_JULIA_SKIP_ZERO_ABSORBER","1") != "1" || active_absorber
    ),))
    nw, nm, nt = length(c.crop), prod(shape), length(c.window)
    chunk = min(Int(input["chunk"]),nm)
    nf, nc = c.envelope ? nt : nt÷2+1, length(c.coarse_rate)
    T = c.envelope ? ComplexF64 : Float64
    stages = c.low_storage ? 0 : length(c.c)
    physical = zeros(ComplexF64,nw,nm)
    spatial = view(reshape(physical,nw,shape...),:,:,:)
    fine, modal = zeros(ComplexF64,nf,chunk), zeros(T,nt,chunk)
    analytic = zeros(ComplexF64,nt,chunk)
    coarse, coarse_spec = zeros(T,nc,chunk), zeros(ComplexF64,nw,chunk)
    w = (
        plasma=plasma_workspace(input,c,nt,chunk,chunk,nm),
        y=zeros(ComplexF64,nw,nm), candidate=zeros(ComplexF64,nw,nm),
        lab=zeros(ComplexF64,nw,nm), source=zeros(ComplexF64,nw,nm),
        fsal=zeros(ComplexF64,c.low_storage ? 0 : nw,c.low_storage ? 0 : nm), error=zeros(ComplexF64,nw,nm),
        stages=zeros(ComplexF64,nw,nm,stages),
        forward=zeros(ComplexF64,nw,nm,1), inverse=zeros(ComplexF64,nw,nm,c.low_storage ? 0 : 1),
        physical=physical, loss=zeros(ComplexF64,nw,nm),
        fine=fine, modal=modal, projected=zeros(T,nt,chunk),
        analytic=analytic, absorber=zeros(T,nt,chunk), absorber_spec=similar(fine),
        coarse=coarse, coarse_spec=coarse_spec, partials=zeros(3,nm),
        spatial_inverse=plan_ifft!(spatial,(2,3);flags=c.fft_flags,num_threads=c.transverse_fft_threads),
        spatial_forward=plan_fft!(spatial,(2,3);flags=c.fft_flags,num_threads=c.transverse_fft_threads),
        inverse_plan=c.envelope ? plan_ifft(fft_storage(fine),1;flags=c.fft_flags) : plan_irfft(fft_storage(fine),nt,1;flags=c.fft_flags),
        forward_plan=c.envelope ? plan_fft(fft_storage(modal),1;flags=c.fft_flags) : plan_rfft(fft_storage(modal),1;flags=c.fft_flags),
        analytic_forward=plan_fft!(fft_storage(analytic),1;flags=c.fft_flags),
        analytic_inverse=plan_ifft!(fft_storage(analytic),1;flags=c.fft_flags),
        coarse_inverse=c.envelope ? plan_ifft(fft_storage(coarse_spec),1;flags=c.fft_flags) : plan_irfft(fft_storage(coarse_spec),nc,1;flags=c.fft_flags),
        coarse_forward=c.envelope ? plan_fft(fft_storage(coarse),1;flags=c.fft_flags) : plan_rfft(fft_storage(coarse),1;flags=c.fft_flags),
    )
    Engine(c,w,false,0,NaN,c.factor)
end

# For an exactly real squared longitudinal wave number, the principal root is
# sqrt(x) for x>0 and i*sqrt(-x) for x<0. Preserve the imaginary signed zero and
# use the library complex root for lossy or nonfinite inputs. The compensated
# subtraction that forms q remains untouched; no near-zero cutoff is introduced.
@inline function free_root(q,specialize)
    x,y=real(q),imag(q)
    if specialize && iszero(y) && isfinite(x)
        if x > 0
            return complex(sqrt(x),y)
        elseif x < 0
            return complex(0.0,copysign(sqrt(-x),y))
        else
            return complex(0.0,y)
        end
    end
    sqrt(q)
end

# Preserve the compensated subtraction order in translate.spatial.longitudinal_root.
# No fast-math or reassociation: near grazing, the residual determines the branch.
@inline function free_element(c,f,m)
    @inbounds begin
        q = (c.k_squared[f]-c.kperp_sq[m]) + (c.k_squared_low[f]-c.kperp_sq_low[m])
        root = free_root(q,c.real_root)
        propagating = real(q) > 0
        attenuation = propagating ? imag(root) : min(abs(imag(root)),c.evanescent_rate)
        rate = c.active[f] ? complex(-attenuation-0.5*c.kperp_rate[m],-(real(root)-c.frame[f])) : 0.0im
        prefactor = propagating ? c.prefactor_numerator[f]/real(root) : c.prefactor_clamped[f]
        return rate-0.5*c.spectral_rate[f], c.active[f] ? prefactor : 0.0im
    end
end

function free_propagators!(engine,h)
    engine.propagator_dz == h && return
    c,w = engine.config,engine.work
    # The zero-distance stage is exactly identity; the final two DOPRI stages
    # share a distance and can reuse this one scratch pair without dense caches.
    if h == 0
        fill!(w.forward,1)
        fill!(w.inverse,1)
        engine.propagator_dz = h
        return
    end
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds begin
            rate,_ = free_element(c,f,m)
            exponent = h*rate
            # Both signs share exact trigonometric inputs, as in the compiled seam.
            scale = exp(real(exponent))
            sine,cosine = sincos(imag(exponent))
            w.forward[i] = scale*complex(cosine,sine)
            if !c.low_storage
                w.inverse[i] = complex(cosine,-sine)/scale
            end
        end
    end
    engine.propagator_dz = h
    nothing
end

# Large source-entry copies are bandwidth passes. Reuse the shared traversal
# and its task threshold without changing element order or owned storage.
function copy_volume!(destination,source,c)
    if c.parallel_copy
        foreach_sample(destination,c.tasks) do f,m,i
            @inbounds destination[i]=source[i]
        end
    else
        copyto!(destination,source)
    end
    nothing
end

function spatial_fft!(a,plan,c)
    shaped = view(reshape(a,size(a,1),c.spatial_shape...),:,:,:)
    fft_apply!(shaped,plan,shaped,c)
    nothing
end

function source_free!(engine,lab)
    c,w = engine.config,engine.work
    nt,chunk = size(w.modal)
    nw,nm = size(lab)
    copy_volume!(w.physical,lab,c)
    spatial_fft!(w.physical,w.spatial_inverse,c)
    # Transverse inversion precedes real temporal synthesis: individual transverse
    # spectral columns do not themselves represent real temporal fields.
    for first in 1:chunk:nm
        count = min(chunk,nm-first+1)
        fill!(w.fine,0)
        foreach_sample(@view(w.physical[:,first:first+count-1]),c.tasks) do f,m,i
            @inbounds w.fine[c.crop[f],m] = w.physical[f,first+m-1]
        end
        fft_apply!(w.modal,w.inverse_plan,w.fine,c)
        if !isnothing(w.plasma)
            # The temporal inverse is unscaled; plasma consumes SI nodal fields,
            # with the same padding normalization as the Kerr response below.
            foreach_sample(w.projected,c.tasks) do t,m,i
                @inbounds w.projected[i]=c.pad_scale*w.modal[i]
            end
            plasma_current!(w.plasma,@view(w.projected[:,1:count]),c,first)
        end
        if c.envelope
            foreach_sample(w.projected,c.tasks) do t,m,i
                @inbounds begin
                    value=c.pad_scale*w.modal[i]
                    w.projected[i] = c.factor*c.analytic_factor*abs2(value)*value*c.window[t]
                end
            end
        elseif c.analytic
            foreach_sample(w.analytic,c.tasks) do t,m,i
                @inbounds w.analytic[i] = c.pad_scale*w.modal[i]
            end
            w.analytic_forward*w.analytic
            foreach_sample(w.analytic,c.tasks) do f,m,i
                @inbounds w.analytic[i] *= (f == 1 || f == nt÷2+1) ? 1 : f <= nt÷2 ? 2 : 0
            end
            fft_apply!(w.analytic,w.analytic_inverse,w.analytic,c)
            polarize!(w.projected,w.analytic,c)
        else
            foreach_sample(w.projected,c.tasks) do t,m,i
                @inbounds w.projected[i] = polarization(c.pad_scale*w.modal[i],c.factor,c.analytic_factor)*c.window[t]
            end
        end
        mul!(w.fine,w.forward_plan,w.projected)
        # The held-current equivalent polarization combines before the spatial
        # forward FFT, exactly as the reference Cartesian source does.
        isnothing(w.plasma) || add_plasma!(w.fine,w.plasma,w.forward_plan,c;direct=true)
        if c.in_stage_absorber
            foreach_sample(w.absorber,c.tasks) do t,m,i
                @inbounds begin
                    rate = m <= count ? c.spatial_rate[first+m-1] : 0.0
                    rate += c.split ? 0.0 : c.rate[t]
                    w.absorber[i] = -0.5*rate*w.modal[i]
                end
            end
            mul!(w.absorber_spec,w.forward_plan,w.absorber)
        end
        foreach_sample(@view(w.physical[:,first:first+count-1]),c.tasks) do f,m,i
            @inbounds begin
                w.physical[f,first+m-1] = w.fine[c.crop[f],m]
                if c.in_stage_absorber
                    w.loss[f,first+m-1] = w.absorber_spec[c.crop[f],m]
                end
            end
        end
    end
    spatial_fft!(w.physical,w.spatial_forward,c)
    c.in_stage_absorber && spatial_fft!(w.loss,w.spatial_forward,c)
    foreach_sample(w.source,c.tasks) do f,m,i
        @inbounds begin
            _,prefactor = free_element(c,f,m)
            w.source[i] = prefactor*c.omega_window[f]*w.physical[i] + (c.in_stage_absorber && c.mask[f] ? w.loss[i] : 0.0im)
        end
    end
    engine.calls += 1
    nothing
end

function split_free!(engine,dz)
    c,w = engine.config,engine.work
    copy_volume!(w.physical,w.candidate,c)
    spatial_fft!(w.physical,w.spatial_inverse,c)
    chunk = size(w.coarse,2)
    for first in 1:chunk:size(w.candidate,2)
        count = min(chunk,size(w.candidate,2)-first+1)
        fill!(w.coarse_spec,0)
        @views copyto!(w.coarse_spec[:,1:count],w.physical[:,first:first+count-1])
        fft_apply!(w.coarse,w.coarse_inverse,w.coarse_spec,c)
        foreach_sample(w.coarse,c.tasks) do t,m,i
            @inbounds w.coarse[i] *= exp(-0.5*dz*c.coarse_rate[t])
        end
        mul!(w.coarse_spec,w.coarse_forward,w.coarse)
        @views copyto!(w.physical[:,first:first+count-1],w.coarse_spec[:,1:count])
    end
    spatial_fft!(w.physical,w.spatial_forward,c)
    copy_volume!(w.candidate,w.physical,c)
    nothing
end

# Reuse the accepted-field observation scan on bounded physical-space chunks.
function population_free_snapshot(engine,input)
    c,w=engine.config,engine.work
    p=w.plasma
    copyto!(w.physical,Matrix{ComplexF64}(input))
    spatial_fft!(w.physical,w.spatial_inverse,c)
    chunk=size(w.modal,2)
    parts=zeros(6,chunk)
    maxima=zeros(6)
    for first in 1:chunk:size(w.physical,2)
        count=min(chunk,size(w.physical,2)-first+1)
        fill!(w.fine,0)
        foreach_sample(@view(w.physical[:,first:first+count-1]),c.tasks) do f,m,i
            @inbounds w.fine[c.crop[f],m]=w.physical[f,first+m-1]
        end
        fft_apply!(w.modal,w.inverse_plan,w.fine,c)
        foreach_sample(w.projected,c.tasks) do t,m,i
            @inbounds w.projected[i]=c.pad_scale*w.modal[i]
        end
        nodes=@view(w.projected[:,1:count])
        AK.foreachindex(1:count;max_tasks=c.tasks,min_elems=1) do node
            population_column!(p,nodes,node,first,parts)
        end
        @inbounds for node in 1:count, i in 1:6
            maxima[i]=max(maxima[i],parts[i,node])
        end
    end
    (copy(p.fractions),copy(p.peaks),maxima)
end
