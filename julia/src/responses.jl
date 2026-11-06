# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Numerical execution of the translated scalar response. Rate tables, constants,
# FIR taps and temporal collar derivatives all come from schema 34 (SEP 18/27).
function plasma_workspace(input, c, nt, chunk, nm, nn)
    (haskey(input,"plasma") && Bool(input["plasma"])) || return nothing
    # Cartesian columns already are physical nodes. Only their current chunk
    # needs temporal current spectra; no modal projection buffers are required.
    projected_rows, projected_columns = c.free ? (0,0) : (nt,nm)
    (
        coefficients=Matrix{Float64}(input["plasma_coefficients"]),
        parameters=Vector{Float64}(input["plasma_parameters"]),
        taps=Vector{Float64}(input["plasma_taps"]), floor=Bool(input["plasma_floor"]),
        dt=Float64(input["dt"]), omega=Vector{Float64}(input["omega"]),
        derivative=Vector{Float64}(input["window_derivative"]),
        rate=zeros(nt,chunk), fraction=zeros(nt,chunk), integrand=zeros(nt,chunk),
        current=zeros(nt,chunk), gated=zeros(nt,chunk), edge=zeros(nt,chunk),
        projected=zeros(projected_rows,projected_columns), edge_projected=zeros(projected_rows,projected_columns),
        coarse=zeros(projected_rows,projected_columns), coarse_edge=zeros(projected_rows,projected_columns),
        spectrum=zeros(ComplexF64,nt÷2+1,nm), edge_spectrum=zeros(ComplexF64,nt÷2+1,nm),
        fractions=zeros(nn), peaks=zeros(nn),
    )
end

@inline function rate_value(value, p)
    x=abs(value)
    x < p.parameters[1] && return 0.0
    x >= p.parameters[2] && return p.parameters[3]
    logarithm=log(x)
    delta=logarithm-p.parameters[4]
    j=clamp(floor(Int,delta/p.parameters[5])+1,1,size(p.coefficients,2))
    t=logarithm-(p.parameters[4]+(j-1)*p.parameters[5])
    @inbounds value=((p.coefficients[1,j]*t+p.coefficients[2,j])*t+p.coefficients[3,j])*t+p.coefficients[4,j]
    min(exp(value),p.parameters[3])
end

# A symmetric FIR followed by a running trapezoid. Point reflection preserves
# the endpoint convention of RunningIntegral; no reassociation of this scan.
@inline function filtered(values,t,node,taps)
    nt=size(values,1)
    @inbounds value=taps[1]*values[t,node]
    @inbounds for k in 1:length(taps)-1
        left=t-k < 1 ? 2*values[1,node]-values[1+k-t+1,node] : values[t-k,node]
        right=t+k > nt ? 2*values[nt,node]-values[2nt-t-k,node] : values[t+k,node]
        value += taps[k+1]*(left+right)
    end
    value
end
function integrate_column!(out,values,node,p)
    previous=filtered(values,1,node,p.taps)
    total=0.0
    @inbounds out[1,node]=0.0
    @inbounds for t in 2:size(values,1)
        current=filtered(values,t,node,p.taps)
        total += p.dt*(current+previous)/2
        out[t,node]=total
        previous=current
    end
end

# On CPU, filter independent time samples before the dependent prefix. Keeping
# the tap loop outside the interior time loop exposes SIMD without reordering
# a sample's tap additions. The output itself is scratch until the strict scan,
# so this traversal does not add a persistent buffer.
# Exact zero support saves filter and prefix work outside the live interval.
# Reflection cannot extend this interval beyond the filter radius: its endpoint
# is already live whenever a reflected endpoint value is nonzero.
function integrate_column!(out::StridedMatrix,values::StridedMatrix,node,p)
    nt=size(values,1)
    half=length(p.taps)-1
    first=1
    @inbounds while first <= nt && iszero(values[first,node])
        first += 1
    end
    fill!(@view(out[:,node]),0)
    first > nt && return nothing
    last=nt
    @inbounds while iszero(values[last,node])
        last -= 1
    end
    first=max(1,first-half)
    last=min(nt,last+half)
    if 2half >= nt
        @inbounds for t in first:last
            out[t,node]=filtered(values,t,node,p.taps)
        end
    else
        @inbounds for t in first:min(half,last)
            out[t,node]=filtered(values,t,node,p.taps)
        end
        @inbounds for t in max(nt-half+1,first):last
            out[t,node]=filtered(values,t,node,p.taps)
        end
        interior_first=max(first,half+1)
        interior_last=min(last,nt-half)
        @inbounds @simd for t in interior_first:interior_last
            out[t,node]=p.taps[1]*values[t,node]
        end
        @inbounds for k in 1:half
            tap=p.taps[k+1]
            @simd for t in interior_first:interior_last
                out[t,node] += tap*(values[t-k,node]+values[t+k,node])
            end
        end
    end
    @inbounds previous=out[1,node]
    total=0.0
    @inbounds out[1,node]=0.0
    stop=min(nt,last+1)
    @inbounds for t in max(2,first):stop
        current=out[t,node]
        total += p.dt*(current+previous)/2
        out[t,node]=total
        previous=current
    end
    @inbounds for t in stop+1:nt
        out[t,node]=total
    end
    nothing
end

function plasma_current!(p,nodes,c,first)
    AK.foreachindex(axes(nodes,2);max_tasks=c.tasks,min_elems=1) do node
        peak=exposure_column!(p,nodes,node)
        # With exactly zero rates and no initial electrons, both current streams
        # vanish. Clear their outputs even if this chunk ionised on a prior call.
        # The unused integrand/current scratch is overwritten on every live call.
        if c.skip_zero_plasma && iszero(peak) && iszero(p.parameters[9])
            fill!(@view(p.gated[:,node]),0)
            fill!(@view(p.edge[:,node]),0)
            @inbounds p.fractions[first+node-1]=0.0
            @inbounds p.peaks[first+node-1]=0.0
            return
        end
        previous_exposure=NaN
        fraction=0.0
        @inbounds for t in axes(nodes,1)
            exposure=p.floor ? max(p.fraction[t,node],0.0) : p.fraction[t,node]
            # An unchanged prefix gives the same fraction, including the exact
            # zero-rate head and constant tail. No threshold or approximation.
            if exposure != previous_exposure
                fraction=1-(1-p.parameters[9])*exp(-exposure)
                previous_exposure=exposure
            end
            p.fraction[t,node]=fraction
            p.integrand[t,node]=fraction*p.parameters[6]*nodes[t,node]
        end
        integrate_column!(p.current,p.integrand,node,p)
        @inbounds for t in axes(nodes,1)
            field=nodes[t,node]
            squared=field*field
            loss=p.rate[t,node] > 0 && squared > 0 ? (p.parameters[7]*p.rate[t,node]*(1-p.fraction[t,node])/squared)*field : 0.0
            current=p.parameters[8]*(p.current[t,node]+loss)
            p.gated[t,node]=c.window[t]*current
            p.edge[t,node]=p.derivative[t]*current
        end
        @inbounds p.fractions[first+node-1]=p.fraction[size(nodes,1),node]
        @inbounds p.peaks[first+node-1]=peak
    end
end
function clear_plasma!(p)
    isnothing(p) && return
    for a in (p.projected,p.edge_projected,p.coarse,p.coarse_edge)
        fill!(a,0)
    end
end
function project_plasma!(p,c,first,last,quadrature)
    count=last-first+1
    for (values,full,coarse) in ((p.gated,p.projected,p.coarse),(p.edge,p.edge_projected,p.coarse_edge))
        mul!(full,@view(values[:,1:count]),@view(c.projection[first:last,:]),1.0,1.0)
        if quadrature && c.embedded
            mul!(coarse,@view(values[:,1:count]),@view(c.projection_coarse[first:last,:]),1.0,1.0)
        end
    end
end
function add_plasma!(destination,p,plan,c;coarse=false,direct=false)
    mul!(p.spectrum,plan,direct ? p.gated : coarse ? p.coarse : p.projected)
    mul!(p.edge_spectrum,plan,direct ? p.edge : coarse ? p.coarse_edge : p.edge_projected)
    foreach_sample(@view(destination[1:length(c.crop),:]),c.tasks) do f,m,i
        @inbounds if c.mask[f]
            k=c.crop[f]
            inverse=1/p.omega[f]
            destination[k,m] += -im*inverse*p.spectrum[k,m]+inverse^2*p.edge_spectrum[k,m]
        end
    end
end
# The source and observation share the translated rate and strict exposure scan.
function exposure_column!(p,nodes,node)
    peak=0.0
    @inbounds for t in axes(nodes,1)
        rate=rate_value(nodes[t,node],p)
        p.rate[t,node]=rate
        peak=max(peak,rate)
    end
    integrate_column!(p.fraction,p.rate,node,p)
    peak
end

# Observation scans read the accepted field afresh. They need rates and exposure,
# not the propagated current, Kerr/Raman response, projection or source FFTs.
function population_column!(p,nodes,node,first,parts)
    nt=size(nodes,1)
    peak=exposure_column!(p,nodes,node)
    dip=rise=paid=early=maximum_integrand=0.0
    previous_exposure=0.0
    previous_product=0.0
    initial_fraction=0.0
    reach=2(length(p.taps)-1)
    @inbounds for t in 1:nt
        exposure=p.fraction[t,node]
        if t > 1
            increment=exposure-previous_exposure
            dip=max(dip,-increment)
            rise=max(rise,increment)
        end
        previous_exposure=exposure
        fraction=1-(1-p.parameters[9])*exp(-(p.floor ? max(exposure,0.0) : exposure))
        p.fraction[t,node]=fraction
        product=p.rate[t,node]*(1-fraction)
        if t == 1
            initial_fraction=fraction
        else
            # Keep this diagnostic trapezoid ordered and strict, like the prefix.
            paid += p.dt*(product+previous_product)/2
        end
        previous_product=product
        integrand=abs(fraction*nodes[t,node])
        maximum_integrand=max(maximum_integrand,integrand)
        if t <= reach
            early=max(early,integrand)
        end
    end
    @inbounds begin
        freed=p.fraction[nt,node]-initial_fraction
        p.fractions[first+node-1]=p.fraction[nt,node]
        p.peaks[first+node-1]=peak
        parts[1,node]=dip
        parts[2,node]=rise
        parts[3,node]=abs(paid-freed)
        parts[4,node]=freed
        parts[5,node]=early
        parts[6,node]=maximum_integrand
    end
    nothing
end
function population_snapshot(engine,input)
    engine.config.free && return population_free_snapshot(engine,input)
    c,w=engine.config,engine.work
    p=w.plasma
    lab=Matrix{ComplexF64}(input)
    fill!(w.fine,0)
    foreach_sample(lab,c.tasks) do f,m,i
        @inbounds w.fine[c.crop[f],m]=lab[i]
    end
    fft_apply!(w.modal,w.inverse_plan,w.fine,c)
    chunk=size(w.nodes,2)
    # Observation-only transient reductions are outside the owned march buffers.
    parts=zeros(6,chunk)
    maxima=zeros(6)
    for first in 1:chunk:size(c.synthesis,2)
        last=min(first+chunk-1,size(c.synthesis,2))
        count=last-first+1
        nodes=@view w.nodes[:,1:count]
        mul!(nodes,w.modal,@view(c.synthesis[:,first:last]))
        AK.foreachindex(1:count;max_tasks=c.tasks,min_elems=1) do node
            population_column!(p,nodes,node,first,parts)
        end
        @inbounds for node in 1:count, i in 1:6
            maxima[i]=max(maxima[i],parts[i,node])
        end
    end
    (copy(p.fractions),copy(p.peaks),maxima)
end

function raman_workspace(input,c,nt,chunk)
    (haskey(input,"raman") && Bool(input["raman"])) || return nothing
    rows=Matrix{Float64}(input["raman_undamped"])
    rates=Vector{Float64}(input["raman_rates"])
    dt=Float64(input["dt"])
    kernel=zeros(2nt,1)
    for term in eachindex(rates), t in 1:2nt
        @inbounds kernel[t] += rows[t,term]*exp(-rates[term]*(t-1)*dt)
    end
    kernel .*= Float64(input["raman_fraction"])
    plan=plan_rfft(fft_storage(kernel),1;flags=c.fft_flags,num_threads=1)
    # MEASURE may overwrite its planning input.
    fill!(kernel,0)
    for term in eachindex(rates), t in 1:2nt
        @inbounds kernel[t] += rows[t,term]*exp(-rates[term]*(t-1)*dt)
    end
    kernel .*= Float64(input["raman_fraction"])
    spectrum=vec(plan*kernel).*dt
    drive=zeros(2nt,chunk)
    frequency=zeros(ComplexF64,nt+1,chunk)
    (kernel=spectrum,factor=Float64(input["raman_factor"]),thg=Bool(input["raman_thg"]),
     drive=drive,frequency=frequency,result=zeros(c.envelope ? ComplexF64 : Float64,nt,chunk),
     forward=plan_rfft(fft_storage(drive),1;flags=c.fft_flags,num_threads=c.raman_fft_threads),
     inverse=plan_irfft(fft_storage(frequency),2nt,1;flags=c.fft_flags,num_threads=c.raman_fft_threads))
end
function raman_response!(r,nodes,analytic,c)
    nt,count=size(nodes)
    fill!(r.drive,0)
    foreach_sample(nodes,c.tasks) do t,m,i
        @inbounds r.drive[t,m]=r.thg ? abs2(nodes[i]) : 0.5*abs2(analytic[t,m])
    end
    mul!(r.frequency,r.forward,r.drive)
    foreach_sample(r.frequency,c.tasks) do f,m,i
        @inbounds r.frequency[i] *= r.kernel[f]
    end
    fft_apply!(r.drive,r.inverse,r.frequency,c)
    foreach_sample(nodes,c.tasks) do t,m,i
        @inbounds r.result[t,m]=r.factor*nodes[i]*r.drive[t,m]
    end
end
function add_raman!(r,nodes,c)
    foreach_sample(nodes,c.tasks) do t,m,i
        @inbounds nodes[i] += r.result[t,m]*c.window[t]
    end
end
