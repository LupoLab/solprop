# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# SEP 16's resident exponential PEC. The Python march supplies scalar polynomial
# tables and chooses steps/orders; state-sized arithmetic and history stay here.
const RECIP_FACTORIAL = ntuple(k -> 1.0/factorial(k-1), 10)
function phi_values(x,::Val{N}) where N
    if abs(x)>3.0
        out=ntuple(_->zero(ComplexF64),N+1)
        out=Base.setindex(out,exp(x),1)
        for k in 1:N
            out=Base.setindex(out,(out[k]-RECIP_FACTORIAL[k])/x,k+1)
        end
        return out
    end
    levels=abs(x)<=0.5 ? 0 : ceil(Int,log2(abs(x)/0.5))
    y=x/2.0^levels
    term=complex(RECIP_FACTORIAL[N+1])
    total=term
    for m in 1:23
        term=term*y/(m+N)
        total+=term
    end
    out=ntuple(_->zero(ComplexF64),N+1)
    out=Base.setindex(out,total,N+1)
    for k in N-1:-1:1
        out=Base.setindex(out,RECIP_FACTORIAL[k+1]+y*out[k+2],k+1)
    end
    out=Base.setindex(out,exp(y),1)
    for _ in 1:levels
        old=out
        out=Base.setindex(out,old[1]*old[1],1)
        for k in 1:N
            value=old[1]*old[k+1]
            for j in 1:k
                value+=old[j+1]*RECIP_FACTORIAL[k-j+1]
            end
            out=Base.setindex(out,value/2.0^k,k+1)
        end
    end
    out
end
phi_snapshot(x,order)=collect(phi_values(ComplexF64(x),Val(Int(order))))

function adams_workspace(input,nw,nm)
    order=haskey(input,"adams_order") ? Int(input["adams_order"]) : 0
    order==0 && return nothing
    slots=order+2
    width=order+1
    compensated=Bool(input["adams_compensated"])
    nphi=haskey(input,"phi_terms") ? size(Array{ComplexF64,3}(input["phi_terms"]),3) : 1
    (store=zeros(ComplexF64,nw,nm,slots),
     states=zeros(ComplexF64,nw,nm,compensated ? slots : 0),
     rates=zeros(ComplexF64,nw,nm,compensated ? slots : 0),
     moments=zeros(ComplexF64,nw,nm,width),
     carried=zeros(ComplexF64,nw,nm,compensated ? width-1 : 0),
     endpoint=zeros(ComplexF64,compensated ? nw : 0,compensated ? nm : 0),
     predictor=zeros(width,width),corrector=zeros(width,width),defects=zeros(3,2width),
     spans=zeros(width),cpredictor=zeros(width),ccorrector=zeros(width),cdefects=zeros(3,width),
     rate_weights=zeros(nphi),slots=zeros(Int,slots),sizes=zeros(Int,3),
     norms=zeros(3,nm),compensated=compensated)
end
function configure_adams!(engine,predictor,corrector,defects,slots,spans,cpredictor,ccorrector,cdefects,rate)
    a=engine.work.adams
    p,q,d=Matrix{Float64}(predictor),Matrix{Float64}(corrector),Matrix{Float64}(defects)
    order,width,count=size(p,1),size(q,1),size(d,1)
    a.sizes .= (order,width,count)
    a.predictor[1:order,1:order] .= p
    a.corrector[1:width,1:width] .= q
    a.defects[1:count,1:2width] .= d
    incoming=Vector{Int}(slots)
    a.slots[1:length(incoming)] .= incoming .+ 1
    if a.compensated
        s,cp,cc=Vector{Float64}(spans),Vector{Float64}(cpredictor),Vector{Float64}(ccorrector)
        a.spans[1:length(s)] .= s
        a.cpredictor[1:length(cp)] .= cp
        a.ccorrector[1:length(cc)] .= cc
        a.cdefects[1:count,1:width] .= Matrix{Float64}(cdefects)
        a.rate_weights .= Vector{Float64}(rate)
    end
    nothing
end
function configure_adams_rate!(engine,rate)
    engine.work.adams.rate_weights .= Vector{Float64}(rate)
    nothing
end
@inline function adams_rate(engine,f,m)
    p,a=engine.work.path,engine.work.adams
    isnothing(p) && return engine.config.linear[f,m]
    value=0.0im
    @inbounds for term in axes(p.phi,3)
        value+=p.phi[f,m,term]*a.rate_weights[term]
    end
    value
end
function adams_start!(engine)
    w=engine.work
    isnothing(w.path) || material_stage!(engine,1)
    source!(engine,w.y)
    copyto!(@view(w.adams.store[:,:,1]),w.source)
    if w.adams.compensated
        copyto!(@view(w.adams.states[:,:,1]),w.y)
        foreach_sample(w.y,engine.config.tasks) do f,m,i
            @inbounds w.adams.rates[f,m,1]=adams_rate(engine,f,m)
        end
    end
    nothing
end
function adams_trial!(engine,h)
    adams_trial_width!(engine,h,Val(engine.work.adams.sizes[2]))
end
function adams_trial_width!(engine,h,::Val{N}) where N
    c,w=engine.config,engine.work
    a=w.adams
    order,_,count=a.sizes
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds begin
            x=h*c.linear[i]
            if !isnothing(w.path)
                x=0.0im
                for term in axes(w.path.phi,3)
                    x+=w.path.phi[f,m,term]*w.path.delta[term,end]
                end
            end
            phi=phi_values(x,Val(N))
            w.forward[f,m,1]=phi[1]
            value=0.0im
            for k in 1:N
                moment=phi[k+1]/RECIP_FACTORIAL[k]
                a.moments[f,m,k]=moment
                if k<=order
                    weighted=0.0im
                    for j in 1:order
                        weighted+=a.predictor[j,k]*a.store[f,m,a.slots[j]]
                    end
                    value+=moment*weighted
                end
            end
            if a.compensated
                correction=0.0im
                for j in 1:N-1
                    slot=a.slots[j]
                    carried=exp(a.spans[j]*x)*(a.rates[f,m,slot]-x/h)*a.states[f,m,slot]
                    a.carried[f,m,j]=carried
                    j<=order && (correction+=a.cpredictor[j]*carried)
                end
                value+=correction
                w.error[i]=adams_rate(engine,f,m)
            end
            predicted=phi[1]*w.y[i]+h*value
            w.lab[i]=predicted
            if a.compensated
                a.endpoint[i]=(w.error[i]-x/h)*predicted
            end
        end
    end
    isnothing(w.path) || material_stage!(engine,length(c.c))
    source!(engine,w.lab;quadrature=true)
    magnitude=sqrt(masked_squared(engine,w.y))
    AK.foreachindex(axes(w.y,2);max_tasks=c.tasks,min_elems=max(1,cld(32768,size(w.y,1)))) do m
        @inbounds begin
            for row in 1:count
                a.norms[row,m]=0.0
            end
            for f in axes(w.y,1)
                corrected=0.0im
                for k in 1:N
                    weighted=a.corrector[1,k]*w.source[f,m]
                    for j in 2:N
                        weighted+=a.corrector[j,k]*a.store[f,m,a.slots[j-1]]
                    end
                    corrected+=a.moments[f,m,k]*weighted
                end
                if a.compensated
                    correction=a.ccorrector[1]*a.endpoint[f,m]
                    for j in 1:order
                        correction+=a.ccorrector[j+1]*a.carried[f,m,j]
                    end
                    corrected+=correction
                end
                w.candidate[f,m]=w.forward[f,m,1]*w.y[f,m]+h*corrected
                if c.norm_mask[f]
                    scale=c.rtol*c.weights[f]*magnitude+c.atol
                    for row in 1:count
                        difference=a.defects[row,1]*w.source[f,m]
                        polynomial=0.0im
                        for j in 2:N
                            difference+=a.defects[row,j]*a.store[f,m,a.slots[j-1]]
                        end
                        for k in 1:N
                            polynomial+=a.defects[row,N+k]*a.moments[f,m,k]
                        end
                        defect=difference*polynomial
                        if a.compensated
                            correction=a.cdefects[row,1]*a.endpoint[f,m]
                            for j in 1:N-1
                                correction+=a.cdefects[row,j+1]*a.carried[f,m,j]
                            end
                            defect+=correction
                        end
                        defect*=h
                        a.norms[row,m]+=scale==0 && defect==0 ? 0.0 : abs2(defect/scale)
                    end
                end
            end
        end
    end
    [sqrt(sum(@view a.norms[row,:])) for row in 1:count]
end
function adams_accept!(engine,slot)
    w=engine.work
    copyto!(@view(w.adams.store[:,:,slot+1]),w.source)
    if w.adams.compensated
        copyto!(@view(w.adams.states[:,:,slot+1]),w.lab)
        copyto!(@view(w.adams.rates[:,:,slot+1]),w.error)
    end
    nothing
end
adams_rows(engine,name)=copy(getproperty(engine.work.adams,Symbol(name)))
function adams_load_rows!(engine,name,rows)
    copyto!(getproperty(engine.work.adams,Symbol(name)),Array{ComplexF64,3}(rows))
    nothing
end
const configure_adams_b=configure_adams!
const configure_adams_rate_b=configure_adams_rate!
const adams_start_b=adams_start!
const adams_trial_b=adams_trial!
const adams_accept_b=adams_accept!
const adams_load_rows_b=adams_load_rows!
