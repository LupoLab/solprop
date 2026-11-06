# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The four register passes of the existing SEP 24 scheme. A constant flow's key
# is the schedule run's mean step, supplied by the shared table_steps policy.
function masked_squared(engine,a)
    c,w=engine.config,engine.work
    AK.foreachindex(axes(a,2);max_tasks=c.tasks,min_elems=1) do m
        value=0.0
        @inbounds for f in axes(a,1)
            if c.norm_mask[f]
                value += abs2(a[f,m])
            end
        end
        w.partials[1,m]=value
    end
    sum(@view w.partials[1,:])
end
function rk4!(engine,dz::Float64,key::Float64)
    c,w=engine.config,engine.work
    free_propagators!(engine,key/2)
    source!(engine,w.y)
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds begin
            w.y[i] *= w.forward[i]
            slope=w.forward[i]*w.source[i]
            w.error[i]=slope
            w.candidate[i]=w.y[i]+(dz/6)*slope
            w.lab[i]=w.y[i]+(dz/2)*slope
        end
    end
    source!(engine,w.lab)
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds begin
            w.error[i]=w.source[i]-w.error[i]
            w.candidate[i] += (dz/3)*w.source[i]
            w.lab[i]=w.y[i]+(dz/2)*w.source[i]
        end
    end
    lower=masked_squared(engine,w.error)
    copy_volume!(w.error,w.source,c)
    source!(engine,w.lab)
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds begin
            w.error[i]=w.source[i]-w.error[i]
            w.candidate[i] += (dz/3)*w.source[i]
            w.lab[i]=w.forward[i]*(w.y[i]+dz*w.source[i])
        end
    end
    upper=masked_squared(engine,w.error)
    source!(engine,w.lab)
    foreach_sample(w.y,c.tasks) do f,m,i
        @inbounds w.candidate[i]=w.forward[i]*w.candidate[i]+(dz/6)*w.source[i]
    end
    (masked_squared(engine,w.candidate),lower > 0 ? 2sqrt(upper/lower) : NaN)
end
