# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# The four register passes of SEP 24's RK4IP. A constant flow's key is the schedule
# run's mean step, from the shared table_steps policy. Each pass also forms the norms
# the march reads: the state's, and the stiffness estimate's two slope differences.

function rk4!(engine, dz::Float64, key::Float64)
    c, w = engine.config, engine.work
    T = real_type(engine)
    (; y, lab, candidate, error, source, forward) = w
    mask = c.norm_mask
    h, half, third, sixth = T(dz), T(dz / 2), T(dz / 3), T(dz / 6)
    c.free ? free_propagators!(engine, key / 2) : modal_flow!(engine, key / 2)
    none = zero(T)
    source!(engine, y)
    engine.version += 1
    each!(c, size(y)...) do f, m
        @inbounds begin
            flow = forward[f, m, 1]
            y[f, m] *= flow
            slope = flow * source[f, m]
            error[f, m] = slope
            candidate[f, m] = muladd(sixth, slope, y[f, m])
            lab[f, m] = muladd(half, slope, y[f, m])
        end
    end
    source!(engine, lab)
    lower = reduce_each(+, none, c, size(y)...) do f, m
        @inbounds begin
            slope = source[f, m]
            difference = slope - error[f, m]
            # The second slope is the third's reference: kept where the first was.
            error[f, m] = slope
            candidate[f, m] = muladd(third, slope, candidate[f, m])
            lab[f, m] = muladd(half, slope, y[f, m])
            mask[f] ? abs2(difference) : none
        end
    end
    source!(engine, lab)
    upper = reduce_each(+, none, c, size(y)...) do f, m
        @inbounds begin
            slope = source[f, m]
            difference = slope - error[f, m]
            error[f, m] = difference
            candidate[f, m] = muladd(third, slope, candidate[f, m])
            lab[f, m] = forward[f, m, 1] * muladd(h, slope, y[f, m])
            mask[f] ? abs2(difference) : none
        end
    end
    source!(engine, lab)
    squared = reduce_each(+, none, c, size(y)...) do f, m
        @inbounds begin
            new = muladd(sixth, source[f, m], forward[f, m, 1] * candidate[f, m])
            candidate[f, m] = new
            mask[f] ? abs2(new) : none
        end
    end
    (Float64(squared), lower > 0 ? 2sqrt(Float64(upper) / Float64(lower)) : NaN)
end

"""A modal route's RK4IP flow over `h`, `exp(h L)`, in slot one (formed in the wide type
and rounded once), kept while `h` is unchanged."""
function modal_flow!(engine, h)
    engine.propagator_dz == h && return 1
    c, w = engine.config, engine.work
    linear, forward = c.linear, w.forward
    step = wide_value(c, Float64(h))
    each!(c, size(linear)...) do f, m
        @inbounds forward[f, m, 1] = first(wide_exp_pair(step * linear[f, m]))
    end
    engine.propagator_dz = h
    1
end
