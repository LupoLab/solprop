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
    free_propagators!(engine, key / 2)
    source!(engine, y)
    each!(c, size(y)...) do f, m
        @inbounds begin
            flow = forward[f, m, 1]
            y[f, m] *= flow
            slope = flow * source[f, m]
            error[f, m] = slope
            candidate[f, m] = y[f, m] + sixth * slope
            lab[f, m] = y[f, m] + half * slope
        end
    end
    source!(engine, lab)
    lower = reduce_each(+, zero(T), c, size(y)...) do f, m
        @inbounds begin
            slope = source[f, m]
            difference = slope - error[f, m]
            # The second slope is the third's reference: kept where the first was.
            error[f, m] = slope
            candidate[f, m] += third * slope
            lab[f, m] = y[f, m] + half * slope
            mask[f] ? abs2(difference) : zero(T)
        end
    end
    source!(engine, lab)
    upper = reduce_each(+, zero(T), c, size(y)...) do f, m
        @inbounds begin
            slope = source[f, m]
            difference = slope - error[f, m]
            error[f, m] = difference
            candidate[f, m] += third * slope
            lab[f, m] = forward[f, m, 1] * (y[f, m] + h * slope)
            mask[f] ? abs2(difference) : zero(T)
        end
    end
    source!(engine, lab)
    squared = reduce_each(+, zero(T), c, size(y)...) do f, m
        @inbounds begin
            new = forward[f, m, 1] * candidate[f, m] + sixth * source[f, m]
            candidate[f, m] = new
            mask[f] ? abs2(new) : zero(T)
        end
    end
    (Float64(squared), lower > 0 ? 2sqrt(Float64(upper) / Float64(lower)) : NaN)
end
