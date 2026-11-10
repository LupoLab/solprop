# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

function adams_control_policy(; kwargs...)
    # Synthetic policy readings test control algebra, not stability of a method.
    fields = (
        order = 5, variable_order = true, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = true, interval_cap = false, dz_min = 1.0e-8, dz_max = 0.3,
        stretch = 1.1, z_end = 1.0, save_planes = [0.25, 0.5, 0.75, 1.0],
        hard_kinks = [0.4], breakpoints = [0.0, 1.0], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
    return SP.AdamsControlPolicy(; merge(fields, (; kwargs...))...)
end

function adams_control_cycle!(policy, stencil, positions, slots, orders, norms)
    SP.adams_select_stencil!(
        stencil, policy, positions, slots, length(slots), 3;
        z = last(positions), dz = 0.01, last_rejected = false,
    )
    SP.adams_factor(policy, 0.3, 3; growth = policy.max_growth)
    SP.adams_stable_factor(policy, 3, 0.01; quiet = false)
    SP.adams_order_weight(policy, 0.3, 3, 0.01; quiet = false)
    SP.adams_choose_order(policy, 3, orders, norms, 3, 0.01; quiet = false)
    SP.adams_rejection(policy, 1.2, 3; rejections = 2, held = 6)
    SP.adams_quiet_cap(policy, 0.03; quiet = true)
    SP.adams_collar_reentry(policy, 0.03, 3; was_quiet = true, quiet = false)
    stop = SP.adams_landing(policy, 0.1, 0.08, 0.5)
    SP.adams_next_landing(policy, 0.1, 0.5)
    SP.adams_approach(policy, 0.1, stop, 0.5)
    SP.adams_replay_kink(policy, 0.4, 8eps(1.0))
    return nothing
end

function adams_control_allocations(policy, stencil, positions, slots, orders, norms)
    adams_control_cycle!(policy, stencil, positions, slots, orders, norms)
    adams_control_cycle!(policy, stencil, positions, slots, orders, norms)
    return @allocated adams_control_cycle!(policy, stencil, positions, slots, orders, norms)
end

@testset "Adams policy owns only translated values" begin
    policy = adams_control_policy()
    @test policy.order == 5 && policy.max_calls === nothing && !policy.carrier
    @test isbitstype(SP.AdamsOrderChoice)
    @test isbitstype(SP.AdamsRejection)
    @test isbitstype(SP.AdamsLanding)
    @test isbitstype(SP.AdamsApproach)
    planes, kinks, edges = [0.5, 1.0], [0.4], [0.0, 1.0]
    owned = adams_control_policy(;
        save_planes = planes, hard_kinks = kinks, breakpoints = edges
    )
    planes[1], kinks[1], edges[1] = 0.7, 0.8, -1.0
    @test owned.save_planes == [0.5, 1.0]
    @test owned.hard_kinks == [0.4]
    @test owned.breakpoints == [0.0, 1.0]
    for invalid in (
            (; order = 0), (; order = 7), (; safety = 0.0), (; safety = NaN),
            (; max_growth = 1.0), (; min_factor = 1.0), (; rejection_growth = 1.1),
            (; hysteresis = 0.9), (; restart_after = 0), (; collar_stability = 0.0),
            (; collar_rate = -1.0), (; carrier_cap = Inf), (; max_calls = 0),
            (; dz_min = -1.0), (; dz_max = Inf), (; stretch = 0.9), (; z_end = NaN),
            (; landing_ulps = 0), (; quiet_interval = 0.0), (; carrier = true),
            (; real_intervals = (1.0,)), (; breakpoints = [0.0]),
            (; hard_kinks = [0.4, 0.4]), (; save_planes = [1.1]),
            (; interpolate = true),
        )
        @test_throws ArgumentError adams_control_policy(; invalid...)
    end
    @test adams_control_policy(; interpolate = true, save_planes = Float64[]).interpolate
end

@testset "Adams control selects numerical prefixes without owning history" begin
    for maximum in 1:6, variable in (false, true), held in 1:(maximum + 1)
        policy = adams_control_policy(; order = maximum, variable_order = variable)
        positions = [1.0 + 0.1index for index in 1:held]
        slots = collect(1:held)
        original = (copy(positions), copy(slots))
        stencil = SP.AdamsStencil(maximum)
        for stored in 1:maximum, rejected in (false, true)
            SP.adams_select_stencil!(
                stencil, policy, positions, slots, held, stored;
                z = last(positions), dz = 0.1, last_rejected = rejected,
            )
            executed = min(stored, held)
            rising = !rejected && held > executed
            scored = [executed]
            if variable
                for candidate in (executed - 1, executed + 1)
                    if 1 <= candidate <= maximum && candidate <= held &&
                            (candidate < executed || rising)
                        push!(scored, candidate)
                    end
                end
            end
            count = min(held, maximum)
            @test stencil.order == executed && stencil.count == count
            @test stencil.slots[1:count] == reverse(slots)[1:count]
            @test stencil.nodes[1:count] ==
                ((reverse(positions) .- last(positions)) ./ 0.1)[1:count]
            @test stencil.orders[1:stencil.n_orders] == scored
            @test all(iszero, stencil.slots[(count + 1):end])
            @test all(iszero, stencil.nodes[(count + 1):end])
            @test all(iszero, stencil.orders[(length(scored) + 1):end])
            @test positions == original[1] && slots == original[2]
        end
    end
    policy = adams_control_policy(; order = 3)
    stencil = SP.AdamsStencil(3)
    positions = [0.8, 0.9, nextfloat(1.0)]
    slots = [3, 1, 2]
    SP.adams_select_stencil!(
        stencil, policy, positions, slots, 3, 2; z = 1.0, dz = 0.1, last_rejected = false
    )
    @test stencil.nodes[1] == (nextfloat(1.0) - 1.0) / 0.1
    @test stencil.nodes[1] > 0
    saved = (copy(stencil.slots), copy(stencil.orders), copy(stencil.nodes))
    for (bad_positions, bad_slots, count, stored, anchor, step) in (
            (positions, slots, 0, 2, 1.0, 0.1),
            (positions, slots, 3, 0, 1.0, 0.1),
            ([0.9, 0.8, 1.0], slots, 3, 2, 1.0, 0.1),
            (positions, [1, 1, 2], 3, 2, 1.0, 0.1),
            (positions, [1, 0, 2], 3, 2, 1.0, 0.1),
            (positions, slots, 3, 2, NaN, 0.1),
            (positions, slots, 3, 2, 1.0, 0.0),
            (Float32.(positions), slots, 3, 2, 1.0, 0.1),
            (positions, UInt.(slots), 3, 2, 1.0, 0.1),
            (stencil.nodes, slots, 3, 2, 1.0, 0.1),
            (positions, view(stencil.slots, :), 3, 2, 1.0, 0.1),
        )
        @test_throws ArgumentError SP.adams_select_stencil!(
            stencil, policy, bad_positions, bad_slots, count, stored;
            z = anchor, dz = step, last_rejected = false,
        )
        @test (stencil.slots, stencil.orders, stencil.nodes) == saved
    end
end

@testset "Adams scalar decision ties and nonfinite readings" begin
    policy = adams_control_policy()
    for norm in (0.0, -1.0, -Inf), order in 1:5
        @test SP.adams_factor(policy, norm, order; growth = 1.7) == 1.7
    end
    for norm in (NaN, Inf), order in 1:5
        @test SP.adams_factor(policy, norm, order; growth = 1.7) == policy.min_factor
    end
    @test SP.adams_factor(policy, 1.0, 3; growth = 1.7) == policy.safety
    @test isinf(SP.adams_stable_factor(policy, 2, 0.01; quiet = false))
    collar = adams_control_policy(; collar_rate = 10.0)
    @test SP.adams_stable_factor(collar, 2, 0.01; quiet = false) == 0.7 / (10.0 * 0.01)
    @test isinf(SP.adams_stable_factor(collar, 2, 0.01; quiet = true))
    @test isnan(SP.adams_order_weight(policy, NaN, 2, 0.01; quiet = false))
    orders = [3, 2, 4]
    for norms in ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [NaN, 0.0, 0.0])
        choice = SP.adams_choose_order(policy, 3, orders, norms, 3, 0.01; quiet = false)
        @test choice.stored_order == choice.control_order == 3
        @test isequal(choice.norm, norms[1])
    end
    choice = SP.adams_choose_order(
        policy, 3, orders, [1.0, NaN, 0.0], 3, 0.01; quiet = false
    )
    @test choice.stored_order == choice.control_order == 4 && iszero(choice.norm)
    fixed = adams_control_policy(; variable_order = false)
    choice = SP.adams_choose_order(fixed, 1, [1], [0.3], 1, 0.01; quiet = false)
    @test choice.stored_order == 5 && choice.control_order == 1 && choice.norm == 0.3
    for norm in (NaN, Inf, -Inf), held in (1, 6)
        rejection = SP.adams_rejection(policy, norm, 3; rejections = 2, held)
        @test rejection.factor == policy.min_factor
        @test rejection.restart == (held > 1)
        @test rejection.rejections == (held > 1 ? 0 : 3)
    end
    rejection = SP.adams_rejection(policy, 1.0, 2; rejections = 0, held = 3)
    @test rejection.factor == policy.safety &&
        rejection.rejections == 1 && !rejection.restart
    @test_throws ArgumentError SP.adams_choose_order(
        policy, 3, [2, 3], [1.0, 1.0], 2, 0.01; quiet = false
    )
    @test_throws ArgumentError SP.adams_choose_order(
        policy, 3, [3, 3], [1.0, 1.0], 2, 0.01; quiet = false
    )
    threshold = adams_control_policy(; hysteresis = 2.0)
    # Both roots are exact powers of two; equality at hysteresis does not switch.
    retained = SP.adams_choose_order(
        threshold, 2, [2, 3], [1.0, 1 / 16], 2, 0.01; quiet = false
    )
    switched = SP.adams_choose_order(
        threshold, 2, [2, 3], [1.0, 1 / 256], 2, 0.01; quiet = false
    )
    @test retained.stored_order == 2 && switched.stored_order == 3
end

@testset "Adams decision admission errors" begin
    policy = adams_control_policy()
    for growth in (0.0, Inf), order in (0, 3)
        @test_throws ArgumentError SP.adams_factor(policy, 1.0, order; growth)
    end
    for order in (0, 6), step in (0.0, 0.01)
        @test_throws ArgumentError SP.adams_stable_factor(
            policy, order, step; quiet = false
        )
        @test_throws ArgumentError SP.adams_order_weight(
            policy, 1.0, order, step; quiet = false
        )
    end
    for (rejections, held) in ((-1, 2), (0, 0), (typemax(Int), 2))
        @test_throws ArgumentError SP.adams_rejection(
            policy, 1.0, 2; rejections, held
        )
    end
    @test_throws DimensionMismatch SP.adams_select_stencil!(
        SP.AdamsStencil(2), policy, [0.0], [1], 1, 1;
        z = 0.0, dz = 0.01, last_rejected = false,
    )
    @test_throws ArgumentError SP.adams_choose_order(
        policy, 3, [3], [1.0], 0, 0.01; quiet = false
    )
    @test_throws ArgumentError SP.adams_collar_reentry(
        policy, 0.01, 0; was_quiet = false, quiet = false
    )
    @test_throws ArgumentError SP.adams_next_landing(policy, NaN, 1.0)
    @test_throws ArgumentError SP.adams_landing(policy, 0.0, 0.01, 2.0)
    @test_throws ArgumentError SP.adams_replay_kink(policy, 0.4, -1.0)
end

@testset "Adams caps, equal approach and coincident kink landings" begin
    policy = adams_control_policy()
    stop = SP.adams_landing(policy, 0.0, 1.0, 1.0)
    @test stop.dz == 0.25 && stop.z_target == 0.25 && stop.kind == SP.AdamsSavePlane
    @test stop.landed && !stop.stretched
    stop = SP.adams_landing(policy, 0.0, 0.24, 1.0)
    @test stop.dz == 0.25 && stop.stretched
    stop = SP.adams_landing(policy, 0.0, 0.2, 1.0)
    @test !stop.landed && stop.kind == SP.AdamsNoLanding && iszero(stop.z_target)
    approach = SP.adams_approach(policy, 0.0, stop, 1.0)
    @test approach.dz == approach.destination == 0.125 && !approach.hard_kink
    coincident = adams_control_policy(; save_planes = [0.4, 1.0], hard_kinks = [0.4, 1.0])
    stop = SP.adams_landing(coincident, 0.2, 0.2, 1.0)
    @test stop.kind == SP.AdamsSavePlane && stop.z_target == 0.4
    @test SP.adams_approach(coincident, 0.2, stop, 1.0).hard_kink
    stop = SP.adams_landing(coincident, 0.8, 0.2, 1.0)
    @test stop.kind == SP.AdamsEnd
    @test !SP.adams_approach(coincident, 0.8, stop, 1.0).hard_kink
    cap_tie = adams_control_policy(;
        dz_max = 0.25, breakpoints = [0.0, 0.25, 1.0], save_planes = [0.75, 1.0],
        hard_kinks = Float64[],
    )
    stop = SP.adams_landing(cap_tie, 0.0, 0.25, 1.0)
    @test stop.kind == SP.AdamsIntervalWidth && !stop.landed
    tolerance = policy.landing_ulps * eps(1.0)
    near = adams_control_policy(;
        save_planes = [0.1 + tolerance, 1.0], hard_kinks = Float64[]
    )
    @test SP.adams_landing(near, 0.1, 0.01, 1.0).z_target == 0.1 + tolerance
    @test SP.adams_next_landing(near, 0.1, 1.0) == 1.0
    @test SP.adams_next_landing(near, 1.0, 1.0) === nothing
    @test_throws ArgumentError SP.adams_landing(policy, 1.0, 0.01, 1.0)
    @test_throws ArgumentError SP.adams_landing(policy, -0.1, 0.01, 1.0)
    @test_throws ArgumentError SP.adams_landing(policy, 0.0, Inf, 1.0)
    @test SP.adams_replay_kink(policy, 0.4 + eps(0.4), 8eps(0.4))
    @test !SP.adams_replay_kink(policy, 0.41, 8eps(0.4))
    quiet = adams_control_policy(; quiet_interval = 0.01, collar_rate = 10.0)
    @test SP.adams_quiet_cap(quiet, 0.2; quiet = true) == 0.01
    @test SP.adams_quiet_cap(quiet, 0.2; quiet = false) == 0.2
    @test SP.adams_quiet_cap(policy, 0.2; quiet = true) == 0.2
    @test SP.adams_collar_reentry(
        quiet, 0.2, 2; was_quiet = true, quiet = false
    ) == 0.7 / 10
    @test SP.adams_collar_reentry(quiet, 0.2, 2; was_quiet = false, quiet = false) == 0.2
    @test SP.adams_collar_reentry(quiet, 0.2, 2; was_quiet = true, quiet = true) == 0.2
    positions, slots = collect(0.1:0.1:0.6), collect(1:6)
    stencil = SP.AdamsStencil(5)
    @test adams_control_allocations(
        quiet, stencil, positions, slots, [3, 2, 4], [0.3, 0.5, 0.1]
    ) == 0
end
