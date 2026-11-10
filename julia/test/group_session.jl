# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as GSP

mutable struct GroupTestSource{E, G}
    execution::E
    gain::G
    calls::Int
    fail_on::Int
end

function (source::GroupTestSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_on &&
        throw(ArgumentError("controlled group source failure"))
    gain = source.gain
    GSP.each!(source.execution, size(output)...) do row, column
        output[row, column] = gain * field[row, column]
    end
    return nothing
end

Base.@kwdef mutable struct GroupTestServices
    warning_step::Int = 0
    stop_step::Int = 0
    barrier::Symbol = :none
    continuation::Symbol = :none
    accepted::Int = 0
    rows::Int = 0
    row_steps::Vector{Int} = zeros(Int, 17)
    calls::Int = 0
    fail_launch::Bool = false
    fail_ack::Bool = false
end

GSP._session_launch!(::GroupTestServices, session) = nothing
function GSP._session_observe!(service::GroupTestServices, session, kind, plane)
    service.fail_launch && kind === :launch &&
        throw(ArgumentError("controlled launch observation failure"))
    return nothing
end
GSP._session_finish!(::GroupTestServices, session) = nothing
GSP._session_quiet(::GroupTestServices, session) = false
GSP._session_quiet_interval(::GroupTestServices, session) = nothing
GSP._session_cap(::GroupTestServices, session, proposal) = proposal
GSP._session_finish_candidate!(
    ::GroupTestServices, session, candidate, dz, endpoint,
) = nothing
GSP._session_barrier(service::GroupTestServices, session) = service.barrier

function _group_test_row!(service, session)
    service.rows += 1
    service.row_steps[service.rows] = session.attempts.count - 1
    return nothing
end

function GSP._session_accepted!(service::GroupTestServices, session, dz)
    service.accepted += 1
    if service.accepted in (service.warning_step, service.stop_step)
        service.barrier = service.accepted == service.stop_step ? :boundary_stop : :warning
        service.continuation = :accepted_rest
    else
        _group_test_row!(service, session)
    end
    return nothing
end

function GSP._session_continue!(service::GroupTestServices, session)
    result = service.continuation
    service.continuation = :none
    if result === :accepted_rest
        service.calls += 1
        _group_test_row!(service, session)
    end
    return result
end

function GSP._session_acknowledge!(service::GroupTestServices, session, generation)
    service.fail_ack && throw(ArgumentError("controlled acknowledgment failure"))
    service.barrier = :none
    service.rows = 0
    return nothing
end

function group_test_member(;
        method = :rk, gain = 0.2 + 0.1im, capacity = 3, services = GroupTestServices(),
        fail_on = 0, z = 0.0, plane_pending = true, steps = fill(0.025, 8),
        edges = z .+ [0.0; cumsum(steps)], planes = [0.1, 0.2],
        shape = (3, 2), stop = GSP.StopFlag(), step_planes = 0, tasks = 1,
        real_gain = false,
    )
    execution = GSP.KernelExecution(GSP.KernelAbstractions.CPU(), tasks, true)
    field = fill(1.0 + 0.3im, shape)
    coefficient = real_gain ? Float64(real(gain)) : ComplexF64(gain)
    source = GroupTestSource(execution, coefficient, 0, fail_on)
    flow = GSP.LinearFlow(
        execution,
        GSP.DiagonalOperator(execution, fill(-0.1im, shape)), field
    )
    norm = GSP.MarchNorm(
        execution, fill(true, shape[1]), fill(true, shape[1]),
        ones(shape[1]), 1.0e-3, 0.0
    )
    schedule = GSP.MarchSchedule(;
        planes, breakpoints = [0.0, 0.4],
        hard_kinks = Float64[], stretch = 1.1, step_planes,
        replay_steps = steps, replay_edges = edges
    )
    if method === :low_storage
        return GSP.LowStorageSession(
            execution, field, z, GSP.RK4IPTableau(), norm,
            schedule, source, flow; stop, capacity, services, plane_pending
        )
    elseif method === :adams
        policy = GSP.AdamsControlPolicy(;
            order = 4, variable_order = false, safety = 0.8, max_growth = 2.0,
            min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
            collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
            collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
            equal_approach = true, interval_cap = false, dz_min = 1.0e-10, dz_max = 0.05,
            stretch = 1.1, z_end = last(planes), save_planes = planes,
            hard_kinks = Float64[], breakpoints = [0.0, 0.4], landing_ulps = 8,
            interpolate = false, quiet_interval = nothing,
        )
        return GSP.AdamsSession(
            execution, field, z, 0.025, policy, norm, schedule,
            source, flow; stop, capacity, services, plane_pending
        )
    end
    tableau = GSP.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    control = GSP.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.05, auto_init = false,
    )
    return GSP.RungeKuttaSession(
        execution, field, z, 0.025, tableau, control, norm,
        schedule, source, flow; stop, capacity, services, plane_pending
    )
end

function group_test_solo!(session)
    positions = Float64[]
    while !session.state.finished
        report = GSP.advance!(session; work_limit = 2, deadline = 60.0)
        append!(positions, @view(session.attempts.z[1:session.attempts.count]))
        GSP.acknowledge!(session, report.generation)
    end
    return positions
end

function group_test_collect!(group; work_limit = 5, deadline = 60.0, warning_error = false)
    positions = [Float64[] for _ in group.members]
    row_steps = [Int[] for _ in group.members]
    chunks = [Int[] for _ in group.members]
    events = [GSP.SessionEvent[] for _ in group.members]
    reports = GSP.GroupReport[]
    for iteration in 1:300
        group.state.finished && break
        report = GSP.advance_group!(group; work_limit, deadline)
        push!(reports, report)
        @test 0 <= report.attempts <= work_limit
        for index in eachindex(group.members)
            held = GSP.group_member_report(group, report.generation, index)
            held.available || continue
            if held.new_chunk
                @test !(held.chunk in chunks[index])
                push!(chunks[index], held.chunk)
                session = GSP.group_member_session(group, report.generation, index)
                append!(
                    positions[index],
                    @view(session.attempts.z[1:session.attempts.count])
                )
                offset = length(positions[index]) - session.attempts.count
                service = session.services
                for row in 1:service.rows
                    push!(row_steps[index], offset + service.row_steps[row])
                end
            end
            if held.new_event
                push!(events[index], held.report.kind)
                if warning_error && held.report.kind === GSP.SESSION_WARNING
                    old_chunk = held.chunk
                    GSP.fail_group_member!(group, report.generation, index)
                    after = GSP.group_member_report(group, report.generation, index)
                    @test after.chunk == old_chunk && after.eligible
                end
            end
        end
        GSP.acknowledge_group!(group, report.generation)
    end
    @test group.state.finished
    return (; positions, row_steps, chunks, events, reports)
end

function group_test_launch!(group)
    report = GSP.advance_group!(group; deadline = 60.0)
    @test report.kind === GSP.GROUP_WINDOW && report.window == -1
    @test all(slot -> slot.report.kind === GSP.SESSION_LAUNCH, group.slots)
    GSP.acknowledge_group!(group, report.generation)
    return nothing
end

function group_test_set_stop!(flag)
    GC.@preserve flag unsafe_store!(flag.address, UInt8(1), :release)
    return nothing
end

@testset "Group fixed replay retains each member's solo trajectory" begin
    for method in (:rk, :adams, :low_storage), work_limit in (1, 5, 256)
        members = (group_test_member(; method), group_test_member(; method, gain = -0.3im))
        group = GSP.FixedScheduleGroup(members; round_steps = 2)
        result = group_test_collect!(
            group; work_limit, deadline = work_limit == 1 ? 0.0 : 60.0,
        )
        @test group.state.outcome === GSP.GROUP_COMPLETED
        for (index, gain) in enumerate((0.2 + 0.1im, -0.3im))
            solo = group_test_member(; method, gain)
            trace = group_test_solo!(solo)
            # Both paths call the same concrete CPU method with task budget one,
            # identical scalar inputs, coefficients and fixed histories. There is no
            # grouped field arithmetic or state-affecting reduction, so the operation
            # traces match. This positive allowance covers output rounding, not a
            # solver forward-error bound. It does not qualify changed task budgets,
            # adaptive orders, fused/device arithmetic or FFT-based group sources.
            bound = 2 * eps(Float64) * maximum(abs, solo.field)
            @test bound > 0 && maximum(abs, members[index].field - solo.field) <= bound
            @test result.positions[index] == trace
            @test result.row_steps[index] == collect(0:7)
            @test members[index].source.calls == solo.source.calls
            @test members[index].state.attempts == solo.state.attempts == 8
            @test count(==(GSP.SESSION_PLANE), result.events[index]) == 2
        end
    end
end

@testset "Group admission is transactional and only constrains its schedule" begin
    @test_throws ArgumentError GSP.FixedScheduleGroup(())
    first = group_test_member()
    @test_throws ArgumentError GSP.FixedScheduleGroup((first, first))
    @test !first.state.closed && first.source.calls == 0 && first.state.generation == 0
    @test_throws ArgumentError GSP.FixedScheduleGroup((first,); round_steps = 0)
    @test_throws ArgumentError GSP.FixedScheduleGroup((first,); stop = first.stop)
    alternative = group_test_member(; shape = (4, 1), gain = -0.8im)
    group = GSP.FixedScheduleGroup((first, alternative))
    @test size(group.members[1].field) != size(group.members[2].field)
    GSP.close_group!(group)
    @test all(member -> member.state.closed, group.members)
    GSP.close_group!(group)
    @test_throws ArgumentError GSP.advance_group!(group)
    @test_throws ArgumentError GSP.request_group_snapshot!(group, 1)
    for second in (
            group_test_member(; steps = fill(0.05, 4)),
            group_test_member(; planes = [0.2]),
            group_test_member(; plane_pending = false),
            group_test_member(; step_planes = 1),
            group_test_member(; steps = Float64[], edges = Float64[]),
        )
        untouched = group_test_member()
        @test_throws ArgumentError GSP.FixedScheduleGroup((untouched, second))
        @test untouched.source.calls == second.source.calls == 0
        @test !untouched.state.closed && !second.state.closed
    end
    @test_throws ArgumentError GSP.FixedScheduleGroup(
        (group_test_member(planes = [0.09, 0.2]),),
    )
end

@testset "Group entrances reject restored history and previously accepted cursors" begin
    prior = group_test_member(;
        z = 0.1, steps = fill(0.025, 4), planes = [0.2],
        plane_pending = false
    )
    @test_throws ArgumentError GSP.FixedScheduleGroup((prior,))
    for method in (:rk, :adams, :low_storage)
        member = group_test_member(; method)
        member.state.attempts = 1
        @test_throws ArgumentError GSP.FixedScheduleGroup((member,))
        @test member.source.calls == 0 && !member.state.closed
    end
    history = group_test_member()
    GSP.record_norm!(history.history, 0.1)
    @test_throws ArgumentError GSP.FixedScheduleGroup((history,))
    adams = group_test_member(; method = :adams)
    adams.history.occupied[1] = true
    @test_throws ArgumentError GSP.FixedScheduleGroup((adams,))
end

@testset "Pending group generations and independently held member tickets" begin
    group = GSP.FixedScheduleGroup((group_test_member(), group_test_member()))
    report = GSP.advance_group!(group)
    held = GSP.group_member_report(group, report.generation, 1)
    @test held.available && held.eligible && held.new_chunk && held.chunk == 1
    @test GSP.group_member_session(group, report.generation, 1; exportable = true) ===
        group.members[1]
    @test_throws ArgumentError GSP.advance_group!(group)
    @test_throws ArgumentError GSP.acknowledge_group!(group, report.generation + 1)
    @test_throws ArgumentError GSP.group_member_report(group, report.generation, 0)
    @test_throws ArgumentError GSP.group_member_report(group, report.generation, 3)
    @test_throws ArgumentError GSP.fail_group_member!(group, report.generation, 1)
    GSP.acknowledge_group!(group, report.generation)
    @test all(member -> member.state.pending, group.members)
    @test_throws ArgumentError GSP.group_member_session(group, report.generation, 1)
    @test_throws ArgumentError GSP.acknowledge_group!(group, report.generation)
    @test_throws ArgumentError GSP.advance_group!(group; work_limit = 0)
    @test_throws ArgumentError GSP.advance_group!(group; deadline = NaN)
    @test_throws ArgumentError GSP.advance_group!(group; deadline = -1.0)
    @test GSP.group_memory(group).slots == sizeof(group.slots)
end

@testset "Warnings at a round edge complete rows without overshooting" begin
    service = GroupTestServices(; warning_step = 1)
    group = GSP.FixedScheduleGroup(
        (group_test_member(; services = service), group_test_member());
        round_steps = 1
    )
    group_test_launch!(group)
    warning = GSP.advance_group!(group; work_limit = 1, deadline = 60.0)
    @test warning.kind === GSP.GROUP_SERVICE
    @test [member.state.replay for member in group.members] == [2, 1]
    held = GSP.group_member_report(group, warning.generation, 1)
    @test held.report.kind === GSP.SESSION_WARNING && group.members[1].attempts.count == 1
    @test service.rows == 0 && service.calls == 0
    GSP.acknowledge_group!(group, warning.generation)
    resumed = GSP.advance_group!(group; work_limit = 1, deadline = 60.0)
    @test service.calls == 1 && service.rows == 1 && service.row_steps[1] == -1
    @test all(member -> member.state.replay == 2, group.members)
    @test resumed.attempts == 1
    GSP.acknowledge_group!(group, resumed.generation)
    group_test_set_stop!(group.stop)
    ended = GSP.advance_group!(group; deadline = 60.0)
    @test ended.outcome === GSP.GROUP_STOPPED
    @test all(member -> member.state.z == 0.025, group.members)
end

@testset "Mid-round cancellation waits for every remaining common edge" begin
    group = GSP.FixedScheduleGroup(ntuple(_ -> group_test_member(), 3); round_steps = 2)
    group_test_launch!(group)
    progress = GSP.advance_group!(group; work_limit = 1, deadline = 0.0)
    @test progress.attempts == 1
    @test [member.state.z for member in group.members] == [0.025, 0.0, 0.0]
    GSP.acknowledge_group!(group, progress.generation)
    group_test_set_stop!(group.stop)
    result = group_test_collect!(group; work_limit = 1, deadline = 0.0)
    @test group.state.outcome === GSP.GROUP_STOPPED
    @test all(member -> member.state.z == 0.05, group.members)
    @test length.(result.positions) == [1, 2, 2]
end

@testset "Numerical failures retain actual cursors and old chunk identities" begin
    group = GSP.FixedScheduleGroup(
        (
            group_test_member(), group_test_member(; fail_on = 1), group_test_member(),
        ); round_steps = 2
    )
    group_test_launch!(group)
    progressed = GSP.advance_group!(group; work_limit = 1, deadline = 0.0)
    old = GSP.group_member_report(group, progressed.generation, 1)
    GSP.acknowledge_group!(group, progressed.generation)
    ended = GSP.advance_group!(group; deadline = 60.0)
    @test ended.kind === GSP.GROUP_TERMINAL && ended.outcome === GSP.GROUP_FAILED
    @test ended.failure_member == 2
    @test [member.state.z for member in group.members] == [0.05, 0.0, 0.0]
    @test all(member -> member.state.finished, group.members)
    first = GSP.group_member_report(group, ended.generation, 1)
    third = GSP.group_member_report(group, ended.generation, 3)
    @test first.chunk == old.chunk + 1 && first.new_chunk && first.eligible
    @test third.chunk == 1 && !third.new_chunk && third.eligible
    @test first.report.generation != old.report.generation
    poisoned = GSP.FixedScheduleGroup(
        (
            group_test_member(; method = :low_storage, fail_on = 1),
            group_test_member(; method = :low_storage),
        )
    )
    group_test_launch!(poisoned)
    bad = GSP.advance_group!(poisoned; deadline = 60.0)
    @test bad.outcome === GSP.GROUP_FAILED
    @test !GSP.group_member_report(poisoned, bad.generation, 1).eligible
    @test GSP.group_member_report(poisoned, bad.generation, 2).eligible
    @test_throws ArgumentError GSP.group_member_session(
        poisoned, bad.generation, 1;
        exportable = true
    )
end

@testset "Boundary stops and raised warnings end only their member" begin
    for raised in (false, true)
        service = raised ? GroupTestServices(; warning_step = 1) :
            GroupTestServices(; stop_step = 1)
        group = GSP.FixedScheduleGroup(
            (
                group_test_member(; services = service),
                group_test_member(),
            ); round_steps = 2
        )
        result = group_test_collect!(group; warning_error = raised)
        @test group.state.outcome === GSP.GROUP_COMPLETED
        @test group.members[1].state.z == 0.025
        @test group.members[2].state.z == last(group.members[2].schedule.replay_edges)
        @test length.(result.positions) == [1, 8]
        @test service.calls == 0
    end
end

@testset "Snapshot barriers precede later trials and retain plane precedence" begin
    group = GSP.FixedScheduleGroup(
        (group_test_member(), group_test_member()); round_steps = 4,
    )
    @test GSP.request_group_snapshot!(group, 1) == (1, false)
    @test GSP.request_group_snapshot!(group, 1) == (1, true)
    group_test_launch!(group)
    snapshot = GSP.advance_group!(group; deadline = 60.0)
    held = GSP.group_member_report(group, snapshot.generation, 1)
    @test snapshot.kind === GSP.GROUP_SERVICE && held.report.kind === GSP.SESSION_SNAPSHOT
    @test held.report.z == 0.025 && group.members[2].state.z == 0.0
    GSP.acknowledge_group!(group, snapshot.generation)
    result = group_test_collect!(group)
    @test group.state.outcome === GSP.GROUP_COMPLETED
    @test length.(result.positions) == [7, 8]
end

@testset "An external group failure replaces events without redelivering chunks" begin
    group = GSP.FixedScheduleGroup((group_test_member(), group_test_member()))
    launched = GSP.advance_group!(group)
    chunks = [
        GSP.group_member_report(group, launched.generation, index).chunk for index in 1:2
    ]
    GSP.fail_group!(
        group, launched.generation, :ConsumerError, "controlled service failure",
    )
    ended = GSP.group_report(group, launched.generation)
    @test ended.kind === GSP.GROUP_TERMINAL && ended.outcome === GSP.GROUP_FAILED
    @test all(
        index -> GSP.group_member_report(group, launched.generation, index).chunk ==
            chunks[index], 1:2
    )
    @test all(member -> member.state.failure_code === :ConsumerError, group.members)
    GSP.acknowledge_group!(group, launched.generation)
    @test_throws ArgumentError GSP.advance_group!(group)
end

@testset "Unlaunched peers and escaped member calls never invent eligible fields" begin
    group = GSP.FixedScheduleGroup(
        (
            group_test_member(; services = GroupTestServices(; fail_launch = true)),
            group_test_member(),
        )
    )
    failed = GSP.advance_group!(group; deadline = 60.0)
    @test failed.outcome === GSP.GROUP_FAILED && failed.failure_member == 1
    peer = GSP.group_member_report(group, failed.generation, 2)
    @test !peer.eligible && !peer.new_chunk && peer.chunk == 0
    @test !group.members[2].state.launched && group.members[2].source.calls == 0
    @test_throws ArgumentError GSP.group_member_session(
        group, failed.generation, 2;
        exportable = true
    )
    escaped = GSP.FixedScheduleGroup((group_test_member(), group_test_member()))
    group_test_launch!(escaped)
    escaped.members[1].services.fail_ack = true
    failure = GSP.advance_group!(escaped; deadline = 60.0)
    @test failure.outcome === GSP.GROUP_FAILED && failure.failure_member == 1
    @test GSP.group_failure(escaped, failure.generation) ==
        (:ArgumentError, "ArgumentError: controlled acknowledgment failure", 1)
    @test !GSP.group_member_report(escaped, failure.generation, 1).eligible
    @test GSP.group_member_report(escaped, failure.generation, 2).eligible
    @test all(member -> member.source.calls == 0, escaped.members)
end

function group_test_advance_ack!(group)
    report = GSP.advance_group!(group; work_limit = 2, deadline = 60.0)
    GSP.acknowledge_group!(group, report.generation)
    return nothing
end

function group_test_allocated(group)
    group_test_advance_ack!(group)
    group_test_advance_ack!(group)
    return @allocated group_test_advance_ack!(group)
end

@testset "Warmed group transport keeps its fixed buffers and allocates nothing" begin
    for method in (:rk, :adams, :low_storage), tasks in (1, 2)
        group = GSP.FixedScheduleGroup(
            (
                group_test_member(; method, tasks),
                group_test_member(; method, tasks),
            ); round_steps = 2
        )
        slots, origins, edges = group.slots, group.replay_origins, group.plane_edges
        bytes = GSP.group_memory(group)
        @test group_test_allocated(group) == 0
        @test group.slots === slots && group.replay_origins === origins &&
            group.plane_edges === edges
        @test GSP.group_memory(group) == bytes
    end
end

@testset "A latched snapshot at a common edge is serviced before a group stop" begin
    group = GSP.FixedScheduleGroup(
        (group_test_member(), group_test_member()); round_steps = 1,
    )
    group_test_launch!(group)
    partial = GSP.advance_group!(group; work_limit = 2, deadline = 60.0)
    @test all(member -> member.state.z == 0.025, group.members)
    GSP.acknowledge_group!(group, partial.generation)
    GSP.request_group_snapshot!(group, 1)
    group_test_set_stop!(group.stop)
    snapshot = GSP.advance_group!(group; deadline = 60.0)
    @test snapshot.kind === GSP.GROUP_SERVICE && snapshot.attempts == 0
    @test GSP.group_member_report(group, snapshot.generation, 1).report.kind ===
        GSP.SESSION_SNAPSHOT
    GSP.acknowledge_group!(group, snapshot.generation)
    ended = GSP.advance_group!(group; deadline = 60.0)
    @test ended.outcome === GSP.GROUP_STOPPED && ended.attempts == 0
    @test all(member -> member.state.z == 0.025, group.members)
end

@testset "Concrete heterogeneous member ownership needs no fused numerical layout" begin
    members = (group_test_member(), group_test_member(; real_gain = true, shape = (4, 1)))
    @test typeof(members[1]) !== typeof(members[2])
    group = GSP.FixedScheduleGroup(members; round_steps = 2)
    @test group_test_allocated(group) == 0
    result = group_test_collect!(group)
    @test group.state.outcome === GSP.GROUP_COMPLETED
    @test members[1].source.calls == members[2].source.calls == 16
    @test size(members[1].field) == (3, 2) && size(members[2].field) == (4, 1)
end

@testset "Generation exhaustion refuses before touching member state" begin
    group = GSP.FixedScheduleGroup((group_test_member(), group_test_member()))
    group.state.generation = typemax(Int)
    @test_throws OverflowError GSP.advance_group!(group)
    @test all(
        member -> member.source.calls == 0 && member.state.generation == 0,
        group.members
    )
    group.state.generation = 0
    group.members[2].state.generation = typemax(Int) - 1
    @test_throws ArgumentError GSP.advance_group!(group)
    @test group.state.generation == 0 && !group.state.pending
    @test group.members[1].state.generation == 0 && group.members[1].source.calls == 0
    failure = GSP.FixedScheduleGroup((group_test_member(), group_test_member()))
    launch = GSP.advance_group!(failure)
    failure.members[2].state.generation = typemax(Int)
    before = failure.members[1].state.generation
    @test_throws ArgumentError GSP.fail_group!(
        failure, launch.generation,
        :ExternalFailure, "failed"
    )
    @test failure.state.outcome === GSP.GROUP_RUNNING && !failure.state.finished
    @test failure.members[1].state.generation == before
    @test !failure.members[1].state.finished
end

@testset "Distinct requested planes cannot share one landed group edge" begin
    for planes in ([0.1, nextfloat(0.1), 0.2], [0.1, nextfloat(0.1, 2), 0.2])
        members = ntuple(_ -> group_test_member(; planes), 2)
        fields = map(member -> copy(member.field), members)
        @test_throws r"distinct increasing replay edges" GSP.FixedScheduleGroup(members)
        for (member, original) in zip(members, fields)
            @test member.field == original && member.source.calls == 0
            @test !member.state.pending && !member.state.closed && !member.state.launched
            @test member.state.generation == member.state.attempts == 0
        end
    end
end

@testset "Deadlines bound launch and continuation turns without taking a trial" begin
    group = GSP.FixedScheduleGroup(ntuple(_ -> group_test_member(), 3))
    for index in 1:3
        report = GSP.advance_group!(group; deadline = 0.0)
        @test report.attempts == 0
        @test report.kind === (index == 3 ? GSP.GROUP_WINDOW : GSP.GROUP_PROGRESS)
        @test count(member -> member.state.launched, group.members) == index
        @test count(slot -> slot.new_event, group.slots) == 1
        @test GSP.group_member_report(group, report.generation, index).new_event
        @test all(member -> member.source.calls == 0, group.members)
        GSP.acknowledge_group!(group, report.generation)
    end
    GSP.close_group!(group)

    service = GroupTestServices(; warning_step = 1)
    resumed = GSP.FixedScheduleGroup(
        (group_test_member(; services = service), group_test_member()); round_steps = 1,
    )
    group_test_launch!(resumed)
    warning = GSP.advance_group!(resumed; deadline = 60.0)
    @test warning.kind === GSP.GROUP_SERVICE && warning.attempts == 1
    GSP.acknowledge_group!(resumed, warning.generation)
    progress = GSP.advance_group!(resumed; deadline = 0.0)
    @test progress.kind === GSP.GROUP_PROGRESS && progress.attempts == 0
    @test service.calls == service.rows == 1 && service.row_steps[1] == -1
    @test [member.state.replay for member in resumed.members] == [2, 1]
    @test resumed.members[2].source.calls == 0
    @test count(slot -> slot.new_event, resumed.slots) == 1
    GSP.acknowledge_group!(resumed, progress.generation)
    next = GSP.advance_group!(resumed; deadline = 0.0)
    @test next.attempts == 1
    @test all(member -> member.state.replay == 2, resumed.members)
    @test service.calls == 1
    GSP.close_group!(resumed)
end
