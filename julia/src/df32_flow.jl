# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public DF32LinearFlow, DF32TrialSite, DF32TrialFlow, prepare_df32_trial!

"""
    DF32TrialSite(origin, step, destination, replay_index)

Retain actual binary64 trial coordinates and the canonical landing fact.

`origin`, positive `step` and `destination` are metres. The raw endpoint is the
separately evaluated `origin+step`; canonical `destination` is not substituted into
that graph. `replay_index` is the one-based actual itinerary cursor, or zero for an
adaptive trial. Construction validates finite advancing raw coordinates; preparation
checks the complete site against its bound schedule. Throw `ArgumentError` otherwise.
"""
struct DF32TrialSite
    origin::Float64
    step::Float64
    destination::Float64
    replay_index::Int

    function DF32TrialSite(
            origin::Float64, step::Float64, destination::Float64, replay_index::Int,
        )
        all(isfinite, (origin, step, destination, origin + step)) && step > 0.0 &&
            origin + step > origin && replay_index >= 0 ||
            throw(ArgumentError("a DF32 trial needs finite advancing coordinates"))
        return new(origin, step, destination, replay_index)
    end
end

# Symbol-bearing WideCertificate and owner-bearing WideOperation are not bank
# elements. Their fixed-size projection preserves every scalar/radius and records
# the status as an exact code, while one operator stays rooted on the flow.
struct _DF32FlowCertificate
    real_error::Float64
    imag_error::Float64
    intermediate::Float64
    status::UInt8
end

function _df32_flow_certificate(certificate::WideCertificate)
    status = certificate.status === :valid ? 0x00 :
        certificate.status === :intermediate_range ? 0x01 :
        certificate.status === :function_domain ? 0x02 :
        certificate.status === :assembly_error ? 0x03 :
        throw(ArgumentError("unknown wide flow certificate status"))
    return _DF32FlowCertificate(
        certificate.real_error, certificate.imag_error, certificate.intermediate, status,
    )
end

struct _DF32FlowOperation
    scalars::_WideScalars{DF32}
    scalar_certificate::_WideScalars{_WideBall}
    certificate::_DF32FlowCertificate
    origin::Float64
    destination::Float64
    step::Float64
    sample_z::Float64
    generation::UInt64
end

function _df32_flow_operation(operation::WideOperation{Nothing}, generation::UInt64)
    operation.kind === :exponent && operation.destination === nothing ||
        throw(ArgumentError("a trial flow needs an ordinary exponent operation"))
    return _DF32FlowOperation(
        operation.scalars, operation.scalar_certificate,
        _df32_flow_certificate(operation.certificate), operation.z0, operation.z1,
        operation.step, operation.sample_z, generation,
    )
end

mutable struct _DF32FlowPlans
    banks::NTuple{2, Vector{_DF32FlowOperation}}
    active::Int
    generation::UInt64
    site::Union{Nothing, DF32TrialSite}

    function _DF32FlowPlans(count::Int)
        banks = (
            Vector{_DF32FlowOperation}(undef, count),
            Vector{_DF32FlowOperation}(undef, count),
        )
        return new(banks, 0, zero(UInt64), nothing)
    end
end

_df32_flow_slots(tableau::EmbeddedTableau) = length(tableau.nodes) + 1
_df32_flow_slots(::Nothing) = 2

"""
    DF32LinearFlow(execution, operator, state::AbstractMatrix{ComplexF32}, spec;
                   tabulate=false)

Prepare a CPU DF32 flow bound to one actual binary32 start and its immutable inputs.

`operator` is the original owned binary64 constant, represented cumulative or
real-weight expansion operator. `state` supplies `(n_omega,n_columns)` shape and
binary32 precision, without being retained. `spec::WideStartSpec` supplies the actual
RK or fixed RK4IP method, schedule, start, geometry and source scales. This constructor
creates one `WideStartBinding` from that same operator; it accepts no independent
operator/certificate pairing. The original operator remains shared immutable input.

Two bounded host operation banks and O(n_columns) failure scratch are owned by this
flow. Certificates are computed before source/state mutation; only their DF32 graph
is launched. Use `prepare_df32_trial!` to get the callable token. A bare flow refuses
numerical entry. Completed starts allocate no operation slots. Neither public DF32
admission nor a device arithmetic model follows from this CPU implementation.

Throw `ArgumentError` before state allocation for non-CPU execution, other precision,
unsupported operator/method, tabulation or invalid immutable inputs. Shape mismatches
raise `DimensionMismatch`. Dense and Adams attachment remain explicitly unsupported.
"""
struct DF32LinearFlow{E, O, B}
    execution::E
    operator::O
    binding::B
    shape::Tuple{Int, Int}
    state_type::Type{ComplexF32}
    plans::_DF32FlowPlans
    failures::Vector{Int}

    function DF32LinearFlow(
            execution::E, operator::O, binding::B, shape::Tuple{Int, Int},
            plans::_DF32FlowPlans, failures::Vector{Int},
        ) where {E <: KernelExecution, O, B <: WideStartBinding}
        return new{E, O, B}(
            execution, operator, binding, shape, ComplexF32, plans, failures,
        )
    end
end

function DF32LinearFlow(
        execution::KernelExecution, operator, state::AbstractMatrix{ComplexF32},
        spec::WideStartSpec; tabulate::Bool = false,
    )
    oncpu(execution) || throw(ArgumentError("DF32 trial flows currently require CPU"))
    spec.method in (:rk, :low_storage) ||
        throw(ArgumentError("DF32 Adams complete factors are not qualified"))
    tabulate && throw(ArgumentError("DF32 linear tabulation is not qualified"))
    state isa StridedMatrix{ComplexF32} && !isempty(state) && all(isfinite, state) ||
        throw(ArgumentError("DF32 flow preparation needs finite strided ComplexF32 state"))
    operator isa Union{DiagonalOperator, RepresentedOperator, ExpansionOperator} ||
        throw(ArgumentError("DF32 separable spatial and other operators are not qualified"))
    shape = _wide_start_shape(operator)
    size(state) == shape ||
        throw(DimensionMismatch("DF32 flow state and operator shapes disagree"))
    binding = prepare_wide_start(execution, operator, spec)
    count = binding.operator === nothing ? 0 : _df32_flow_slots(spec.tableau)
    plans = _DF32FlowPlans(count)
    return DF32LinearFlow(execution, operator, binding, shape, plans, zeros(Int, shape[2]))
end

function DF32LinearFlow(execution, operator, state, spec; tabulate::Bool = false)
    throw(ArgumentError("DF32 linear flow requires a binary32 state and actual start spec"))
end

"""
    DF32TrialFlow

Apply only the intervals of one successfully certified DF32 trial plan.

Obtain this immutable token from `prepare_df32_trial!`. It binds its owning flow,
actual site and published generation. A later successful plan makes this token stale;
a refused plan preserves it. Calling `(out,input,z0,z1)` selects the corrected forward
or backward factor of an exactly bound interval and returns `nothing`.

Caller matrices have the flow's shape and ComplexF32 element type. Output is writable
strided storage; the inputs are disjoint or exactly the same array. Both must be
disjoint from every retained flow array. Unsupported coordinates, stale generations,
invalid storage and overlap refuse before writes. Nonfinite numerical results throw
outside the launched body, preserving the enclosing march's failure policy.
"""
struct DF32TrialFlow{F}
    flow::F
    site::DF32TrialSite
    generation::UInt64

    function DF32TrialFlow(
            flow::F, site::DF32TrialSite, generation::UInt64,
        ) where {F <: DF32LinearFlow}
        return new{F}(flow, site, generation)
    end
end

_flow_shape(flow::DF32LinearFlow) = flow.shape
_inplace_flow(::DF32LinearFlow) = true

function _flow_arithmetic(flow::DF32LinearFlow, state, arithmetic)
    eltype(state) === ComplexF32 && arithmetic == flow.binding.spec.arithmetic ||
        throw(ArgumentError("DF32 flow and march precision or source scales differ"))
    size(state) == flow.shape ||
        throw(DimensionMismatch("DF32 flow and march shapes differ"))
    return nothing
end

function _df32_flow_scheme(flow, scheme)
    spec = flow.binding.spec
    valid = spec.method === :rk ? scheme === spec.tableau : scheme isa RK4IPTableau
    valid || throw(ArgumentError("DF32 trial method differs from its bound start"))
    return nothing
end

function _flow_start(flow::DF32LinearFlow, scheme, position, schedule)
    _df32_flow_scheme(flow, scheme)
    isequal(position, flow.binding.spec.start) && schedule === flow.binding.spec.schedule ||
        throw(ArgumentError("DF32 flow belongs to another actual start or schedule"))
    return nothing
end

function _df32_flow_site(flow, site)
    spec, schedule = flow.binding.spec, flow.binding.spec.schedule
    prepared = flow.binding.operator
    prepared === nothing && throw(ArgumentError("a completed DF32 start has no trial"))
    spec.start <= site.origin < last(schedule.planes) ||
        throw(ArgumentError("DF32 trial origin lies outside its actual run"))
    geometry = prepared.geometry
    geometry.lower <= site.destination <= geometry.upper ||
        throw(ArgumentError("DF32 canonical destination lies outside its geometry"))
    if isempty(schedule.replay_steps)
        site.destination <= last(schedule.planes) ||
            throw(ArgumentError("DF32 adaptive destination exceeds its requested end"))
        iszero(site.replay_index) ||
            throw(ArgumentError("adaptive DF32 trials have no replay cursor"))
        abs(site.destination - (site.origin + site.step)) <=
            _march_slack(site.origin + site.step, site.destination) ||
            throw(ArgumentError("DF32 canonical destination disagrees with its endpoint"))
    else
        finish = last(schedule.planes)
        site.destination <= finish ||
            site.destination - finish <= _replay_slack(schedule, finish) ||
            throw(ArgumentError("DF32 replay destination exceeds its requested end"))
        index = site.replay_index
        1 <= index <= length(schedule.replay_steps) ||
            throw(ArgumentError("DF32 trial replay cursor is outside its itinerary"))
        isequal(site.step, schedule.replay_steps[index]) &&
            isequal(site.destination, schedule.replay_edges[index + 1]) &&
            abs(site.origin - schedule.replay_edges[index]) <=
            _replay_slack(schedule, site.origin) ||
            throw(ArgumentError("DF32 trial differs from its actual replay step or edge"))
    end
    return prepared
end

_df32_flow_interval(scheme::EmbeddedTableau, site, index) = (
    site.origin,
    index <= length(scheme.nodes) ? site.origin + scheme.nodes[index] * site.step :
        site.origin + site.step,
)
function _df32_flow_interval(::RK4IPTableau, site, index)
    midpoint = site.origin + 0.5site.step
    return index == 1 ? (site.origin, midpoint) : (midpoint, site.origin + site.step)
end

"""
    prepare_df32_trial!(flow, scheme, site)

Certify every actual RK or RK4IP flow before publishing a bounded trial token.

`flow::DF32LinearFlow` owns the original operator binding. `scheme` is its exact
`EmbeddedTableau` or `RK4IPTableau`; `site::DF32TrialSite` retains supplied step, raw
coordinate origin, canonical destination and actual replay cursor separately. Fill
only the inactive operation bank, preserving every scalar-bundle error and certificate
radius. Publish a new generation only after every complete exponent is valid.

Return a `DF32TrialFlow`. A refusal preserves the published bank and generation,
state, source calls and accepted method history; the inactive scratch may change.
Preparation is allocation-free after warmup when consumed by a trial. Throw
`ArgumentError` for mismatched method/site, exhausted generation or certificate range.
"""
function prepare_df32_trial!(
        flow::DF32LinearFlow, scheme::Union{EmbeddedTableau, RK4IPTableau},
        site::DF32TrialSite,
    )
    _df32_flow_scheme(flow, scheme)
    prepared = _df32_flow_site(flow, site)
    plans = flow.plans
    plans.generation < typemax(UInt64) ||
        throw(ArgumentError("DF32 trial generation capacity is exhausted"))
    generation = plans.generation + one(UInt64)
    bank_index = plans.active == 1 ? 2 : 1
    bank = plans.banks[bank_index]
    for index in eachindex(bank)
        origin, destination = _df32_flow_interval(scheme, site, index)
        operation = wide_operation(prepared, origin, destination; step = site.step)
        bank[index] = _df32_flow_operation(operation, generation)
        _wide_check_operation(operation, :exponent)
    end
    plans.active, plans.site, plans.generation = bank_index, site, generation
    return DF32TrialFlow(flow, site, generation)
end

function _df32_flow_token(trial)
    plans = trial.flow.plans
    trial.generation == plans.generation && plans.active != 0 &&
        trial.site === plans.site ||
        throw(ArgumentError("DF32 trial token is stale or belongs to another site"))
    return plans.banks[plans.active]
end

function _check_trial_flow(flow::DF32LinearFlow, scheme, position, step, state, work)
    throw(ArgumentError("a DF32 numerical trial requires its certified trial-flow token"))
end

function _check_trial_flow(trial::DF32TrialFlow, scheme, position, step, state, work)
    _df32_flow_token(trial)
    _df32_flow_scheme(trial.flow, scheme)
    isequal(position, trial.site.origin) && isequal(step, trial.site.step) ||
        throw(ArgumentError("DF32 numerical trial differs from its certified coordinates"))
    _df32_flow_state(trial.flow, state)
    _df32_flow_work(trial.flow, work)
    return nothing
end

function _prepare_trial_flow(
        flow::DF32LinearFlow, scheme, position, step, destination, replay_index,
    )
    return prepare_df32_trial!(
        flow, scheme, DF32TrialSite(position, step, destination, replay_index),
    )
end

function _flow_start(flow::DF32TrialFlow, scheme, position, schedule)
    throw(ArgumentError("a DF32 session requires its owning flow, not a trial token"))
end

function _df32_flow_retained(array, flow)
    _retained_alias(array, flow) && return true
    # The general owner walk treats arrays as leaves. Explicitly visit the saved
    # matrices within the binding's optional rate vector as well.
    for rate in flow.binding.rates
        _retained_alias(array, rate) && return true
    end
    return false
end

function _df32_flow_state(flow, state)
    state isa StridedMatrix{ComplexF32} && size(state) == flow.shape ||
        throw(ArgumentError("DF32 trial state differs from its prepared storage"))
    _df32_flow_retained(state, flow) &&
        throw(ArgumentError("DF32 trial state overlaps retained flow data"))
    return nothing
end

_df32_flow_work_arrays(flow, ::Tuple{}) = nothing
function _df32_flow_work_arrays(flow, arrays::Tuple)
    _df32_flow_retained(first(arrays), flow) &&
        throw(ArgumentError("DF32 trial workspace overlaps retained flow data"))
    _df32_flow_work_arrays(flow, Base.tail(arrays))
    return nothing
end

function _df32_flow_work(flow, work::RungeKuttaWork)
    eltype(work.candidate) === ComplexF32 ||
        throw(ArgumentError("DF32 RK workspace requires binary32 state storage"))
    scale = work.norm_scale
    scale isa _Float32NormScale ||
        throw(ArgumentError("DF32 RK workspace requires binary32 norm storage"))
    return _df32_flow_work_arrays(
        flow, (
            work.stages, work.combination, work.lab, work.source, work.transported,
            work.candidate, work.defect, work.norm_parts, scale.mantissa, scale.exponent,
        )
    )
end

function _df32_flow_work(flow, work::LowStorageWork)
    eltype(work.accumulator) === ComplexF32 ||
        throw(ArgumentError("DF32 low-storage workspace requires binary32 storage"))
    return _df32_flow_work_arrays(
        flow, (
            work.accumulator, work.stage, work.source, work.norm_mask,
            work.norm_parts, work.finite_columns,
        )
    )
end

function _df32_flow_buffers(flow, output, input)
    output isa StridedMatrix{ComplexF32} && input isa StridedMatrix{ComplexF32} ||
        throw(ArgumentError("DF32 flow buffers require strided ComplexF32 storage"))
    size(output) == size(input) == flow.shape ||
        throw(DimensionMismatch("DF32 flow buffers do not match their preparation"))
    _transfer_writable_destination(output)
    output !== input && _transfer_arrays_alias(output, input) &&
        throw(ArgumentError("DF32 flow refuses partial input/output overlap"))
    (_df32_flow_retained(output, flow) || _df32_flow_retained(input, flow)) &&
        throw(ArgumentError("DF32 caller buffers overlap retained flow data"))
    return nothing
end

function (trial::DF32TrialFlow)(output, input, origin::Float64, destination::Float64)
    bank = _df32_flow_token(trial)
    selected, backward = 0, false
    for index in eachindex(bank)
        operation = bank[index]
        if isequal(origin, operation.origin) && isequal(destination, operation.destination)
            selected, backward = index, false
            break
        elseif isequal(origin, operation.destination) &&
                isequal(destination, operation.origin)
            selected, backward = index, true
            break
        end
    end
    selected > 0 ||
        throw(ArgumentError("DF32 flow interval was not certified for this trial"))
    operation = bank[selected]
    operation.generation == trial.generation && iszero(operation.certificate.status) ||
        throw(ArgumentError("DF32 flow operation has an invalid certificate or generation"))
    flow = trial.flow
    _df32_flow_buffers(flow, output, input)
    _source_finite!(flow.failures, input, flow.execution, :flow_input)
    if origin == destination
        copyto!(output, input)
        return nothing
    end
    prepared = flow.binding.operator
    coefficients, terms, kind, bundle =
        prepared.coefficients, prepared.terms, prepared.kind, operation.scalars
    direction = backward ? 2 : 1
    each!(flow.execution, size(output)...) do row, column
        index = row + (column - 1) * size(output, 1)
        rate = _wide_graph(
            coefficients, terms, nothing, nothing, kind, index, bundle,
            Val(:compute)
        )
        phase = _wide_result(rate, bundle, 0x00, rate)
        pair = _df32_exp_pair(phase)
        output[row, column] = pair[direction] * input[row, column]
    end
    _source_finite!(flow.failures, output, flow.execution, :flow)
    return nothing
end

function (flow::DF32LinearFlow)(output, input, origin::Float64, destination::Float64)
    throw(ArgumentError("DF32 flow application requires a certified trial token"))
end
