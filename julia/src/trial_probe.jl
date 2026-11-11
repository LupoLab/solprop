# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public TrialProbe, start_trial_probe, trial_probe!, copy_trial_probe_field!
public trial_probe_status, trial_probe_memory, close_trial_probe!

mutable struct _TrialProbeState
    loaded_z::Float64
    default_dz::Float64
    trial_z::Float64
    trial_dz::Float64
    ready::Bool
    closed::Bool
    failed::Bool
    in_flight::Bool
    source_evaluations::Int
    norm_evaluations::Int
    trials::Int
    error_norm::Float64
end

struct _TrialProbeSource{S}
    source::S
    state::_TrialProbeState
end

@inline function (sample::_TrialProbeSource)(output, field, position)
    sample.state.source_evaluations += 1
    return sample.source(output, field, position)
end

"""
    TrialProbe(execution, initial, z, dz_init, tableau, norm, schedule, source, flow;
               split=nothing, noise=nothing, resume=false)

Own independent single-trial arithmetic without a propagation session or controller.

Copy the finite CPU binary64 `initial` matrix `(n_omega,n_columns)` in internal units
and the FFT convention. Copy `norm::MarchNorm`, add optional binary64 `InputNoise`
only under its fresh-launch rule, and apply the exact active mask. The input array is
never changed. `source`, `flow::LinearFlow`, optional split workspace and noise work
are independently owned numerical preparations adopted by the probe; they must not
alias the originating field. Immutable source/operator data remain shared with their
preparation. An in-stage collar is enabled independently of any prior monitor state.

`z` and `dz_init` are metres. The load must lie in the `MarchSchedule` operator span
and at a replay edge if one exists. The probe retains no schedule/plane arrays. For an
`EmbeddedTableau`, including the request's tableau when its march is Adams, the default
length is `dz_init`. For `RK4IPTableau`, a fixed schedule is required and the default is
the step at the loaded edge (the last step at the final edge). Overriding a trial's
anchor does not change that loaded default, matching the shared timing-probe contract.

Every workspace, counter and unread candidate belongs to this probe. No method
continuation, monitor, history, event, acknowledgment or acceptance policy is created.
Throw `ArgumentError` on invalid admission and `DimensionMismatch` for shape mismatch.
"""
struct TrialProbe{T, E, A, W, N, S, F, B, I}
    tableau::T
    execution::E
    field::A
    work::W
    norm::N
    source::S
    flow::F
    split::B
    noise::I
    state::_TrialProbeState
    sampler::_TrialProbeSource{S}
end

function _trial_probe_default(z, dz_init, schedule::MarchSchedule, tableau)
    isfinite(z) && first(schedule.breakpoints) <= z <= last(schedule.breakpoints) &&
        isfinite(dz_init) && dz_init > 0 ||
        throw(ArgumentError("invalid timing-probe load position or initial step"))
    index = 1
    if !isempty(schedule.replay_steps)
        index = argmin(abs.(schedule.replay_edges .- z))
        abs(schedule.replay_edges[index] - z) <= _replay_slack(schedule, z) ||
            throw(ArgumentError("a timing probe must load at a recorded replay edge"))
    end
    if tableau isa RK4IPTableau
        !isempty(schedule.replay_steps) ||
            throw(ArgumentError("a low-storage timing probe needs a fixed schedule"))
        return schedule.replay_steps[min(index, length(schedule.replay_steps))]
    end
    return dz_init
end

function _trial_probe_mask!(field, active, execution)
    each!(execution, size(field)...) do row, column
        if !active[row]
            field[row, column] = zero(eltype(field))
        end
    end
    return nothing
end

function TrialProbe(
        execution::KernelExecution, initial::AbstractMatrix{ComplexF64}, z::Float64,
        dz_init::Float64, tableau::Union{EmbeddedTableau, RK4IPTableau}, norm::MarchNorm,
        schedule::MarchSchedule, source, flow::LinearFlow;
        split = nothing, noise::Union{Nothing, InputNoise} = nothing, resume::Bool = false,
    )
    oncpu(execution) || throw(ArgumentError("timing probes currently require CPU"))
    !isempty(initial) && all(isfinite, initial) ||
        throw(ArgumentError("a timing probe needs a finite nonempty initial field"))
    size(initial) == size(flow.exponent) && size(initial, 1) == length(norm.active) ||
        throw(DimensionMismatch("timing-probe field, norm and flow shapes disagree"))
    _retained_alias(initial, (source, flow, split, noise)) &&
        throw(ArgumentError("timing-probe input aliases adopted numerical storage"))
    step = _trial_probe_default(z, dz_init, schedule, tableau)
    owned_norm = MarchNorm(
        execution, norm.active, norm.observed, norm.weights, norm.rtol, norm.atol,
    )
    field = upload(execution, initial)
    if noise === nothing
        _trial_probe_mask!(field, owned_norm.active, execution)
    else
        apply_input_noise!(field, noise, owned_norm.active, z, resume, execution)
    end
    work = tableau isa EmbeddedTableau ? RungeKuttaWork(field, tableau) :
        LowStorageWork(execution, field, tableau; norm_mask = owned_norm.observed)
    state = _TrialProbeState(z, step, z, 0.0, false, false, false, false, 0, 0, 0, NaN)
    # A timing probe never attaches/resumes a monitor. Its source is freshly active,
    # even if a direct Julia caller adopted a previously quieted owned source.
    _source_has_temporal(source) && set_temporal_source!(source, true)
    return TrialProbe(
        tableau, execution, field, work, owned_norm, source, flow, split, noise, state,
        _TrialProbeSource(source, state),
    )
end

function TrialProbe(
        execution, initial, z, dz_init, tableau, norm, schedule, source, flow;
        split = nothing, noise = nothing, resume = false
    )
    throw(
        ArgumentError(
            "timing probes need CPU binary64 state and prepared numerical records",
        )
    )
end

"""
    start_trial_probe(preparation, transfer, field, z, noise=nothing, resume=false,
                      planning=nothing)

Copy a Python start into an independent numerical timing probe through PythonCall.

The optional extension copies the canonical `(n_modes,1,n_omega)` field and resolved
tableau/norm/schedule records. It builds a fresh source and flow, ignores method
continuation and all monitoring/statistics, and uses only the immutable optional
split-rate policy. `planning` is an optional explicit per-preparation FFT context.
The Python owner consumes its Start only after this construction succeeds and roots
the probe until close; no Python wrapper escapes a narrow bridge helper.
"""
function start_trial_probe end

function _trial_probe_open(probe::TrialProbe)
    !probe.state.closed || throw(ArgumentError("the timing probe is closed"))
    !probe.state.failed ||
        throw(ArgumentError("the timing probe failed; construct a new probe"))
    return nothing
end

_trial_probe_calls(probe::TrialProbe{<:EmbeddedTableau}) = length(probe.tableau.nodes)
_trial_probe_calls(probe::TrialProbe{RK4IPTableau}) = 4

function _trial_probe_arithmetic!(probe::TrialProbe{<:EmbeddedTableau}, z, dz)
    rk_trial!(
        probe.work, probe.field, z, dz, probe.tableau,
        probe.sampler, probe.flow, probe.execution,
    )
    norm = probe.norm
    probe.state.error_norm = rk_error_norm!(
        probe.work, probe.field, norm.observed, norm.weights, norm.rtol, norm.atol,
        probe.execution,
    )
    probe.state.norm_evaluations += 1
    probe.state.ready = true
    return nothing
end

function _trial_probe_arithmetic!(probe::TrialProbe{RK4IPTableau}, z, dz)
    probe.state.in_flight = true
    facts = low_storage_trial!(
        probe.work, probe.field, z, dz, probe.tableau,
        probe.sampler, probe.flow, probe.execution,
    )
    probe.state.norm_evaluations += facts.norm_calls
    facts.finite && isfinite(facts.state_norm) ||
        throw(DomainError(facts.state_norm, "low-storage timing probe became nonfinite"))
    probe.state.error_norm = facts.error_norm
    probe.state.in_flight = false
    # The existing probe times the raw engine step on discarded registers. Accepted
    # masks/splitting/position updates belong to the march, which is not entered here.
    return nothing
end

"""
    trial_probe!(probe, z=nothing, dz=nothing)

Execute one trial from the probe's loaded state without a session or acceptance.

Optional finite `z` and positive `dz` are metres; defaults are the loaded anchor and
resolved loaded step. The interval must advance and remain finite. Embedded RK uses
every stage and scores the normal defect, replacing any unread candidate while keeping
the load unchanged. Low-storage RK4IP overwrites discarded registers and has no readable
trial field. Neither family advances the loaded position or touches history/monitors.

Return `nothing`. Invalid arguments/counter capacity refuse before invalidating a ready
trial. A numerical failure retains exact attempted source counts, clears readiness and
latches failure; the caller must close it and construct a new probe. Warmed calls use
only retained buffers and allocate nothing on the qualified CPU path.
"""
function trial_probe!(
        probe::TrialProbe, z::Union{Nothing, Float64} = nothing,
        dz::Union{Nothing, Float64} = nothing,
    )
    _trial_probe_open(probe)
    state = probe.state
    position = z === nothing ? state.loaded_z : z
    step = dz === nothing ? state.default_dz : dz
    isfinite(position) && isfinite(step) && step > 0 &&
        isfinite(position + step) && position + step > position ||
        throw(ArgumentError("a timing probe needs a finite advancing positive interval"))
    if probe.tableau isa EmbeddedTableau
        all(node -> isfinite(position + node * step), probe.tableau.nodes) ||
            throw(ArgumentError("a timing-probe stage position must remain finite"))
    end
    calls = _trial_probe_calls(probe)
    state.source_evaluations <= typemax(Int) - calls &&
        state.norm_evaluations < typemax(Int) && state.trials < typemax(Int) ||
        throw(ArgumentError("timing-probe counters cannot represent another trial"))
    state.ready = false
    state.trial_z, state.trial_dz = position, step
    try
        _trial_probe_arithmetic!(probe, position, step)
        state.trials += 1
    catch
        state.ready, state.failed = false, true
        rethrow()
    end
    return nothing
end

"""
    copy_trial_probe_field!(output, probe)

Finish and copy the last unread embedded trial into caller-owned internal field storage.

`output` is a writable matching complex binary64 matrix. Validate shape, affine layout
and retained-storage/foreign-span alias freedom before any numerical write. Apply the
active mask, optional temporal split and final mask once, then copy without power or
optics conversion. Success consumes the unread trial; the loaded state stays unchanged.
Invalid destinations preserve readiness. Numerical finishing failure poisons the probe.

Return `nothing`. Throw `ArgumentError` for a closed/failed probe, no unread trial,
low-storage family, unsupported destination or alias; incompatible shapes raise
`DimensionMismatch`. The host owner allocates and keeps its own canonical copy.
"""
function copy_trial_probe_field!(output, probe::TrialProbe{<:EmbeddedTableau})
    _trial_probe_open(probe)
    probe.state.ready || throw(ArgumentError("take a trial before reading its field"))
    output isa AbstractMatrix{ComplexF64} ||
        throw(ArgumentError("timing-probe output requires a binary64 complex matrix"))
    size(output) == size(probe.field) ||
        throw(DimensionMismatch("timing-probe output shape differs from its field"))
    _transfer_writable_destination(output)
    _retained_alias(output, probe) &&
        throw(ArgumentError("timing-probe output aliases retained storage"))
    try
        candidate = probe.work.candidate
        _trial_probe_mask!(candidate, probe.norm.active, probe.execution)
        if probe.split !== nothing
            finish_temporal_split!(
                candidate, probe.split, probe.state.trial_dz, probe.execution,
            )
        end
        _source_finite!(probe.flow.failures, candidate, probe.execution, :probe_candidate)
        _trial_probe_mask!(candidate, probe.norm.active, probe.execution)
        copyto!(output, candidate)
        probe.state.ready = false
    catch
        probe.state.ready, probe.state.failed = false, true
        rethrow()
    end
    return nothing
end

function copy_trial_probe_field!(output, probe::TrialProbe{RK4IPTableau})
    _trial_probe_open(probe)
    throw(ArgumentError("the low-storage timing probe has no trial field"))
end

"""
    trial_probe_status(probe)

Return detached timing-probe lifecycle, coordinates and actual work counters.

The pointer-free named tuple contains loaded/trial positions and lengths in metres,
readiness/failure/closure flags, exact cumulative source/norm/completed-trial counts,
and the last dimensionless error norm (NaN before a trial). Reading does not consume
an unread candidate and remains available after failure or close.
"""
function trial_probe_status(probe::TrialProbe)
    state = probe.state
    return (
        loaded_z = state.loaded_z, default_dz = state.default_dz,
        trial_z = state.trial_z, trial_dz = state.trial_dz, ready = state.ready,
        low_storage = probe.tableau isa RK4IPTableau, closed = state.closed,
        failed = state.failed, in_flight = state.in_flight,
        source_evaluations = state.source_evaluations,
        norm_evaluations = state.norm_evaluations,
        trials = state.trials, error_norm = state.error_norm,
    )
end

"""
    close_trial_probe!(probe)

Close a timing probe idempotently before its owning registry releases the buffers.

Clear unread-field eligibility and refuse subsequent arithmetic/readback. Retain
diagnostic status and arrays until the owner drops the record. Return `nothing`.
"""
function close_trial_probe!(probe::TrialProbe)
    probe.state.closed, probe.state.ready = true, false
    return nothing
end

_trial_probe_source_data(source) = nothing
_trial_probe_source_data(source::Union{ModalSource, RadialSource, CartesianSource}) =
    source.data

"""
    trial_probe_memory(probe)

Return owned array bytes by name, excluding shared immutable source/operator preparation.

The common owner traversal counts views/repeated references once, including noise,
split work and numerical norms/stages. FFT vendor plan sizes are explicitly estimates.
Ledger creation may allocate; trial arithmetic and field copying do not build a ledger.
"""
function trial_probe_memory(probe::TrialProbe)
    ledger = Dict{Symbol, Int}()
    seen = IdDict{Any, Bool}()
    _memory_arrays!(ledger, seen, "shared.source", _trial_probe_source_data(probe.source))
    _memory_arrays!(ledger, seen, "shared.operator", probe.flow.operator)
    empty!(ledger)
    _memory_arrays!(ledger, seen, "probe", probe)
    return _memory_record(ledger)
end
