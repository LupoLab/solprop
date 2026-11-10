# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public PIControl, NormHistory, StepFloorError
public record_norm!, replace_last_norm!, control_history, restore_control!
public control_decide, control_start, control_propose, control_reject

# CPython float power uses the platform C library. Julia's default OpenLibm power
# need not make identical scalar control decisions on the same recorded readings.
# The ordinary positive-base path is CPython 3.14's float_pow:
# https://github.com/python/cpython/blob/v3.14.0/Objects/floatobject.c
const _CONTROL_LIBM = Sys.isapple() ? "libSystem.B.dylib" :
    Sys.iswindows() ? "ucrtbase.dll" : "libm.so.6"

@inline function _control_pow(base::Float64, exponent::Float64)
    result = ccall((:pow, _CONTROL_LIBM), Float64, (Float64, Float64), base, exponent)
    if isfinite(base) && isfinite(exponent) && isinf(result)
        throw(OverflowError("a scalar controller power overflowed"))
    end
    return result
end

@inline _control_min(first, second) = second < first ? second : first
@inline _control_max(first, second) = second > first ? second : first

"""
    StepFloorError(proposed, minimum)

Report an adaptive proposal below its permitted step floor, both in metres.

The caller records this convergence failure without changing its last accepted state,
history or proposal. `proposed` and `minimum` are binary64 scalar distances.
"""
struct StepFloorError <: Exception
    proposed::Float64
    minimum::Float64
end

function Base.showerror(io::IO, error::StepFloorError)
    return print(io, "step proposal ", error.proposed, " m is below ", error.minimum, " m")
end

"""
    PIControl(; safety, max_factor, min_factor, beta1, beta2, lag, reject_free,
              estimator_order, start_growth, dz_min, dz_max, auto_init)

Own the translated scalar Runge–Kutta controller policy without choosing defaults.

Gains, factors and `start_growth` are dimensionless; `dz_min` and `dz_max` are in
metres. `lag` is a nonnegative number of accepted steps and `estimator_order` is positive.
All floating values are finite binary64. The tolerance and spectral weights belong to
the numerical norm, not this scalar decision record. The formulas and evaluation order
are the reference `solprop.backends.rk_core` and `solprop.stepper.controller` contract.

# Throws
- `ArgumentError`: For inconsistent factors, gains, step bounds or unsupported platform.
"""
struct PIControl
    safety::Float64
    max_factor::Float64
    min_factor::Float64
    beta1::Float64
    beta2::Float64
    lag::Int
    reject_free::Bool
    estimator_order::Int
    start_growth::Float64
    dz_min::Float64
    dz_max::Float64
    auto_init::Bool

    function PIControl(;
            safety::Float64, max_factor::Float64, min_factor::Float64,
            beta1::Float64, beta2::Float64, lag::Int, reject_free::Bool,
            estimator_order::Int, start_growth::Float64, dz_min::Float64,
            dz_max::Float64, auto_init::Bool,
        )
        if !(Sys.isapple() || Sys.islinux() || Sys.iswindows())
            throw(ArgumentError("this platform has no Julia scalar-control math binding"))
        end
        finite_policy = all(
            isfinite, (
                safety, max_factor, min_factor, beta1, beta2, start_growth, dz_min, dz_max,
            )
        )
        if !finite_policy || !(0.0 < safety <= 1.0) ||
                !(0.0 < min_factor < 1.0 < max_factor) ||
                beta1 <= 0.0 || lag < 0 || estimator_order < 1 || start_growth < 1.0 ||
                !(0.0 <= dz_min <= dz_max) || dz_max == 0.0
            throw(ArgumentError("inconsistent translated PI controller policy"))
        end
        return new(
            safety, max_factor, min_factor, beta1, beta2, lag, reject_free,
            estimator_order, start_growth, dz_min, dz_max, auto_init,
        )
    end
end

"""
    NormHistory(control::PIControl)

Allocate a bounded ring retaining the last `control.lag + 2` accepted error readings.

Entries are dimensionless binary64 equivalent norms, ordered by acceptance. Rejected
attempts do not enter this history. The total accepted count remains exact after wrap.
Snapshots retain `control_history` and `count`; `restore_control!` restores both.
"""
mutable struct NormHistory
    readings::Vector{Float64}
    next::Int
    count::Int
end

NormHistory(control::PIControl) = NormHistory(zeros(control.lag + 2), 1, 0)

"""
    record_norm!(history, norm::Float64)

Append an accepted equivalent norm to bounded history, overwriting its oldest slot.

This allocation-free operation returns `nothing`. Nonfinite readings remain visible:
the scalar controller applies its specified conservative factor when it reads one.
"""
function record_norm!(history::NormHistory, norm::Float64)
    history.readings[history.next] = norm
    history.next = mod1(history.next + 1, length(history.readings))
    history.count += 1
    return nothing
end

"""
    replace_last_norm!(history, norm::Float64)

Replace the newest accepted reading after a successful startup decision.

The replacement allocates nothing and changes neither count nor ring position.
An empty history raises `ArgumentError`; the call returns `nothing` on success.
"""
function replace_last_norm!(history::NormHistory, norm::Float64)
    history.count > 0 || throw(ArgumentError("cannot replace an empty control history"))
    history.readings[mod1(history.next - 1, length(history.readings))] = norm
    return nothing
end

@inline function _control_read(history::NormHistory, age::Int)
    return history.readings[mod1(history.next - 1 - age, length(history.readings))]
end

"""
    control_history(history)

Return an owned vector of retained equivalent norms, oldest first, for a snapshot.

This export may allocate and is not called in warmed numerical advancement. Its
length is bounded by the history capacity even after many accepted steps.
"""
function control_history(history::NormHistory)
    held = min(history.count, length(history.readings))
    return [_control_read(history, age) for age in (held - 1):-1:0]
end

"""
    restore_control!(history, readings, total::Int)

Restore retained accepted norms and their original total count from a snapshot.

`readings` is a binary64 vector ordered oldest first. Its length must equal the lesser
of the ring capacity and the nonnegative total accepted count. Validation precedes
mutation; restoration may allocate an owned copy and returns `nothing`. Keeping `total`
separate preserves exact accounting after the bounded ring has wrapped.
"""
function restore_control!(
        history::NormHistory, readings::AbstractVector{Float64}, total::Int,
    )
    total >= 0 && length(readings) == min(total, length(history.readings)) ||
        throw(ArgumentError("retained control readings disagree with the total count"))
    owned = Vector{Float64}(undef, length(readings))
    for (destination, index) in enumerate(axes(readings, 1))
        owned[destination] = readings[index]
    end
    fill!(history.readings, 0.0)
    copyto!(history.readings, 1, owned, 1, length(owned))
    history.next = mod1(length(owned) + 1, length(history.readings))
    history.count = total
    return nothing
end

function _control_equivalent(control::PIControl, norm, taken, proposal)
    if !(isfinite(taken) && taken > 0.0 && isfinite(proposal) && proposal > 0.0)
        throw(ArgumentError("controller step lengths must be finite and positive"))
    end
    isfinite(norm) || return norm
    return norm * _control_pow(proposal / taken, Float64(control.estimator_order + 1))
end

"""
    control_decide(control, norm, taken, proposal)

Return the equivalent error norm and acceptance of an executed Runge–Kutta trial.

`norm` is dimensionless; positive `taken` and `proposal` are in metres. Acceptance
uses the executed norm. History learns the norm rescaled to the unclipped proposal.
A nonfinite norm is rejected even under the translated `reject_free` policy.
"""
function control_decide(
        control::PIControl, norm::Float64, taken::Float64, proposal::Float64,
    )
    equivalent = _control_equivalent(control, norm, taken, proposal)
    return equivalent, isfinite(norm) && (control.reject_free || norm <= 1.0)
end

function _control_limit(control::PIControl, proposal, factor)
    following = _control_min(proposal * factor, control.dz_max)
    if following < control.dz_min
        throw(StepFloorError(following, control.dz_min))
    end
    return following
end

function _control_deadbeat(control::PIControl, norm, maximum)
    if !isfinite(norm) || norm <= 0.0
        return norm == 0.0 ? maximum : control.min_factor
    end
    raw = control.safety * _control_pow(1.0 / norm, 1.0 / (control.estimator_order + 1))
    return _control_min(_control_max(raw, control.min_factor), maximum)
end

"""
    control_start(control, proposal, norm)

Return a startup proposal and replacement reading after the first accepted step.

The caller applies this only when `control.auto_init` is true. The returned proposal
is in metres, its replacement norm dimensionless. A floor failure raises `StepFloorError`
before either value is installed in mutable session state.
"""
function control_start(control::PIControl, proposal::Float64, norm::Float64)
    factor = _control_deadbeat(control, norm, control.start_growth)
    following = _control_limit(control, proposal, factor)
    recorded = factor >= control.start_growth ? 0.0 :
        _control_equivalent(control, norm, proposal, following)
    return following, recorded
end

"""
    control_propose(control, proposal, history)

Return the next proposal after an accepted step, reading its completed lagged history.

The proposal is in metres. The ring must have the capacity `control.lag + 2`.
Until its lag has filled, the proposal remains unchanged except for its maximum cap.
A step below the configured floor raises `StepFloorError` without mutating history.
"""
function control_propose(control::PIControl, proposal::Float64, history::NormHistory)
    if length(history.readings) != control.lag + 2
        throw(ArgumentError("control history capacity does not match the policy lag"))
    end
    if history.count <= control.lag
        return _control_limit(control, proposal, 1.0)
    end
    current = _control_read(history, control.lag)
    factor = if !isfinite(current)
        control.min_factor
    elseif current <= 0.0
        control.max_factor
    else
        previous = history.count > control.lag + 1 ?
            _control_read(history, control.lag + 1) : 1.0
        if !isfinite(previous) || previous <= 0.0
            previous = 1.0
        end
        raw = control.safety * _control_pow(1.0 / current, control.beta1) *
            _control_pow(1.0 / previous, control.beta2)
        _control_min(_control_max(raw, control.min_factor), control.max_factor)
    end
    return _control_limit(control, proposal, factor)
end

"""
    control_reject(control, proposal, equivalent)

Return the shrunk proposal after rejection, using the rejected equivalent norm.

The proposal is in metres. A rejection never grows it, and a nonfinite reading uses
the maximum configured shrink. A below-floor result raises `StepFloorError` before
the caller changes its proposal. Accepted history remains unchanged.
"""
function control_reject(control::PIControl, proposal::Float64, equivalent::Float64)
    factor = if !isfinite(equivalent)
        control.min_factor
    elseif equivalent <= 0.0
        1.0
    else
        _control_deadbeat(control, equivalent, 1.0)
    end
    return _control_limit(control, proposal, factor)
end
