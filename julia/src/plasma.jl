# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public IonisationRateTable, PlasmaChannel, RunningIntegralRule, FieldPlasmaWorkspace
public PlasmaCode, PlasmaSuccess, PlasmaNonfiniteInput, PlasmaNonfiniteResult
public PlasmaQuadratureFailure, PlasmaFailure, plasma_failure, field_plasma!
public CycleQuadrature, EnvelopePlasmaWorkspace, envelope_plasma!
public PopulationWorkspace, field_populations!, envelope_populations!

# Only preparation uses this token after revalidating fresh host-owned data and
# uploading an exact copy. Device copies are checked by shape/type, never reduced
# or scalar-indexed from the host to repeat the already completed validation.
struct _PlasmaValidatedCopy end

"""
    IonisationRateTable(coefficients, field_min, field_max, rate_max,
                        log_field_min, d_log_field)

Retain a translated instantaneous ionisation-rate spline with explicit clipping.

# Arguments
- `coefficients`: Finite real `(4, n_intervals)` matrix of cubic coefficients for
  `log(W)` versus `log(abs(E))`, highest power first; caller-owned prepared storage.
- `field_min`, `field_max`: Finite positive increasing field bounds, V/m. The rate
  is exactly zero below the lower bound and exactly `rate_max` at/above the upper bound.
- `rate_max`: Finite nonnegative rate ceiling in s⁻¹.
- `log_field_min`, `d_log_field`: Finite logarithmic knot origin and positive spacing.

# Returns
An immutable numerical table. No rate model, species object or Python value is retained.
Scalar clipping/log-grid metadata remains binary64 at preparation. Each call shifts
the clipping coordinates to the stored field scale. Binary32 logarithmic coordinates
use local compensated binary32 words to avoid subtracting rounded absolute logarithms;
the coefficient matrix and final spline offset use response precision.

# Throws
- `ArgumentError`: If shape, finiteness, clipping bounds or spacing are invalid.
"""
struct IonisationRateTable{T <: AbstractFloat, A <: AbstractMatrix{T}}
    coefficients::A
    field_min::Float64
    field_max::Float64
    rate_max::Float64
    log_field_min::Float64
    d_log_field::Float64

    function IonisationRateTable(
            coefficients::A, field_min::Real, field_max::Real, rate_max::Real,
            log_field_min::Real, d_log_field::Real,
        ) where {T <: AbstractFloat, A <: AbstractMatrix{T}}
        size(coefficients, 1) == 4 && size(coefficients, 2) > 0 &&
            all(isfinite, coefficients) ||
            throw(ArgumentError("rate spline needs four finite coefficients per interval"))
        all(isfinite, (field_min, field_max, rate_max, log_field_min, d_log_field)) &&
            0 < field_min < field_max && rate_max >= 0 && d_log_field > 0 ||
            throw(ArgumentError("invalid ionisation table bounds or logarithmic grid"))
        scalars = map(Float64, (field_min, field_max, rate_max, log_field_min, d_log_field))
        all(isfinite, scalars) ||
            throw(ArgumentError("ionisation table host scalars must fit binary64"))
        return new{T, A}(coefficients, scalars...)
    end

    function IonisationRateTable(
            ::_PlasmaValidatedCopy, validated::IonisationRateTable{T}, coefficients::A,
        ) where {T <: AbstractFloat, A <: AbstractMatrix{T}}
        size(coefficients) == size(validated.coefficients) ||
            throw(DimensionMismatch("uploaded rate coefficient shape differs"))
        return new{T, A}(
            coefficients, validated.field_min, validated.field_max, validated.rate_max,
            validated.log_field_min, validated.d_log_field,
        )
    end
end

IonisationRateTable(::_PlasmaValidatedCopy, ::IonisationRateTable, ::AbstractArray) =
    throw(ArgumentError("uploaded rate coefficients must preserve type and rank"))

"""
    PlasmaChannel(rate, charge_mass_ratio, ionisation_potential)

Bind one independent electron population to its translated rate and material constants.

# Arguments
- `rate`: `IonisationRateTable` at the response arithmetic precision.
- `charge_mass_ratio`: Finite positive translated `e²/m_e`, in C²/kg.
- `ionisation_potential`: Finite positive energy per electron freed, in joules.

# Returns
An immutable channel with binary64 host material constants. Stage density (m⁻³)
and preionisation (dimensionless) remain
independent inputs to every evaluation; a mixture evaluates each channel separately.

# Throws
- `ArgumentError`: If either material constant is nonfinite or nonpositive.
"""
struct PlasmaChannel{T <: AbstractFloat, R}
    rate::R
    charge_mass_ratio::Float64
    ionisation_potential::Float64

    function PlasmaChannel(
            rate::IonisationRateTable{T}, charge_mass_ratio::Real,
            ionisation_potential::Real,
        ) where {T}
        constants = (charge_mass_ratio, ionisation_potential)
        all(value -> isfinite(value) && value > 0, constants) ||
            throw(ArgumentError("plasma material constants must be finite and positive"))
        charge, potential = Float64(charge_mass_ratio), Float64(ionisation_potential)
        isfinite(charge) && isfinite(potential) && charge > 0 && potential > 0 ||
            throw(ArgumentError("plasma material constants must fit positive binary64"))
        return new{T, typeof(rate)}(rate, charge, potential)
    end
end

"""
    RunningIntegralRule(taps, dt; floor_exposure=false)

Retain the translated symmetric FIR and running trapezoid for field-grid plasma.

# Arguments
- `taps`: Finite nonempty vector `(c_0, ..., c_K)`, centre first. The trapezoid
  uses `[1]`; a certified band rule supplies its own taps without refitting here.
- `dt`: Finite positive sample spacing in seconds, matching the translated rule.
- `floor_exposure`: Apply `max(exposure, 0)` to populations, as the band rule requires.
  This is a pointwise floor, not a running maximum, and does not alter stored exposure.

# Returns
An immutable rule with point-reflected endpoints: `y[1-k] = 2y[1] - y[1+k]`.

# Throws
- `ArgumentError`: If taps or sample spacing are invalid.
"""
struct RunningIntegralRule{T <: AbstractFloat, A <: AbstractVector{T}}
    taps::A
    dt::T
    floor_exposure::Bool

    function RunningIntegralRule(
            taps::A, dt::T; floor_exposure::Bool = false,
        ) where {T <: AbstractFloat, A <: AbstractVector{T}}
        !isempty(taps) && all(isfinite, taps) && isfinite(dt) && dt > 0 ||
            throw(ArgumentError("running integral requires finite taps and positive dt"))
        return new{T, A}(taps, dt, floor_exposure)
    end

    function RunningIntegralRule(
            ::_PlasmaValidatedCopy, validated::RunningIntegralRule{T}, taps::A,
        ) where {T <: AbstractFloat, A <: AbstractVector{T}}
        size(taps) == size(validated.taps) ||
            throw(DimensionMismatch("uploaded integral tap shape differs"))
        return new{T, A}(taps, validated.dt, validated.floor_exposure)
    end
end

RunningIntegralRule(::_PlasmaValidatedCopy, ::RunningIntegralRule, ::AbstractArray) =
    throw(ArgumentError("uploaded integral taps must preserve type and rank"))

"""Identify a numerical plasma result without throwing inside a scalar kernel."""
@enum PlasmaCode::UInt8 begin
    PlasmaSuccess = 0
    PlasmaNonfiniteInput = 1
    PlasmaNonfiniteResult = 2
    PlasmaQuadratureFailure = 3
end

"""Identify a finite, successfully evaluated plasma sample."""
PlasmaSuccess

"""Identify a nonfinite field sample before plasma arithmetic begins."""
PlasmaNonfiniteInput

"""Identify nonfinite plasma arithmetic, including accumulation into earlier channels."""
PlasmaNonfiniteResult

"""Identify scalar cycle moments that exhaust the supplied refinement rules."""
PlasmaQuadratureFailure

"""
    PlasmaFailure(code, node, sample)

Describe the first failing plasma sample in node-major order.

`code` is a `PlasmaCode`; `node` and `sample` are one-based indices. A successful
result has `PlasmaSuccess` and both indices zero. This immutable value owns no wrapper
or exception; the session converts a failure into its named numerical error policy.
"""
struct PlasmaFailure
    code::PlasmaCode
    node::Int
    sample::Int
end

"""
    plasma_failure(status)

Read a CPU status matrix and return the first failure in node-major order.

# Arguments
- `status`: CPU-readable `UInt8` matrix `(n_time, n_nodes)` written by a plasma kernel.
  A device caller must first transfer its status through its qualified host boundary.

# Returns
A `PlasmaFailure`; successful matrices return code `PlasmaSuccess` and zero indices.
No allocation or Julia exception construction occurs on this reporting path.
"""
function plasma_failure(status::AbstractMatrix{UInt8})
    for node in axes(status, 2), sample in axes(status, 1)
        code = status[sample, node]
        iszero(code) || return PlasmaFailure(PlasmaCode(code), node, sample)
    end
    return PlasmaFailure(PlasmaSuccess, 0, 0)
end

"""
    FieldPlasmaWorkspace(execution, T, layout)

Allocate independently owned work buffers for one scalar field-plasma channel at a time.

# Arguments
- `execution`: Numerical backend used only for preparation allocations.
- `T`: Response floating-point scalar type; CPU binary64 is the admitted initial path.
- `layout`: Scalar `ResponseLayout`, including inactive node capacity.

# Returns
A workspace owning seven real `(n_time, n_nodes)` matrices: `rate` (s⁻¹), raw
`exposure` (dimensionless), `fraction` and `survival` (dimensionless), `integrand`
(current acceleration per unit number density), `current` (A m per atom), and
`filtered` (the current integral's FIR scratch). `status` is a `UInt8` matrix of
the same shape. All retained buffers are explicit for the session's memory ledger.

# Throws
- `ArgumentError`: If the layout requests unsupported vector plasma.
"""
struct FieldPlasmaWorkspace{T, A <: AbstractMatrix{T}, U <: AbstractMatrix{UInt8}}
    layout::ResponseLayout
    rate::A
    exposure::A
    fraction::A
    survival::A
    integrand::A
    current::A
    filtered::A
    status::U
end

function FieldPlasmaWorkspace(
        execution, ::Type{T}, layout::ResponseLayout,
    ) where {T <: AbstractFloat}
    layout.n_components == 1 ||
        throw(ArgumentError("Julia field plasma currently requires scalar polarisation"))
    dimensions = (layout.n_time, layout.n_nodes)
    return FieldPlasmaWorkspace(
        layout, ntuple(_ -> dzeros(execution, T, dimensions...), 7)...,
        dzeros(execution, UInt8, dimensions...),
    )
end

@inline function _rate_parameters(table::IonisationRateTable{Float64}, exponent::Int = 0)
    iszero(exponent) || return _scaled_rate_parameters(table, exponent)
    return (
        table.coefficients, table.field_min, table.field_max, table.rate_max,
        table.log_field_min, table.d_log_field,
    )
end

struct _PlasmaLogCoordinate
    origin::DF32
    spacing::DF32
    logtwo::DF32
    exponent::Int
end

function _plasma_log_pair(value::Float64)
    high = Float32(value)
    low = Float32(value - Float64(high))
    isfinite(high) && isfinite(low) ||
        throw(ArgumentError("plasma reduced logarithmic coordinate overflows"))
    # This source-local split retains gradual underflow. Its absolute conversion
    # allowance is u^2*abs(value)+eta, unlike prepare_df32's normal-only admission.
    high, low = _df32_fast_two_sum(high, low)
    isfinite(high) && isfinite(low) ||
        throw(ArgumentError("plasma reduced logarithmic coordinate overflows"))
    return _df32_raw(high, low)
end

function _rate_parameters(table::IonisationRateTable{Float32}, exponent::Int = 0)
    parameters = _scaled_rate_parameters(table, exponent)
    # The clipping field provides a bounded integer anchor even for an unusual
    # translated log origin; no logarithm-to-Int conversion is required.
    anchor = last(frexp(table.field_min))
    logtwo = log(2.0)
    reduced = _PlasmaLogCoordinate(
        _plasma_log_pair(fma(-Float64(anchor), logtwo, table.log_field_min)),
        _plasma_log_pair(table.d_log_field), _plasma_log_pair(logtwo), exponent - anchor,
    )
    return parameters..., reduced
end

function _scaled_rate_parameters(table::IonisationRateTable{T}, exponent::Int) where {T}
    -1074 <= exponent <= 1023 ||
        throw(ArgumentError("plasma field exponent must be a finite binary64 power"))
    arithmetic = SourceArithmetic(T)
    lower = source_coefficient(
        arithmetic, :plasma_lower_field, table.field_min;
        exponent = -exponent
    )
    upper = source_coefficient(
        arithmetic, :plasma_upper_field, table.field_max;
        exponent = -exponent
    )
    # Ceil the clipping endpoints: comparisons against representable stored real
    # magnitudes then retain the original threshold side after an exact power scale.
    lower_exact = ldexp(table.field_min, -exponent)
    upper_exact = ldexp(table.field_max, -exponent)
    lower = lower < lower_exact ? nextfloat(lower) : lower
    upper = upper < upper_exact ? nextfloat(upper) : upper
    lower < upper && isfinite(upper) || throw(
        ArgumentError("plasma scaled clipping interval collapses at source precision")
    )
    cap = source_coefficient(arithmetic, :plasma_rate_ceiling, table.rate_max)
    origin = T(table.log_field_min - exponent * log(2.0))
    spacing = source_coefficient(arithmetic, :plasma_log_spacing, table.d_log_field)
    isfinite(origin) || throw(ArgumentError("plasma scaled logarithmic origin overflows"))
    return table.coefficients, lower, upper, cap, origin, spacing
end

function _plasma_coefficients(channel::PlasmaChannel{T}, density, exponent::Int) where {T}
    if T === Float64 && iszero(exponent)
        return channel.charge_mass_ratio, channel.ionisation_potential, T(density)
    end
    arithmetic = SourceArithmetic(T)
    charge = source_coefficient(
        arithmetic, :plasma_free_current,
        density, channel.charge_mass_ratio
    )
    loss = source_coefficient(
        arithmetic, :plasma_ionisation_loss,
        density, channel.ionisation_potential; exponent = -2exponent
    )
    return charge, loss, one(T)
end

@inline function _ionisation_rate(magnitude, parameters)
    coefficients, lower, upper, cap, origin, spacing = parameters
    magnitude < lower && return zero(cap)
    magnitude >= upper && return cap
    logarithm = log(magnitude)
    coordinate = floor((logarithm - origin) / spacing)
    interval = Int(clamp(coordinate, zero(coordinate), size(coefficients, 2) - 1)) + 1
    offset = logarithm - (origin + (interval - 1) * spacing)
    polynomial = muladd(
        muladd(
            muladd(coefficients[1, interval], offset, coefficients[2, interval]),
            offset, coefficients[3, interval]
        ), offset, coefficients[4, interval]
    )
    return min(exp(polynomial), cap)
end

@inline function _plasma_log_scale(value::DF32, shift::Int)
    high, low = _df32_fast_two_sum(ldexp(value.hi, shift), ldexp(value.lo, shift))
    return _df32_raw(high, low)
end

@inline function _plasma_log_offset(
        magnitude::Float32, reduced::_PlasmaLogCoordinate, intervals::Int,
    )
    mantissa, exponent = frexp(magnitude)
    if mantissa < sqrt(0.5f0)
        mantissa *= 2.0f0
        exponent -= 1
    end
    # mantissa-1 is exact by Sterbenz. The logarithm's argument stays near one;
    # its rounded absolute value no longer contains the physical field exponent.
    logarithm = _df32_raw(log1p(mantissa - 1.0f0), 0.0f0)
    power = _df32_raw(Float32(exponent + reduced.exponent), 0.0f0)
    coordinate = logarithm + power * reduced.logtwo - reduced.origin
    estimate = coordinate.hi / reduced.spacing.hi
    interval = if estimate <= 0.0f0
        1
    elseif estimate >= Float32(intervals)
        intervals
    else
        # Exact binary scaling puts the divisor in the reviewed [8, 2^42]
        # reciprocal domain. Scaling a low word can only add gradual-underflow
        # error; it cannot overflow while this guarded quotient is below the
        # number of stored spline intervals.
        shift = 4 - last(frexp(reduced.spacing.hi))
        numerator = _plasma_log_scale(coordinate, shift)
        denominator = _plasma_log_scale(reduced.spacing, shift)
        fraction = numerator / denominator
        isfinite(fraction) || return 1, NaN32
        leading = floor(Int, fraction.hi)
        trailing = floor(Int, (fraction.hi - Float32(leading)) + fraction.lo)
        clamp(leading + trailing, 0, intervals - 1) + 1
    end
    # Integer knot indices may exceed binary32's exact-integer range. Split the
    # index before multiplying rather than dropping its low bits implicitly.
    index = interval - 1
    high = Float32(index)
    low = Float32(index - Int(high))
    offset = coordinate - _df32_raw(high, low) * reduced.spacing
    return interval, offset.hi
end

@inline function _ionisation_rate(
        magnitude::Float32,
        parameters::Tuple{
            A, Float32, Float32, Float32, Float32, Float32, _PlasmaLogCoordinate,
        },
    ) where {A <: AbstractMatrix{Float32}}
    coefficients, lower, upper, cap = parameters[1:4]
    magnitude < lower && return 0.0f0
    magnitude >= upper && return cap
    interval, offset = _plasma_log_offset(magnitude, parameters[7], size(coefficients, 2))
    polynomial = muladd(
        muladd(
            muladd(coefficients[1, interval], offset, coefficients[2, interval]),
            offset, coefficients[3, interval]
        ), offset, coefficients[4, interval]
    )
    return min(exp(polynomial), cap)
end

@inline function _integrate_plasma_column!(output, filtered, input, node, taps, dt)
    n_time = size(input, 1)
    # Write the entire FIR footprint before any scan reads it. In particular, the
    # zero after a compactly supported input is not allowed to retain old scratch
    # (the donor failure reproduced in issue #622).
    for row in 1:n_time
        value = taps[1] * input[row, node]
        for offset in 1:(length(taps) - 1)
            left = row - offset < 1 ?
                2input[1, node] - input[2 - row + offset, node] : input[row - offset, node]
            right = row + offset > n_time ?
                2input[n_time, node] - input[2n_time - row - offset, node] :
                input[row + offset, node]
            value = muladd(taps[offset + 1], left + right, value)
        end
        filtered[row, node] = value
    end
    if eltype(output) === Float32
        output[1, node] = 0
        for row in 2:n_time
            output[row, node] = (filtered[row - 1, node] + filtered[row, node]) * (dt / 2)
        end
        _plasma_prefix!(output, node)
        return nothing
    end
    total = zero(dt)
    output[1, node] = total
    for row in 2:n_time
        total += (filtered[row - 1, node] + filtered[row, node]) * (dt / 2)
        output[row, node] = total
    end
    return nothing
end

@inline function _plasma_prefix!(output, node)
    # Reverse traversal retains the previous level without extra storage. Each
    # prefix has at most ceil(log2(n)) rounded additions, matching the protected
    # Hillis--Steele graph of translate.precision.prefix_sum, not sequential cumsum.
    stride, count = 1, size(output, 1)
    while stride < count
        for row in count:-1:(stride + 1)
            output[row, node] += output[row - stride, node]
        end
        stride > count ÷ 2 && break
        stride *= 2
    end
    return nothing
end

@inline function _envelope_exposure!(exposure, rate, node, spacing)
    T = eltype(exposure)
    exposure[1, node] = zero(T)
    total = zero(T)
    for row in 2:size(rate, 1)
        increment = (rate[row - 1, node] / 2 + rate[row, node] / 2) * spacing
        total = T === Float32 ? increment : total + increment
        exposure[row, node] = total
    end
    T === Float32 && _plasma_prefix!(exposure, node)
    return nothing
end

@inline function _field_plasma_buffers(workspace)
    return (
        rate = workspace.rate, exposure = workspace.exposure,
        fraction = workspace.fraction, survival = workspace.survival,
        integrand = workspace.integrand, current = workspace.current,
        filtered = workspace.filtered, status = workspace.status,
    )
end

@inline function _clear_field_plasma!(buffers, gated, edge, node)
    for row in axes(gated, 1)
        buffers.rate[row, node] = 0
        buffers.exposure[row, node] = 0
        buffers.fraction[row, node] = 0
        buffers.survival[row, node] = 0
        buffers.integrand[row, node] = 0
        buffers.current[row, node] = 0
        buffers.filtered[row, node] = 0
        gated[row, node] = 0
        edge[row, node] = 0
    end
    return nothing
end

function _plasma_stage(layout, density, initial, count, window, derivative = nothing)
    _response_count(layout, count)
    _response_window(window, layout)
    _response_window(derivative, layout)
    isfinite(density) && density >= 0 ||
        throw(ArgumentError("plasma stage density must be finite and nonnegative"))
    isfinite(initial) && 0 <= initial <= 1 ||
        throw(ArgumentError("plasma preionisation must lie in [0, 1]"))
    return nothing
end

"""
    field_plasma!(gated, edge, samples, channel, rule, workspace, execution;
                  density, preionisation, window=nothing, derivative=nothing,
                  count=workspace.layout.n_nodes, add=false, field_exponent=0)

Evaluate one independent field-plasma population and its two gated current streams.

# Arguments
- `gated`, `edge`: Caller-owned real matrices with the workspace layout. Receive
  `window * J` in A/m² and `window_derivative * J` in A/(m² s), respectively.
- `samples`: Stored real field, with the same `(n_time, n_nodes)` layout. Its
  physical value in V/m is `samples * 2^field_exponent`.
- `channel`: Prepared `PlasmaChannel`; constants have not absorbed any stage profile.
- `rule`: Prepared `RunningIntegralRule`. Its FIR radius must be smaller than `n_time`.
- `workspace`: `FieldPlasmaWorkspace`, reusable between channels and calls.
- `execution`: Synchronous numerical launcher; one sequential column body per node.
- `density`: This channel's finite nonnegative stage number density, m⁻³.
- `preionisation`: This channel's initial fraction in `[0, 1]`.
- `window`, `derivative`: Optional dimensionless source gate and its time derivative
  in s⁻¹, each length `n_time`. Defaults are one and zero, respectively.
- `count`: Active node count. Inactive work and output columns are fully zeroed.
- `add`: Add this channel's active streams to an earlier channel's streams.
- `field_exponent`: Integer physical field scale. Outputs are the physical currents
  divided by the same power. Rate lookup shifts the table coordinates on the host;
  charge/density and loss/density/inverse-square powers combine before one rounding.

# Returns
`nothing`. Read `plasma_failure(workspace.status)` before consuming the output.
Nonfinite input/result is recorded explicitly and never silently propagated.
Warm CPU calls allocate nothing; inputs and all output/work arrays must be disjoint.

# Throws
- `ArgumentError`: If stage values, scalar layout or integration support are invalid.
- `DimensionMismatch`: If input/output shapes disagree with the workspace.

# Notes
The independent-atom survival is `(1-f0)exp(-S)`, with `S = integral(W dt)`;
`f = f0 + (1-f0)(-expm1(-S))` avoids cancellation at small exposure. The free current
per atom is `(e²/m_e) integral(f E dt)`. Its loss current is
`I_p W survival E/|E|²`, so `E*J_loss = I_p W survival`. Density is applied once to
their sum. Keeping survival separately avoids subtracting a nearly unit population.
"""
function field_plasma!(
        gated, edge, samples, channel::PlasmaChannel{T}, rule::RunningIntegralRule{T},
        workspace::FieldPlasmaWorkspace{T}, execution;
        density, preionisation, window = nothing, derivative = nothing,
        count::Int = workspace.layout.n_nodes, add::Bool = false,
        field_exponent::Int = 0,
    ) where {T}
    layout = workspace.layout
    _response_matrix(gated, layout)
    _response_matrix(edge, layout)
    _response_matrix(samples, layout)
    eltype(samples) <: Real || throw(ArgumentError("field plasma requires real samples"))
    length(rule.taps) <= layout.n_time ||
        throw(ArgumentError("plasma FIR support exceeds the temporal grid"))
    _plasma_stage(layout, density, preionisation, count, window, derivative)
    parameters = _rate_parameters(channel.rate, field_exponent)
    buffers = _field_plasma_buffers(workspace)
    charge, potential, stage_density =
        _plasma_coefficients(channel, density, field_exponent)
    taps, dt, floor_exposure = rule.taps, rule.dt, rule.floor_exposure
    initial, keep = T(preionisation), T(1 - Float64(preionisation))
    each_column!(execution, layout.n_nodes) do node
        invalid = false
        for row in axes(samples, 1)
            sample = node <= count ? samples[row, node] : zero(T)
            code = isfinite(sample) ? PlasmaSuccess : PlasmaNonfiniteInput
            rate = code == PlasmaSuccess ?
                _ionisation_rate(abs(sample), parameters) : zero(T)
            if !isfinite(rate)
                code = PlasmaNonfiniteResult
                rate = zero(T)
            end
            buffers.rate[row, node] = rate
            buffers.status[row, node] = UInt8(code)
            invalid |= code != PlasmaSuccess
        end
        if node > count || invalid
            _clear_field_plasma!(buffers, gated, edge, node)
            return nothing
        end
        _integrate_plasma_column!(
            buffers.exposure, buffers.filtered, buffers.rate, node, taps, dt
        )
        for row in axes(samples, 1)
            if !(
                    isfinite(buffers.exposure[row, node]) &&
                        isfinite(buffers.filtered[row, node])
                )
                buffers.status[row, node] = UInt8(PlasmaNonfiniteResult)
                invalid = true
                continue
            end
            exposure = floor_exposure ? max(buffers.exposure[row, node], zero(T)) :
                buffers.exposure[row, node]
            survival = keep * exp(-exposure)
            fraction = initial + keep * -expm1(-exposure)
            buffers.survival[row, node] = survival
            buffers.fraction[row, node] = fraction
            buffers.integrand[row, node] = (charge * fraction) * samples[row, node]
        end
        if invalid
            _clear_field_plasma!(buffers, gated, edge, node)
            return nothing
        end
        _integrate_plasma_column!(
            buffers.current, buffers.filtered, buffers.integrand, node, taps, dt
        )
        for row in axes(samples, 1)
            sample, rate = samples[row, node], buffers.rate[row, node]
            squared = sample * sample
            loss = rate > 0 && squared > 0 ?
                (potential * rate * buffers.survival[row, node] / squared) * sample :
                zero(T)
            total = stage_density * (buffers.current[row, node] + loss)
            value = _response_weight(window, row, total) * total
            correction = derivative === nothing ? zero(T) : derivative[row] * total
            value = add ? gated[row, node] + value : value
            correction = add ? edge[row, node] + correction : correction
            if !(isfinite(value) && isfinite(correction))
                buffers.status[row, node] = UInt8(PlasmaNonfiniteResult)
                value, correction = zero(T), zero(T)
            end
            gated[row, node] = value
            edge[row, node] = correction
        end
        return nothing
    end
    return nothing
end

"""
    CycleQuadrature(nodes, weights, ends; moment_rtol, negligible_rate)

Retain the translated nested refinement rules for scalar envelope-plasma moments.

# Arguments
- `nodes`, `weights`: Finite real vectors containing each Gauss-Legendre rule on
  `[-1, 1]` in succession. Nodes are strictly inside the interval and weights positive.
- `ends`: Tuple of increasing integer end offsets, one per rule; at least two rules.
  Production lowering supplies the accepted orders 64, 128, 256 and 512.
- `moment_rtol`: Positive finite relative refinement budget from `CycleIntegralData`.
- `negligible_rate`: Positive finite denominator floor in s⁻¹ from the same descriptor.

# Returns
An immutable numerical quadrature record. Both positive directional moments must
meet the supplied refinement budget; the scalar model does not discard the sine test.

# Throws
- `ArgumentError`: If the rule dimensions, nodes, weights or budgets are invalid.
"""
struct CycleQuadrature{T <: AbstractFloat, A <: AbstractVector{T}, E}
    nodes::A
    weights::A
    ends::E
    moment_rtol::T
    negligible_rate::T

    function CycleQuadrature(
            nodes::A, weights::A, ends::NTuple{N, Int};
            moment_rtol::T, negligible_rate::T,
        ) where {T <: AbstractFloat, A <: AbstractVector{T}, N}
        N >= 2 && length(nodes) == length(weights) && ends[end] == length(nodes) ||
            throw(ArgumentError("cycle quadrature requires at least two complete rules"))
        all(value -> isfinite(value) && abs(value) < 1, nodes) &&
            all(value -> isfinite(value) && value > 0, weights) ||
            throw(ArgumentError("invalid cycle quadrature nodes or weights"))
        all(value -> isfinite(value) && value > 0, (moment_rtol, negligible_rate)) ||
            throw(ArgumentError("cycle quadrature budgets must be finite and positive"))
        first = 1
        for last in ends
            first + 1 <= last <= length(nodes) ||
                throw(ArgumentError("cycle rules need increasing valid end offsets"))
            total = zero(T)
            for index in first:last
                index == first || nodes[index] > nodes[index - 1] ||
                    throw(ArgumentError("cycle rule nodes must be increasing"))
                total += weights[index]
            end
            abs(total - 2) <= 32eps(T) ||
                throw(ArgumentError("cycle rule weights must integrate a constant"))
            first = last + 1
        end
        return new{T, A, typeof(ends)}(nodes, weights, ends, moment_rtol, negligible_rate)
    end

    function CycleQuadrature(
            ::_PlasmaValidatedCopy, validated::CycleQuadrature{T}, nodes::A, weights::A,
        ) where {T <: AbstractFloat, A <: AbstractVector{T}}
        size(nodes) == size(validated.nodes) && size(weights) == size(validated.weights) ||
            throw(DimensionMismatch("uploaded cycle-rule shapes differ"))
        return new{T, A, typeof(validated.ends)}(
            nodes, weights, validated.ends,
            validated.moment_rtol, validated.negligible_rate,
        )
    end
end

CycleQuadrature(
    ::_PlasmaValidatedCopy, ::CycleQuadrature, ::AbstractArray, ::AbstractArray
) =
    throw(ArgumentError("uploaded cycle arrays must preserve type and rank"))

"""
    EnvelopePlasmaWorkspace(execution, T, layout)

Allocate owned scalar-envelope rate, population, loss-moment and status storage.

# Arguments
- `execution`: Numerical backend used for preparation allocations.
- `T`: Real response scalar type; the initial admitted path is CPU binary64.
- `layout`: Scalar `ResponseLayout`, including inactive node capacity.

# Returns
A workspace owning four real `(n_time, n_nodes)` matrices: `rate` (s⁻¹), `exposure`,
`fraction` and `survival` (dimensionless); complex `loss_moment` in m/(V s), and
`UInt8` `status` of the same shape. All buffers are reused between channels/stages.

# Throws
- `ArgumentError`: If the layout requests unsupported vector/elliptical plasma.
"""
struct EnvelopePlasmaWorkspace{T, A <: AbstractMatrix{T}, C, U <: AbstractMatrix{UInt8}}
    layout::ResponseLayout
    rate::A
    exposure::A
    fraction::A
    survival::A
    loss_moment::C
    status::U
end

function EnvelopePlasmaWorkspace(
        execution, ::Type{T}, layout::ResponseLayout,
    ) where {T <: AbstractFloat}
    layout.n_components == 1 ||
        throw(ArgumentError("Julia envelope plasma currently requires scalar polarisation"))
    dimensions = (layout.n_time, layout.n_nodes)
    return EnvelopePlasmaWorkspace(
        layout, ntuple(_ -> dzeros(execution, T, dimensions...), 4)...,
        dzeros(execution, Complex{T}, dimensions...),
        dzeros(execution, UInt8, dimensions...),
    )
end

@inline function _cycle_parameters(quadrature)
    return (
        quadrature.nodes, quadrature.weights, quadrature.ends,
        quadrature.moment_rtol, quadrature.negligible_rate,
    )
end

@inline function _compensated_add(sum, correction, value)
    combined = sum + value
    correction += abs(sum) >= abs(value) ?
        (sum - combined) + value : (value - combined) + sum
    return combined, correction
end

@inline function _scalar_panel(
        major, lower, upper, ceiling, rate_parameters, quadrature, first, last,
    )
    T = typeof(major)
    upper <= lower && return zero(T), zero(T)
    nodes, weights = quadrature[1], quadrature[2]
    width, midpoint_scale = (upper - lower) / T(pi), (upper - lower) / 2
    cosine_sum, cosine_correction = zero(T), zero(T)
    sine_sum, sine_correction = zero(T), zero(T)
    for index in first:last
        theta = muladd(nodes[index] + 1, midpoint_scale, lower)
        sine, cosine = sincos(theta)
        # Panels are split at the lower cutoff and the ceiling. Clamping only
        # inside this live panel prevents rounded boundary angles from changing
        # the discontinuous clipping branch between two refinement rules.
        rate = ceiling ? rate_parameters[4] : _ionisation_rate(
                clamp(major * cosine, rate_parameters[2], rate_parameters[3]),
                rate_parameters,
            )
        cosine_value = weights[index] * rate
        sine_value = cosine_value * (sine / cosine)^2
        cosine_sum, cosine_correction = _compensated_add(
            cosine_sum, cosine_correction, cosine_value
        )
        sine_sum, sine_correction = _compensated_add(
            sine_sum, sine_correction, sine_value
        )
    end
    return width * (cosine_sum + cosine_correction), width * (sine_sum + sine_correction)
end

@inline function _scalar_cycle_rule(major, rate_parameters, quadrature, first, last)
    T = typeof(major)
    lower_field, upper_field = rate_parameters[2], rate_parameters[3]
    lower_cut = acos(clamp(lower_field / major, zero(T), one(T)))
    upper_cut = acos(clamp(upper_field / major, zero(T), one(T)))
    capped = _scalar_panel(
        major, zero(T), upper_cut, true, rate_parameters, quadrature, first, last
    )
    live = _scalar_panel(
        major, upper_cut, lower_cut, false, rate_parameters, quadrature, first, last
    )
    return capped[1] + live[1], capped[2] + live[2]
end

@inline function _scalar_cycle_moments(major, rate_parameters, quadrature)
    T = typeof(major)
    # At the exact lower threshold only one angle is live, a set of zero measure.
    # The instantaneous table's nonzero value at that single point does not create
    # a finite cycle average.
    major <= rate_parameters[2] && return zero(T), zero(T), PlasmaSuccess
    ends, rtol, floor = quadrature[3], quadrature[4], quadrature[5]
    previous = _scalar_cycle_rule(major, rate_parameters, quadrature, 1, ends[1])
    for level in 2:length(ends)
        current = _scalar_cycle_rule(
            major, rate_parameters, quadrature, ends[level - 1] + 1, ends[level]
        )
        if isfinite(current[1]) && isfinite(current[2]) &&
                abs(current[1] - previous[1]) <= rtol * max(current[1], floor) &&
                abs(current[2] - previous[2]) <= rtol * max(current[2], floor)
            return current[1], current[2], PlasmaSuccess
        end
        previous = current
    end
    return zero(T), zero(T), PlasmaQuadratureFailure
end

"""
    envelope_plasma!(drive, loss, samples, channel, quadrature, workspace, execution;
                     density, preionisation, dt, window=nothing,
                     count=workspace.layout.n_nodes, add=false, field_exponent=0)

Evaluate one scalar envelope-plasma population and its Drude-drive/loss-current streams.

# Arguments
- `drive`, `loss`: Caller-owned complex matrices with the workspace layout, receiving
  the Drude drive in A/(m² s) and the ionisation-loss current in A/m².
- `samples`: Stored complex envelope, shape `(n_time, n_nodes)`. Its physical value
  in V/m is `samples * 2^field_exponent`.
- `channel`: Prepared `PlasmaChannel` with an instantaneous isotropic rate table.
- `quadrature`: Translated `CycleQuadrature`; both directional moments are refined.
- `workspace`: Caller-owned `EnvelopePlasmaWorkspace`, including numerical status.
- `execution`: Synchronous numerical launcher.
- `density`, `preionisation`: Independent channel stage number density (m⁻³) and
  initial ionised fraction in `[0, 1]`.
- `dt`: Finite positive time spacing in seconds; envelope exposure uses a trapezoid.
- `window`: Optional dimensionless temporal source gate, applied to both streams.
- `count`: Active node count; every inactive work/output cell is cleared.
- `add`: Add active streams to previously evaluated channels instead of replacing them.
- `field_exponent`: Integer physical field scale; outputs have the same inverse
  scale. Table coordinates and the loss coefficient absorb it outside the kernel.

# Returns
`nothing`. Read `plasma_failure(workspace.status)` before using the streams. A rule
that does not converge records `PlasmaQuadratureFailure`, never an unlabelled NaN.
Warmed CPU calls allocate nothing; input, output and workspace arrays must be disjoint.

# Throws
- `ArgumentError`: If stage values, scalar layout or time spacing are invalid.
- `DimensionMismatch`: If input/output dimensions disagree with the workspace.

# Notes
For scalar `A`, the physical field is `Re(A exp(-im*omega*t))`. Over a quadrant,
`Mc = (2/pi) integral(W(abs(A)*cos(theta)) dtheta)` is its mean ionisation rate.
The second moment replaces the integrand by `W*tan(theta)^2`. The loss envelope per
unit ionisation work is `2Mc*A/abs2(A)`, which pays the cycle-averaged ionisation work.
Exposure gives survival `(1-f0)exp(-integral(Mc dt))` and its cancellation-free
complement. Projection, Fourier transforms, and the translated `-1/omega²` and
`-im/omega` source factors are applied by the caller exactly once.
"""
function envelope_plasma!(
        drive, loss, samples, channel::PlasmaChannel{T}, quadrature::CycleQuadrature{T},
        workspace::EnvelopePlasmaWorkspace{T}, execution;
        density, preionisation, dt, window = nothing,
        count::Int = workspace.layout.n_nodes, add::Bool = false,
        field_exponent::Int = 0,
    ) where {T}
    layout = workspace.layout
    _response_matrix(drive, layout)
    _response_matrix(loss, layout)
    _response_matrix(samples, layout)
    eltype(samples) <: Complex ||
        throw(ArgumentError("envelope plasma requires complex envelope samples"))
    _plasma_stage(layout, density, preionisation, count, window)
    spacing = T(dt)
    isfinite(spacing) && spacing > 0 ||
        throw(ArgumentError("envelope plasma dt must be finite and positive"))
    parameters = _rate_parameters(channel.rate, field_exponent)
    rules = _cycle_parameters(quadrature)
    rate, exposure = workspace.rate, workspace.exposure
    fraction, survival = workspace.fraction, workspace.survival
    moment, status = workspace.loss_moment, workspace.status
    charge, potential, stage_density =
        _plasma_coefficients(channel, density, field_exponent)
    initial, keep = T(preionisation), T(1 - Float64(preionisation))
    each!(execution, layout.n_time, layout.n_nodes; heavy = true) do row, node
        sample = node <= count ? samples[row, node] : zero(eltype(samples))
        major = abs(sample)
        code = isfinite(major) ? PlasmaSuccess : PlasmaNonfiniteInput
        cosine = zero(T)
        if code == PlasmaSuccess
            cosine, _, code = _scalar_cycle_moments(major, parameters, rules)
        end
        rate[row, node] = cosine
        moment[row, node] = iszero(major) || code != PlasmaSuccess ? zero(eltype(moment)) :
            (2 * (cosine / major)) * (sample / major)
        status[row, node] = UInt8(code)
    end
    each_column!(execution, layout.n_nodes) do node
        invalid = false
        for row in axes(samples, 1)
            invalid |= !iszero(status[row, node])
        end
        if node > count || invalid
            for row in axes(samples, 1)
                rate[row, node] = 0
                exposure[row, node] = 0
                fraction[row, node] = 0
                survival[row, node] = 0
                moment[row, node] = 0
                drive[row, node] = 0
                loss[row, node] = 0
            end
            return nothing
        end
        _envelope_exposure!(exposure, rate, node, spacing)
        for row in axes(samples, 1)
            total = exposure[row, node]
            neutral = keep * exp(-total)
            ionised = initial + keep * -expm1(-total)
            survival[row, node], fraction[row, node] = neutral, ionised
            weight = _response_weight(window, row, stage_density)
            first = weight * (charge * stage_density * ionised) * samples[row, node]
            second = weight * (potential * stage_density * neutral) * moment[row, node]
            first = add ? drive[row, node] + first : first
            second = add ? loss[row, node] + second : second
            # Saturation can make both currents finite despite infinite exposure
            # (for example at zero density). Do not hide that failed integral (#701).
            if !(
                    isfinite(total) && isfinite(neutral) && isfinite(ionised) &&
                        isfinite(first) && isfinite(second)
                )
                status[row, node] = UInt8(PlasmaNonfiniteResult)
                first, second = zero(eltype(drive)), zero(eltype(loss))
            end
            drive[row, node] = first
            loss[row, node] = second
        end
        return nothing
    end
    return nothing
end

"""
    PopulationWorkspace(execution, T, layout)

Allocate independent rate/population storage without any current-response buffers.

`layout` is a scalar `ResponseLayout`; `T` is the real response precision. Owned
`rate` and `filtered` are in s⁻¹; `exposure`, `fraction` and `survival` are
dimensionless. These arrays and `UInt8` `status` all have shape `(n_time,n_nodes)`.
The filtered rate is scratch for the field's FIR rule.
Envelope populations use the same storage and clear unused filter values. Current
coefficients and density are absent: populations depend only on rate and initial
fraction. Invalid vector layouts raise `ArgumentError` before allocation.
Population calls require independent writable destinations with immutable affine
index metadata. Mutable gather destinations and overlapping foreign storage are
refused before any numerical write, including workspaces assembled positionally.
"""
struct PopulationWorkspace{T, A <: AbstractMatrix{T}, U <: AbstractMatrix{UInt8}}
    layout::ResponseLayout
    rate::A
    exposure::A
    fraction::A
    survival::A
    filtered::A
    status::U
end

function PopulationWorkspace(execution, ::Type{T}, layout::ResponseLayout) where
    {T <: AbstractFloat}
    layout.n_components == 1 ||
        throw(ArgumentError("Julia populations currently require scalar polarisation"))
    dimensions = (layout.n_time, layout.n_nodes)
    return PopulationWorkspace(
        layout,
        ntuple(_ -> dzeros(execution, T, dimensions...), 5)...,
        dzeros(execution, UInt8, dimensions...)
    )
end

@inline _population_arrays(work) =
    (work.rate, work.exposure, work.fraction, work.survival, work.filtered, work.status)

@inline _population_disjoint(destination, ::Tuple{}) = nothing
@inline function _population_disjoint(destination, inputs::Tuple)
    _transfer_arrays_alias(destination, first(inputs)) &&
        throw(ArgumentError("population workspace aliases numerical input or scratch"))
    _population_disjoint(destination, Base.tail(inputs))
    return nothing
end

@inline _population_storage(::Tuple{}, samples, prepared) = nothing
@inline function _population_storage(arrays::Tuple, samples, prepared)
    destination, rest = first(arrays), Base.tail(arrays)
    _population_disjoint(destination, (samples, prepared..., rest...))
    _population_storage(rest, samples, prepared)
    return nothing
end

@inline _population_shapes(layout, ::Tuple{}) = nothing
@inline function _population_shapes(layout, arrays::Tuple)
    _response_matrix(first(arrays), layout)
    _population_shapes(layout, Base.tail(arrays))
    return nothing
end

function _population_admit(work, samples, prepared, initial, count)
    _response_matrix(samples, work.layout)
    _response_count(work.layout, count)
    isfinite(initial) && 0 <= initial <= 1 ||
        throw(ArgumentError("population preionisation must lie in [0, 1]"))
    arrays = _population_arrays(work)
    _population_shapes(work.layout, arrays)
    # Disjoint storage alone does not make a mutable gather destination safe:
    # repeated or later-changed indices can redirect a numerical write.
    _transfer_admit_destinations(arrays)
    _population_storage(arrays, samples, prepared)
    return nothing
end

@inline function _clear_population_column!(
        rate, exposure, fraction, survival, filtered, node,
    )
    for row in axes(rate, 1)
        rate[row, node] = 0
        exposure[row, node] = 0
        fraction[row, node] = 0
        survival[row, node] = 0
        filtered[row, node] = 0
    end
    return nothing
end

@inline function _population_pair(exposure, initial, keep)
    neutral = keep * exp(-exposure)
    fraction = initial + keep * -expm1(-exposure)
    valid = isfinite(exposure) && isfinite(neutral) && isfinite(fraction)
    return fraction, neutral, valid
end

"""
    field_populations!(samples, rate, rule, workspace, execution;
                       preionisation, count=workspace.layout.n_nodes, field_exponent=0)

Evaluate fresh scalar field rates and populations without constructing a current.

`samples` is a real `(n_time,n_nodes)` electric field whose physical value in V/m
is `samples * 2^field_exponent`. `rate` is its
`IonisationRateTable`, `rule` its `RunningIntegralRule`, and `workspace` an owned
`PopulationWorkspace`, all in the same real precision. `preionisation` lies in
`[0,1]`; `count` is the active node count. The complete rate/FIR/prefix footprint is
overwritten on each call and inactive columns are zeroed. Preserve the field and
prepared arrays, which must be disjoint from every workspace array.

Return `nothing`; check `plasma_failure(workspace.status)` before reading fractions
or peak rates. Nonfinite fields/rates/exposures/populations receive explicit status.
Warm calls allocate nothing. Invalid types, shapes, ranges, aliases or FIR support
raise before mutation. The survival and stable fraction use the same independent-atom
formula as `field_plasma!`; density, charge and ionisation potential do not enter.
"""
function field_populations!(
        samples, table::IonisationRateTable{T},
        rule::RunningIntegralRule{T}, work::PopulationWorkspace{T}, execution;
        preionisation, count::Int = work.layout.n_nodes, field_exponent::Int = 0,
    ) where {T}
    eltype(samples) === T ||
        throw(ArgumentError("field populations require matching real field precision"))
    length(rule.taps) <= work.layout.n_time ||
        throw(ArgumentError("population FIR support exceeds the temporal grid"))
    _population_admit(work, samples, (table.coefficients, rule.taps), preionisation, count)
    rate, exposure, fraction, survival, filtered, status = _population_arrays(work)
    parameters = _rate_parameters(table, field_exponent)
    taps, dt, floor_exposure = rule.taps, rule.dt, rule.floor_exposure
    initial, keep = T(preionisation), T(1 - Float64(preionisation))
    each_column!(execution, work.layout.n_nodes) do node
        invalid = false
        for row in axes(samples, 1)
            sample = node <= count ? samples[row, node] : zero(T)
            code = isfinite(sample) ? PlasmaSuccess : PlasmaNonfiniteInput
            value = code == PlasmaSuccess ?
                _ionisation_rate(abs(sample), parameters) : zero(T)
            if !isfinite(value)
                code, value = PlasmaNonfiniteResult, zero(T)
            end
            rate[row, node], status[row, node] = value, UInt8(code)
            invalid |= code != PlasmaSuccess
        end
        if node > count || invalid
            _clear_population_column!(rate, exposure, fraction, survival, filtered, node)
            return nothing
        end
        _integrate_plasma_column!(exposure, filtered, rate, node, taps, dt)
        for row in axes(samples, 1)
            if !(isfinite(exposure[row, node]) && isfinite(filtered[row, node]))
                status[row, node] = UInt8(PlasmaNonfiniteResult)
                invalid = true
                continue
            end
            value = floor_exposure ? max(exposure[row, node], zero(T)) : exposure[row, node]
            ionised, neutral, valid = _population_pair(value, initial, keep)
            fraction[row, node], survival[row, node] = valid ?
                (ionised, neutral) : (zero(T), zero(T))
            valid || (status[row, node] = UInt8(PlasmaNonfiniteResult))
            invalid |= !valid
        end
        invalid &&
            _clear_population_column!(rate, exposure, fraction, survival, filtered, node)
        return nothing
    end
    return nothing
end


"""
    envelope_populations!(samples, rate, quadrature, workspace, execution;
                          preionisation, dt, count=workspace.layout.n_nodes,
                          field_exponent=0)

Evaluate fresh scalar envelope rates and populations without loss/current algebra.

`samples` is a complex `(n_time,n_nodes)` envelope whose physical value in V/m is
`samples * 2^field_exponent`. `rate` is the instantaneous
`IonisationRateTable`, `quadrature` its translated `CycleQuadrature`, and `workspace`
an independent `PopulationWorkspace` in the matching real precision. `dt` is positive
in s; `preionisation` lies in `[0,1]` and `count` bounds active nodes. Both directional
cycle moments must meet the same refinement tests used by `envelope_plasma!`, even
though population accumulation consumes only the scalar rate.

Preserve inputs and overwrite every rate/population/status cell, clearing inactive
columns and unused filter scratch. Return `nothing`; check `plasma_failure` before
consuming rates or fractions. Invalid layouts, precision, stage values or aliases
raise before mutation. Failed quadrature and nonfinite numerical inputs/results use
status codes. Warm evaluation allocates nothing and reads no current coefficients.
"""
function envelope_populations!(
        samples, table::IonisationRateTable{T},
        quadrature::CycleQuadrature{T}, work::PopulationWorkspace{T}, execution;
        preionisation, dt, count::Int = work.layout.n_nodes, field_exponent::Int = 0,
    ) where {T}
    eltype(samples) === Complex{T} ||
        throw(ArgumentError("envelope populations require matching complex precision"))
    spacing = T(dt)
    isfinite(spacing) && spacing > 0 ||
        throw(ArgumentError("population time spacing must be finite and positive"))
    _population_admit(
        work, samples,
        (table.coefficients, quadrature.nodes, quadrature.weights), preionisation, count
    )
    rate, exposure, fraction, survival, filtered, status = _population_arrays(work)
    parameters = _rate_parameters(table, field_exponent)
    rules = _cycle_parameters(quadrature)
    initial, keep = T(preionisation), T(1 - Float64(preionisation))
    each!(execution, work.layout.n_time, work.layout.n_nodes; heavy = true) do row, node
        sample = node <= count ? samples[row, node] : zero(eltype(samples))
        major = abs(sample)
        code = isfinite(major) ? PlasmaSuccess : PlasmaNonfiniteInput
        value = zero(T)
        if code == PlasmaSuccess
            value, _, code = _scalar_cycle_moments(major, parameters, rules)
        end
        rate[row, node], status[row, node] = value, UInt8(code)
    end
    each_column!(execution, work.layout.n_nodes) do node
        invalid = false
        for row in axes(samples, 1)
            invalid |= !iszero(status[row, node])
        end
        if node > count || invalid
            _clear_population_column!(rate, exposure, fraction, survival, filtered, node)
            return nothing
        end
        _envelope_exposure!(exposure, rate, node, spacing)
        for row in axes(samples, 1)
            total = exposure[row, node]
            filtered[row, node] = zero(T)
            ionised, neutral, valid = _population_pair(total, initial, keep)
            fraction[row, node], survival[row, node] = valid ?
                (ionised, neutral) : (zero(T), zero(T))
            valid || (status[row, node] = UInt8(PlasmaNonfiniteResult))
        end
        return nothing
    end
    return nothing
end
