# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public PlasmaDiagnosticWorkspace, reset_plasma_diagnostics!
public plasma_diagnostics!, plasma_diagnostic_values

"""
    PlasmaDiagnosticWorkspace(execution, T, n_nodes)

Allocate bounded numerical storage for the three field-plasma integration diagnostics.

`T` is currently `Float64`; `n_nodes` is a positive chunk capacity. The owned `parts`
matrix has shape `(6,n_nodes)` and `maxima` has six entries. Their order is exposure
dip, exposure rise, absolute work defect, freed fraction, early free-current integrand
and peak free-current integrand. Only these arrays are retained. No current response,
material coefficient, source workspace or field is owned by this record.

See SEP 18 sections 4.4--4.5 and its Part II ratio-of-global-maxima correction.
Invalid precision or capacity raises `ArgumentError` before allocation.
"""
struct PlasmaDiagnosticWorkspace{A, V}
    parts::A
    maxima::V
end

function PlasmaDiagnosticWorkspace(execution, ::Type{T}, n_nodes::Int) where {T}
    T === Float64 && n_nodes > 0 ||
        throw(ArgumentError("plasma diagnostics require binary64 and a positive chunk"))
    return PlasmaDiagnosticWorkspace(
        dzeros(execution, T, 6, n_nodes), dzeros(execution, T, 6),
    )
end

function _plasma_diagnostic_shape(work)
    size(work.parts, 1) == length(work.maxima) == 6 && size(work.parts, 2) > 0 ||
        throw(DimensionMismatch("plasma diagnostic workspace dimensions disagree"))
    _transfer_writable_destination(work.parts)
    _transfer_writable_destination(work.maxima)
    _transfer_arrays_alias(work.parts, work.maxima) &&
        throw(ArgumentError("plasma diagnostic arrays must be disjoint"))
    return nothing
end

"""
    reset_plasma_diagnostics!(work)

Clear all six aggregate maxima and per-node parts before a fresh observed field.

`work` is a `PlasmaDiagnosticWorkspace`. A scan calls this once, then accumulates
every requested field-plasma channel and chunk with `plasma_diagnostics!`.
Return `nothing`; malformed or aliased owned arrays raise before writes.
"""
function reset_plasma_diagnostics!(work::PlasmaDiagnosticWorkspace)
    _plasma_diagnostic_shape(work)
    fill!(work.parts, 0)
    fill!(work.maxima, 0)
    return nothing
end

function _plasma_diagnostic_inputs(work, samples, population, rule, count)
    _plasma_diagnostic_shape(work)
    size(samples) == size(population.rate) == size(population.exposure) ==
        size(population.fraction) == size(population.status) && size(samples, 1) > 1 &&
        size(samples, 2) == size(work.parts, 2) &&
        0 <= count <= size(samples, 2) ||
        throw(DimensionMismatch("plasma diagnostics need matching scalar time/node arrays"))
    eltype(samples) === eltype(population.rate) === eltype(population.exposure) ===
        eltype(population.fraction) === Float64 ||
        throw(ArgumentError("plasma diagnostics require binary64 field populations"))
    isfinite(rule.dt) && rule.dt > 0 && 1 <= length(rule.taps) <= size(samples, 1) ||
        throw(ArgumentError("plasma diagnostic integral is invalid"))
    (
        _retained_alias(work.parts, (samples, population, rule)) ||
            _retained_alias(work.maxima, (samples, population, rule))
    ) &&
        throw(ArgumentError("plasma diagnostics alias their observed inputs"))
    all(value -> isfinite(value) && value >= 0, work.maxima) ||
        throw(ArgumentError("plasma diagnostic maxima must be finite and nonnegative"))
    for node in 1:count, row in axes(samples, 1)
        isfinite(samples[row, node]) && isfinite(population.exposure[row, node]) &&
            isfinite(population.rate[row, node]) && population.rate[row, node] >= 0 &&
            isfinite(population.fraction[row, node]) &&
            0 <= population.fraction[row, node] <= 1 &&
            population.status[row, node] == UInt8(PlasmaSuccess) ||
            throw(ArgumentError("plasma diagnostics need successful finite populations"))
    end
    return nothing
end

"""
    plasma_diagnostics!(work, samples, population, rule, execution;
                        count=size(samples,2))

Accumulate one fresh field-plasma channel's diagnostic maxima over a node chunk.

`samples` is the physical real field in V/m, shape `(n_time,n_nodes)`.
`population` is the matching successful `PopulationWorkspace`; its raw, unfloored
exposure, raw rate and ionised fraction are read without mutation. `rule` is the
`RunningIntegralRule` that produced them. `count` selects the active leading nodes.

For each node, form the negative/positive exposure increments, trapezoidal paid work
`integral(rate*(1-fraction))`, freed fraction `fraction[end]-fraction[1]`, and
`abs(fraction*field)` on the first `2K` samples and the whole window, where
`K=length(rule.taps)-1`. A zero-width rule has exactly zero early part. Take componentwise
maxima over nodes, channels and chunks before any division. The work statistic is
dimensionless and independent of density, charge and ionisation potential.

Return `nothing`. Invalid/aliased inputs raise before writes. Nonfinite intermediate
parts raise before changing the aggregate maxima; scratch may then contain the failed
parts. Warm calls allocate nothing. The complete per-node scratch is overwritten.
"""
function plasma_diagnostics!(
        work::PlasmaDiagnosticWorkspace, samples, population::PopulationWorkspace,
        rule::RunningIntegralRule, execution; count::Int = size(samples, 2),
    )
    _plasma_diagnostic_inputs(work, samples, population, rule, count)
    parts = work.parts
    rate, exposure, fraction = population.rate, population.exposure, population.fraction
    half_width = length(rule.taps) - 1
    reach = half_width <= size(samples, 1) ÷ 2 ? 2half_width : size(samples, 1)
    half_dt = rule.dt / 2
    each_column!(execution, size(parts, 2)) do node
        dip = rise = paid = early = peak = 0.0
        freed = 0.0
        if node <= count
            previous = rate[1, node] * (1 - fraction[1, node])
            for row in axes(samples, 1)
                integrand = abs(fraction[row, node] * samples[row, node])
                peak = max(peak, integrand)
                row <= reach && (early = max(early, integrand))
                if row > 1
                    increment = exposure[row, node] - exposure[row - 1, node]
                    dip, rise = max(dip, -increment), max(rise, increment)
                    current = rate[row, node] * (1 - fraction[row, node])
                    paid += half_dt * (previous + current)
                    previous = current
                end
            end
            freed = fraction[end, node] - fraction[1, node]
        end
        parts[1, node], parts[2, node] = dip, rise
        parts[3, node], parts[4, node] = abs(paid - freed), max(freed, 0.0)
        parts[5, node], parts[6, node] = early, peak
    end
    all(isfinite, parts) || throw(ArgumentError("plasma diagnostic arithmetic overflowed"))
    for part in eachindex(work.maxima)
        value = work.maxima[part]
        for node in axes(parts, 2)
            value = max(value, parts[part, node])
        end
        work.maxima[part] = value
    end
    return nothing
end

"""
    plasma_diagnostic_values(work)

Return the physical dimensionless undershoot, work-defect and boundary-level ratios.

Each numerator and denominator is a separate maximum over the complete scan.
A zero denominator returns exactly zero. `work` is a `PlasmaDiagnosticWorkspace`;
malformed, negative or nonfinite maxima and overflowing ratios raise `ArgumentError`.
No array is mutated and no storage is allocated.
"""
function plasma_diagnostic_values(work::PlasmaDiagnosticWorkspace)
    length(work.maxima) == 6 ||
        throw(DimensionMismatch("plasma diagnostic maxima need six entries"))
    all(value -> isfinite(value) && value >= 0, work.maxima) ||
        throw(ArgumentError("plasma diagnostic maxima must be finite and nonnegative"))
    values = ntuple(3) do index
        numerator, denominator = work.maxima[2index - 1], work.maxima[2index]
        denominator > 0 ? Float64(numerator / denominator) : 0.0
    end
    all(isfinite, values) || throw(ArgumentError("plasma diagnostic ratio overflowed"))
    return values
end
