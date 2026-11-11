# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct _CartesianTemporalSplit{B, T}
    band::B
    temporal::T
end

function _temporal_split_workspace(execution, data::SpatialObservationPreparation, rate)
    temporal = TemporalSplitWorkspace(
        execution, data.kind, rate, data.n_modes; arithmetic = data.arithmetic,
    )
    data.map isa RadialObservationMap && return temporal
    band = CartesianBandWork(
        execution, data.map.shape, length(data.grid.omega); arithmetic = data.arithmetic,
    )
    return _CartesianTemporalSplit(band, temporal)
end

function finish_temporal_split!(candidate, work::_CartesianTemporalSplit, dz, execution)
    candidate isa StridedMatrix{eltype(work.band.band)} &&
        size(candidate) == size(work.band.band) ||
        throw(DimensionMismatch("Cartesian split candidate has an incompatible layout"))
    dz = Float64(dz)::Float64
    isfinite(dz) && dz > 0 && all(isfinite, candidate) ||
        throw(ArgumentError("Cartesian splitting needs a finite field and positive step"))
    _transfer_writable_destination(candidate)
    _retained_alias(candidate, work) &&
        throw(ArgumentError("Cartesian split candidate aliases its retained buffers"))
    work.temporal.inactive && return nothing
    # A transverse spectral column need not be a real field's temporal spectrum.
    # Match the reference: transverse inverse, real/complex temporal pair, forward.
    physical = cartesian_inverse!(work.band, candidate)
    finish_temporal_split!(physical, work.temporal, dz, execution)
    cartesian_forward!(physical, work.band)
    all(isfinite, physical) ||
        throw(DomainError(dz, "Cartesian temporal splitting produced nonfinite values"))
    copyto!(candidate, physical)
    return nothing
end
