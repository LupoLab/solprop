# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public TemporalFFT, temporal_forward!, temporal_inverse!
public SpatialFFT, spatial_forward!, spatial_inverse!
public SpectralPadding, pad_spectrum!, crop_spectrum!
public apply_weights!, linear_map!, synthesize!, project!

"""
    TemporalFFT(execution, time_buffer, spectrum_buffer)

Prepare raw forward and normalized inverse temporal transforms along matrix rows.

`time_buffer` is a dense `Matrix` with shape `(n_time, n_columns)`, real for a field and
complex elements for an envelope. `spectrum_buffer` has matching complex precision
and shape `(n_time ÷ 2 + 1, n_columns)` or `(n_time, n_columns)`, respectively.
Both buffers are disjoint caller-owned dense matrices. Plans preserve their inputs.
Views are refused during preparation: FFTW's out-of-place plans choose dense outputs,
which cannot in general match a view's strides or alignment in both directions.
The internal forward sign is negative; no physical units or Fourier conventions change.

CPU plans use one FFTW thread, staying inside every positive execution budget. The
backward plan is unnormalized; `temporal_inverse!` applies `inverse_scale = 1/n_time`.
Planning restores FFTW's previous planner thread setting.

# Throws
- `ArgumentError`: For unsupported precision, storage, backend or aliased buffers.
- `DimensionMismatch`: For inconsistent transform dimensions.
"""
struct TemporalFFT{F, I, T}
    forward::F
    inverse::I
    inverse_scale::T
    n_time::Int
    n_frequency::Int
    n_columns::Int
end

function TemporalFFT(
        execution::KernelExecution, time_buffer::Matrix{T},
        spectrum_buffer::Matrix{Complex{R}},
    ) where {T, R <: Union{Float32, Float64}}
    oncpu(execution) || throw(ArgumentError("this backend has no qualified FFT planner"))
    real_signal = T === R
    real_signal || T === Complex{R} ||
        throw(ArgumentError("time and spectral buffers must have matching precision"))
    n_time, n_columns = size(time_buffer)
    n_time > 0 && n_columns > 0 ||
        throw(ArgumentError("FFT dimensions must be positive"))
    n_frequency = real_signal ? div(n_time, 2) + 1 : n_time
    size(spectrum_buffer) == (n_frequency, n_columns) ||
        throw(DimensionMismatch("spectral buffer does not match the temporal layout"))
    Base.mightalias(time_buffer, spectrum_buffer) &&
        throw(ArgumentError("temporal transform buffers must be disjoint"))
    flags = FFTW.ESTIMATE | FFTW.PRESERVE_INPUT
    if real_signal
        forward = FFTW.plan_rfft(time_buffer, 1; flags, num_threads = 1)
        inverse = FFTW.plan_brfft(spectrum_buffer, n_time, 1; flags, num_threads = 1)
    else
        forward = FFTW.plan_fft(time_buffer, 1; flags, num_threads = 1)
        inverse = FFTW.plan_bfft(spectrum_buffer, 1; flags, num_threads = 1)
    end
    return TemporalFFT(forward, inverse, inv(R(n_time)), n_time, n_frequency, n_columns)
end

function TemporalFFT(execution::KernelExecution, time_buffer, spectrum_buffer)
    throw(ArgumentError("temporal FFTs require matching dense binary32/binary64 matrices"))
end

"""
    temporal_forward!(spectrum, plan::TemporalFFT, time)

Write the unnormalized negative-sign temporal DFT into the prepared spectral buffer.

Shapes and strides must match the arrays used to prepare `plan`. Inputs are preserved;
the output is returned. The plan's single FFTW thread finishes before return.
"""
function temporal_forward!(spectrum, plan::TemporalFFT, time)
    LinearAlgebra.mul!(spectrum, plan.forward, time)
    return spectrum
end

"""
    temporal_inverse!(time, plan::TemporalFFT, spectrum, execution)

Write the inverse temporal DFT, including its explicit `1/n_time` normalization.

Shapes and strides match the prepared buffers. The spectrum is preserved and the
time buffer is returned. Scaling uses the execution's bounded synchronous launcher.
"""
function temporal_inverse!(time, plan::TemporalFFT, spectrum, execution::KernelExecution)
    LinearAlgebra.mul!(time, plan.inverse, spectrum)
    inverse_scale = plan.inverse_scale
    each!(execution, size(time)...) do row, column
        time[row, column] *= inverse_scale
    end
    return time
end

"""
    SpatialFFT(execution, buffer, axes)

Prepare in-place complex DFTs over explicitly named spatial axes.

`buffer` is strided binary32 or binary64 complex storage; `axes` is a nonempty tuple
of distinct dimension indices. Other dimensions are independent batches. For example,
`(frequency, component, x, y)` storage transforms axes `(3,4)`, preserving the agreed
component-first flattened state. No physical spatial scale is folded into the plans.

The forward transform is raw and negative-sign. The backward plan is unnormalized;
`spatial_inverse!` applies `1/prod(size(buffer, axis) for axis in axes)` explicitly.
Single-thread FFTW plans stay within the execution budget and restore planner settings.
Preparation preserves the supplied buffer; all transform cells are overwritten in use.

# Throws
- `ArgumentError`: For unsupported storage/backend, empty dimensions or invalid axes.
"""
struct SpatialFFT{F, I, T}
    forward::F
    inverse::I
    inverse_scale::T
end

function SpatialFFT(
        execution::KernelExecution, buffer::StridedArray{Complex{T}},
        axes::Tuple{Vararg{Integer}},
    ) where {T <: Union{Float32, Float64}}
    oncpu(execution) || throw(ArgumentError("this backend has no qualified FFT planner"))
    isempty(buffer) && throw(ArgumentError("spatial FFT dimensions must be positive"))
    region = map(Int, axes)
    !isempty(region) && all(axis -> 1 <= axis <= ndims(buffer), region) &&
        length(unique(region)) == length(region) ||
        throw(ArgumentError("spatial FFT axes must be nonempty, distinct and in range"))
    count = 1
    for axis in region
        count = Base.checked_mul(count, size(buffer, axis))
    end
    forward = FFTW.plan_fft!(buffer, region; flags = FFTW.ESTIMATE, num_threads = 1)
    inverse = FFTW.plan_bfft!(buffer, region; flags = FFTW.ESTIMATE, num_threads = 1)
    return SpatialFFT(forward, inverse, inv(T(count)))
end

function SpatialFFT(execution::KernelExecution, buffer, axes)
    throw(ArgumentError("spatial FFTs require strided complex storage and an axes tuple"))
end

"""
    spatial_forward!(buffer, plan::SpatialFFT)

Replace the whole prepared spatial buffer with its raw negative-sign DFT.

Shape and strides match preparation. The single FFTW thread finishes before return.
Return `buffer` without introducing physical normalization.
"""
function spatial_forward!(buffer, plan::SpatialFFT)
    LinearAlgebra.mul!(buffer, plan.forward, buffer)
    return buffer
end

"""
    spatial_inverse!(buffer, plan::SpatialFFT, execution)

Replace the prepared buffer with its spatial inverse, including explicit DFT normalization.

Every cell is transformed and scaled. No temporal normalization or translated spatial
scale is added. Return `buffer` after bounded synchronous completion.
"""
function spatial_inverse!(buffer, plan::SpatialFFT, execution::KernelExecution)
    LinearAlgebra.mul!(buffer, plan.inverse, buffer)
    scale = plan.inverse_scale
    each!(execution, length(buffer), 1) do index, unused
        buffer[index] *= scale
    end
    return buffer
end

"""
    SpectralPadding(execution, crop_index, n_fine_frequency, pad_scale, crop_scale;
                    index_base = 0)
    SpectralPadding(execution, n_coarse_time, n_fine_time; real_signal)

Copy an authoritative spectral crop map and its normalization into owned storage.

Each coarse spectral bin maps to one unique fine bin. The explicit constructor accepts
the translated zero-based `crop_index`; `index_base=1` accepts Julia indices. The two
positive finite scales are carried exactly as supplied. The derived constructor uses
solprop's field leading bins or envelope leading/trailing halves and the time-count
ratios. Time counts in that constructor are even, at least four, with an integer ratio.

For a real field, a nonzero coarse Nyquist bin is copied without splitting. As in
`solprop.fourier.oversample`, coincident time samples require that bin to be zero when
oversampling; translated field grids enforce this condition with their mask.

# Throws
- `ArgumentError`: For invalid indices, scales or time counts.
"""
struct SpectralPadding{I, T}
    crop::I
    uncrop::I
    pad_scale::T
    crop_scale::T
end

function SpectralPadding(
        execution::KernelExecution, crop_index::AbstractVector{<:Integer},
        n_fine_frequency::Integer, pad_scale::Real, crop_scale::Real;
        index_base::Integer = 0,
    )
    n_fine_frequency > 0 ||
        throw(ArgumentError("the fine spectral length must be positive"))
    index_base in (0, 1) || throw(ArgumentError("the crop index base must be zero or one"))
    isempty(crop_index) && throw(ArgumentError("the crop map must not be empty"))
    scales = promote(float(pad_scale), float(crop_scale))
    all(x -> isfinite(x) && x > zero(x), scales) ||
        throw(ArgumentError("spectral scales must be finite and positive"))
    crop = Int.(crop_index) .+ (1 - index_base)
    all(index -> 1 <= index <= n_fine_frequency, crop) ||
        throw(ArgumentError("the crop map points outside the fine spectrum"))
    length(unique(crop)) == length(crop) ||
        throw(ArgumentError("the crop map must not repeat a fine bin"))
    uncrop = zeros(Int, n_fine_frequency)
    for (coarse, fine) in enumerate(crop)
        uncrop[fine] = coarse
    end
    return SpectralPadding(upload(execution, crop), upload(execution, uncrop), scales...)
end

function SpectralPadding(
        execution::KernelExecution, n_coarse_time::Integer, n_fine_time::Integer;
        real_signal::Bool,
    )
    if n_coarse_time < 4 || isodd(n_coarse_time) || n_fine_time < n_coarse_time ||
            isodd(n_fine_time) || rem(n_fine_time, n_coarse_time) != 0
        throw(ArgumentError("temporal padding needs even counts and an integer ratio"))
    end
    if real_signal
        crop = collect(0:div(n_coarse_time, 2))
        n_fine_frequency = div(n_fine_time, 2) + 1
    else
        half = div(n_coarse_time, 2)
        crop = vcat(collect(0:(half - 1)), collect((n_fine_time - half):(n_fine_time - 1)))
        n_fine_frequency = n_fine_time
    end
    return SpectralPadding(
        execution, crop, n_fine_frequency,
        n_fine_time / n_coarse_time, n_coarse_time / n_fine_time,
    )
end

@inline function _active_columns(output, input, active_columns)
    0 <= active_columns <= min(size(output, 2), size(input, 2)) ||
        throw(DimensionMismatch("active columns exceed a prepared buffer"))
    return Int(active_columns)
end

"""
    pad_spectrum!(fine, coarse, padding, execution; active_columns = size(coarse, 2))

Write a scaled coarse spectrum into its fine bins and zero every other output cell.

Buffers are disjoint matrices with row lengths matching `padding`. Only the declared
active columns are read; all padding and inactive output columns are overwritten.
Return `fine`, preserving `coarse` exactly.
"""
function pad_spectrum!(
        fine, coarse, padding::SpectralPadding, execution::KernelExecution;
        active_columns::Integer = size(coarse, 2),
    )
    size(coarse, 1) == length(padding.crop) && size(fine, 1) == length(padding.uncrop) ||
        throw(DimensionMismatch("spectral buffers do not match the padding map"))
    active = _active_columns(fine, coarse, active_columns)
    Base.mightalias(fine, coarse) &&
        throw(ArgumentError("spectral padding buffers must be disjoint"))
    uncrop, scale = padding.uncrop, padding.pad_scale
    neutral = zero(eltype(fine))
    each!(execution, size(fine)...) do row, column
        index = uncrop[row]
        fine[row, column] = column <= active && index != 0 ?
            coarse[index, column] * scale : neutral
    end
    return fine
end

"""
    crop_spectrum!(coarse, fine, padding, execution; active_columns = size(coarse, 2))

Write scaled fine-grid bins into the coarse spectrum and zero inactive output columns.

Buffers are disjoint matrices with the prepared row lengths. Return `coarse`, preserving
`fine` exactly. No bins outside the authoritative map contribute to the result.
"""
function crop_spectrum!(
        coarse, fine, padding::SpectralPadding, execution::KernelExecution;
        active_columns::Integer = size(coarse, 2),
    )
    size(coarse, 1) == length(padding.crop) && size(fine, 1) == length(padding.uncrop) ||
        throw(DimensionMismatch("spectral buffers do not match the padding map"))
    active = _active_columns(coarse, fine, active_columns)
    Base.mightalias(fine, coarse) &&
        throw(ArgumentError("spectral cropping buffers must be disjoint"))
    crop, scale = padding.crop, padding.crop_scale
    neutral = zero(eltype(coarse))
    each!(execution, size(coarse)...) do row, column
        coarse[row, column] = column <= active ? fine[crop[row], column] * scale : neutral
    end
    return coarse
end

@inline _weight(weights::AbstractVector, row, column) = weights[row]
@inline _weight(weights::AbstractMatrix, row, column) = weights[row, column]

"""
    apply_weights!(output, input, weights, execution; active_columns = size(input, 2))

Apply prepared row or matrix weights, clearing zero weights and inactive cells exactly.

`input` and `output` have equal row counts; weights have one entry per row or the input
matrix shape. In-place use is supported. Hard zeros do not multiply a discarded input,
so a masked nonfinite value cannot contaminate the output. Return `output`.
"""
function apply_weights!(
        output, input, weights::Union{AbstractVector, AbstractMatrix},
        execution::KernelExecution; active_columns::Integer = size(input, 2),
    )
    size(output, 1) == size(input, 1) && size(weights, 1) == size(input, 1) ||
        throw(DimensionMismatch("weight and field row counts must agree"))
    ndims(weights) == 1 || size(weights, 2) >= active_columns ||
        throw(DimensionMismatch("matrix weights omit active columns"))
    active = _active_columns(output, input, active_columns)
    Base.mightalias(output, input) && output !== input &&
        throw(ArgumentError("partially overlapping weighted buffers are not supported"))
    Base.mightalias(output, weights) &&
        throw(ArgumentError("the weighted output must not overwrite its weights"))
    neutral = zero(eltype(output))
    each!(execution, size(output)...) do row, column
        if column <= active
            weight = _weight(weights, row, column)
            output[row, column] = iszero(weight) ? neutral : input[row, column] * weight
        else
            output[row, column] = neutral
        end
    end
    return output
end

"""
    linear_map!(output, input, weights, execution;
                active_inputs = size(input, 2), active_columns = size(output, 2))

Apply a prepared dense column map without invoking a separate BLAS thread pool.

Arrays use `(n_rows, n_columns)` storage; `weights[in_column, out_column]` carries the
translated map and its units. Sum only the declared active input columns, overwrite
every output cell, and clear inactive output columns. Input/output storage is disjoint.
The scalar contraction order is fixed; independent rows and columns use `execution`.
Return `output`.
"""
function linear_map!(
        output, input, weights::AbstractMatrix, execution::KernelExecution;
        active_inputs::Integer = size(input, 2),
        active_columns::Integer = size(output, 2),
    )
    size(output, 1) == size(input, 1) ||
        throw(DimensionMismatch("linear-map row counts must agree"))
    0 <= active_inputs <= min(size(input, 2), size(weights, 1)) ||
        throw(DimensionMismatch("linear-map input columns exceed the prepared map"))
    0 <= active_columns <= min(size(output, 2), size(weights, 2)) ||
        throw(DimensionMismatch("linear-map output columns exceed the prepared map"))
    Base.mightalias(output, input) &&
        throw(ArgumentError("linear-map buffers must be disjoint"))
    neutral = zero(eltype(output))
    each!(execution, size(output)...) do row, column
        total = neutral
        if column <= active_columns
            for input_column in 1:active_inputs
                total += input[row, input_column] * weights[input_column, column]
            end
        end
        output[row, column] = total
    end
    return output
end

"""
    synthesize!(nodal, modal, synthesis, execution; n_components, active_nodes)

Synthesize independent components from modes into flattened nodal columns.

`synthesis[node, mode]` maps each modal component to a node. Column indices are
`component + n_components*(mode-1)` for modal and
`component + n_components*(node-1)` for nodal arrays. `active_nodes` defaults to all
prepared nodes. Every inactive nodal column is zeroed. Return `nodal`.
"""
function synthesize!(
        nodal, modal, synthesis::AbstractMatrix, execution::KernelExecution;
        n_components::Integer, active_nodes::Integer = size(synthesis, 1),
    )
    n_components > 0 || throw(ArgumentError("the component count must be positive"))
    n_modes = size(synthesis, 2)
    size(modal, 2) == n_components * n_modes && size(nodal, 1) == size(modal, 1) ||
        throw(DimensionMismatch("modal synthesis buffers do not match the map"))
    0 <= active_nodes <= size(synthesis, 1) &&
        n_components * active_nodes <= size(nodal, 2) ||
        throw(DimensionMismatch("active synthesis nodes exceed the prepared buffers"))
    Base.mightalias(nodal, modal) &&
        throw(ArgumentError("modal synthesis buffers must be disjoint"))
    neutral = zero(eltype(nodal))
    each!(execution, size(nodal)...) do row, column
        node = div(column - 1, n_components) + 1
        component = mod(column - 1, n_components) + 1
        total = neutral
        if node <= active_nodes
            for mode in 1:n_modes
                total += modal[row, component + n_components * (mode - 1)] *
                    synthesis[node, mode]
            end
        end
        nodal[row, column] = total
    end
    return nodal
end

"""
    project!(modal, nodal, projection, execution; n_components, active_nodes)

Project flattened nodal columns into independent modal components.

`projection[mode, node]` includes the translated quadrature and physical normalization.
Flattened columns follow `synthesize!`. Only `active_nodes` are read and every modal
output is overwritten; buffers are disjoint. Return `modal`.
"""
function project!(
        modal, nodal, projection::AbstractMatrix, execution::KernelExecution;
        n_components::Integer, active_nodes::Integer = size(projection, 2),
    )
    n_components > 0 || throw(ArgumentError("the component count must be positive"))
    n_modes = size(projection, 1)
    size(modal, 2) == n_components * n_modes && size(nodal, 1) == size(modal, 1) ||
        throw(DimensionMismatch("modal projection buffers do not match the map"))
    0 <= active_nodes <= size(projection, 2) &&
        n_components * active_nodes <= size(nodal, 2) ||
        throw(DimensionMismatch("active projection nodes exceed the prepared buffers"))
    Base.mightalias(nodal, modal) &&
        throw(ArgumentError("modal projection buffers must be disjoint"))
    neutral = zero(eltype(modal))
    each!(execution, size(modal)...) do row, column
        mode = div(column - 1, n_components) + 1
        component = mod(column - 1, n_components) + 1
        total = neutral
        for node in 1:active_nodes
            total += nodal[row, component + n_components * (node - 1)] *
                projection[mode, node]
        end
        modal[row, column] = total
    end
    return modal
end
