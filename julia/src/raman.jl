# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public RamanDrive, RamanKernel, RamanConvolution
public raman_drive!, raman_products!, rebuild_raman!, convolve_raman!, raman_polarisation!

"""
    RamanDrive(layout; kind=:field, thg=true)

Prepare the scalar Raman quadratic drive on an explicitly declared time/node layout.

# Arguments
- `layout`: `ResponseLayout` with one physical component per node.
- `kind`: `:field` or `:envelope`, using the internal Fourier convention.
- `thg`: Retain the real field's twice-carrier beat. Ignored for an envelope.

# Returns
An immutable drive description. A real harmonic-retaining field uses `E^2`; a
harmonic-free real field uses `abs2(analytic_signal(E))/2`; an envelope uses `abs2(A)/2`.

# Throws
- `ArgumentError`: If the grid kind is unknown or the layout requests tensor Raman.
"""
struct RamanDrive
    layout::ResponseLayout
    analytic_signal::Bool
    half_weight::Bool

    function RamanDrive(layout::ResponseLayout; kind::Symbol = :field, thg::Bool = true)
        kind in (:field, :envelope) || throw(ArgumentError("unknown Raman grid kind"))
        layout.n_components == 1 ||
            throw(ArgumentError("Julia Raman currently requires scalar polarisation"))
        layout.n_time <= typemax(Int) ÷ 2 ||
            throw(ArgumentError("Raman time count does not fit the doubled grid"))
        return new(layout, kind == :field && !thg, kind == :envelope || !thg)
    end
end

"""
    RamanKernel(undamped, dt)

Retain translated causal Raman rows and their sample spacing for stage reconstruction.

# Arguments
- `undamped`: Finite real matrix `(2n_time, n_terms)` of independently damped causal
  rows, in the translated response's SI units. Its second half is exactly zero.
  The caller owns immutable prepared storage.
- `dt`: Finite positive sample spacing in seconds, at the rows' scalar precision.

# Returns
An immutable description. Term weights and decay rates are supplied separately at
each stage; no Kerr, operator, density or preionisation profile is substituted.

# Throws
- `ArgumentError`: If dimensions, finiteness, zero padding or `dt` are invalid.
"""
struct RamanKernel{T <: AbstractFloat, A <: AbstractMatrix{T}}
    undamped::A
    dt::T

    function RamanKernel(
            undamped::A, dt::T,
        ) where {T <: AbstractFloat, A <: AbstractMatrix{T}}
        size(undamped, 1) >= 2 && iseven(size(undamped, 1)) && size(undamped, 2) > 0 ||
            throw(ArgumentError("Raman rows need a nonempty doubled temporal grid"))
        all(isfinite, undamped) || throw(ArgumentError("Raman rows must be finite"))
        n_time = size(undamped, 1) ÷ 2
        all(iszero, view(undamped, (n_time + 1):size(undamped, 1), :)) ||
            throw(ArgumentError("the second half of each Raman row must be zero padding"))
        isfinite(dt) && dt > 0 ||
            throw(ArgumentError("Raman dt must be finite and positive"))
        return new{T, A}(undamped, dt)
    end
end

"""
    RamanConvolution(n_time, kernel, forward, inverse; inverse_scale=1/(2n_time))

Bind a doubled-grid Raman spectrum to caller-prepared transform plans.

# Arguments
- `n_time`: Positive number of physical time samples, before zero padding.
- `kernel`: Complex vector of length `n_time + 1`, equal to `dt * rfft(h)` on the
  doubled grid. Its prepared storage may be updated by `rebuild_raman!`.
- `forward`, `inverse`: Plans accepted by `LinearAlgebra.mul!` for real-to-complex
  and complex-to-real column transforms. Planning and thread policy belong to preparation.
- `inverse_scale`: Explicit inverse normalisation: `1/(2n_time)` for an unnormalised
  `brfft` plan, or one for a normalised `irfft` plan. Finite and positive.

# Returns
An immutable, concretely typed plan/kernel record, retaining no scratch arrays.

# Throws
- `ArgumentError`: If the temporal size or inverse scale is invalid.
- `DimensionMismatch`: If the kernel does not fit the doubled-grid half spectrum.
"""
struct RamanConvolution{T <: AbstractFloat, K, F, I}
    n_time::Int
    kernel::K
    forward::F
    inverse::I
    inverse_scale::T

    function RamanConvolution(
            n_time::Int, kernel::AbstractVector{Complex{T}}, forward, inverse;
            inverse_scale = inv(T(2) * T(n_time)),
        ) where {T <: AbstractFloat}
        n_time > 0 && n_time <= typemax(Int) ÷ 2 ||
            throw(ArgumentError("Raman time count must fit a positive doubled grid"))
        length(kernel) == n_time + 1 ||
            throw(DimensionMismatch("Raman kernel does not fit the doubled grid"))
        scale = T(inverse_scale)
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("Raman inverse scale must be finite and positive"))
        return new{T, typeof(kernel), typeof(forward), typeof(inverse)}(
            n_time, kernel, forward, inverse, scale
        )
    end
end

"""
    raman_drive!(drive, samples, prepared, execution;
                 analytic_samples=nothing, count=prepared.layout.n_nodes)

Fill the complete doubled-grid scalar Raman drive, including inactive and padded zeros.

# Arguments
- `drive`: Real caller-owned matrix `(2n_time, n_nodes)`, in (V/m)².
- `samples`: Physical field or envelope matrix `(n_time, n_nodes)`, in V/m.
- `prepared`: `RamanDrive` with explicit grid and layout metadata.
- `execution`: Synchronous numerical launcher.
- `analytic_samples`: Caller-owned complex analytic signal with the samples' layout;
  required only for a real field whose harmonic drive is excluded.
- `count`: Active node count in `[0, n_nodes]`; all other output cells are overwritten.

# Returns
`nothing`. No sample at row `n_time + 1` or later is read, including in the
non-harmonic path. Warmed CPU calls allocate nothing.

# Throws
- `ArgumentError`: If a required analytic signal is absent or the count is invalid.
- `DimensionMismatch`: If an array does not match the declared layout.
"""
function raman_drive!(
        drive, samples, prepared::RamanDrive, execution;
        analytic_samples = nothing, count::Int = prepared.layout.n_nodes,
    )
    layout = prepared.layout
    _response_matrix(samples, layout)
    _response_count(layout, count)
    size(drive) == (2layout.n_time, layout.n_nodes) ||
        throw(DimensionMismatch("Raman drive does not match its doubled layout"))
    if prepared.analytic_signal
        analytic_samples === nothing &&
            throw(ArgumentError("non-harmonic field Raman requires an analytic signal"))
        _response_matrix(analytic_samples, layout)
        _fill_raman_drive!(drive, analytic_samples, prepared, execution, count)
    else
        _fill_raman_drive!(drive, samples, prepared, execution, count)
    end
    return nothing
end

function _fill_raman_drive!(drive, signal, prepared, execution, count)
    n_time = prepared.layout.n_time
    weight = prepared.half_weight ? eltype(drive)(0.5) : one(eltype(drive))
    each!(execution, size(drive, 1), size(drive, 2)) do row, node
        drive[row, node] = row > n_time || node > count ? zero(eltype(drive)) :
            weight * abs2(signal[row, node])
    end
    return nothing
end

"""
    raman_products!(drive, signals, first, second, execution; thg=true)

Fill doubled-grid modal product drives without reading the zero-padded input region.

# Arguments
- `drive`: Real caller-owned matrix `(2n_time, n_pairs)`, in (V/m)².
- `signals`: Real modal fields when `thg=true`, or their supplied complex analytic
  signals otherwise, shape `(n_time, n_modes)`, in V/m.
- `first`, `second`: Prepared integer vectors of modal indices, one per output pair.
- `execution`: Synchronous numerical launcher.
- `thg`: Form `a_m * a_n` for a real field; otherwise form
  `real(a_m * conj(a_n))/2`. Projection weights and pair multiplicity belong to the caller.

# Returns
`nothing`. Every output cell is overwritten. The padding branch precedes all signal
loads, fixing the small non-THG out-of-bounds path recorded in issue #613.

# Throws
- `DimensionMismatch`: If pair or temporal dimensions disagree.
- `ArgumentError`: If an index is out of range or a real drive receives complex input.
"""
function raman_products!(drive, signals, first, second, execution; thg::Bool = true)
    size(drive, 1) == 2size(signals, 1) &&
        size(drive, 2) == length(first) == length(second) ||
        throw(DimensionMismatch("Raman product dimensions do not match"))
    all(index -> 1 <= index <= size(signals, 2), first) &&
        all(index -> 1 <= index <= size(signals, 2), second) ||
        throw(ArgumentError("Raman modal product index is out of range"))
    thg && !(eltype(signals) <: Real) &&
        throw(ArgumentError("harmonic-retaining Raman products require real fields"))
    n_time = size(signals, 1)
    each!(execution, size(drive, 1), size(drive, 2)) do row, pair
        if row > n_time
            drive[row, pair] = zero(eltype(drive))
        else
            a, b = signals[row, first[pair]], signals[row, second[pair]]
            drive[row, pair] = thg ? a * b :
                (real(a) * real(b) + imag(a) * imag(b)) / 2
        end
    end
    return nothing
end

"""
    rebuild_raman!(time_kernel, spectrum, prepared, rates, weights, forward, execution)

Rebuild every sample and spectral bin of a stage's weighted, damped Raman kernel.

# Arguments
- `time_kernel`: Real caller-owned vector of length `2n_time`.
- `spectrum`: Complex caller-owned vector of length `n_time + 1`; receives `dt*rfft(h)`.
- `prepared`: `RamanKernel` holding undamped rows and spacing in seconds.
- `rates`: Stage decay rates in s⁻¹, one finite nonnegative entry per row term.
- `weights`: Stage term weights, one finite entry per term, already containing the
  appropriate Raman component density and fraction. These are independent profiles.
- `forward`: Caller-prepared real FFT plan from `time_kernel` to `spectrum`.
- `execution`: Synchronous numerical launcher.

# Returns
`nothing`. Zero weights produce an exact zero kernel. There is no exponent lookup,
hidden last-stage cache, allocation, or mutation of the supplied rates/weights.

# Throws
- `DimensionMismatch`: If scratch or stage arrays do not match the prepared rows.
- `ArgumentError`: If a rate is negative/nonfinite or a weight is nonfinite.
"""
function rebuild_raman!(
        time_kernel, spectrum, prepared::RamanKernel, rates, weights, forward, execution,
    )
    rows, dt = prepared.undamped, prepared.dt
    length(time_kernel) == size(rows, 1) && length(spectrum) == size(rows, 1) ÷ 2 + 1 &&
        length(rates) == length(weights) == size(rows, 2) ||
        throw(DimensionMismatch("Raman reconstruction dimensions do not match"))
    all(rate -> isfinite(rate) && rate >= 0, rates) ||
        throw(ArgumentError("Raman decay rates must be finite and nonnegative"))
    all(isfinite, weights) || throw(ArgumentError("Raman term weights must be finite"))
    each!(execution, length(time_kernel), 1) do row, _
        value = zero(eltype(time_kernel))
        time = (row - 1) * dt
        for term in axes(rows, 2)
            value += weights[term] * rows[row, term] * exp(-rates[term] * time)
        end
        time_kernel[row] = value
    end
    LinearAlgebra.mul!(spectrum, forward, time_kernel)
    each!(execution, length(spectrum), 1) do row, _
        spectrum[row] *= dt
    end
    return nothing
end

"""
    convolve_raman!(drive, frequency, prepared, execution)

Convolve every zero-padded drive column with its prepared Raman spectrum in place.

# Arguments
- `drive`: Caller-owned real matrix `(2n_time, n_columns)`, filled by a drive kernel.
- `frequency`: Caller-owned complex matrix `(n_time + 1, n_columns)`.
- `prepared`: `RamanConvolution`; its kernel includes `dt`, not inverse normalisation.
- `execution`: Synchronous numerical launcher for the spectral products.

# Returns
`nothing`; `drive` contains the doubled-grid convolution. With causal kernels and a
zero-padded drive, its first `n_time` rows equal the discrete causal linear convolution.
No transform plan or scratch storage is created during the call.

Both sequences vanish beyond `n_time`; their linear product has at most `2n_time-1`
samples, so a length-`2n_time` circular convolution cannot wrap. Its retained sample
`i` is `dt * sum(h[i-j+1] * drive[j] for j in 1:i)`.

# Throws
- `DimensionMismatch`: If either buffer does not match the prepared temporal grid.
"""
function convolve_raman!(drive, frequency, prepared::RamanConvolution, execution)
    size(drive, 1) == 2prepared.n_time && size(frequency, 1) == prepared.n_time + 1 &&
        size(drive, 2) == size(frequency, 2) ||
        throw(DimensionMismatch("Raman convolution buffers do not match the prepared grid"))
    LinearAlgebra.mul!(frequency, prepared.forward, drive)
    kernel, scale = prepared.kernel, prepared.inverse_scale
    each!(execution, size(frequency, 1), size(frequency, 2)) do row, column
        frequency[row, column] *= kernel[row] * scale
    end
    LinearAlgebra.mul!(drive, prepared.inverse, frequency)
    return nothing
end

"""
    raman_polarisation!(destination, samples, convolved, layout, execution;
                        factor, window=nothing, count=layout.n_nodes, add=false)

Multiply a scalar Raman history by the physical field and its explicit material factor.

# Arguments
- `destination`, `samples`: Output C/m² and input V/m matrices with `layout`.
- `convolved`: Real matrix `(2n_time, n_nodes)` containing the Raman convolution.
- `layout`: Scalar `ResponseLayout`.
- `execution`: Synchronous numerical launcher.
- `factor`: Translated `epsilon_0` times the outside Raman density weight. Use only
  `epsilon_0` when density was already included in the reconstructed kernel's weights.
- `window`: Optional dimensionless temporal source window of length `n_time`.
- `count`: Active node count. Inactive destination columns are cleared on every call.
- `add`: Add the active polarisation to the destination instead of replacing it.

# Returns
`nothing`. Input and destination may be identical. Warmed CPU calls allocate nothing.

# Throws
- `ArgumentError`: If tensor Raman, invalid count or a nonfinite factor is supplied.
- `DimensionMismatch`: If any buffer/window does not match the layout.
"""
function raman_polarisation!(
        destination, samples, convolved, layout::ResponseLayout, execution;
        factor, window = nothing, count::Int = layout.n_nodes, add::Bool = false,
    )
    layout.n_components == 1 ||
        throw(ArgumentError("Julia Raman currently requires scalar polarisation"))
    _response_matrix(destination, layout)
    _response_matrix(samples, layout)
    _response_count(layout, count)
    _response_window(window, layout)
    size(convolved) == (2layout.n_time, layout.n_nodes) ||
        throw(DimensionMismatch("Raman history does not match its doubled layout"))
    isfinite(factor) || throw(ArgumentError("Raman material factor must be finite"))
    each!(execution, layout.n_time, layout.n_nodes) do row, node
        if node > count
            destination[row, node] = zero(eltype(destination))
        else
            value = factor * samples[row, node] * convolved[row, node] *
                _response_weight(window, row, factor)
            destination[row, node] = add ? destination[row, node] + value : value
        end
    end
    return nothing
end
