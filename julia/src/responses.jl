# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public ResponseLayout, KerrCoefficients, kerr_polarisation!

"""
    ResponseLayout(n_time, n_components, n_nodes)

Describe contiguous time rows and component-major columns within each spatial node.

# Arguments
- `n_time`: Positive number of temporal samples.
- `n_components`: Positive number of physical Cartesian components per node.
- `n_nodes`: Positive node capacity. Column `component + n_components * (node - 1)`
  belongs to that component and node; different nodes never share an intensity sum.

# Returns
An immutable layout for a matrix of size `(n_time, n_components * n_nodes)`.
The layout does not itself admit a route or response into a propagation session.

# Throws
- `ArgumentError`: If a count is not positive or the column count overflows `Int`.
"""
struct ResponseLayout
    n_time::Int
    n_components::Int
    n_nodes::Int

    function ResponseLayout(n_time::Integer, n_components::Integer, n_nodes::Integer)
        all(>(0), (n_time, n_components, n_nodes)) ||
            throw(ArgumentError("response layout counts must be positive"))
        all(<=(typemax(Int)), (n_time, n_components, n_nodes)) ||
            throw(ArgumentError("response layout counts must fit Int"))
        n_components <= typemax(Int) ÷ n_nodes ||
            throw(ArgumentError("response column count overflows Int"))
        return new(Int(n_time), Int(n_components), Int(n_nodes))
    end
end

"""
    KerrCoefficients(coefficient, form; self_factor=1/2, conjugate_factor=1/4,
                     harmonic_factor=0, take_real=false)

Hold the translated instantaneous cubic coefficients without evaluating material data.

# Arguments
- `coefficient`: Finite real `epsilon_0 * chi3` per unit stage-density weight, in
  SI units. No analytic-signal factor or stage density is folded into this value.
- `form`: `:real` for `(sum(E_q^2)) E_p`, or `:analytic` for
  `self_factor * sum(abs2(a_q)) * a_p + conjugate_factor * sum(a_q^2) * conj(a_p)`.
- `self_factor`, `conjugate_factor`: Finite dimensionless analytic-cubic factors.
  Their isotropic values are `1/2` and `1/4`; a scalar fundamental then has `3/4`.
- `harmonic_factor`: Finite dimensionless factor multiplying
  `carrier * sum(a_q^2) * a_p` in the analytic form.
- `take_real`: Return the real part after an analytic cubic, as required for a real
  field whose third harmonic is excluded. The caller supplies its analytic signal.

# Returns
An immutable record at the floating-point precision of `coefficient`. Algebraic
factors follow the internal-field conventions and the compiled Kerr contract.

# Throws
- `ArgumentError`: If a coefficient is nonfinite, the form is unknown, or a real
  cubic requests a separate harmonic carrier.
"""
struct KerrCoefficients{T <: AbstractFloat, F}
    coefficient::T
    self_factor::T
    conjugate_factor::T
    harmonic_factor::T
    take_real::Bool
end

function KerrCoefficients(
        coefficient::T, form::Symbol;
        self_factor = T(1) / 2, conjugate_factor = T(1) / 4,
        harmonic_factor = zero(T), take_real::Bool = false,
    ) where {T <: AbstractFloat}
    factors = (coefficient, T(self_factor), T(conjugate_factor), T(harmonic_factor))
    all(isfinite, factors) || throw(ArgumentError("Kerr coefficients must be finite"))
    form in (:real, :analytic) || throw(ArgumentError("unknown Kerr cubic form"))
    form == :real && !iszero(harmonic_factor) &&
        throw(ArgumentError("the real cubic already contains its harmonic"))
    return KerrCoefficients{T, form}(factors..., take_real)
end

function _response_matrix(array, layout)
    size(array) == (layout.n_time, layout.n_components * layout.n_nodes) ||
        throw(DimensionMismatch("response matrix does not match its declared layout"))
    return nothing
end

function _response_count(layout, count)
    0 <= count <= layout.n_nodes ||
        throw(ArgumentError("active response nodes exceed the declared capacity"))
    return nothing
end

function _response_window(window, layout)
    window === nothing || length(window) == layout.n_time ||
        throw(DimensionMismatch("response window does not match the time axis"))
    return nothing
end

@inline _response_weight(::Nothing, row, prototype) = one(prototype)
@inline _response_weight(window, row, prototype) = window[row]
@inline _response_carrier(::Nothing, row, prototype) = zero(prototype)
@inline _response_carrier(carrier, row, prototype) = carrier[row]

"""
    kerr_polarisation!(destination, samples, layout, coefficients, execution;
                       density=1, carrier=nothing, window=nothing,
                       count=layout.n_nodes, add=false)

Write the instantaneous Kerr polarisation into caller-owned time/node storage.

# Arguments
- `destination`: Matrix of size `(n_time, n_components * n_nodes)`, in C/m².
- `samples`: Matrix with the same layout, in V/m: real field samples for `:real`,
  or a complex envelope/analytic signal for `:analytic`.
- `layout`: Explicit `ResponseLayout`; component sums stay within one node.
- `coefficients`: Prepared `KerrCoefficients`, independent of other response profiles.
- `execution`: Synchronous `KernelExecution` used to launch each time/node body.
- `density`: Finite nonnegative stage weight, m⁻³ for per-atom susceptibility or
  dimensionless for an absolute material susceptibility.
- `carrier`: `exp(2im * omega0 * t)` on the internal time grid, length `n_time`.
  Required for a nonzero analytic harmonic coefficient; otherwise optional.
- `window`: Optional real dimensionless temporal source window, length `n_time`.
- `count`: Number of active nodes, between zero and the declared capacity.
- `add`: Add to active destination samples instead of replacing them. Inactive
  columns are always cleared, before any input sample in those columns is read.

# Returns
`nothing`. Identical input and output matrices are supported: both component sums
are formed before any component is overwritten. Otherwise, the input and output
storage must be disjoint. Warmed CPU calls allocate nothing.

# Throws
- `DimensionMismatch`: If storage, window or carrier shapes disagree with the layout.
- `ArgumentError`: If stage density/count is invalid, a required carrier is absent,
  real-cubic samples are complex, or real projection has an incompatible destination.

# Notes
For a real isotropic field, `P = epsilon_0 * chi3 * (E ⋅ E) E`. Expanding the
positive-frequency part of that cubic gives the self/conjugate factors above;
the separately supplied carrier restores the third-harmonic term for an envelope.
No Fourier conjugation or amplitude-normalisation change occurs in this kernel.
"""
function kerr_polarisation!(
        destination, samples, layout::ResponseLayout,
        coefficients::KerrCoefficients{T, F}, execution;
        density = one(T), carrier = nothing, window = nothing,
        count::Int = layout.n_nodes, add::Bool = false,
    ) where {T, F}
    _response_matrix(destination, layout)
    _response_matrix(samples, layout)
    _response_count(layout, count)
    _response_window(window, layout)
    isfinite(density) && density >= 0 ||
        throw(ArgumentError("Kerr stage density must be finite and nonnegative"))
    carrier === nothing || length(carrier) == layout.n_time ||
        throw(DimensionMismatch("Kerr carrier does not match the time axis"))
    !iszero(coefficients.harmonic_factor) && carrier === nothing &&
        throw(ArgumentError("the analytic Kerr harmonic requires a carrier"))
    F == :real && !(eltype(samples) <: Real) &&
        throw(ArgumentError("the real Kerr cubic requires real samples"))
    F == :analytic && !coefficients.take_real && !(eltype(destination) <: Complex) &&
        throw(ArgumentError("the analytic Kerr cubic requires complex destination storage"))
    factor = coefficients.coefficient * T(density)
    components = layout.n_components
    each!(execution, layout.n_time, layout.n_nodes) do row, node
        first = components * (node - 1) + 1
        if node > count
            for component in 0:(components - 1)
                destination[row, first + component] = zero(eltype(destination))
            end
            return nothing
        end
        modulus = zero(T)
        square = zero(eltype(samples))
        for component in 0:(components - 1)
            sample = samples[row, first + component]
            modulus += abs2(sample)
            square += sample * sample
        end
        weight = _response_weight(window, row, factor)
        harmonic = _response_carrier(carrier, row, square)
        for component in 0:(components - 1)
            column = first + component
            sample = samples[row, column]
            value = if F == :real
                factor * square * sample
            else
                factor * (
                    coefficients.self_factor * modulus * sample +
                        coefficients.conjugate_factor * square * conj(sample) +
                        coefficients.harmonic_factor * harmonic * square * sample
                )
            end
            result = weight * (coefficients.take_real ? real(value) : value)
            destination[row, column] = add ? destination[row, column] + result : result
        end
        return nothing
    end
    return nothing
end
