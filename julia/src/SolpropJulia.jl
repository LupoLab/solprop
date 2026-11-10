# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    SolpropJulia

Execute translated propagation data in a Julia-owned engine.

The engine uses SI units and the internal Fourier convention defined in solprop's
developer conventions. Python objects enter only through the optional PythonCall extension.
"""
module SolpropJulia

import AcceleratedKernels
import Adapt
import Atomix
import FFTW
import KernelAbstractions
import LinearAlgebra
import Polyester
import PrecompileTools
import StrideArraysCore

public copy_preparation
public StopFlag, stop_address, stop_requested, stop_setter_address

include("cancellation.jl")
include("execution.jl")
include("phi_functions.jl")
include("transforms.jl")
include("tables.jl")
include("power_weights.jl")
include("operators.jl")
include("runge_kutta.jl")
include("adams.jl")
include("low_storage.jl")
include("control.jl")
include("responses.jl")
include("raman.jl")
include("plasma.jl")
include("statistics.jl")
include("monitor.jl")
include("source.jl")
include("composition.jl")
include("march.jl")
include("transfer.jl")

"""
    copy_preparation(values::AbstractArray)

Copy translated numerical data into independently owned Julia storage.

# Arguments
- `values`: An array of fixed-size numerical values, of any dimension, in the translated
  input's units. Dimensions, element type and values are preserved; no unit conversion or
  propagation admission is performed.

# Returns
An `Array` with the same size and element type. The result retains no input reference.

# Throws
- `ArgumentError`: If the element type is not a fixed-size subtype of `Number`.
"""
function copy_preparation(values::AbstractArray{T, N}) where {T, N}
    if !(isbitstype(T) && T <: Number)
        throw(ArgumentError("preparation arrays require fixed-size numerical elements"))
    end
    owned = Array{T, N}(undef, size(values))
    copyto!(owned, values)
    return owned
end

end
