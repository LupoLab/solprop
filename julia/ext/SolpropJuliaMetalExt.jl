# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SolpropJuliaMetalExt

import Metal
import KernelAbstractions
import LinearAlgebra
import SolpropJulia

include("metal_execution.jl")
include("metal_transforms.jl")

end
