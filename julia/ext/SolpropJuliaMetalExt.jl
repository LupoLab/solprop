# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Apple GPUs: the engine's arrays become MtlArrays, its transforms MPSGraph plans and
# its products Metal.jl's; every pass is the same scalar body as on a CPU. Metal has no
# binary64: the engine runs binary32 with the operator's exponents and the Adams flow
# in double-float (dfloat.jl), and refuses anything else at construction.
module SolpropJuliaMetalExt
using Metal
import SolpropJulia

SolpropJulia.device_backend(::Val{:metal}) = (Metal.allowscalar(false); MetalBackend())
SolpropJulia.device_label(::MetalBackend) = "metal: " * String(Metal.device().name)

end
