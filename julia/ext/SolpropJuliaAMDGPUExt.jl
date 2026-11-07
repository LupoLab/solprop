# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# AMD GPUs: the engine's arrays become ROCArrays, its transforms rocFFT plans and its
# products rocBLAS calls; every pass is the same scalar body as on a CPU or an NVIDIA GPU.
module SolpropJuliaAMDGPUExt
using AMDGPU
import SolpropJulia

SolpropJulia.device_backend(::Val{:rocm}) = (AMDGPU.allowscalar(false); ROCBackend())
SolpropJulia.device_label(::ROCBackend) = "rocm: " * string(AMDGPU.device())

end
