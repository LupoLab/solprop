# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# NVIDIA GPUs: the engine's arrays become CuArrays, its transforms cuFFT plans and its
# products cuBLAS calls; every pass is the same scalar body as on a CPU.
module SolpropJuliaCUDAExt
using CUDA
import SolpropJulia

SolpropJulia.device_backend(::Val{:cuda}) = (CUDA.allowscalar(false); CUDABackend())
SolpropJulia.device_label(::CUDABackend) = "cuda: " * CUDA.name(CUDA.device())

end
