# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# NVIDIA GPUs: the engine's arrays become CuArrays, its transforms cuFFT plans and its
# products cuBLAS calls; every pass is the same scalar body as on a CPU.
module SolpropJuliaCUDAExt
using CUDA
import SolpropJulia

SolpropJulia.device_backend(::Val{:cuda}) = (CUDA.allowscalar(false); CUDABackend())
SolpropJulia.device_label(::CUDABackend) = "cuda: " * CUDA.name(CUDA.device())
# cuFFT's binary64 transforms of a few thousand points run one block a transform
# (96 us at 5120 on an A40): the engine splits them (`LongFFT`).
SolpropJulia.long_transforms(::CUDABackend) = true
# cuFFT's strided transverse planes are 3.4 times contiguous ones (`Transposed`).
SolpropJulia.transposed_planes(::CUDABackend) = true
# A source's passes as a CUDA graph (`graph_source!`): stream capture records them,
# kernels, cuFFT and cuBLAS alike, on the task's stream.
SolpropJulia.graph_capable(::CUDABackend) = true
SolpropJulia.graph_capture(::CUDABackend, f) = CUDA.instantiate(CUDA.capture(f))
SolpropJulia.graph_launch(exec::CUDA.CuGraphExec) = CUDA.launch(exec)

end
