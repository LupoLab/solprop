# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPU backend: fused numba kernels over FFTW plans.

The execution chosen by a bake-off of candidate backends (``docs/dev/backends.md``
section 5): the seven numbered passes of ``architecture.md`` section 8 collapsed into a
handful of compiled kernels, the transforms planned once by FFTW on buffers that never
move, and all five thread pools owned rather than inherited.

- :mod:`solprop.backends.numba_cpu.kernels` -- the fused stage kernels, built from the
  per-element bodies of :mod:`solprop.backends.bodies`, which every backend shares.
- :mod:`solprop.backends.numba_cpu.fft` -- the plans, the wisdom and the pools.
- :mod:`solprop.backends.numba_cpu.backend` -- the seam and the stage driver.
"""

from solprop.backends.numba_cpu.backend import NumbaCpuBackend

__all__ = ["NumbaCpuBackend"]
