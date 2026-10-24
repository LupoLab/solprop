# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``mlx`` backend: single-precision propagation on Apple-silicon GPUs (SEP 33).

Registered as ``"mlx"`` and installed with the ``mlx`` extra on macOS on Apple silicon.
It runs ``precision="single"`` problems only, validated against
:mod:`solprop.backends.numpy_ref.single`, its oracle (ADR-052). The package is named
``mlx_gpu`` so that it does not shadow the ``mlx`` library inside it.

Importing it sets ``MLX_ENABLE_TF32=0`` when the variable is unset, so that MLX computes
float32 matrix products in float32 rather than TF32; MLX reads the setting when its
subsystem first initialises, so a process that used MLX before is checked by
:func:`~solprop.backends.mlx_gpu.arithmetic.gemm_probe` and refused if its products are
TF32.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")
