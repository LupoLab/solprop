# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Julia backend experiment: CPU, or an NVIDIA GPU with ``SOLPROP_JULIA_DEVICE``."""

from solprop.backends.julia._backend import JuliaBackend
from solprop.backends.julia._runtime import source_digest

__all__ = ["JuliaBackend", "source_digest"]
