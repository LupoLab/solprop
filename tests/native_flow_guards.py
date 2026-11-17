# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native coupled-flow guards' host transcription, the oracle their refusals match.

The native state refuses a flow action whose exponent needs too many squarings, and a
step whose measured conditioning exceeds the flow's certificate, with the messages
these functions raise (``Workspace::guard_flow``). Until SEP 47's phase 5 the backend
also ran them itself on its host-packed Runge-Kutta trial; that path is gone, and the
transcription lives here, so the native guards stay held to solprop's reference rule
(``numpy_ref.coupled_flow._flow_squarings`` and ``check_flow_conditioning``).
"""

import math
from typing import Any

import numpy as np

from solprop.backends.exponential import FLOW_MAX_SQUARINGS, FLOW_TAYLOR_SCALE
from solprop.exceptions import BackendError


def flow_squarings(
    largest: float,
    *,
    scale: float = FLOW_TAYLOR_SCALE,
    admitted: int = FLOW_MAX_SQUARINGS,
) -> None:
    """Refuse a length whose exponent would need more squarings than admitted.

    Parameters
    ----------
    largest : float
        The section's largest row-sum norm times the length.
    scale : float, optional
        The row-sum norm the exponent is scaled below, solprop's by default.
    admitted : int, optional
        The most squarings admitted, solprop's by default.

    Raises
    ------
    solprop.exceptions.BackendError
        Past the admitted squarings, with the native message.
    """
    squarings = 0
    if largest > scale:
        squarings = math.ceil(math.log2(largest / scale))
    if squarings > admitted:
        raise BackendError(
            f"native cannot take a coupled flow action whose exponent has row-sum "
            f"norm {largest:.4g}: it would need {squarings} squarings, above the "
            f"{admitted} this backend admits. Cap dz_max"
        )


def check_flow_conditioning(
    backend: Any, flow: Any, section: int, measured: np.ndarray, dz: float
) -> None:
    """Refuse a step whose measured ``norm(U) norm(U^-1)`` exceeds the certificate.

    Parameters
    ----------
    backend : module
        ``solprop_native.backend``, whose ``_flow_allowance`` is read.
    flow : solprop_native.backend._HostFlow
        The decomposed flow.
    section : int
        The step's section.
    measured : numpy.ndarray
        The engine's measurement per active bin.
    dz : float
        The step, m.

    Raises
    ------
    solprop.exceptions.BackendError
        Above the certificate times the rounding allowance, with the native message.
    """
    if not measured.size:
        return
    declared = flow.condition_bound[section]
    allowance = backend._flow_allowance(int(flow.vectors.shape[-1]))
    excess = measured / (declared * allowance)
    worst = int(np.argmax(excess))
    if excess[worst] > 1.0:
        raise BackendError(
            f"native measured a coupled flow conditioning of "
            f"{float(measured[worst]):.12g} over a step of {dz:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at "
            f"active bin {worst} of section {section} -- exceeding it by a factor "
            f"{float(excess[worst]):.12g} of the rounding allowance: the inverse "
            "action carries the nonlinear source back by the measured factor. Lower "
            "dz_max, which is what the bound was declared over"
        )
