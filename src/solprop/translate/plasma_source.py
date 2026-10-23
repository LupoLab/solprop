# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Certify and identify the held field-plasma source (SEP 27).

The certificate tests the collar's steady-current identity, not arbitrary product
aliasing. Its denominator is the complete fine-grid transform, including DC.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np

from solprop.exceptions import TranslationError
from solprop.translate.masks import Grids

__all__ = ["PLASMA_SOURCE_CERTIFICATE_LIMIT", "plasma_source_record"]

PLASMA_SOURCE_CERTIFICATE_LIMIT = 1e-10
"""Largest admitted steady-current residual, relative to the full collar transform."""

# Only the private benchmark harness sets this, for a new simulation and its entire
# run. Backends capture it at construction; the driver binds it into the run identity.
_BENCHMARK_GATED: ContextVar[bool] = ContextVar("plasma_benchmark_gated", default=False)


@dataclass(frozen=True)
class _Certificate:
    """Retain one grid's scalar admission evidence without retaining its arrays."""

    residual: float
    leading: int
    trailing: int


_CERTIFICATES: WeakKeyDictionary[Grids, _Certificate] = WeakKeyDictionary()


def _certificate(grids: Grids) -> _Certificate:
    """Compute one immutable grid's certificate once.

    Parameters
    ----------
    grids : Grids
        Fine collar and derivative, active angular frequencies and crop indices.

    Returns
    -------
    _Certificate
        Dimensionless full-transform residual and each taper's sample count.

    Raises
    ------
    TranslationError
        If an active frequency is nonpositive or the collar is identically zero.
    """
    cached = _CERTIFICATES.get(grids)
    if cached is not None:
        return cached
    gate = np.fft.rfft(grids.t_window_fine)
    slope = np.fft.rfft(grids.t_window_fine_derivative)
    active = grids.active_mask
    omega = grids.omega[active]
    if np.any(omega <= 0.0):
        raise TranslationError(
            "the held plasma source requires positive active frequencies"
        )
    crop = grids.crop_index[active]
    denominator = float(np.linalg.norm(gate))
    if denominator == 0.0:
        raise TranslationError(
            "the held plasma source requires a nonzero temporal collar"
        )
    half = grids.n_to // 2
    record = _Certificate(
        float(np.linalg.norm(gate[crop] - slope[crop] / (1j * omega))) / denominator,
        int(np.count_nonzero(grids.t_window_fine[:half] < 1.0)),
        int(np.count_nonzero(grids.t_window_fine[half:] < 1.0)),
    )
    _CERTIFICATES[grids] = record
    return record


def plasma_source_record(grids: Grids) -> dict[str, object]:
    """Validate and describe the held plasma source's collar certificate.

    Parameters
    ----------
    grids : solprop.translate.masks.Grids
        Immutable field-grid arrays, including the analytic collar derivative.

    Returns
    -------
    dict
        The source form, dimensionless certificate and limit, and the number of
        samples below the plateau on each half of the temporal lattice.

    Raises
    ------
    TranslationError
        If the steady-current identity is unresolved or an active frequency is zero.

    Notes
    -----
    SEP 27 sections 4.4--4.6 derive the identity and its operand-based error budget.
    The private historical benchmark uses its original collar and equation; its
    record is labelled non-resumable and does not claim the held-current certificate.
    """
    if _BENCHMARK_GATED.get():
        return {"form": "gated_benchmark", "resumable": False}
    certificate = _certificate(grids)
    value = certificate.residual
    tapers = [certificate.leading, certificate.trailing]
    if not np.isfinite(value) or value > PLASMA_SOURCE_CERTIFICATE_LIMIT:
        raise TranslationError(
            f"held plasma current collar certificate {value:.3e} exceeds "
            f"{PLASMA_SOURCE_CERTIFICATE_LIMIT:.1e}; taper samples {tapers}; "
            "use a wider temporal window or wider temporal tapers"
        )
    return {
        "form": "held_current",
        "certificate": value,
        "limit": PLASMA_SOURCE_CERTIFICATE_LIMIT,
        "taper_samples": tapers,
    }
