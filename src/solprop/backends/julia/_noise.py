# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own one supplied binary64 input-noise draw without changing its model or units."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex

from solprop.backends.julia._descriptors import owned
from solprop.exceptions import BackendError, TranslationError
from solprop.noise import NoiseRealisation
from solprop.translate.masks import noise_spectrum
from solprop.translate.problem import PropagationProblem

__all__ = ["InputNoiseTransfer", "lower_input_noise"]


@dataclass(frozen=True, eq=False)
class InputNoiseTransfer:
    """Own the selected draw and the problem's absolute launch position.

    Attributes
    ----------
    spectrum : numpy.ndarray
        Independent read-only Fortran-contiguous complex128 storage, shape
        ``(n_omega,n_columns)``, where columns enumerate components inside modes.
        Values retain the internal descriptor's spectral amplitude units and FFT
        convention. No power, transform or noise-model normalization is applied.
    launch : float
        Problem launch position in m. Julia adds the draw only for a fresh start
        at this exact position; the same draw remains available to the monitor
        floor on resume.
    """

    spectrum: Complex[np.ndarray, "n_omega n_columns"]
    launch: float


def lower_input_noise(
    problem: PropagationProblem, *, noise: NoiseRealisation | None = None
) -> InputNoiseTransfer | None:
    """Copy the supplied input draw according to the shared initial-state policy.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Translated problem whose initial descriptor owns the active noise model,
        default draw, shape and launch position. Active noise requires binary64.
    noise : solprop.noise.NoiseRealisation or None, optional
        Per-start replacement draw in internal descriptor units. The shared
        ``noise_spectrum`` rule broadcasts singleton mode and component axes.
        Its model metadata does not change the problem's active model, exactly
        as in ``HostDriver.set_initial``. An inactive model ignores this argument.

    Returns
    -------
    InputNoiseTransfer or None
        Owned numerical data, or ``None`` for an inactive noise model. The result
        retains no problem, start, realisation or Python callback.

    Raises
    ------
    BackendError
        If modified noise, active binary32 noise, an incompatible shape or a
        nonfinite draw is requested. These refusals precede Julia allocation.

    Notes
    -----
    This helper draws no random numbers and performs no field addition, masking
    or phase evaluation. Julia owns those numerical operations and their resume
    predicate. An override replaces only the draw, including when its metadata
    names the other shared model.
    """
    initial = problem.initial
    if initial.noise_model == "none":
        return None
    if initial.noise_model != "input":
        raise BackendError("julia input noise does not support the modified model")
    if problem.precision.single:
        raise BackendError("julia input noise requires binary64 precision")
    try:
        spectrum = (
            initial.noise0 if noise is None else noise_spectrum(noise, problem.shape)
        )
    except TranslationError as error:
        msg = f"julia input noise shape is incompatible: {error}"
        raise BackendError(msg) from error
    if spectrum is None or spectrum.shape != problem.shape:
        raise BackendError("julia input noise requires a draw matching the state shape")
    if not np.all(np.isfinite(spectrum)):
        raise BackendError("julia input noise requires a finite draw")
    frequency_first = spectrum.reshape(-1, problem.shape[-1]).T
    return InputNoiseTransfer(owned(frequency_first), float(initial.z))
