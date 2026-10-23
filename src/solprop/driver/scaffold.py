# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A fixed-step interaction-picture integrator for the mode-averaged GNLSE.

Not the solver, and not how a simulation is run: that is
:func:`solprop.driver.interface.prop_gnlse` and
:class:`solprop.driver.simulation.Simulation`. This is a second, independent integrator
of the same physics, which is why it **stays** now that the driver has landed: it
composes the nonlinear term from the physics layer and steps it with the classical
interaction-picture RK4, sharing nothing with the reference backend's passes but the
response itself, so agreement between the two checks the stage combination and the
propagator factors rather than the physics they share
(``tests/backends/test_numpy_ref_soliton.py``). It is plain NumPy, fixed step, envelope
grid, mode-averaged, Kerr only, and allocates freely.

The scheme is the classical fourth-order Runge-Kutta method in the interaction picture
[1]_: the linear flow ``exp(D h)`` is applied exactly, which the diagonal operator
allows (``docs/guide/numerics/linear-operator.md``), and only the nonlinear term is
integrated
numerically.

References
----------
.. [1] J. Hult, "A fourth-order Runge-Kutta in the interaction picture method for
       simulating supercontinuum generation in optical fibers", J. Lightwave Technol.
       25, 3770 (2007). doi:10.1109/JLT.2007.909373
"""

import logging
import math
from collections.abc import Callable

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.exceptions import GridError, StepperError
from solprop.grid import TimeGrid
from solprop.linear import Frame, dispersion_operator
from solprop.nonlinear.gnlse import GnlseProblem
from solprop.nonlinear.normalisation import mode_average_prefactor, sqrtw_to_vm

__all__ = ["gnlse_source", "propagate_fixed_rk4"]

logger = logging.getLogger(__name__)


def gnlse_source(
    problem: GnlseProblem, grid: TimeGrid
) -> Callable[[Complex[np.ndarray, " n_omega"]], Complex[np.ndarray, " n_omega"]]:
    """Return the scaffold's nonlinear term of the mode-averaged GNLSE.

    The term :func:`propagate_fixed_rk4` integrates, composed from the physics layer
    alone (``docs/dev/architecture.md`` section 2): the spectrum is oversampled onto
    the fine grid, transformed, scaled to volts per metre, handed to the Kerr response,
    and the polarisation is transformed back, cropped and multiplied by
    :func:`solprop.nonlinear.normalisation.mode_average_prefactor`. Exposed so that one
    evaluation can be compared with a backend's (SEP 24 section 7.1).

    Parameters
    ----------
    problem : solprop.nonlinear.gnlse.GnlseProblem
        The fibre and the nonlinearity.
    grid : solprop.grid.TimeGrid
        An envelope grid.

    Returns
    -------
    collections.abc.Callable
        ``N[state]``: a stored spectrum of shape ``(n_omega,)`` in the square root of
        watts to the nonlinear term, same shape, in the square root of watts per metre.

    Raises
    ------
    GridError
        If the grid is not an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.driver.scaffold import gnlse_source
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> problem = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0,
    ...                        gamma=0.11, shock=False)
    >>> source = gnlse_source(problem, grid)
    >>> bool(np.all(source(np.zeros(grid.n_omega, dtype=complex)) == 0.0))
    True
    """
    if grid.kind != "envelope":
        msg = (
            "the scaffold propagates envelope grids only; a field grid has no path here"
        )
        raise GridError(msg)
    prefactor = mode_average_prefactor(
        problem.mode, grid, shock=problem.shock, index=problem.prefactor_index
    )
    to_field = sqrtw_to_vm(problem.aeff)
    n_t, n_to = grid.n_t, grid.n_to

    def nonlinear(
        state: Complex[np.ndarray, " n_omega"],
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the nonlinear term of the propagation equation for one spectrum.

        Parameters
        ----------
        state : numpy.ndarray
            Stored spectrum, shape ``(n_omega,)``, in the square root of watts.

        Returns
        -------
        numpy.ndarray
            ``N[state]``, shape ``(n_omega,)``, in the square root of watts per metre.
        """
        fine = fourier.oversample(state, n_to, kind="envelope")
        envelope = fourier.inverse(fine, kind="envelope") * to_field
        polarisation = problem.kerr.evaluate(envelope[np.newaxis], 1.0, kind="envelope")
        transformed = fourier.forward(polarisation[0], kind="envelope")
        return prefactor * fourier.crop(transformed, n_t, kind="envelope")

    return nonlinear


def propagate_fixed_rk4(
    problem: GnlseProblem,
    grid: TimeGrid,
    spectrum: Complex[np.ndarray, " n_omega"],
    z_end: float,
    n_steps: int,
    *,
    n_save: int = 1,
) -> tuple[Float[np.ndarray, " n_z"], Complex[np.ndarray, "n_z n_omega"]]:
    """Propagate a mode-averaged envelope spectrum with fixed-step IP-RK4.

    The state is the stored spectrum in the internal FFT convention, in the square root
    of watts, as :mod:`solprop.pulses` builds it. Each step applies the exact linear
    factor ``exp(D h / 2)`` around the four nonlinear evaluations of the classical
    Runge-Kutta scheme. The nonlinear term is :func:`gnlse_source`'s. The linear
    operator is evaluated once, at ``z = 0``, which is exact for a dispersion-specified
    mode because it does not vary along the fibre.

    The state is masked to the grid's active band once at the start. Outside
    the band both the operator and the prefactor are zero, so nothing there ever
    changes.

    A fibre that varies along z is refused: the operator and the area are evaluated
    once here, and a varying effective area belongs to the tabulated route of
    :func:`solprop.driver.interface.prop_gnlse`.

    Parameters
    ----------
    problem : solprop.nonlinear.gnlse.GnlseProblem
        The fibre and the nonlinearity, from
        :func:`solprop.nonlinear.gnlse.problem_from`.
    grid : solprop.grid.TimeGrid
        An envelope grid. A field grid is refused: this scaffold has no field path.
    spectrum : numpy.ndarray
        Initial stored spectrum, shape ``(n_omega,)``, complex, in the square root of
        watts and the internal convention.
    z_end : float
        Propagation length, m, positive.
    n_steps : int
        Number of equal steps, at least 1 and a multiple of ``n_save``.
    n_save : int, optional
        Number of planes to save after ``z = 0``, at least 1. Default 1: the end only.

    Returns
    -------
    tuple
        ``(z, spectra)``: the saved positions, shape ``(n_save + 1,)``, m, starting at
        zero; and the stored spectra there, shape ``(n_save + 1, n_omega)``.

    Raises
    ------
    GridError
        If the grid is not an envelope grid, or the spectrum does not match it.
    StepperError
        If the fibre varies along z, ``z_end`` is not positive and finite,
        ``n_steps`` or ``n_save`` is below 1, or ``n_steps`` is not a multiple of
        ``n_save``.

    Examples
    --------
    A fundamental soliton is stationary up to the phase ``-i xi / 2`` of the internal
    convention, ``xi`` being the distance in dispersion lengths:

    >>> import numpy as np
    >>> from solprop.driver.scaffold import propagate_fixed_rk4
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> beta2, gamma, tau0 = -11.83e-27, 0.11, 100e-15
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> problem = problem_from(
    ...     [0.0, 0.0, beta2], omega_ref=grid.omega0, gamma=gamma, shock=False
    ... )
    >>> peak = abs(beta2) / (gamma * tau0**2)
    >>> start = SechPulse(wavelength=835e-9, width=tau0, peak_power=peak).build(grid)
    >>> z, spectra = propagate_fixed_rk4(
    ...     problem, grid, start[0, 0], tau0**2 / abs(beta2), 200
    ... )
    >>> error = np.linalg.norm(spectra[-1] - np.exp(-0.5j) * spectra[0])
    >>> bool(error / np.linalg.norm(spectra[0]) < 1e-9)
    True
    """
    if not problem.mode.z_constant:
        msg = (
            "the scaffold evaluates the linear operator and the effective area once, "
            "at z = 0, so its fibre must be z-constant; this one varies along z. Run "
            "a varying-area fibre through prop_gnlse or Simulation, which tabulate it"
        )
        raise StepperError(msg)
    nonlinear = gnlse_source(problem, grid)
    values = np.asarray(spectrum, dtype=np.complex128)
    if values.shape != (grid.n_omega,):
        msg = f"spectrum has shape {values.shape}, expected {(grid.n_omega,)}"
        raise GridError(msg)
    if not (math.isfinite(z_end) and z_end > 0.0):
        msg = f"z_end must be positive and finite, got {z_end!r}"
        raise StepperError(msg)
    if n_steps < 1:
        msg = f"n_steps must be at least 1, got {n_steps!r}"
        raise StepperError(msg)
    if n_save < 1:
        msg = f"n_save must be at least 1, got {n_save!r}"
        raise StepperError(msg)
    if n_steps % n_save:
        msg = f"n_steps ({n_steps}) must be a multiple of n_save ({n_save})"
        raise StepperError(msg)

    operator = dispersion_operator(problem.mode, grid, Frame.comoving(grid))
    step = z_end / n_steps
    half = np.exp(0.5 * step * operator)
    per_save = n_steps // n_save
    state = values * grid.active_mask
    saved = np.empty((n_save + 1, grid.n_omega), dtype=np.complex128)
    saved[0] = state
    logger.debug(
        "scaffold IP-RK4: %d steps of %.6e m over %.6e m, saving %d planes",
        n_steps,
        step,
        z_end,
        n_save,
    )
    for index in range(n_steps):
        interaction = half * state
        k1 = half * nonlinear(state)
        k2 = nonlinear(interaction + 0.5 * step * k1)
        k3 = nonlinear(interaction + 0.5 * step * k2)
        k4 = nonlinear(half * (interaction + step * k3))
        state = half * (interaction + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3))
        state = state + step / 6.0 * k4
        if (index + 1) % per_save == 0:
            saved[(index + 1) // per_save] = state
    return np.linspace(0.0, z_end, n_save + 1), saved
