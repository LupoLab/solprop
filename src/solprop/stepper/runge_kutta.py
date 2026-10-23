# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The interaction-picture Runge-Kutta march as a run's integrator.

A run is stepped by one of two integrators: :class:`RungeKutta`, the embedded explicit
Runge-Kutta pair in the interaction picture (IP-DOPRI5 by default), or
:class:`solprop.stepper.multistep.Multistep`, the low-call Adams method, which is the
default. ``docs/guide/numerics/choosing-a-stepper.md`` compares them.
"""

from dataclasses import dataclass

from solprop.stepper.tableau import DOPRI5, Tableau

__all__ = ["RungeKutta"]


@dataclass(frozen=True)
class RungeKutta:
    """The interaction-picture Runge-Kutta march with an embedded error estimate.

    Every accepted step pays the tableau's stages in source calls (six for DOPRI5, the
    first-same-as-last stage recycled) and keeps no history, where the Adams method
    holds from seven to about sixty further copies of the state, by backend and
    formulation: the integrator to choose when a source evaluation is cheap or the
    state is too large for the Adams history
    (``docs/guide/numerics/choosing-a-stepper.md``).

    Equality is by the tableau's identity, as :class:`~solprop.stepper.tableau.Tableau`
    compares.

    Attributes
    ----------
    tableau : solprop.stepper.tableau.Tableau
        The scheme. Default DOPRI5, whose continuous extension interpolates save planes.

    Examples
    --------
    >>> from solprop.stepper.runge_kutta import RungeKutta
    >>> from solprop.stepper.tableau import TSIT5
    >>> RungeKutta()
    RungeKutta(tableau=DOPRI5)
    >>> RungeKutta(TSIT5).name
    'Tsit5'
    """

    tableau: Tableau = DOPRI5

    def __repr__(self) -> str:
        """Return the representation, naming the tableau.

        Returns
        -------
        str
            ``RungeKutta(tableau=<name>)``.
        """
        return f"RungeKutta(tableau={self.tableau.name})"

    @property
    def name(self) -> str:
        """Return the tableau's name, the method's record name.

        Returns
        -------
        str
            The tableau's name.
        """
        return self.tableau.name
