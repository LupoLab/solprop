# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The mixed-precision path of the reference backend: SEP 36's oracle.

:class:`NumpyMixedBackend` holds everything that touches the propagated state in
binary64 and evaluates the source in binary32, in the operation order of
``docs/dev/seps/sep-0036-native-reduced-precision.md`` section 4.2. Per stage ``i``:

1. *Stage input, binary64:* ``y_i = exp(dPhi_i) (A + dz sum_j a_ij k_j)``, with the
   state, every stage value and the propagator binary64; the modified noise model's
   phase-transported realisation is added here, in binary64
   (:func:`solprop.backends.numpy_ref.passes.stage_input`).
2. *Narrowing, one rounding:* ``fl32(y_i / s)``, ``s`` the launch's exact power-of-two
   scale (:func:`solprop.translate.precision.state_scale`), so the division is exact.
3. *Source, binary32:* SEP 33's single-precision source evaluation of the scaled input
   -- the transforms, the responses with their running integrals in SEP 33's orders,
   the in-stage absorber terms and the matrix products, with the tables narrowed at
   construction -- returning ``N(s y) / s`` in binary32. It is the source of
   :class:`~solprop.backends.numpy_ref.single.NumpySingleBackend`, reached through the
   same hooks with a unit inverse propagator.
4. *Widening and pull-back, binary64:* ``k_i = exp(-dPhi_i) * s * wide(n_i)``. The
   widening and the multiplication by ``s`` are exact; the inverse propagator is
   applied in binary64 and the stage value stored in binary64.

Everything else is the reference's binary64 path: the state, the propagators and their
inverses, the stage combinations, every row the march stores in the interaction
variable (the Runge-Kutta stages, the low-storage registers, the Adams history, the
dense-output rows), the band mask and the split-step absorber of ``_finish`` with its
transforms, the error norm and the statistics. So no step's damping is restricted:
``exp(-dPhi)`` multiplies binary64 values only.

**Dtype guards.** The state, every stage value and the accepted state are checked
complex128 at each trial; the source's input must be the narrowing of a complex128
array and its output complex64, so a binary64 array inside the source or a binary32
array in a stored row is refused by name (SEP 36 section 6.2).

The oracle's own binary64 host evaluations inside the source (the ionisation rate's
table lookup, the envelope-grid plasma's cycle moments) carry over from SEP 33
unchanged (SEP 36 section 4.5).
"""

from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Complex

from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEBinding, TDSEPurpose

__all__ = ["NumpyMixedBackend"]

# The dtype of the state and of every row stored beside it.
_WIDE = np.dtype(np.complex128)


class NumpyMixedBackend(NumpySingleBackend):
    """Propagate a mixed-precision problem: a binary64 state, a binary32 source.

    Constructed by the registry for ``backend="numpy_ref"`` when the problem's
    precision is mixed. See the module docstring for the operation order. The source's
    tables, coefficients and passes are the single-precision oracle's; every hook that
    touches the state is the binary64 reference's.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated problem with ``precision.mixed``.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, needed only by :meth:`stats`.
    modes : collections.abc.Sequence, optional
        The propagated modes, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Must be ``None``: TDSE responses are refused under mixed precision.

    Raises
    ------
    BackendError
        If the problem is not mixed precision, or a source table cannot be narrowed
        without losing range that its pass does not carry.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.backends.registry import select
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=50e-15,
    ...     peak_power=10.0, pulseshape="sech", backend="numpy_ref",
    ...     precision="mixed", save_planes=[0.01],
    ... )
    >>> problem = simulation.translate()
    >>> backend = select("numpy_ref", problem, grid=simulation.grid)
    >>> type(backend).__name__, backend.scale
    ('NumpyMixedBackend', 2.0)
    >>> backend.set_initial(problem.initial.field0, None)
    >>> backend.propagate_to(0.01).met_tolerance
    True
    >>> backend.observe().field.dtype
    dtype('complex128')
    >>> backend.close()
    """

    precision_name = "mixed"

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
    ) -> None:
        """Adopt a mixed-precision problem, narrowing its source tables only."""
        super().__init__(
            problem,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
        )
        # The state is physical and binary64: the controller's own tolerances, and a
        # lab-frame unit propagator of the state's precision.
        self._controller = problem.stepper.controller
        self._lab_frame = np.broadcast_to(
            np.ones((), dtype=np.complex128), problem.shape
        )
        # The unit inverse propagator the binary32 source evaluation is called with:
        # the source returns its lab-frame value, pulled back in binary64 afterwards.
        self._source_frame = np.broadcast_to(
            np.ones((), dtype=self._complex), problem.shape
        )

    # The state and what crosses to the host: the reference's.

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: object,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state in binary64, with its noise realisation.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout, physical units.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry, or ``None`` for the problem's own; drawn,
            transported and added in binary64 (SEP 36 section 4.4).
        z : float, optional
            The position the state belongs to, m.
        """
        NumpyRefBackend.set_initial(
            self,
            field,
            noise,  # pyright: ignore[reportArgumentType]
            z=z,
        )

    def _observed_field(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the state, which is physical and binary64.

        Parameters
        ----------
        state : numpy.ndarray
            The complex128 state.

        Returns
        -------
        numpy.ndarray
            `state`.
        """
        return NumpyRefBackend._observed_field(self, state)

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect statistics from the binary64 state, as the reference does.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            The statistics to collect.

        Returns
        -------
        dict
            As :meth:`solprop.backends.numpy_ref.backend.NumpyRefBackend.stats`.
        """
        return NumpyRefBackend.stats(self, spec)

    # The linear part, the norm and the end of a step: the reference's, in binary64.

    def _propagator(
        self, exponent: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``exp(dPhi)`` in binary64 (SEP 36 section 4.3).

        Parameters
        ----------
        exponent : numpy.ndarray
            The binary64 exponent or its negative.

        Returns
        -------
        numpy.ndarray
            The complex128 propagator.
        """
        return NumpyRefBackend._propagator(self, exponent)

    def _error_norm(
        self,
        error: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        advanced: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> float:
        """Return the reference's error norm, reduced from the binary64 state.

        Parameters
        ----------
        error : numpy.ndarray
            The embedded difference carried to the end of the step.
        state : numpy.ndarray
            The state at the anchor.
        advanced : numpy.ndarray
            The trial state at the end of the step.

        Returns
        -------
        float
            The norm the controller reads.
        """
        return NumpyRefBackend._error_norm(self, error, state, advanced)

    def _finish(
        self,
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        dz: float,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Mask the accepted state and apply a split-step collar, in binary64.

        The split-step absorber's transforms are the reference's binary64 ones: a
        once-per-step treatment acts on the whole state, where a binary32 rounding
        would enter it unscaled (SEP 36 section 4.2).

        Parameters
        ----------
        state : numpy.ndarray
            The accepted state.
        dz : float
            The step's length, m.

        Returns
        -------
        numpy.ndarray
            The finished complex128 state.
        """
        finished = NumpyRefBackend._finish(self, state, dz)
        _require(np.asarray(finished), _WIDE, "the finished state")
        return finished

    def _adams_adopt(
        self, advanced: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"], dz: float
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Adopt an Adams step's binary64 state.

        The Adams history is binary64 under mixed precision, as the state is (SEP 36
        section 4.6; admitted in phase 5, Q2).

        Parameters
        ----------
        advanced : numpy.ndarray
            The accepted state.
        dz : float
            The step's length, m.

        Returns
        -------
        numpy.ndarray
            The finished state.
        """
        _require(np.asarray(advanced), _WIDE, "the Adams state")
        return NumpyRefBackend._adams_adopt(self, advanced, dz)

    def _trial_step(
        self, z: float, dz: float
    ) -> tuple[float, Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]:
        """Take the reference's trial step, then check every stored row is binary64.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the trial step, m.

        Returns
        -------
        tuple
            The scaled error norm and the state at the end of the step.

        Raises
        ------
        BackendError
            If the state or a stage value is not complex128: a binary32 array reached
            a row stored in the interaction variable (SEP 36 section 4.2).
        """
        norm, advanced = NumpyRefBackend._trial_step(self, z, dz)
        _require(advanced, _WIDE, "the trial state")
        for index, stage in enumerate(self._last_stages):
            _require(stage, _WIDE, f"stage {index + 1}")
        return norm, advanced

    # The source: narrowed once, evaluated in binary32, pulled back in binary64.

    def _nonlinear(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate one stage value in SEP 36's operation order (section 4.2).

        Parameters
        ----------
        lab : numpy.ndarray
            The binary64 lab-frame stage input, noise included.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        inverse_propagator : numpy.ndarray
            ``exp(-dPhi_i)``, binary64.
        purpose : str, optional
            Source-evaluation label; unused, TDSE responses being refused.

        Returns
        -------
        numpy.ndarray
            ``exp(-dPhi_i) * s * wide(N32(fl32(lab / s)))``, complex128.

        Raises
        ------
        BackendError
            If the input is not complex128 or the source's value is not complex64.
        """
        del purpose  # TDSE responses are refused under mixed precision
        _require(np.asarray(lab), _WIDE, "the stage input")
        narrowed = (lab * (1.0 / self._scale)).astype(self._complex)
        source = super()._nonlinear(narrowed, bundle, self._source_frame)
        _require(np.asarray(source), self._complex, "the source evaluation")
        return inverse_propagator * (source.astype(np.complex128) * self._scale)


def _require(values: np.ndarray, dtype: np.dtype, what: str) -> None:
    """Refuse an array of the wrong precision, by name.

    Parameters
    ----------
    values : numpy.ndarray
        The array.
    dtype : numpy.dtype
        The precision it must have.
    what : str
        What it is, for the message.

    Raises
    ------
    BackendError
        If the dtype differs.
    """
    if values.dtype != dtype:
        stored = dtype == np.complex128
        msg = (
            f"the mixed-precision path computed {what} as {values.dtype}, not "
            f"{dtype}: "
            + (
                "a binary32 array reached a row held in binary64"
                if stored
                else "a binary64 array reached the binary32 source"
            )
        )
        raise BackendError(msg)
