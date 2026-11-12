# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run's input: its pulse, its launch, its shot noise and its peak field.

The keyword routes shape their pulses and draw their noise here; a derived run's input
is checked and installed here; and the input's peak field sets the range of each
optmat ionisation rate a run carries (SEP 3).
"""

import dataclasses
import functools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from jaxtyping import Complex

from solprop.coupling import (
    CouplingLaunch,
    CouplingModel,
    FieldPlane,
    PlaneGeometry,
    couple,
)
from solprop.driver._launch import launch_pulse
from solprop.driver.defaults import NoiseModel
from solprop.driver.routes.keywords import _PULSE_KEYWORDS, _about
from solprop.driver.settings import SimulationSettings
from solprop.exceptions import CouplingError, DriverError
from solprop.fourier import power_amplitude
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.modes.collection import ModeCollection
from solprop.modes.guide import ModalGuide
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import TransverseQuadrature, mode_matrix, on_axis_rule
from solprop.modes.slab import PlanarGrid
from solprop.noise import (
    NoiseRealisation,
    input_noise,
    modified_noise,
    spatial_noise,
    transformed_noise,
)
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import (
    OptmatRate,
    bind_reference_field,
)
from solprop.nonlinear.responses import PlasmaResponse
from solprop.output.schema import _recorded_inline
from solprop.pulses import (
    GaussianPulse,
    PlanarPulse,
    Pulse,
    PulseSum,
    SechPulse,
    SpatialPulse,
    SpatialPulseSum,
)
from solprop.spatial import SpatialFlux, SpatialMedium, peak_field
from solprop.stats import analytic_field
from solprop.translate.linop import power_weight_at
from solprop.translate.masks import noise_spectrum
from solprop.translate.problem import (
    PropagationProblem,
    _held_block,
    _held_state,
    _hold,
    _state,
)

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["ModalLaunch", "default_field_range"]

logger = logging.getLogger(__name__)

_FIELD_RANGE_BELOW = 1e-3
"""Lower end of the default ionisation table, as a factor on the input's peak field."""

_FIELD_RANGE_ABOVE = 30.0
"""Upper end of the default ionisation table, as a factor on the input's peak field."""

ModalLaunch = (
    Mapping[str | int, float | Mapping[str, float]]
    | Sequence[Mapping[str | int, float | Mapping[str, float]]]
)
"""What a modal constructor's ``launch`` keyword takes (SEP 41, S5).

A mapping from a mode's display label or index to its share, or to
``{"share": s, "phase": phi}`` with the phase in radians; or a list of such mappings,
one for each input pulse (``docs/guide/setup/pulses.md``).
"""


def _realisation(problem: PropagationProblem) -> NoiseRealisation | None:
    """Return a derived problem's noise as the backend loads it, or ``None``.

    - Returns: its realisation, or ``None`` without noise.
    """
    initial = problem.initial
    if initial.noise0 is None or initial.noise_model == "none":
        return None
    return NoiseRealisation(
        spectrum=initial.noise0, seed=None, model=initial.noise_model
    )


def _check_initial_position(
    simulation: "Simulation", settings: SimulationSettings
) -> None:
    """Validate an absolute launch before translation can normalize its field.

    - ``simulation``: declared optical input, grid and absolute launch position in m.
    """
    z = simulation.initial_z
    if (
        not np.isfinite(z)
        or z < 0
        or z >= simulation.grid.length
        or np.any(np.asarray(settings.save_planes) <= z)
    ):
        raise DriverError(
            "initial_z must be finite, nonnegative and before every save plane"
        )
    if z == 0:
        return
    if (
        not isinstance(simulation.initial, np.ndarray)
        or simulation.spatial_grid is not None
    ):
        raise DriverError("an interior launch requires an explicit modal spectrum")
    if (
        simulation.noise is not None and simulation.noise.model == "modified"
    ) or settings.schedule is not None:
        raise DriverError(
            "an interior launch refuses the modified noise model and a fixed schedule; "
            "input noise takes its draw at the launch (SEP 37)"
        )


def _replaced_launch(
    original: PropagationProblem,
    field0: Complex[np.ndarray, "n_modes n_pol n_omega"],
    launch_z: float,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a replaced input as the derived problem's launch, without a further copy.

    On a free-space or spatial route the launch is a view of the derived simulation's
    own copy; on a modal route it is the power conversion of that copy, a second array
    beside it.

    - ``field0``: the replacement input, in true sqrt(W) on a modal route.
    - ``launch_z``: where it is launched, m.
    """
    if original.transform.geometry is not None:
        return _state(field0, original.shape[-1])
    # The conversion is a new array, held as it is rather than copied again.
    converted = power_amplitude(
        field0, power_weight_at(original, launch_z), inverse=True
    )
    return _hold(np.asarray(converted, dtype=np.complex128))


def _check_input_identity(
    identity: str | None,
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray | CouplingLaunch,
) -> None:
    """Refuse a stated identity that is empty, or of anything but an array input."""
    if identity is None:
        return
    if not identity:
        raise DriverError("input_identity must say what the input is: it is empty")
    if not isinstance(initial, np.ndarray):
        msg = (
            "input_identity states an array input's identity; "
            f"a {type(initial).__name__} is recorded as itself"
        )
        raise DriverError(msg)


def _given_input(
    given: np.ndarray, held: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> np.ndarray:
    """Return an array input as the caller gave it, for the record: a read-only view.

    The record keeps the caller's dtype and values, so that a file written from such
    an input before the simulation copied it still resumes; a view of a read-only owner
    cannot be made writeable, so the record and the run cannot be changed through it.

    - Returns: a read-only view of `held` when `given` is complex128, otherwise of a
      held copy of `given` in its own dtype; :class:`~solprop.translate.problem.Initial`
      takes either as held.
    """
    # The whole dtype, byte order included: the record serialises its exact string.
    source = held if given.dtype == held.dtype else _held_block(given, given.dtype)
    return _hold(source.view())


def _check_planar_input(shape: tuple[int, ...], n_columns: int, n_omega: int) -> None:
    """Refuse a planar input whose shape is not ``(n_p*n_y, 1, n_omega)``.

    - ``shape``: the input's shape.
    """
    if shape != (n_columns, 1, n_omega):
        msg = (
            "a planar input has shape (n_p*n_y, 1, n_omega) = "
            f"{(n_columns, 1, n_omega)}, got {shape}"
        )
        raise DriverError(msg)


def _check_spatial_input(shape: tuple[int, ...], n_columns: int, n_omega: int) -> None:
    """Refuse a spatial input whose shape is not ``(n_columns, n_pol, n_omega)``.

    - ``shape``: the input's shape.
    """
    if len(shape) != 3 or shape[0] != n_columns or shape[2] != n_omega:
        msg = "spatial input must have shape (n_columns, n_pol, n_omega)"
        raise DriverError(msg)


def _built(
    initial: Pulse | np.ndarray, grid: TimeGrid
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the initial spectrum, building a pulse against the grid when given one.

    - Returns: the spectrum, complex128, shape ``(n_modes, n_pol, n_omega)``, read-only
      and the simulation's own: a copy of an array (of its repeated block for a
      broadcast), or a solprop pulse's new build.
    """
    if isinstance(initial, np.ndarray):
        # Copied now, so that changing the caller's array later changes nothing, as
        # the problem holds it: a broadcast copies its block alone, and an array a
        # simulation already holds (its own, passed back) is taken as it is.
        return _held_state(initial)
    # A runtime-checkable protocol is satisfied by the presence of `build`, not by its
    # being callable, so the guard below is still the one that names the mistake.
    build = getattr(initial, "build", None)
    if not callable(build):
        msg = (
            f"the input must be a pulse with a build(grid) method or a built spectrum, "
            f"got {type(initial).__name__}"
        )
        raise DriverError(msg)
    return _built_state(initial, build(grid))


def _built_state(
    pulse: object, built: object
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a pulse's built spectrum as the problem holds it, copied only if foreign.

    A build of solprop's own pulse classes is a new array solprop made, so it is held
    as it is; any other pulse's build may be an array the pulse keeps, and is copied
    (:class:`~solprop.translate.problem.Initial`).

    - Returns: the spectrum, complex128, read-only and held.
    """
    spectrum = np.asarray(built)
    build = getattr(type(pulse), "build", None)
    own = getattr(build, "__module__", None) == _PULSE_MODULE
    if own and spectrum.dtype == np.complex128:
        return _hold(spectrum)
    return _held_state(spectrum)


_PULSE_MODULE: Final = SpatialPulse.__module__
"""The module of solprop's pulse classes, whose ``build`` always makes a new array."""


def _shaped_pulse(
    shape: str,
    *,
    wavelength: float,
    fwhm: float | None,
    width: float | None,
    energy: float | None,
    peak_power: float | None,
    phases: Sequence[float] | np.ndarray,
    pulses: Sequence[Pulse] | None = None,
) -> GaussianPulse | SechPulse | PulseSum:
    """Return the shaped input pulse a keyword interface starts from.

    - ``wavelength``: centre wavelength, m, and the reference of the spectral phase.
    - ``fwhm``: intensity full width at half maximum, s.
    - ``width``: natural width of a ``sech**2``, s.
    - ``energy``: pulse energy, J.
    - ``peak_power``: peak power, W.
    - ``phases``: Taylor coefficients of the optics-convention spectral phase.
    """
    if pulses is not None:
        given = [
            name
            for name, value in (
                ("fwhm", fwhm),
                ("width", width),
                ("energy", energy),
                ("peak_power", peak_power),
            )
            if value is not None
        ]
        if len(phases):
            given.append("phases")
        if shape != "gauss":
            given.append("shape")
        if given:
            msg = (
                f"pulses= describes every input pulse itself, and this call also sets "
                f"{', '.join(given)}: give each pulse its own duration, energy and "
                "phase, and leave the single-pulse keywords unset"
            )
            raise DriverError(
                msg, parameter=("pulses", *given), rule="pulses-or-pulse-keywords"
            )
        with _about("pulses"):
            return PulseSum(tuple(pulses))
    if shape == "sech":
        # The pulse names its own arguments, which are the keywords but for phase.
        with _about(renamed=_PULSE_KEYWORDS):
            return SechPulse(
                wavelength=wavelength,
                fwhm=fwhm,
                width=width,
                energy=energy,
                peak_power=peak_power,
                phase=phases,
            )
    if shape != "gauss":
        msg = f"shape must be 'gauss' or 'sech', got {shape!r}"
        raise DriverError(msg, parameter="shape")
    if width is not None or fwhm is None:
        msg = (
            "a Gaussian input is specified by its intensity fwhm; give fwhm= rather "
            "than width=, which is the natural width of a sech**2 pulse"
        )
        raise DriverError(
            msg,
            parameter="fwhm" if width is None else ("width", "shape"),
            rule="gauss-fwhm",
        )
    with _about(renamed=_PULSE_KEYWORDS):
        return GaussianPulse(
            wavelength=wavelength,
            fwhm=fwhm,
            energy=energy,
            peak_power=peak_power,
            phase=phases,
        )


def _shot_noise(
    noise: str | None, grid: TimeGrid, seed: int | None, *, n_modes: int = 1
) -> NoiseRealisation | None:
    """Draw the shot-noise realisation a run carries.

    The model is a plain string rather than :data:`NoiseModel` for the reason
    :func:`solprop.fourier._check_kind` gives: this is the run-time guard for callers
    that ignore the annotation, and the test session's ``beartype`` hook would otherwise
    reject the bad value before the check below is reached.

    - ``noise``: which model to use, or ``None`` for a noiseless run.
    - ``seed``: entropy of the generator; ``None`` draws from the operating system, and
      the entropy it drew is recorded on the realisation either way.
    """
    if noise is None:
        return None
    if noise not in ("input", "modified"):
        msg = f"noise must be 'default', None, 'input' or 'modified', got {noise!r}"
        raise DriverError(msg, parameter="noise")
    generator = np.random.default_rng(seed)
    draw = input_noise if noise == "input" else modified_noise
    return draw(grid, generator, n_modes=n_modes)


def _general_noise(
    noise: str | None,
    grid: TimeGrid,
    seed: int | None,
    guide: ModalGuide,
) -> NoiseRealisation | None:
    """Draw a general modal run's shot noise, in the basis it is defined in.

    The shot-noise draw (``docs/guide/media/noise.md``) is one photon per channel per
    bin, which is an energy-normalised true-power amplitude only in a basis orthonormal
    in true power. A guide in the provider's own coordinates **is** that basis, so
    this is :func:`_shot_noise` unchanged. A unitary change of basis draws
    **once, in the reference basis, with the same seed and the same stream**, and
    carries that one realisation into the run's coordinates through the inverse
    transform (:func:`solprop.noise.transformed_noise`): a second draw in the declared
    basis would be a different stochastic model, because a fixed-modulus random-phase
    draw is not invariant in distribution under unitary mixing.

    - ``noise``: which model to use, or ``None`` for a noiseless run.
    """
    drawn = _shot_noise(noise, grid, seed, n_modes=len(guide.mode_collection))
    sampling = guide.sampling
    if drawn is None or sampling is None or sampling.basis is None:
        return drawn
    return transformed_noise(drawn, sampling.basis.transform)


def _spatial_shot_noise(
    noise: NoiseModel, flux: SpatialFlux, n_pol: int, seed: int | None
) -> NoiseRealisation:
    """Draw a radial or Cartesian run's shot noise at its entrance plane.

    - ``seed``: entropy of the generator; ``None`` draws from the operating system.
    """
    return spatial_noise(flux, np.random.default_rng(seed), model=noise, n_pol=n_pol)


def _spatial_draw(
    noise: NoiseModel, simulation: "Simulation", medium: SpatialMedium
) -> Callable[[int | None], NoiseRealisation | None]:
    """Return the draw of a spatial run's shot noise, measured at its entrance plane."""
    flux = SpatialFlux.resolve(
        simulation._base_grid,
        cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid),
        medium,
        z=simulation.initial_z,
    )
    return functools.partial(
        _spatial_shot_noise, noise, flux, simulation._field0.shape[1]
    )


def _with_noise(
    problem: PropagationProblem, noise: NoiseRealisation | None
) -> PropagationProblem:
    """Return `problem` carrying `noise`, every other leaf shared by identity.

    A realisation reaches a translation only through the initial record and the
    ``has_noise`` flag; what else consults it is a refusal keyed on the model, and a
    replacement that changes the model is checked by its caller.

    - ``noise``: the realisation in true sqrt(W) on a modal route, or ``None`` for no
      noise.
    - Returns: the problem with the realisation in its internal, vacuum-normalised
      amplitude.
    """
    noise0 = None if noise is None else noise_spectrum(noise, problem.shape)
    if noise0 is not None and problem.transform.geometry is None:
        noise0 = power_amplitude(
            noise0, power_weight_at(problem, problem.initial.z), inverse=True
        )
    return dataclasses.replace(
        problem,
        initial=dataclasses.replace(
            problem.initial,
            noise0=noise0,
            noise_model="none" if noise is None else noise.model,
        ),
        flags=dataclasses.replace(problem.flags, has_noise=noise0 is not None),
    )


def _redrawable(
    simulation: "Simulation",
    draw: Callable[[int | None], NoiseRealisation | None],
    seed: int | None,
) -> "Simulation":
    """Let a realisation drawn without a seed be drawn again from a chosen one.

    A call with ``seed=None`` records the operating-system entropy it drew, so the
    same call made twice describes two runs. Keeping the draw lets a resume take the
    seed its file recorded and a scan point take its own (:meth:`Simulation._reseed`);
    a seeded realisation keeps the one it was given.

    - Returns: `simulation`, able to redraw its noise when `seed` was ``None``.
    """
    if seed is None and simulation.noise is not None:
        simulation._noise_redraw = draw
    return simulation


def default_field_range(peak_field: float) -> tuple[float, float]:
    """Return the field amplitudes an ionisation table covers, from the input's peak.

    The table is built in ``log`` field and its knots double until it is converged
    (:func:`solprop.translate.ionisation.build_rate_table`), so a generous range costs
    little; getting it wrong is what costs. Below ``field_min`` the rate is taken as
    zero, which is right because every rate this package carries is exactly zero below
    its threshold, and the threshold is where the lower limit ends up anyway. Above
    ``field_max`` the table *holds its last value*, so an upper limit that is too low
    is a silent clip of the rate exactly where ionisation matters most.

    Hence three decades below the input peak and thirty times above it. The lower
    factor only has to reach under the rate's threshold, which for the noble gases at
    the intensities a capillary reaches sits one to two decades below the peak. The
    upper factor has to clear the self-compressed peak: a capillary compressing a pump
    to a few optical cycles raises the on-axis field by a small single-digit factor
    before ionisation-induced defocusing and loss clamp it, so thirty is roughly an
    order of margin on that -- a thousand times in intensity.

    Parameters
    ----------
    peak_field : float
        Peak field amplitude of the input, V/m, > 0: on a capillary the largest
        over the axis and the transverse nodes.

    Returns
    -------
    tuple of float
        Lower and upper field amplitude in V/m.

    Raises
    ------
    DriverError
        If the peak field is not positive, which means the input is empty.

    Examples
    --------
    >>> from solprop.driver.inputs import default_field_range
    >>> low, high = default_field_range(1e10)
    >>> f"{low:.3g} {high:.3g}"
    '1e+07 3e+11'
    """
    if not peak_field > 0.0:
        msg = (
            f"the input carries no field (peak {peak_field:.3e} V/m), so "
            "the ionisation table has no range to cover; give "
            "ionisation_field_range explicitly"
        )
        raise DriverError(msg)
    return (peak_field * _FIELD_RANGE_BELOW, peak_field * _FIELD_RANGE_ABOVE)


def _general_pulse(
    incident: SpatialPulse | None,
    channel: int | str | None,
    shape: str,
    *,
    wavelength: float,
    fwhm: float | None,
    width: float | None,
    energy: float | None,
    peak_power: float | None,
    phases: Sequence[float] | np.ndarray,
    pulses: Sequence[Pulse] | None,
) -> GaussianPulse | SechPulse | PulseSum | SpatialPulse:
    """Return a general modal run's input: a shaped pulse, or an incident beam.

    - ``incident``: the incident beam, which carries its own temporal pulse; ``None``
      builds the pulse from the keywords.
    - ``shape``, ``wavelength``, ``fwhm``, ``width``, ``energy``, ``peak_power``,
      ``phases``, ``pulses``: the pulse keywords, as :func:`_shaped_pulse` takes them.
    """
    if incident is None:
        return _shaped_pulse(
            shape,
            wavelength=wavelength,
            fwhm=fwhm,
            width=width,
            energy=energy,
            peak_power=peak_power,
            phases=phases,
            pulses=pulses,
        )
    if channel is not None:
        msg = (
            "channel= and incident= are two ways to say what is launched and this "
            "call gives both: name a channel and the pulse enters it unchanged, or "
            "declare an incident beam and the channels capture what they overlap"
        )
        raise DriverError(msg)
    given = [
        name
        for name, value in (
            ("fwhm", fwhm),
            ("width", width),
            ("energy", energy),
            ("peak_power", peak_power),
            ("pulses", pulses),
        )
        if value is not None
    ]
    if len(phases):
        given.append("phases")
    if shape != "gauss":
        given.append("shape")
    if given:
        msg = (
            f"incident= carries its own temporal pulse, and this call also sets "
            f"{', '.join(given)}: give the beam's pulse its duration, energy and "
            "phase, and leave the pulse keywords unset"
        )
        raise DriverError(msg)
    if isinstance(incident, PlanarPulse):
        msg = (
            "a PlanarPulse launches into a planar guide's slab modes; the incident "
            "beam of a general modal guide is a SpatialPulse"
        )
        raise DriverError(msg)
    if incident.propz != 0:
        msg = (
            f"incident= is launched {incident.propz!r} m from its waist; carrying a "
            "beam to the guide's entrance is a transport step this launch does not "
            "model, so the beam is launched at its waist (propz=0)"
        )
        raise DriverError(msg)
    return incident


_BEAM_EXTENT = 60.0
"""Field exponent at the edge of an incident beam's radial plane.

The plane extends to where a super-Gaussian of order N has fallen to
``exp(-_BEAM_EXTENT)`` in field, ``r = waist * _BEAM_EXTENT**(1/(2N))``, times the
margin :data:`_BEAM_MARGIN`; its intensity there is below 1e-52 of the peak.
"""

_BEAM_MARGIN = 1.5
"""Factor on the incident beam's plane radius beyond its decay radius."""

_BEAM_NODES = 256
"""Radial nodes of an incident beam's plane.

With the plane radius above, the largest transverse wavenumber is about
``pi * 256 / (1.5 * 60**(1/2N) * waist)``, 69/w for a Gaussian, where its spectrum
is below exp(-1000): the beam is resolved to rounding.
"""


def _beam_grid(pulse: SpatialPulse) -> RadialGrid:
    """Return the radial plane an incident beam is sampled on before coupling."""
    radius = _BEAM_MARGIN * pulse.waist * _BEAM_EXTENT ** (1.0 / (2.0 * pulse.order))
    return RadialGrid(radius, _BEAM_NODES)


def _incident_launch(
    simulation: "Simulation",
    pulse: SpatialPulse,
    model: CouplingModel,
) -> "Simulation":
    """Couple an incident beam into a general modal simulation and install it.

    The beam becomes a radial :class:`~solprop.coupling.plane.FieldPlane` in the
    guide's core index (:meth:`~solprop.coupling.plane.FieldPlane.from_pulse`, its
    request met over its whole plane in the paraxial measure), and
    :func:`~solprop.coupling.coupler.couple` recovers its coordinates in the guide's
    channels under `model`. **Nothing is rescaled to the requested energy**: a
    badly captured beam launches the little the channels take, and a beam they
    cannot see is a valid zero launch. The launch is installed through
    :meth:`Simulation.with_input`, which records its provenance as
    ``configuration["coupling"]``; the coupling's report is recorded as
    ``configuration["launch"]`` and the beam as ``configuration["pulse"]``.
    """
    port = simulation.input_port()
    index = (
        np.zeros(simulation.grid.n_omega)
        if port.reference_index is None
        else (np.asarray(port.reference_index))
    )
    try:
        plane = FieldPlane.from_pulse(
            pulse,
            simulation.grid,
            PlaneGeometry(port.geometry.z),
            space=_beam_grid(pulse),
            index=index,
        )
        result = couple(plane, port, model=model)
    except CouplingError as error:
        raise DriverError(str(error)) from error
    report = result.report
    logger.info(
        "incident beam coupled into %d channel(s) under %s: %.4g J incident, "
        "%.4g J launched",
        port.shape[0],
        model.kind,
        report.incident_energy,
        report.launched_energy,
    )
    launched = simulation.with_input(
        result.launch,
        noise=simulation.noise,
        configuration={"pulse": pulse, "launch": report},
    )
    # The launch carries the constructor's realisation unchanged, so a draw made
    # without a seed may still be drawn again from a recorded one.
    launched._noise_redraw = simulation._noise_redraw
    return launched


def _hollow_input(
    pulse: Pulse,
    launch: ModalLaunch | None,
    grid: TimeGrid,
    modes: Sequence[Mode],
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], dict[str, object] | None]:
    """Return a hollow guide's initial state, and its launch record if it has one.

    - ``launch``: the ``launch`` keyword; ``None`` puts the input in the first mode.
    - Returns: ``configuration["modal_launch"]``, or ``None`` without a launch.
    """
    if launch is None:
        return _in_lowest_mode(pulse.build(grid), len(modes)), None
    # A hollow guide's modes are orthonormal in true power, so the shares are energy
    # fractions (SEP 41, S5).
    return launch_pulse(
        pulse, launch, grid, tuple(mode.channel_label for mode in modes)
    )


def _in_lowest_mode(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"], n_modes: int
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a one-mode input placed in the lowest mode of an `n_modes` set.

    A single pulse enters the fundamental and the higher modes start empty, filling
    through the nonlinearity.
    """
    if built.shape[0] > n_modes:
        msg = (
            f"the input carries {built.shape[0]} modes and the set has {n_modes}; a "
            "pulse cannot be placed in a mode that is not propagated"
        )
        raise DriverError(msg, parameter=("pulses", "modes"))
    if built.shape[0] == n_modes:
        return built
    state = np.zeros((n_modes, *built.shape[1:]), dtype=np.complex128)
    state[: built.shape[0]] = built
    return state


def _capillary_input_peak(
    field0: Complex[np.ndarray, "n_modes n_pol n_omega"],
    modes: ModeCollection,
    grid: TimeGrid,
    rule: TransverseQuadrature | None,
) -> tuple[float, float]:
    """Return the input's peak field, which sets the ionisation table's range.

    The peak is the largest field amplitude of the analytic signal over the axis and the
    run's own transverse nodes, which are where the plasma response is evaluated. On the
    axis it is the quantity :data:`solprop.stats.CATALOGUE`'s ``peak_intensity`` is the
    square of, and for an ``HE_1m`` launch the axis is the peak, so the range is
    anchored to a number the run itself reports. A ``TE_0m`` or ``TM_0m`` field vanishes
    on the axis (azimuthal order zero, but an azimuthal or radial vector field), so for
    those the nodes carry the peak. The same peak is the reference field of an optmat
    rate's constant energy per event (SEP 3, D8).

    - ``modes``: the mode set, whose profiles are evaluated on the axis and at the
      nodes.
    - ``rule``: the run's transverse quadrature; ``None`` for a mode-averaged run, whose
      table the axis alone bounds: the area-mean field it samples lies below it.
    - Returns: the peak field amplitude in V/m, and the circular share ``|S3| / S0`` of
      the field's transverse components where it peaks, zero for a linear field.
    """
    analytic = analytic_field(grid, field0)
    limits = modes.dimlimits()[0]
    peak_power = 0.0
    circular = 0.0
    for where in (
        (on_axis_rule(limits),) if rule is None else (on_axis_rule(limits), rule)
    ):
        matrices = mode_matrix(modes.modes, where, 0.0, components=("x", "y"))
        rows = matrices.synthesis.reshape(matrices.n_nodes, matrices.n_pol, -1)
        # One node at a time: the field at every node and time together would be
        # n_nodes times the state's size, which a full angular rule makes large.
        for node in rows:
            columns = np.tensordot(node, analytic, axes=([1], [0]))
            power = np.sum(np.abs(columns) ** 2, axis=tuple(range(columns.ndim - 1)))
            index = int(np.argmax(power))
            if float(power[index]) > peak_power:
                peak_power = float(power[index])
                # The Stokes parameter of the x and y components at the peak sample.
                x, y = columns.reshape(columns.shape[0], -1, columns.shape[-1])[
                    :, 0, index
                ]
                circular = abs(2.0 * float(np.imag(x * np.conj(y)))) / peak_power
    peak = math.sqrt(peak_power)
    logger.debug("input peak field over the axis and the nodes %.4e V/m", peak)
    return peak, circular


@dataclasses.dataclass(frozen=True)
class _InputPeak:
    """What the input's peak field is found from, evaluated only when it is needed.

    Attributes
    ----------
    field0 : numpy.ndarray
        The input state, shape ``(n_modes, n_pol, n_omega)``.
    modes : solprop.modes.collection.ModeCollection
        The mode set.
    grid : solprop.grid.TimeGrid
        The grid the state is stored on.
    rule : solprop.modes.quadrature.TransverseQuadrature or None
        The run's transverse quadrature; ``None`` for a mode-averaged run.
    """

    field0: Complex[np.ndarray, "n_modes n_pol n_omega"]
    modes: ModeCollection
    grid: TimeGrid
    rule: TransverseQuadrature | None

    def evaluate(self) -> tuple[float, float]:
        """Return the peak field in V/m and its circular share."""
        return _capillary_input_peak(self.field0, self.modes, self.grid, self.rule)


def _ionisation_inputs(
    responses: Sequence[object],
    field_range: tuple[float, float] | None,
    peak: _InputPeak | None,
) -> tuple[list[object], tuple[float, float] | None, float | None]:
    """Return a capillary run's ionisation table range and its bound rates.

    The input's peak field is found once, when the table's range is the default one
    (:func:`default_field_range`) or an optmat rate waits for its reference field.

    - ``field_range``: the caller's table range in V/m, or ``None`` for the default.
    - ``peak``: where the input's peak is found from; ``None`` for a run without plasma.
    - Returns: the responses (optmat rates bound to the peak), the table range, and the
      reference field in V/m the rates were bound to, or ``None``.
    """
    unbound = _unbound_rates(responses)
    if peak is None or not (field_range is None or unbound):
        return list(responses), field_range, None
    found = peak.evaluate()
    if field_range is None:
        field_range = default_field_range(found[0])
    if not unbound:
        return list(responses), field_range, None
    return _bound_rates(responses, found, field_range), field_range, found[0]


def _check_rate_inputs(
    simulation: "Simulation",
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
) -> None:
    """Check a run's input against each optmat rate it carries, once per run.

    The reference field and the table range stay those the run was built with; the
    input is checked against each rate's maximum field and, when it is not linearly
    polarised, the Keldysh parameter at its peak
    (:meth:`solprop.nonlinear.optmat_rates.OptmatRate.check_input`). On a mode-set
    route the peak and the polarisation are the field's over the axis and the nodes
    (:func:`_capillary_input_peak`); on a free-space route the peak is the physical
    intensity's maximum and the polarisation the pulse's Jones vector, which a
    spectrum given as an array does not declare, so it is taken as linear there.
    """
    rates = [
        one.rate
        for one in simulation.responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
    ]
    if not rates:
        return
    if simulation.spatial_grid is not None:
        peak, circular = _spatial_input_peak(simulation, initial)
    elif simulation.general is None and simulation.settings.quadrature is not None:
        peak, circular = _capillary_input_peak(
            simulation._field0,
            ModeCollection(list(simulation.modes)),
            simulation.grid,
            simulation.settings.quadrature,
        )
    else:
        return
    for rate in rates:
        rate.check_input(peak, circular=circular)


def _bind_spatial_rates(
    simulation: "Simulation",
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
) -> None:
    """Bind a free-space or planar run's optmat rates to its input, and check it.

    The spatial routes -- radial, Cartesian, line and planar -- bind an optmat rate's
    constant energy per event to the input's peak field, as the capillary route does
    (SEP 3, Q14); binding also checks the input against each rate. A run whose rates
    are all bound is only checked.
    """
    if _unbound_rates(simulation.responses):
        simulation.responses = tuple(
            _bound_rates(
                simulation.responses,
                _spatial_input_peak(simulation, initial),
                simulation.settings.ionisation_field_range,
            )
        )
    else:
        _check_rate_inputs(simulation, initial)


def _spatial_input_peak(
    simulation: "Simulation",
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray,
) -> tuple[float, float]:
    """Return a free-space or planar run's peak input field and its circular share.

    - Returns: the peak physical field amplitude in V/m, and the circular share
      ``|S3|/S0`` of the pulse's Jones vector; a spectrum given as an array declares
      none and is taken as linear.
    """
    space = cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid)
    # Ionisation is driven by the physical field, never by a signed axial flux.
    peak = peak_field(simulation.grid, space, simulation._field0)
    jones = (
        np.asarray(initial.jones, dtype=np.complex128)
        if isinstance(initial, SpatialPulse | SpatialPulseSum)
        else np.ones(1, dtype=np.complex128)
    )
    circular = (
        abs(2.0 * float(np.imag(jones[0] * np.conj(jones[1]))))
        / float(np.sum(np.abs(jones) ** 2))
        if jones.size == 2
        else 0.0
    )
    return peak, circular


def _unbound_rates(responses: Sequence[object]) -> bool:
    """Return whether a plasma response carries an optmat rate without a reference.

    - Returns: True when a rate's constant energy per event still waits for the input's
      peak.
    """
    return any(
        isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(one.rate, OptmatRate)
        and one.rate.reference_field is None
        for one in responses
    )


def _bound_rates(
    responses: Sequence[object],
    peak: tuple[float, float],
    field_range: tuple[float, float] | None,
) -> list[object]:
    """Bind a run's optmat rates to the input's peak field, and check the input.

    - ``peak``: the input's peak field in V/m and its circular share there.
    - ``field_range``: the field amplitudes in V/m the run tabulates its rates over.
    - Warns once per run when the peak exceeds a rate's maximum field, or the input is
      not linearly polarised at a Keldysh parameter above one
      (:meth:`solprop.nonlinear.optmat_rates.OptmatRate.check_input`).
    """
    field, circular = peak
    bound = list(bind_reference_field(responses, field, field_range))
    for one in bound:
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse) and isinstance(
            one.rate, OptmatRate
        ):
            one.rate.check_input(field, circular=circular)
    return bound


def _borrowed_input(
    initial: Pulse | SpatialPulse | SpatialPulseSum | np.ndarray, *, stated: bool
) -> Pulse | SpatialPulse | SpatialPulseSum | np.ndarray:
    """Return a runner's input as its private derived simulation may take it, uncopied.

    A derived simulation copies an array so that its record and its run outlive the
    caller's changes. A runner's lives only until the load, on the caller's thread and
    inside the call, so an array needs no copy unless the record outlives it: one
    recorded inline stays a copy, in the handler's record.

    - Returns: a read-only view of a complex128 array the record does not hold, marked
      as held so that nothing copies it; `initial` itself otherwise. The caller's own
      flags are not touched.
    """
    if (
        isinstance(initial, np.ndarray)
        and initial.dtype == np.complex128
        and (stated or not _recorded_inline(initial))
    ):
        return _hold(np.asarray(initial).view())
    return initial


def _without_input(problem: PropagationProblem) -> PropagationProblem:
    """Return a loaded problem without its input, for the march that follows the load.

    The save-plane loop reads the launch position and never the launch, so the state
    is replaced by a broadcast of one zero, which holds no volume (:class:`~solprop.
    translate.problem.Initial`). The problem is good for the march alone.
    """
    placeholder = np.broadcast_to(
        np.zeros(1, dtype=np.complex128), problem.initial.field0.shape
    )
    return dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=_hold(placeholder))
    )
