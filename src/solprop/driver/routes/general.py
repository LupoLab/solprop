# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route: a supplied guide's modes and its keywords."""

import functools
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop.boundaries import Absorber
from solprop.coupling import CouplingModel
from solprop.driver._launch import launch_pulse, refuse_beside
from solprop.driver.defaults import resolve_noise
from solprop.driver.inputs import (
    ModalLaunch,
    _general_noise,
    _general_pulse,
    _incident_launch,
    _redrawable,
)
from solprop.driver.routes.hollow import _capillary_grid
from solprop.driver.routes.materials import _selected_guide
from solprop.driver.settings import (
    NoiseSetting,
    SimulationSettings,
    _boundary,
    _keyword_controller,
    _save_planes,
)
from solprop.exceptions import DriverError, TranslationError
from solprop.grid import TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.material import MaterialRegion
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryCheck
from solprop.output.schema import ModeRecord
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import Pulse, SpatialPulse
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.coupled import metric_at
from solprop.translate.general import general_modal_route
from solprop.translate.masks import AbsorberScheme
from solprop.translate.responses import KerrForm

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["GAS_KEYWORD_DEFAULTS"]

logger = logging.getLogger(__name__)


def _from_general(
    cls: type["Simulation"],
    provider: ModalGuide,
    modes: Sequence[Mode],
    length: float,
    *,
    channel: int | str | None,
    incident: SpatialPulse | None,
    coupling: CouplingModel | None,
    kerr_form: KerrForm,
    observable_rtol: float | None,
    observable_atol: float | None,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
    grid: TimeGrid | None,
    fwhm: float | None,
    width: float | None,
    shape: str,
    energy: float | None,
    peak_power: float | None,
    phases: Sequence[float] | np.ndarray,
    pulses: Sequence[Pulse] | None,
    absorber: Absorber | Literal[False] | None,
    absorber_scheme: AbsorberScheme | None,
    boundary_check: BoundaryCheck | None,
    noise: NoiseSetting,
    seed: int | None,
    rtol: float | None,
    atol: float,
    dz_max: float | None,
    dz_init: float | None,
    schedule: FixedSchedule | None,
    save_planes: Sequence[float] | np.ndarray | None,
    n_save: int | None,
    stats: StatsSpec | None,
    backend: str,
    materials: MaterialSelection | None = None,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None,
    launch: ModalLaunch | None = None,
    interpolate_planes: bool = False,
    step_planes: int | None = None,
    polarisation_taper: bool = False,
    **refused: object,
) -> "Simulation":
    """Build a general modal run from a guide that solves its own fields.

    The provider half of :meth:`from_waveguide`, kept apart because it shares no
    physics with the gas-filled one: no fill, no mixture, no static transverse
    quadrature, no declared response list, and a launch that crosses no
    normalisation boundary. Every keyword of the gas-filled path arrives in
    `refused` and is refused by name if it was set.

    - ``length``: guide length, m.
    - ``channel``: which channel the pulse enters; ``None`` takes the first.
    - ``observable_rtol``, ``observable_atol``: accepted only as ``None``, which
      means the solver's own tolerances.
    - ``fwhm``, ``width``, ``shape``, ``energy``, ``peak_power``, ``phases``,
      ``pulses``: the input pulse, as :func:`_shaped_pulse` takes it.
    - ``absorber``: the collar rates; ``None`` builds the grid's default collar, and
      ``False`` switches both soft collars off and keeps the hard band mask.
    - ``absorber_scheme``: how the temporal collar is applied; ``None`` is the
      default scheme.
    - ``stats``: which statistics to record; ``None`` takes :func:`_general_stats`.
    - ``integrator``: the method that steps the run; ``None`` takes ``Multistep()``,
      and ``LowStorageRungeKutta()`` replays `schedule` in three registers.
    - ``step_planes``: also save the end of every `step_planes`-th accepted step;
      ``None`` is ``0``.
    - ``**refused``: every gas-filled keyword; each is refused by name when it is
      set.
    """
    _refuse_gas_keywords(refused)
    provider = _selected_guide(
        provider, MaterialSelection() if materials is None else materials
    )
    for name, value in (
        ("observable_rtol", observable_rtol),
        ("observable_atol", observable_atol),
    ):
        if value is not None:
            msg = (
                f"{name}={value!r}: a general modal run's observables are its "
                "channel amplitudes in a basis orthonormal in true power, so "
                "the controller's own rtol and atol already bound them, and "
                "the separate allocation a different number would ask for -- "
                "over the table's frequency interpolation, the transverse rule "
                "and the step -- is not implemented. Pass None, and set rtol "
                "and atol"
            )
            raise DriverError(msg)
    working = _capillary_grid(
        grid,
        length,
        wavelength,
        wavelength_limits,
        trange,
        envelope=False,
        thg=False,
    )
    if working.kind != "field":
        msg = (
            f"a general modal guide propagates on a field grid and this run is "
            f"on an {working.kind} grid: its leaves are tabulated on the "
            "canonical increasing coarse axis, and its source is the "
            "analytic-signal polarisation of a carrier-resolved field"
        )
        raise DriverError(msg)
    channels = tuple(provider.mode_collection.modes)
    if tuple(modes) not in ((), channels):
        msg = (
            f"a general modal guide supplies its own {len(channels)} ordered "
            f"channel(s), validated against the provider's fields, and this "
            f"call passes {len(tuple(modes))} mode(s) of its own: pass "
            "modes=() beside provider=, or the guide's own "
            "provider.mode_collection.modes"
        )
        raise DriverError(msg)
    planes = _save_planes(length, save_planes, n_save)
    refuse_beside(launch, channel=channel, incident=incident, coupling=coupling)
    pulse = _general_pulse(
        incident,
        channel,
        shape,
        wavelength=wavelength,
        fwhm=fwhm,
        width=width,
        energy=energy,
        peak_power=peak_power,
        phases=phases,
        pulses=pulses,
    )
    built = (pulse.temporal if isinstance(pulse, SpatialPulse) else pulse).build(
        working
    )
    try:
        route = general_modal_route(
            provider,
            working,
            z_max=float(np.max(planes)),
            omega_reference=float(working.omega0),
            kerr_form=kerr_form,
            selection=MaterialSelection() if materials is None else materials,
            # The pulse is built first so that the route's occupied-support basis
            # gate is weighted by where this run's light actually is rather than
            # over the whole declared band. The launch's channel distribution does
            # not enter that weight -- it is summed over channels -- so the pulse
            # alone is enough, which is what lets a projected launch, which needs
            # the route's own dual, still be weighted by its own spectrum. It
            # reaches no leaf.
            launch=built,
        )
    except TranslationError as error:
        # The bridge belongs to the translation layer and says so; a caller of
        # `from_waveguide` is told what this entry point promises, and
        # `DriverError` is what its `Raises` names. The message is the bridge's
        # own, unchanged, and the cause is kept.
        raise DriverError(str(error)) from error
    beam = pulse if isinstance(pulse, SpatialPulse) else None
    if (beam is None) != (coupling is None):
        msg = (
            "an incident beam is coupled under a declared model: give incident= "
            "with coupling=CouplingModel('vector_paraxial', model_error=...), or "
            "neither"
        )
        raise DriverError(msg)
    launched: dict[str, object] | None = None
    if beam is not None:
        # Checked before the port is built: a guide whose first region is off the
        # axis has no core for a beam to enter through.
        _core_index(provider.regions, working)
        # The launch is the coupling's, installed through `with_input` below.
        field0 = np.zeros((len(channels), 1, working.n_omega), dtype=np.complex128)
    elif launch is None:
        field0 = _in_channel(built, provider, channel)
    else:
        # A declared change of basis carries a dense flux metric, under which the
        # shares are a coordinate split; in the provider's own coordinates the
        # metric is the identity and they are energy fractions (SEP 41, S5).
        field0, launched = launch_pulse(
            # No beam, so the input is the shaped pulse.
            cast("Pulse", pulse),
            launch,
            working,
            provider.mode_collection.channel_labels,
            metric=metric_at(route.metric_tables, 0.0),
        )
    boundary, scheme = _boundary(absorber, absorber_scheme, working)
    controller, default = _keyword_controller("waveguide", rtol, atol)
    settings = SimulationSettings(
        controller=controller,
        _default_controller=default,
        _route="waveguide",
        save_planes=planes,
        dz_max=length / 2 if dz_max is None else dz_max,
        dz_init=length * 1e-4 if dz_init is None else dz_init,
        auto_init=dz_init is None,
        schedule=schedule,
        integrator=integrator,
        interpolate_planes=interpolate_planes,
        step_planes=0 if step_planes is None else step_planes,
        absorber=boundary,
        absorber_scheme=scheme,
        boundary_check=boundary_check,
        # The route carries its nodes, its materials and its normalisation in
        # its own leaves: the static transverse quadrature, the component
        # selection, the density path and the index override are all refused
        # by the translation layer and are not settings of this run.
        stats=_general_stats() if stats is None else stats,
        polarisation_taper=polarisation_taper,
        backend=backend,
    )
    noise_model, resolution = resolve_noise("waveguide", noise, seed)
    draw = functools.partial(_general_noise, noise_model, working, guide=provider)
    simulation = cls(
        working,
        channels,
        [],
        field0,
        settings=settings,
        noise=draw(seed),
        noise_resolution=resolution,
        configuration={
            "waveguide": {
                "modes": [ModeRecord.from_mode(mode) for mode in channels],
                "length": length,
                "provider": dict(provider.identity),
            },
            **({} if beam is not None else {"pulse": pulse}),
            **({} if launched is None else {"modal_launch": launched}),
        },
        general=route,
        guide=provider,
        materials=materials,
    )
    _redrawable(simulation, draw, seed)
    if beam is None:
        return simulation
    return _incident_launch(simulation, beam, cast("CouplingModel", coupling))


GAS_KEYWORD_DEFAULTS: Mapping[str, object] = {
    "gas": None,
    "pressure": None,
    "pressure_floor": None,
    "vacuum_end": None,
    "susceptibility": None,
    "temperature": None,
    "kerr": True,
    "thg": None,
    "plasma": None,
    "preionisation": None,
    "plasma_integration": None,
    "raman": None,
    "mode_average": False,
    "n_r": None,
    "n_theta": None,
    "radial_rule": "gauss_kronrod",
    "ionisation_field_range": None,
    "polarisation": None,
    "envelope": False,
    "node_budget": None,
    "taper_coupling": True,
}
"""Every gas-filled `from_waveguide` keyword, with the value that means "not given".

A general modal guide owns its materials, its responses, its transverse rule and
its grid kind, so each of these is refused by name when it departs from the value
here; a keyword silently ignored would let a caller believe a fill, a response or
a resolution was applied.
"""

_GAS_KEYWORD_REASONS: Mapping[str, str] = {
    "gas": "a guide's materials are its regions' own laws, not a fill",
    "pressure": "a guide's materials are its regions' own laws, not a fill",
    "pressure_floor": "a guide has no fill and so no pressure path",
    "vacuum_end": "a guide has no fill and so no gas flow to choke",
    "susceptibility": "a guide's dispersion is its provider's, solved from the "
    "regions' permittivity laws",
    "temperature": "a material region's law is a declared state, and a guide "
    "carries no gas whose density a temperature would set",
    "kerr": "a guide's Kerr response is its regions'; declare a region with no "
    "MaterialKerr to make it linear",
    "thg": "the general route's third harmonic is set by kerr_form=: the real "
    "cubic carries it and the analytic-signal form does not",
    "plasma": "a general guide carries the instantaneous Kerr term alone; "
    "ionisation is not implemented on the general route",
    "preionisation": "a general guide carries no plasma channel",
    "plasma_integration": "a general guide carries no plasma column to integrate",
    "raman": "a guide's delayed response is its material regions' own, the silica "
    "tensor Raman model",
    "mode_average": "a general guide's channels are resolved on its own node rule; "
    "mode averaging is the gas-filled route's",
    "n_r": "the transverse rule is the guide's own node rule, declared when the "
    "guide was built",
    "n_theta": "the transverse rule is the guide's own node rule, declared when "
    "the guide was built",
    "radial_rule": "the transverse rule is the guide's own node rule, declared when "
    "the guide was built",
    "taper_coupling": "a guide that moves along z carries its own connection in the "
    "general route's flow; the opt-out is the capillary route's",
    "ionisation_field_range": "a general guide carries no ionisation table",
    "polarisation": "a general guide puts every physical channel, including each "
    "parity partner, in its own row; launch into them by name with channel=, or "
    "build the launch yourself and pass it to Simulation.with_input",
    "envelope": "the general route synthesises an analytic field from a "
    "carrier-resolved spectrum and propagates on a field grid",
    "node_budget": "the general route sizes its node chunk from its own memory "
    "ledger, which counts different buffers",
}
"""Why each gas-filled keyword cannot be carried by a general modal guide."""


def _means_not_given(value: object, default: object) -> bool:
    """Return whether a supplied keyword carries the value that means "not given".

    - Returns: True when the keyword was, in effect, not given.
    """
    if value is default:
        return True
    if isinstance(value, np.ndarray) and value.ndim > 0:
        # `array == default` is elementwise and its truth value raises; a keyword
        # whose "not given" value is a scalar or None is never an array anyway, so
        # an array is always something the caller meant.
        return False
    return bool(value == default)


def _refuse_gas_keywords(supplied: Mapping[str, object]) -> None:
    """Refuse, by name, every gas-filled keyword given beside a provider."""
    for name, value in supplied.items():
        # Indexed, not `.get`: a keyword this table does not know is a keyword
        # `from_waveguide` would silently ignore on this route, and
        # `test_from_waveguide_refuses_or_names_every_keyword_it_takes` locks the
        # two together so that adding one cannot pass unnoticed.
        default = GAS_KEYWORD_DEFAULTS[name]
        if _means_not_given(value, default):
            continue
        msg = (
            f"{name}={value!r} beside a provider=: {_GAS_KEYWORD_REASONS[name]}. "
            "The general modal route takes its physics from the guide "
            "(docs/guide/waveguides/general-modal.md)"
        )
        raise DriverError(msg)


def _general_stats() -> StatsSpec:
    """Return the statistics a general modal run records by default.

    The route's **own** names, not the legacy ones: ``physical_energy`` is the
    metric-weighted energy through the plane and ``analytic_flux`` the peak total
    analytic Poynting flux, both scalars, because a coordinate energy and a
    coordinate power are not additive on a basis that is not orthonormal
    (``docs/guide/results/statistics.md``). ``energy``, ``peak_power``,
    ``peak_power_total`` and ``peak_intensity`` are refused by name on this basis
    rather than redefined. ``omega0``,
    ``fwhm_t`` and ``fwhm_t_outer`` stay: they are functionals of the
    same-frequency flux ``a.H W a`` and take the basis metric
    (:data:`solprop.stats.METRIC_WEIGHTED_STATISTICS`), so they are exact on a
    declared basis as well as on the identity one.
    ``quadrature_error`` is not here either: a provider's node rule has
    no embedded coarse pair to difference.
    """
    return StatsSpec(
        names=("omega0", "physical_energy", "analytic_flux", "fwhm_t", "fwhm_t_outer")
    )


def _in_channel(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"],
    guide: ModalGuide,
    channel: int | str | None,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Place a one-channel pulse in a named or indexed channel of a guide.

    **The amplitude is placed unchanged.** A provider normalises every channel to
    unit longitudinal Poynting flux, so a channel amplitude is already a true-power
    amplitude and the launched energy is the pulse's; there is no ``sqrt(Re n_eff)``
    boundary on this route and nothing here crosses one.

    - ``channel``: row index, display label, or ``None`` for the first channel.
    """
    labels = [label.display_label for label in guide.mode_collection.channel_labels]
    if built.shape[0] != 1:
        msg = (
            f"a general modal launch places one pulse in one channel and this "
            f"input carries {built.shape[0]}; build the state yourself and pass it "
            "to Simulation.with_input to launch into several at once"
        )
        raise DriverError(msg)
    if channel is None:
        index = 0
    elif isinstance(channel, str):
        if channel not in labels:
            msg = f"channel={channel!r} is not one of this guide's: {labels}"
            raise DriverError(msg)
        index = labels.index(channel)
    else:
        index = int(channel)
        if not 0 <= index < len(labels):
            msg = (
                f"channel={channel!r} is outside this guide's {len(labels)} "
                f"channel(s): {labels}"
            )
            raise DriverError(msg)
    state = np.zeros((len(labels), *built.shape[1:]), dtype=np.complex128)
    state[index] = built[0]
    logger.debug("general modal launch into channel %d (%s)", index, labels[index])
    return state


def _core_index(
    regions: Sequence[MaterialRegion], grid: TimeGrid
) -> Float[np.ndarray, " n_omega"]:
    """Return the real phase index of a guide's core at each of a grid's frequencies.

    The incident medium of a beam launched into the guide: the innermost material
    region, whose law does not move with z, so it is the entrance's.

    - Returns: ``Re sqrt(eps)`` on the active positive band and zero elsewhere, shape
      ``(n_omega,)``.
    """
    core = regions[0]
    if not isinstance(core.inner_radius, float | int) or core.inner_radius != 0:
        msg = (
            f"an incident beam enters a guide through its core, the region on the "
            f"axis, and this guide's first region {core.name!r} does not start there"
        )
        raise DriverError(msg)
    live = grid.active_mask & (grid.omega > 0)
    index = np.zeros(grid.n_omega)
    index[live] = np.real(np.sqrt(core.material.permittivity(grid.omega[live])))
    return index
