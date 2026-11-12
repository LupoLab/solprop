# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hollow-guide route: capillaries and supplied gas-filled mode sets.

The grid, modes, transverse quadrature, responses and statistics of a gas-filled hollow
guide, the checks of its walls, gas flow and plasma settings, and the route itself.
"""

import dataclasses
import functools
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
from scipy.special import j0, jn_zeros

from solprop.boundaries import Absorber, warn_walkoff
from solprop.driver.defaults import WAVEGUIDE_ATOL, resolve_noise, route_row
from solprop.driver.inputs import (
    ModalLaunch,
    _hollow_input,
    _InputPeak,
    _ionisation_inputs,
    _redrawable,
    _shaped_pulse,
    _shot_noise,
)
from solprop.driver.mixtures import (
    ResponseSelection,
    capillary_fill,
    is_air,
    mixture_profiles,
)
from solprop.driver.routes.keywords import (
    _GRID_KEYWORDS,
    _MATERIAL_WARNINGS,
    _about,
    _unset,
    _warnings_about,
)
from solprop.driver.routes.materials import _check_selection_carrier, _default_cladding
from solprop.driver.settings import (
    NoiseSetting,
    _boundary,
    _save_planes,
    keyword_settings,
)
from solprop.driver.tdse_run import _entry_overrides, _tdse_entry, _tdse_stats
from solprop.exceptions import (
    DriverError,
    GasFlowWarning,
    SampledPeakWarning,
    SolpropWarning,
)
from solprop.gasflow import VacuumEnd
from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.linear import Frame, frame_quantities
from solprop.modes.antiresonant import TubeLattice
from solprop.modes.capillary import (
    CoherentWall,
    GlassIndex,
    LossSpec,
    MarcatiliMode,
    MarcatiliModel,
)
from solprop.modes.collection import ModeCollection
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import (
    RadialRule,
    TransverseQuadrature,
    cartesian_nodes,
    transverse_quadrature,
)
from solprop.modes.slab import SlabMode
from solprop.modes.stepindex import StepIndexMode
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate
from solprop.nonlinear.mixtures import electronic_response, responses_for
from solprop.nonlinear.optmat_raman import mixture_raman, raman_status
from solprop.nonlinear.optmat_rates import selected_rate
from solprop.nonlinear.raman import MolecularOptions
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    KerrResponse,
    PlasmaResponse,
)
from solprop.nonlinear.tdse import TDSEExecution, TDSEGasResponse
from solprop.output.records import ModeRecord
from solprop.physdata import Gas, Mixture, species
from solprop.physdata.mixtures import Pressure
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.physdata.species import Species, built_in_population
from solprop.profiles import PressureGradient, ZProfile, as_profile
from solprop.pulses import Pulse
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec, default_spec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.masks import AbsorberScheme
from solprop.translate.problem import DEFAULT_COMPONENTS

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = [
    "BACKEND_STATISTICS",
    "ORTHONORMALITY_TOLERANCE",
    "PLASMA_STATISTICS",
    "default_radial_order",
]

logger = logging.getLogger(__name__)

PLASMA_STATISTICS = ("electron_density", "peak_ionisation_rate")
"""Backend statistics that read the plasma the run compiled; refused without one."""

BACKEND_STATISTICS = ("peak_intensity", "quadrature_error", *PLASMA_STATISTICS)
"""The statistics only a backend can compute, recorded by a capillary run by default.

Each needs something the kernel contract does not carry to a later pass: the quadrature
nodes the plasma was evaluated at, the mode set the axis needs, or the embedded
coarse-rule projection, which is gone once the step is accepted. The two plasma entries
are recorded only by a run that has a plasma response; the backend refuses them
otherwise, and a default that always asked for them would make ``plasma=False`` fail.
"""

ORTHONORMALITY_TOLERANCE = 1e-6
"""How far a given mode set may stray from orthonormality before it is refused.

The transform is one synthesis matrix and its dual, which represent an orthonormal
basis and nothing else, so a set that is not one is propagating a different problem
from the one it describes. The bound is loose because the overlaps are integrated
adaptively by :func:`scipy.integrate.dblquad`, whose own accuracy on a set of Bessel
profiles is around 1e-9.
"""

_MINIMUM_RADIAL_ORDER = 32
"""Floor on the default radial order; see :func:`default_radial_order`."""

_RADIAL_NODES_PER_MODE = 8
"""Radial nodes per mode in the default order; see :func:`default_radial_order`."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class _HollowRequest:
    """The keywords of :func:`_from_hollow`, as the public constructor took them.

    - ``radius``: core radius, m, constant or along z.
    - ``length``: capillary length, m.
    - ``gas``: Built-in gas, mole fractions keyed by species name, or ordered
      populations carrying their own density paths. A mapping uses total pressure.
    - ``pressure``: None uses one bar for a gas or fractions. Omit pressure beside a
      Mixture; every explicit pressure is refused because it already carries its
      paths. Fill pressure in **bar**: a number for a constant fill, a pair ``(p_in,
      p_out)`` for the steady-flow gradient of
      :meth:`solprop.profiles.PressureGradient.two_point`, or ``(z, p)`` with the
      positions and the pressures there for
      :meth:`solprop.profiles.PressureGradient.from_stages`.
    - ``pressure_floor``: lowest pressure the fill is allowed to reach, bar;
      ``None`` takes :data:`solprop.profiles.DEFAULT_PRESSURE_FLOOR_FRACTION` of the
      highest stage pressure. With ``vacuum_end="floor"``, pass ``0.0`` for a true
      vacuum end, which warns, because ``dp/dz`` diverges there. Refused beside a
      constant pressure or a pressure profile, which have no end to apply it to.
    - ``vacuum_end``: how a pressure gradient's outer end below its choked pressure
      is treated. ``None`` (the default) means ``"choked"`` for a circular hollow
      guide: the fill follows the bore (Pennetta et al. 2019, eq. 6) and an end
      exhausting into vacuum takes the choked pressure of the flow
      (:func:`solprop.gasflow.steady_fill`), with a warning if the flow is not
      laminar. ``"floor"`` concerns the ends only: it keeps the declared end
      pressures, raised only to `pressure_floor`, for a comparison with a model that
      imposes them, while a taper's fill still follows the bore (on a uniform bore,
      bitwise the uniform law). Refused beside a constant pressure, a pressure
      profile or a Mixture, which have no end to choke, and on the general route.
      Every mode declaring ``core_radius`` -- every ``MarcatiliMode`` subclass, a
      ``TabulatedMode`` built from arrays included -- is taken as a circular bore of
      that radius; a rectangular guide keeps the uniform law.
    - ``wavelength``: reference wavelength, m, and the pulse's centre wavelength.
    - ``wavelength_limits``: shortest and longest wavelength of the frequency
      window, m. Required unless `grid` is given.
    - ``trange``: requested width of the time window, s. Required unless `grid` is
      given.
    - ``grid``: a grid of one's own, instead of the three keywords above; the only
      way to reach ``dt``, ``ffac`` or the transform size.
    - ``fwhm``: pulse duration at half maximum, s.
    - ``width``: natural width of a sech pulse, s; give this or `fwhm`.
    - ``energy``: pulse energy, J. Give this or `peak_power`.
    - ``peak_power``: peak power, W, realised after the spectral phases.
    - ``phases``: Taylor spectral phases about `wavelength`, optics convention: CEP
      in rad, group delay in s, GDD in s^2, and so on.
    - ``pulses``: several pulses launched together, for two-colour and multi-colour
      inputs: each a :mod:`solprop.pulses` object with its own wavelength, duration,
      energy or peak power, phase and delay; the input is the sum of their fields
      (:class:`~solprop.pulses.PulseSum`). Replaces the single-pulse keywords
      `fwhm`, `width`, `shape`, `energy`, `peak_power` and `phases`, which must then
      be left unset; `wavelength` still sets the grid's reference wavelength.
    - ``launch``: the modes each pulse enters, with their shares and phases, as
      :meth:`from_capillary` documents it; ``None`` puts the input in the first
      mode.
    - ``mode_average``: propagate the one mode **mode-averaged**: a uniform disc of
      the mode's effective area instead of a transverse rule
      (``docs/guide/waveguides/mode-averaging.md``). The plasma then sees the
      area-mean field, below the on-axis one (``HE_11``'s axis intensity is 1.768
      times ``P/A_eff``), while ``peak_intensity`` stays on the axis. ``False``, the
      default, resolves the mode on the radial rule. Refused beside more than one
      mode and beside `n_r`, `n_theta` or a `radial_rule`.
    - ``lattice``: tube geometry for an integer mode count, selecting Vincetti
      modes. Refused with supplied modes, which carry their own geometry, and with
      ``model="reduced"``. None selects ordinary capillary modes.
    - ``wall``: declare the scalar **coherent** wall block: the outgoing Robin
      boundary condition condensed onto the retained basis, which couples every
      ``HE_1m`` channel to every other with a definite phase instead of giving each
      its own Marcatili attenuation. It needs the contiguous set ``HE_11`` to
      ``HE_1N`` -- or, with ``impedance="te_tm"``, one azimuthal order -- on a
      constant or a varying guide, and a scalar run; it carries the default
      ``"input"`` noise and refuses ``"modified"``. ``numpy_ref`` and ``native``
      execute it; every other mode set or geometry is refused by name. ``None``,
      the default, carries no wall: the modes keep their own Marcatili loss, and a
      multimode taper its taper coupling (below).
    - ``taper_coupling``: on a multimode capillary whose radius varies along z,
      carry the **taper coupling** -- the connection ``<psi_m|d_z psi_n>`` that
      moves power between the ``HE_1m`` channels as the bore changes -- lossless or
      not. ``True``, the default; ``False`` is the explicit opt-out to the
      **adiabatic** route, on which each channel follows its local mode and
      exchanges no power. It is refused beside `wall`, which always carries the
      connection, and has no effect on one channel, a constant bore (a pressure
      gradient included, which does not change the modes' shape) or a set that is
      not the contiguous ``HE_11`` to ``HE_1N`` block
      (``docs/guide/waveguides/capillaries.md``, "Tapers").
    - ``model``: which Marcatili model the modes use; beside a supplied set, refused
      unless it is the default or the modes' own.
    - ``loss``: propagation loss of the modes; beside a supplied set, refused unless
      it is the default or the modes' own.
    - ``cladding``: cladding index; ``None`` is the package's cited silica. Beside a
      supplied set, refused unless it is the default or the modes' own.
    - ``temperature``: gas temperature, K, or a path of it along z in m. Sets the
      density the pressure path maps to, so it changes both the dispersion and the
      nonlinearity. A path also weights a steady-flow gradient by the gas's ``eta(T)
      T`` (docs/guide/waveguides/profiles.md), makes a shared mixture's populations
      independent parameters, and holds the rotational Raman populations at one
      reference temperature with a
      :class:`~solprop.exceptions.RamanTemperatureWarning`
      (:func:`solprop.nonlinear.mixtures.raman_temperature`). A constant path is the
      number, bit for bit. ``None`` (the default) is 293.15 K. A Mixture owns its
      temperature: beside one, a temperature must be omitted or be the mixture's own
      (the same number or path object), and any other is refused by name.
    - ``thg``: include third-harmonic generation in the Kerr term; ``None`` takes
      ``not envelope``.
    - ``plasma``: include photoionisation. ``None`` takes ``not envelope``. ``True``
      selects optmat's recommended rate model of each built-in gas (SEP 3, D1;
      :class:`~solprop.nonlinear.optmat_rates.OptmatRate`, or the model the
      selection's ``ionisation`` names), for a single gas and for every component of
      a mixture; a custom species needs its own rate. A rate object is used as
      given; a mapping selects a rate or switch per component label, and omitted
      labels are off.
    - ``preionisation``: initial ionised fraction in [0, 1], globally or per label:
      a number, or a path along z in m giving the fraction ionised before the pulse
      arrives at each position. None and omitted mapping labels retain the
      component's existing declaration.
    - ``plasma_integration``: how the field-grid plasma forms its population and
      free-current integrals (SEP 18): ``'band'`` filters each integrand with taps
      designed for the grid, exact to a certified 1e-5 across the active band;
      ``'trapezoid'`` is the plain cumulative trapezoid, short by up to 11 % per
      integral at the top of the default active band, which the frozen witness
      comparisons use. ``None`` takes
      :class:`~solprop.nonlinear.responses.PlasmaResponse`'s default. Refused with
      envelope plasma, whose currents are converted exactly, and when the plasma is
      off.
    - ``raman``: ``None`` (the default) turns the Raman response on for every
      species whose optmat response is curated -- N2, O2, H2, D2, CH4 and SF6 --
      leaves an atom without one, and warns
      (:class:`solprop.exceptions.SolpropWarning`) for a molecule whose response is
      provisional (N2O, unless the selection's ``raman`` is ``"provisional"``)
      rather than leaving it out silently; with ``kerr=False`` the default is off.
      ``False`` disables it; ``True`` selects the available models of a mixture and
      is refused by name for a single gas without data; explicit labels or
      :class:`~solprop.physdata.raman_types.MolecularOptions` select components and
      terms, and a named unavailable model fails explicitly.
    - ``n_r``: radial nodes of the fixed rule. ``None`` takes
      :func:`default_radial_order` of the mode count.
    - ``ionisation_field_range``: field amplitudes in V/m the rate table must cover.
      ``None`` derives them from the input's peak field over the axis and the
      transverse nodes; see :func:`solprop.driver.inputs.default_field_range`.
    - ``absorber``: the collar rates; ``None`` builds the default collar of the
      grid, and ``False`` switches both soft collars off and keeps the hard band
      mask (:func:`solprop.boundaries.no_absorber`; SEP 21).
    - ``boundary_check``: how the run watches its time window's edges (SEP 21): the
      default warns, then stops, when light above the shot-noise floor reaches them;
      ``None`` watches nothing, for a window kept narrow on purpose.
    - ``rtol``: relative step tolerance; ``None`` takes the route's default
      (:func:`~solprop.driver.defaults.default_rtol`). Calibrate it for a production
      run (``docs/dev/stepper.md`` section 3).
    - ``dz_max``: cap on the step, m; ``None`` takes half the capillary.
    - ``dz_init``: first step, m; ``None`` takes ``length * 1e-4``.
    - ``integrator``: the method that steps the run
      (``docs/guide/numerics/choosing-a-stepper.md``). ``None``, the default, takes
      ``Multistep()``: the low-call exponential Adams method, one response
      evaluation per step. ``RungeKutta()`` is the interaction-picture DOPRI5 march,
      for a cheap response or a state too large for the Adams history.
      ``LowStorageRungeKutta()`` replays `schedule` in three registers (SEP 24).
    - ``route``: the interface building the run, whose row of
      :data:`~solprop.driver.defaults.ROUTE_DEFAULTS` gives what the caller left:
      ``"capillary"`` (its TDSE variant beside a TDSE response) or ``"waveguide"``
      (its slab variant for a slab mode set).
    - ``step_planes``: also save the end of every `step_planes`-th accepted step
      (SEP 14); ``None`` is ``0``, or ``1`` with `tdse` (SEP 20 section 5.8).
    - ``save_planes``: positions to save at, m. Give this or `n_save`.
    - ``stats``: which statistics to record; ``None`` takes
      :func:`solprop.stats.default_spec` for a multimode run with the profiles, or
      for a mode-averaged one under `mode_average`, plus every statistic only a
      backend can compute.
    - ``node_budget``: bytes of node-chunk buffers the response pass may hold, an
      execution setting; ``None``, the default, leaves the backend's automatic
      choice.
    - ``backend``: registry name, or ``"auto"`` (the default) for ``native``, which
      refuses if it isn't installed or this processor is below its baseline. There
      is no fallback, and import or construction errors stay visible.
    - ``materials``: take the run's built-in materials from optmat under this
      selection: a gas or fraction mapping of built-in gases, ``"air"`` and
      ``Gas.Air`` resolve through :func:`solprop.physdata.species` and
      :meth:`solprop.physdata.mixtures.Mixture.air` with it, and the default
      cladding (``cladding=None`` with a numbered mode set) is optmat's recommended
      silica. A supplied mixture, mode set, cladding or susceptibility is taken as
      it is. The run records its material manifest. ``None`` (the default) is
      ``MaterialSelection()``.
    """

    radius: float | ZProfile | None
    length: float
    gas: Gas | str | Mapping[str, float] | Mixture
    pressure: Pressure | None = None
    waveguide: bool = False
    pressure_floor: float | None = None
    vacuum_end: VacuumEnd | None = None
    wavelength: float
    wavelength_limits: tuple[float, float] | None = None
    trange: float | None = None
    grid: TimeGrid | None = None
    envelope: bool = False
    fwhm: float | None = None
    width: float | None = None
    shape: str = "gauss"
    energy: float | None = None
    peak_power: float | None = None
    phases: Sequence[float] | np.ndarray = ()
    pulses: Sequence[Pulse] | None = None
    launch: ModalLaunch | None = None
    modes: int | Sequence[Mode] = 1
    mode_average: bool = False
    lattice: TubeLattice | None = None
    wall: CoherentWall | None = None
    taper_coupling: bool = True
    model: MarcatiliModel = "full"
    loss: LossSpec = True
    cladding: GlassIndex | complex | None = None
    susceptibility: GasSusceptibility | None = None
    temperature: float | ZProfile | None = None
    kerr: bool = True
    thg: bool | None = None
    plasma: bool | IonisationRate | Mapping[str, bool | IonisationRate] | None = None
    preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None = None
    plasma_integration: IntegrationPolicy | None = None
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions] | None = None
    n_r: int | None = None
    n_theta: int | None = None
    radial_rule: RadialRule = "gauss_kronrod"
    ionisation_field_range: tuple[float, float] | None = None
    absorber: Absorber | Literal[False] | None = None
    absorber_scheme: AbsorberScheme | None = None
    boundary_check: BoundaryCheck | None = DEFAULT_BOUNDARY_CHECK
    noise: NoiseSetting = "default"
    seed: int | None = None
    rtol: float | None = None
    atol: float = WAVEGUIDE_ATOL
    dz_max: float | None = None
    dz_init: float | None = None
    schedule: FixedSchedule | None = None
    integrator: Multistep | RungeKutta | LowStorageRungeKutta | None = None
    route: Literal["capillary", "waveguide"] = "capillary"
    interpolate_planes: bool = False
    step_planes: int | None = None
    save_planes: Sequence[float] | np.ndarray | None = None
    n_save: int | None = None
    stats: StatsSpec | None = None
    polarisation_taper: bool = False
    node_budget: int | None = None
    backend: str = "auto"
    polarisation: str | float | None = None
    materials: MaterialSelection | None = None
    tdse: TDSEGasResponse | None = None
    tdse_execution: TDSEExecution | None = None


def _from_hollow(  # noqa: PLR0915 - the route's whole build, keyword by keyword
    cls: type["Simulation"], request: _HollowRequest
) -> "Simulation":
    """Build a simulation of a pulse in a gas-filled hollow waveguide.

    The shared body of :meth:`from_capillary` and :meth:`from_waveguide`. The mode
    set is the first `modes` HE_1m modes of the capillary unless a set is given; the
    transverse integral is a fixed quadrature rule
    (``docs/guide/waveguides/mode-sets.md``); the
    gas density along z is the pressure path through the equation of state, and the
    same path sets the modes' core index, so a gradient fill disperses and broadens
    consistently.

    See ``docs/guide/setup/running.md`` for the discussion and a worked example.
    """
    backend = request.backend
    materials = request.materials
    n_save = request.n_save
    plasma = request.plasma
    raman = request.raman
    step_planes = request.step_planes
    thg = request.thg
    # The TDSE response owns the electronic response: build no other.
    atomic, plasma, raman, backend, n_save, step_planes = _entry_overrides(
        _tdse_entry(
            request.tdse,
            request.tdse_execution,
            trange=request.trange,
            kerr=request.kerr,
            mode_average=request.mode_average,
            unset={
                "plasma": plasma,
                "raman": raman,
                "preionisation": request.preionisation,
                "plasma_integration": request.plasma_integration,
                "ionisation_field_range": request.ionisation_field_range,
                "thg": thg,
            },
            backend=backend,
            step_planes=step_planes,
            save_planes=request.save_planes,
            n_save=n_save,
        ),
        plasma,
        raman,
        backend,
        n_save,
        step_planes,
    )
    if raman and not request.kerr:
        msg = (
            "raman=True needs kerr=True: a medium with a Raman response has an "
            "electronic one too, and on a field grid the drive form reaches the "
            "kernel through the Kerr term (docs/guide/media/raman.md)"
        )
        raise DriverError(msg, parameter=("raman", "kerr"), rule="raman-needs-kerr")
    if request.polarisation is not None:
        msg = (
            f"polarisation={request.polarisation!r}: this path is scalar and "
            "propagates one transverse component per mode row. Declare the "
            "polarisation in the modes instead -- a rectangular or slab guide's own "
            "polarisation, each partner a row of its own -- or, for a vector "
            "capillary run, a polarisation-resolved mode set, which is not "
            "validated end to end yet"
        )
        raise DriverError(
            msg, parameter="polarisation", rule="scalar-capillary-polarisation"
        )
    _check_wall_declaration(
        request.wall,
        model=request.model,
        loss=request.loss,
        lattice=request.lattice,
        taper_coupling=request.taper_coupling,
    )
    thg, plasma = (
        (not request.envelope) if thg is None else thg,
        (not request.envelope) if plasma is None else plasma,
    )
    active_plasma = (
        any(value is not False for value in plasma.values())
        if isinstance(plasma, Mapping)
        else plasma is not False
    )
    _refuse_plasma_settings(
        request.plasma_integration,
        envelope=request.envelope,
        active=active_plasma,
        thg=thg,
    )

    working = _capillary_grid(
        request.grid,
        request.length,
        request.wavelength,
        request.wavelength_limits,
        request.trange,
        envelope=request.envelope,
        thg=thg,
    )
    materials = MaterialSelection() if materials is None else materials
    _check_selection_carrier(
        materials, working.reference_wavelength, request.wavelength
    )
    bore = (
        _flow_radius(request.modes, request.pressure)
        if request.waveguide
        else as_profile(cast("float | ZProfile", request.radius))
    )
    # The fill's own models warn about the conditions they are evaluated at,
    # which the gas keyword chose.
    with _warnings_about("gas", _MATERIAL_WARNINGS):
        mixture = capillary_fill(
            request.gas,
            request.pressure,
            length=request.length,
            temperature=request.temperature,
            floor=request.pressure_floor,
            susceptibility=request.susceptibility,
            preionisation=request.preionisation,
            radius=bore,
            vacuum_end=request.vacuum_end,
            selection=materials,
        )
    _warn_boreless_gradient(mixture, bore)
    if request.waveguide:
        if isinstance(request.modes, int):
            raise DriverError(
                "from_waveguide requires supplied gas-filled modes",
                parameter="modes",
            )
        mode_set = _given_gas_modes(request.modes, mixture)
    else:
        mode_set = _capillary_modes(
            request.modes,
            cast("float | ZProfile", request.radius),
            mixture,
            length=request.length,
            lattice=request.lattice,
            model=request.model,
            loss=request.loss,
            cladding=_default_cladding(request.cladding)
            if isinstance(request.modes, int)
            else request.cladding,
        )
    rule = (
        _no_quadrature(
            mode_set, request.n_r, request.n_theta, radial_rule=request.radial_rule
        )
        if request.mode_average
        else _capillary_quadrature(
            mode_set, request.n_r, request.n_theta, radial_rule=request.radial_rule
        )
    )
    profiles = mixture_profiles(mixture)
    if request.waveguide:
        for mode in mode_set:
            paths: Mapping[str, ZProfile] = getattr(mode, "parameter_paths", {})
            profiles.update(
                {
                    name: path
                    for name, path in paths.items()
                    if name not in profiles
                    and (not path.is_constant or name == "core_radius")
                }
            )
    else:
        profiles["core_radius"] = as_profile(cast("float | ZProfile", request.radius))
    if raman is None:
        # The default follows the Kerr switch: a run without the electronic
        # response is a deliberately reduced model, and a field-grid Raman term
        # cannot stand without a Kerr term beside it anyway.
        raman = request.kerr and _default_raman(
            mixture, provisional=materials.raman == "provisional"
        )
    if atomic is not None:
        responses = [atomic.response]
    elif (
        isinstance(request.gas, (Gas, str))
        and not is_air(request.gas)
        and isinstance(raman, bool)
        and not isinstance(plasma, Mapping)
        and not isinstance(request.preionisation, Mapping)
    ):
        # The Raman response is optmat's, built from the fill (SEP 3); the other
        # responses keep this path.
        with _warnings_about("gas", _MATERIAL_WARNINGS):
            responses = _capillary_responses(
                request.gas,
                mixture.components[0].species,
                kerr=request.kerr,
                plasma=plasma,
                preionisation=0.0
                if request.preionisation is None
                else request.preionisation,
                integration=request.plasma_integration,
                wavelength=request.wavelength,
                envelope=request.envelope,
                # The single gas's electronic response is optmat's, at the
                # carrier or dispersive, as the selection's electronic policy
                # says (SEP 6).
                electronic=electronic_response(
                    mixture.components[0].species,
                    selection=materials,
                    wavelength=request.wavelength,
                    thg=thg,
                )
                if request.kerr
                else None,
                selection=materials,
            )
        # optmat's Raman response is the raman keyword's choice.
        with _about("raman"), _warnings_about("raman", _MATERIAL_WARNINGS):
            responses += (
                _optmat_raman(
                    request.gas,
                    mixture,
                    materials,
                    thg,
                    request.wavelength,
                    request.length,
                )
                if raman
                else []
            )
    else:
        with _warnings_about("gas", _MATERIAL_WARNINGS):
            responses = list(
                responses_for(
                    mixture,
                    kerr=request.kerr,
                    thg=thg,
                    plasma=plasma,
                    raman=raman,
                    wavelength=request.wavelength,
                    envelope=request.envelope,
                    length=request.length,
                    selection=materials,
                    plasma_integration=request.plasma_integration,
                )
            )
    with _about(("preionisation", "plasma")):
        _refuse_idle_preionisation(mixture, responses)
    pulse = _shaped_pulse(
        request.shape,
        wavelength=request.wavelength,
        fwhm=request.fwhm,
        width=request.width,
        energy=request.energy,
        peak_power=request.peak_power,
        phases=request.phases,
        pulses=request.pulses,
    )
    field0, launched = _hollow_input(pulse, request.launch, working, mode_set.modes)
    ionising = any(
        isinstance(one, PlasmaResponse | EnvelopePlasmaResponse) for one in responses
    )
    boundary, scheme = _boundary(request.absorber, request.absorber_scheme, working)
    # A hollow guide's default is the grid's collar, so there is always one.
    _warn_capillary_walkoff(mode_set.modes, working, cast("Absorber", boundary))
    responses, field_range, reference = _ionisation_inputs(
        responses,
        request.ionisation_field_range,
        _InputPeak(field0, mode_set, working, rule) if ionising else None,
    )
    row = route_row(
        request.route,
        tdse=atomic is not None,
        slab=any(isinstance(mode, SlabMode) for mode in mode_set.modes),
    )
    settings = keyword_settings(
        row,
        request.length,
        rtol=request.rtol,
        atol=request.atol,
        dz_max=request.dz_max,
        dz_init=request.dz_init,
        save_planes=_save_planes(request.length, request.save_planes, n_save),
        schedule=request.schedule,
        integrator=request.integrator,
        interpolate_planes=request.interpolate_planes,
        step_planes=step_planes,
        density=None,
        absorber=boundary,
        absorber_scheme=scheme,
        boundary_check=request.boundary_check,
        quadrature=rule,
        components=None
        if rule is None
        else _hollow_components(supplied=not isinstance(request.modes, int)),
        ionisation_field_range=field_range,
        node_budget=request.node_budget,
        stats=(
            _with_integration_statistics(
                _hollow_stats(
                    with_plasma=ionising,
                    mixture=mixture,
                    profiles=profiles,
                    rule=rule,
                )
                if request.waveguide
                else _capillary_stats(
                    with_plasma=ionising, mixture=mixture, averaged=rule is None
                ),
                responses,
            )
            if request.stats is None
            else request.stats
        )
        if atomic is None or request.stats is not None
        else _tdse_stats(
            _capillary_stats(with_plasma=False, mixture=mixture, averaged=rule is None)
        ),
        polarisation_taper=request.polarisation_taper,
        backend=backend,
        tdse_execution=None if atomic is None else atomic.execution,
    )
    noise_model, resolution = resolve_noise(row, request.noise, request.seed)
    draw = functools.partial(_shot_noise, noise_model, working, n_modes=len(mode_set))
    simulation = cls(
        working,
        mode_set.modes,
        responses,
        field0,
        settings=settings,
        noise=draw(request.seed),
        noise_resolution=resolution,
        profiles=profiles,
        wall=request.wall,
        taper_coupling=request.taper_coupling,
        materials=materials,
        configuration={
            "medium": mixture,
            **(
                {
                    "waveguide": {
                        "modes": [ModeRecord.from_mode(mode) for mode in mode_set],
                        "length": request.length,
                        "temperature": mixture.temperature,
                    }
                }
                if request.waveguide
                else {
                    "capillary": {
                        "radius": request.radius,
                        "length": request.length,
                        "temperature": mixture.temperature,
                    }
                }
            ),
            "pulse": pulse,
            **({} if launched is None else {"modal_launch": launched}),
            **({} if atomic is None else {"tdse_entry": atomic.record}),
        },
    )
    # A TDSE run keeps no selection: a mixture update must not rebuild the
    # rate-based responses in its place.
    simulation._mixture_selection = (
        None
        if atomic is not None
        else ResponseSelection(
            request.kerr,
            thg,
            plasma,
            raman,
            request.wavelength,
            request.envelope,
            request.length,
            selection=materials,
            reference_field=reference,
            field_range=field_range,
            plasma_integration=request.plasma_integration,
        )
    )
    return _redrawable(simulation, draw, request.seed)


def _refuse_plasma_settings(
    integration: IntegrationPolicy | None, *, envelope: bool, active: bool, thg: bool
) -> None:
    """Refuse plasma settings that cannot act together.

    Envelope plasma cannot carry the third harmonic, and a running-integral rule
    belongs to the field-grid plasma column: given beside envelope plasma, or with
    the plasma off, nothing would read it.
    """
    if envelope and active and thg:
        msg = "envelope plasma cannot be combined with THG"
        raise DriverError(
            msg, parameter=("envelope", "plasma", "thg"), rule="envelope-plasma-no-thg"
        )
    if integration is None:
        return
    if envelope:
        msg = (
            f"plasma_integration={integration!r} selects the field-grid plasma "
            "column's running-integral rule; envelope plasma converts its currents "
            "with exact inverse-frequency factors and takes no such rule"
        )
        raise DriverError(
            msg,
            parameter=("plasma_integration", "envelope"),
            rule="plasma-integration-field-grid",
        )
    if not active:
        msg = (
            f"plasma_integration={integration!r} is given but the plasma is off; "
            "turn it on (plasma=True) or drop plasma_integration"
        )
        raise DriverError(
            msg,
            parameter=("plasma_integration", "plasma"),
            rule="plasma-integration-needs-plasma",
        )


def _refuse_idle_preionisation(mixture: Mixture, responses: Sequence[object]) -> None:
    """Refuse a declared initial ionised fraction that no plasma channel carries.

    A component's preionisation reaches the run only through its plasma or
    envelope-plasma response. Without one the declaration would be recorded in the
    run's metadata and ignored by the propagation, so it is refused by name.
    """
    carried = {
        _population_label(one, mixture)
        for one in responses
        if isinstance(one, PlasmaResponse | EnvelopePlasmaResponse)
    }
    for component in mixture.components:
        value = component.preionisation
        declared = isinstance(value, ZProfile) or value != 0
        if declared and component.label not in carried:
            msg = (
                f"preionisation is declared for {component.label!r}, which has no "
                "plasma response, so no channel would carry it; turn its plasma on "
                "(plasma=True or plasma={label: True}) or drop the preionisation"
            )
            raise DriverError(msg)


def _population_label(
    response: PlasmaResponse | EnvelopePlasmaResponse, mixture: Mixture
) -> str:
    """Resolve an existing plasma declaration to its already-validated population."""
    if response.species is not None:
        return response.species
    if len(mixture.components) == 1:
        return mixture.labels[0]
    name = getattr(response.gas, "name", str(response.gas))
    return next(
        component.label
        for component in mixture.components
        if component.species.name == name
    )


def _check_wall_declaration(
    wall: CoherentWall | None,
    *,
    model: MarcatiliModel,
    loss: LossSpec,
    lattice: TubeLattice | None,
    taper_coupling: bool = True,
) -> None:
    """Refuse, at the keyword the caller typed, a wall this run cannot carry.

    The highest of the coherent wall's doors, and the only one that can name the
    argument rather than the object it became: the backend selection, the loss
    setting the block would replace, the capillary model it is not derived for and
    the structured cladding it does not describe. Each is refused again further
    down -- in :func:`solprop.modes.capillary.wall_operator` and at the backend --
    so nothing depends on this one being reached.

    - ``wall``: what was declared. ``None`` leaves every other argument alone.
    """
    if wall is None:
        return
    if not taper_coupling:
        msg = (
            "taper_coupling=False declines the taper connection, and a coherent wall "
            "block always carries it: its basis is attached to the local radius. "
            "Leave taper_coupling at its default with wall=, or pass wall=None for "
            "the adiabatic route"
        )
        raise DriverError(
            msg, parameter=("wall", "taper_coupling"), rule="wall-keeps-taper-coupling"
        )
    if loss is not True:
        msg = (
            f"a coherent wall block is the loss model, and this run declares "
            f"loss={loss!r}: the condensed Robin boundary condition replaces the "
            "Marcatili attenuation of every channel, so a LossSpec beside it would "
            "be discarded in silence. Pass loss=True with wall=, or leave wall=None "
            "and keep the uncoupled loss you asked for"
        )
        raise DriverError(msg, parameter=("wall", "loss"), rule="wall-is-the-loss")
    if model != "full":
        msg = (
            f"a coherent wall block needs one capillary model, the full one, and "
            f"this run declares model={model!r}: the condensation inverts the full "
            "model's transverse eigenvalue, and the reduced model's leading-order "
            "carrier beside it would be overridden rather than propagated"
        )
        raise DriverError(msg, parameter=("wall", "model"), rule="wall-full-model")
    if lattice is not None:
        msg = (
            "a coherent wall block assumes one uniform dielectric wall at a single "
            "radius, and this run declares a tube lattice: an antiresonant guide's "
            "loss is its tubes' resonances, not a Robin condition on a bare "
            "capillary. Leave wall=None and keep the Vincetti loss, or drop the "
            "lattice"
        )
        raise DriverError(msg, parameter=("wall", "lattice"), rule="wall-no-lattice")


# Positions along the capillary at which the modes' group delays are read.
_WALKOFF_SAMPLES: Final = 33


_PACKAGE_FILES: Final = (str(Path(__file__).resolve().parents[1]),)
"""Frames a fill warning skips, so that it names the caller's line, not solprop's."""


def _flow_radius(
    modes: int | Sequence[Mode], pressure: Pressure | None
) -> ZProfile | None:
    """Return the core radius a supplied mode set's gas flows through, if circular.

    Every mode that declares ``core_radius`` -- every ``MarcatiliMode`` subclass: a
    capillary, a Vincetti mode, a tabulated hollow mode and a ``TabulatedMode`` built
    from arrays -- is taken as a circular bore of that declared radius, which sets the
    Poiseuille conductance and the choke (for a tube lattice the gaps between the tubes
    are ignored). A rectangular guide's duct resistance and a slab's flow between plates
    are not modelled: they keep the uniform law, and warn if their widths or gap vary
    under a pressure gradient.

    - Returns: the first mode's core radius in m, or ``None`` for a guide without one.
    """
    if isinstance(modes, int) or not modes:
        return None
    first = modes[0]
    radius = getattr(first, "core_radius", None)
    if isinstance(radius, ZProfile):
        return radius
    widths = [getattr(first, name, None) for name in ("width_x", "width_y")]
    varying = any(isinstance(w, ZProfile) and not w.is_constant for w in widths)
    if varying and isinstance(pressure, Sequence):
        # A slab declares only width_x, its half-gap: flow between plates goes as
        # a^3 per unit width, which is not modelled either.
        message = (
            "the slab's gap varies along z, and its pressure gradient follows the "
            "uniform law: flow between plates is not modelled, so the fill ignores "
            "the gap"
            if widths[1] is None
            else "the rectangular guide's widths vary along z, and its pressure "
            "gradient follows the uniform law: a rectangular duct's hydraulic "
            "resistance is not modelled, so the fill ignores the widths"
        )
        warnings.warn(
            message,
            GasFlowWarning,
            stacklevel=2,
            skip_file_prefixes=_PACKAGE_FILES,
        )
    return None


def _warn_boreless_gradient(mixture: Mixture, radius: ZProfile | None) -> None:
    """Warn if a gradient built without the bore fills a guide whose radius varies.

    A :class:`~solprop.profiles.PressureGradient` supplied directly, or inside a
    Mixture, was built by its caller; if it does not know the radius, its ``p^2`` is
    linear in ``z`` although a taper's is linear in ``int a^-4 dz``.

    - ``radius``: the guide's core radius in m, if it has one.
    """
    shared = mixture.shared
    if (
        radius is None
        or radius.is_constant
        or not isinstance(shared, PressureGradient)
        or shared.is_constant
        or shared.radius is not None
    ):
        return
    warnings.warn(
        "the pressure gradient was built without the bore, and the guide's core "
        "radius varies along z: its p^2 is linear in z, whereas a taper's steady flow "
        "makes p^2 linear in the hydraulic resistance int a^-4 dz, so a narrow waist "
        "carries most of the drop. Build it with PressureGradient.from_stages(..., "
        "radius=) or solprop.gasflow.steady_fill, or pass the end pressures as a pair "
        "and let the constructor build the fill",
        GasFlowWarning,
        stacklevel=2,
        skip_file_prefixes=_PACKAGE_FILES,
    )


def _warn_capillary_walkoff(
    modes: Sequence[Mode], grid: TimeGrid, boundary: Absorber
) -> None:
    """Warn if the mode set's group-delay walk-off would survive the temporal collar.

    A mode set shares one co-moving frame, so the group delay between its modes is
    physics the operator carries: mode ``p`` drifts against the first by the integral of
    ``beta_1,p - beta_1,1`` along the fibre, 43 fs per metre between HE_11 and HE_12 of
    a 125 um capillary. The span of the set, integrated over the length with the modes'
    own paths, is what :func:`solprop.boundaries.warn_walkoff` needs. It is read at the
    reference frequency alone: a band-wide span would be dominated by band edges
    carrying nothing.

    - ``boundary``: the collar the run will use, for the length that sets its temporal
      strength. A spectral-only collar has no temporal part to check.
    """
    if len(modes) < 2 or not np.any(boundary.temporal_rate):
        return
    frame = Frame.comoving(grid)
    positions = np.linspace(0.0, grid.length, _WALKOFF_SAMPLES)
    delays = np.array(
        [
            [frame_quantities(mode, frame, float(z))[1] for mode in modes]
            for z in positions
        ]
    )
    spread = np.max(delays, axis=1) - np.min(delays, axis=1)
    span = float(np.trapezoid(spread, positions))
    warn_walkoff(grid, span, boundary_length=boundary.in_stage_length)


def default_radial_order(n_modes: int) -> int:
    """Return the radial order of the fixed rule a capillary run uses by default.

    The rule has to integrate the *quartic* of the mode profiles, because the Kerr
    polarisation is cubic in the field and its projection multiplies by one more
    profile. Those are Bessel functions of ``r/a``, so a Gauss rule converges spectrally
    and the only question is where. Two measurements set the number: the quartic overlap
    tensor of four HE_1m modes is converged to 2e-14 at ``n_r = 32``
    (``docs/guide/waveguides/mode-sets.md``), and the embedded error estimate for three
    modes falls 1.6e-1, 2.1e-2, 4.1e-5, 7.2e-9, 1.2e-15 at ``n_r = 8, 16, 24, 32, 48``.
    Thus 32 clears the alarm by two orders on three modes and is the reference's
    rounding on four. Eight nodes per mode carries that scaling to larger sets,
    since the highest radial order in the quartic grows with the set.

    A flat, larger order whatever the mode count would be safer and slower; 32 nodes on
    the sets a capillary run actually uses, with the alarm watching, is the trade this
    makes. The number rests on these two measurements and has not been calibrated
    further.

    Parameters
    ----------
    n_modes : int
        How many modes the set carries, >= 1.

    Returns
    -------
    int
        Radial nodes asked of
        :func:`solprop.modes.quadrature.transverse_quadrature`, which with
        ``radial_rule="gauss_kronrod"`` rounds an even count up by one so that the
        rule is the Kronrod extension of the half-order Gauss rule.

    Examples
    --------
    >>> from solprop.driver.routes.hollow import default_radial_order
    >>> default_radial_order(1), default_radial_order(4), default_radial_order(8)
    (32, 32, 64)
    """
    return max(_MINIMUM_RADIAL_ORDER, _RADIAL_NODES_PER_MODE * n_modes)


def _capillary_grid(
    grid: TimeGrid | None,
    length: float,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
    *,
    envelope: bool,
    thg: bool,
) -> TimeGrid:
    """Return the grid a capillary run propagates on.

    - ``length``: capillary length, m.
    - ``wavelength``: reference wavelength, m.
    - ``wavelength_limits``: window limits, m.
    - ``trange``: requested time window, s.
    """
    if grid is not None:
        if wavelength_limits is not None or trange is not None:
            msg = (
                "give a grid or the wavelength_limits and trange to build one, not "
                "both: the keywords would be ignored"
            )
            given = {"wavelength_limits": wavelength_limits, "trange": trange}
            raise DriverError(
                msg,
                parameter=(
                    "grid",
                    *(name for name, value in given.items() if value is not None),
                ),
                rule="grid-or-limits",
            )
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "a capillary run needs wavelength_limits and trange to build its grid, "
            "or a grid of your own"
        )
        raise DriverError(
            msg,
            parameter=_unset(wavelength_limits=wavelength_limits, trange=trange),
            rule="grid-or-limits",
        )
    with _about(renamed=_GRID_KEYWORDS):
        if envelope:
            return EnvelopeGrid.from_limits(
                length,
                wavelength,
                wavelength_limits,
                trange,
                thg=thg,
            )
        return FieldGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _capillary_modes(
    modes: int | Sequence[Mode],
    radius: float | ZProfile,
    mixture: Mixture,
    *,
    length: float,
    lattice: TubeLattice | None = None,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> ModeCollection:
    """Return the mode set under the declared capillary material.

    - ``radius``: core radius in m, used when constructing a numbered mode set.
    - ``length``: the run's length in m, over which a supplied set's radius is compared.
    """
    if isinstance(modes, int) and modes < 1:
        raise DriverError(
            f"a capillary mode set needs at least one mode, got {modes}",
            parameter="modes",
        )
    if lattice is not None:
        if not isinstance(modes, int):
            raise DriverError(
                "lattice is refused: the given modes carry their own geometry",
                parameter=("lattice", "modes"),
                rule="lattice-builds-modes",
            )
        if model == "reduced":
            raise DriverError(
                'lattice is incompatible with model="reduced"',
                parameter=("lattice", "model"),
                rule="lattice-full-model",
            )
        return ModeCollection.vincetti(
            radius,
            count=modes,
            mixture=mixture,
            lattice=lattice,
            loss=loss,
            cladding=cladding,
        )
    if isinstance(modes, int):
        return ModeCollection(
            tuple(
                MarcatiliMode(
                    radius,
                    mixture=mixture,
                    m=index + 1,
                    model=model,
                    loss=loss,
                    cladding=cladding,
                )
                for index in range(modes)
            )
        )
    if not modes:
        # The capillary wording, which the int branch above already uses: the
        # waveguide entry has its own.
        raise DriverError(
            "a capillary mode set needs at least one mode", parameter="modes"
        )
    given = _given_gas_modes(modes, mixture)
    _refuse_ignored_geometry(
        given, radius, length=length, model=model, loss=loss, cladding=cladding
    )
    return given


# Two radius paths agree when they agree to this relative tolerance at every sampled
# position: a supplied set and a declared radius built from the same number or the
# same profile agree exactly, so this only forgives the last bits of a radius
# computed two ways, and a real disagreement is a percent or more.
_RADIUS_RTOL = 1e-12


def _refuse_ignored_geometry(
    modes: Sequence[Mode],
    radius: float | ZProfile,
    *,
    length: float,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> None:
    """Refuse, by name, a declared capillary argument a supplied mode set ignores.

    A supplied set is the run's geometry: its modes keep their own radius, model, loss
    and cladding and are only rebound to the declared fill, so an argument that
    disagrees with them would be silently discarded. The radius is always declared -- it
    is positional -- and is compared with each mode's own ``core_radius`` path over the
    run, breakpoints included, so a taper beside a constant bore, or a different bore,
    is named. `model`, `loss` and `cladding` are compared only when they differ from
    their defaults, which is the only way this layer can tell a declaration from an
    omission: a default beside a mode that says otherwise was not asked for.

    - ``modes``: the supplied set, non-empty.
    - ``radius``: the declared core radius, m.
    - ``length``: the run's length, m, over which the radius paths are compared.
    """
    declared = as_profile(radius)
    for index, mode in enumerate(modes):
        own = getattr(mode, "core_radius", None)
        if own is None:
            msg = (
                "from_capillary's radius is the capillary's bore, and supplied mode "
                f"{index} ({type(mode).__name__}) declares no core radius to compare "
                "it with, so the radius would be ignored: propagate a mode set that "
                "is not a capillary's with from_waveguide or prop_waveguide"
            )
            raise DriverError(msg, parameter=("radius", f"modes[{index}]"))
        if not _same_radius(declared, own, length):
            msg = (
                f"from_capillary's radius disagrees with supplied mode {index} "
                f"({mode.channel_label.display_label}): a supplied mode set is the "
                "run's geometry and keeps its own core radius, so the declared one "
                "would be ignored. Declare the modes' own radius (the same number or "
                "profile) or build the modes on the radius wanted"
            )
            raise DriverError(msg, parameter=("radius", f"modes[{index}]"))
        _refuse_ignored_setting(index, mode, "model", model, default="full")
        _refuse_ignored_setting(index, mode, "loss", loss, default=True)
        _refuse_ignored_setting(index, mode, "cladding", cladding, default=None)


def _refuse_ignored_setting(
    index: int, mode: Mode, name: str, declared: object, *, default: object
) -> None:
    """Refuse a non-default capillary setting that a supplied mode does not carry.

    - ``name``: the keyword, which is also the mode's attribute: ``"model"``, ``"loss"``
      or ``"cladding"``.
    """
    if declared is default or declared == default:
        return
    own = getattr(mode, name, None)
    same = own is declared or (
        not callable(own) and not callable(declared) and own == declared
    )
    if not same:
        msg = (
            f"from_capillary's {name}={declared!r} disagrees with supplied mode "
            f"{index}, which carries {name}={own!r}: a supplied mode set keeps its "
            f"own {name}, so the declaration would be ignored. Declare it on the "
            f"modes, or leave {name} at its default beside them"
        )
        raise DriverError(msg, parameter=(name, f"modes[{index}]"))


def _same_radius(declared: ZProfile, own: ZProfile, length: float) -> bool:
    """Return whether two radius paths agree over a run of `length`.

    - ``declared``: the declared core radius, m.
    - ``own``: a supplied mode's core radius, m.
    - ``length``: the run's length, m.
    - Returns: whether they agree to ``_RADIUS_RTOL`` at 65 equally spaced positions,
      both paths' breakpoints and the midpoints between all of them.
    """
    if declared is own:
        return True
    start = max(0.0, declared.z_span[0], own.z_span[0])
    end = min(length, declared.z_span[1], own.z_span[1])
    edges = np.unique(
        np.concatenate(
            [np.linspace(start, end, 65), declared.breakpoints, own.breakpoints]
        ).clip(start, end)
    )
    positions = np.unique(np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])]))
    return bool(
        np.allclose(own(positions), declared(positions), rtol=_RADIUS_RTOL, atol=0.0)
    )


def _given_gas_modes(modes: Sequence[Mode], mixture: Mixture) -> ModeCollection:
    """Bind supplied modes and retain the existing orthonormality gate."""
    if not modes:
        raise DriverError(
            "a waveguide mode set needs at least one mode", parameter="modes"
        )
    rebound: list[Mode] = []
    for index, mode in enumerate(modes):
        with _about(f"modes[{index}]"):
            rebound.append(_rebind_mode(mode, mixture))
    given = ModeCollection(tuple(rebound))
    if len(given) > 1:
        overlaps = given.orthonormality()
        worst = float(np.max(np.abs(overlaps - np.eye(len(given)))))
        if worst > ORTHONORMALITY_TOLERANCE:
            raise DriverError(
                f"the given modes are not an orthonormal set: the largest deviation "
                f"of the overlap matrix from the identity is {worst:.3e}, above "
                f"{ORTHONORMALITY_TOLERANCE:.0e}. One synthesis matrix and its dual "
                "represent an orthonormal basis and nothing else",
                parameter="modes",
            )
    return given


_NODE_MAXIMUM_PLASMA = (
    "electron_density",
    "peak_ionisation_rate",
    "species_electron_density",
    "species_peak_ionisation_rate",
)
"""The plasma statistics read at a sampled node rather than on the axis."""

_SAMPLED_PEAK_THRESHOLD = 0.999
"""Innermost field scale below which a node maximum warns (SEP 19, Q8).

At 0.999 the innermost node sees 99.8% of the axial intensity, so a rate of order 10
there is within 2% of its axial value; every Gauss-Kronrod and Gauss rule in r/a of
eight or more nodes is closer still."""


def _warn_sampled_peak(rule: TransverseQuadrature | None, spec: StatsSpec) -> None:
    """Warn once per run when a node maximum is read far from the axis (SEP 19, Q8).

    A field-scale Gauss rule of few nodes integrates the source but has no node near
    the axis, so a node-maximum plasma statistic samples the axial peak at its
    innermost node, which a steep rate under-reports. The value is unchanged; the
    warning names the node and the on-axis statistics that read the axis.
    """
    if rule is None or rule.radial_rule != "field_scale":
        return
    asked = sorted(set(_NODE_MAXIMUM_PLASMA) & {*spec.names, *spec.periodic})
    if not asked:
        return
    eta = float(j0(jn_zeros(0, 1)[0] * float(np.min(rule.xi_r))))
    if eta < _SAMPLED_PEAK_THRESHOLD:
        warnings.warn(
            f"the field-scale rule's innermost node is at eta = {eta:.3f}, "
            f"{eta**2:.1%} of the axial intensity, so {asked} are node maxima "
            f"there: a rate of order K reads about eta^(2K) of its axial value. "
            "Request the on_axis_* statistics for the axis, or use the Radau rule, "
            "whose first node is on it",
            SampledPeakWarning,
            stacklevel=3,
        )


def _capillary_quadrature(
    modes: ModeCollection,
    n_r: int | None,
    n_theta: int | None,
    *,
    radial_rule: RadialRule,
) -> TransverseQuadrature:
    """Return the fixed transverse rule of a capillary run.

    - ``n_r``: radial nodes; ``None`` takes the module's
      :func:`~solprop.driver.routes.hollow.default_radial_order`.
    - ``n_theta``: azimuthal nodes; ``None`` takes the rule's own default where one is
      needed.
    """
    if modes.modes[0].dimlimits(0.0)[0] == "slab":
        if n_theta is not None:
            msg = (
                f"n_theta={n_theta} was given for a slab set: the planar guide is "
                "uniform in y, so its rule has one y node of unit width and n_r "
                "alone is the x node count"
            )
            raise DriverError(msg, parameter="n_theta")
        # The measured bound of the x rule's quartic products is the Cartesian
        # per-axis one; the default takes whichever is larger.
        needed = cartesian_nodes(max(mode.azimuthal_order for mode in modes.modes))
        order = max(default_radial_order(len(modes)), needed) if n_r is None else n_r
        return transverse_quadrature(modes.modes, n_r=order, radial_rule=radial_rule)
    if n_theta is not None and not modes.needs_full_rule:
        msg = (
            f"n_theta={n_theta} was given for a set whose modes all have azimuthal "
            "order zero: |J_0(u r/a)| does not depend on the angle, so the rule is "
            "radial and every azimuthal node is a copy of the same integrand"
        )
        raise DriverError(msg, parameter="n_theta")
    order = default_radial_order(len(modes)) if n_r is None else n_r
    if modes.modes[0].dimlimits(0.0)[0] == "cartesian":
        # For a Cartesian set n_r and n_theta are the x and y node counts, so
        # both defaults take the measured aliasing bound of the set's highest
        # transverse order rather than the radial and azimuthal defaults, which
        # under-resolve y from order two upwards.
        needed = cartesian_nodes(max(mode.azimuthal_order for mode in modes.modes))
        return transverse_quadrature(
            modes.modes,
            n_r=max(order, needed) if n_r is None else n_r,
            n_theta=needed if n_theta is None else n_theta,
            radial_rule=radial_rule,
        )
    if n_theta is None:
        return transverse_quadrature(modes.modes, n_r=order, radial_rule=radial_rule)
    return transverse_quadrature(
        modes.modes, n_r=order, n_theta=n_theta, radial_rule=radial_rule
    )


def _no_quadrature(
    modes: ModeCollection,
    n_r: int | None,
    n_theta: int | None,
    *,
    radial_rule: RadialRule,
) -> None:
    """Check a mode-averaged hollow run and return its absent transverse rule.

    A mode-averaged run carries one mode as a uniform disc of its effective area, the
    one-node transform (:func:`solprop.translate.modal.mode_averaged_transform`), so it
    has no rule for the keywords that shape one to act on.

    - ``modes``: the mode set, which must hold exactly one mode.
    - ``n_r``: radial nodes, which must not be given.
    - ``n_theta``: azimuthal nodes, which must not be given.
    - ``radial_rule``: the radial rule family, which must be the default.
    """
    if len(modes) != 1:
        msg = (
            f"mode_average=True needs exactly one mode, and the set has {len(modes)}: "
            "energy moving between modes is what a mode-averaged run cannot carry "
            "(docs/guide/waveguides/mode-averaging.md)"
        )
        raise DriverError(
            msg, parameter=("mode_average", "modes"), rule="mode-average-one-mode"
        )
    given = [
        name
        for name, value, unset in (
            ("n_r", n_r, None),
            ("n_theta", n_theta, None),
            ("radial_rule", radial_rule, "gauss_kronrod"),
        )
        if value != unset
    ]
    if given:
        msg = (
            f"{', '.join(given)} given beside mode_average=True: they shape a "
            "transverse rule, and a mode-averaged run has none, its mode being a "
            "uniform disc of the effective area"
        )
        raise DriverError(
            msg, parameter=("mode_average", *given), rule="mode-average-no-quadrature"
        )


def _capillary_responses(
    gas: Gas | str,
    material: Species,
    *,
    kerr: bool,
    plasma: bool | IonisationRate,
    preionisation: float | ZProfile,
    integration: IntegrationPolicy | None,
    wavelength: float,
    envelope: bool,
    electronic: KerrResponse | DispersiveKerrResponse | None,
    selection: MaterialSelection,
) -> list[object]:
    """Return the Kerr and plasma responses of a single-gas capillary run.

    The Raman response, optmat's, is the fill's (:func:`_optmat_raman`).

    - ``material``: the fill's species as resolved under the run's selection, which the
      plasma response carries: it is never rebuilt from the name with the default
      selection, so the material manifest lists only the models that acted (Q45).
    - ``preionisation``: fraction already ionised before the pulse, a number or a path
      along z in m.
    - ``integration``: the field plasma column's running-integral rule; ``None`` takes
      the response's default.
    - ``wavelength``: reference wavelength, m, the rate's carrier unless the selection
      names one.
    - ``electronic``: the gas's electronic response under the selection's policy
      (:func:`solprop.nonlinear.mixtures.electronic_response`); ``None`` without one.
    """
    responses: list[object] = []
    if kerr:
        # Per particle: the density path supplies the rest (ADR-026,
        # docs/dev/decisions.md).
        assert electronic is not None
        responses.append(electronic)
    if plasma is not False:
        rate = _ionisation_rate(gas, plasma, wavelength, selection)
        if envelope:
            responses.append(
                EnvelopePlasmaResponse(material, rate, preionisation=preionisation)
            )
        else:
            chosen: dict[str, IntegrationPolicy] = (
                {} if integration is None else {"integration": integration}
            )
            responses.append(
                PlasmaResponse(material, rate, preionisation=preionisation, **chosen)
            )
    if not responses:
        msg = (
            "a capillary run with neither kerr nor plasma has no nonlinearity; use "
            "the linear propagator instead of a solver"
        )
        raise DriverError(
            msg, parameter=("kerr", "plasma"), rule="capillary-needs-nonlinearity"
        )
    return responses


def _default_raman(mixture: Mixture, *, provisional: bool = False) -> bool:
    """Decide what ``raman=None`` means for a fill.

    On when any component carries a Raman response: a built-in gas whose optmat
    response is curated (:func:`solprop.nonlinear.optmat_raman.raman_status`), or a
    custom species carrying a model with at least one term. An atom contributes
    nothing and says nothing. A molecule whose optmat response is provisional (N2O's
    bands, SEP 3, D13) is off and warns rather than silently going without: a user
    expecting every molecule's Raman response would otherwise not learn that nitrous
    oxide is the one curated only provisionally. Under
    ``MaterialSelection(raman="provisional")`` it takes that response instead (Q17),
    which warns once per run with its status and is recorded.
    """
    on = False
    for component in mixture.components:
        material = component.species
        if not built_in_population(material):
            model = material.raman
            on = on or (
                model is not None
                and (model.rotational is not None or model.vibrational is not None)
            )
            continue
        status = raman_status(material.name)
        if status == "validated" or (status == "provisional" and provisional):
            on = True
        elif status == "provisional":
            warnings.warn(
                SolpropWarning(
                    f"{component.label} ({material.name}) is a molecule whose Raman "
                    "response optmat has only provisionally, so it is off; pass "
                    "raman=False to say so, or MaterialSelection(raman='provisional') "
                    "to admit it",
                    parameter="raman",
                ),
                stacklevel=3,
            )
    return on


def _optmat_raman(  # noqa: PLR0917 - called positionally
    gas: Gas | str,
    mixture: Mixture,
    selection: MaterialSelection,
    thg: bool,
    wavelength: float,
    length: float,
) -> list[object]:
    """Return a single-gas fill's Raman response from optmat (SEP 3).

    - ``wavelength``: the grid reference wavelength in m, the carrier unless the
      selection names one.
    - ``length``: the run's length in m.
    """
    logger.debug("optmat Raman response of the %s fill", gas)
    response = mixture_raman(
        mixture,
        True,
        selection=selection,
        thg=thg,
        wavelength=wavelength,
        length=length,
        explicit=True,
    )
    return [] if response is None else [response]


def _ionisation_rate(
    gas: Gas | str,
    plasma: bool | IonisationRate,
    wavelength: float,
    selection: MaterialSelection,
) -> IonisationRate:
    """Return the ionisation rate a capillary run uses.

    ``plasma=True`` takes optmat's rate model
    (:func:`solprop.nonlinear.optmat_rates.selected_rate`, SEP 3, D1); an explicit
    rate object passes through unchanged. The models' physical limits are discussed
    in ``docs/guide/media/ionisation.md``.

    - ``wavelength``: reference wavelength, m, the rate's carrier unless the selection
      names one.
    """
    if isinstance(plasma, bool):
        return selected_rate(species(gas, selection=selection), selection, wavelength)
    return plasma


def _with_integration_statistics(
    spec: StatsSpec, responses: Sequence[object]
) -> StatsSpec:
    """Add the plasma integration diagnostics to a default spec under the band rule.

    The undershoot, work defect and boundary level of SEP 18 are recorded by default
    wherever the field plasma column integrates by the band rule, so the run's warning
    can read them back; a run on the plain trapezoid records none of them.
    """
    if not any(
        isinstance(one, PlasmaResponse) and one.integration == "band"
        for one in responses
    ):
        return spec
    extra = tuple(
        name for name in PLASMA_INTEGRATION_STATISTICS if name not in spec.names
    )
    return dataclasses.replace(spec, names=(*spec.names, *extra))


def _capillary_stats(
    *, with_plasma: bool, mixture: Mixture, averaged: bool = False
) -> StatsSpec:
    """Return the statistics a capillary run records unless told otherwise.

    The multimode defaults with the z-profiles, plus every statistic only a backend can
    compute: it holds the nodes the plasma was evaluated at, the mode set the axis
    needs, and the embedded quadrature estimate a later pass cannot reconstruct.

    - ``averaged``: whether the run is mode-averaged: the mode-averaged defaults then
      replace the multimode ones, and the quadrature estimate, which has no rule to
      read, is left out.
    """
    wanted = [
        name
        for name in BACKEND_STATISTICS
        if (with_plasma or name not in PLASMA_STATISTICS)
        and not (averaged and name == "quadrature_error")
    ]
    base = default_spec(
        "mode_averaged" if averaged else "multimode", with_profiles=True
    )
    wanted.extend(
        (
            "species_ionisation",
            "species_electron_density",
            "species_peak_ionisation_rate",
            "species_density",
            "species_ionisation_energy",
        )
    )
    if mixture.shared is None:
        base = dataclasses.replace(
            base, names=tuple(name for name in base.names if name != "pressure")
        )
    return dataclasses.replace(base, names=(*base.names, *wanted))


def _hollow_components(*, supplied: bool) -> tuple[str, ...] | None:
    """Return the transverse components a hollow-guide run resolves.

    The capillary route's own set (a mode count) is unrotated HE_1m, which points along
    x (``docs/dev/conventions.md`` section 5), and keeps ``("x",)`` bitwise. A supplied
    set -- through either entry, with or without a coherent wall -- resolves the
    components its modes need (``None``, inferred by
    :func:`solprop.translate.problem.assemble`), so a TE_0m, TM_0m, EH or rotated set
    keeps its E_y. A wall's coupled flow contracts the channel axis of the one-component
    state and never reads the synthesis, so the backends admit it beside two synthesised
    components. ``from_waveguide`` refuses a mode count before this, so only the
    capillary entry reaches the first branch.

    - Returns: ``("x",)``, or ``None`` to infer the components from the modes.
    """
    return None if supplied else DEFAULT_COMPONENTS


def _rebind_mode(mode: Mode, mixture: Mixture) -> Mode:
    """Bind a gas fill through the mode's explicit material interface.

    The mode's own gas, pressure and temperature (a temperature path included) are
    replaced by the mixture's; only its geometry is kept.
    """
    rebind: Callable[[Mixture], Mode] | None = getattr(mode, "with_mixture", None)
    if not callable(rebind):
        # The hint belongs to the mode that has a route: a step-index fibre is a
        # mode-averaged GNLSE run. Anything else gets the plain refusal.
        message = f"{type(mode).__name__} cannot replace its material with a Mixture"
        if isinstance(mode, StepIndexMode):
            message += "; use prop_gnlse(fibre=...) instead"
        raise DriverError(message)
    return rebind(mixture)


def _hollow_stats(
    *,
    with_plasma: bool,
    mixture: Mixture,
    profiles: Mapping[str, ZProfile],
    rule: TransverseQuadrature | None,
) -> StatsSpec:
    """Keep capillary statistics where the supplied geometry defines them.

    - ``rule``: the run's fixed transverse rule, whose embedded coarse rule the
      quadrature alarm needs; ``None`` for a mode-averaged run.
    - Returns: statistics with unsupported geometry paths omitted.
    """
    base = _capillary_stats(
        with_plasma=with_plasma, mixture=mixture, averaged=rule is None
    )
    dropped = {name for name in ("core_radius",) if name not in profiles}
    if rule is not None and not rule.kronrod:
        # Without an embedded coarse rule the estimate is NaN at every plane, so
        # the statistic is dropped rather than recorded empty. A Cartesian
        # product rule has no Kronrod extension at all (quadrature_rule stores
        # kronrod only for a polar rule), so the alarm is off for every
        # rectangular run, and the log below says so rather than leaving it silent.
        dropped.add("quadrature_error")
        logger.info(
            "the %s rule has no embedded coarse rule, so the quadrature alarm is "
            "off and quadrature_error is not recorded",
            rule.kind,
        )
    return dataclasses.replace(
        base, names=tuple(name for name in base.names if name not in dropped)
    )
