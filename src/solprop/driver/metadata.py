# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The record that says what produced a run, and whether it may be resumed.

:func:`metadata_for` builds a run's :class:`solprop.output.metadata.RunMetadata`: its
grid, modes, configuration, stepper, plane reference and recipe, whose configuration
hash is the resume key. :class:`RecipeSource` is what the driver needs of the recipe
that built a simulation: the record its output keeps. The manual pages are
``docs/guide/setup/running.md`` and ``docs/guide/results/resuming.md``.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from solprop.coupling import PlaneReference
from solprop.driver.material_output import material_record, processing_record
from solprop.grid import TimeGrid
from solprop.linear import Frame, frame_origin
from solprop.modes.collection import transverse_measure
from solprop.modes.material import JsonValue
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import TransverseQuadrature
from solprop.noise import NoiseRealisation
from solprop.output.metadata import RunMetadata
from solprop.output.records import (
    DefaultsResolution,
    GridRecord,
    ModeRecord,
    NoiseRecord,
    NoiseResolution,
    RecipeRecord,
    StepLimits,
    StepperRecord,
)
from solprop.physdata import Mixture
from solprop.physdata.selection import MaterialManifest
from solprop.spatial import SpatialMedium
from solprop.stepper.settings import StepperSettings
from solprop.translate.coupling import frame_record
from solprop.translate.envelope_plasma import assembly_record
from solprop.translate.flux import flux_record
from solprop.translate.plasma_integration import integration_record
from solprop.translate.plasma_source import plasma_source_record
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of, tdse_descriptor, transverse_of
from solprop.translate.run_request import RunRequest

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = ["RecipeSource", "metadata_for"]


class RecipeSource(Protocol):
    """What built a simulation from a recipe: it writes the record its output keeps.

    :class:`solprop.recipe.core.Recipe` is one (SEP 41, S2); the driver needs only this.
    """

    def record(self, simulation: "Simulation") -> RecipeRecord:
        """Return the authored and resolved recipe of a run about to start.

        Parameters
        ----------
        simulation : Simulation
            The simulation, translated, with its settings at run start.

        Returns
        -------
        solprop.output.records.RecipeRecord
            The record its output keeps.
        """
        ...


def _stepper_record(settings: StepperSettings) -> StepperRecord:
    """Return the run's stepper record, naming an Adams integrator where one steps it.

    The method is recorded in full, and the interval cap and start-up rule with the
    limits, so that a run and its resume hash the stepping they were taken with.
    """
    return StepperRecord.from_settings(
        settings.tableau,
        settings.controller,
        schedule=settings.schedule,
        limits=StepLimits(
            dz_min=settings.dz_min,
            dz_max=settings.dz_max,
            dz_init=settings.dz_init,
            stretch=settings.stretch,
            interval_cap=settings.interval_cap,
            auto_init=settings.auto_init,
            interpolate_planes=settings.interpolate_planes,
            step_planes=settings.step_planes,
        ),
        method=None if settings.integrator is None else repr(settings.integrator),
    )


def metadata_for(
    problem: PropagationProblem,
    request: RunRequest,
    grid: TimeGrid,
    *,
    modes: Sequence[Mode] = (),
    configuration: Mapping[str, object] | None = None,
    noise: NoiseRealisation | None = None,
    modal: Mapping[str, JsonValue] | None = None,
    materials: MaterialManifest | None = None,
    quadrature: TransverseQuadrature | None = None,
    frame: Frame | None = None,
    noise_resolution: NoiseResolution | None = None,
    defaults: DefaultsResolution | None = None,
) -> RunMetadata:
    """Build the record that says what produced a run, and whether it may be resumed.

    Everything that describes *what is computed* goes into the digest
    :meth:`solprop.output.metadata.RunMetadata.configuration_hash` takes -- the grid,
    the modes, the configuration, the noise and the stepper settings, plus the save
    planes and the statistics, which are read off the run's request here -- and nothing
    that describes *when* or *with which build*, so a resume survives an unrelated
    commit or a NumPy upgrade.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem the run will step.
    request : solprop.translate.run_request.RunRequest
        The request translation returned with it: its stepper settings, save planes
        and statistics enter the hash exactly as they did when the problem carried
        them.
    grid : solprop.grid.TimeGrid
        The grid the state lives on.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order.
    configuration : collections.abc.Mapping, optional
        The physics objects that built the run -- responses, the input, profiles,
        boundaries -- recorded by :func:`solprop.output.serialisation.serialise`.
        Nothing is dropped: a callable is recorded as its ``repr`` and marked
        unreproducible, which also makes the digest unstable across processes, so prefer
        objects that carry their data (``docs/guide/setup/running.md``).
    noise : solprop.noise.NoiseRealisation or None, optional
        The realisation the run carries, recorded with its seed.
    modal : collections.abc.Mapping or None, optional
        The general modal identity, as
        :func:`solprop.translate.general.modal_identity` writes it: the channel
        labels, the table, metric, response and guide identities, the frame, the
        active band and the accessors the basis admits. ``None`` for every other
        route. It is part of the resume key.
    materials : solprop.physdata.selection.MaterialManifest or None, optional
        The run's material manifest
        (:func:`solprop.driver.material_output.material_manifest`); ``None`` is the
        empty manifest of a run of solprop's own materials. Its identities and
        policies are part of the resume key.
    quadrature : solprop.modes.quadrature.TransverseQuadrature or None, optional
        The transverse rule of a resolved modal run, recorded with its nodes and
        physical weights on the first mode's cross-section and part of the resume
        key (SEP 19). ``None`` for every other route.
    frame : solprop.linear.Frame or None, optional
        The run's frame, whose reference frequency is the origin of the linear frame
        term on an envelope grid; ``None`` is the grid's co-moving frame. It enters
        the plane-reference record (SEP 31), not the hash.
    noise_resolution : solprop.output.records.NoiseResolution or None, optional
        How the noise model was chosen (SEP 37). Outside the hash.
    defaults : solprop.output.records.DefaultsResolution or None, optional
        Which stepping choices took the route's default. Outside the hash.

    Returns
    -------
    solprop.output.metadata.RunMetadata
        The record, ready for :meth:`solprop.output.store.Output.open`.

    Examples
    --------
    >>> from solprop.driver.metadata import metadata_for
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.assembly import assemble
    >>> grid = EnvelopeGrid.from_limits(0.01, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem, request = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> metadata = metadata_for(problem, request, grid, modes=[gnlse.mode])
    >>> metadata.simulation_kind, len(metadata.configuration_hash())
    ('mode_averaged', 64)
    """
    settings = request.stepper
    record = dict(configuration or {})
    if bool(problem.responses.plasma_channels):
        # The equation and admission policy are identity; the measured residual
        # belongs to backend execution evidence, like the TDSE certificate. Its
        # last bits may change with the host FFT implementation.
        source = plasma_source_record(problem.grids)
        record["plasma_source"] = {
            key: value for key, value in source.items() if key != "certificate"
        }
    # The planes and the statistics come from the request rather than from the caller:
    # a resume continues at a plane index, so two runs that differ in either are not the
    # same run, and neither is carried by StepperRecord.
    record["save_planes"] = np.asarray(settings.save_planes, dtype=np.float64)
    record["statistics"] = request.observation.stats
    measure = transverse_measure(modes) if modes else "area"
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "free" and geometry.shape[1] == 1:
        # A Cartesian line (one y row of unit measure) is per metre of y however the
        # simulation was built, not only through from_free.
        measure = "per_unit_width"
    if measure != "area":
        # A run uniform in y reports its powers per metre of y; processing and
        # plotting label them from this key (RunMetadata.transverse_measure). It is
        # written only where it is not the default, "area", which a reader assumes
        # where it is absent, so the record stays short.
        record["transverse_measure"] = measure
    if problem.responses.envelope_plasma is not None:
        record["statistic_rate_conventions"] = {
            name: "cycle-averaged"
            for name in ("peak_ionisation_rate", "species_peak_ionisation_rate")
            if name in request.observation.stats.names
            or name in request.observation.stats.periodic
        }
        record["envelope_plasma"] = assembly_record(
            problem.responses.envelope_plasma, problem.grids.omega
        )
    if problem.responses.plasma_integral.policy != "trapezoid":
        # The band rule's taps and policies (SEP 18, Q9): a resume that would
        # integrate the plasma differently is refused through the hash.
        record["plasma_integration"] = integration_record(
            problem.responses.plasma_integral
        )
    material = record.get("medium")
    if isinstance(material, SpatialMedium):
        record["spatial_medium"] = material
        material = material.material
    if not isinstance(material, Mixture) and modes:
        material = getattr(modes[0], "mixture", None)
    if isinstance(material, Mixture):
        record["medium"] = material_record(material)
        record["mixture_processing"] = processing_record(
            problem, request.observation.save_planes, grid, material, modes
        )
    geometry = problem.transform.geometry
    spatial_record: dict[str, object] | None = None
    if geometry is not None:
        # A hybrid planar column is a slab amplitude per y frequency: its x axis is the
        # slab rule's nodes, its spectrum the mode index and k_y, and its stored unit a
        # modal amplitude's.
        slab = geometry.kind == "slab"
        spatial_record = {
            "kind": geometry.kind,
            "shape": geometry.shape,
            "axes": geometry.axes,
            "axis_names": ("r",) if geometry.kind == "radial" else ("x", "y"),
            "axis_units": "m",
            "spectral_axes": geometry.spectral_axes,
            "spectral_axis_names": ("k_r",)
            if geometry.kind == "radial"
            else ("k_y",)
            if slab
            else ("k_x", "k_y"),
            "spectral_axis_units": "rad/m",
            "column_order": "r"
            if geometry.kind == "radial"
            else "p + n_p*iy"
            if slab
            else "ix + n_x*iy",
            "physical_field_units": "V/m",
            "stored_field_units": "V m"
            if geometry.kind == "radial"
            else "sqrt(W/m)"
            if slab
            else "V/m",
            "spatial_weights": geometry.spatial_weights,
            "spectral_weights": geometry.spectral_weights,
            "forward_scale": geometry.forward_scale,
            "inverse_scale": geometry.inverse_scale,
            "window": geometry.window,
            "absorber_rate": geometry.absorber_rate,
            "flux_model": "scalar-axial-admittance",
            "flux_plan": flux_record(flux_of(problem.route)),
        }
    descriptor = tdse_descriptor(problem.route)
    tdse_record = None if descriptor is None else dataclasses.asdict(descriptor)
    design = transverse_of(problem.route)
    return RunMetadata(
        reference=_plane_reference_record(problem, grid, record, frame=frame),
        transverse=None if design is None else design.record(),
        grid=GridRecord.from_grid(grid),
        modes=tuple(ModeRecord.from_mode(mode) for mode in modes),
        simulation_kind=(
            geometry.kind
            if geometry is not None
            else "mode_averaged"
            if problem.flags.mode_averaged
            else "multimode"
        ),
        configuration=record,
        stepper=_stepper_record(settings),
        noise=None if noise is None else NoiseRecord.from_realisation(noise),
        noise_resolution=noise_resolution,
        defaults=defaults,
        spatial=spatial_record,
        quadrature=None
        if quadrature is None or not modes
        else quadrature.record(modes[0].dimlimits(0.0)),
        modal=None if modal is None else dict(modal),
        materials=MaterialManifest() if materials is None else materials,
        initial_z=problem.initial.z,
        tdse=tdse_record,
    )


def _plane_reference_record(
    problem: PropagationProblem,
    grid: TimeGrid,
    configuration: Mapping[str, object],
    *,
    frame: Frame | None,
) -> dict[str, object]:
    """Return the plane-reference record of a run (SEP 31).

    A plane's basis is not recorded: it is derived when the plane is read
    (:meth:`solprop.coupling.plane.PlaneBasis.of_run`), so a run that is never
    coupled computes no basis identity.

    - ``configuration``: the run's configuration; a ``coupling`` entry supplies the
      inherited origin and the parent's provenance.
    - ``frame``: the run's frame, or ``None`` for the co-moving one.
    - Returns: ``carrier``, ``origin``, ``parent`` and ``frame``, as
      :attr:`solprop.output.metadata.RunMetadata.reference` documents them.
    """
    coupling = configuration.get("coupling")
    origin: object = PlaneReference().record()
    parent: object = None
    if isinstance(coupling, Mapping):
        coupled = cast("Mapping[str, object]", coupling)
        origin = coupled["origin"]
        parent = coupled["source"]
    return {
        "carrier": float(grid.omega0),
        "origin": origin,
        "parent": parent,
        "frame": _plane_frame(problem, grid, frame),
    }


def _plane_frame(
    problem: PropagationProblem, grid: TimeGrid, frame: Frame | None
) -> dict[str, object] | None:
    """Return the record of the frame a run's planes are referenced to (SEP 31).

    - ``frame``: the run's frame, or ``None`` for the co-moving one.
    - Returns: the record, or ``None`` for an operator supplied without frame tables.
    """
    # The co-moving frame keeps the carrier phase exactly when a harmonic term is
    # present, and then measures its delay from zero frequency (`frame_origin`).
    # `assemble` chose it from the declared responses and checked that choice against
    # the compiled carrier, so reading the compiled one here records the same frame.
    if frame is None:
        frame = Frame.comoving(grid, thg=problem.responses.harmonic_carrier)
    return frame_record(problem.linear, frame_origin(frame, grid.kind))
