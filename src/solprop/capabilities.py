# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What solprop can do: for a route, a species and a results file (SEP 41, S8, S16).

A front end asks :func:`route` what a route takes: its keywords, the values of each
enumerated one, and the **combination rules** its builder enforces, each with an id and
a reason. Every refusal that enforces a rule raises with that id as its ``rule`` and the
keywords as its ``parameter`` (:class:`~solprop.exceptions.SolpropError`), and a test
holds the raise sites to :data:`RULES`. A refusal that depends on a value (grid
coverage, a node budget, a material's validity range) is not a rule:
:meth:`~solprop.recipe.core.Recipe.check` finds it.

It asks :func:`species` what a built-in gas brings: its ionisation models and Raman
response with optmat's status of each, and the ranges optmat records for it.

A front end that opens a results file asks :func:`output` what it may offer: the
file's basis, and for each derived quantity whether it is a **physical flux** or a
**coordinate quantity**; which planes keep their field; which operations a reader
admits and which it refuses, with the refusal the reader itself gives; and whether
the file records a recipe and can continue. The refusals come from the same gates the
readers raise from (:func:`solprop.output.capability.refusal_for` and the
compressors' basis check), so the statement and the refusal cannot drift.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import numpy as np

from solprop.exceptions import DriverError, ProcessingError
from solprop.output.capability import (
    BasisCapability,
    capability_of,
    refusal_for,
    withheld_from,
)
from solprop.output.hdf5 import open_file
from solprop.output.resumable import Resumability, resumability
from solprop.output.schema import Output
from solprop.physdata.gases import Gas, as_gas, is_molecule
from solprop.physdata.selection import MaterialRecord

__all__ = [
    "OPERATIONS",
    "RULES",
    "IonisationModel",
    "OutputCapabilities",
    "Quantity",
    "QuantityKind",
    "RouteCapabilities",
    "Rule",
    "SpeciesCapabilities",
    "output",
    "route",
    "species",
]

QuantityKind = Literal["physical", "coordinate"]
"""Whether a derived quantity is a physical flux, or a quantity of the coordinates."""

OPERATIONS: Final = (
    "energy",
    "peak_power",
    "peak_intensity",
    "physical_energy",
    "reconstructed_flux",
    "electric_field",
    "compression",
)
"""The operations a statement covers: the accessors whose admission depends on the
basis, and compression, which needs orthogonal modal powers."""


@dataclass(frozen=True)
class Quantity:
    """What a derived quantity of a file is.

    Attributes
    ----------
    kind : {'physical', 'coordinate'}
        A physical flux, which sums to the pulse's power and energy, or a quantity
        of the basis's coordinates, which does not on a basis with cross terms.
    unit : str
        Its unit.
    """

    kind: QuantityKind
    unit: str


@dataclass(frozen=True)
class OutputCapabilities:
    """What a results file can give a reader.

    Attributes
    ----------
    basis : {'modal', 'general_modal', 'spatial'}
        The file's basis.
    simulation_kind : str
        The kind of run, as its record names it.
    quantities : collections.abc.Mapping
        :class:`Quantity` for ``power``, ``energy``, ``spectral_density`` and
        ``field``.
    field_planes : tuple of int
        The saved planes that keep their field.
    statistics_planes : tuple of int
        The saved planes that keep only their statistics row.
    refused : collections.abc.Mapping
        For each of :data:`OPERATIONS` a reader refuses, its refusal.
    needs : collections.abc.Mapping
        For an operation a reader admits only with something the file does not hold,
        what it needs: a general modal file's reconstruction needs its guide and
        reference tables, of which it holds the identities.
    recipe : bool
        Whether the file records the recipe it was built from.
    resumability : solprop.output.resumable.Resumability or None
        Whether it can continue, for a file given by its path.
    """

    basis: BasisCapability
    simulation_kind: str
    quantities: dict[str, Quantity]
    field_planes: tuple[int, ...]
    statistics_planes: tuple[int, ...]
    refused: dict[str, str]
    needs: dict[str, str]
    recipe: bool
    resumability: Resumability | None

    def admits(self, operation: str) -> bool:
        """Return whether a reader admits an operation on this file.

        Parameters
        ----------
        operation : str
            One of :data:`OPERATIONS`.

        Returns
        -------
        bool
            False when it is refused.
        """
        return operation not in self.refused


def _quantities(basis: BasisCapability) -> dict[str, Quantity]:
    """Return what each derived quantity is on a basis.

    Parameters
    ----------
    basis : {'modal', 'general_modal', 'spatial'}
        The basis.

    Returns
    -------
    dict
        The quantities: physical on an orthogonal modal basis, coordinate on a general
        one (whose physical energy is :func:`solprop.processing.modal.physical_energy`)
        and on a spatial grid, whose amplitudes are per transverse column.
    """
    kind: QuantityKind = "physical" if basis == "modal" else "coordinate"
    return {
        "power": Quantity(kind, "W"),
        "energy": Quantity(kind, "J"),
        "spectral_density": Quantity(kind, "J s/rad"),
        "field": Quantity(kind, "sqrt(W)"),
    }


def _compression_refusal(handler: Output) -> str | None:
    """Return the compressors' refusal of a file, or ``None``.

    Parameters
    ----------
    handler : solprop.output.schema.Output
        The opened file.

    Returns
    -------
    str or None
        The refusal text, as the compressors raise it.
    """
    from solprop.processing.compression import (  # noqa: PLC0415 - layer above
        _require_modal,  # pyright: ignore[reportPrivateUsage]
    )

    try:
        _require_modal(handler)
    except ProcessingError as error:
        return str(error)
    return None


def output(source: str | Path | Output) -> OutputCapabilities:
    """Say what a results file can give a reader (SEP 41, S16).

    Parameters
    ----------
    source : str, pathlib.Path or solprop.output.schema.Output
        The file, or an opened handler (then nothing is said of resuming it).

    Returns
    -------
    OutputCapabilities
        The statement.

    Raises
    ------
    solprop.exceptions.OutputError
        If the file cannot be read or holds no run record.

    Examples
    --------
    >>> from solprop.capabilities import output
    >>> from solprop.output.schema import _example_metadata
    >>> from solprop.output import MemoryOutput
    >>> handler = MemoryOutput()
    >>> handler.open(_example_metadata())
    >>> statement = output(handler)
    >>> statement.basis, statement.quantities["energy"].kind
    ('modal', 'physical')
    """
    if isinstance(source, Output):
        return _statement(source, None)
    path = Path(source)
    with open_file(path) as handler:
        return _statement(handler, resumability(path))


def _statement(handler: Output, resumable: Resumability | None) -> OutputCapabilities:
    """Return the statement for an opened handler.

    Parameters
    ----------
    handler : solprop.output.schema.Output
        The opened file.
    resumable : Resumability or None
        Its resumability, for a file.

    Returns
    -------
    OutputCapabilities
        The statement.
    """
    metadata = handler.metadata
    if metadata is None:
        from solprop.exceptions import OutputError  # noqa: PLC0415 - only here

        msg = "the output holds no run record"
        raise OutputError(msg)
    basis = capability_of(metadata)
    withheld = withheld_from(metadata)
    refused: dict[str, str] = {}
    for name in OPERATIONS[:-1]:
        refusal = refusal_for(basis, [name], withheld=withheld)
        if refusal is not None:
            refused[name] = refusal
    compression = _compression_refusal(handler)
    if compression is not None:
        refused["compression"] = compression
    needs: dict[str, str] = {}
    if basis == "general_modal" and metadata.modal is not None:
        held = sorted(key for key in ("tables", "guide") if key in metadata.modal)
        needs["reconstructed_flux"] = (
            "the guide and the reference tables that produced the file, which it "
            f"identifies ({', '.join(held) or 'nothing'}) but does not hold"
        )
    n_saved = handler.n_saved
    # Every handler answers which saved planes kept their field (SEP 22).
    kept = tuple(int(index) for index in np.asarray(handler["field_planes"]))
    others = tuple(index for index in range(n_saved) if index not in kept)
    recipe = getattr(handler, "recipe", None) is not None
    return OutputCapabilities(
        basis,
        metadata.simulation_kind,
        _quantities(basis),
        kept,
        others,
        refused,
        needs,
        recipe,
        resumable,
    )


# ------------------------------------------------------------------- routes (S8)


@dataclass(frozen=True)
class Rule:
    """A combination of keywords, route or species that a builder refuses.

    Attributes
    ----------
    id : str
        A stable short identifier in kebab case, which the refusal carries as its
        ``rule``.
    keywords : tuple of str
        The keywords the rule concerns: build keywords of its routes, or keys of the
        ``[execution]`` table. A refusal's ``parameter`` names those of them it
        found set, or missing.
    reason : str
        Why the combination is refused, in a sentence a front end can show.
    routes : tuple of str
        The routes whose builders enforce it.
    species_condition : str or None
        For a rule that depends on the fill, the species or form of ``gas`` it
        concerns; ``None`` for one that does not.

    Examples
    --------
    >>> from solprop.capabilities import RULES
    >>> RULES["raman-needs-kerr"].keywords
    ('raman', 'kerr')
    """

    id: str
    keywords: tuple[str, ...]
    reason: str
    routes: tuple[str, ...]
    species_condition: str | None = None


_BOTH: Final = ("gnlse", "capillary")
_GNLSE: Final = ("gnlse",)
_CAPILLARY: Final = ("capillary",)

_RULES: Final = (
    # The grid, the input pulse and the saved planes, on both routes.
    Rule(
        "grid-or-limits",
        ("grid", "wavelength_limits", "trange"),
        "A run's grid is given, or built from wavelength_limits and trange: one "
        "way, and complete.",
        _BOTH,
    ),
    Rule(
        "pulses-or-pulse-keywords",
        ("pulses", "fwhm", "width", "energy", "peak_power", "phases", "shape"),
        "pulses= describes every input pulse itself; the single-pulse keywords "
        "are then left unset.",
        _BOTH,
    ),
    Rule(
        "gauss-fwhm",
        ("fwhm", "width", "shape"),
        "A Gaussian input is specified by its intensity fwhm; width is the natural "
        "width of a sech**2 pulse.",
        _BOTH,
    ),
    Rule(
        "fwhm-or-width",
        ("fwhm", "width"),
        "A sech**2 input takes exactly one of fwhm and width.",
        _BOTH,
    ),
    Rule(
        "energy-or-peak-power",
        ("energy", "peak_power"),
        "An input pulse takes exactly one of energy and peak_power.",
        _BOTH,
    ),
    Rule(
        "save-planes-or-n-save",
        ("save_planes", "n_save"),
        "The saved planes are given as positions or as a count, not both.",
        _BOTH,
    ),
    Rule(
        "fields-planes-step-planes",
        ("fields", "step_planes"),
        "fields='planes' names saved planes by index, and step_planes adds the "
        "march's own steps, so the plane count is not known in advance.",
        _BOTH,
    ),
    Rule(
        "absorber-scheme-needs-absorber",
        ("absorber", "absorber_scheme"),
        "absorber=False switches the temporal collar off, leaving nothing for "
        "absorber_scheme to apply.",
        _BOTH,
    ),
    Rule(
        "seed-needs-noise",
        ("seed", "noise"),
        "A noise seed needs a noise model to draw.",
        _BOTH,
    ),
    # The integrator.
    Rule(
        "low-storage-needs-schedule",
        ("integrator", "schedule"),
        "The low-storage march replays a prescribed schedule.",
        _BOTH,
    ),
    Rule(
        "low-storage-no-interpolation",
        ("integrator", "interpolate_planes"),
        "The low-storage march lands on its schedule's edges and has no dense output.",
        _BOTH,
    ),
    Rule(
        "integrator-backends",
        ("integrator", "backend"),
        "The Multistep and low-storage integrators run on numpy_ref, numba_cpu and "
        "native only.",
        _BOTH,
    ),
    # The gnlse route.
    Rule(
        "gnlse-envelope-grid",
        ("grid",),
        "The GNLSE route propagates an envelope grid; a field-resolved run is the "
        "capillary route's.",
        _GNLSE,
    ),
    Rule(
        "betas-beta-or-fibre",
        ("betas", "beta", "fibre"),
        "The dispersion is given exactly once: as betas, as beta or by a fibre.",
        _GNLSE,
    ),
    Rule(
        "gamma-or-n2",
        ("gamma", "n2"),
        "The nonlinearity is given exactly once: as gamma or as n2.",
        _GNLSE,
    ),
    Rule(
        "n2-needs-fibre-area",
        ("n2", "fibre"),
        "n2 needs a fibre that carries an effective area.",
        _GNLSE,
    ),
    Rule(
        "fibre-carries-loss",
        ("fibre", "loss_db_per_m"),
        "A fibre carries its own loss, which is scaled on the mode.",
        _GNLSE,
    ),
    Rule(
        "gnlse-carrier-kerr",
        ("materials",),
        "The GNLSE route keeps a constant gamma, so a selection's dispersive "
        "electronic policy is refused.",
        _GNLSE,
    ),
    # The capillary route: the medium and its responses.
    Rule(
        "raman-needs-kerr",
        ("raman", "kerr"),
        "A medium with a Raman response has an electronic one too, and on a field "
        "grid the Raman drive reaches the kernel through the Kerr term.",
        _CAPILLARY,
    ),
    Rule(
        "raman-needs-molecule",
        ("raman", "gas"),
        "A Raman response asked for explicitly needs a molecule: an atom has no "
        "Raman-active mode.",
        _CAPILLARY,
        "a molecule optmat curates",
    ),
    Rule(
        "capillary-needs-nonlinearity",
        ("kerr", "plasma"),
        "A capillary run with neither kerr nor plasma has no nonlinearity.",
        _CAPILLARY,
    ),
    Rule(
        "envelope-plasma-no-thg",
        ("envelope", "plasma", "thg"),
        "Envelope plasma cannot be combined with the third harmonic.",
        _CAPILLARY,
    ),
    Rule(
        "plasma-integration-field-grid",
        ("plasma_integration", "envelope"),
        "plasma_integration chooses the field-grid plasma column's rule; envelope "
        "plasma takes none.",
        _CAPILLARY,
    ),
    Rule(
        "plasma-integration-needs-plasma",
        ("plasma_integration", "plasma"),
        "plasma_integration needs the plasma on.",
        _CAPILLARY,
    ),
    Rule(
        "mixture-carries-pressure",
        ("gas", "pressure", "pressure_floor", "vacuum_end"),
        "A Mixture carries its own pressure paths.",
        _CAPILLARY,
        "gas given as a Mixture",
    ),
    Rule(
        "mixture-carries-susceptibility",
        ("gas", "susceptibility"),
        "A Mixture, a fraction mapping and air carry their species' susceptibilities.",
        _CAPILLARY,
        "gas given as a Mixture, a fraction mapping or air",
    ),
    Rule(
        "scalar-capillary-polarisation",
        ("polarisation",),
        "The capillary route is scalar and propagates one transverse component "
        "per mode; the polarisation is declared in the modes.",
        _CAPILLARY,
    ),
    # The capillary route: the guide and its modes.
    Rule(
        "lattice-builds-modes",
        ("lattice", "modes"),
        "A tube lattice shapes the modes the route builds; given modes carry their "
        "own geometry.",
        _CAPILLARY,
    ),
    Rule(
        "lattice-full-model",
        ("lattice", "model"),
        "A tube lattice needs the full capillary model.",
        _CAPILLARY,
    ),
    Rule(
        "wall-keeps-taper-coupling",
        ("wall", "taper_coupling"),
        "A coherent wall block always carries the taper connection.",
        _CAPILLARY,
    ),
    Rule(
        "wall-cf4-numpy-ref",
        ("wall", "backend"),
        "A coherent wall with method='cf4' runs on numpy_ref alone.",
        _CAPILLARY,
    ),
    Rule(
        "wall-is-the-loss",
        ("wall", "loss"),
        "A coherent wall block is the loss model, so loss stays True beside it.",
        _CAPILLARY,
    ),
    Rule(
        "wall-full-model",
        ("wall", "model"),
        "A coherent wall block needs the full capillary model.",
        _CAPILLARY,
    ),
    Rule(
        "wall-no-lattice",
        ("wall", "lattice"),
        "A coherent wall block assumes one uniform wall, not a tube lattice.",
        _CAPILLARY,
    ),
    Rule(
        "mode-average-one-mode",
        ("mode_average", "modes"),
        "A mode-averaged run has exactly one mode.",
        _CAPILLARY,
    ),
    Rule(
        "mode-average-no-quadrature",
        ("mode_average", "n_r", "n_theta", "radial_rule"),
        "A mode-averaged run has no transverse rule to shape.",
        _CAPILLARY,
    ),
    # The capillary route's TDSE response.
    Rule(
        "tdse-helium",
        ("tdse", "gas"),
        "The TDSE response is helium's: the fill is optmat's helium alone, not "
        "a mixture and not preionised.",
        _CAPILLARY,
        "helium alone",
    ),
    Rule(
        "tdse-owns-electronic",
        (
            "tdse",
            "kerr",
            "mode_average",
            "plasma",
            "raman",
            "preionisation",
            "plasma_integration",
            "ionisation_field_range",
            "thg",
        ),
        "The TDSE response is the whole electronic response, so the keywords of "
        "the others are left unset beside it.",
        _CAPILLARY,
    ),
    Rule(
        "tdse-execution-needs-tdse",
        ("tdse_execution", "tdse"),
        "tdse_execution is the resources of a TDSE response, which tdse declares.",
        _CAPILLARY,
    ),
    Rule(
        "tdse-double-precision",
        ("tdse_execution", "precision"),
        "No backend runs the TDSE response in binary32.",
        _CAPILLARY,
    ),
    Rule(
        "tdse-numpy-ref",
        ("tdse_execution", "backend"),
        "The TDSE response runs on numpy_ref alone, named explicitly.",
        _CAPILLARY,
    ),
    Rule(
        "tdse-multistep",
        ("tdse_execution", "integrator"),
        "The TDSE response steps with the Multistep integrator.",
        _CAPILLARY,
    ),
)

RULES: Final[Mapping[str, Rule]] = MappingProxyType({one.id: one for one in _RULES})
"""Every combination rule the builders enforce, by id.

A test holds it to the source: every ``rule=`` given to an exception names a rule
here, and every rule here is given at least once.
"""


@dataclass(frozen=True)
class RouteCapabilities:
    """What a route takes: its keywords, their choices and its combination rules.

    Attributes
    ----------
    route : str
        The route.
    keywords : tuple of str
        Its build keywords, in the constructor's order
        (:func:`solprop.recipe.metadata.parameters`).
    choices : collections.abc.Mapping
        For each enumerated keyword, its allowed values, the ``"none"`` spelling
        included where the keyword takes it.
    rules : tuple of Rule
        The combination rules its builder enforces.
    """

    route: str
    keywords: tuple[str, ...]
    choices: Mapping[str, tuple[str, ...]]
    rules: tuple[Rule, ...]

    def rules_for(self, keyword: str) -> tuple[Rule, ...]:
        """Return the rules that concern a keyword.

        Parameters
        ----------
        keyword : str
            A build keyword of the route.

        Returns
        -------
        tuple of Rule
            The rules naming it, in :data:`RULES`' order.
        """
        return tuple(rule for rule in self.rules if keyword in rule.keywords)


def route(name: str) -> RouteCapabilities:
    """Say what a route takes (SEP 41, S8).

    Parameters
    ----------
    name : str
        A route with parameter metadata: ``"gnlse"`` or ``"capillary"``.

    Returns
    -------
    RouteCapabilities
        Its keywords, the values of each enumerated one, and its combination rules.

    Raises
    ------
    solprop.exceptions.DriverError
        If the route has no parameter metadata, naming the routes that do.

    Examples
    --------
    >>> from solprop.capabilities import route
    >>> capillary = route("capillary")
    >>> capillary.choices["shape"], capillary.choices["noise"]
    (('gauss', 'sech'), ('default', 'input', 'modified', 'none'))
    >>> [rule.id for rule in capillary.rules_for("tdse_execution")][:2]
    ['tdse-execution-needs-tdse', 'tdse-double-precision']
    """
    # The recipe layer sits above this module, which the driver imports.
    from solprop.recipe.core import ROUTES  # noqa: PLC0415 - layer above
    from solprop.recipe.metadata import parameters  # noqa: PLC0415 - layer above

    if name not in ROUTES:
        msg = (
            f"no capabilities for route {name!r}: the routes with parameter metadata "
            f"are {', '.join(ROUTES)}"
        )
        raise DriverError(msg, parameter="route")
    table = parameters(name)
    choices: dict[str, tuple[str, ...]] = {}
    for keyword, entry in table.items():
        values = tuple(
            value
            for form in entry.forms
            if form.kind == "choice"
            for value in form.choices
        )
        # A keyword whose only choice is the "none" spelling is not enumerated.
        if set(values) - {"none"}:
            choices[keyword] = values
    return RouteCapabilities(
        name,
        tuple(table),
        MappingProxyType(choices),
        tuple(rule for rule in _RULES if name in rule.routes),
    )


# ------------------------------------------------------------------ species (S8)


_RATE_WAVELENGTH: Final = 800e-9
"""The carrier, m, at which a rate model is built to read its status.

The status belongs to the model, not to the carrier; optmat lists its models at
800 nm too.
"""


@dataclass(frozen=True)
class IonisationModel:
    """One ionisation-rate model optmat has for a species.

    Attributes
    ----------
    name : str
        The name a selection gives it (``MaterialSelection(ionisation=name)``):
        ``"recommended"`` for optmat's default, or a literature form.
    status : str
        optmat's status of the model, one of
        :data:`solprop.physdata.selection.STATUSES`.
    reason : str
        optmat's reason for the status, where it gives one.
    """

    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class SpeciesCapabilities:
    """What a built-in gas brings to a run, with optmat's status of each part.

    Attributes
    ----------
    name : str
        The gas.
    molecule : bool
        Whether it is molecular rather than a single atom.
    ionisation : tuple of IonisationModel
        optmat's rate models for it, the recommended one first; empty for a gas
        optmat has no rate for, whose plasma needs a rate of the caller's own.
    raman : str
        Its Raman response's status
        (:func:`solprop.nonlinear.optmat_raman.raman_status`): ``"atom"`` (no
        Raman-active mode), ``"validated"``, ``"provisional"`` (taken only under
        ``MaterialSelection(raman="provisional")``) or ``"none"`` (a molecule
        optmat has no response of).
    raman_by_default : bool
        Whether ``raman=None`` turns the response on for a fill of it.
    wavelength_range : tuple of float
        The vacuum wavelengths in m over which its linear model is valid; beyond
        them a run warns (``MaterialRangeWarning``).
    temperature_range : tuple of float
        The temperatures in K over which its equation of state is valid.
    records : tuple of solprop.physdata.selection.MaterialRecord
        The records of the optmat models a fill of it uses (linear, equation of
        state, transport and Kerr), each with its status.
    """

    name: str
    molecule: bool
    ionisation: tuple[IonisationModel, ...]
    raman: str
    raman_by_default: bool
    wavelength_range: tuple[float, float]
    temperature_range: tuple[float, float]
    records: tuple[MaterialRecord, ...]


def _ionisation_models(gas: Gas) -> tuple[IonisationModel, ...]:
    """Return optmat's ionisation-rate models of a gas with their statuses.

    The models are the ones :func:`solprop.nonlinear.optmat_rates.selected_rate`
    resolves a selection's name from, each built once to read its record.

    Parameters
    ----------
    gas : solprop.physdata.Gas
        The gas.

    Returns
    -------
    tuple of IonisationModel
        The recommended model first, then the literature forms in optmat's order;
        empty for a gas neither of optmat's rate families covers.
    """
    from optmat.gas.ionisation_rates import (  # noqa: PLC0415 - optmat is lazy
        ionisation_rate_coverage,
        ionisation_rate_model,
    )
    from optmat.gas.molecular_ionisation_rates import (  # noqa: PLC0415 - as above
        molecular_rate_coverage,
        molecular_rate_model,
    )

    for coverage, build in (
        (ionisation_rate_coverage(), ionisation_rate_model),
        (molecular_rate_coverage(), molecular_rate_model),
    ):
        names = [
            name
            for name, gases in coverage.items()
            if any(one.value == gas.value for one in gases)
        ]
        if "default" not in names:
            continue
        ordered = ["default", *(name for name in names if name != "default")]
        models = []
        for name in ordered:
            record = MaterialRecord.from_identity(
                build(gas.value, _RATE_WAVELENGTH, name).identity
            )
            models.append(
                IonisationModel(
                    "recommended" if name == "default" else name,
                    record.status,
                    record.status_reason,
                )
            )
        return tuple(models)
    return ()


@functools.cache
def _species(gas: Gas) -> SpeciesCapabilities:
    """Return a built-in gas's capabilities, once per gas.

    Parameters
    ----------
    gas : solprop.physdata.Gas
        The gas, not air.

    Returns
    -------
    SpeciesCapabilities
        The statement.
    """
    from solprop.nonlinear.optmat_raman import raman_status  # noqa: PLC0415 - layer
    from solprop.physdata.species import OptmatGas  # noqa: PLC0415 - as above
    from solprop.physdata.species import species as built_in  # noqa: PLC0415

    material = built_in(gas)
    state = material.equation_of_state
    # A built-in gas's state is always optmat's; the check narrows the protocol.
    assert isinstance(state, OptmatGas)
    status = raman_status(gas.value)
    low, high = material.susceptibility.wavelength_range
    cold, hot = state.temperature_range
    return SpeciesCapabilities(
        gas.value,
        is_molecule(gas),
        _ionisation_models(gas),
        status,
        status == "validated",
        (float(low), float(high)),
        (float(cold), float(hot)),
        material.material_records,
    )


def species(name: str | Gas) -> SpeciesCapabilities:
    """Say what a built-in gas brings to a run (SEP 41, S8).

    Parameters
    ----------
    name : str or solprop.physdata.Gas
        A built-in gas's exact name, as the ``gas`` keyword takes it (``"N2"``).

    Returns
    -------
    SpeciesCapabilities
        Its ionisation models and Raman response with optmat's status of each, and
        the validity ranges optmat records for it.

    Raises
    ------
    solprop.exceptions.MaterialError
        If the name is not a built-in gas.
    solprop.exceptions.DriverError
        For air, which a capillary fills as nitrogen and oxygen: ask of each.

    Examples
    --------
    >>> from solprop.capabilities import species
    >>> nitrogen = species("N2")
    >>> nitrogen.raman, nitrogen.raman_by_default, nitrogen.ionisation[0].name
    ('validated', True, 'recommended')
    >>> species("He").raman, species("N2O").raman
    ('atom', 'provisional')
    """
    from solprop.driver.mixtures import is_air  # noqa: PLC0415 - layer above

    if is_air(name):
        msg = (
            'the capillary\'s "air" is a mixture of nitrogen and oxygen, each '
            "species with its own models: ask species() of N2 and O2"
        )
        raise DriverError(msg, parameter="gas")
    return _species(as_gas(name))
