# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical E and H on a saved plane of a general modal run, and what they measure.

A general modal guide propagates channel amplitudes, and its file stores those: the
mode axis is a coordinate axis and not a physical field. This module is the way back.
Given the run's own reference tables -- the ones
:func:`solprop.translate.general.general_modal_route` built, rebuilt from the guide the
file names -- it reconstructs ``E(x, y, t)`` and ``H(x, y, t)`` in V/m and A/m at the
guide's quadrature nodes, and forms the quantities a detector would see: the
cycle-averaged :func:`reconstructed_flux`, the real :func:`instantaneous_flux` with its
sum-frequency term, and the per-coordinate :func:`coordinate_intensity`, which is
**not additive**.

**What a file carries and what it does not.** The run record holds the guide's
*identity* -- the channel labels, the provider, the sampling, the table and metric
identities -- and not the reference arrays themselves, which are tens of megabytes per
run and are rebuilt from the guide in **tens of milliseconds** (28 ms on the slice's
one-channel 288-node fixture; about 0.71 s if the guide has still to be solved, which
is where that cost is). Physical profile storage for output is separate from the
execution tables and may be a hash-identified sidecar.
Every function here therefore takes the tables beside the handler and **checks them
against the stored identity**, path by path, refusing the first entry that disagrees,
so a reconstruction is never made from the wrong guide. ``guide=`` is **required**
beside them, and its sampling and its regions are checked too, which the tables'
identity cannot see: a guide re-sampled at other interior knots, or declaring another
``chi3``, builds tables whose identity agrees entry for entry. Every
caller has the guide, because that is what the tables were rebuilt from, so there is no
reading of this module in which the narrower check is the one wanted; the keyword has
no default and no alias.

**A guide whose geometry moves along z** is reconstructed at each plane's own
position: the guide is solved again there on the rule that has moved with its
interfaces, so the fields come back on that position's nodes and the flux is weighted
by that position's areas.
:func:`channel_amplitudes` recovers amplitudes on the entrance reference only.

**The accessor whitelist.** These names are admitted on a general modal basis and
refused on every other, because an orthogonal modal run and a spatial run carry
neither the certified metric nor the E/H reference they promise
(:mod:`solprop.output.capability`). The refusal is the same text the run loop uses.

The manual page is ``docs/guide/results/output.md``.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop import fourier, stats
from solprop.exceptions import ModeError, OutputError, TranslationError
from solprop.modes.guide import ModalGuide
from solprop.modes.protocol import ChannelLabel
from solprop.modes.vector import ModalNodeRule
from solprop.output.capability import capability_of, refusal_for, withheld_from
from solprop.output.schema import Convention, Output, RunMetadata
from solprop.processing.axes import _plane_field, grid_of
from solprop.translate.chebyshev import _EDGE_SLACK_FRACTION
from solprop.translate.general import guide_identity
from solprop.translate.modal_path import path_rule, path_snapshot
from solprop.translate.modal_tables import ModalTransformTables

__all__ = [
    "channel_amplitudes",
    "channel_labels",
    "coordinate_intensity",
    "electric_field",
    "instantaneous_flux",
    "magnetic_field",
    "physical_energy",
    "plane_nodes",
    "reconstructed_flux",
]

logger = logging.getLogger(__name__)


def _require_general(output: Output, name: str) -> RunMetadata:
    """Return the record of a general modal run, refusing any other basis by name.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    name : str
        The accessor being asked for, which the refusal names.

    Returns
    -------
    solprop.output.schema.RunMetadata
        The record, whose ``modal`` identity is not ``None``.

    Raises
    ------
    solprop.exceptions.OutputError
        If the handler holds no record, or its basis does not admit `name`.
    """
    metadata = output.metadata
    if metadata is None:
        msg = f"{name} needs an opened output holding its run record"
        raise OutputError(msg)
    refusal = refusal_for(
        capability_of(metadata), [name], withheld=withheld_from(metadata)
    )
    if refusal is not None:
        raise OutputError(refusal)
    return metadata


def _in_convention(
    signal: Complex[np.ndarray, "*shape"], convention: Convention
) -> Complex[np.ndarray, "*shape"]:
    """Return an analytic signal in the sign convention the caller asked for.

    The reference is solved, and the state is stored, in the internal FFT
    convention; every other accessor of :mod:`solprop.processing` hands back the
    optics convention by default, and this one does too, through the one sanctioned
    boundary (:func:`solprop.fourier.spectrum_to_optics`).

    Parameters
    ----------
    signal : numpy.ndarray
        The analytic signal in the internal convention.
    convention : {'optics', 'fft'}
        What the caller asked for.

    Returns
    -------
    numpy.ndarray
        The same signal in that convention.
    """
    return fourier.to_optics(signal) if convention == "optics" else signal


def _identity_leaves(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Yield the dotted path and value of every leaf of one JSON identity.

    An identity is a tree of mappings, lists and scalars, and what a refusal has to
    name is the **path** at which two of them part company -- ``model_id``,
    ``sampling.omega.6``, ``regions.0.kerr.chi3`` -- not the top-level block that
    contains it. Flattening both sides to paths is what makes that possible, and
    also what makes a stored block that is a subset of the expected one visible as
    the missing paths it is.

    Parameters
    ----------
    value : object
        A mapping, a sequence or a scalar, as
        :func:`solprop.translate.general.modal_identity` writes them and JSON
        reads them back.
    prefix : str, optional
        The path of `value` itself; empty at the root.

    Yields
    ------
    tuple
        The dotted path and the leaf value at it.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _identity_leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            head = f"{prefix}.{index}" if prefix else str(index)
            yield from _identity_leaves(item, head)
    else:
        yield prefix, value


_IDENTITY_SUBJECT: Mapping[str, str] = {
    "tables": "these tables are not the ones that produced this file",
    "guide": "this guide is not the one that produced this file",
}
"""The opening clause a refusal uses for each identity block."""


def _identity_refusal(
    block: str, expected: Mapping[str, object], stored: object
) -> str | None:
    """Return why a stored identity block is not this one's, or ``None``.

    The walk is over the **expected** identity's own paths, so a stored block that
    carries fewer of them is refused as incomplete rather than silently agreeing
    on the few it has: the file must say everything the rebuilt reference says, or
    what was not compared was not checked.

    Parameters
    ----------
    block : {'tables', 'guide'}
        Which entry of ``metadata.modal`` is being checked, for the message.
    expected : collections.abc.Mapping
        The identity of what the caller supplied, rebuilt now.
    stored : object
        The identity the file recorded, read back from JSON.

    Returns
    -------
    str or None
        The refusal, or ``None`` when the two agree path for path.
    """
    root = f"modal.{block}"
    if not isinstance(stored, Mapping):
        return (
            f"this file records no {root} identity to check against: it carries "
            f"{stored!r}. A record written by "
            "solprop.translate.general.modal_identity carries one"
        )
    live = {f"{root}.{path}": value for path, value in _identity_leaves(expected)}
    saved = {f"{root}.{path}": value for path, value in _identity_leaves(stored)}
    missing = [path for path in live if path not in saved]
    extra = [path for path in saved if path not in live]
    if missing or extra:
        side = (
            f"carries no {missing[0]}"
            if missing
            else f"carries {extra[0]}, which this reference has no entry for"
        )
        return (
            f"this file's {root} identity does not have the same entries as the one "
            f"offered here: it {side}. An identity that is a subset of the other "
            "cannot be checked -- what was not compared was not checked -- so the "
            "first entry that differs in shape is named instead of a value. Rebuild "
            "the reference from the guide this file names"
        )
    for path, value in live.items():
        if saved[path] != value:
            return (
                f"{_IDENTITY_SUBJECT[block]}: their {path} is {value!r} and the file "
                f"records {saved[path]!r}. Rebuild the reference from the guide the "
                "file names in metadata.modal['guide'], with the same sampling"
            )
    return None


def _checked_tables(
    metadata: RunMetadata,
    tables: ModalTransformTables,
    guide: ModalGuide,
) -> ModalTransformTables:
    """Return the tables, having checked they are the ones that produced the file.

    Each identity is compared **path by path**, so the refusal names the entry that
    disagrees -- ``model_id`` for a different guide, ``n_nodes`` for a different node
    rule, ``omega_span`` for a different run band -- rather than saying only that the
    two are not the same.

    A table identity cannot see everything a reconstruction depends on. It records
    the run's own coarse and fine extents, the node count and the three declared
    IDs, and **not** the frequencies the provider was solved at or what each region
    declared: a guide re-sampled at different interior knots, or with a different
    ``chi3``, builds tables whose identity is the same to the entry. So the caller
    hands over the guide the tables were rebuilt from -- which it has, since that is
    what :func:`solprop.translate.general.general_modal_route` takes -- and
    ``metadata.modal['guide']`` is checked too, which closes the gap. **Both
    identities are always checked**: the guide is required and the gate has no
    narrower mode.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The record of a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The reference the caller rebuilt from the guide.
    guide : solprop.modes.guide.ModalGuide
        The guide it was rebuilt from. It is **required**: a caller that rebuilt
        the tables has the guide, because that is what they were rebuilt from, and
        a check that closes the hole only when it is asked for is not a gate.

    Returns
    -------
    solprop.translate.modal_tables.ModalTransformTables
        The same tables.

    Raises
    ------
    solprop.exceptions.OutputError
        Naming the first identity path that disagrees, in the tables or in the
        guide.
    """
    modal = metadata.modal
    assert modal is not None
    refusal = _identity_refusal("tables", tables.identity, modal["tables"])
    if refusal is None:
        try:
            expected = guide_identity(guide)
        except TranslationError as absent:
            # A guide built by `from_step_index` declares no sampling and so has no
            # identity at all. That is the translation layer's refusal, and it is
            # re-raised in this module's own class because an accessor documents one
            # exception and a caller should not have to catch two to ask one question.
            msg = f"this guide has no identity to check against the file's: {absent}"
            raise OutputError(msg) from absent
        refusal = _identity_refusal("guide", expected, modal["guide"])
    if refusal is not None:
        raise OutputError(refusal)
    return tables


def _amplitudes(
    output: Output, plane: int | None, z: float | None
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return one plane's channel amplitudes in the internal convention.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Plane index, negative from the end.
    z : float or None
        Plane position, m.

    Returns
    -------
    numpy.ndarray
        The amplitudes, shape ``(n_modes, n_pol, n_omega)``, square root of watts.

    Raises
    ------
    solprop.exceptions.OutputError
        If neither a plane index nor a position selects exactly one plane.
    """
    if plane is None and z is None:
        msg = (
            "a reconstruction is made on one saved plane: give plane= or z=. A "
            "whole run of physical fields is n_z times the size of the run itself"
        )
        raise OutputError(msg)
    return _plane_field(output, plane, z, convention="fft")


def _plane_position(output: Output, plane: int | None, z: float | None) -> float:
    """Return the position of the one saved plane a selection names, in m.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Plane index, negative from the end.
    z : float or None
        Plane position, m; the nearest saved plane is the one meant, as
        :func:`_amplitudes` reads it.

    Returns
    -------
    float
        The stored position of that plane, m.
    """
    positions = output["z"]
    if z is not None:
        return float(positions[int(np.argmin(np.abs(positions - z)))])
    assert plane is not None
    return float(positions[plane])


@dataclass(frozen=True, eq=False)
class _PlaneReference:
    """The E/H reference and the area weights a plane is reconstructed on.

    Attributes
    ----------
    electric, magnetic : numpy.ndarray
        Reference E and H in V/m and A/m per sqrt(W), shape
        ``(n_nodes, 3, n_modes, n_omega)``, on the run's coarse grid.
    weights : numpy.ndarray
        The area weights of the nodes the reference lives on, m^2, shape
        ``(n_nodes,)``.
    """

    electric: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"]
    magnetic: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"]
    weights: Float[np.ndarray, " n_nodes"]


def _moves(tables: ModalTransformTables) -> bool:
    """Return whether a reference's maps are expanded along z (a moving guide).

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's reference.

    Returns
    -------
    bool
        True when the synthesis has more than one interval or coefficient in z.
    """
    synthesis = tables.synthesis
    return synthesis is not None and synthesis.shape[0] * synthesis.shape[1] > 1


def _moved_rule(guide: ModalGuide, position: float) -> tuple[float, ModalNodeRule]:
    """Return a moving guide's rule at a plane, and the position it was built at.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        The moving guide the reference was rebuilt from.
    position : float
        The plane's own position, m.

    Returns
    -------
    tuple
        The position, snapped onto the path's end when within the edge slack, and
        the provider's own rule there
        (:func:`solprop.translate.modal_path.path_rule`).

    Raises
    ------
    solprop.exceptions.OutputError
        If the guide records no path.
    solprop.exceptions.ModeError
        If the provider refuses the position; the caller says which plane.
    """
    sampling = guide.sampling
    if sampling is None or sampling.path is None:
        msg = (
            "these tables move along z and this guide records no path to solve the "
            "plane's own modes along: rebuild the reference from the moving guide the "
            "file names"
        )
        raise OutputError(msg)
    # A plane the run landed on by summing its steps sits within a few ulp of the
    # path's end rather than on it; within the edge slack a schedule is matched to
    # its kinks with, it is the end.
    edges = sampling.path.z_edges
    slack = _EDGE_SLACK_FRACTION * float(edges[-1] - edges[0])
    if edges[0] - slack <= position <= edges[-1] + slack:
        position = min(max(position, float(edges[0])), float(edges[-1]))
    return position, path_rule(guide.field_provider, sampling.nodes, position)


def _plane_rule(
    tables: ModalTransformTables, guide: ModalGuide, position: float
) -> tuple[Float[np.ndarray, "n_nodes 2"], Float[np.ndarray, " n_nodes"]]:
    """Return the nodes and area weights a plane at `position` is reconstructed on.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's reference, already checked against the file.
    guide : solprop.modes.guide.ModalGuide
        The guide it was rebuilt from.
    position : float
        The plane's own position, m.

    Returns
    -------
    tuple of numpy.ndarray
        Node positions, shape ``(n_nodes, 2)`` in m, and weights in m^2.

    Raises
    ------
    solprop.exceptions.OutputError
        If a moving guide records no path, or its provider refuses the position.
    """
    if not _moves(tables):
        return tables.node_xy[0], tables.weights[0]
    try:
        _, rule = _moved_rule(guide, position)
    except ModeError as refused:
        msg = f"the plane at z = {position:.6g} m cannot be solved again: {refused}"
        raise OutputError(msg) from refused
    return rule.xy, rule.weights


def _plane_reference(
    tables: ModalTransformTables, guide: ModalGuide, position: float
) -> _PlaneReference:
    """Return the reference a plane at `position` is reconstructed on.

    A guide whose geometry does not move has one reference, the tables' own. A
    moving guide's saved amplitudes are coefficients of **that plane's** modes, on
    the rule that has moved with the interfaces to that plane (node ``q`` is the point
    ``x_q(z)``), so the guide is solved again there exactly as
    its path was sampled -- the provider's own rule asked for as the entrance rule was
    (:func:`solprop.translate.modal_path.path_rule`), in the provider's own gauge
    (:func:`solprop.translate.modal_path.path_snapshot`), then carried to the run's
    bins by the same cubic in frequency the maps were built with. It is the same
    field the run's Chebyshev-expanded synthesis stores at that position, to the
    path's held-out accuracy, and H comes from the same solve; the expansion carries
    no H at all, which is why the solve is repeated rather than read.

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's reference, already checked against the file.
    guide : solprop.modes.guide.ModalGuide
        The guide it was rebuilt from.
    position : float
        The plane's own position, m.

    Returns
    -------
    _PlaneReference
        The reference and weights at that plane.

    Raises
    ------
    solprop.exceptions.OutputError
        If the guide records no path to solve along, or the provider refuses the
        position.
    """
    if not _moves(tables):
        return _PlaneReference(
            tables.electric[0], tables.magnetic[0], tables.weights[0]
        )
    sampling = guide.sampling
    try:
        position, rule = _moved_rule(guide, position)
        assert sampling is not None
        snapshot = path_snapshot(
            guide.field_provider,
            sampling.omega,
            position,
            rule,
            degenerate_blocks=sampling.degenerate_blocks,
            basis=sampling.basis,
        )
    except ModeError as refused:
        msg = f"the plane at z = {position:.6g} m cannot be solved again: {refused}"
        raise OutputError(msg) from refused
    shape = (rule.weights.size, 3, tables.electric.shape[3], tables.omega.size)
    electric = np.zeros(shape, np.complex128)
    magnetic = np.zeros(shape, np.complex128)
    # Outside the solved band the maps are explicit zeros, as the run's are.
    inside = (sampling.omega[0] <= tables.omega) & (tables.omega <= sampling.omega[-1])
    for column in np.flatnonzero(inside):
        sample = snapshot.evaluate(float(tables.omega[column]), position)
        electric[..., column] = sample.electric
        magnetic[..., column] = sample.magnetic
    logger.debug(
        "reconstruction reference solved again at z = %.6e m on %d nodes",
        position,
        rule.weights.size,
    )
    return _PlaneReference(electric, magnetic, rule.weights)


def _spectra(
    output: Output,
    tables: ModalTransformTables,
    plane: int | None,
    z: float | None,
    name: str,
    guide: ModalGuide,
) -> tuple[
    Complex[np.ndarray, "n_nodes 3 n_omega"],
    Complex[np.ndarray, "n_nodes 3 n_omega"],
    Float[np.ndarray, " n_nodes"],
]:
    """Return the nodal E and H spectra of one saved plane, in V/m and A/m.

    On a moving guide they are the plane's own modes at that plane's own nodes
    (:func:`_plane_reference`).

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference.
    plane : int or None
        Plane index, negative from the end.
    z : float or None
        Plane position, m.
    name : str
        The accessor being asked for, which a refusal names.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, whose identity is checked
        against the file's own.

    Returns
    -------
    tuple of numpy.ndarray
        The electric and magnetic spectra, each ``(n_nodes, 3, n_omega)``, and the
        area weights of the nodes they live on, m^2.

    Raises
    ------
    solprop.exceptions.OutputError
        If the basis, the tables, the guide or the plane selection is wrong, or
        the stored spectrum and the reference do not live on the same frequency
        axis.
    """
    metadata = _require_general(output, name)
    _checked_tables(metadata, tables, guide)
    field = _amplitudes(output, plane, z)
    grid = grid_of(output)
    if not np.array_equal(grid.omega, tables.omega):
        msg = (
            f"the reference is tabulated on {tables.omega.size} frequencies from "
            f"{tables.omega[0]:.6e} to {tables.omega[-1]:.6e} rad/s and the file's "
            f"grid carries {grid.omega.size} from {grid.omega[0]:.6e} to "
            f"{grid.omega[-1]:.6e}: rebuild the route on this run's own grid"
        )
        raise OutputError(msg)
    reference = _plane_reference(tables, guide, _plane_position(output, plane, z))
    # n_pol is one on this route, and the backend refuses anything else; summing
    # it keeps the contraction honest if that ever changes.
    amplitudes = np.sum(field, axis=1)
    electric = np.einsum("qcmw,mw->qcw", reference.electric, amplitudes)
    magnetic = np.einsum("qcmw,mw->qcw", reference.magnetic, amplitudes)
    return electric, magnetic, reference.weights


def channel_labels(output: Output) -> tuple[ChannelLabel, ...]:
    """Return the ordered channel labels a file records for its mode axis.

    The run record persists one :class:`solprop.modes.protocol.ChannelLabel` per
    row of the mode axis, so a plot legend, a selection by name and a resume check
    all read the same identity. A record whose mode records were built without
    labels names its channels by class and position only, and is refused.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.

    Returns
    -------
    tuple
        One label per mode-axis row, in state order.

    Raises
    ------
    solprop.exceptions.OutputError
        If the handler holds no record, or the record carries no labels.

    Examples
    --------
    >>> from solprop.processing._examples import example_general_output
    >>> from solprop.processing.modal import channel_labels
    >>> [label.display_label for label in channel_labels(example_general_output())]
    ['HE1,1:even']
    """
    metadata = output.metadata
    if metadata is None:
        msg = "channel_labels needs an opened output holding its run record"
        raise OutputError(msg)
    missing = [one for one in metadata.modes if one.label is None]
    if missing or not metadata.modes:
        msg = (
            "this run's record carries no channel labels: its mode records were "
            "built without them, and name a channel only by its class and its "
            "position on the mode axis. A run's own record (ModeRecord.from_mode) "
            "carries them"
        )
        raise OutputError(msg)
    return tuple(
        ChannelLabel.from_mapping(dict(one.label))
        for one in metadata.modes
        if one.label is not None
    )


def electric_field(  # noqa: PLR0913 - a plane, a representation, a convention, a guide
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    analytic: bool = False,
    convention: Convention = "optics",
    guide: ModalGuide,
) -> Inexact[np.ndarray, "n_nodes 3 n_t"]:
    """Return the electric field at the guide's nodes on one saved plane, in V/m.

    ``E_i(x_q, t) = sum_m e_i^m(x_q, omega) a_m(omega)`` transformed back to time,
    with ``e`` the stored reference in V/(m sqrt(W)) and ``a`` the saved amplitudes
    in the square root of watts. The three components are ``E_x``, ``E_y``, ``E_z``.
    On a guide whose geometry moves along z, ``e`` and the nodes are the guide's at
    the plane's own position, solved again there (the module's notes).

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference, rebuilt from the guide the record names; its
        identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    analytic : bool, optional
        Return the complex analytic signal instead of the real physical field. The
        real field is ``numpy.real`` of it, and the two flux accessors differ in
        exactly this choice.
    convention : {'optics', 'fft'}, optional
        Sign convention of the analytic signal, as every other accessor of
        :mod:`solprop.processing` takes it and with the same default: the optics
        convention of ``docs/dev/conventions.md`` section 2 unless ``'fft'`` asks
        for the internal one the reference is solved in. The **real** field is
        convention-free, so this is read only with `analytic`.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from. Its sampling and its regions are
        checked against the file's own guide identity, which the tables' identity
        cannot see. It is **required** and has no default.

    Returns
    -------
    numpy.ndarray
        The field, shape ``(n_nodes, 3, n_t)``, V/m; real unless `analytic`.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, or no single plane was selected.
    """
    spectrum, _, _ = _spectra(output, tables, plane, z, "electric_field", guide)
    signal = stats.analytic_field(grid_of(output), spectrum)
    return _in_convention(signal, convention) if analytic else np.real(signal)


def magnetic_field(  # noqa: PLR0913 - a plane, a representation, a convention, a guide
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    analytic: bool = False,
    convention: Convention = "optics",
    guide: ModalGuide,
) -> Inexact[np.ndarray, "n_nodes 3 n_t"]:
    """Return the magnetic field at the guide's nodes on one saved plane, in A/m.

    The partner of :func:`electric_field`, from the stored ``magnetic`` reference in
    A/(m sqrt(W)). The pair is what makes a flux: the provider normalises every
    channel to unit longitudinal Poynting flux, so ``int Re(E x conj(H))/2 . z dA``
    is one watt for a unit amplitude and the reconstruction carries SI units
    throughout.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference; its identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    analytic : bool, optional
        Return the complex analytic signal instead of the real physical field.
    convention : {'optics', 'fft'}, optional
        Sign convention of the analytic signal, as :func:`electric_field` takes it
        and with the same default.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, checked as in
        :func:`electric_field`. Required.

    Returns
    -------
    numpy.ndarray
        The field, shape ``(n_nodes, 3, n_t)``, A/m; real unless `analytic`.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, or no single plane was selected.
    """
    _, spectrum, _ = _spectra(output, tables, plane, z, "magnetic_field", guide)
    signal = stats.analytic_field(grid_of(output), spectrum)
    return _in_convention(signal, convention) if analytic else np.real(signal)


def _longitudinal(
    electric: Inexact[np.ndarray, "n_nodes 3 n_t"],
    magnetic: Inexact[np.ndarray, "n_nodes 3 n_t"],
    *,
    cycle_averaged: bool,
) -> Float[np.ndarray, "n_nodes n_t"]:
    """Return the longitudinal Poynting flux density of one E/H pair, in W/m^2.

    Parameters
    ----------
    electric : numpy.ndarray
        The electric field or its analytic signal, shape ``(n_nodes, 3, n_t)``.
    magnetic : numpy.ndarray
        The magnetic field or its analytic signal, same shape.
    cycle_averaged : bool
        Whether to take ``Re(E x conj(H))/2``, which drops the sum-frequency term,
        or the real product ``E x H``, which keeps it.

    Returns
    -------
    numpy.ndarray
        The flux density, shape ``(n_nodes, n_t)``, W/m^2. Negative values are
        interference and are never clipped.
    """
    if cycle_averaged:
        cross = electric[:, 0] * np.conj(magnetic[:, 1]) - electric[:, 1] * np.conj(
            magnetic[:, 0]
        )
        return 0.5 * np.real(cross)
    return np.real(electric[:, 0] * magnetic[:, 1] - electric[:, 1] * magnetic[:, 0])


def reconstructed_flux(
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    density: bool = False,
    guide: ModalGuide,
) -> Float[np.ndarray, "*shape"]:
    """Return the cycle-averaged Poynting flux of one saved plane, in watts.

    ``Re(E_a x conj(H_a))/2 . z`` integrated over the cross-section with the guide's
    own area weights, from the reconstructed analytic fields. This is the physical
    power through the plane at each instant, a trace in time, and it is **not** the
    trace the run statistic :func:`solprop.stats.analytic_flux` is the maximum of
    (hence the different name): that statistic is a scalar, the peak of the
    same-frequency diagonal, ``max_t sum_m |a_m(t)|^2``, while this carries
    the guide's cross-frequency flux kernel
    ``K_ij = sum_q w_q [e_x(w_i) conj(h_y(w_j)) - e_y(w_i) conj(h_x(w_j))]/2`` as
    well. The two peaks differ by at most ``sum_ij |B_i||B_j||K_ij - 1|``, measured
    at 1.3e-5 of the peak inside a bound of 3.4e-5 on the general-basis test
    fixture; their **time integrals** agree, because the cross-frequency
    terms are an integer number of cycles in the window. This one is the quantity a
    detector sees.

    **Negative values are kept.** A broadband field's Hermitian flux can go negative
    where two channels interfere; clipping it would make the trace look like a power
    and stop it integrating to the right energy.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference; its identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    density : bool, optional
        Return the flux **density** at each node, ``(n_nodes, n_t)`` in W/m^2,
        instead of its weighted sum over the cross-section.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, checked as in
        :func:`electric_field`. Required.

    Returns
    -------
    numpy.ndarray
        The flux, shape ``(n_t,)`` in W, or ``(n_nodes, n_t)`` in W/m^2 with
        `density`.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, or no single plane was selected.
    """
    electric, magnetic, weights = _spectra(
        output, tables, plane, z, "reconstructed_flux", guide
    )
    grid = grid_of(output)
    flux = _longitudinal(
        stats.analytic_field(grid, electric),
        stats.analytic_field(grid, magnetic),
        cycle_averaged=True,
    )
    return flux if density else np.einsum("q,qt->t", weights, flux)


def instantaneous_flux(
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    density: bool = False,
    guide: ModalGuide,
) -> Float[np.ndarray, "*shape"]:
    """Return the instantaneous Poynting flux of one saved plane, in watts.

    ``(E x H) . z`` of the **real** fields, integrated over the cross-section. It
    carries the sum-frequency term the cycle-averaged :func:`reconstructed_flux`
    drops -- the oscillation at twice the carrier -- and averages to it over a
    cycle. Which one a measurement wants is a statement about the detector: an
    instrument slower than the carrier sees the analytic flux, and a field-resolved
    measurement sees this one.

    **The sum-frequency oscillation is aliased when the grid does not resolve it.**
    The trace is formed from products of two real fields sampled on the run's own
    time axis, so the term at ``2 omega_0`` is only representable while
    ``2 omega_0`` is below the grid's Nyquist frequency. The **sample values are
    exact** either way -- each is the product of two exact samples -- and so is the
    integral identity below; what moves is the apparent frequency of the plotted
    oscillation, which folds to ``abs(2 omega_0 - 2 omega_Nyq)``. A field grid
    declared for a carrier near the top of its band will therefore show a slow
    ripple where the physical one is fast, and a plot that means to show the
    carrier beat needs a window sampled for it.

    ``int inst dt`` equals ``int analytic dt`` **when no sum-frequency content
    folds onto DC**: the cross terms are an integer number of cycles in the window
    and integrate away only while none of them aliases to zero frequency. That
    holds whenever the spectrum's support, doubled, stays clear of the sampling
    frequency's multiples, which a run whose carrier is inside its declared band
    satisfies.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference; its identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    density : bool, optional
        Return the flux **density** at each node, ``(n_nodes, n_t)`` in W/m^2,
        instead of its weighted sum over the cross-section.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, checked as in
        :func:`electric_field`. Required.

    Returns
    -------
    numpy.ndarray
        The flux, shape ``(n_t,)`` in W, or ``(n_nodes, n_t)`` in W/m^2 with
        `density`.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, or no single plane was selected.
    """
    electric, magnetic, weights = _spectra(
        output, tables, plane, z, "instantaneous_flux", guide
    )
    grid = grid_of(output)
    flux = _longitudinal(
        np.real(stats.analytic_field(grid, electric)),
        np.real(stats.analytic_field(grid, magnetic)),
        cycle_averaged=False,
    )
    return flux if density else np.einsum("q,qt->t", weights, flux)


def coordinate_intensity(
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    guide: ModalGuide,
) -> Float[np.ndarray, "n_modes n_nodes n_t"]:
    """Return the flux density each channel would carry alone, in W/m^2.

    The coordinate decomposition of the intensity: channel ``m``'s own analytic flux
    density ``Re(E^m x conj(H^m))/2 . z``, with ``E^m`` synthesised from that channel
    alone. It is what a per-mode intensity means on a modal basis.

    **It is not additive.** The sum over channels of these densities is the physical
    flux density only where the channels do not interfere; in general the difference
    is the cross term, which is exactly what :func:`reconstructed_flux` with ``density``
    carries and this does not. Even a positive spectral metric does not make an
    instantaneous cross-frequency flux a sum of coordinate powers. Nothing is clipped,
    here or there.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference; its identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, checked as in
        :func:`electric_field`. Required.

    Returns
    -------
    numpy.ndarray
        The per-channel flux density, shape ``(n_modes, n_nodes, n_t)``, W/m^2.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, or no single plane was selected.
    """
    metadata = _require_general(output, "coordinate_intensity")
    _checked_tables(metadata, tables, guide)
    amplitudes = np.sum(_amplitudes(output, plane, z), axis=1)
    reference = _plane_reference(tables, guide, _plane_position(output, plane, z))
    grid = grid_of(output)
    return np.stack(
        [
            _longitudinal(
                stats.analytic_field(grid, reference.electric[:, :, mode] * row),
                stats.analytic_field(grid, reference.magnetic[:, :, mode] * row),
                cycle_averaged=True,
            )
            for mode, row in enumerate(amplitudes)
        ]
    )


def plane_nodes(
    output: Output,
    tables: ModalTransformTables,
    *,
    plane: int | None = None,
    z: float | None = None,
    guide: ModalGuide,
) -> tuple[Float[np.ndarray, "n_nodes 2"], Float[np.ndarray, " n_nodes"]]:
    """Return where one saved plane is reconstructed: its nodes and area weights.

    The field accessors of this module return values at the guide's quadrature
    nodes and integrate them with its area weights; this returns those nodes and
    weights, in the same order as the node axis of :func:`electric_field`,
    :func:`reconstructed_flux` with ``density=True`` and :func:`coordinate_intensity`.
    For a guide whose geometry does not move they are the tables' own, the same at
    every plane. For a **moving** guide they are the plane's own: the rule that has
    moved with the interfaces to that plane's position, which is the provider's
    rule asked for there exactly as the entrance rule was
    (:func:`solprop.translate.modal_path.path_rule`) -- the same rule the field
    accessors solve on, so no caller needs the translation layer to plot a plane.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The run's own reference; its identity is checked against the file's.
    plane : int or None, optional
        Index of the plane wanted, negative from the end.
    z : float or None, optional
        Position of the plane wanted, m. Exactly one of `plane` and `z` is given.
    guide : solprop.modes.guide.ModalGuide
        The guide the reference was rebuilt from, checked as in
        :func:`electric_field`. Required.

    Returns
    -------
    tuple of numpy.ndarray
        The node positions ``(x, y)`` in the cross-section, shape ``(n_nodes, 2)``
        in m, and their area weights, shape ``(n_nodes,)`` in m^2, so that
        ``weights @ reconstructed_flux(..., density=True)`` is
        ``reconstructed_flux(...)``.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one, the tables or the guide are not the
        ones that produced the file, no single plane was selected, or a moving
        guide cannot be solved again at the plane's position.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_general_run
    >>> from solprop.processing.modal import plane_nodes, reconstructed_flux
    >>> run, tables, guide = example_general_run()
    >>> xy, weights = plane_nodes(run, tables, plane=-1, guide=guide)
    >>> xy.shape == (weights.size, 2)
    True
    >>> density = reconstructed_flux(run, tables, plane=-1, density=True, guide=guide)
    >>> flux = reconstructed_flux(run, tables, plane=-1, guide=guide)
    >>> bool(np.allclose(weights @ density, flux, rtol=1e-12, atol=0.0))
    True
    """
    metadata = _require_general(output, "plane_nodes")
    _checked_tables(metadata, tables, guide)
    # Read for its refusals alone: the same plane selection the field accessors make.
    _amplitudes(output, plane, z)
    return _plane_rule(tables, guide, _plane_position(output, plane, z))


def physical_energy(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
) -> Float[np.ndarray, "*shape"]:
    """Return the physical energy through a plane, or through every plane, in joules.

    :func:`solprop.stats.physical_energy`, which is the same-frequency product of the
    state with the basis's flux metric integrated over frequency, so a number read
    off a file and the number the run recorded agree to the bit. It is a scalar per
    plane because a coordinate energy is not additive on a basis that is not
    orthonormal. The run record holds the metric's **identity** and not its
    arrays, so this accessor evaluates the unweighted form and
    **refuses by name** a record whose ``modal.metric.kind`` is not ``"identity"``
    -- the number it could form from the stored state alone is the answer for a
    different basis. Read the run's own recorded ``physical_energy`` statistic, or
    :func:`reconstructed_flux` here, which reconstructs the physical flux from the
    guide's stored reference and needs no metric at all.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding a general modal run.
    plane : int or None, optional
        Index of the plane wanted, negative from the end; with neither `plane` nor
        `z`, every plane is returned with a leading ``n_z`` axis.
    z : float or None, optional
        Position of the plane wanted, m.

    Returns
    -------
    numpy.ndarray
        The energy, J, a scalar per plane.

    Raises
    ------
    solprop.exceptions.OutputError
        If the run is not a general modal one. If the recorded basis metric is not
        the identity, so that the weighted energy cannot be formed from the stored
        state.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_general_output
    >>> from solprop.processing.modal import physical_energy
    >>> values = physical_energy(example_general_output())
    >>> bool(np.allclose(values, values[0], rtol=1e-12))
    True
    """
    _require_identity_metric(_require_general(output, "physical_energy"))
    field = _plane_field(output, plane, z, convention="fft")
    grid = grid_of(output)
    if field.ndim == 3:
        return stats.physical_energy(grid, field)
    return np.stack([stats.physical_energy(grid, one) for one in field])


def _require_identity_metric(metadata: RunMetadata) -> None:
    """Refuse a metric-weighted accessor on a file whose metric is not in it.

    :data:`solprop.stats.METRIC_WEIGHTED_STATISTICS` are functionals of the physical
    same-frequency flux ``a.H W a``. The run record holds the **identity** of the
    basis metric -- its kind, its gauge and its measured defect -- and not the
    metric's arrays, exactly as it records the identity of the reference tables and
    not the tables, so a file-time accessor that reduces the stored
    state alone cannot evaluate one on a basis whose ``W`` is not the identity. It
    refuses by name rather than returning the unweighted number, which would be the
    answer for a different basis: the run's own recorded ``physical_energy``
    statistic carries the weighted value, and :func:`reconstructed_flux` here
    reconstructs the physical flux from the guide's stored reference, which needs no
    metric at all.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The record of a general modal run.

    Raises
    ------
    solprop.exceptions.OutputError
        If the recorded basis metric is not the identity.
    """
    modal = metadata.modal
    metric = modal.get("metric") if isinstance(modal, dict) else None
    kind = metric.get("kind") if isinstance(metric, dict) else None
    if kind in (None, "identity"):
        return
    gauge = metric.get("gauge_id") if isinstance(metric, dict) else None
    msg = (
        f"physical_energy is unavailable for this run: its basis metric is of kind "
        f"{kind!r} in the gauge {gauge!r}, and a metric-weighted energy is the "
        "same-frequency product of the state with that metric, whose arrays the "
        "run record does not store -- only its identity. Read the "
        "run's own recorded physical_energy statistic, or reconstructed_flux, which "
        "reconstructs the physical flux from the guide's stored reference and needs "
        "no metric at all"
    )
    raise OutputError(msg)


def channel_amplitudes(
    electric: Complex[np.ndarray, "n_nodes 3 n_omega"],
    magnetic: Complex[np.ndarray, "n_nodes 3 n_omega"],
    tables: ModalTransformTables,
) -> Complex[np.ndarray, "n_modes n_omega"]:
    """Recover channel amplitudes from a nodal E/H pair, in the square root of watts.

    The inverse of the synthesis :func:`electric_field` and :func:`magnetic_field`
    perform, through the coefficient dual of
    :class:`~solprop.translate.modal_tables.ModalTransformTables`:
    ``a_m = sum_q w_q [W^-1]_mn (E x conj(h_n) + conj(e_n) x H)_z / 4``, the same
    Hermitian pairing the flux metric is built from. It is amplitude recovery and
    **not** source projection, which carries the separate ``-i omega`` factor.

    Reprojecting a physical field that was not synthesised from this basis creates a
    new initial-value problem; it is not a resume path.

    Parameters
    ----------
    electric : numpy.ndarray
        Nodal electric spectrum, shape ``(n_nodes, 3, n_omega)``, V/m, internal
        convention.
    magnetic : numpy.ndarray
        Nodal magnetic spectrum, same shape, A/m.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The reference the fields were synthesised from, carrying its
        ``coordinate_dual``.

    Returns
    -------
    numpy.ndarray
        The amplitudes, shape ``(n_modes, n_omega)``, square root of watts.

    Raises
    ------
    solprop.exceptions.OutputError
        If the record carries no coefficient dual, or the fields do not match its
        node and frequency extents.
    """
    dual = tables.coordinate_dual
    if dual is None:
        msg = (
            "these tables carry no coordinate_dual, so an amplitude cannot be "
            "recovered from a physical field: build the reference with "
            "solprop.translate.modal_tables.modal_transform_tables, which fills it"
        )
        raise OutputError(msg)
    expected = (dual.shape[2], 3, dual.shape[4])
    if electric.shape != expected or magnetic.shape != expected:
        msg = (
            f"the nodal fields have shapes {electric.shape} and {magnetic.shape} "
            f"and this reference's dual expects {expected} (nodes, components, "
            "frequencies)"
        )
        raise OutputError(msg)
    stacked = np.concatenate([electric, magnetic], axis=1)
    return np.einsum("mqcw,qcw->mw", dual[0], stacked)
