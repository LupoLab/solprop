# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a stored basis admits: the accessor whitelist of the run record.

A saved run is read through one protocol, and for every route but one a run's mode
axis means the same thing: an orthogonal one- or two-component basis whose
coordinate powers add up to the physical power. A **general modal guide** resolves
three Cartesian components at its own quadrature nodes, and its coordinate quantities
are not that: ``energy`` and ``peak_power`` are defined for it only because its metric
happens to be the identity, and ``peak_intensity`` is not defined for it at all. Rather
than redefine the old names on a basis they do not describe, the record names the
**capability** that produced the file and refuses, by name, every accessor that
capability does not admit.

This module is the **gate**; the sets it reads are in :mod:`solprop.stats`, beside the
catalogue of names they select from, so that a translation module can record the list a
file publishes without importing an output one.

The gate is applied at four doors: at :class:`solprop.stats.StatsSpec` admission, where
the driver knows the problem; in each CPU backend's ``stats``, for a caller who builds
one and asks it directly; in the accessors of :mod:`solprop.processing.modal`; and in
the legacy processing accessors of :mod:`solprop.processing.time`, which see only the
stored record. A modal run of any other route carries no general identity, and is read
as the ordinary modal run it is: the general names are refused for it, because the
certified metric and the stored E/H reference they promise are exactly what such a file
does not have.

The manual page is ``docs/guide/results/statistics.md``.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from solprop.output.schema import RunMetadata
from solprop.stats import (
    _GAS_OR_SPATIAL_NAMES,
    _STEP_DIAGNOSTICS,
    ACCESSOR_ADVICE,
    GENERAL_MODAL_ACCESSORS,
    WITHHELD_WITH_A_FLOW,
)

__all__ = [
    "ACCESSOR_ADVICE",
    "ACCESSOR_WHITELIST",
    "CAPABILITIES",
    "GENERAL_MODAL_ACCESSORS",
    "BasisCapability",
    "capability_of",
    "declared_capability",
    "refusal_for",
    "withheld_accessors",
    "withheld_from",
]

BasisCapability = Literal["modal", "general_modal", "spatial"]
"""Which kind of basis a run's mode axis is, and so which accessors it admits."""

CAPABILITIES: tuple[BasisCapability, ...] = ("modal", "general_modal", "spatial")
"""Every declared basis capability, in the order the manual lists them."""

_GENERAL_ONLY: frozenset[str] = frozenset(
    {
        "physical_energy",
        "spectral_flux",
        "analytic_flux",
        "reconstructed_flux",
        "coordinate_intensity",
        "instantaneous_flux",
        "electric_field",
        "magnetic_field",
        "plane_nodes",
        "channel_amplitudes",
    }
)
"""Accessors that need a certified metric and a stored E/H reference.

These ten are what an ordinary modal or spatial run is refused.
``channel_amplitudes`` is among them although it takes
two arrays and a table rather than an opened handler, and so is never asked of a
record: it is here so that the one direction a caller could reach -- inverting a nodal
E/H pair a run of another basis produced -- is refused for the right reason.
"""

ACCESSOR_WHITELIST: Mapping[BasisCapability, frozenset[str] | None] = {
    "modal": None,
    "general_modal": GENERAL_MODAL_ACCESSORS,
    "spatial": None,
}
"""Which accessors each capability admits.

``None`` means "every name this run has always had, and no general one".

An orthogonal modal run and a spatial run keep every accessor they have always had --
this table never narrows an existing route -- and refuse the ten names of
``_GENERAL_ONLY``, which need a certified flux metric and a stored E/H reference
they do not carry. A general modal run admits exactly its own list, plus the
basis-independent step diagnostics section 6 allows.
"""

_GENERIC_ADVICE = (
    "physical_energy, analytic_flux, or reconstructed E/H with an explicit detector "
    "definition"
)


def declared_capability(*, general_modal: bool, spatial: bool) -> BasisCapability:
    """Return the capability a translated problem declares.

    The problem-side companion of :func:`capability_of`, for the two places that hold
    a problem rather than a record: the run loop's statistics admission and the
    reference backend's own ``stats``.

    Parameters
    ----------
    general_modal : bool
        Whether ``transform.modal_tables`` is set, which is what makes a run a
        general modal one.
    spatial : bool
        Whether ``transform.geometry`` is set, which is what makes it a radial or
        Cartesian beam.

    Returns
    -------
    str
        One of :data:`CAPABILITIES`.

    Examples
    --------
    >>> from solprop.output.capability import declared_capability
    >>> declared_capability(general_modal=True, spatial=False)
    'general_modal'
    >>> declared_capability(general_modal=False, spatial=False)
    'modal'
    """
    if general_modal:
        return "general_modal"
    return "spatial" if spatial else "modal"


def capability_of(metadata: RunMetadata) -> BasisCapability:
    """Return the capability a stored record declares.

    A record whose modal identity declares the **general modal basis** is a general
    modal run; a radial or Cartesian run is spatial; everything else -- including
    a record that carries no modal identity at all, and
    a capillary carrying a coherent wall block, whose identity declares the same
    orthogonal ``HE_1m`` basis it always had -- is the ordinary modal basis. It is
    keyed on ``modal["basis"]`` rather than on the identity being present at all
    because a second route (a capillary with a coherent wall) carries an identity of
    its own, and a coupled flow through a basis does not change the basis; a general
    modal record declares ``"general_modal"`` there, so nothing already stored reads
    differently.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The record read back from a handler.

    Returns
    -------
    str
        One of :data:`CAPABILITIES`.

    Examples
    --------
    >>> from solprop.output.capability import capability_of
    >>> from solprop.output.schema import _example_metadata
    >>> capability_of(_example_metadata())
    'modal'
    """
    if metadata.modal is not None and metadata.modal.get("basis") == "general_modal":
        return "general_modal"
    return (
        "spatial" if metadata.simulation_kind in ("radial", "free", "slab") else "modal"
    )


def refusal_for(
    capability: BasisCapability,
    names: Sequence[str],
    *,
    withheld: Sequence[str] = (),
) -> str | None:
    """Return why the first inadmissible accessor is inadmissible, or ``None``.

    The one text both doors use, so that a name refused at admission and the same
    name refused in processing say the same thing. The message names the basis and
    the replacement, and never redefines the old name.

    **Each name is refused for its own reason.** A legacy power name is refused
    because a general basis's coordinate quantities are not additive; a gas or
    spatial name is refused because this guide has no gas fill and no transverse
    geometry, which is not a statement about the basis at all; and a
    basis-independent step diagnostic is admitted here whatever the basis, so that
    whoever owns its real reason is the one that refuses it.

    Parameters
    ----------
    capability : {'modal', 'general_modal', 'spatial'}
        What the run's basis is, from :func:`capability_of` or
        :func:`declared_capability`.
    names : collections.abc.Sequence
        The accessor or statistic names asked for, in the order they were asked.
    withheld : collections.abc.Sequence, optional
        Names the basis admits in general but **this run** does not publish: a
        declared basis of a split block executes through a coupled flow whose
        channels beat, so a per-channel propagation constant is the block's mean and
        ``zdw`` from it is a number that propagates nothing. The route withholds the
        name from its record and every door refuses it by name.

    Returns
    -------
    str or None
        The refusal, or ``None`` when every name is admitted.

    Examples
    --------
    >>> from solprop.output.capability import refusal_for
    >>> refusal_for("general_modal", ["peak_power"])[:34]
    'peak_power is unavailable for this'
    >>> refusal_for("modal", ["energy"]) is None
    True
    >>> refusal_for("general_modal", ["quadrature_error"]) is None
    True
    >>> refusal_for("general_modal", ["zdw"], withheld=("zdw",))[:30]
    'zdw is unavailable for this ru'
    """
    allowed = ACCESSOR_WHITELIST[capability]
    for name in names:
        if name in withheld:
            return (
                f"{name} is unavailable for this run: its declared change of basis "
                "couples channels whose propagation constants differ, so the block "
                "executes through a coupled flow and a per-channel propagation "
                "constant is the block's mean, from which the statistic would be a "
                "number that propagates nothing. Ask "
                "the plain basis for it"
            )
        if allowed is None:
            if name not in _GENERAL_ONLY:
                continue
            return (
                f"{name} is unavailable for this {capability} run: it needs a "
                "general modal guide's certified flux metric and the stored E/H "
                "reference of its run record, and this record carries neither, so "
                "the number would be a promise the file cannot keep. Use the "
                "existing energy, peak_power and peak_intensity accessors, which "
                "are what this basis does define"
            )
        if name in allowed or name in _STEP_DIAGNOSTICS:
            continue
        if name in _GAS_OR_SPATIAL_NAMES:
            return (
                f"{name} is unavailable for this general modal guide: it needs a "
                "gas fill or a spatial geometry this guide has none of. A guide's "
                "material regions declare their own responses at fixed density, "
                "and its channels are solved on a transverse quadrature rather "
                "than on a radial or Cartesian grid, so there is no pressure, no "
                "ionised species and no beam width to report. This is not a "
                "statement about the basis"
            )
        return (
            f"{name} is unavailable for this nonorthogonal or three-component "
            f"modal basis; use {ACCESSOR_ADVICE.get(name, _GENERIC_ADVICE)}. A "
            "general modal guide resolves E_x, E_y and E_z at its own quadrature "
            "nodes, and its coordinate quantities are not additive, so the old "
            "name is refused rather than redefined. "
            f"This basis admits {', '.join(sorted(allowed))}"
        )
    return None


def withheld_accessors(*, general_modal: bool, coupled_flow: bool) -> tuple[str, ...]:
    """Return the names a problem withholds beyond its basis's whitelist.

    A general modal problem carrying ``linear.coupled_flow`` executes a declared
    basis of a split block, whose per-channel propagation constant is the block's
    mean, so it withholds :data:`solprop.stats.WITHHELD_WITH_A_FLOW`; every other
    problem withholds nothing. The two execution doors pass this to
    :func:`refusal_for`, and the route leaves the same names out of the record's
    published list so that the file doors agree (:func:`withheld_from`).

    Parameters
    ----------
    general_modal : bool
        Whether ``transform.modal_tables`` is set.
    coupled_flow : bool
        Whether ``linear.coupled_flow`` is set.

    Returns
    -------
    tuple of str
        The withheld names, sorted; empty for every problem without a split block.

    Examples
    --------
    >>> from solprop.output.capability import withheld_accessors
    >>> withheld_accessors(general_modal=True, coupled_flow=True)
    ('zdw',)
    >>> withheld_accessors(general_modal=False, coupled_flow=True)
    ()
    """
    if not (general_modal and coupled_flow):
        return ()
    return tuple(sorted(WITHHELD_WITH_A_FLOW))


def withheld_from(metadata: RunMetadata) -> tuple[str, ...]:
    """Return the names a stored record withholds beyond its basis's whitelist.

    The file doors' half of :func:`withheld_accessors`: a general modal record whose
    published ``accessors`` list leaves out a name of
    :data:`solprop.stats.WITHHELD_WITH_A_FLOW` was written by a split-block route,
    and the name is refused from the file exactly as it was from the run. A record
    of any other basis, or one with no published list, withholds nothing.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The stored record.

    Returns
    -------
    tuple of str
        The withheld names, sorted.
    """
    if capability_of(metadata) != "general_modal" or metadata.modal is None:
        return ()
    published = metadata.modal.get("accessors")
    if not isinstance(published, list):
        return ()
    return tuple(sorted(name for name in WITHHELD_WITH_A_FLOW if name not in published))
