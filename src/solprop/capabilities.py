# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What solprop can give: here, what a results file can give (SEP 41, S16).

A front end that opens a results file asks :func:`output` what it may offer: the
file's basis, and for each derived quantity whether it is a **physical flux** or a
**coordinate quantity**; which planes keep their field; which operations a reader
admits and which it refuses, with the refusal the reader itself gives; and whether
the file records a recipe and can continue. The refusals come from the same gates the
readers raise from (:func:`solprop.output.capability.refusal_for` and the
compressors' basis check), so the statement and the refusal cannot drift.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from solprop.exceptions import ProcessingError
from solprop.output.capability import (
    BasisCapability,
    capability_of,
    refusal_for,
    withheld_from,
)
from solprop.output.hdf5 import open_file
from solprop.output.resumable import Resumability, resumability
from solprop.output.schema import Output

__all__ = ["OPERATIONS", "OutputCapabilities", "Quantity", "QuantityKind", "output"]

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
