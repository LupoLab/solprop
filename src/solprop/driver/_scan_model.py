# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Describe indexed parameter scans and dispatch their independent simulations.

Scan indices are zero-based and use C order, with the last declared variable
fastest. Selecting a subset never renumbers points. See
``docs/guide/workflows/scans.md`` for execution, seeds and storage.
"""

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from solprop._display import Displayed, line
from solprop.exceptions import DriverError

__all__ = ["Scan", "ScanPoint"]


@dataclass(frozen=True)
class ScanPoint:
    """Identify one combination independently of its executor.

    Attributes
    ----------
    index : int
        Zero-based index in the complete Cartesian product.
    values : dict
        Variable names and scalar values, in declaration order and user units.
    multi_index : tuple of int
        Zero-based coordinates in the complete scan shape.
    seed : int or None
        Explicit seed-axis value or derived non-negative integer noise seed.
    """

    index: int
    values: dict[str, object]
    multi_index: tuple[int, ...]
    seed: int | None = None


def name_seed(name: str, index: int) -> int:
    """Return the noise seed of a point whose scan declares none.

    A builder that leaves its noise unseeded on a scan without a ``seed`` axis or a
    ``base_seed`` would draw operating-system entropy, so the point could be neither
    reproduced nor rerun against its record. :func:`~solprop.driver.scan.run_scan`
    draws it from this seed instead: the root entropy is the first eight bytes of the
    SHA-256 digest of the scan's name, little-endian, and the point's is
    ``SeedSequence([root, global_index]).generate_state(1, dtype=uint64)[0]``, the
    derivation ``base_seed`` uses.

    Parameters
    ----------
    name : str
        The scan's name.
    index : int
        The point's global index.

    Returns
    -------
    int
        A non-negative 64-bit seed.

    Examples
    --------
    >>> name_seed("rdw", 3) == name_seed("rdw", 3) != name_seed("rdw", 4)
    True
    """
    root = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    sequence = np.random.SeedSequence([root, index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class Scan(Displayed):
    """Declare an ordered Cartesian product of one-dimensional variable arrays.

    Parameters
    ----------
    name : str
        Nonempty scan name; must not contain path separators.
    base_seed : int or None, optional
        Non-negative root entropy. An explicit ``seed`` variable takes precedence.
    **variables : array_like
        One-dimensional arrays in declaration order. Values use the builder's
        documented units; no unit conversions are performed by the scan.

    Attributes
    ----------
    name : str
        Name used in metadata and durable point filenames.
    base_seed : int or None
        Root entropy used for index-based seed derivation.

    Examples
    --------
    >>> scan = Scan("example", pressure=[1., 2.], energy=[1e-6, 2e-6])
    >>> scan.shape, scan.point(2).multi_index
    ((2, 2), (1, 0))
    >>> [point.index for point in scan.chunk(1, 2).points()]
    [1, 3]
    """

    def __init__(
        self, name: str, *, base_seed: int | None = None, **variables: ArrayLike
    ) -> None:
        if not name or name in (".", "..") or any(c in name for c in "/\\\x00"):
            raise DriverError(
                "scan name must be nonempty and contain no path separators"
            )
        if base_seed is not None and (
            not isinstance(base_seed, int)
            or isinstance(base_seed, bool)
            or base_seed < 0
        ):
            raise DriverError("base_seed must be a non-negative integer")
        self.name = name
        self.base_seed = base_seed
        self._variables: dict[str, np.ndarray] = {}
        self._indices: tuple[int, ...] | None = None
        for key, values in variables.items():
            self.add(key, values)

    def __repr__(self) -> str:
        """Return a one-line description: the name and the axes' lengths.

        Returns
        -------
        str
            For example ``Scan('rdw', pressure=5, energy=4)``.
        """
        axes = [(name, values.size) for name, values in self._variables.items()]
        return line("Scan", [("", repr(self.name)), *axes])

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        rows: list[tuple[str, object]] = [
            (name, f"{values.size} values, {values[0]:.6g} to {values[-1]:.6g}")
            for name, values in self._variables.items()
        ]
        rows.append(("base seed", self.base_seed))
        if self._indices is not None:
            rows.append(("subset", f"{len(self._indices)} of {self.size} points"))
        return f"Scan {self.name!r}: {self.size} points", rows

    @property
    def variables(self) -> Mapping[str, np.ndarray]:
        """Return copies of the named axes in declaration order.

        Returns
        -------
        collections.abc.Mapping
            One-dimensional arrays; changing them cannot alter this scan.
        """
        return {name: values.copy() for name, values in self._variables.items()}

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the complete Cartesian shape, including for a subset.

        Returns
        -------
        tuple of int
            Axis lengths in declaration order; empty before any variable is added.
        """
        return tuple(values.size for values in self._variables.values())

    @property
    def size(self) -> int:
        """Return the number of points in the complete scan.

        Returns
        -------
        int
            Non-negative Cartesian product size; zero for no variables.
        """
        return prod(self.shape) if self.shape else 0

    def __len__(self) -> int:
        """Return the number of selected points.

        Returns
        -------
        int
            Number of selected global indices.
        """
        return self.size if self._indices is None else len(self._indices)

    def add(self, name: str, values: ArrayLike) -> None:
        """Append an axis before selecting or running the scan.

        Parameters
        ----------
        name : str
            Unique nonempty variable name without path separators.
        values : array_like
            One-dimensional axis in the builder's units; may be empty.

        Raises
        ------
        DriverError
            If the name repeats, the values are not an axis or this is a subset.
        """
        if self._indices is not None:
            raise DriverError("cannot add variables to a selected scan")
        if not name or any(c in name for c in "/\\\x00") or name in self._variables:
            raise DriverError(f"invalid or repeated scan variable {name!r}")
        array = np.array(values, copy=True)
        if array.ndim != 1:
            raise DriverError(f"scan variable {name!r} must be one-dimensional")
        if name == "seed" and (array.dtype.kind not in "iu" or np.any(array < 0)):
            raise DriverError("seed values must be non-negative integers")
        array.setflags(write=False)
        self._variables[name] = array

    def point(self, index: int) -> ScanPoint:
        """Look up a point by its scan-wide index without building the product.

        Parameters
        ----------
        index : int
            Global index, from zero through ``size - 1``.

        Returns
        -------
        ScanPoint
            Coordinates, values and resolved noise seed.

        Raises
        ------
        DriverError
            If the index is outside the complete product.
        """
        if not 0 <= index < self.size:
            raise DriverError(f"point index {index} is outside scan size {self.size}")
        coordinates = tuple(int(i) for i in np.unravel_index(index, self.shape))
        values = {
            name: axis[i].item() if isinstance(axis[i], np.generic) else axis[i]
            for (name, axis), i in zip(
                self._variables.items(), coordinates, strict=True
            )
        }
        seed = None
        if "seed" in values:
            seed = int(self._variables["seed"][coordinates[list(values).index("seed")]])
        elif self.base_seed is not None:
            sequence = np.random.SeedSequence([self.base_seed, index])
            seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        return ScanPoint(index, values, coordinates, seed)

    def points(self) -> Iterator[ScanPoint]:
        """Iterate over the selected points in selection order.

        Yields
        ------
        ScanPoint
            A point retaining its global index.
        """
        for index in range(self.size) if self._indices is None else self._indices:
            yield self.point(index)

    def subset(self, indices: Sequence[int]) -> "Scan":  # noqa: UP037 - runtime hook evaluates the class body
        """Select unique global indices without changing the Cartesian axes.

        Parameters
        ----------
        indices : sequence of int
            Global indices, in requested order, within the current selection.

        Returns
        -------
        Scan
            Independent selection with the original shape and root seed.

        Raises
        ------
        DriverError
            If an index is repeated, out of range or outside this selection.
        """
        selected = tuple(indices)
        allowed = range(self.size) if self._indices is None else self._indices
        if len(set(selected)) != len(selected) or any(
            i not in allowed for i in selected
        ):
            raise DriverError("subset indices must be unique members of this scan")
        result = Scan(self.name, base_seed=self.base_seed)
        result._variables = dict(self._variables)
        result._indices = selected
        return result

    def chunk(self, k: int, n: int) -> "Scan":  # noqa: UP037 - runtime hook evaluates the class body
        """Select one round-robin chunk by the global index modulo its count.

        Parameters
        ----------
        k : int
            Zero-based chunk number, smaller than ``n``.
        n : int
            Positive number of chunks.

        Returns
        -------
        Scan
            Points whose global index modulo ``n`` equals ``k``.

        Raises
        ------
        DriverError
            If the chunk count or number is invalid.
        """
        if n < 1 or not 0 <= k < n:
            raise DriverError("chunk requires n > 0 and 0 <= k < n")
        return self.subset(
            [point.index for point in self.points() if point.index % n == k]
        )
