# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The error a recipe raises when it cannot be read, checked or written."""

import functools

from solprop.exceptions import SolpropError

__all__ = ["RecipeError"]


class RecipeError(SolpropError, ValueError):
    """A recipe, or a value in it, cannot be read, checked or written.

    Parameters
    ----------
    message : str
        What is wrong, and where it can be, what to write instead.
    path : str or None, optional
        The canonical parameter path of the value refused (``"pressure"``,
        ``"pulses[1].energy"``), or ``None`` when the error concerns the document
        rather than one value. When given, the message starts with it.

    Attributes
    ----------
    message : str
        The message without the path.
    path : str or None
        The parameter path given, or ``None``; the same as ``parameter``, which
        every :class:`~solprop.exceptions.SolpropError` carries (SEP 41, S10).

    Examples
    --------
    >>> error = RecipeError("a pressure cannot be negative", path="pressure")
    >>> error.path, error.parameter
    ('pressure', 'pressure')
    >>> str(error)
    'pressure: a pressure cannot be negative'
    """

    message: str
    path: str | None

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(f"{path}: {message}" if path else message, parameter=path)
        self.message = message
        self.path = path

    def __reduce__(
        self,
    ) -> "tuple[functools.partial[RecipeError], tuple[str]]":  # noqa: UP037
        """Pickle the error with its path.

        The path is keyword-only, which the default exception pickling (the class
        called with ``args``) would drop, and a scan worker's refusal crosses a
        process.

        Returns
        -------
        tuple
            The class with the path bound, and the message.
        """
        return functools.partial(type(self), path=self.path), (self.message,)
