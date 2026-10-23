# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compile with numba's ``njit`` behind a disk cache that writers cannot cross-wire.

numba keeps one index file per function, which maps each key -- the argument types, the
target and a digest of the bytecode and the closure's cells -- to a numbered data file.
Its ``IndexDataCacheFile.save`` reads the index, takes the lowest unused number and
writes the index and the data back with no lock. Two processes saving different keys of
one function at the same moment (two closure variants, or two signatures) can both take
``.1.nbc``: the index ends up naming one key and the data file holding the other's
code. numba loads an entry without comparing its signature to the request, so every
later process asking for that key runs the other compilation. With incompatible
arguments that is ``TypeError: can't unbox array from PyObject into native value``;
with compatible ones it is the other variant's body, silently. Twelve variants of
``make_raman_finish`` compiled at once into an empty cache cross-wired entries in three
trials of four under numba 0.67.0: two raised that error, one ran other variants' bodies
silently. An ``xdist`` run, or a batch of jobs started together on a shared source tree,
is that situation.

Here there is no index. A data file is named by a digest of the key's signature, target
and closure cells, and begins with a header naming what it holds: the numba version, the
source file's stamp and the whole key. A load reads the header first and treats any
mismatch as a miss, so a file can only ever be used for the key that wrote it, and with
nothing to read, modify and write back, concurrent writers cannot lose each other's
entries either. The bytecode digest is left out of the name, so that editing a function
overwrites its files rather than adding to them, as numba's numbering does.

The header also carries a digest of what the compilation took from **outside** the
function, which numba's key and source stamp do not see: numba invalidates an entry only
when the defining file changes, yet it compiles the bodies a kernel inlines (the
``define_bodies`` family of ``backends/bodies.py``) and freezes the global constants it
reads (``EPS0``, ``PHI_JOIN``) into the machine code. An edit to ``bodies.py`` alone
left every kernel serving the old body. The digest follows the names the function's
bytecode references, and its closure cells: a dispatcher contributes its function's
bytecode, constants and names, and is followed in turn; a number, string, NumPy scalar
or array, or a tuple of them, contributes its value; a dispatcher's compiler options
(its licence, whether it is inlined) count as well. It is taken from the code in
memory, the code that was compiled, rather than from the files on disk, so a process
that loaded an old body cannot save it as current.

**The floating-point licence** (SEP 25 section 5) is applied here, so that no kernel can
compile under another. Every compilation takes ``fastmath={"contract"}``: a fused
multiply-add rounds once where the pair rounds twice. A kernel that needs a reduction
or a stencil vectorised asks for ``fastmath={"contract", "reassoc"}``. A protected
helper -- one that forms a small difference or relies on an exact cancellation, such
as ``expm1``-like population maps, the phi-functions and the multistep defects --
takes ``strict=True`` and compiles with no flag at all. numba applies its flags to a
whole compiled function, including what it inlines, so a strict helper must be
compiled separately and called: ``strict=True`` with ``inline="always"`` is refused.
Every other flag, and ``fastmath=True``, is refused. The licence is part of the cache
key, so a function compiled under two licences keeps two entries.

This module imports nothing from the rest of the package, so that any layer may use it.
"""

import contextlib
import hashlib
import logging
import pickle
import sys
import types
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

import numba
import numpy as np
from numba.core import serialize, typeinfer
from numba.core.caching import FunctionCache
from numba.core.dispatcher import Dispatcher
from numba.core.typing.templates import Signature

logger = logging.getLogger(__name__)

__all__ = ["CONTRACT", "REASSOCIATE", "licence", "njit"]

CONTRACT = frozenset({"contract"})
"""The default licence: contraction to fused multiply-add, nothing else."""

REASSOCIATE = frozenset({"contract", "reassoc"})
"""The licence a kernel asks for on loops that need reassociation to vectorise."""

_LICENCES = (CONTRACT, REASSOCIATE)


class _KeyedCacheFile:
    """Hold the data files of one function, each named by its key and stating it.

    Parameters
    ----------
    cache_path : str
        The directory numba's locator chose for this function's cache.
    filename_base : str
        The numba file name stem for the function, ``module.qualname-line.pyXY``.
    source_stamp : object
        The source file's modification time and size, as numba's locator gives them.
    function : collections.abc.Callable
        The Python function compiled, whose dependencies the header digests.
    options : str, optional
        The compiler options in a canonical form, the floating-point licence among
        them. numba's own key leaves them out, so a function compiled under two
        licences would otherwise share one entry.
    """

    def __init__(
        self,
        cache_path: str,
        filename_base: str,
        source_stamp: object,
        function: Callable[..., Any],
        options: str = "",
    ) -> None:
        self._directory = Path(cache_path)
        self._filename_base = filename_base
        self._source_stamp = source_stamp
        self._function = function
        self._options = options
        self._dependencies: str | None = None

    def _path(self, key: tuple[Any, ...]) -> Path:
        """Return the file that holds ``key``'s compilation.

        Parameters
        ----------
        key : tuple
            The numba key, ``(signature, target, (bytecode digest, closure digest))``.

        Returns
        -------
        pathlib.Path
            The data file, named by a digest of all but the bytecode digest, and of
            the compiler options.
        """
        # The repr is the same in every process: numba types repr as their names, with
        # dtype, layout, read-only and alignment spelled out, and the rest are strings.
        signature, target, (_, closure) = key
        named = repr((signature, target, closure, self._options)).encode()
        digest = hashlib.sha256(named).hexdigest()[:32]
        return self._directory / f"{self._filename_base}.{digest}.nbc"

    def _header(self, key: tuple[Any, ...]) -> tuple[str, object, str, str, str]:
        """Return what a file holding ``key``'s compilation must begin with.

        Parameters
        ----------
        key : tuple
            The numba cache key.

        Returns
        -------
        tuple
            The numba version, the source stamp, the dependency digest, the key's repr
            and the compiler options.
        """
        # Digested at the first compilation rather than at decoration, when globals
        # defined further down the module are not bound yet.
        if self._dependencies is None:
            self._dependencies = _dependency_digest(self._function)
        return (
            numba.__version__,
            self._source_stamp,
            self._dependencies,
            repr(key),
            self._options,
        )

    def save(self, key: tuple[Any, ...], data: object) -> None:
        """Write the compilation under its key, replacing the file atomically.

        Parameters
        ----------
        key : tuple
            The numba cache key.
        data : object
            The reduced compilation numba hands its cache.
        """
        payload = serialize.dumps(data)
        try:
            with _replacing(self._path(key)) as stream:
                pickle.dump(self._header(key), stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.write(payload)
        except PermissionError:
            # Windows refuses to replace a file another process holds open, a
            # concurrent load of this key, which already holds the same compilation:
            # the write is lost, never wrong, and the next process writes it.
            if sys.platform != "win32":
                raise
            logger.debug("numba cache write lost to a reader: %s", self._path(key))

    def load(self, key: tuple[Any, ...]) -> object:
        """Return the key's compilation, or ``None`` if no file holds exactly it.

        Parameters
        ----------
        key : tuple
            The numba cache key.

        Returns
        -------
        object
            The reduced compilation, or ``None`` for a miss.
        """
        try:
            with self._path(key).open("rb") as stream:
                if pickle.load(stream) != self._header(key):
                    return None
                return pickle.loads(stream.read())
        except OSError:  # absent, or removed by hand while it was being read
            return None

    def flush(self) -> None:
        """Do nothing: without an index there is nothing to empty."""


def _dependency_digest(function: Callable[..., Any]) -> str:
    """Digest the dispatchers and constants a compilation of ``function`` reads.

    Parameters
    ----------
    function : collections.abc.Callable
        The Python function compiled.

    Returns
    -------
    str
        A hexadecimal digest, the same in every process for the same code and values.
    """
    parts: list[tuple[object, ...]] = []
    # By function, not by code: two closures over one code object can hold different
    # cells, and each counts.
    seen: set[int] = set()
    pending = [function]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        code = current.__code__
        if current is not function:  # its own bytecode is in numba's key already
            parts.append(("code", code.co_qualname, _code_value(code)))
        cells = [cell.cell_contents for cell in current.__closure__ or ()]
        named = [
            (name, current.__globals__[name])
            for name in sorted(_referenced_names(code))
            if name in current.__globals__
        ]
        for name, value in [*named, *(("<cell>", cell) for cell in cells)]:
            if isinstance(value, Dispatcher):
                # Its options too: a callee's licence, or whether it is inlined,
                # changes the machine code of what calls it (SEP 25 section 5).
                options = _canonical(dict(value.targetoptions))
                parts.append(("options", code.co_qualname, name, options))
                pending.append(value.py_func)
            elif (frozen := _frozen_value(value)) is not None:
                parts.append(("value", code.co_qualname, name, frozen))
    # Sorted, so that the order the walk met them in does not matter.
    text = repr(sorted(parts, key=repr))
    return hashlib.sha256(text.encode()).hexdigest()


def _referenced_names(code: types.CodeType) -> set[str]:
    """Return every global or attribute name a code object and its nested ones use.

    Parameters
    ----------
    code : types.CodeType
        The code object.

    Returns
    -------
    set of str
        The names; an attribute that shares a global's name is followed too, which
        can only invalidate more often, never less.
    """
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _referenced_names(constant)
    return names


def _code_value(code: types.CodeType) -> tuple[object, ...]:
    """Return a code object's bytecode, constants and names, comparable across runs.

    Parameters
    ----------
    code : types.CodeType
        The code object.

    Returns
    -------
    tuple
        The bytecode, the constants (nested code objects in the same form) and the
        names.
    """
    return code.co_code, tuple(_constant(one) for one in code.co_consts), code.co_names


def _constant(constant: object) -> object:
    """Return a code constant in a form whose repr is the same in every process.

    Parameters
    ----------
    constant : object
        An entry of ``co_consts``.

    Returns
    -------
    object
        The constant, with code objects and frozensets made comparable.
    """
    if isinstance(constant, types.CodeType):
        return _code_value(constant)
    if isinstance(constant, frozenset):
        # A frozenset's iteration order follows string hashing, salted per process.
        return ("frozenset", sorted((_constant(one) for one in constant), key=repr))
    if isinstance(constant, tuple):
        return tuple(_constant(one) for one in constant)
    return repr(constant)


def _frozen_value(value: object) -> object:
    """Return a global's value as numba freezes it, or ``None`` if it freezes none.

    Parameters
    ----------
    value : object
        A global or closure cell the compiled code reads.

    Returns
    -------
    object
        The value's repr, an array's dtype, shape and data digest, or a tuple of
        these; ``None`` for anything else (a module, a class, a plain function), whose
        repr may hold an address and so differ between processes.
    """
    if isinstance(value, np.ndarray):
        data = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return value.dtype.str, value.shape, data
    if isinstance(value, tuple):
        frozen = [_frozen_value(one) for one in value]
        return None if any(one is None for one in frozen) else tuple(frozen)
    if isinstance(value, bool | int | float | complex | str | bytes | np.generic):
        return repr(value)
    return None


@contextlib.contextmanager
def _replacing(path: Path) -> Iterator[IO[bytes]]:
    """Write a temporary file beside ``path`` and move it over ``path`` when done.

    Parameters
    ----------
    path : pathlib.Path
        The file to replace.

    Yields
    ------
    typing.IO
        The temporary file, open for binary writing.
    """
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex[:16]}")
    try:
        with temporary.open("wb") as stream:
            yield stream
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class _KeyedFunctionCache(FunctionCache):
    """Cache one function's compilations as numba does, in :class:`_KeyedCacheFile`.

    Parameters
    ----------
    py_func : collections.abc.Callable
        The Python function the dispatcher compiles.
    options : str, optional
        The canonical compiler options, which key the entries beside numba's key.
    """

    def __init__(self, py_func: Callable[..., Any], options: str = "") -> None:
        super().__init__(py_func)
        self._cache_file = _KeyedCacheFile(
            cache_path=self._cache_path,
            filename_base=self._impl.filename_base,
            source_stamp=self._impl.locator.get_source_stamp(),
            function=py_func,
            options=options,
        )


def licence(fastmath: object = None, *, strict: bool = False) -> frozenset[str]:
    """Return the fastmath flags a compilation may take, or refuse the request.

    Parameters
    ----------
    fastmath : set of str, optional
        The flags a kernel asks for: ``None`` for the default ``{"contract"}``, or
        ``{"contract", "reassoc"}``.
    strict : bool, default False
        A protected helper: no flag at all.

    Returns
    -------
    frozenset of str
        The flags, empty for a strict helper.

    Raises
    ------
    ValueError
        For ``fastmath=True`` or ``False``, any other flag set, or flags given to a
        strict helper.

    Examples
    --------
    >>> from solprop._numba_cache import licence
    >>> sorted(licence())
    ['contract']
    >>> sorted(licence({"reassoc", "contract"}))
    ['contract', 'reassoc']
    >>> licence(strict=True)
    frozenset()
    """
    if strict:
        if fastmath is not None:
            msg = (
                "a strict helper compiles without fastmath flags; pass one or the other"
            )
            raise ValueError(msg)
        return frozenset()
    if fastmath is None:
        return CONTRACT
    if isinstance(fastmath, set | frozenset) and frozenset(fastmath) in _LICENCES:
        return frozenset(fastmath)
    msg = (
        f"fastmath={fastmath!r} is outside the floating-point licence: a kernel takes "
        "the default {'contract'}, asks for {'contract', 'reassoc'}, or is strict=True "
        "(docs/dev/backends.md section 3.2)"
    )
    raise ValueError(msg)


def _canonical(options: dict[str, Any]) -> str:
    """Return compiler options as a string that is the same in every process.

    Parameters
    ----------
    options : dict
        The keywords handed to :func:`numba.njit`.

    Returns
    -------
    str
        The sorted options, sets as sorted tuples.
    """
    items = [
        (name, tuple(sorted(value)) if isinstance(value, set | frozenset) else value)
        for name, value in sorted(options.items())
    ]
    return repr(items)


def njit(
    signature: str | Signature | list[str | Signature] | None = None,
    /,
    *,
    cache: bool = False,
    fastmath: object = None,
    strict: bool = False,
    **options: Any,
) -> Callable[[Callable[..., Any]], Any]:
    """Return a decorator that compiles like :func:`numba.njit`, caching safely.

    Every compilation takes the floating-point licence of SEP 25 (:func:`licence`):
    contraction by default, reassociation where a kernel asks for it, nothing for a
    strict helper.

    Parameters
    ----------
    signature : str or numba.core.typing.templates.Signature or list of them, optional
        An explicit signature, or a list of them, compiled at once and then frozen,
        as numba does with a positional signature.
    cache : bool, default False
        Write compilations to the disk cache, each in a file that names the key it
        holds, so that concurrent processes cannot cross-wire them.
    fastmath : set of str, optional
        ``{"contract", "reassoc"}`` for a kernel whose loops need reassociation;
        ``None`` takes the default ``{"contract"}``. Nothing else is accepted.
    strict : bool, default False
        Compile a protected helper with no fastmath flag. It must be called, never
        inlined: ``inline="always"`` would compile it under its caller's flags.
    **options
        Any other keyword :func:`numba.njit` takes (``nogil``, ``parallel``,
        ``inline``, ...).

    Returns
    -------
    collections.abc.Callable
        The decorator, which returns a numba dispatcher (or, under
        ``NUMBA_DISABLE_JIT``, the function itself, as numba does).

    Raises
    ------
    ValueError
        If the flags are outside the licence, or a strict helper is to be inlined.

    Examples
    --------
    >>> from solprop._numba_cache import njit
    >>> twice = njit(cache=False)(lambda x: 2 * x)
    >>> twice(3)
    6
    """
    flags = licence(fastmath, strict=strict)
    if strict and options.get("inline", "never") != "never":
        msg = (
            "a strict helper is compiled separately and called: inlined, it would "
            "compile under its caller's fastmath flags"
        )
        raise ValueError(msg)
    options = {**options, "fastmath": set(flags) if flags else False}

    def decorate(function: Callable[..., Any]) -> Any:
        """Compile ``function``, with the keyed cache when one was asked for.

        Parameters
        ----------
        function : collections.abc.Callable
            The Python function to compile.

        Returns
        -------
        object
            Its dispatcher, or the function itself under ``NUMBA_DISABLE_JIT``.
        """
        dispatcher = numba.njit(**options)(function)
        if not isinstance(dispatcher, Dispatcher):
            return dispatcher
        if cache:
            # What numba's ``enable_caching`` does, with the keyed cache; it must happen
            # before an eager signature compiles, so the signature is compiled here, as
            # ``numba.jit`` compiles it. ``_cache`` is the slot numba's own method sets.
            dispatcher._cache = _KeyedFunctionCache(function, _canonical(options))  # pyright: ignore[reportAttributeAccessIssue]
        if signature is not None:
            signatures = signature if isinstance(signature, list) else [signature]
            with typeinfer.register_dispatcher(dispatcher):
                for one in signatures:
                    dispatcher.compile(one)
                dispatcher.disable_compile()
        return dispatcher

    return decorate
