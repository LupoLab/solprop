# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frames: plain data with arrays, as bytes for a file or a stream (design 13.4).

A run's record and the helper's messages carry plain data (dicts, lists, strings,
numbers, ``None``) and NumPy arrays. A frame writes one such value as a fixed preamble,
a JSON header and the arrays' raw bytes, so no pickle ever crosses a network or sits in
a run's folder, and arrays cost no text conversion::

    MAGIC (4 bytes) | header length (uint32) | body length (uint64) | header | body

The header is the value with each array replaced by a marker naming its place in the
body, its dtype and its shape. All integers are little-endian. The magic lets a reader
skip whatever a remote shell prints before the first frame and refuse a stream that is
not one. A reader of a file being appended to takes only the frames that are complete.
"""

import json
import math
import struct
from collections.abc import Mapping
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from solprop.service.messages import ServiceError

__all__ = [
    "MAGIC",
    "Readable",
    "Writable",
    "decode",
    "encode",
    "read_frame",
    "split_frames",
    "write_frame",
]

MAGIC: Final = b"SPf1"
"""The four bytes every frame starts with: the format and its version."""


@runtime_checkable
class Readable(Protocol):
    """A binary stream frames are read from: a pipe, a file, an SSH session's output."""

    def read(self, size: int = -1, /) -> bytes:
        """Return up to ``size`` bytes; empty at the stream's end.

        Parameters
        ----------
        size : int, optional
            The most bytes; -1 for all there are.

        Returns
        -------
        bytes
            The bytes.
        """
        ...


@runtime_checkable
class Writable(Protocol):
    """A binary stream frames are written to."""

    def write(self, data: bytes, /) -> int:
        """Write bytes.

        Parameters
        ----------
        data : bytes
            The bytes.

        Returns
        -------
        int
            How many were written.
        """
        ...

    def flush(self) -> None:
        """Send what is buffered."""
        ...


_PREAMBLE: Final = struct.Struct("<4sIQ")
_ARRAY: Final = "__ndarray__"
_FLOAT: Final = "__float__"
_LARGEST: Final = 1 << 34
"""The largest header or body accepted, bytes: a guard against a corrupt length."""


def _plain(value: object, arrays: list[np.ndarray]) -> object:
    """Return a value's JSON form, collecting its arrays."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {_FLOAT: repr(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            msg = "a frame carries numeric arrays only, not object arrays"
            raise ServiceError(msg)
        # Not ascontiguousarray, which makes a 0-d array 1-d.
        arrays.append(value if value.flags.c_contiguous else value.copy(order="C"))
        return {_ARRAY: len(arrays) - 1}
    if isinstance(value, np.generic):
        return _plain(value.item(), arrays)
    if isinstance(value, Mapping):
        plain: dict[str, object] = {}
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(key, str):
                msg = f"a frame's keys are strings, not {type(key).__name__}"  # pyright: ignore[reportUnknownArgumentType]
                raise ServiceError(msg)
            plain[key] = _plain(item, arrays)
        return plain
    if isinstance(value, list | tuple):
        return [_plain(item, arrays) for item in value]  # pyright: ignore[reportUnknownVariableType]
    msg = f"a frame carries plain data and arrays, not {type(value).__name__}"
    raise ServiceError(msg)


def encode(value: object) -> bytes:
    """Return one frame holding a value.

    Parameters
    ----------
    value : object
        Plain data (dicts with string keys, lists and tuples, strings, numbers,
        booleans, ``None``) and NumPy arrays; tuples come back as lists.

    Returns
    -------
    bytes
        The frame.

    Raises
    ------
    solprop.service.messages.ServiceError
        For a value that is not plain data.

    Examples
    --------
    >>> frame = encode({"z": 0.5, "power": np.arange(3.0)})
    >>> frame[:4] == MAGIC
    True
    >>> value, used = decode(frame)
    >>> value["z"], value["power"].tolist(), used == len(frame)
    (0.5, [0.0, 1.0, 2.0], True)
    """
    arrays: list[np.ndarray] = []
    plain = _plain(value, arrays)
    blocks: list[dict[str, object]] = []
    offset = 0
    for array in arrays:
        blocks.append(
            {"dtype": array.dtype.str, "shape": list(array.shape), "offset": offset}
        )
        offset += array.nbytes
    header = json.dumps(
        {"value": plain, "arrays": blocks}, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    body = b"".join(array.tobytes() for array in arrays)
    return _PREAMBLE.pack(MAGIC, len(header), len(body)) + header + body


def _restore(value: object, arrays: list[np.ndarray]) -> object:
    """Return a header's value with its markers replaced."""
    if isinstance(value, dict):
        if _ARRAY in value and len(value) == 1:  # pyright: ignore[reportUnknownArgumentType]
            return arrays[int(value[_ARRAY])]  # pyright: ignore[reportUnknownArgumentType]
        if _FLOAT in value and len(value) == 1:  # pyright: ignore[reportUnknownArgumentType]
            return float(value[_FLOAT])  # pyright: ignore[reportUnknownArgumentType]
        return {key: _restore(item, arrays) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_restore(item, arrays) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def _lengths(preamble: bytes) -> tuple[int, int]:
    """Return a frame's header and body lengths from its preamble.

    - ``preamble``: the frame's first bytes, at least the preamble's size.
    - Returns: the header's and the body's lengths, bytes.
    """
    magic, header, body = _PREAMBLE.unpack_from(preamble)
    if magic != MAGIC:
        msg = f"not a solprop frame: it starts {magic!r}"
        raise ServiceError(msg)
    if header > _LARGEST or body > _LARGEST:
        msg = "a frame's length is beyond any message solprop sends"
        raise ServiceError(msg)
    return header, body


def _value(header: bytes, body: bytes) -> object:
    """Return the value a frame's header and body hold.

    - ``body``: the arrays' bytes.
    """
    content: dict[str, Any] = json.loads(header.decode("utf-8"))
    arrays: list[np.ndarray] = []
    for block in content["arrays"]:
        dtype = np.dtype(block["dtype"])
        shape = tuple(int(n) for n in block["shape"])
        count = math.prod(shape)
        array = np.frombuffer(body, dtype=dtype, count=count, offset=block["offset"])
        arrays.append(array.reshape(shape).copy())
    return _restore(content["value"], arrays)


def decode(data: bytes) -> tuple[object, int]:
    """Return the value of the frame at the start of some bytes.

    Parameters
    ----------
    data : bytes
        The bytes, a whole frame at least.

    Returns
    -------
    tuple
        The value, and the frame's length in bytes.

    Raises
    ------
    solprop.service.messages.ServiceError
        If the bytes do not start with a whole frame.

    Examples
    --------
    >>> decode(encode([1, "a", None, float("nan")]))[0]
    [1, 'a', None, nan]
    """
    if len(data) >= _PREAMBLE.size:
        header, body = _lengths(data)
        end = _PREAMBLE.size + header + body
        if end <= len(data):
            start = _PREAMBLE.size
            return _value(data[start : start + header], data[start + header : end]), end
    msg = "the bytes hold no whole frame"
    raise ServiceError(msg)


def split_frames(data: bytes) -> tuple[list[object], int]:
    """Return the values of the whole frames at the start of some bytes.

    A file still being appended to may end in part of a frame; that part is left for
    the next read.

    Parameters
    ----------
    data : bytes
        The bytes.

    Returns
    -------
    tuple
        The values in order, and how many bytes they took.

    Raises
    ------
    solprop.service.messages.ServiceError
        If the bytes are not frames.

    Examples
    --------
    >>> data = encode({"a": 1}) + encode({"b": 2})
    >>> split_frames(data + data[:7]) == ([{"a": 1}, {"b": 2}], len(data))
    True
    """
    values: list[object] = []
    used = 0
    while len(data) - used >= _PREAMBLE.size:
        header, body = _lengths(data[used : used + _PREAMBLE.size])
        end = used + _PREAMBLE.size + header + body
        if end > len(data):
            break
        start = used + _PREAMBLE.size
        values.append(_value(data[start : start + header], data[start + header : end]))
        used = end
    return values, used


def _read_exactly(stream: Readable, size: int) -> bytes | None:
    """Return the next bytes of a stream, or ``None`` at its end.

    - ``size``: how many bytes.
    - Returns: exactly that many; ``None`` when the stream ended before the first.
    """
    chunks: list[bytes] = []
    left = size
    while left:
        chunk = stream.read(left)
        if not chunk:
            if left == size:
                return None
            msg = "the stream ended inside a frame"
            raise ServiceError(msg)
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: Readable, *, skip_preface: bool = False) -> object:
    r"""Return the next frame's value from a stream, waiting for it.

    Parameters
    ----------
    stream : Readable
        A binary stream (a pipe, an SSH session's output).
    skip_preface : bool, optional
        Skip any bytes before the first frame's magic, as a remote shell's greeting.

    Returns
    -------
    object
        The value.

    Raises
    ------
    EOFError
        If the stream ends before a frame starts.
    solprop.service.messages.ServiceError
        If what comes is not a frame, or ends inside one.

    Examples
    --------
    >>> import io
    >>> read_frame(io.BytesIO(b"Welcome!\n" + encode("hi")), skip_preface=True)
    'hi'
    """
    if skip_preface:
        seen = b""
        while not seen.endswith(MAGIC):
            byte = stream.read(1)
            if not byte:
                raise EOFError
            seen = (seen + byte)[-len(MAGIC) :]
        rest = _read_exactly(stream, _PREAMBLE.size - len(MAGIC))
        if rest is None:
            msg = "the stream ended inside a frame"
            raise ServiceError(msg)
        preamble = MAGIC + rest
    else:
        found = _read_exactly(stream, _PREAMBLE.size)
        if found is None:
            raise EOFError
        preamble = found
    header, body = _lengths(preamble)
    content = _read_exactly(stream, header + body) if header + body else b""
    if content is None:
        msg = "the stream ended inside a frame"
        raise ServiceError(msg)
    return _value(content[:header], content[header:])


def write_frame(stream: Writable, value: object) -> None:
    """Write one frame to a stream and flush it.

    Parameters
    ----------
    stream : Writable
        The stream.
    value : object
        The value (:func:`encode`).

    Examples
    --------
    >>> import io
    >>> buffer = io.BytesIO()
    >>> write_frame(buffer, {"ok": True})
    >>> decode(buffer.getvalue())[0]
    {'ok': True}
    """
    stream.write(encode(value))
    stream.flush()
