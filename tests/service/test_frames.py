# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frames: plain data and arrays through bytes, and the ends no doctest reaches."""

import io
import struct
from typing import Any

import numpy as np
import pytest

from solprop.service import frames
from solprop.service.frames import (
    MAGIC,
    decode,
    encode,
    read_frame,
    split_frames,
    write_frame,
)
from solprop.service.messages import ServiceError


def test_plain_data_and_arrays_come_back_as_they_went():
    value = {
        "none": None,
        "flag": True,
        "count": 3,
        "text": "λ",
        "pair": (1, 2.5),
        "scalar": np.float32(1.5),
        "inf": float("inf"),
        "nan": float("nan"),
        "complex": np.array([1 + 2j, 3 - 4j]),
        "map": np.arange(6, dtype=np.int16).reshape(2, 3),
        "empty": np.zeros((0, 4)),
        "zero_d": np.array(7.0),
        "mask": np.array([True, False]),
        "strided": np.arange(10.0)[::3],
    }
    decoded, used = decode(encode(value))
    back: Any = decoded
    assert used == len(encode(value))
    assert back["none"] is None and back["flag"] is True and back["count"] == 3
    assert back["text"] == "λ" and back["pair"] == [1, 2.5] and back["scalar"] == 1.5
    assert back["inf"] == float("inf") and np.isnan(back["nan"])
    for name in ("complex", "map", "empty", "zero_d", "mask", "strided"):
        assert back[name].dtype == np.asarray(value[name]).dtype
        assert np.array_equal(back[name], value[name])
    back["map"][0, 0] = 9  # writable: the reader owns its arrays


@pytest.mark.parametrize(
    ("value", "says"),
    [
        (np.array([object()]), "numeric arrays"),
        ({1: "a"}, "keys are strings"),
        ({"a": {1, 2}}, "plain data"),
    ],
)
def test_what_is_not_plain_data_is_refused(value, says):
    with pytest.raises(ServiceError, match=says):
        encode(value)


def test_bytes_that_are_not_frames_are_refused():
    with pytest.raises(ServiceError, match="not a solprop frame"):
        split_frames(b"HTTP/1.1 200 OK and more")
    with pytest.raises(ServiceError, match="no whole frame"):
        decode(encode("x")[:-1])
    with pytest.raises(ServiceError, match="no whole frame"):
        decode(b"SP")
    huge = struct.pack("<4sIQ", MAGIC, 1, 1 << 40)
    with pytest.raises(ServiceError, match="beyond any message"):
        split_frames(huge)


def test_a_stream_gives_its_frames_and_says_where_it_ends():
    buffer = io.BytesIO()
    write_frame(buffer, {"a": 1})
    write_frame(buffer, [2])
    buffer.seek(0)
    assert read_frame(buffer) == {"a": 1}
    assert read_frame(buffer) == [2]
    with pytest.raises(EOFError):
        read_frame(buffer)


def test_a_stream_that_ends_inside_a_frame_is_refused():
    frame = encode({"power": np.arange(4.0)})
    for cut in (6, len(frame) - 3):
        with pytest.raises(ServiceError, match="inside a frame"):
            read_frame(io.BytesIO(frame[:cut]))
    with pytest.raises(ServiceError, match="inside a frame"):
        read_frame(io.BytesIO(b"banner" + frame[:6]), skip_preface=True)


def test_a_preface_is_skipped_only_when_asked():
    stream = io.BytesIO(b"motd\n" + encode("hi"))
    with pytest.raises(ServiceError, match="not a solprop frame"):
        read_frame(stream)
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(b"no frame at all"), skip_preface=True)


def test_an_empty_frame_has_no_body():
    frame = encode(None)
    assert read_frame(io.BytesIO(frame)) is None
    assert frame[:4] == frames.MAGIC


def test_frames_cut_after_their_preamble_wait_or_are_refused():
    frame = encode({"power": [1.0, 2.0, 3.0]})
    assert split_frames(frame + frame[:20]) == ([decode(frame)[0]], len(frame))
    with pytest.raises(ServiceError, match="inside a frame"):
        read_frame(io.BytesIO(frame[:16]))  # the preamble, and nothing after it
    with pytest.raises(ServiceError, match="inside a frame"):
        read_frame(io.BytesIO(b"motd" + MAGIC), skip_preface=True)
