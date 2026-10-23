# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the separate host Tools collector's concurrency and lossless labels."""

import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import solprop_native


def test_label_collector_counts_concurrent_calls_and_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not solprop_native.build_info().coverage_instrumented:
        pytest.skip("label module is installed by the separate coverage build")
    library = next(Path(solprop_native.__file__).parent.glob("solprop_native_labels.*"))
    collector = ctypes.CDLL(str(library))
    begin = collector.kokkosp_begin_parallel_for
    begin.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    begin.restype = None
    label = 'solprop_native.test.\\quoted"\t\n'

    def call(_: int) -> int:
        identifier = ctypes.c_uint64()
        begin(label.encode(), 0, ctypes.byref(identifier))
        return identifier.value

    with ThreadPoolExecutor(max_workers=4) as pool:
        identifiers = list(pool.map(call, range(200)))
    assert len(set(identifiers)) == 200
    output = tmp_path / "labels.jsonl"
    monkeypatch.setenv("SOLPROP_NATIVE_LABEL_FILE", str(output))
    collector.kokkosp_finalize_library()
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"kind": "parallel_for", "label": label, "count": 200}]
