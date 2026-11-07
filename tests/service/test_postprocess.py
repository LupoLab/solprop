# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Post-process step's compressor: its stages, and the kernel's chain on a run."""

import warnings
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from PySide6.QtWidgets import QLineEdit
from solprop_gui.model.compressor import (
    payload,
)

from service.test_results import beam, run  # noqa: F401 - fixtures
from solprop.service import handlers


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    """Ignore the material-range warnings of the wide bands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, body: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](body)


def chain(path: str, *stages: Any, **extra: object) -> Any:
    body = {"path": path, "index": -1, "stages": payload(stages), **extra}
    return ask("compress_chain", body)


def test_the_catalogue_names_glasses_gases_and_mirrors():
    found = ask("optics_catalogue", {})
    assert "fused silica" in found["glasses"] and "Air" in found["gases"]
    assert found["media"] == found["glasses"] + found["gases"]
    assert "PC70" in found["mirrors"]


# ------------------------------------------------------------------ the page


def type_into(field: QLineEdit, text: str) -> None:
    field.setText(text)
    field.editingFinished.emit()
