# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Manufactured adapters the source worker serves in the tests, inside the worker.

``make_adapter`` is a provider-agnostic adapter whose requests misbehave on purpose:
it echoes, hangs, raises, exits, prints, closes its pipe or writes a malformed frame.
The ``*_tdse`` factories serve the TDSE binding with Pyroraptor's adapter around the
manufactured haCC provider of ``hacc_cases`` (SEP 49 section 7.2), the module this
worker imports in place of ``pyroraptor``.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from solprop.driver import _pyroraptor, _source_worker

HANG_SECONDS = 30.0
"""How long a manufactured hang lasts: bounded, so that a broken cancellation fails a
test (the call returns) instead of hanging it, and far above any cancellation's time."""


class _Manufactured:
    """An adapter with no provider: each request names its behaviour."""

    def identity(self):
        return {"adapter": "manufactured", "pid": os.getpid()}

    def classify(self, error):
        del error
        return "manufactured"

    def computed(self, value):
        del value
        return {}

    def handle(self, kind, value):
        if kind == "echo":
            return value
        if kind == "hang":
            time.sleep(HANG_SECONDS)
            return "woke"
        if kind == "raise":
            raise ValueError("manufactured provider failure")
        if kind == "exit":
            sys.stderr.write("the manufactured provider gives up\n")
            sys.stderr.flush()
            os._exit(7)
        if kind == "print":
            sys.stdout.write("noise on standard output\n")
            sys.stdout.flush()
            os.write(1, b"raw noise on descriptor 1\n")
            return "printed"
        if kind == "oversized":
            return np.zeros(value["bytes"], dtype=np.uint8)
        if kind == "malformed":
            payload = b"{not json"
            _source_worker._write_all(
                _source_worker.PROTOCOL_DESCRIPTORS[1],
                _source_worker._HEADER.pack(len(payload)) + payload,
            )
            time.sleep(HANG_SECONDS)
        if kind == "close_pipe":
            os.close(_source_worker.PROTOCOL_DESCRIPTORS[1])
            time.sleep(HANG_SECONDS)
        if kind == "modules":
            spec = getattr(sys.modules["__main__"], "__spec__", None)
            return {
                "modules": sorted(sys.modules),
                "main": None if spec is None else spec.name,
            }
        if kind == "environment":
            return dict(os.environ)
        raise KeyError(kind)

    def close(self):
        pass


def make_adapter():
    return _Manufactured()


def failing_adapter():
    raise ValueError("the manufactured provider cannot start")


def _provider(**changes):
    from hacc_cases import FakeHaccProvider, alpha_law  # noqa: PLC0415 - worker only

    provider = FakeHaccProvider(**changes)
    # The run tests' law: a 2% linear model error and a dyadic fifth-order remainder.
    provider.response = alpha_law(1.02, lambda5=2.0**-200)
    marker = os.environ.get("MANUFACTURED_FAIL_ONCE")
    if marker is not None:
        _fail_once(provider, marker)
    hang = os.environ.get("MANUFACTURED_HANG_AFTER")
    if hang is not None:
        _hang_after(provider, int(hang))
    return provider


def _fail_once(provider, marker):
    """Make the first atomic call of all the workers sharing `marker` raise."""
    from hacc_cases import FakeDrivenError  # noqa: PLC0415 - worker only

    original = provider.hacc.ResponsePlan

    def plan(model, **values):
        built = original(model, **values)
        evaluate = built.evaluate

        def failing(fields):
            if not Path(marker).exists():
                Path(marker).touch()
                raise FakeDrivenError("manufactured provider failure")
            return evaluate(fields)

        built.evaluate = failing
        return built

    provider.hacc.ResponsePlan = plan


def _hang_after(provider, calls):
    """Make every atomic call after the first `calls` hang."""
    original = provider.hacc.ResponsePlan

    def plan(model, **values):
        built = original(model, **values)
        evaluate = built.evaluate

        def slow(fields):
            if len(provider.calls) >= calls:
                sys.stderr.write("MANUFACTURED CALL HANGS\n")
                sys.stderr.flush()
                time.sleep(HANG_SECONDS)
            return evaluate(fields)

        built.evaluate = slow
        return built

    provider.hacc.ResponsePlan = plan


def _adapter(provider):
    return _pyroraptor.PyroraptorAdapter(cast("Any", provider))


def manufactured_tdse():
    return _adapter(_provider())


def stale_tdse():
    from hacc_cases import FakeBuild  # noqa: PLC0415 - worker only

    return _adapter(_provider(build=FakeBuild(source_revision="old0000")))


def cuda_tdse():
    from hacc_cases import FakeBuild  # noqa: PLC0415 - worker only

    return _adapter(_provider(build=FakeBuild(execution_space="Cuda")))
