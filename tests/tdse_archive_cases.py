# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Manufactured provider records and checks for the TDSE archive tests (SEP 20).

The records carry every field of the provider's four public result types, with
the provider's dtypes (float64 histories, bool validity and flags, int64 step
indices), deterministic values from a seeded generator, and optional non-finite
and invalid samples, so a round trip can be compared exactly.
"""

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal

from solprop.output.tdse_archive import (
    HACC_GRID_RESPONSE,
    HACC_TOTAL_CURRENT_RESPONSE,
    SAE_GRID_RESPONSE,
    SAE_TOTAL_CURRENT_RESPONSE,
    RecordSchema,
    TDSEArchive,
    TDSEArchiveWriter,
    structured_json,
)


@dataclasses.dataclass(frozen=True)
class Outcome:
    failure: str = "none"
    failure_step: int | None = None

    @property
    def ok(self):
        return self.failure == "none"


@dataclasses.dataclass(frozen=True)
class Info:
    rows: int
    samples: int
    spacing: float
    outputs: str
    shadow_order: int
    nested: object = None


@dataclasses.dataclass(frozen=True, eq=False)
class SaeMonitors:
    initial_norm: np.ndarray
    norm: np.ndarray
    peak_norm: np.ndarray
    field_free_energy: np.ndarray
    monitored_step: np.ndarray
    removed_component: np.ndarray | None
    step_error_estimate: np.ndarray | None
    peak_step_error_estimate: np.ndarray | None


@dataclasses.dataclass(frozen=True, eq=False)
class HaccMonitors:
    initial_norm: np.ndarray
    norm: np.ndarray
    peak_norm: np.ndarray
    monitored_step: np.ndarray
    non_finite: np.ndarray


class Shadow(SimpleNamespace):
    """A provider shadow record with its ``component`` accessor."""

    def component(self, quantity, order):
        return getattr(self, quantity)[:, self.orders.index(order), :]


def _histories(rng, rows, samples, *names):
    return {name: rng.standard_normal((rows, samples)) for name in names}


def _ordered(rng, rows, orders, samples):
    return rng.standard_normal((rows, len(orders), samples))


def sae_monitors(rng, rows, *, optional=True):
    return SaeMonitors(
        initial_norm=np.ones(rows),
        norm=rng.uniform(0.9, 1.0, rows),
        peak_norm=np.ones(rows),
        field_free_energy=rng.standard_normal(rows) * 1e-18,
        monitored_step=np.arange(rows, dtype=np.int64) + 40,
        removed_component=rng.uniform(0, 1e-12, rows) if optional else None,
        step_error_estimate=None,
        peak_step_error_estimate=None,
    )


def hacc_monitors(rng, rows):
    return HaccMonitors(
        initial_norm=np.ones(rows),
        norm=rng.uniform(0.9, 1.0, rows),
        peak_norm=np.ones(rows),
        monitored_step=np.full(rows, 40, dtype=np.int64),
        non_finite=np.zeros(rows, dtype=bool),
    )


def common(rng, rows, samples, *, outputs, shadow_order, spacing=1e-17):
    return {
        "time": np.arange(samples) * spacing,
        "outcomes": tuple(Outcome() for _ in range(rows)),
        "valid": np.ones((rows, samples), dtype=bool),
        "info": Info(
            rows,
            samples,
            spacing,
            outputs,
            shadow_order,
            nested=SimpleNamespace(build={"threads": 1}, edges=np.array([0.0, 1.0])),
        ),
    }


def sae_full(rng, rows=2, samples=12, orders=(1, 3)):
    plasma = SimpleNamespace(
        radius=4e-9,
        potential_slope_bound=1.5e-9,
        **_histories(
            rng,
            rows,
            samples,
            "inbox_acceleration",
            "free_acceleration",
            "total_acceleration",
            "inbox_current",
            "free_current",
            "total_current",
            "outside_integral",
            "neglected_acceleration_bound",
            "neglected_current_bound",
        ),
    )
    shadow = Shadow(
        orders=orders,
        radius=4e-9,
        charge_dipole=_ordered(rng, rows, orders, samples),
        current=_ordered(rng, rows, orders, samples),
        total_acceleration=_ordered(rng, rows, orders, samples),
        total_current=_ordered(rng, rows, orders, samples),
    )
    return SimpleNamespace(
        **common(rng, rows, samples, outputs="full", shadow_order=max(orders)),
        **_histories(
            rng,
            rows,
            samples,
            "vector_potential",
            "electric_field",
            "charge_dipole",
            "current",
            "dipole_acceleration",
            "outside",
        ),
        plasma=plasma,
        cut_term=1.25e-3,
        monitors=sae_monitors(rng, rows),
        shadow=shadow,
    )


def sae_compact(rng, rows=2, samples=12, orders=(1, 3)):
    return SimpleNamespace(
        **common(rng, rows, samples, outputs="total_current", shadow_order=max(orders)),
        total_current=rng.standard_normal((rows, samples)),
        shadow_total_current=_ordered(rng, rows, orders, samples),
        shadow_orders=orders,
        radius=4e-9,
        potential_slope_bound=1.5e-9,
        neglected_acceleration_bound=rng.uniform(0, 1, (rows, samples)),
        neglected_current_bound=rng.uniform(0, 1, (rows, samples)),
        monitors=sae_monitors(rng, rows, optional=False),
    )


def hacc_full(rng, rows=2, samples=12, *, closure=True):
    shadow = SimpleNamespace(
        orders=(1,),
        charge_dipole=_ordered(rng, rows, (1,), samples),
        current=_ordered(rng, rows, (1,), samples),
        total_current=_ordered(rng, rows, (1,), samples) if closure else None,
    )
    return SimpleNamespace(
        **common(rng, rows, samples, outputs="full", shadow_order=1),
        **_histories(
            rng,
            rows,
            samples,
            "vector_potential",
            "electric_field",
            "charge_dipole",
            "current",
            "total_current",
            "outside",
        ),
        neglected_current_bound=rng.uniform(0, 1, (rows, samples)) if closure else None,
        monitors=hacc_monitors(rng, rows),
        shadow=shadow,
    )


def hacc_compact(rng, rows=2, samples=12):
    return SimpleNamespace(
        **common(rng, rows, samples, outputs="total_current", shadow_order=1),
        total_current=rng.standard_normal((rows, samples)),
        shadow_total_current=_ordered(rng, rows, (1,), samples),
        shadow_orders=(1,),
        radius=20 * 5.29e-11,
        neglected_current_bound=rng.uniform(0, 1, (rows, samples)),
        monitors=hacc_monitors(rng, rows),
    )


RECORDS = {
    "sae_full": (sae_full, SAE_GRID_RESPONSE),
    "sae_compact": (sae_compact, SAE_TOTAL_CURRENT_RESPONSE),
    "hacc_full": (hacc_full, HACC_GRID_RESPONSE),
    "hacc_compact": (hacc_compact, HACC_TOTAL_CURRENT_RESPONSE),
}
"""Each manufactured record with its serializer."""


def writer(
    path, *, outputs: Any = "full", run_uuid="run-1", kind: Any = "run", **kwargs
):
    """Open a writer session on `path` with a small context."""
    return TDSEArchiveWriter.open(
        TDSEArchive(path, outputs=outputs),
        run_uuid=run_uuid,
        kind=kind,
        optical=kwargs.pop("optical", None),
        context={"test": True},
        **kwargs,
    )


def labels(**changes):
    values = {
        "trial_id": 0,
        "role": "predictor",
        "purpose": "predictor",
        "z": 0.25,
        "density": 2.5e24,
        "input_identity": "abc",
    }
    values.update(changes)
    return values


def begin(archive, rng, rows=2, samples=12, chunks=1, **changes):
    """Begin one evaluation with `chunks` chunks of `rows` rows."""
    prewindow = rng.standard_normal((rows * chunks, samples + 4))
    drives = [
        (
            index * rows,
            (index + 1) * rows,
            prewindow[index * rows : (index + 1) * rows, 2:-2],
        )
        for index in range(chunks)
    ]
    identifier = archive.begin_evaluation(
        labels(**changes),
        prewindow=prewindow,
        drives=drives,
        gate=np.linspace(0.0, 1.0, samples + 4),
        binding={"build": {"revision": "272ef73"}, "chunks": rows},
    )
    return identifier, prewindow, drives


def assert_record_equal(raw, stored, schema: RecordSchema):
    """Assert that a stored record reproduces every field of `raw` exactly.

    Lossless storage is a structural contract: exact, not a rounding comparison.
    """
    assert stored["serializer"] == schema.serializer
    for item in schema.fields:
        value = getattr(raw, item.name)
        got = stored[item.name]
        if value is None:
            assert got is None, item.name
        elif item.kind == "array":
            assert isinstance(got, np.ndarray), item.name
            assert got.dtype == value.dtype and got.shape == value.shape, item.name
            assert_array_equal(got, value, strict=True)
        elif item.kind == "scalar":
            assert got == value and np.asarray(got).dtype == np.asarray(value).dtype
        elif item.kind == "labels":
            assert got == value, item.name
        elif item.kind == "outcomes":
            assert got == [
                {"failure": one.failure, "failure_step": one.failure_step}
                for one in value
            ]
        elif item.kind == "structured":
            assert got == json.loads(structured_json(value))
        else:
            assert item.record is not None
            assert_record_equal(value, got, item.record)
