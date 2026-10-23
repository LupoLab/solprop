# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An optmat adapter evaluates each frequency array once (SEP 3, Q22).

A material's index or polarizability does not depend on z, yet a translation reads it
at every sample of a z table. Each adapter keeps its last few evaluations
(:class:`~solprop.physdata._optmat.EvaluationCache`): a repeated frequency array takes
the stored arrays, exactly what a fresh evaluation in the same process gives (a cache
stores a result, it computes no other one: a structural contract), and the warnings and
range
records of the first evaluation are raised and recorded again under the same
once-per-context policy. The capillary route's frame tables sample the mode's fill and
wall once per frequency vector besides.
"""

import contextlib
import copy
import dataclasses
import math
import pickle

import numpy as np
import pytest
from optmat.exceptions import StateError

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.modes.capillary import MarcatiliMode
from solprop.physdata import _optmat, fused_silica
from solprop.physdata.glasses import OptmatGlass
from solprop.physdata.selection import MaterialLog
from solprop.physdata.sellmeier import OptmatSusceptibility
from solprop.translate import problem as problem_module

OMEGA = 2.0 * math.pi * C / np.geomspace(300e-9, 2e-6, 33)
"""A band inside fit/SiO2's and fit/Ar's qualified intervals, rad/s."""

BEYOND = 2.0 * math.pi * C / np.array([60e-9, 800e-9])
"""A band reaching below fit/Ar's shortest qualified wavelength, rad/s."""


def silica() -> OptmatGlass:
    """Return a fresh adapter of optmat's recommended silica."""
    adapted = fused_silica(model="recommended")
    assert isinstance(adapted, OptmatGlass)
    return adapted


def counted(monkeypatch, cls, *names):
    """Count the calls of some methods of an optmat model class."""
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(cls, name)

        def wrapper(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, wrapper)
    return calls


def same_bits(left, right):
    """Return whether two arrays are the same to the bit, shape and type.

    The cache's contract is structural: it hands back the arrays the same function of
    the same inputs produced in this process, so exact identity is what it owes.
    """
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def test_a_repeated_band_is_evaluated_once_and_exactly_as_directly(monkeypatch):
    adapted = silica()
    model = _optmat.untyped(adapted.model)
    wavelength = _optmat.omega_to_wavelength(OMEGA)
    direct_index = np.asarray(model.refractive_index(wavelength), dtype=complex)
    direct_slope = np.asarray(model.dindex_domega(wavelength), dtype=complex)
    calls = counted(monkeypatch, type(model), "refractive_index", "dindex_domega")
    readings = [
        adapted.index(OMEGA),
        adapted.dindex_domega(OMEGA.copy()),
        adapted.permittivity(list(OMEGA)),
        adapted.dpermittivity_domega(OMEGA),
        adapted.index(OMEGA),
    ]
    # The permittivity and its derivative read both quantities each: once in all.
    assert calls == {"refractive_index": 1, "dindex_domega": 1}
    assert same_bits(readings[0], direct_index)
    assert same_bits(readings[1], direct_slope)
    assert same_bits(readings[2], direct_index**2)
    assert same_bits(readings[3], 2.0 * direct_index * direct_slope)
    assert same_bits(readings[4], direct_index)
    # A fresh adapter, which has evaluated nothing, evaluates and gives the same bits.
    assert same_bits(silica().index(OMEGA), readings[0])
    assert calls == {"refractive_index": 2, "dindex_domega": 2}
    # Another band, or the same values in another shape, is another evaluation.
    adapted.index(OMEGA[:5])
    adapted.index(OMEGA[:1].reshape(()))
    assert calls == {"refractive_index": 4, "dindex_domega": 4}


def test_a_returned_array_is_the_callers_own(monkeypatch):
    adapted = silica()
    first = adapted.index(OMEGA)
    held = first.copy()
    assert first.flags.writeable
    first[:] = 0.0
    assert same_bits(adapted.index(OMEGA), held)


def test_the_cache_keeps_a_bounded_number_of_bands():
    adapted = silica()
    for count in range(3 * _optmat.EVALUATION_CACHE_SIZE):
        adapted.index(OMEGA * (1.0 + count * 1e-3))
    assert len(adapted._evaluations) == _optmat.EVALUATION_CACHE_SIZE
    assert (
        repr(adapted._evaluations)
        == f"EvaluationCache({_optmat.EVALUATION_CACHE_SIZE} of "
        f"{_optmat.EVALUATION_CACHE_SIZE})"
    )
    adapted._evaluations.clear()
    assert not len(adapted._evaluations)


def test_the_least_recently_used_band_is_evicted_first(monkeypatch):
    cache = _optmat.EvaluationCache(size=2)
    model = silica().model
    computed: list[float] = []

    def evaluate(omega):
        def compute():
            computed.append(float(omega[0]))
            return (omega * 2.0,)

        return cache.evaluate(
            "x",
            omega,
            model,
            material="SiO2",
            wavelength=_optmat.omega_to_wavelength(omega),
            compute=compute,
        )

    one, two, three = (np.array([value]) for value in (1.0, 2.0, 3.0))
    evaluate(one)
    evaluate(two)
    evaluate(one)  # refreshes one, so two is the oldest
    evaluate(three)
    assert computed == [1.0, 2.0, 3.0]
    evaluate(one)
    evaluate(two)
    assert computed == [1.0, 2.0, 3.0, 2.0]
    assert len(cache) == 2
    with pytest.raises(ValueError, match="at least one entry"):
        _optmat.EvaluationCache(size=0)


def test_a_refused_evaluation_is_translated_and_not_kept():
    cache = _optmat.EvaluationCache()
    model = silica().model

    def refuse():
        raise StateError("refused")

    with pytest.raises(MaterialError, match="SiO2: optmat refuses the index"):
        cache.evaluate(
            "index",
            OMEGA,
            model,
            material="SiO2",
            wavelength=_optmat.omega_to_wavelength(OMEGA),
            compute=refuse,
        )
    assert not len(cache)


def test_a_copied_or_pickled_adapter_starts_with_an_empty_cache():
    adapted = silica()
    adapted.index(OMEGA)
    assert len(adapted._evaluations) == 1
    for other in (copy.deepcopy(adapted), pickle.loads(pickle.dumps(adapted))):
        assert other._evaluations is not adapted._evaluations
        assert not len(other._evaluations)
        assert other._evaluations.size == adapted._evaluations.size
        assert same_bits(other.index(OMEGA), adapted.index(OMEGA))


def test_a_stored_band_warns_and_records_as_a_fresh_one_does():
    argon = OptmatSusceptibility.resolve("Ar")
    reference = OptmatSusceptibility.resolve("Ar")
    # Outside every context each evaluation warns, stored or not.
    with pytest.warns(MaterialRangeWarning, match="below its shortest") as caught:
        first = argon.susceptibility(BEYOND)
        second = argon.susceptibility(BEYOND)
    assert len(caught) == 2
    assert same_bits(first, second)
    assert len(argon._evaluations) == 1
    # Inside a context, once; a second context warns and records afresh.
    logs = []
    for adapter in (argon, argon, reference):
        log = MaterialLog()
        with (
            pytest.warns(MaterialRangeWarning) as caught,
            _optmat.material_context(log),
        ):
            for _ in range(3):
                adapter.susceptibility(BEYOND)
                adapter.dsusceptibility_domega(BEYOND)
        assert len(caught) == 1
        logs.append(log)
    held = [(log.seen, log.ranges) for log in logs]
    assert held[0] == held[1] == held[2]
    assert logs[0].ranges


def test_the_gas_susceptibility_and_its_derivative_are_kept_apart(monkeypatch):
    argon = OptmatSusceptibility.resolve("Ar")
    model = _optmat.untyped(argon.model)
    wavelength = _optmat.omega_to_wavelength(OMEGA)
    from solprop.constants import EPS0  # noqa: PLC0415 - the one test that needs it

    direct = (np.asarray(model.complex_polarizability(wavelength)) / EPS0).real
    slope = (np.asarray(model.dpolarizability_domega(wavelength)) / EPS0).real
    calls = counted(
        monkeypatch, type(model), "complex_polarizability", "dpolarizability_domega"
    )
    for _ in range(4):
        assert same_bits(argon.susceptibility(OMEGA), direct)
        assert same_bits(argon.dsusceptibility_domega(OMEGA), slope)
    assert calls == {"complex_polarizability": 1, "dpolarizability_domega": 1}


def test_a_transparent_index_records_its_dropped_loss_once_per_adapter():
    window = silica().as_transparent()
    edge = 2.0 * math.pi * C / np.array([800e-9, 120e-9])
    with pytest.warns(MaterialRangeWarning, match="beyond the transparent") as caught:
        first = window.index(edge)
        second = window.index(edge)
    assert len(caught) == 1
    assert same_bits(first, second)
    record = dict(window.transparency)
    fresh = silica().as_transparent()
    with pytest.warns(MaterialRangeWarning, match="beyond the transparent"):
        assert same_bits(fresh.index(edge), first)
    assert dict(fresh.transparency) == record
    assert record["bound_exceeded"]


def _gradient_capillary():
    """Return a short nitrogen gradient walled by optmat's silica, under a selection."""
    from optmat_parity_cases import _capillary, _gradient  # noqa: PLC0415 - tests/

    return _capillary(_gradient({"N2": 1.0}, (1.5, 0.3)), pressure=None)


@pytest.mark.medium
@pytest.mark.filterwarnings(
    f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
)
def test_a_gradient_translates_exactly_as_before_and_samples_its_wall_a_few_times(
    monkeypatch,
):
    """The frame tables read the mode at every z sample; its terms are sampled once.

    The unshared, uncached translation is the one before SEP 3's Q22: the scope is
    replaced by a null one and every adapter evaluation computes afresh. The shared
    scope and the cache reuse the results of the same functions of the same inputs, so
    every leaf is exactly the unshared one's: a structural contract, not arithmetic.
    The count of the wall's evaluations is the translation-time guard: it does not
    depend on the machine's load.
    """
    shared = _gradient_capillary()
    mode = shared.modes[0]
    assert isinstance(mode, MarcatiliMode)
    assert isinstance(mode.cladding_index, OptmatGlass)
    walls = counted(monkeypatch, type(mode.cladding_index.model), "refractive_index")
    problem = shared.translate()
    shared_walls = walls["refractive_index"]

    computes = {"count": 0}

    def uncached(self, quantity, omega, model, *, material, wavelength, compute):  # noqa: PLR0913 - the method it replaces
        computes["count"] += 1
        with _optmat.translated(
            model, quantity, material=material, wavelength=wavelength
        ):
            return compute()

    monkeypatch.setattr(_optmat.EvaluationCache, "evaluate", uncached)
    monkeypatch.setattr(problem_module, "_shared_sampling", contextlib.nullcontext)
    walls["refractive_index"] = 0
    unshared = _gradient_capillary()
    before = unshared.translate()
    assert walls["refractive_index"] > 100 * shared_walls
    assert shared_walls <= 4
    assert computes["count"] > 100
    after, reference = _leaves(problem), _leaves(before)
    assert after.keys() == reference.keys()
    assert len(after) > 100
    for path, value in reference.items():
        assert same_bits(after[path], value), path


def _leaves(value, path="", within=()):
    """Return every array and scalar reachable from a translated problem, by path."""
    if id(value) in within:
        return {}
    within = (*within, id(value))
    if isinstance(value, np.ndarray | int | float | complex | str | bool | None):
        return {path: np.asarray(value)}
    found = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            found |= _leaves(getattr(value, field.name), f"{path}.{field.name}", within)
    elif isinstance(value, dict):
        for key, item in value.items():
            found |= _leaves(item, f"{path}[{key!r}]", within)
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            found |= _leaves(item, f"{path}[{index}]", within)
    return found
