# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batched sampling in translation reproduces the interval-by-interval tables exactly.

The adaptive Chebyshev split resolves the intervals its walk is about to visit in
batches, and frozen capillary copies share their origin's frequency-only terms. Neither
may change a table: each test builds the same thing with and without them and compares
the tables' structure exactly and their Chebyshev coefficients to a few units in the
last place. The coefficients come from NumPy's FFT, which on some platforms (the Linux
aarch64 wheel) rounds a batch of transforms differently from the same transforms one at
a time.
"""

import contextlib
import dataclasses
import enum
import itertools
import warnings

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.driver.mixtures import capillary_fill
from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.modes import capillary as capillary_module
from solprop.profiles import PressureGradient
from solprop.translate import chebyshev
from solprop.translate import problem as problem_module
from solprop.translate.chebyshev import PiecewiseChebyshev, build
from solprop.translate.linop import build_phi

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]


def _unbatched(monkeypatch):
    """Resolve one interval at a time, as the split did before it batched."""
    monkeypatch.setattr(chebyshev, "_look_ahead", lambda *args, **kwargs: None)


def _unshared(monkeypatch):
    """Sample every mode's terms afresh: no frozen copy shares, no scope shares."""
    monkeypatch.setattr(capillary_module, "_same_rows", lambda first, second: False)
    monkeypatch.setattr(problem_module, "_shared_sampling", contextlib.nullcontext)


def _assert_coefficients_close(one, other, what: str):
    """Agree to 8 units in the last place of the array's largest coefficient."""
    assert one.dtype == other.dtype and one.shape == other.shape, what
    scale = float(np.max(np.abs(other), initial=0.0))
    atol = 8 * np.finfo(one.real.dtype).eps * scale
    np.testing.assert_allclose(one, other, rtol=0, atol=atol, err_msg=what)


def _assert_tables_equal(first: PiecewiseChebyshev, second: PiecewiseChebyshev):
    assert first.budget_exceeded == second.budget_exceeded
    _assert_coefficients_close(first.coefficients, second.coefficients, "coefficients")
    for name in ("breakpoints", "degrees", "hard_kinks"):
        one, other = getattr(first, name), getattr(second, name)
        assert one.dtype == other.dtype, name
        np.testing.assert_array_equal(one, other, err_msg=name, strict=True)


def _counting(f):
    calls = []

    def counted(z):
        calls.append(z.size)
        return f(z)

    return counted, calls


def _kinked(z):
    """A vector of three columns with a kink and a square-root entrance."""
    z = np.asarray(z, dtype=float)
    return np.stack(
        [np.abs(z - 0.31) ** 1.5, np.sqrt(z + 1e-3), np.exp(np.sin(7.0 * z))], axis=1
    )


def _complex(z):
    z = np.asarray(z, dtype=float)
    return np.stack(
        [np.exp(1j * 40.0 * z**2), (1.0 + 2j) * np.tanh(30.0 * (z - 0.5))], 1
    )


def _scalar_column(z):
    """One element per position, where NumPy's axis-0 sum is pairwise."""
    z = np.asarray(z, dtype=float)
    return np.abs(z - 0.43)[:, np.newaxis] ** 2.5


def _stepped(z):
    return 1.0 + 0.6 * z + np.where(z >= 0.4, 0.8, 0.0)


SAMPLERS = {
    "kinked": _kinked,
    "complex": _complex,
    "scalar_column": _scalar_column,
    "stepped": _stepped,
}


@pytest.mark.parametrize("name", sorted(SAMPLERS))
@pytest.mark.parametrize(
    "options",
    [{}, {"mandatory_breakpoints": (0.25, 0.7)}, {"tol": 1e-14, "max_intervals": 24}],
    ids=["default", "breakpoints", "exhausted"],
)
def test_batched_split_reproduces_the_interval_walk(name, options, monkeypatch):
    f = SAMPLERS[name]
    batched, batched_calls = _counting(f)
    table = build(batched, 0.0, 1.0, **options)
    with monkeypatch.context() as patch:
        _unbatched(patch)
        single, single_calls = _counting(f)
        reference = build(single, 0.0, 1.0, **options)
    _assert_tables_equal(table, reference)
    # Never more calls of the sampler than one per resolution round of each interval.
    assert len(batched_calls) <= len(single_calls)


@pytest.mark.parametrize("name", ["kinked", "complex"])
def test_batching_samples_nothing_extra_within_the_budget(name, monkeypatch):
    """Off the budget, the look-ahead resolves exactly the intervals the walk visits."""
    f = SAMPLERS[name]
    batched, batched_calls = _counting(f)
    table = build(batched, 0.0, 1.0, tol=1e-13)
    assert not table.budget_exceeded
    with monkeypatch.context() as patch:
        _unbatched(patch)
        single, single_calls = _counting(f)
        build(single, 0.0, 1.0, tol=1e-13)
    assert sum(batched_calls) == sum(single_calls)
    assert len(batched_calls) < len(single_calls)


def test_large_samples_are_not_batched_beyond_the_memory_bound(monkeypatch):
    """A sample this large fits once per batch at the highest degree: no look-ahead."""
    monkeypatch.setattr(chebyshev, "_BATCH_BYTES", 65 * 3 * 8)
    batched, batched_calls = _counting(_kinked)
    table = build(batched, 0.0, 1.0, tol=1e-12)
    with monkeypatch.context() as patch:
        _unbatched(patch)
        reference = build(_kinked, 0.0, 1.0, tol=1e-12)
    _assert_tables_equal(table, reference)
    assert max(batched_calls) <= 65


def test_the_sampler_cuts_its_calls_at_the_memory_bound(monkeypatch):
    """Groups are joined into calls of at most the bound, and split back exactly.

    The first group is sampled alone and fixes a sample's size; the rest are joined
    while their samples fit in :data:`chebyshev._BATCH_BYTES`: after the first of six
    groups of ten three-column samples, a bound of twenty samples takes the other five
    in three calls.
    """
    monkeypatch.setattr(chebyshev, "_BATCH_BYTES", 20 * 3 * 8)
    counted, calls = _counting(_kinked)
    sampler = chebyshev._BatchedSampler(counted)
    groups = [np.linspace(0.1 * k, 0.1 * k + 0.05, 10) for k in range(6)]
    sampled = sampler(groups)
    assert calls == [10, 20, 20, 10]
    assert len(sampled) == len(groups)
    for positions, values in zip(groups, sampled, strict=True):
        np.testing.assert_array_equal(values, _kinked(positions), strict=True)


def test_a_sampler_that_fails_fails_where_the_interval_walk_failed(monkeypatch):
    def failing(z):
        z = np.asarray(z, dtype=float)
        values = _kinked(z)
        values[z > 0.83] = np.nan
        return values

    with pytest.raises(TranslationError, match="non-finite") as batched:
        build(failing, 0.0, 1.0)
    with monkeypatch.context() as patch:
        _unbatched(patch)
        with pytest.raises(TranslationError, match="non-finite") as single:
            build(failing, 0.0, 1.0)
    assert str(batched.value) == str(single.value)


@pytest.mark.parametrize("shape", [(17, 1), (65, 1), (17, 4), (33, 2, 3)])
def test_the_batched_acceptance_test_is_the_one_interval_test_bitwise(shape):
    """Tails and noise floors themselves, not just the decisions they lead to."""
    rng = np.random.default_rng(25)
    edges = np.cumsum(rng.uniform(0.1, 1.0, 17))
    intervals = [(float(lo), float(hi)) for lo, hi in itertools.pairwise(edges)]
    # Samples spanning decades, so that summation order shows in the last bits.
    values = [
        rng.standard_normal(shape) * 10.0 ** rng.uniform(-6, 6, shape)
        for _ in intervals
    ]
    values[1] = np.zeros(shape)  # an identically zero series
    coefficients, ratios, noises = chebyshev._tails_many(values, intervals)
    for index, (samples, (lo, hi)) in enumerate(zip(values, intervals, strict=True)):
        expected = chebyshev.lobatto_coefficients(samples)
        np.testing.assert_array_equal(coefficients[index], expected, strict=True)
        assert ratios[index] == chebyshev._tail_ratio(expected)
        assert noises[index] == chebyshev._sampling_noise(expected, lo, hi)


# ------------------------------------------------------------- the mode level


@pytest.fixture(scope="module")
def grid():
    return FieldGrid.from_limits(0.02, 800e-9, (200e-9, 2e-6), 200e-15)


def _gradient_modes():
    fill = PressureGradient.two_point(0.02, 0.8, 0.2)
    return [MarcatiliMode(125e-6, "He", fill, m=m) for m in (1, 2)]


def test_a_frozen_copy_shares_its_origins_terms_bitwise():
    mode = _gradient_modes()[0]
    omega = np.linspace(1.0e15, 4.0e15, 64)
    values = {
        name: float(path(np.array([0.013]))[0])
        for name, path in mode.parameter_paths.items()
    }
    frozen = mode.at_parameters(values)
    assert frozen._terms_origin is mode
    assert frozen.at_parameters(values)._terms_origin is mode
    with capillary_module._shared_sampling():
        shared = frozen.sampled_terms(omega)
        assert mode.sampled_terms(omega) is shared
        own = frozen._sample_terms(omega.copy())
    for field in dataclasses.fields(shared):
        np.testing.assert_array_equal(
            getattr(shared, field.name), getattr(own, field.name), strict=True
        )


def test_a_frozen_copy_whose_rows_change_keeps_its_own_terms():
    """A mixture's population frozen to vacuum samples a zero row: it must not share."""
    fill = capillary_fill(
        "He",
        (0.8, 0.0),
        length=0.02,
        temperature=None,
        floor=None,
        susceptibility=None,
        preionisation=None,
    )
    mode = MarcatiliMode(125e-6, mixture=fill)
    assert set(mode.parameter_paths) == {"core_radius", "density"}
    filled = {"core_radius": 100e-6, "density": 1e24}
    assert mode.at_parameters(filled)._terms_origin is mode
    assert mode.at_parameters(filled | {"density": 0.0})._terms_origin is None


def _walk(value, path, out, seen):
    if id(value) in seen:
        return
    if isinstance(value, np.ndarray):
        out[path] = value
        return
    if isinstance(value, (int, float, complex, str, bool, type(None), enum.Enum)):
        out[path] = value
        return
    if isinstance(value, np.generic):
        out[path] = value.item()
        return
    seen.add(id(value))
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _walk(getattr(value, field.name), f"{path}.{field.name}", out, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk(item, f"{path}[{key!r}]", out, seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", out, seen)


def _assert_problems_equal(first, second):
    one, other = {}, {}
    _walk(first, "problem", one, set())
    _walk(second, "problem", other, set())
    assert one.keys() == other.keys()
    for key, value in one.items():
        if isinstance(value, np.ndarray) and key.endswith(".coefficients"):
            _assert_coefficients_close(value, other[key], key)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other[key], err_msg=key, strict=True)
        elif value == value:  # noqa: PLR0124 - NaN scalars compare unequal to themselves
            assert value == other[key], key


def _capillary(**changes):
    parameters = {
        "radius": 125e-6,
        "length": 0.02,
        "gas": "He",
        "pressure": (0.8, 0.3),
        "lambda0": 800e-9,
        "fwhm": 10e-15,
        "energy": 50e-6,
        "trange": 200e-15,
        "wavelength_limits": (150e-9, 2e-6),
        "n_save": 3,
    } | changes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Simulation.from_capillary(
            parameters.pop("radius"),
            parameters.pop("length"),
            parameters.pop("gas"),
            parameters.pop("pressure"),
            **{"noise": None, **parameters},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"modes": 3},
        # A shallow gradient whose power-weight table splits to its budget: the
        # look-ahead's batches and its spare resolutions are both exercised.
        {"pressure": (0.8, 0.7947), "modes": 2},
        {"pressure": 1.5, "gas": "Ar", "radius": 50e-6, "modes": 2},
    ],
    ids=["gradient", "gradient-modes", "shallow-gradient-modes", "constant-modes"],
)
def test_translation_reproduces_the_unbatched_unshared_tables(changes, monkeypatch):
    problem = _capillary(**changes).translate()
    with monkeypatch.context() as patch:
        _unbatched(patch)
        _unshared(patch)
        reference = _capillary(**changes).translate()
    _assert_problems_equal(problem, reference)


class _UserMode:
    """A user's mode: a capillary behind the public scalar-z interface alone."""

    scale_invariant = True
    azimuthal_order = 0

    def __init__(self, inner):
        self._inner = inner
        self.positions = []

    @property
    def z_constant(self):
        return False

    @property
    def channel_label(self):
        return self._inner.channel_label

    def principal_component(self, z=0.0):
        return self._inner.principal_component(z)

    def neff(self, omega, z=0.0):
        self.positions.append(z)
        return self._inner.neff(omega, z)

    def neff_minus_one(self, omega, z=0.0):
        self.positions.append(z)
        return self._inner.neff_minus_one(omega, z)

    def dneff_domega(self, omega, z=0.0):
        self.positions.append(z)
        return self._inner.dneff_domega(omega, z)

    def aeff(self, z=0.0):
        self.positions.append(z)
        return self._inner.aeff(z)

    def dimlimits(self, *args, **kwargs):
        return self._inner.dimlimits(*args, **kwargs)

    def field(self, *args, **kwargs):
        return self._inner.field(*args, **kwargs)


def test_a_user_mode_keeps_the_scalar_path(grid, monkeypatch):
    mode = _UserMode(_gradient_modes()[0])
    frame = Frame.comoving(grid)
    table = build_phi(mode, grid, frame, z_max=0.02)
    assert mode.positions
    assert all(isinstance(z, float) for z in mode.positions)
    with monkeypatch.context() as patch:
        _unbatched(patch)
        reference = build_phi(_UserMode(_gradient_modes()[0]), grid, frame, z_max=0.02)
    for name in ("phi_coefficients", "phi_breakpoints", "phi_degrees"):
        np.testing.assert_array_equal(
            getattr(table, name), getattr(reference, name), strict=True
        )
