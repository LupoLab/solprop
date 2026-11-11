# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify supplied input-noise records without loading an execution engine."""

import gc
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from backends.julia.cases import cpu_case, material_case
from solprop.backends.julia._noise import lower_input_noise
from solprop.exceptions import BackendError, TranslationError
from solprop.noise import NoiseRealisation
from solprop.translate.masks import noise_spectrum
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Initial


@pytest.fixture(scope="module")
def problem():
    return cpu_case().problem


def _with_noise(
    problem,
    spectrum=None,
    *,
    model: Literal["input", "modified"] = "input",
    launch=0.0,
):
    if spectrum is None:
        spectrum = np.full(problem.shape, 0.125 - 0.25j, dtype=np.complex128)
    return replace(
        problem,
        initial=Initial(problem.initial.field0, spectrum, model, launch),
        flags=replace(problem.flags, has_noise=True),
    )


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_descriptor_draw_keeps_units_layout_and_absolute_launch(kind):
    case, _, _ = material_case(kind, "kerr")
    n_omega = case.problem.shape[-1]
    draw = (np.arange(n_omega) / 8 + 0.25j).reshape(1, 1, n_omega)
    source = _with_noise(case.problem, draw, launch=2e-6)
    prepared = lower_input_noise(source)
    assert prepared is not None
    assert prepared.launch == 2e-6
    assert prepared.spectrum.shape == (n_omega, 1)
    assert prepared.spectrum.dtype == np.complex128
    assert prepared.spectrum.flags.f_contiguous
    assert prepared.spectrum.flags.owndata
    assert not prepared.spectrum.flags.writeable
    # Layout/copy is a structural operation: no normalization or phase arithmetic.
    np.testing.assert_array_equal(prepared.spectrum, draw.reshape(-1, n_omega).T)
    assert not np.shares_memory(prepared.spectrum, source.initial.noise0)


@pytest.mark.parametrize("override_model", ["input", "modified"])
@pytest.mark.parametrize("draw_shape", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_override_broadcasts_components_inside_modes_without_switching_policy(
    problem, override_model, draw_shape
):
    n_omega = problem.shape[-1]
    shape = (2, 2, n_omega)
    field = np.zeros(shape, dtype=np.complex128)
    source = replace(
        problem,
        shape=shape,
        initial=Initial(field, np.ones_like(field), "input"),
    )
    values = np.arange(np.prod((*draw_shape, n_omega)), dtype=float)
    values = (values / 16 + 0.5j).reshape(*draw_shape, n_omega)
    replacement = NoiseRealisation(values, 17, override_model)
    prepared = lower_input_noise(source, noise=replacement)
    assert prepared is not None
    reference = noise_spectrum(replacement, shape)
    np.testing.assert_array_equal(prepared.spectrum, reference.reshape(-1, n_omega).T)
    assert not np.shares_memory(prepared.spectrum, replacement.spectrum)
    assert not np.shares_memory(prepared.spectrum, values)
    assert not prepared.spectrum.flags.writeable
    values[:] = 0
    np.testing.assert_array_equal(prepared.spectrum, reference.reshape(-1, n_omega).T)


@pytest.mark.parametrize("single", [False, True])
def test_inactive_model_ignores_nonfinite_incompatible_override(problem, single):
    source = replace(problem, precision=SINGLE) if single else problem
    ignored = NoiseRealisation(np.full((7, 3, 2), complex(np.inf, 0)), 3, "modified")
    assert lower_input_noise(source, noise=ignored) is None
    assert lower_input_noise(source) is None


@pytest.mark.parametrize("override", [False, True])
def test_modified_active_policy_refuses_before_selecting_the_draw(problem, override):
    source = _with_noise(problem, model="modified")
    replacement = (
        NoiseRealisation(np.zeros(problem.shape, dtype=complex), 1, "input")
        if override
        else None
    )
    with pytest.raises(BackendError, match=r"julia.*modified"):
        lower_input_noise(source, noise=replacement)


def test_active_binary32_noise_is_refused_by_name(problem):
    source = replace(_with_noise(problem), precision=SINGLE)
    with pytest.raises(BackendError, match=r"julia.*noise.*binary64"):
        lower_input_noise(source)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, complex(0, np.inf)])
@pytest.mark.parametrize("override", [False, True])
def test_nonfinite_draws_are_refused_before_publication(problem, value, override):
    bad = np.full(problem.shape, value, dtype=np.complex128)
    if override:
        source = _with_noise(problem)
        replacement = NoiseRealisation(bad, None, "input")
    else:
        source = _with_noise(problem, bad)
        replacement = None
    with pytest.raises(BackendError, match=r"julia.*noise.*finite"):
        lower_input_noise(source, noise=replacement)


def test_override_shape_failure_preserves_shared_cause(problem):
    source = _with_noise(problem)
    replacement = NoiseRealisation(np.zeros((3, 1, 5), dtype=complex), None, "input")
    with pytest.raises(BackendError, match=r"julia.*noise.*shape") as failure:
        lower_input_noise(source, noise=replacement)
    assert isinstance(failure.value.__cause__, TranslationError)


def test_mismatched_descriptor_draw_is_refused_without_a_published_record(problem):
    # An independently replaced shape is not a validated PropagationProblem; the
    # lowering boundary still refuses it before returning an engine input record.
    source = replace(_with_noise(problem), shape=(2, 1, problem.shape[-1]))
    with pytest.raises(BackendError, match=r"julia.*noise.*shape"):
        lower_input_noise(source)


def test_result_is_frozen_independent_and_does_not_retain_input_owners():
    problem = _with_noise(cpu_case().problem, launch=0.01)
    replacement = NoiseRealisation(np.zeros(problem.shape, dtype=complex), 23, "input")
    problem_ref, replacement_ref = weakref.ref(problem), weakref.ref(replacement)
    prepared = lower_input_noise(problem, noise=replacement)
    second = lower_input_noise(problem, noise=replacement)
    assert prepared is not None and second is not None
    assert prepared is not second
    assert not np.shares_memory(prepared.spectrum, second.spectrum)
    with pytest.raises(FrozenInstanceError):
        prepared.__setattr__("launch", 0.0)
    with pytest.raises(ValueError, match="read-only"):
        prepared.spectrum[0, 0] = 1
    del problem, replacement
    gc.collect()
    assert problem_ref() is None and replacement_ref() is None
    assert prepared.launch == 0.01
    assert not np.any(prepared.spectrum)


def test_lowering_draws_no_random_numbers(problem, monkeypatch):
    def refuse_rng(*args, **kwargs):
        raise AssertionError("lowering must not draw new noise")

    monkeypatch.setattr(np.random, "default_rng", refuse_rng)
    prepared = lower_input_noise(_with_noise(problem))
    assert prepared is not None
    np.testing.assert_array_equal(prepared.spectrum, 0.125 - 0.25j)


def test_fresh_process_lowers_without_native_or_julia_import(monkeypatch, tmp_path):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class RefuseEngines(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                engines = ('solprop_native', 'juliacall', 'juliapkg')
                if fullname.split('.')[0] in engines:
                    raise AssertionError('execution engine imported: ' + fullname)
        sys.meta_path.insert(0, RefuseEngines())
        from dataclasses import replace
        import numpy as np
        from backends.julia.cases import cpu_case
        from solprop.backends.julia._noise import lower_input_noise
        from solprop.translate.problem import Initial
        problem = cpu_case().problem
        source = replace(problem, initial=Initial(problem.initial.field0,
                         np.zeros(problem.shape, dtype=complex), 'input'))
        lowered = lower_input_noise(source)
        assert lowered is not None and not np.any(lowered.spectrum)
    """)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
