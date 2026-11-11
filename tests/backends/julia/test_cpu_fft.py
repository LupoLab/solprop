# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise managed FFT publication and reuse through complete start transactions."""

import os

import pytest

from backends.julia.cases import cpu_case
from solprop.backends.julia._fft_cache import forget_fft_cache
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.exceptions import BackendError

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _owner(case):
    return JuliaBackend(
        case.problem,
        case.request,
        options=ExecutionOptions(
            backend_options={"setup_id": os.environ["SOLPROP_TEST_JULIA_SETUP_ID"]}
        ),
    )


def test_repeated_prepared_starts_offer_the_previous_successful_measurements(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOLPROP_FFTW_WISDOM", str(tmp_path / "plans"))
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "measure")
    forget_fft_cache()
    case = cpu_case()
    owner = _owner(case)
    try:
        prepared = owner.prepare(case.problem, case.request)
        with prepared.start(Start.entrance(case.problem), owns=False) as session:
            initial = session.execution_record()
            assert initial["fft_cache_written"] is True
            assert initial["fft_cache_supplied"] is False
            assert initial["fft_plan_effort"] == "measure"
            assert initial["fft_plan_rejections"] == {}
        assert owner._fft_cache is not None and owner._fft_cache.path is not None
        assert owner._fft_cache.path.is_file()
        with prepared.start(Start.entrance(case.problem), owns=False) as session:
            later = session.execution_record()
            assert later["fft_cache_supplied"] is True
            assert later["fft_plan_fingerprint"] == initial["fft_plan_fingerprint"]
            assert later["fft_plan_rejections"] == {}
            # Exact reuse is keyed by the new allocations' actual alignment;
            # different alignment can legitimately require another measurement.
            assert later["fft_plans_applied"] == initial["fft_plans_applied"]
    finally:
        owner.close()
        forget_fft_cache()


def test_refused_numerical_start_does_not_publish_planning_measurements(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOLPROP_FFTW_WISDOM", str(tmp_path / "plans"))
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "measure")
    forget_fft_cache()
    case = cpu_case()
    owner = _owner(case)
    try:
        prepared = owner.prepare(case.problem, case.request)
        field = case.problem.initial.field0
        start = Start(field, 1.0)
        with pytest.raises(BackendError):
            prepared.start(start, owns=False)
        assert start.field is field
        assert owner._fft_cache is not None and owner._fft_cache.path is not None
        assert not owner._fft_cache.path.exists()
        assert not owner._sessions
        with prepared.start(Start.entrance(case.problem), owns=False) as session:
            assert session.execution_record()["fft_cache_written"] is True
    finally:
        owner.close()
        forget_fft_cache()
