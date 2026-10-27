# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from bounds import gamma
from solprop.exceptions import OutputError
from solprop.processing import coherence

IDENTITY = gamma(64)
"""Rounding of a coherence that is one in exact arithmetic, relative.

``g12`` is a ratio of two sums over at most ``N^2 = 16`` pair products here, each a
complex product of a few roundings: one is a mathematical identity reached through
inexact arithmetic, so it is asserted to that budget rather than exactly (SEP 25).
"""


def test_identical_realisations_are_fully_coherent():
    one = np.array([1.0 + 0.5j, 2.0 - 1.0j, 0.25j])
    np.testing.assert_allclose(
        coherence.g12(np.stack([one] * 4)), np.ones(3), rtol=IDENTITY, atol=0.0
    )


def test_two_known_realisations_match_the_definition():
    first = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    second = np.array([1.0 + 0.0j, -1.0 + 0.0j])
    # With two realisations the definition is |2 Re(E1* E2)| / (|E1|^2 + |E2|^2), which
    # is 1 where they agree and 1 where they are exactly out of phase: the modulus of
    # the cross term, not its sign, is what g12 reports.
    assert np.allclose(coherence.g12(np.stack([first, second])), [1.0, 1.0])
    third = np.array([1.0 + 0.0j, 1.0j])
    # Orthogonal phasors have no cross term at all.
    assert coherence.g12(np.stack([first, third]))[1] == pytest.approx(0.0, abs=1e-15)


def test_independent_realisations_have_no_coherence():
    rng = np.random.default_rng(20260905)
    shape = (400, 32)
    ensemble = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    values = coherence.g12(ensemble)
    # The unbiased estimator has mean zero and a spread of order 1/sqrt(N); one that
    # keeps the i == j terms would sit at 1/N = 0.0025 plus that spread, so the
    # bound below separates the two definitions as well as testing this one.
    assert float(values.mean()) < 0.1
    assert np.all(values >= 0.0)


def test_a_partly_coherent_ensemble_lies_between():
    rng = np.random.default_rng(7)
    common = np.ones(16, dtype=complex)
    noise = rng.standard_normal((64, 16)) + 1j * rng.standard_normal((64, 16))
    values = coherence.g12(common + 0.5 * noise)
    assert 0.1 < float(values.mean()) < 0.95


def test_an_empty_bin_has_no_coherence():
    empty = np.zeros((3, 2), dtype=complex)
    empty[:, 0] = 1.0
    values = coherence.g12(empty)
    assert values[0] == pytest.approx(1.0, rel=IDENTITY, abs=0.0)
    assert np.isnan(values[1])


def test_g12_needs_two_realisations():
    with pytest.raises(OutputError, match="at least two realisations"):
        coherence.g12(np.ones((1, 4), dtype=complex))


def test_g12_of_reads_an_ensemble_of_runs(synthetic, tmp_path):
    memory = synthetic.build_run("envelope")
    ensemble = [memory, synthetic.as_file(memory, tmp_path / "run.h5")]
    omega, values = coherence.g12_of(ensemble, plane=-1)
    finite = np.isfinite(values)
    assert omega.shape == values.shape
    # The two handlers hold the same run, so the ensemble is perfectly coherent.
    assert np.any(finite)
    np.testing.assert_allclose(values[finite], 1.0, rtol=IDENTITY, atol=0.0)
    ensemble[1].close()


def test_g12_of_covers_every_plane_when_none_is_selected(synthetic):
    ensemble = [synthetic.build_run("envelope") for _ in range(2)]
    omega, values = coherence.g12_of(ensemble)
    assert values.shape == (synthetic.n_z, omega.size)


def test_g12_of_separates_two_different_runs(synthetic):
    ensemble = [
        synthetic.build_run("envelope", t0=0.0),
        synthetic.build_run("envelope", t0=60e-15),
    ]
    _, values = coherence.g12_of(ensemble, plane=0)
    # A delay is a linear spectral phase, so the pair decorrelates across the band as
    # the phase difference walks through 2 pi.
    assert float(np.nanmean(values)) < 0.9


def test_g12_of_needs_two_runs(synthetic):
    with pytest.raises(OutputError, match="at least two runs"):
        coherence.g12_of([synthetic.build_run("envelope")])


def test_g12_of_refuses_runs_of_different_shape(synthetic):
    ensemble = [
        synthetic.build_run("envelope", n_modes=1),
        synthetic.build_run("envelope", n_modes=2),
    ]
    with pytest.raises(OutputError, match="same shape"):
        coherence.g12_of(ensemble, plane=0)


def test_the_bands_split_the_occupied_span_equally():
    weights = np.zeros(70)
    weights[3:67] = 1.0
    bands = coherence.spectral_bands(weights, 32)
    assert len(bands) == 32
    assert bands[0][0] == 3
    assert bands[-1][-1] == 66
    assert {band.size for band in bands} == {2}
    assert coherence.spectral_bands(np.zeros(4)) == []


def test_band_coherence_weights_each_bin_by_the_reference_spectrum():
    rng = np.random.default_rng(1)
    n_x = 64
    reference = np.ones((8, n_x), dtype=complex)
    # Two bins per band: a coherent bin and an incoherent one whose reference weight
    # is three times larger, so the band mean is (1 * 1 + 3 * g) / 4.
    reference[:, 1::2] *= np.sqrt(3.0)
    phases = np.exp(2j * np.pi * rng.random((8, n_x)))
    spectra = np.ones((8, n_x), dtype=complex)
    spectra[:, 1::2] = phases[:, 1::2]
    per_bin = coherence.g12(spectra)
    values, energy = coherence.band_coherence(spectra, reference)
    expected = (per_bin[0::2] + 3.0 * per_bin[1::2]) / 4.0
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    np.testing.assert_allclose(energy, np.full(32, 4.0 * 8 / 8), rtol=1e-12)


def test_band_coherence_scores_only_bands_above_the_floor():
    reference = np.ones((4, 64), dtype=complex)
    reference[:, 32:] *= 1e-2  # -40 dB in energy
    values, energy = coherence.band_coherence(reference, reference)
    assert np.all(np.isfinite(values[:16]))
    assert np.all(np.isnan(values[16:]))
    assert energy[16] == pytest.approx(energy[0] * 1e-4)
    # A shape mismatch is refused, by the annotation's checker where it runs.
    with pytest.raises(Exception, match=r"shape|n_x"):
        coherence.band_coherence(reference, reference[:, :32])
