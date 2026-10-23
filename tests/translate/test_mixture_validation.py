# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Refuse ambiguous labels and inconsistent schema-six component declarations."""

import dataclasses

import numpy as np
import pytest

from mixture_cases import mixture_case
from solprop.backends.numpy_ref import passes
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Mixture
from solprop.translate.linop import tabulate
from solprop.translate.problem import Flags, assemble
from solprop.translate.responses import compile_responses


@pytest.fixture
def case():
    return mixture_case()


def compile_for(case, responses):
    _, grid, mode, _, _ = case
    return compile_responses(
        responses, grid, mode, density=mode.density, z_max=grid.length
    )


@pytest.mark.parametrize("damage", ["bare", "unknown", "duplicate", "thg"])
def test_kerr_bindings_require_unique_known_labels_and_one_drive(case, damage):
    kerr = [r for r in case[3] if isinstance(r, KerrResponse)]
    if damage == "bare":
        kerr[0] = dataclasses.replace(kerr[0], species=None)
        message = "needs a component label"
    elif damage == "unknown":
        kerr[0] = dataclasses.replace(kerr[0], species="absent")
        message = "unknown response component label"
    elif damage == "duplicate":
        kerr[1] = kerr[0]
        message = "same species"
    else:
        kerr[1] = dataclasses.replace(kerr[1], thg=True)
        message = "agree on thg"
    with pytest.raises(TranslationError, match=message):
        compile_for(case, kerr)


def test_unique_plasma_species_can_bind_without_an_explicit_label(case):
    plasma = tuple(
        dataclasses.replace(r, species=None)
        for r in case[3]
        if isinstance(r, PlasmaResponse)
    )
    compiled = compile_for(case, plasma)
    assert (
        tuple(channel.label for channel in compiled.plasma_channels) == case[0].labels
    )
    # A rate bound to an absent species is ambiguous even if its numerical table
    # happens to be identical to a declared channel's table.
    unknown = dataclasses.replace(plasma[0], gas="Ar")
    with pytest.raises(TranslationError, match="needs a component label"):
        compile_for(case, [unknown])


def test_duplicate_raman_components_are_refused(case):
    response = RamanResponse(
        (DampedOscillator(1e-37, 2e14, undamped_dephasing()),),
        thg=False,
        species=case[0].labels[0],
    )
    with pytest.raises(TranslationError, match="same species"):
        compile_for(case, [*case[3], response, response])


@pytest.mark.parametrize("damage", ["empty_label", "electron_count", "rate_count"])
def test_channel_contract_checks_metadata_and_table_counts(case, damage):
    problem = case[-1]
    channel = problem.responses.plasma_channels[0]
    with pytest.raises(TranslationError, match=r"label|electrons_per_event|one rate"):
        if damage == "empty_label":
            dataclasses.replace(channel, label="")
        elif damage == "electron_count":
            dataclasses.replace(channel, electrons_per_event=0.0)
        else:
            responses = dataclasses.replace(problem.responses, ionisation_table=())
            dataclasses.replace(problem, responses=responses).validate()


def test_rate_at_zero_is_checked_even_for_a_broken_rate_object(case, monkeypatch):
    problem = case[-1]
    rate = problem.responses.plasma_channels[0].rate
    monkeypatch.setattr(type(rate), "__call__", lambda self, field: 1.0)
    with pytest.raises(TranslationError, match="zero at zero field"):
        problem.validate()


@pytest.mark.parametrize("weights", [np.zeros(2), np.array([-1.0]), np.array([np.nan])])
def test_physical_modal_weights_are_validated(case, weights):
    with pytest.raises(TranslationError, match="node_weights"):
        dataclasses.replace(case[-1].transform, node_weights=weights)


def test_multi_channel_pass_cannot_fall_back_to_one_unlabelled_density(case):
    problem = case[-1]
    with pytest.raises(BackendError, match="labelled stage scalars"):
        passes.nonlinear_polarisation(
            np.zeros((1, 1, problem.grids.n_to)),
            problem.responses,
            np.ones(problem.grids.n_to),
            kind="field",
            density=1.0,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
        )


@pytest.fixture
def raman_problem(case):
    extra = tuple(
        RamanResponse(
            (DampedOscillator(1e-37, 2e14, undamped_dephasing()),),
            thg=False,
            species=component.label,
        )
        for component in case[0].components
    )
    compiled = compile_for(case, (*case[3], *extra))
    original = case[-1]
    return dataclasses.replace(
        original,
        responses=compiled,
        flags=Flags.from_content(
            shape=original.shape,
            linear=original.linear,
            transform=original.transform,
            responses=compiled,
            initial=original.initial,
        ),
    )


@pytest.mark.parametrize("damage", ["label", "shape", "density", "thg"])
def test_each_raman_component_is_validated(raman_problem, damage):
    terms = list(raman_problem.responses.raman_terms)
    change = {
        "label": {"label": "absent"},
        "shape": {"undamped": terms[0].undamped[:, :-1]},
        "density": {"density": None},
        "thg": {"thg": True},
    }[damage]
    terms[0] = dataclasses.replace(terms[0], **change)
    problem = dataclasses.replace(
        raman_problem,
        responses=dataclasses.replace(
            raman_problem.responses, raman_terms=tuple(terms)
        ),
    )
    with pytest.raises(TranslationError, match=r"Raman|raman_terms"):
        problem.validate()


def test_raman_drive_agreement_is_checked_on_an_envelope_too(raman_problem):
    terms = raman_problem.responses.raman_terms
    problem = dataclasses.replace(
        raman_problem,
        grid_kind="envelope",
        responses=dataclasses.replace(
            raman_problem.responses,
            raman_terms=(dataclasses.replace(terms[0], thg=True), terms[1]),
        ),
    )
    with pytest.raises(TranslationError, match="agree on thg"):
        problem._check_raman()


@pytest.mark.parametrize("damage", ["labels", "kerr_density", "total_shape"])
def test_species_metadata_and_density_shapes_are_validated(case, damage):
    original = case[-1]
    responses = original.responses
    if damage == "labels":
        responses = dataclasses.replace(responses, labels=("", ""))
    elif damage == "kerr_density":
        responses = dataclasses.replace(
            responses,
            kerr_coefficients=dataclasses.replace(
                responses.kerr_coefficients, density=None
            ),
        )
    else:
        responses = dataclasses.replace(
            responses,
            density=tabulate(lambda z: np.ones((len(z), 2)), *original.linear.z_span),
        )
    problem = dataclasses.replace(original, responses=responses)
    with pytest.raises(TranslationError, match=r"labels|density|scalar table"):
        problem._check_species()


def test_explicit_response_density_cannot_override_a_mixture(case):
    _, grid, mode, responses, original = case
    with pytest.raises(TranslationError, match=r"density.*mixture|mixture.*density"):
        assemble(
            grid,
            mode,
            responses,
            original.initial.field0,
            controller=original.stepper.controller,
            save_planes=[grid.length],
            dz_max=1e-4,
            dz_init=1e-5,
            density=1.0,
        )


@pytest.mark.parametrize("kind", ["kerr", "raman"])
def test_bare_and_explicit_aliases_cannot_duplicate_one_population(case, kind):
    component = case[0].components[0]
    mixture = Mixture.of(component)
    mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
    kerr = KerrResponse(1e-52, thg=False)
    if kind == "kerr":
        responses = (kerr, dataclasses.replace(kerr, species=component.label))
    else:
        raman = RamanResponse(
            (DampedOscillator(1e-37, 2e14, undamped_dephasing()),), thg=False
        )
        responses = (kerr, raman, dataclasses.replace(raman, species=component.label))
    with pytest.raises(TranslationError, match="same species"):
        compile_responses(
            responses, case[1], mode, density=mode.density, z_max=case[1].length
        )


def test_compiler_requires_one_raman_drive_on_an_envelope(case):
    grid = EnvelopeGrid.from_limits(
        case[1].length, 800e-9, (600e-9, 1.2e-6), 80e-15, thg=False
    )
    responses = tuple(
        RamanResponse(
            (DampedOscillator(1e-37, 2e14, undamped_dephasing()),),
            thg=bool(index),
            species=label,
        )
        for index, label in enumerate(case[0].labels)
    )
    with pytest.raises(TranslationError, match="agree on thg"):
        compile_responses(
            responses, grid, case[2], density=case[2].density, z_max=grid.length
        )
