# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Refuse unsupported TDSE declarations and tampered seam records before setup."""

import dataclasses

import numpy as np
import pytest

from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import DispersiveKerrResponse, KerrResponse
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.physdata import Component, Mixture, species
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.profiles import Constant, Linear
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.translate.masks import AbsorberScheme
from solprop.translate.problem import assemble
from solprop.translate.responses import (
    ResponseStage,
    compile_responses,
    kerr_coefficients,
)
from solprop.translate.tdse import TDSEDescriptor, refine_response_grid
from solprop.translate.tdse_source import admit_tdse


@pytest.fixture
def declaration():
    # The native source is gated by the grid collar, which this short 8 fs lattice
    # resolves only refined: q = 64 certifies it.
    grid = refine_response_grid(
        FieldGrid.from_limits(0.001, 800e-9, (500e-9, 1e-6), 8e-15), TDSESampling(64)
    )
    mode = MarcatiliMode(125e-6, "He", 0.1, cladding=1.45, loss=False)
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 3, 1e-16, source="admission fixture"
    )
    calibration = TDSECalibration(
        "admission-fixture",
        "research",
        basis.identity,
        (500e-9, 1e-6),
        source="admission fixture",
        limitations=("no atomic qualification",),
    )
    response = TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling(64)
    )
    return grid, mode, response


def compile_declaration(
    declaration, *, density=None, absorber_scheme: AbsorberScheme = "nonlinear"
):
    grid, mode, response = declaration
    return assemble(
        grid,
        mode,
        [response],
        np.zeros((1, 1, grid.n_omega), dtype=np.complex128),
        controller=PIController(rtol=1e-2, atol=1e-12),
        save_planes=[0.001],
        dz_init=1e-5,
        dz_max=1e-4,
        density=density,
        absorber_scheme=absorber_scheme,
        quadrature=transverse_quadrature([mode], n_r=4),
        components=("x",),
        integrator=Multistep(formulation="interaction"),
        stats_spec=StatsSpec(names=("energy",)),
    )


@pytest.fixture
def admitted(declaration):
    return compile_declaration(declaration)


def test_legacy_helium_mode_refuses_a_separate_nonlinear_density(declaration):
    _, mode, _ = declaration
    assert mode.mixture is None
    with pytest.raises(TranslationError, match=r"density.*helium fill"):
        compile_declaration(declaration, density=1.0)


def test_admitted_native_problem_needs_no_optional_provider(admitted):
    admitted.validate()
    assert admitted.tdse is not None
    assert admitted.tdse.shadow_order == 1
    assert admitted.responses.kerr_coefficients is None


@pytest.mark.parametrize("z", [-1.0, np.nan, np.inf])
def test_initial_seam_refuses_invalid_absolute_positions(admitted, z):
    with pytest.raises(TranslationError, match=r"initial.z.*non-negative"):
        dataclasses.replace(admitted.initial, z=z)


@pytest.mark.parametrize("missing", ["tdse", "current_numerator"])
def test_seam_requires_both_atomic_descriptor_and_current_factor(admitted, missing):
    with pytest.raises(TranslationError, match="must coexist"):
        dataclasses.replace(admitted, **{missing: None}).validate()


@pytest.mark.parametrize("periodic", [False, True])
def test_manual_atomic_seam_refuses_legacy_plasma_statistics(admitted, periodic):
    stats = StatsSpec(
        names=("energy", "electron_density"),
        periodic=("electron_density",) if periodic else (),
    )
    with pytest.raises(TranslationError, match="TDSE does not define legacy"):
        dataclasses.replace(admitted, stats_spec=stats).validate()


def test_current_factor_must_be_read_only_even_when_numerically_correct(admitted):
    assert admitted.current_numerator is not None
    current = admitted.current_numerator.copy()
    assert current.flags.writeable
    with pytest.raises(TranslationError, match="frozen finite source row"):
        dataclasses.replace(admitted, current_numerator=current).validate()


@pytest.mark.parametrize("offset", [0.0, 1e-5])
def test_manual_seam_refuses_save_planes_at_or_before_entrance(admitted, offset):
    initial = dataclasses.replace(
        admitted.initial, z=admitted.stepper.save_planes[0] + offset
    )
    with pytest.raises(TranslationError, match="save planes must follow"):
        dataclasses.replace(admitted, initial=initial).validate()


def test_atomic_assembly_refuses_a_split_step_collar(declaration):
    with pytest.raises(TranslationError, match="in-stage collar"):
        compile_declaration(declaration, absorber_scheme="split_step")


@pytest.mark.parametrize("change", ["samples", "spacing"])
def test_seam_rejects_a_descriptor_for_a_different_response_lattice(admitted, change):
    grids = dataclasses.replace(
        admitted.grids,
        **(
            {
                "n_to": admitted.grids.n_to * 2,
                "t_window_fine": np.repeat(admitted.grids.t_window_fine, 2),
                "t_window_fine_derivative": np.repeat(
                    admitted.grids.t_window_fine_derivative, 2
                ),
            }
            if change == "samples"
            else {"dt_fine": admitted.grids.dt_fine * 2}
        ),
    )
    with pytest.raises(TranslationError, match="sampling differs"):
        dataclasses.replace(admitted, grids=grids).validate()


def test_seam_rejects_a_descriptor_resolved_on_another_refinement(
    admitted, declaration
):
    grid, _, response = declaration
    refined = refine_response_grid(grid, TDSESampling(2))
    descriptor = TDSEDescriptor.from_response(
        dataclasses.replace(response, sampling=TDSESampling(2)),
        refined,
    )
    with pytest.raises(TranslationError, match="sampling differs"):
        dataclasses.replace(admitted, tdse=descriptor).validate()


@pytest.mark.parametrize("scale", [0.0, 0.5, -1.0])
def test_seam_rejects_a_changed_current_sign_or_crop_factor(admitted, scale):
    assert admitted.current_numerator is not None
    changed = admitted.current_numerator * scale
    changed.setflags(write=False)
    with pytest.raises(
        TranslationError, match=r"current numerator.*physical coefficient"
    ):
        dataclasses.replace(admitted, current_numerator=changed).validate()


@pytest.mark.parametrize(
    "flag", ["has_noise", "has_plasma", "has_raman", "mode_averaged"]
)
def test_seam_rejects_unsupported_flags_before_atomic_setup(admitted, flag):
    damaged = dataclasses.replace(
        admitted, flags=dataclasses.replace(admitted.flags, **{flag: True})
    )
    with pytest.raises(TranslationError, match="scalar resolved modal Adams"):
        damaged.validate()


@pytest.mark.parametrize("interpolate", [False, True], ids=["no-adams", "dense"])
def test_seam_requires_adams_and_landed_save_planes(admitted, interpolate):
    damaged = dataclasses.replace(
        admitted,
        stepper=dataclasses.replace(
            admitted.stepper,
            integrator=Multistep(formulation="interaction") if interpolate else None,
            interpolate_planes=interpolate,
        ),
    )
    with pytest.raises(TranslationError, match="scalar resolved modal Adams"):
        damaged.validate()


def test_seam_refuses_split_step_absorption(admitted):
    damaged = dataclasses.replace(
        admitted,
        responses=dataclasses.replace(admitted.responses, absorber_scheme="split_step"),
    )
    with pytest.raises(TranslationError, match="scalar resolved modal Adams"):
        damaged.validate()


@pytest.mark.parametrize("model", ["tdse_native", "tdse_shadow"])
def test_atomic_cubic_models_refuse_an_extra_compiled_kerr(
    admitted, declaration, model
):
    grid, _, response = declaration
    response = dataclasses.replace(
        response,
        kerr_model=model,
        cubic_scale=2.0 if model == "tdse_shadow" else None,
    )
    ordinary = dataclasses.replace(
        admitted.responses,
        kerr_coefficients=kerr_coefficients(KerrResponse(chi3=1e-50, thg=True), grid),
        response_order=(ResponseStage.KERR, ResponseStage.ACCUMULATE),
    )
    damaged = dataclasses.replace(
        admitted,
        tdse=TDSEDescriptor.from_response(response, grid),
        responses=ordinary,
    )
    with pytest.raises(TranslationError, match=r"cubic|Kerr"):
        damaged.validate()


def test_optmat_closure_refuses_an_absent_compiled_cubic(admitted, declaration):
    grid, _, response = declaration
    replacement = dataclasses.replace(
        response, kerr_model="optmat_cubic", remainder_scale=2.0
    )
    damaged = dataclasses.replace(
        admitted,
        tdse=TDSEDescriptor.from_response(replacement, grid),
    )
    with pytest.raises(TranslationError, match=r"cubic|Kerr"):
        damaged.validate()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"responses": "duplicate"}, "owns"),
        ({"shape": (2, 1, 1, 16)}, "unbatched"),
        ({"modes": "multiple"}, "resolved HE_11"),
        ({"modes": "rotated"}, "x-polarised"),
        ({"modes": "tapered"}, "tapered"),
        ({"modes": "argon"}, "helium fill"),
        ({"modes": "mixture"}, "gas mixtures"),
        ({"modes": "preionised"}, "preionised"),
        ({"modes": "custom"}, "selected optmat"),
        ({"quadrature": None}, "radial quadrature"),
        ({"integrator": None}, "Multistep"),
        ({"interpolate_planes": True}, "landed"),
        ({"has_noise": True}, "noise=None"),
        ({"shock": False}, "normalisation"),
        ({"prefactor_index": 1.0}, "normalisation"),
    ],
)
def test_unsupported_physical_routes_fail_before_operator_setup(
    declaration, change, message
):
    grid, mode, response = declaration
    options = {
        "responses": (response,),
        "grid": grid,
        "modes": (mode,),
        "shape": (1, 1, 1, grid.n_omega),
        "quadrature": transverse_quadrature([mode], n_r=4),
        "integrator": Multistep(formulation="interaction"),
        "has_noise": False,
        "shock": True,
        "prefactor_index": None,
    }
    change = dict(change)
    kind = change.get("modes")
    if kind == "multiple":
        change["modes"] = (mode, mode)
    elif kind == "rotated":
        change["modes"] = (dataclasses.replace(mode, phi=0.5),)
    elif kind == "tapered":
        change["modes"] = (
            MarcatiliMode(
                Linear(125e-6, 126e-6, grid.length), "He", 0.1, cladding=1.45
            ),
        )
    elif kind == "argon":
        change["modes"] = (MarcatiliMode(125e-6, "Ar", 0.1, cladding=1.45),)
    elif kind in ("mixture", "preionised", "custom"):
        material = species("He")
        if kind == "custom":
            material = dataclasses.replace(material, kerr=None)
        components = [
            Component(
                material,
                Constant(1e23),
                preionisation=(0.01 if kind == "preionised" else 0.0),
            )
        ]
        if kind == "mixture":
            components.append(Component(species("Ar"), Constant(1e23)))
        change["modes"] = (
            MarcatiliMode(125e-6, mixture=Mixture.of(*components), cladding=1.45),
        )
    if change.get("responses") == "duplicate":
        change["responses"] = (response, KerrResponse(chi3=1e-50))
    options.update(change)
    with pytest.raises(TranslationError, match=message):
        admit_tdse(**options)


@pytest.mark.parametrize("model", ["tdse_native", "optmat_cubic", "ordinary"])
def test_admission_preserves_the_selected_cubic_ownership(declaration, model):
    grid, mode, response = declaration
    if model == "optmat_cubic":
        response = dataclasses.replace(response, kerr_model=model, remainder_scale=2.0)
    declarations = [KerrResponse(chi3=1e-50)] if model == "ordinary" else [response]
    ordinary, descriptor = admit_tdse(
        declarations,
        grid,
        [mode],
        shape=(1, 1, 1, grid.n_omega),
        quadrature=transverse_quadrature([mode], n_r=4),
        integrator=Multistep(formulation="interaction"),
        has_noise=False,
        shock=True,
        prefactor_index=None,
    )
    if model == "tdse_native":
        assert ordinary == ()
        assert descriptor is not None and descriptor.kerr_model == model
    elif model == "ordinary":
        assert ordinary == tuple(declarations)
        assert descriptor is None
    else:
        assert len(ordinary) == 1 and isinstance(ordinary[0], KerrResponse)
        assert ordinary[0].thg
        assert descriptor is not None and descriptor.kerr_model == model


def test_filtered_seam_refuses_a_collar_not_bound_to_its_descriptor(declaration):
    problem = compile_declaration(declaration)
    problem.validate()
    changed = problem.grids.t_window_fine.copy()
    changed[changed.size // 2] *= 0.5
    changed.setflags(write=False)
    grids = dataclasses.replace(problem.grids, t_window_fine=changed)
    with pytest.raises(TranslationError, match="source collar differs"):
        dataclasses.replace(problem, grids=grids).validate()


def test_a_dispersive_kerr_response_beside_the_tdse_is_refused(admitted, declaration):
    """The TDSE owns its gas's whole electronic response (SEP 6, section 2)."""
    grid, mode, _ = declaration
    helium = OptmatKerr("He").dispersive_model()
    leaf = compile_responses(
        [DispersiveKerrResponse(helium, "symmetric")],
        grid,
        mode,
        density=Constant(2.5e24),
        z_max=0.001,
    ).dispersive_kerr
    damaged = dataclasses.replace(
        admitted,
        responses=dataclasses.replace(admitted.responses, dispersive_kerr=leaf),
    )
    with pytest.raises(TranslationError, match="refuses a dispersive Kerr response"):
        damaged.validate()
