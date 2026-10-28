# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The printing rule: a one-line repr without arrays, a block for str and IPython."""

import logging
import re
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID
from record_numbers import (
    assert_record_numbers,
    record_skeleton,
    skeleton_hash,
    without_sep21_absorber,
)
from solprop import _display, tools
from solprop.coupling import AdmissionItem, CouplingReport
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.scan import Scan
from solprop.exceptions import MaterialRangeWarning, ModeError, OutputError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import (
    CoherentWall,
    GradedIndexProvider,
    IndexMaterial,
    IndexTable,
    MarcatiliMode,
    MaterialKerr,
    ModalConnection,
    ModalGuide,
    ModeCollection,
    PowerLawPermittivity,
    RectMode,
    StepIndexMode,
    StepIndexProvider,
    TabulatedHollowMode,
    TabulatedMode,
    TubeLattice,
    VincettiMode,
    vector_wall_operator,
    wall_operator,
)
from solprop.modes.guide import ModalPath, circular_pair
from solprop.modes.protocol import ChannelLabel
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.optics import Chain, ChirpedMirror, Material, TaylorPhase
from solprop.output.hdf5 import open_file, resume
from solprop.output.schema import StepHistory
from solprop.physdata import MaterialManifest, Mixture, calcium_fluoride, species
from solprop.physdata.mixtures import Component
from solprop.processing import compression
from solprop.processing._examples import example_grid, example_output
from solprop.processing.scans import ScanOutput
from solprop.profiles import (
    Constant,
    DensityPath,
    Linear,
    PiecewiseLinear,
    PressureGradient,
)
from solprop.pulses import (
    CustomPulse,
    CWPulse,
    DataPulse,
    GaussianPulse,
    SechPulse,
    ShapedPulse,
    SpatialPulse,
)
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController, RungeKutta
from solprop.translate.coupled import CoupledFlowTables, ModalMetricTables
from solprop.translate.general import GeneralModalRoute
from solprop.translate.modal_path import sample_modal_path
from solprop.translate.modal_tables import (
    modal_transform_tables,
    project_transverse_field,
    sample_modal_table,
)
from solprop.translate.responses import (
    RegionalResponseTables,
    regional_responses,
)

WINDOW = calcium_fluoride(model="malitson_1963/sellmeier")


def flat_top(t):
    return np.where(np.abs(t) < 10e-15, 1.0, 0.0)


def quiet(call, *arguments, **keywords):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return call(*arguments, **keywords)


def index_table():
    wavelength = np.linspace(400e-9, 2000e-9, 64)
    omega = 2.0 * np.pi * 299792458.0 / wavelength[::-1]
    return IndexTable(omega, 1.0 - 1e-4 * (2.4e15 / omega) ** 2 + 1e-9j)


def simulation():
    return quiet(
        Simulation.from_capillary,
        125e-6,
        0.1,
        "He",
        1.0,
        lambda0=800e-9,
        # The grid the record's literals below were taken on.
        grid=FieldGrid.from_limits(0.1, 800e-9, (200e-9, 2e-6), 300e-15, **LEGACY_GRID),
        fwhm=10e-15,
        energy=100e-6,
        modes=2,
        # The record's hash literals below were taken with the Runge-Kutta march, and
        # with the plasma on the cumulative trapezoid, which records nothing; SEP 18's
        # band rule, the default since, adds its taps to the record.
        integrator=RungeKutta(),
        plasma_integration="trapezoid",
        noise=None,
    )


def scan_output(path):
    """Return a scan handle on a path without opening it (reading takes a lock)."""
    output = ScanOutput.__new__(ScanOutput)
    output.path = path
    return output


def modal_records():
    """Return the bound-provider records, on the smallest rule that is still one."""
    omega = 2.0 * np.pi * 299792458.0 / 835e-9
    strand = StepIndexMode(250e-9, core=1.45, cladding=1.0, profile_wavelength=835e-9)
    guide = ModalGuide.from_step_index(
        [strand], core=IndexMaterial(1.45), cladding=IndexMaterial(1.0)
    )
    provider = guide.field_provider
    assert isinstance(provider, StepIndexProvider)
    rule = provider.node_rule(omega, n_r=3, n_theta=5)
    grin_omega = 2.0 * np.pi * 299792458.0 / 1.03e-6
    grin = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        radius=25e-6,
        family="TE",
        count=1,
        elements=8,
        extent=2.0,
    )
    moving_grin = replace(grin, radius=Linear(25e-6, 26e-6, 1e-3))
    dense = sample_modal_table(
        provider,
        np.linspace(0.99 * omega, 1.01 * omega, 4),
        np.array([0.0]),
        rule,
        # The smallest rule that is still one: its sampled basis is 0.31 from
        # orthonormal, so this printing example declares a budget that admits
        # it rather than a rule anyone would propagate on.
        gram_atol=1.0,
    )
    grin_guide = ModalGuide.from_provider(
        grin, np.linspace(0.99 * grin_omega, 1.01 * grin_omega, 4)
    )
    printed_tables = modal_transform_tables(
        dense, np.linspace(0.995 * omega, 1.005 * omega, 4), profile_error=1e-9
    )
    printed_field = printed_tables.electric[0, :, :, 0, :]
    printed_wall = wall_operator(
        [MarcatiliMode(125e-6, "He", 1.0, m=index + 1) for index in range(3)],
        np.array([2.30e15, 2.356e15, 2.40e15]),
    )
    printed_vector_wall = vector_wall_operator(
        [
            MarcatiliMode(125e-6, "He", 1.0, n=n, m=index + 1)
            for n in (1, -1)
            for index in range(2)
        ],
        np.array([2.30e15, 2.356e15, 2.40e15]),
    )
    return {
        "ChannelLabel": strand.channel_label,
        # The declared circular basis of a parity pair, built from the two
        # labels an HE_1m channel of any provider carries.
        "ChannelBasis": circular_pair(
            (
                ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
                ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
            ),
            (0, 1),
        ),
        "ModalGuide": guide,
        # The coherent wall block: its declaration, the ratios it was admitted under and
        # the operator itself.
        "CoherentWall": CoherentWall(),
        # Its CF4 declaration, whose growth limit and fallback only a CF4 wall prints.
        "CoherentWallCF4": CoherentWall(method="cf4", fallback="frozen_residual"),
        # The moving-basis connection: a synthetic two-channel record, because the
        # measured one costs three provider solves and this gallery is about the
        # printing.
        "ModalConnection": ModalConnection(
            np.array([[0.0, 3.05e2], [-3.05e2, 0.0]], dtype=complex),
            np.eye(2, dtype=complex),
            np.zeros((2, 2), dtype=complex),
            2.9e-8,
            1.53e-8,
            5e-4,
            2.2559e15,
        ),
        "WallOperator": printed_wall,
        "WallLimits": printed_wall.limits,
        "VectorWallOperator": printed_vector_wall,
        "ModalTransformTables": modal_transform_tables(
            dense, np.linspace(0.995 * omega, 1.005 * omega, 4), profile_error=1e-9
        ),
        # A declared transverse field projected onto this guide's one channel: the
        # channel's own profile, so the projection captures all of it.
        "ProjectedLaunch": project_transverse_field(printed_tables, printed_field),
        # A coupling's account: synthetic per-bin energies, a third captured.
        "CouplingReport": CouplingReport(
            "scalar_paraxial",
            0.05,
            np.full(3, 1.00028),
            np.array([2e-7, 6e-7, 2e-7]),
            np.array([0.6e-7, 2e-7, 0.7e-7]),
            np.array([0.6e-7, 2e-7, 0.7e-7]),
            "",
            (
                AdmissionItem("angular", 6.3e-3, None, True, "field", "cone + tail"),
                AdmissionItem("contrast", None, None, False, "model", "not applied"),
            ),
            "zero",
            None,
            {"source": "pulse"},
            "0" * 64,
        ),
        "ProviderChannelMode": grin_guide.mode_collection.modes[0],
        "ModalSampling": grin_guide.sampling,
        # A moving guide's path record: synthetic numbers of the shape a sampled path
        # carries, since the printing is the subject here.
        "ModalPath": ModalPath(
            np.array([0.0, 4e-4, 1e-3]), 4, 1, 4.6e-5, 2.0e-6, 5.8e-6
        ),
        # And its snapshot table, sampled at degree one on the graded-index sector's
        # own tapered rule, which is the cheapest path there is.
        "ModalPathTable": sample_modal_path(
            moving_grin,
            np.linspace(0.99 * grin_omega, 1.01 * grin_omega, 4),
            moving_grin.node_rule(0.99 * grin_omega, 0.0),
            np.array([0.0, 1e-3]),
            rtol=1.0,
            degree=1,
            measured=(2.0e-6, 5.8e-6),
        ),
        "StepIndexProvider": provider,
        "MaterialRegion": guide.regions[0],
        # The linear cladding above and a responding core: an empty `responses`
        # is how a region is said to be linear, so both sides of that print.
        "RespondingRegion": replace(
            guide.regions[0], name="core", responses=(MaterialKerr(n2=2.6e-20),)
        ),
        "MaterialKerr": MaterialKerr(chi3=2.2e-22),
        # An n2 declares the wavelength it was measured at, and both the
        # one-line form and the block say so.
        "MeasuredKerr": MaterialKerr(n2=2.6e-20, at=1550e-9),
        "SilicaTensorResponse": SilicaTensorResponse(),
        "ModalNodeRule": rule,
        "ModalSample": guide.solve(omega, 0.0, nodes=rule),
        "DenseModalTable": dense,
        "GradedIndexProvider": grin,
        "GeneralModalRoute": GeneralModalRoute(
            modal_transform_tables(
                dense, np.linspace(0.995 * omega, 1.005 * omega, 4), profile_error=1e-9
            ),
            regional_responses([2.2e-22], [0], n_to=8, dt_fine=1e-16),
            dense.gram_defect,
            # The same budget the table above declares: this printing example is the
            # smallest rule that is still one, and the route refuses a defect it has
            # not been told to admit.
            gram_atol=1.0,
        ),
        **seam_leaves(),
    }


def seam_leaves():
    """Return the three schema-12 contract leaves the general modal route carries."""
    omega = np.linspace(1.0e15, 2.0e15, 4)
    return {
        "ModalMetricTables": ModalMetricTables(
            "diagonal",
            omega,
            np.array([0.0, 1e-3]),
            np.full((1, 2, 2, 4), 0.5),
            None,
            np.full((1, 4), 1e-12),
            np.full((1, 4), 0.5),
            np.full((1, 4), 0.5),
            "lab",
            "sampled",
        ),
        "CoupledFlowTables": CoupledFlowTables(
            omega,
            np.array([0.0, 1e-3]),
            None,
            np.zeros((2, 2, 2, 4), dtype=complex),
            np.zeros((1, 2, 2), dtype=complex),
            np.zeros((1, 2, 4), dtype=complex),
            np.zeros((1, 2, 4), dtype=complex),
            None,
            np.zeros((1, 4)),
            np.zeros((1, 4)),
            np.ones((1, 4)),
            ("frozen_residual",),
            "lab",
            "factored",
        ),
        "RegionalResponseTables": RegionalResponseTables(
            np.zeros(1, dtype=np.int32),
            np.full((1, 1, 1), 2.2e-52),
            np.ones((1, 3, 3, 3, 3)),
            np.zeros((1, 0, 3, 3, 3, 3)),
            np.zeros((1, 1, 1, 0, 16)),
            np.zeros((1, 0), dtype=np.int32),
            np.full(8, 1e-15),
            "isotropic-kerr",
            "dense-nodes",
        ),
        # The same leaf with a region's delayed response declared beside the
        # instantaneous one: the `+`-joined tag list and a non-empty term axis are
        # what a reader has to be able to tell from the Kerr-only case above.
        "RegionalResponseTablesDelayed": regional_responses(
            [2.2e-22],
            [0],
            raman=[SilicaTensorResponse()],
            n_to=8,
            dt_fine=1e-15,
            form="real",
        ),
    }


def build_examples(folder):
    """Return one representative instance per class that prints specially."""
    run = example_output(n_modes=2, n_pol=1)
    example_output().to_file(folder / "run.h5")
    plane = run.planes()[-1]
    gaussian = GaussianPulse(
        wavelength=800e-9, fwhm=30e-15, energy=1e-6, phase=(0.3, 0.0, 250e-30)
    )
    data = DataPulse(
        omega=np.linspace(2e15, 3e15, 512), spectrum=np.ones(512, complex), energy=1e-6
    )
    lattice = TubeLattice(180e-9, 9e-6, 6)
    table = index_table()
    mirror = ChirpedMirror.lookup("PC70").bounces(4)
    sim = simulation()
    return {
        "FieldGrid": example_grid("field"),
        "EnvelopeGrid": example_grid("envelope"),
        "RadialGrid": RadialGrid(2e-3, 64),
        "FreeGrid": FreeGrid(2e-3, 32, 1e-3, 16),
        "GaussianPulse": gaussian,
        "SechPulse": SechPulse(wavelength=1030e-9, width=20e-15, peak_power=1e6),
        "CustomPulse": CustomPulse(wavelength=800e-9, energy=1e-6, shape_fn=flat_top),
        "DataPulse": data,
        "CWPulse": CWPulse(wavelength=1064e-9, average_power=10.0, bandwidth=1e-9),
        "SpatialPulse": SpatialPulse(gaussian, 100e-6),
        "MarcatiliMode": MarcatiliMode(125e-6, "He", 1.2),
        "VincettiMode": VincettiMode(15e-6, "Ar", 1.0, lattice=lattice),
        "TubeLattice": lattice,
        "IndexTable": table,
        "TabulatedMode": TabulatedMode(15e-6, table=table),
        "TabulatedHollowMode": TabulatedHollowMode(
            15e-6, table=table, gas="Ar", pressure=1.0
        ),
        "StepIndexMode": StepIndexMode(1.25e-6, profile_wavelength=835e-9),
        "RectMode": RectMode(100e-6, 200e-6, "Ar", 1.0),
        "ModeCollection": ModeCollection(
            (MarcatiliMode(125e-6, "He", 1.0), MarcatiliMode(125e-6, "He", 1.0, m=2))
        ),
        "Simulation": sim,
        "SimulationSettings": sim.settings,
        "MemoryOutput": run,
        "HDF5Output": open_file(folder / "run.h5"),
        "ResumeState": resume(folder / "run.h5"),
        "ScanOutput": scan_output(folder / "scan.h5"),
        "RunMetadata": run.metadata,
        "SavedPlane": plane,
        "StepHistory": plane.step_history,
        "StepStats": plane.step_stats,
        "PiecewiseLinear": PiecewiseLinear(
            np.linspace(0.0, 1.5, 12), np.linspace(1.0, 0.0, 12)
        ),
        "Linear": Linear(1.0, 0.0, 1.5),
        "PressureGradient": PressureGradient.two_point(1.0, 1.2, 0.0),
        "DensityPath": DensityPath("He", PressureGradient.two_point(1.0, 1.2, 0.0)),
        "Mixture": Mixture.air(1.0, argon=False),
        "IndependentMixture": Mixture.of(*Mixture.air(1.0).components),
        "Component": Mixture.air(1.0).components[0],
        "Species": species("N2"),
        "Scan": Scan("rdw", pressure=[0.5, 1.0, 1.5], energy=[1e-4, 2e-4]),
        "SolitonParams": tools.capillary_params(
            125e-6,
            "He",
            0.4,
            energy=300e-6,
            fwhm=10e-15,
            wavelength=800e-9,
            cladding=1.45,
        ),
        "GasRatio": tools.gas_ratio("Ar", "He", 800e-9),
        "TaylorPhase": TaylorPhase((0.0, 0.0, -250e-30, 4000e-45), 800e-9),
        "Material": Material(WINDOW, 1.5e-3, 800e-9, loss=True),
        "GasPath": Material("Ar", 2.0, pressure=1.5),
        "MixturePath": Material(Mixture.fill({"Ar": 1.0}, 1.0), 2.0),
        "ChirpedMirror": mirror,
        "PhaseOnlyMirror": ChirpedMirror.lookup("HD59"),
        "Chain": Chain((mirror, Material(WINDOW, 1e-3, 800e-9))),
        "EmptyChain": Chain(()),
        "CompressionTaylor": compression.compress_taylor(run),
        "CompressionMirrors": compression.compress_mirrors(
            run,
            mirror="PC70",
            max_reflections=3,
            medium=WINDOW,
            thickness_range=(0, 1e-3),
        ),
        "CompressionMaterial": compression.compress_material(
            run, medium=WINDOW, thickness_range=(-0.2, 0.2)
        ),
        "CompressionBounces": compression.compress_mirrors(
            run, mirror="PC70", max_reflections=2
        ),
        "CepOptimum": compression.optimise_cep(example_output("field")),
        **modal_records(),
    }


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    return build_examples(tmp_path_factory.mktemp("display"))


NAMES = sorted(
    [
        "FieldGrid",
        "EnvelopeGrid",
        "RadialGrid",
        "FreeGrid",
        "GaussianPulse",
        "SechPulse",
        "CustomPulse",
        "DataPulse",
        "CWPulse",
        "SpatialPulse",
        "MarcatiliMode",
        "VincettiMode",
        "TubeLattice",
        "IndexTable",
        "TabulatedMode",
        "TabulatedHollowMode",
        "StepIndexMode",
        "RectMode",
        "ModeCollection",
        "Simulation",
        "SimulationSettings",
        "MemoryOutput",
        "HDF5Output",
        "ResumeState",
        "ScanOutput",
        "RunMetadata",
        "SavedPlane",
        "StepHistory",
        "StepStats",
        "PiecewiseLinear",
        "Linear",
        "PressureGradient",
        "DensityPath",
        "Mixture",
        "IndependentMixture",
        "Component",
        "Species",
        "Scan",
        "SolitonParams",
        "GasRatio",
        "TaylorPhase",
        "Material",
        "GasPath",
        "MixturePath",
        "ChirpedMirror",
        "PhaseOnlyMirror",
        "Chain",
        "EmptyChain",
        "CompressionTaylor",
        "CompressionMirrors",
        "CompressionMaterial",
        "CompressionBounces",
        "CepOptimum",
        "CoherentWall",
        "CoherentWallCF4",
        "ModalConnection",
        "WallOperator",
        "WallLimits",
        "VectorWallOperator",
        "ModalGuide",
        "StepIndexProvider",
        "MaterialRegion",
        "RespondingRegion",
        "MaterialKerr",
        "MeasuredKerr",
        "SilicaTensorResponse",
        "ModalNodeRule",
        "ModalSampling",
        "ModalPath",
        "ModalPathTable",
        "ModalSample",
        "DenseModalTable",
        "GradedIndexProvider",
        "ChannelLabel",
        "ChannelBasis",
        "ProviderChannelMode",
        "ModalTransformTables",
        "ProjectedLaunch",
        "CouplingReport",
        "GeneralModalRoute",
        "ModalMetricTables",
        "CoupledFlowTables",
        "RegionalResponseTables",
        "RegionalResponseTablesDelayed",
    ]
)


class Recorder:
    """Stand in for IPython's pretty printer."""

    def __init__(self):
        self.written = []

    def text(self, value):
        self.written.append(value)


def test_the_list_of_names_is_the_list_of_examples(examples):
    assert sorted(examples) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_a_repr_is_one_short_line_without_an_array(examples, name):
    text = repr(examples[name])
    assert "\n" not in text
    assert "array(" not in text
    assert len(text) < 200
    assert text.startswith(type(examples[name]).__name__)


@pytest.mark.parametrize("name", NAMES)
def test_a_block_fits_the_page_and_never_holds_an_array(examples, name, caplog):
    with caplog.at_level(logging.WARNING):
        text = str(examples[name])
    assert "array(" not in text
    assert max(len(row) for row in text.split("\n")) <= 88
    assert not caplog.records


# GasRatio is a plain dataclass of two floats: its generated repr obeys the rule.
@pytest.mark.parametrize("name", [name for name in NAMES if name != "GasRatio"])
def test_ipython_is_shown_the_block_and_a_cycle_the_repr(examples, name):
    instance = examples[name]
    assert isinstance(instance, _display.Displayed)
    shown, cycled = Recorder(), Recorder()
    instance._repr_pretty_(shown, cycle=False)
    instance._repr_pretty_(cycled, cycle=True)
    assert shown.written == [str(instance)]
    assert cycled.written == [repr(instance)]


def test_the_reprs_that_existed_keep_their_text(examples):
    assert repr(examples["MarcatiliMode"]) == (
        "MarcatiliMode(HE11, a=125 um, He, model=full, loss=True)"
    )
    assert repr(examples["VincettiMode"]) == (
        "VincettiMode(HE11, a=15 um, t=180 nm, r_ext=9 um, N=6, Ar 1 bar, loss=True)"
    )
    assert repr(examples["IndexTable"]) == "IndexTable(64 rows, 400-2000 nm, loss=log)"
    assert repr(examples["TabulatedMode"]) == (
        "TabulatedMode(HE11, a=15 um, None, pressure=0.0, "
        "IndexTable(64 rows, 400-2000 nm, loss=log))"
    )
    assert repr(examples["TabulatedHollowMode"]) == (
        "TabulatedHollowMode(HE11, a=15 um, Ar, pressure=1.0, "
        "IndexTable(64 rows, 400-2000 nm, loss=log))"
    )
    assert repr(MarcatiliMode(50e-6)) == (
        "MarcatiliMode(HE11, a=50 um, evacuated, model=full, loss=True)"
    )


OPTMAT_RELEASE = re.compile(r"optmat \d+\.\d+\.\d+")
"""The release an optmat citation names, as in "(optmat 0.8.0, fit/He)"."""


def without_optmat_release(value: object, *, materials: bool = False) -> object:
    """Return a hashed record with what an optmat release writes into it neutralised.

    An optmat release that moves no value still moves the record: its version, in every
    material's ``optmat_version`` and in the citations that name it; the citations'
    text; and a revised model's digest in the material manifest. These are optmat's to
    pin, by its own tests; what is left is solprop's record, every number included.
    """
    if isinstance(value, dict):
        return {
            key: (
                "<optmat>"
                if key in {"optmat_version", "citation"}
                or (materials and key == "digest")
                else without_optmat_release(
                    item, materials=materials or key == "materials"
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [without_optmat_release(item, materials=materials) for item in value]
    if isinstance(value, str):
        return OPTMAT_RELEASE.sub("optmat <release>", value)
    return value


def test_the_run_record_is_what_it_was_before_the_printing(examples):
    # The run record, without what an optmat release writes into it, is what it was
    # when the printing was added. Output schema 6 adds the power convention and the
    # factored power weight to what is hashed, schema 7 the channel labels and the
    # general modal identity, schema 8 the material manifest, schema 9 the stepping
    # method, schema 10 the interval cap and the start-up rule, schema 11 interpolated
    # and step planes. Every built-in material is optmat's since SEP 3, so the manifest
    # is never empty here, and a record carrying one has no older schema: below 8 the
    # manifest would be outside the resume key.
    #
    # Since the pin to optmat 0.8.0 the optmat version, the citations and the
    # manifest's model digests are neutralised (without_optmat_release), so that only a
    # change to solprop's record, or to a number in it, fails here. The record was a
    # single SHA-256 literal per schema until SEP 25; that hash also covered the last
    # bits of its derived numbers, which a platform's libm moves. Its structure is
    # still hashed exactly (identity metadata); its numbers are stored beside this test
    # and compared to RECORD_RTOL. Both were recorded from the state the former
    # literals certified; the literals were re-recorded at output schema 21, which
    # records every dataclass field at its default too.
    metadata = quiet(examples["Simulation"].metadata)
    assert not metadata.materials.empty
    for schema in (5, 7):
        with pytest.raises(OutputError, match="requires output schema 8"):
            replace(
                metadata, schema_version=schema, quadrature=None
            ).configuration_hash()
    literals = {
        11: "a8fc917e8b825754e8aa38a01d0d9249c34e6c8093501e9c06b96a404d2b0db8",
        10: "2266cc52fc3f5ae76c5e77303ed61af034563da132ae63e0bfe7447b764f1e2d",
        9: "3604e39fa502e6cd8299d00f0b31906338da5f924434b6bdddefcd30a508afba",
        8: "f33ef97965fd144e5274756173e9a02ac81f1f23183c6270ec1f1f62bbaf789e",
    }
    numbers, floors = {}, {}
    for schema, literal in literals.items():
        # Keep the historical projection: schema 22 adds the held-current identity,
        # whose record and hash separation have their own held-source contract test,
        # and schema 26 the precision, which its own resume test separates (SEP 33).
        configuration = {
            key: value
            for key, value in metadata.configuration.items()
            if key not in ("plasma_source", "precision")
        }
        record = replace(
            metadata,
            schema_version=schema,
            quadrature=None,
            configuration=configuration,
        )
        # SEP 21 records the absorber's spatial flag (output schema 27).
        skeleton, values, floor = record_skeleton(
            without_optmat_release(without_sep21_absorber(record.hashed()))
        )
        assert skeleton_hash(skeleton) == literal, schema
        numbers[schema] = np.asarray(values, dtype=float)
        floors[schema] = np.asarray(floor, dtype=float)
    # The older schemas hash a subset of the same record, so their numbers are the
    # same values, copied: exact.
    for schema in (8, 9, 10):
        assert np.array_equal(numbers[schema], numbers[11]), schema
    assert_record_numbers(
        numbers[11],
        floors[11],
        Path(__file__).with_suffix("")
        / "test_the_run_record_is_what_it_was_before_the_printing.npz",
    )


def test_the_run_record_names_its_power_convention(examples):
    # The amplitude label is part of what a file means, so the block prints it beside
    # the spectral convention; a schema-5 record (which carries no material manifest)
    # reads back as the vacuum label.
    metadata = quiet(examples["Simulation"].metadata)
    assert "optics spectra, poynting amplitudes, schema 27" in str(metadata)
    older = replace(
        metadata, schema_version=5, quadrature=None, materials=MaterialManifest()
    )
    assert "vacuum amplitudes, schema 5" in str(older)


def test_a_mode_built_over_a_mixture_names_its_fill():
    # The run interfaces build every capillary mode over a mixture, and such a mode
    # must not print as evacuated.
    mode = MarcatiliMode(125e-6, mixture=Mixture.air(1.0, argon=False))
    assert repr(mode) == "MarcatiliMode(HE11, a=125 um, N2+O2, model=full, loss=True)"
    assert "N2+O2 1 bar" in str(mode)
    assert "evacuated" in str(MarcatiliMode(50e-6))
    apart = MarcatiliMode(125e-6, mixture=Mixture.of(*Mixture.air(1.0).components))
    assert "independent populations" in str(apart)


def test_a_block_names_what_a_person_asks_about(examples):
    expected = {
        "FieldGrid": ("time window", "spectral band", "reference"),
        "GaussianPulse": ("wavelength", "fwhm", "energy", "GDD 250 fs^2"),
        "Simulation": ("grid", "mode 0", "mode 1", "responses", "save planes"),
        "MemoryOutput": ("planes", "field per plane", "statistics"),
        "SolitonParams": ("MODE", "FILL", "DISPERSION", "SOLITON", "on axis"),
        "CompressionTaylor": ("found", "duration", "peak power", "quality", "search"),
        "CompressionMirrors": ("per count", "reflections"),
        "Mixture": ("total pressure", "N2", "O2", "mole fraction"),
        "ChirpedMirror": ("reflections", "band", "GDD at band centre"),
        "PhaseOnlyMirror": ("phase only",),
        "Scan": ("pressure", "energy", "6 points"),
    }
    for name, keys in expected.items():
        text = str(examples[name])
        for key in keys:
            assert key in text, f"{name}: {key!r} missing"


def test_the_unbounded_exterior_region_says_it_is_unbounded(examples):
    # The cladding's outer radius is None, and a region prints its radii without
    # evaluating a declared path.
    exterior = examples["ModalGuide"].regions[-1]
    assert "unbounded" in repr(exterior)
    assert "unbounded" in str(exterior)


def test_a_long_profile_is_shown_by_its_ends(examples):
    text = str(examples["PiecewiseLinear"])
    assert "12 points" in text
    assert "..." in text
    assert len(text.split("\n")) == 9
    short = PiecewiseLinear(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert len(str(short).split("\n")) == 3


def test_an_hdf5_output_says_what_state_its_file_is_in(tmp_path):
    path = tmp_path / "run.h5"
    assert "not written yet" in repr(open_new(path))
    example_output().to_file(path)
    handler = open_file(path)
    assert repr(handler) == "HDF5Output('run.h5', 5 planes, open, read-only)"
    assert "0 m to 100 mm" in str(handler)
    # A long path is shown by its end, where the file's own name is.
    assert str(path)[-40:] in str(handler)
    handler.close()
    assert "closed, read-only" in repr(handler)
    assert "positions unavailable" in str(handler)
    state = resume(path)
    assert repr(state).startswith("ResumeState(5 planes saved")
    assert "last plane" in str(state)
    path.unlink()
    assert "file missing" in repr(handler)
    assert "file missing" in str(handler)


def open_new(path):
    from solprop.output.hdf5 import HDF5Output  # noqa: PLC0415 - used by one test

    return HDF5Output(path)


def test_an_empty_store_and_an_empty_history_say_so():
    from solprop.output.memory import MemoryOutput  # noqa: PLC0415 - used by one test

    empty = MemoryOutput()
    assert repr(empty) == "MemoryOutput(0 planes, open)"
    assert "no plane saved yet" in str(empty)
    plane = example_output().planes()[0]
    assert str(plane.step_history) == "StepHistory: no step attempted"
    stepped = example_output(with_step_stats=True)
    assert "between planes:  energy" in str(stepped)


def test_a_mode_in_a_message_stays_on_one_line(caplog):
    # The printed block is for a prompt. A log record and an exception message take the
    # one-line form, whatever `str` has become.
    mode = MarcatiliMode(125e-6, "He", 1.0)
    with pytest.raises(ModeError) as refusal:
        mode.at_parameters({"pressure": 1.0})
    assert "\n" not in str(refusal.value)
    assert repr(mode) in str(refusal.value)
    curve = replace(mode, loss=lambda omega: np.full_like(omega, 2.0))
    with caplog.at_level("WARNING", logger="solprop"):
        curve.neff(np.array([2.4e15]))
    (record,) = [r for r in caplog.records if "derivative" in r.getMessage()]
    assert "\n" not in record.getMessage()
    assert repr(curve) in record.getMessage()
    assert "loss:" in str(curve)
    assert "callable" in str(curve)


def test_a_mode_without_a_parity_prints_without_one():
    mode = StepIndexMode(
        2e-6, core=1.47, cladding=1.45, kind="TE", n=0, profile_wavelength=1e-6
    )
    assert repr(mode).startswith("StepIndexMode(TE01, a=2 um")


def test_a_loss_factor_of_zero_prints_as_the_number_it_is():
    assert "loss=0.0" in repr(MarcatiliMode(125e-6, "He", 1.0, loss=0.0))
    assert "loss=False" in repr(MarcatiliMode(125e-6, "He", 1.0, loss=False))


def test_a_scan_file_is_described_and_a_missing_one_is_not_an_error(tmp_path):
    output = scan_output(tmp_path / "gone.h5")
    assert repr(output) == "ScanOutput('gone.h5')"
    assert "state:    missing" in str(output)
    # A file that is there is reported as there and is NOT opened: reading a scan takes
    # a shared lock, which would block while a controller writes. These bytes are not
    # HDF5, so an attempt to open them would raise.
    output.path = tmp_path / "scan.h5"
    output.path.write_bytes(b"not an HDF5 file")
    assert "state:    present" in str(output)


def test_printing_large_objects_is_cheap():
    samples = 2**16
    pulse = DataPulse(
        omega=np.linspace(2e15, 3e15, samples),
        spectrum=np.ones(samples, complex),
        energy=1e-6,
    )
    run = example_output(n_z=201)
    for instance in (pulse, run):
        repr(instance)
        start = time.perf_counter()
        text = repr(instance) + str(instance)
        elapsed = time.perf_counter() - start
        # A block is built from numbers the object already holds; 50 ms is a loose
        # guard against a formatter that starts evaluating physics or dumping arrays,
        # loose enough for a busy machine.
        assert elapsed < 0.05
        assert len(text) < 2000


def test_the_formatter_handles_the_awkward_numbers():
    assert _display.engineering(999.96, "m", digits=4) == "1 km"
    assert _display.engineering(-2.5e-7, "m") == "-250 nm"
    assert _display.engineering(1e30, "W") == "1e+12 EW"
    assert _display.engineering(1e-30, "s") == "1e-12 as"
    assert _display.engineering(float("inf"), "m") == "inf"
    assert _display.engineering(12.3456) == "12.35"
    assert _display.summarise(np.array([])) == "<0 values>"
    assert _display.summarise(3.0) == "<scalar values>"
    assert _display.phase_text((0, 0, 0, 0, 0, 0, 2e-90)) == "order 6 2 fs^6"
    assert _display.block("Title", []) == "Title"
    long = _display.block("T", [("key", "x" * 200)])
    assert long.split("\n")[1].endswith("...")
    assert len(long.split("\n")[1]) == 88


def test_a_long_value_wraps_under_itself():
    # A line past 88 columns breaks at a space and continues under its value, so
    # nothing is lost from a run's summary (its absorber line passed the width).
    value = "temporal and spectral collars, in-stage; edges watched: " * 3
    text = _display.block("Run", [("absorber", value.strip()), ("seed", 7)])
    rows = text.split("\n")
    assert all(len(row) <= 88 for row in rows)
    assert "..." not in text
    assert rows[2].startswith(" " * 12) and not rows[2].startswith(" " * 13)
    assert " ".join(row.strip() for row in rows[1:-1]) == f"absorber: {value.strip()}"
    assert rows[-1] == "  seed:     7"


ROUNDING_DIAGNOSTICS = re.compile(
    r"(profile error|missed share|residual|unitary defect):( +)\S+"
)
"""The printed fields whose value is a rounding diagnostic, masked by the comparison."""


def rounding(block: str) -> str:
    """Return a printed block with its rounding diagnostics masked."""
    return ROUNDING_DIAGNOSTICS.sub(r"\1:\2<rounding>", block)


@pytest.mark.parametrize(
    "name",
    [
        "FieldGrid",
        "GaussianPulse",
        "ModeCollection",
        "Simulation",
        "MemoryOutput",
        "SolitonParams",
        "CompressionTaylor",
        "CoherentWall",
        "CoherentWallCF4",
        "ModalConnection",
        "WallOperator",
        "WallLimits",
        "VectorWallOperator",
        "ModalGuide",
        "StepIndexProvider",
        "MaterialRegion",
        "RespondingRegion",
        "MaterialKerr",
        "MeasuredKerr",
        "SilicaTensorResponse",
        "ModalNodeRule",
        "ModalSampling",
        "ModalPath",
        "ModalPathTable",
        "ModalSample",
        "DenseModalTable",
        "GradedIndexProvider",
        "ChannelLabel",
        "ChannelBasis",
        "ProviderChannelMode",
        "ModalTransformTables",
        "ProjectedLaunch",
        "CouplingReport",
        "GeneralModalRoute",
        "ModalMetricTables",
        "CoupledFlowTables",
        "RegionalResponseTables",
        "RegionalResponseTablesDelayed",
    ],
)
def test_the_blocks_are_what_they_were(examples, name, file_regression):
    text = str(examples[name])
    # The one thing in a block that changes from run to run.
    text = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]*", "<created>", text)
    # And the one that changes with what is installed: "auto" resolves to Kokkos when
    # its wheel is present.
    text = re.sub(r"backend: +\w+", "backend:     <backend>", text)
    # And the one that changes with SciPy: how many times a search called its objective.
    text = re.sub(r"\(\d+ evaluations\)", "(<n> evaluations)", text)
    # The rounding diagnostics -- residuals and defects at the level of epsilon -- are
    # the last bits of the platform and build that printed them, and the layout of a
    # block is what this test is about. Both sides are compared with them masked; the
    # stored file keeps the values printed when it was recorded, and is rewritten only
    # when something else in the block changes.
    stored = Path(__file__).with_suffix("") / f"{name}.txt"
    if stored.exists() and rounding(text + "\n") == rounding(stored.read_text()):
        return
    file_regression.check(text + "\n", extension=".txt", basename=name)


def test_the_branches_the_representative_instances_do_not_reach():
    # A super-Gaussian order, a delay, a sech given by its FWHM, a monochromatic wave,
    # and the family's base, which adds no shape field of its own.
    shifted = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6, m=2, t0=5e-15)
    assert repr(shifted) == (
        "GaussianPulse(wavelength=800 nm, fwhm=30 fs, m=2, energy=1 uJ, t0=5 fs)"
    )
    assert "fwhm=20 fs" in repr(SechPulse(wavelength=1030e-9, fwhm=20e-15, energy=1e-6))
    assert repr(CWPulse(wavelength=1064e-9, average_power=10.0)) == (
        "CWPulse(wavelength=1.064 um, average_power=10 W)"
    )
    assert repr(ShapedPulse(wavelength=800e-9, energy=1e-6)) == (
        "ShapedPulse(wavelength=800 nm, energy=1 uJ)"
    )
    # A scan narrowed to some of its points, a population labelled apart from its
    # species, and a history that did step.
    scan = Scan("rdw", pressure=[0.5, 1.0, 1.5], energy=[1e-4, 2e-4]).subset([0, 2, 4])
    assert "3 of 6 points" in str(scan)
    labelled = Component(species("He"), Constant(1e24), label="buffer")
    assert repr(labelled) == "Component(buffer, species=He, preionisation=0)"
    history = StepHistory(
        z=np.array([0.0, 1e-3, 2e-3]),
        h=np.array([1e-3, 1e-3, 5e-4]),
        norm=np.array([0.5, 1.2, 0.4]),
        accepted=np.array([True, False, True]),
        wall_time=np.array([0.01, 0.01, 0.02]),
    )
    assert repr(history) == "StepHistory(3 attempts, 1 rejected, 40 ms)"
    assert "3, of which 1 rejected" in str(history)
    assert "largest error norm" in str(history)


def test_a_spatial_simulation_names_its_space():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = RadialGrid(6e-4, 32)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )
    simulation = Simulation.from_radial(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    assert "RadialGrid" in repr(simulation)
    assert "space:" in str(simulation)
