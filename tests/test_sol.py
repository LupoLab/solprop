# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The session namespace: what it holds, and what importing it does."""

import importlib
import logging
import subprocess
import sys
import textwrap
import warnings

import pytest

from solprop import nonlinear, optics, plotting, processing, profiles, sol, tools
from solprop.driver import interface
from solprop.exceptions import MaterialRangeWarning
from solprop.nonlinear import ionisation
from solprop.processing import compression

# Public names of the home modules that the session namespace deliberately leaves out,
# each with its reason. Anything else public in these modules must be in `sol`.
EXCLUDED = {
    "plotting": {
        # Constants and the styling helpers the plots are built from: for someone
        # writing a plot, who imports them from their home.
        "CORNERS",
        "DEFAULT_TRANGE",
        "DEFAULT_WAVELENGTH_RANGE",
        "OTHER_STATISTICS",
        "POWER_UNITS",
        "PULSE_STATISTICS",
        "SLICE_QUANTITIES",
        "Corner",
        "SliceQuantity",
        "cmap_colours",
        "cmap_white",
        "corner_text",
        "fwhm_arrows",
        "mode_labels",
        "per_width",
        "power_unit",
        "spectral_display",
        "spectral_limits",
        "subplot_grid",
    },
    "processing": {
        # Constants, type aliases and the axis bookkeeping behind the accessors.
        "ARRIVAL_METHODS",
        "BANDS",
        "DEFAULT_BANDPASS_BINS",
        "DEFAULT_BANDPASS_FRACTION",
        "FIELD_KINDS",
        "SPECTRAL_AXES",
        "ArrivalMethod",
        "Band",
        "FieldKind",
        "ModeSelection",
        "SpectralAxis",
        "band_mask",
        "grid_from_record",
        "layout",
        "spectral_axis",
        "spectral_order",
        # The unit bookkeeping of a run uniform in y, behind the plot labels.
        "TransverseUnits",
        "transverse_units",
        # The mixture diagnostics need a run with per-species channels and are reached
        # from their home by the few who want them.
        "absorbed_spectral_energy",
        "node_fluence",
        "species_ionisation",
        "species_ionisation_energy",
        # The coupling extraction belongs to the solprop.coupling workflow, which the
        # session does not flatten.
        "PLANE_TOLERANCE",
        "field_plane",
    },
    # Two type aliases.
    "tools": {"Fill", "PulseShape"},
    # A type alias, and the two array-level functions under the elements: the session
    # applies an element with `transfer=` or `apply_transfer`, never with these.
    "optics": {"Medium", "stored_factor", "taylor_phase"},
    # A type alias.
    "compression": {"ModeChoice"},
    "interface": set(),
    "nonlinear": {
        # The Raman model's building blocks and options, reached from their home by
        # whoever declares a molecule.
        "Dephasing",
        "IntermediateBroadening",
        "MolecularOptions",
        "RamanModel",
        "RamanTensorTerm",
        # A material region's declared delayed response, reached from its home by
        # whoever builds a guide, exactly as `MaterialKerr` and `MaterialRegion`
        # are -- none of the three is in the flat session namespace.
        "SilicaTensorResponse",
        # The optmat silica bundle's sampled shape and its response builder, reached
        # from their home by whoever declares silica's response under a selection.
        "SampledShape",
        "bundle_response",
        # Translation-side types and helpers: what the driver passes to a backend, not
        # what a person types.
        "Domain",
        "EnvelopePlasmaSources",
        "GnlseProblem",
        "Polarisation",
        "ResponseTraits",
        "attenuation_from_db",
        "kerr_susceptibility",
        "modal_kerr_coefficient",
        "mode_average_prefactor",
        "normalised_damped_oscillator",
        "problem_from",
        "electronic_response",
        "raman_drive",
        "response_for",
        "responses_for",
        "rotational_nonrigid",
        "sqrtw_to_vm",
        "thg_carrier",
        "undamped_dephasing",
        "uppe_prefactor",
        "vibrational",
    },
    "ionisation": {
        # Constants, protocols and the rate helpers behind a rate (the rate itself is
        # optmat's, OptmatRate, or the caller's).
        "MAX_RATE",
        "DT_RTOL",
        "EvaluableRate",
        # SEP 18's running-integral rule: designed by translation for a grid, and
        # handed to a direct evaluation; not an interactive name.
        "IntegrationPolicy",
        "RunningIntegral",
        "IonisationRate",
        "RateConvention",
        "barrier_suppression_field",
        "bind_rate_energy",
        "ionisation_fraction",
        "keldysh",
        "rate_convention",
        "rate_energy",
        "rate_species",
    },
    "profiles": {
        # Constants, the protocol, and the composition helpers a script reaches from
        # their home.
        "DEFAULT_PRESSURE_FLOOR_FRACTION",
        "MIN_STAGE_PRESSURE",
        "FromCallable",
        "Scaled",
        "profile_identity",
        "Sum",
        # The flow law's bore integral, a helper of solprop.gasflow's calculators.
        "hydraulic_integral",
        # The flow weight's representation, built by
        # solprop.gasflow.temperature_weight rather than by hand.
        "SeriesOfPath",
        "ZProfile",
        "as_profile",
    },
}
RENAMED = {("plotting", "energy"): "plot_energy", ("plotting", "stats"): "plot_stats"}
HOMES = {
    "plotting": plotting,
    "processing": processing,
    "tools": tools,
    "optics": optics,
    "compression": compression,
    "interface": interface,
    "nonlinear": nonlinear,
    "ionisation": ionisation,
    "profiles": profiles,
}


def run_python(code):
    """Run a snippet in a fresh interpreter and return what it printed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


@pytest.fixture
def restored():
    """Leave the package logger and the warning filters as they were found."""
    package = logging.getLogger("solprop")
    handlers, level = list(package.handlers), package.level
    with warnings.catch_warnings():
        yield
    for handler in list(package.handlers):
        if handler not in handlers:
            package.removeHandler(handler)
            handler.close()
    # `end_session` removes what `configure_logging` installed, whoever installed it.
    for handler in handlers:
        if handler not in package.handlers:
            package.addHandler(handler)
    package.setLevel(level)


# ------------------------------------------------------------------------- contents


def test_every_name_is_the_object_its_home_module_defines():
    assert len(set(sol.__all__)) == len(sol.__all__)
    assert not {"np", "plt", "numpy", "matplotlib"} & set(sol.__all__)
    for name in sol.__all__:
        assert hasattr(sol, name), name
    assert sol.prop_capillary is interface.prop_capillary
    assert sol.prop_2d is plotting.prop_2d
    assert sol.plot_energy is plotting.energy
    assert sol.plot_stats is plotting.stats
    assert sol.energy is processing.energy
    assert sol.capillary_params is tools.capillary_params
    assert sol.compress_taylor is compression.compress_taylor
    assert sol.ChirpedMirror is optics.ChirpedMirror


@pytest.mark.parametrize("home", HOMES)
def test_nothing_public_in_a_home_module_is_forgotten(home):
    module = HOMES[home]
    for name in module.__all__:
        if name in EXCLUDED[home]:
            assert name not in sol.__all__, f"{home}.{name} is excluded and exported"
            continue
        exported = RENAMED.get((home, name), name)
        assert exported in sol.__all__, f"{home}.{name} is missing from solprop.sol"
        assert getattr(sol, exported) is getattr(module, name)


def test_the_exclusions_name_things_that_exist():
    for home, names in EXCLUDED.items():
        assert names <= set(HOMES[home].__all__), home


def test_a_star_import_binds_the_names_and_nothing_else():
    lines = run_python(
        """
        np = plt = "mine"
        from solprop.sol import *
        print(prop_capillary.__name__, prop_2d.__name__, capillary_params.__name__)
        print(compress_taylor.__name__, np, plt)
        print("logging" in dir(), "warnings" in dir())
        """
    )
    assert lines == [
        "prop_capillary prop_2d capillary_params",
        "compress_taylor mine mine",
        "False False",
    ]


# ---------------------------------------------------------------------- side effects


def test_importing_the_package_alone_does_none_of_this():
    lines = run_python(
        """
        import logging, sys, warnings
        import solprop
        package = logging.getLogger("solprop")
        print([type(h).__name__ for h in package.handlers], package.level)
        print("solprop.sol" in sys.modules, "matplotlib" in sys.modules)
        print(any(f[2].__name__ == "MaterialRangeWarning" for f in warnings.filters))
        """
    )
    assert lines == ["['NullHandler'] 0", "False False", "False"]


def test_importing_the_namespace_configures_logging_and_the_filter_once():
    lines = run_python(
        """
        import importlib, logging, warnings
        import solprop.sol as sol
        from solprop.exceptions import MaterialRangeWarning

        def state():
            package = logging.getLogger("solprop")
            null = logging.NullHandler
            real = [h for h in package.handlers if not isinstance(h, null)]
            ours = [f[0] for f in warnings.filters if f[2] is MaterialRangeWarning]
            return len(real), logging.getLevelName(package.level), ours

        print(state())
        import solprop.sol
        importlib.reload(sol)
        sol.start_session()
        print(state())
        sol.set_log_level("DEBUG")
        print(state())
        sol.end_session()
        print(state())
        """
    )
    assert lines == [
        "(1, 'INFO', ['ignore'])",
        "(1, 'INFO', ['ignore'])",
        "(1, 'DEBUG', ['ignore'])",
        # Ending the session removes the one filter it installed.
        "(0, 'NOTSET', [])",
    ]


def test_a_range_warning_is_silent_with_the_namespace_and_loud_without():
    # 200 um lies beyond optmat's silica, the default since SEP 3's Phase 4a, which is
    # qualified from 24.8 nm to 125 um (the probe was 150 nm on the registry's).
    probe = """
        import sys, warnings
        {first}
        import numpy as np
        from solprop.physdata import fused_silica
        with warnings.catch_warnings(record=True) as caught:
            {second}
            fused_silica().index(np.array([2 * np.pi * 299792458.0 / 200e-6]))
        print(len(caught))
    """
    assert run_python(probe.format(first="", second="pass")) == ["1"]
    assert run_python(probe.format(first="import solprop.sol", second="pass")) == ["0"]
    shown = probe.format(
        first="import solprop.sol as sol", second="sol.material_warnings(True)"
    )
    assert run_python(shown) == ["1"]


def test_a_users_own_stricter_filter_is_neither_removed_nor_shadowed(restored):
    # The same category, which is the case that matters: someone who made the range
    # warning an error before importing the namespace gets their error back.
    warnings.filterwarnings("error", category=MaterialRangeWarning)
    own = warnings.filters[0]
    sol.material_warnings(False)
    sol.material_warnings(False)
    assert [e[0] for e in warnings.filters if e[2] is MaterialRangeWarning] == [
        "ignore",
        "error",
    ]
    sol.material_warnings(True)
    assert [e for e in warnings.filters if e[2] is MaterialRangeWarning] == [own]
    with pytest.raises(MaterialRangeWarning):
        warnings.warn("outside the fit", MaterialRangeWarning, stacklevel=1)
    sol.material_warnings(True)
    assert own in warnings.filters


def test_starting_twice_installs_one_handler_and_ending_removes_it(restored):
    package = logging.getLogger("solprop")

    def installed():
        # Only the handlers `configure_logging` marks as its own: under a random order
        # another test's reporter may have left one of another kind on the logger.
        return [h for h in package.handlers if getattr(h, "_solprop_installed", False)]

    sol.start_session("WARNING")
    sol.start_session("WARNING")
    assert len(installed()) == 1
    assert package.level == logging.WARNING
    sol.end_session()
    assert installed() == []
    assert package.level == logging.NOTSET


def test_the_level_can_be_changed_for_the_session(restored):
    sol.start_session("WARNING")
    sol.set_log_level("DEBUG")
    assert logging.getLogger("solprop").level == logging.DEBUG
    sol.end_session()


def test_the_suites_collection_hook_ends_the_session(restored):
    # Collection imports the namespace (the doctest run imports every module), and the
    # hook in conftest.py ends its session before the first test. The hook is called
    # here rather than its effect looked for, because under a random order another
    # test may have configured logging since.
    conftest = importlib.import_module("conftest")
    sol.start_session("WARNING")
    conftest.pytest_collection_finish(session=None)
    package = logging.getLogger("solprop")
    assert not [h for h in package.handlers if getattr(h, "_solprop_installed", False)]
    assert not [e for e in warnings.filters if e[2] is MaterialRangeWarning]
