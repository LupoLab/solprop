# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Declare immutable benchmark workloads without importing their builders.

Every case is a complete guide run with its interface's default integrator and
tolerances (SEP 37), a fixed noise seed (the N = 3 soliton runs noiseless) and the
native backend unless asked otherwise (SEP 38). The ``tracking`` set is what
``python -m benchmarks track`` times by default; every other case runs only when named.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from benchmarks.records import Record, SuiteError, identity

__all__ = [
    "CASES",
    "SEED",
    "SETS",
    "STEPPERS",
    "TRACKING",
    "WIDTHS",
    "Case",
    "select",
]

SETS = (
    "tracking",
    "modal-capillary",
    "free-space",
    "mode-averaged",
    "advanced",
    "backend-exercise",
)
TRACKING = "tracking"
STEPPERS = ("default", "multistep", "multistep-interaction", "rk")
WIDTHS = ("1", "production")
SEED = 20261027
"""The noise seed every case fixes, so that its repeats take the same steps."""


@dataclass(frozen=True)
class Case:
    """Name one complete workload, how it is observed and what it costs.

    Attributes
    ----------
    name : str
        Registry key; a changed definition keeps the name and raises `revision`.
    revision : int
        Definition revision; history rows compare only within one.
    sets : tuple of str
        Selection groups; ``tracking`` puts the case in the default run.
    interface : str
        The ``prop_*`` interface whose defaults the case runs with.
    builder : str
        Builder family in ``benchmarks.builders``.
    length : float
        Guide length in metres; a run propagates all of it.
    dimensions : tuple of int
        Main array sizes, for the plan only: the modes (and polarisations) or the
        transverse grid, then the time samples and those of the oversampled
        nonlinear grid, as the builder realises them.
    source : str
        Where the definition comes from.
    widths : tuple of str
        Thread widths the case runs at by default: ``"1"`` and ``"production"``.
    planes : int
        Evenly spaced observations, the last at `length`.
    fields : str
        ``"all"`` keeps every observed field, ``"planes"`` only the last.
    step : float or None
        A fixed step in metres; the case then runs its declared fixed schedule and
        admits no stepper override.
    cost : tuple of float or None
        Seconds per complete run at 1 and 16 threads, for the plan's estimate;
        ``None`` if unknown. Measured for the release on a 16-core Google Cloud
        c2d-highcpu-32 (AMD EPYC 7B13, one thread per core) by the tracking run, with
        the boundary monitor attached; `ctc`'s one-thread value is from a run without
        the monitor, which costs it nothing measurable.
    """

    name: str
    revision: int
    sets: tuple[str, ...]
    interface: str
    builder: str
    length: float
    dimensions: tuple[int, ...]
    source: str
    widths: tuple[str, ...] = WIDTHS
    planes: int = 11
    fields: str = "all"
    step: float | None = None
    cost: tuple[float | None, float | None] = (None, None)

    @property
    def tracked(self) -> bool:
        """Whether the default ``track`` selection includes the case."""
        return TRACKING in self.sets

    @property
    def steppers(self) -> tuple[str, ...]:
        """Stepper choices the case admits; a fixed schedule admits its own only."""
        return ("default",) if self.step is not None else STEPPERS

    @property
    def digest(self) -> str:
        """Identify the definition, excluding selection groups and estimates."""
        record = asdict(self)
        for key in ("sets", "widths", "cost"):
            record.pop(key)
        return identity(record)

    def configuration(
        self,
        stepper: str,
        width: int,
        backend: str = "native",
        fraction: float = 1.0,
        warmup: float = 0.02,
        *,
        device: bool = False,
    ) -> Record:
        """Resolve one measurement without constructing a scientific object.

        On a device build, `width` is the host's thread count and `device` says the
        native backend must run on the accelerator.
        """
        if stepper not in self.steppers:
            raise SuiteError(f"{self.name} admits only {', '.join(self.steppers)}")
        if not 0 < fraction <= 1 or not 0 <= warmup < 1:
            raise SuiteError("length fraction must be in (0, 1], warm-up in [0, 1)")
        return {
            "case": self.name,
            "revision": self.revision,
            "case_hash": self.digest,
            "stepper": stepper,
            "threads": width,
            "backend": backend,
            "device": device,
            "stop": self.length * fraction,
            "fraction": fraction,
            "warmup": warmup,
            "planes": self.planes,
            "fields": self.fields,
        }


_TRACKED = (
    Case(
        "gnlse_raman_shock",
        1,
        ("tracking", "mode-averaged"),
        "prop_gnlse",
        "gnlse",
        0.15,
        (1, 16384, 16384),
        "Dudley, Genty and Coen, Rev. Mod. Phys. 78, 1135 (2006), section III: "
        "50 fs FWHM",
        cost=(49.9, 69.0),
    ),
    Case(
        "gnlse_soliton",
        1,
        ("tracking", "mode-averaged"),
        "prop_gnlse",
        "gnlse",
        1.3278075458959397,
        (1, 3072, 3072),
        "docs/introduction/tutorials/plot_02_soliton.py; one soliton period, noiseless",
        cost=(0.79, 0.80),
    ),
    Case(
        "rdw",
        1,
        ("tracking", "modal-capillary"),
        "prop_capillary",
        "capillary",
        1.5,
        (4, 5120, 10240),
        "He resonant dispersive wave, four resolved modes",
        cost=(18.1, 15.2),
    ),
    Case(
        "he_rdw_averaged",
        1,
        ("tracking", "mode-averaged"),
        "prop_capillary",
        "capillary",
        1.5,
        (1, 5120, 10240),
        "rdw on the mode-averaged route",
        cost=(2.49, 2.52),
    ),
    # 0.2 m and 160 nm-4 um, as SEP 37 phase 0 measured it, rather than the worked
    # example's 0.4 m and 300 nm edge.
    Case(
        "argon_blueshift",
        1,
        ("tracking", "modal-capillary"),
        "prop_capillary",
        "capillary",
        0.2,
        (1, 3200, 6400),
        "docs/examples/processes/plot_06_ionisation_blue_shift.py, shortened",
        cost=(44.8, 24.6),
    ),
    Case(
        "argon_blueshift_averaged",
        1,
        ("tracking", "mode-averaged"),
        "prop_capillary",
        "capillary",
        0.2,
        (1, 3200, 6400),
        "argon_blueshift on the mode-averaged route",
        cost=(10.5, 10.5),
    ),
    Case(
        "argon_blueshift_envelope",
        1,
        ("tracking", "modal-capillary"),
        "prop_capillary",
        "capillary",
        0.2,
        (1, 1728, 1728),
        "argon_blueshift on the envelope grid",
        cost=(2.38, 2.62),
    ),
    Case(
        "h2_raman_averaged",
        1,
        ("tracking", "mode-averaged"),
        "prop_capillary",
        "capillary",
        0.4,
        (1, 4608, 9216),
        "docs/examples/processes/plot_05_raman_self_frequency_shift.py, 10 bar",
        cost=(14.0, 15.2),
    ),
    # Three modes, 0.13 m, 2 ps and 220 nm-4 um: the definition SEP 37 phase 0
    # measured, shorter than the 0.3 m comb-to-continuum test case.
    Case(
        "ctc",
        1,
        ("tracking", "modal-capillary"),
        "prop_capillary",
        "capillary",
        0.13,
        (3, 10240, 20480),
        "H2 comb-to-continuum, three modes",
        widths=("production",),
        cost=(520.0, 226.3),
    ),
    Case(
        "radial_filament_256",
        1,
        ("tracking", "free-space"),
        "prop_radial",
        "spatial",
        0.03,
        (256, 864, 1728),
        "Ar filament with plasma, 256 radial nodes",
        planes=4,
        cost=(8.05, 9.07),
    ),
    # A 2 cm guide, the window SEP 37 phase 0 measured, rather than the example's 0.1 m.
    Case(
        "grin_nonlinear",
        1,
        ("tracking", "advanced"),
        "prop_waveguide",
        "advanced",
        0.02,
        (4, 3072, 6144),
        "docs/examples/waveguides/plot_04_graded_index_multimode.py, 2 cm",
        widths=("production",),
        planes=5,
        cost=(None, 25.7),
    ),
    # Four 0.5 um steps: a fixed-step run costs the same per step, so four track the
    # work of a longer schedule.
    Case(
        "kerr_beam_384",
        1,
        ("tracking", "free-space"),
        "prop_free",
        "kerr",
        2e-6,
        (384, 384, 750, 750),
        "fused-silica Kerr beam, four fixed RK4IP steps",
        widths=("production",),
        planes=1,
        fields="planes",
        step=0.5e-6,
        cost=(None, 29.4),
    ),
    Case(
        "kerr_beam_192",
        1,
        ("tracking", "free-space"),
        "prop_free",
        "kerr",
        2e-6,
        (192, 192, 384, 384),
        "fused-silica Kerr beam, four fixed RK4IP steps",
        widths=("1",),
        planes=1,
        fields="planes",
        step=0.5e-6,
        cost=(15.3, None),
    ),
)

_MODAL = (
    Case(
        "rdw_one_mode",
        1,
        ("modal-capillary",),
        "prop_capillary",
        "capillary",
        1.5,
        (1, 5120, 10240),
        "rdw with one resolved mode",
    ),
    Case(
        "ctc_reduced",
        1,
        ("modal-capillary",),
        "prop_capillary",
        "capillary",
        0.1,
        (1, 9216, 18432),
        "the reduced comb-to-continuum test case: the fundamental alone over 0.1 m "
        "on a 2 ps window",
    ),
    Case(
        "ctc_reduced_two_mode",
        1,
        ("modal-capillary",),
        "prop_capillary",
        "capillary",
        0.1,
        (2, 9216, 18432),
        "ctc_reduced with two resolved modes",
    ),
    Case(
        "h2_raman_6ps",
        1,
        ("modal-capillary",),
        "prop_capillary",
        "capillary",
        2.0,
        (1, 18432, 36864),
        "docs/examples/processes/plot_05_raman_self_frequency_shift.py; 6 ps window",
    ),
    Case(
        "h2_raman_6ps_averaged",
        1,
        ("mode-averaged",),
        "prop_capillary",
        "capillary",
        2.0,
        (1, 18432, 36864),
        "h2_raman_6ps on the mode-averaged route",
    ),
)

_FREE = (
    Case(
        "radial_filament_512",
        1,
        ("free-space",),
        "prop_radial",
        "spatial",
        0.03,
        (512, 864, 1728),
        "radial_filament_256 at 512 nodes",
    ),
    *(
        Case(
            f"cartesian_filament_{size}",
            1,
            ("free-space",),
            "prop_free",
            "spatial",
            0.03,
            (size, size, 864, 1728),
            "the Ar filament on a Cartesian grid",
        )
        for size in (128, 256)
    ),
    Case(
        "kerr_beam_768",
        1,
        ("free-space",),
        "prop_free",
        "kerr",
        3e-6,
        (768, 768, 750, 750),
        "fused-silica Kerr beam, fixed RK4IP steps",
        planes=1,
        fields="planes",
        step=0.5e-6,
    ),
    Case(
        "line",
        1,
        ("free-space",),
        "prop_free",
        "spatial",
        0.08,
        (128, 576, 1152),
        "Ar Kerr beam on a line grid",
    ),
    Case(
        "planar",
        1,
        ("free-space",),
        "prop_planar",
        "spatial",
        0.05,
        (2, 32, 384, 384),
        "two-mode Ar slab",
    ),
    Case(
        "spatial_absorber",
        1,
        ("free-space",),
        "prop_radial",
        "spatial",
        0.08,
        (64, 576, 1152),
        "radial Kerr beam meeting its absorbing boundary",
    ),
)

_ADVANCED = (
    *(
        Case(
            name,
            1,
            ("advanced",),
            "prop_capillary",
            "advanced",
            2e-4,
            (4 if name == "vector_wall" else 3, 864, 1728),
            f"docs/tutorials/capillaries/{source}",
        )
        for name, source in (
            ("scalar_wall", "plot_07_coherent_wall.py"),
            ("vector_wall", "plot_09_vector_wall.py"),
            ("taper", "plot_08_coherent_wall_taper.py"),
            ("taper_wall", "plot_08_coherent_wall_taper.py"),
        )
    ),
    *(
        Case(name, 1, ("advanced",), "prop_waveguide", "advanced", length, size, source)
        for name, length, size, source in (
            ("step_index_pair", 1e-3, (2, 648, 1296), "nonlinear HE11 parity pair"),
            (
                "grin_linear",
                0.0007846800630498693,
                (2, 128, 256),
                "docs/tutorials/waveguides/plot_06_graded_index.py",
            ),
            (
                "moving_solid_guide",
                0.015,
                (2, 2560, 5120),
                "docs/examples/waveguides/plot_05_moving_solid_guide.py",
            ),
        )
    ),
)

_EXERCISES = tuple(
    Case(
        name,
        1,
        ("backend-exercise",),
        interface,
        "exercise",
        length,
        dimensions,
        source,
        planes=8 if name in ("dense_output", "far_field_stats") else 1,
        fields="planes" if name == "far_field_stats" else "all",
    )
    for name, interface, length, dimensions, source in (
        (
            "vector_raman",
            "prop_capillary",
            2e-4,
            (1, 2, 192, 192),
            "the vector Raman tutorial's circular pump and delayed probe",
        ),
        (
            "envelope_plasma",
            "prop_capillary",
            8e-6,
            (1, 1, 256, 256),
            "docs/tutorials/capillaries/plot_05_envelope_plasma.py",
        ),
        (
            "mixture_gradient",
            "prop_capillary",
            1.5,
            (1, 1, 576, 1152),
            "docs/guide/media/mixtures.md",
        ),
        (
            "dispersive_kerr",
            "prop_capillary",
            0.5,
            (1, 1, 1536, 3072),
            "docs/tutorials/capillaries/plot_11_dispersive_kerr.py",
        ),
        ("plasma_band", "prop_capillary", 1e-3, (2, 1, 576, 1152), "band plasma"),
        (
            "plasma_trapezoid",
            "prop_capillary",
            1e-3,
            (2, 1, 576, 1152),
            "trapezoid plasma",
        ),
        ("noise", "prop_gnlse", 1e-3, (1, 1, 1536, 1536), "modified-noise fibre"),
        ("dense_output", "prop_gnlse", 1.0, (1, 1, 1024, 1024), "dense output"),
        (
            "absorber_schemes",
            "prop_gnlse",
            1e-3,
            (1, 1, 1536, 1536),
            "split-step absorber",
        ),
        (
            "ordinary_stats",
            "prop_capillary",
            1e-3,
            (2, 1, 576, 1152),
            "ordinary statistics on two modes",
        ),
        (
            "far_field_stats",
            "prop_free",
            1e-3,
            (224, 64, 160, 160),
            "docs/tutorials/free_space/plot_07_far_field_statistics.py",
        ),
    )
)

CASES = {
    case.name: case for case in (*_TRACKED, *_MODAL, *_FREE, *_ADVANCED, *_EXERCISES)
}
CASES["smoke"] = Case(
    "smoke",
    1,
    (),
    "prop_gnlse",
    "smoke",
    1e-4,
    (1, 192, 192),
    "a short noiseless fibre for testing the runner; not a workload",
    planes=2,
)


def select(
    cases: tuple[str, ...] = (), sets: tuple[str, ...] = (), *, all_cases: bool = False
) -> tuple[Case, ...]:
    """Expand selectors in declared order; no selection means the tracking set."""
    if unknown := [name for name in cases if name not in CASES]:
        raise SuiteError(f"unknown case: {', '.join(unknown)}")
    if any(name not in SETS for name in sets):
        raise SuiteError("unknown set; use --all for every case")
    if all_cases:
        return tuple(case for case in CASES.values() if case.name != "smoke")
    selected = sets or (() if cases else (TRACKING,))
    return tuple(
        case
        for case in CASES.values()
        if case.name in cases or set(case.sets).intersection(selected)
    )
