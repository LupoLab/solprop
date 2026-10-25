# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Declare immutable benchmark workloads without importing their builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from benchmarks.records import Record, SuiteError, identity
from benchmarks.schedules import ENTRANCE_SCHEDULES

__all__ = [
    "ADAPTIVE",
    "CASES",
    "FIXED",
    "KERR",
    "SETS",
    "TRACKS",
    "Case",
    "ResolvedConfiguration",
    "WarningRule",
    "select",
]

ADAPTIVE = ("adaptive-multistep", "adaptive-rk")
FIXED = "fixed-rk"
KERR = "fixed-rk4ip"
TRACKS = (*ADAPTIVE, FIXED, KERR)
SETS = (
    "core",
    "modal-capillary",
    "free-space",
    "mode-averaged",
    "advanced",
    "backend-exercise",
)

# First ten accepted entrance steps of the independent NumPy default-RK pilot
# at rtol=1e-6. SEP 30 Part II records the pilot and verification. Keep these
# separate from Case so freezing a diagnostic does not rename adaptive windows.
_FIXED_SCHEDULES = {
    **{
        (name, 1): edges
        for name, edges in ENTRANCE_SCHEDULES.items()
        if name
        not in ("ctc_reduced", "scalar_wall", "vector_wall", "taper", "taper_wall")
    },
    ("radial_filament_256", 1): (
        0.0,
        3e-6,
        3.7247892534444475e-5,
        6.857136108217572e-5,
        9.611546370159467e-5,
        0.00012252784589431744,
        0.00014998314854780485,
        0.00017906017049297354,
        0.0002095001546572779,
        0.00024121867141398048,
        0.00027452363203191686,
    ),
}


@dataclass(frozen=True)
class WarningRule:
    """Declare one reviewed warning by phase, category and narrow message pattern."""

    phase: str
    category: str
    pattern: str


@dataclass(frozen=True)
class ResolvedConfiguration:
    """Freeze planned measurement semantics before constructing scientific objects."""

    case: str
    revision: int
    case_hash: str
    track: str
    start: float
    stop: float | None
    schedule: tuple[float, ...]
    step: float | None
    rtol: float | None
    atol: float | None
    observations: int
    fields: str
    statistics: str = "source-default"
    noise_seed: int | None = None
    reset: str = "launch-and-lead-in-v1"
    measurement_version: int = 1


@dataclass(frozen=True)
class Case:
    """Name the definition, supported measurements and qualification status."""

    name: str
    revision: int
    sets: tuple[str, ...]
    builder: str
    tracks: tuple[str, ...]
    length: float
    start: float
    stop: float
    step: float | None
    dimensions: tuple[int, ...]
    source: str
    status: str = "implemented"
    pending_reason: str | None = None
    estimate_bytes: int | None = None
    atol: float = 1e-10
    planes: int = 1
    warnings: tuple[WarningRule, ...] = ()

    @property
    def digest(self) -> str:
        """Identify immutable definition values, excluding selection aliases."""
        record = asdict(self)
        for key in ("sets", "status", "pending_reason", "estimate_bytes"):
            record.pop(key)
        return identity(record)

    def measurement(self, track: str, schedule: tuple[float, ...] = ()) -> Record:
        """Resolve a measurement without constructing any scientific object."""
        if track not in self.tracks:
            raise SuiteError(f"{self.name} does not admit {track}")
        if track == FIXED and not schedule:
            schedule = _FIXED_SCHEDULES.get((self.name, self.revision), ())
        adaptive = track in ADAPTIVE
        start = self.start if adaptive or track == KERR else 0.0
        stop = (
            self.stop
            if adaptive or track == KERR
            else (
                schedule[-1]
                if schedule
                else 10 * self.step
                if self.step is not None
                else None
            )
        )
        return asdict(
            ResolvedConfiguration(
                self.name,
                self.revision,
                self.digest,
                track,
                start,
                stop,
                schedule,
                self.step,
                1e-6 if track != KERR else None,
                self.atol if track != KERR else None,
                self.planes if adaptive else 1,
                "planes"
                if adaptive and self.name == "far_field_stats"
                else "all"
                if adaptive
                else "none",
                noise_seed=41
                if self.name == "grin_nonlinear"
                else 13
                if self.name == "noise"
                else None,
            )
        )


_CORE = (
    Case(
        "rdw",
        1,
        ("core", "modal-capillary"),
        "capillary",
        (*ADAPTIVE, FIXED),
        1.5,
        0.02,
        0.04,
        0.002,
        (4, 4096, 8192),
        "benchmarks/cases.py:rdw",
    ),
    Case(
        "ctc",
        1,
        ("core", "modal-capillary"),
        "capillary",
        (*ADAPTIVE, FIXED),
        0.2,
        0.00017,
        0.00034,
        0.000017,
        (6, 32768, 65536),
        "benchmarks/cases.py:ctc",
    ),
    Case(
        "kerr_beam_384",
        1,
        ("core", "free-space"),
        "kerr",
        (KERR,),
        5.5e-6,
        0.5e-6,
        5.5e-6,
        0.5e-6,
        (384, 384, 600),
        "benchmarks/sep28/cases.py:kerr_beam_384",
        atol=0.0,
        warnings=(
            WarningRule(
                "construction",
                "MaterialRangeWarning",
                r"malitson 'Malitson 1965' is valid over 210-3710 nm; "
                r"the requested spectrum spans 130\.2-1200 nm\. Source: "
                r"Malitson, JOSA 55, 1205 \(1965\), doi:10\.1364/JOSA\.55\.001205",
            ),
        ),
    ),
    Case(
        "radial_filament_256",
        1,
        ("core", "free-space"),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.03,
        0.0,
        0.03,
        None,
        (256, 250),
        "benchmarks/sep16/free_space.py:radial_filament",
        atol=0.0,
        planes=4,
    ),
)

_MODAL = (
    Case(
        "rdw_one_mode",
        1,
        ("modal-capillary",),
        "capillary",
        (*ADAPTIVE, FIXED),
        1.5,
        0.0,
        ENTRANCE_SCHEDULES["rdw_one_mode"][-1],
        None,
        (1, 4096, 8192),
        "benchmarks/cases.py:rdw with one resolved mode",
    ),
    # Revision 2 (SEP 35 phase 4): a 2 ps window, as a production run of this pulse
    # takes; revision 1's 1 ps crowded the 400 fs pulse. Its entrance re-piloted.
    Case(
        "ctc_reduced",
        2,
        ("modal-capillary",),
        "capillary",
        (*ADAPTIVE, FIXED),
        0.1,
        0.0,
        ENTRANCE_SCHEDULES["ctc_reduced"][-1],
        0.00012,
        (2, 8192, 16384),
        "benchmarks/cases.py:ctc_reduced with a 2 ps window",
    ),
    Case(
        "argon_blueshift",
        1,
        ("modal-capillary",),
        "capillary",
        (*ADAPTIVE, FIXED),
        0.4,
        0.0,
        ENTRANCE_SCHEDULES["argon_blueshift"][-1],
        None,
        (1, 2048, 4096),
        "docs/examples/processes/plot_06_ionisation_blue_shift.py",
    ),
    Case(
        "h2_raman_6ps",
        1,
        ("modal-capillary",),
        "capillary",
        (*ADAPTIVE, FIXED),
        2.0,
        0.0,
        ENTRANCE_SCHEDULES["h2_raman_6ps"][-1],
        None,
        (1, 16384, 32768),
        "docs/examples/processes/plot_05_raman_self_frequency_shift.py; 6 ps window",
    ),
    Case(
        "he_rdw_averaged",
        1,
        ("mode-averaged",),
        "capillary",
        (*ADAPTIVE, FIXED),
        1.5,
        0.0,
        ENTRANCE_SCHEDULES["he_rdw_averaged"][-1],
        None,
        (1, 4096, 8192),
        "benchmarks/cases.py:rdw; benchmarks/sep25/cases.py averaged route",
    ),
    Case(
        "argon_blueshift_averaged",
        1,
        ("mode-averaged",),
        "capillary",
        (*ADAPTIVE, FIXED),
        0.4,
        0.0,
        ENTRANCE_SCHEDULES["argon_blueshift_averaged"][-1],
        None,
        (1, 2048, 4096),
        "docs/examples/processes/plot_06_ionisation_blue_shift.py",
    ),
    Case(
        "h2_raman_6ps_averaged",
        1,
        ("mode-averaged",),
        "capillary",
        (*ADAPTIVE, FIXED),
        2.0,
        0.0,
        ENTRANCE_SCHEDULES["h2_raman_6ps_averaged"][-1],
        None,
        (1, 16384, 32768),
        "docs/examples/processes/plot_05_raman_self_frequency_shift.py; 6 ps window",
    ),
    Case(
        "gnlse_soliton",
        1,
        ("mode-averaged",),
        "gnlse",
        (*ADAPTIVE, FIXED),
        1.3278075458959397,
        0.0,
        ENTRANCE_SCHEDULES["gnlse_soliton"][-1],
        None,
        (1, 4096, 4096),
        "docs/introduction/tutorials/plot_02_soliton.py",
    ),
    Case(
        "gnlse_raman_shock",
        1,
        ("mode-averaged",),
        "gnlse",
        (*ADAPTIVE, FIXED),
        0.15,
        0.0,
        ENTRANCE_SCHEDULES["gnlse_raman_shock"][-1],
        None,
        (1, 16384, 16384),
        "benchmarks/sep13/gnlse.py",
    ),
)

_FREE = (
    Case(
        "radial_filament_512",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.03,
        0.0,
        ENTRANCE_SCHEDULES["radial_filament_512"][-1],
        None,
        (512, 1024, 2048),
        "benchmarks/sep16/free_space.py:radial_filament_large",
        atol=0.0,
    ),
    Case(
        "cartesian_filament_128",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.03,
        0.0,
        ENTRANCE_SCHEDULES["cartesian_filament_128"][-1],
        None,
        (128, 128, 1024, 2048),
        "benchmarks/sep16/free_space.py:cartesian_filament_large",
        atol=0.0,
    ),
    Case(
        "cartesian_filament_256",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.03,
        0.0,
        ENTRANCE_SCHEDULES["cartesian_filament_256"][-1],
        None,
        (256, 256, 1024, 2048),
        "benchmarks/sep28/cases.py:cartesian_filament_256",
        atol=0.0,
    ),
    Case(
        "kerr_beam_192",
        1,
        ("free-space",),
        "kerr",
        (KERR,),
        5.5e-6,
        0.5e-6,
        5.5e-6,
        0.5e-6,
        (192, 192, 300),
        "benchmarks/sep28/cases.py:kerr_beam_192",
        atol=0.0,
        warnings=(
            WarningRule(
                "construction",
                "MaterialRangeWarning",
                r"malitson 'Malitson 1965' is valid over 210-3710 nm; "
                r"the requested spectrum spans 130\.4-1182 nm\. Source: "
                r"Malitson, JOSA 55, 1205 \(1965\), doi:10\.1364/JOSA\.55\.001205",
            ),
        ),
    ),
    Case(
        "kerr_beam_768",
        1,
        ("free-space",),
        "kerr",
        (KERR,),
        3e-6,
        0.5e-6,
        3e-6,
        0.5e-6,
        (768, 768, 600),
        "benchmarks/sep28/cases.py:kerr_beam_768",
        atol=0.0,
        warnings=_CORE[2].warnings,
    ),
    Case(
        "line",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.08,
        0.0,
        ENTRANCE_SCHEDULES["line"][-1],
        None,
        (128, 512, 1024),
        "benchmarks/sep16/free_space.py:line",
        atol=0.0,
    ),
    Case(
        "planar",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.05,
        0.0,
        ENTRANCE_SCHEDULES["planar"][-1],
        None,
        (2, 32, 512, 512),
        "benchmarks/sep16/free_space.py:planar",
        atol=0.0,
    ),
    Case(
        "spatial_absorber",
        1,
        ("free-space",),
        "spatial",
        (*ADAPTIVE, FIXED),
        0.08,
        0.0,
        ENTRANCE_SCHEDULES["spatial_absorber"][-1],
        None,
        (64, 512, 1024),
        "benchmarks/sep16/free_space.py:radial_boundary",
        atol=0.0,
    ),
)

_GUIDE_WARNINGS = {
    "scalar_wall": (
        WarningRule(
            "construction",
            "SolpropWarning",
            "the temporal absorber attenuates a packet walking off by 0\\.15"
            " fs over 0\\.0002 m by only 0\\.00065 nepers, below the 6\\.9 "
            "that stops it wrapping around the time window; widen trange, "
            "or raise boundary_N above 2\\.12e\\+05",
        ),
        WarningRule(
            "construction",
            "GrazingIncidenceWarning",
            "the coherent wall block's quasi_static reaches 0\\.0449, within"
            " a factor ten of the 0\\.25 limit; the condensed paraxial model"
            " is marginal over this band",
        ),
    ),
    "vector_wall": (
        WarningRule(
            "construction",
            "SolpropWarning",
            "the temporal absorber attenuates a packet walking off by "
            "0\\.141 fs over 0\\.0002 m by only 0\\.00069 nepers, below the "
            "6\\.9 that stops it wrapping around the time window; widen "
            "trange, or raise boundary_N above 2e\\+05",
        ),
        WarningRule(
            "construction",
            "GrazingIncidenceWarning",
            "the coherent wall block's quasi_static reaches 0\\.06069, "
            "within a factor ten of the 0\\.25 limit; the condensed paraxial"
            " model is marginal over this band",
        ),
    ),
    "taper": (
        WarningRule(
            "construction",
            "SolpropWarning",
            "the temporal absorber attenuates a packet walking off by 0\\.19"
            " fs over 0\\.0002 m by only 0\\.00051 nepers, below the 6\\.9 "
            "that stops it wrapping around the time window; widen trange, "
            "or raise boundary_N above 2\\.69e\\+05",
        ),
    ),
    "taper_wall": (
        WarningRule(
            "construction",
            "SolpropWarning",
            "the temporal absorber attenuates a packet walking off by 0\\.19"
            " fs over 0\\.0002 m by only 0\\.00051 nepers, below the 6\\.9 "
            "that stops it wrapping around the time window; widen trange, "
            "or raise boundary_N above 2\\.69e\\+05",
        ),
        WarningRule(
            "construction",
            "GrazingIncidenceWarning",
            "the coherent wall block's quasi_static reaches 0\\.05613, "
            "within a factor ten of the 0\\.25 limit; the condensed paraxial"
            " model is marginal over this band",
        ),
    ),
    "step_index_pair": (),
    "grin_linear": (),
    "grin_nonlinear": (),
    "moving_solid_guide": (),
}

_ADVANCED = (
    *(
        Case(
            name,
            1,
            ("advanced",),
            "advanced",
            (*ADAPTIVE, FIXED),
            2e-4,
            0.0,
            ENTRANCE_SCHEDULES[name][-1],
            2e-5,
            (4 if name == "vector_wall" else 3, 1024, 2048),
            f"docs/tutorials/capillaries/{source}",
            warnings=_GUIDE_WARNINGS[name],
        )
        for name, source in (
            ("scalar_wall", "plot_07_coherent_wall.py"),
            ("vector_wall", "plot_09_vector_wall.py"),
            ("taper", "plot_08_coherent_wall_taper.py"),
            ("taper_wall", "plot_08_coherent_wall_taper.py"),
        )
    ),
    *(
        Case(
            name,
            1,
            ("advanced",),
            "advanced",
            (*ADAPTIVE, FIXED),
            length,
            0.0,
            ENTRANCE_SCHEDULES[name][-1],
            None,
            dimensions,
            source,
            warnings=_GUIDE_WARNINGS[name],
        )
        for name, length, dimensions, source in (
            (
                "step_index_pair",
                1e-3,
                (2, 1024, 2048),
                "benchmarks/sep25/cases.py:_general",
            ),
            (
                "grin_linear",
                0.0007846800630498693,
                (2, 128, 256),
                "docs/tutorials/waveguides/plot_06_graded_index.py",
            ),
            (
                "grin_nonlinear",
                0.1,
                (4, 4096, 8192),
                "docs/examples/waveguides/plot_04_graded_index_multimode.py",
            ),
            (
                "moving_solid_guide",
                0.015,
                (2, 2048, 4096),
                "docs/examples/waveguides/plot_05_moving_solid_guide.py",
            ),
        )
    ),
)

CASES = {case.name: case for case in (*_CORE, *_MODAL, *_FREE, *_ADVANCED)}
CASES["smoke"] = Case(
    "smoke",
    1,
    (),
    "smoke",
    (*ADAPTIVE, FIXED),
    1e-4,
    0.0,
    1e-4,
    1e-5,
    (),
    "suite integration smoke; not an adoption workload",
)
CASES["batch"] = Case(
    "batch",
    1,
    ("backend-exercise",),
    "batch",
    (FIXED,),
    1e-3,
    0.0,
    1e-3,
    1e-4,
    (2, 1, 1024, 1024),
    "tests/backends/test_batch_lockstep.py:simulation; ten fixed RK steps",
)

# Original guides with frozen default-RK entrance pilots; the 8 µm plasma
# tutorial ends after four steps, a window retained by design.
for _name, _length, _dimensions, _source in (
    (
        "vector_raman",
        2e-4,
        (1, 2, 256, 256),
        "tests/vector_raman_cases.py and vector Raman tutorial",
    ),
    (
        "envelope_plasma",
        8e-6,
        (1, 1, 256, 256),
        "docs/tutorials/capillaries/plot_05_envelope_plasma.py",
    ),
    (
        "mixture_gradient",
        1.5,
        (1, 1, 512, 1024),
        "docs/guide/media/mixtures.md; capillary interface case",
    ),
    (
        "dispersive_kerr",
        0.5,
        (1, 1, 2048, 4096),
        "docs/tutorials/capillaries/plot_11_dispersive_kerr.py",
    ),
    (
        "plasma_band",
        1e-3,
        (2, 1, 512, 1024),
        "tests/backends/test_band_plasma_parity.py:modal",
    ),
    (
        "plasma_trapezoid",
        1e-3,
        (2, 1, 512, 1024),
        "tests/backends/test_band_plasma_parity.py:modal",
    ),
    (
        "noise",
        1e-3,
        (1, 1, 1024, 1024),
        "tests/backends/test_batch_lockstep.py:simulation",
    ),
    ("dense_output", 1.0, (1, 1, 1024, 1024), "tests/backends/test_dense_output.py"),
    (
        "absorber_schemes",
        1e-3,
        (1, 1, 1024, 1024),
        "tests/backends/test_batch_lockstep.py:simulation",
    ),
    (
        "ordinary_stats",
        1e-3,
        (2, 1, 512, 1024),
        "tests/driver/test_prop_capillary.py:build",
    ),
    (
        "far_field_stats",
        1e-3,
        (224, 64, 128, 128),
        "docs/tutorials/free_space/plot_07_far_field_statistics.py",
    ),
):
    CASES[_name] = Case(
        _name,
        1,
        ("backend-exercise",),
        "exercise",
        ADAPTIVE if _name == "dense_output" else (*ADAPTIVE, FIXED),
        _length,
        0.0,
        ENTRANCE_SCHEDULES[_name][-1],
        None,
        _dimensions,
        _source,
        atol=1e-14
        if _name == "envelope_plasma"
        else 1e-18
        if _name == "dense_output"
        else 1e-12
        if _name == "far_field_stats"
        else 1e-10,
        planes=8 if _name in ("dense_output", "far_field_stats") else 1,
        warnings=(
            WarningRule(
                "construction",
                "MaterialRangeWarning",
                r"H2: optmat's Raman response raman_response/H2@293\.15K,800nm "
                r"is used outside its validated density range "
                r"\(0\.3722 to 0\.3722 amagat\)\. optmat: density below 0\.94 amagat, "
                r"where the motion of the molecules sets the widths: "
                r"the Dicke terms are capped at the Doppler widths \(TC-\d+\)",
            ),
        )
        if _name == "vector_raman"
        else (),
    )


def select(
    cases: tuple[str, ...] = (), sets: tuple[str, ...] = (), *, all_cases: bool = False
) -> tuple[Case, ...]:
    """Expand selectors once, retaining declared order and excluding no failures."""
    if any(name not in CASES for name in cases):
        raise SuiteError(
            f"unknown case: {', '.join(name for name in cases if name not in CASES)}"
        )
    if any(name not in SETS for name in sets):
        raise SuiteError("unknown set; use --all for the union")
    selected_sets = SETS if all_cases else sets or (() if cases else ("core",))
    return tuple(
        case
        for case in CASES.values()
        if case.name in cases or set(case.sets).intersection(selected_sets)
    )
