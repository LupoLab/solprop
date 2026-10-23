# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent high-precision wavefront and Treacy references for the compressors.

This module deliberately uses linear solves and vector Snell rather than production's
intersection algebra or residual-phase evaluation. mpmath is supplied by the existing
sympy development dependency, as in the rectangular-mode reference tests.
"""

import mpmath as mp
import numpy as np

from solprop.constants import C
from solprop.optics import GratingPair
from solprop.physdata import SellmeierGlass

MALITSON_POLES = (
    ("0.6961663", "0.0684043"),
    ("0.4079426", "0.1162414"),
    ("0.8974794", "9.896161"),
)
"""Malitson (1965), J. Opt. Soc. Am. 55, 1205: Sellmeier strengths and poles in um."""

MALITSON_SILICA = SellmeierGlass(
    name="SiO2",
    source="malitson (test data)",
    wavelength_range=(0.21e-6, 3.71e-6),
    citation=(
        "I. H. Malitson, J. Opt. Soc. Am. 55, 1205-1208 (1965), "
        "doi:10.1364/JOSA.55.001205; the paper's air-relative coefficients"
    ),
    poles=tuple((float(b), float(p)) for b, p in MALITSON_POLES),
)
"""The prism glass of the compressor tests, a caller's own data: the same expansion
the high-precision reference below evaluates, so the two share one index (solprop's
built-in silica is optmat's and absolute, SEP 3)."""


def silica(omega):
    wavelength = 2 * mp.pi * C / omega * 10**6
    terms = MALITSON_POLES
    return mp.sqrt(
        1
        + sum(
            mp.mpf(b) * wavelength**2 / (wavelength**2 - mp.mpf(p) ** 2)
            for b, p in terms
        )
    )


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def vector(x, y):
    return mp.matrix([x, y])


def refract(direction, normal, ratio):
    tangent = ratio * (direction - dot(direction, normal) * normal)
    return tangent + mp.sqrt(1 - dot(tangent, tangent)) * normal


def intersect(point, direction, apex, face):
    distance, along = mp.lu_solve(
        mp.matrix([[direction[0], -face[0]], [direction[1], -face[1]]]), apex - point
    )
    return point + distance * direction, distance, along


def prism_reference(pair):
    geometry = pair.geometry
    half = mp.mpf(geometry.apex_angle) / 2
    theta = mp.mpf(geometry.incidence_angle)
    reference = 2 * mp.pi * C / mp.mpf(pair.wavelength)
    first_face = vector(-mp.sin(half), -mp.cos(half))
    first_exit = vector(mp.sin(half), -mp.cos(half))
    second_entry = vector(-mp.sin(half), mp.cos(half))
    second_exit = vector(mp.sin(half), mp.cos(half))
    normal_in = vector(mp.cos(half), -mp.sin(half))
    normal_out = vector(mp.cos(half), mp.sin(half))
    incident = vector(mp.cos(theta - half), mp.sin(theta - half))
    start = mp.mpf(geometry.first_face_distance) * first_face

    def interior_angle(omega):
        return mp.asin(mp.sin(theta) / silica(omega))

    def exit_angle(omega):
        return mp.asin(silica(omega) * mp.sin(2 * half - interior_angle(omega)))

    angle0 = exit_angle(reference)
    apex = mp.mpf(geometry.separation) * vector(
        mp.cos(half - angle0), mp.sin(half - angle0)
    ) + vector(0, -mp.mpf(geometry.insertion))

    def ray(omega):
        index = silica(omega)
        inside = refract(incident, normal_in, 1 / index)
        first, glass1, face1 = intersect(start, inside, vector(0, 0), first_exit)
        between = refract(inside, normal_out, index)
        entry, air, face2 = intersect(first, between, apex, second_entry)
        inside2 = refract(between, normal_out, 1 / index)
        exit_point, glass2, face3 = intersect(entry, inside2, apex, second_exit)
        output = refract(inside2, normal_in, index)
        return {
            "lengths": (glass1, air, glass2),
            "faces": (face1, face2, face3),
            "glass": glass1 + glass2,
            "air": air,
            "output": output,
            "incident": incident,
            "exit_point": exit_point,
        }

    fixed_plane = ray(reference)["exit_point"]
    passes = 2 if pair.double_pass else 1

    def wavefront(omega):
        traced = ray(omega)
        path = silica(omega) * traced["glass"] + traced["air"]
        path += dot(fixed_plane - traced["exit_point"], incident)
        return passes * omega / C * path

    def gdd7(omega):
        traced = ray(omega)
        index = silica(omega)
        material = (
            traced["glass"]
            / C
            * (2 * mp.diff(silica, omega) + omega * mp.diff(silica, omega, 2))
        )
        air = omega / C * traced["air"] * mp.diff(exit_angle, omega) ** 2
        angular_glass = (
            index * omega / C * traced["glass"] * mp.diff(interior_angle, omega) ** 2
        )
        return passes * (material - air - angular_glass)

    return reference, wavefront, ray, gdd7, (interior_angle, exit_angle), apex


def grating_wavefront(pair, omega):
    """Use the full ray distance, output-plane projection and groove phase."""
    sine = mp.sin(mp.mpf(pair.incidence_angle))
    density = mp.mpf(pair.line_density)
    diffracted = mp.asin(sine + 2 * mp.pi * C * pair.order * density / omega)
    distance = mp.mpf(pair.separation)
    lateral = distance * mp.tan(diffracted)
    path = distance / mp.cos(diffracted) - lateral * sine
    groove_phase = -2 * mp.pi * pair.order * density * lateral
    return (2 if pair.double_pass else 1) * (omega / C * path + groove_phase)


def reference_phase(pair, omega):
    with mp.workdps(50):
        if isinstance(pair, GratingPair):
            reference = 2 * mp.pi * C / mp.mpf(pair.wavelength)

            def raw(omega):
                return grating_wavefront(pair, omega)
        else:
            reference, raw, *_ = prism_reference(pair)
        phase0, slope = raw(reference), mp.diff(raw, reference)
        return np.array(
            [
                float(
                    raw(mp.mpf(float(value)))
                    - phase0
                    - slope * (mp.mpf(float(value)) - reference)
                )
                for value in omega
            ]
        )
