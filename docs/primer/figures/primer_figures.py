# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Draw the computed figures of the long-form primer in ``docs/primer/``.

This script is **not** run by the docs build. ``poe figures`` runs it beside the other
figure scripts whenever a figure changes, and the SVGs it writes next to it are
committed; the primer's pages only include them. Every number a figure shows comes from
solprop's public API or from a closed-form expression stated in its caption, and none of
the figures is a convergence or qualification study.

The SVGs are written deterministically: a fixed hash salt, no date in the metadata,
and trailing whitespace stripped, so that redrawing an unchanged figure changes no
byte. Each figure has a white background, so that it reads the same in the light and
dark themes of the manual.

Usage::

    uv run python docs/primer/figures/primer_figures.py            # every figure
    uv run python docs/primer/figures/primer_figures.py --only raman-silica

Each figure takes a second or two.
"""

# pyright: reportUnknownMemberType=false
# matplotlib's stubs leave most Axes methods partly unknown; the calls here are plain.

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType
from scipy.special import j0, jn_zeros

from solprop.constants import AU_EFIELD, AU_ENERGY, E_CHARGE, EPS0, M_E, C
from solprop.modes import MarcatiliMode
from solprop.modes.base import alpha, beta2
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata.glasses import SilicaBundle

HERE = Path(__file__).resolve().parent
"""The directory the SVGs are written to, beside this script."""

COLOURS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9")
"""A colour-blind-safe cycle (Okabe and Ito), shared by every figure."""

STYLE: dict[RcKeyType, object] = {
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.prop_cycle": cycler(color=COLOURS),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.hashsalt": "solprop-primer",
    "svg.fonttype": "path",
}
"""The rc settings every figure is drawn under."""


def _save(figure: Figure, name: str, source: str) -> None:
    """Write ``figure`` as ``name``.svg beside this script, deterministically.

    Parameters
    ----------
    figure : Figure
        The figure to write. It is closed afterwards.
    name : str
        The file name without its extension.
    source : str
        A one-line statement of what the figure is computed from, stored in the SVG's
        metadata.
    """
    path = HERE / f"{name}.svg"
    with mpl.rc_context(STYLE):
        figure.savefig(
            path,
            format="svg",
            metadata={
                "Date": None,
                "Description": f"{source}. Drawn by primer_figures.py.",
            },
        )
    plt.close(figure)
    text = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    path.write_text(text)


def representations() -> None:
    """Draw a few-cycle pulse as a real field, an analytic signal and an envelope."""
    with mpl.rc_context(STYLE):
        wavelength = 800e-9
        omega0 = 2 * np.pi * C / wavelength
        n_t = 4096
        t = (np.arange(n_t) - n_t // 2) * 0.05e-15
        envelope = np.exp(-2 * np.log(2) * (t / 6e-15) ** 2)
        field = envelope * np.cos(omega0 * t)
        # Optics convention: E(omega) = int E exp(+i omega t) dt, so numpy's ifft.
        spectrum = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(field))) * n_t
        omega = np.fft.fftshift(np.fft.fftfreq(n_t, t[1] - t[0])) * 2 * np.pi
        analytic = np.where(omega > 0, 2 * spectrum, 0)
        scale = np.max(np.abs(spectrum))

        figure, (left, right) = plt.subplots(
            1, 2, figsize=(7.6, 3.0), layout="constrained", width_ratios=(1.0, 1.3)
        )
        left.plot(t * 1e15, field, lw=1.0, label="$E(t)$")
        left.plot(t * 1e15, envelope, lw=1.6, label=r"$|E_a(t)|=|A(t)|$")
        left.plot(t * 1e15, -envelope, lw=1.6, color=COLOURS[1])
        left.set_xlim(-15, 15)
        left.set_xlabel("time (fs)")
        left.set_ylabel("field (arb.)")
        left.legend(loc="lower left", fontsize=8.5)
        left.set_title("a 6 fs pulse at 800 nm")

        x = omega / omega0
        right.plot(
            x,
            np.abs(spectrum) / scale,
            lw=1.6,
            label=r"$|\tilde E(\omega)|$, real field",
        )
        right.plot(
            x,
            np.abs(analytic) / scale,
            lw=1.4,
            ls="--",
            label=r"$|\tilde E_a(\omega)|$, analytic signal",
        )
        right.plot(
            x - 1,
            np.abs(analytic) / scale,
            lw=1.4,
            ls=":",
            label=r"$|\tilde A(\Omega)|$, envelope, $\Omega=\omega-\omega_0$",
        )
        right.set_xlim(-2.2, 2.2)
        right.set_ylim(0, 2.3)
        right.set_xlabel(r"frequency / $\omega_0$")
        right.set_ylabel("spectral amplitude (arb.)")
        right.legend(loc="upper left", fontsize=8)
        right.set_title("the same pulse in three descriptions")
    _save(figure, "representations", "numpy FFT of a Gaussian pulse, optics convention")


def capillary_dispersion() -> None:
    """Draw the HE11 dispersion of an argon-filled capillary, and the loss."""
    with mpl.rc_context(STYLE):
        wavelength = np.linspace(150e-9, 1500e-9, 400)
        omega = 2 * np.pi * C / wavelength
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(7.6, 3.0), layout="constrained"
        )
        for pressure in (0.1, 0.3, 1.0, 3.0):
            mode = MarcatiliMode(125e-6, "Ar", pressure)
            left.plot(
                wavelength * 1e9, beta2(mode, omega) * 1e30, label=f"{pressure:g} bar"
            )
        left.axhline(0.0, color="0.4", lw=0.8)
        left.set_ylim(-60, 60)
        left.set_xlabel("wavelength (nm)")
        left.set_ylabel(r"$\beta_2$ (fs$^2$/m)")
        left.legend(fontsize=8.5, title="argon", title_fontsize=8.5)
        left.set_title(r"HE$_{11}$, 125 µm radius")

        for m in (1, 2, 3):
            mode = MarcatiliMode(125e-6, "Ar", 1.0, m=m)
            loss = alpha(mode, omega) * 10 / np.log(10)
            right.semilogy(wavelength * 1e9, loss, label=rf"HE$_{{1{m}}}$")
        right.set_xlabel("wavelength (nm)")
        right.set_ylabel("loss (dB/m)")
        right.legend(fontsize=8.5)
        right.set_title(r"loss grows as $\lambda^2 u_{1m}^2/a^3$")
    _save(
        figure,
        "capillary-dispersion",
        "solprop MarcatiliMode, beta2 and alpha, default silica wall",
    )


def field_scale() -> None:
    """Draw the HE11 intensity across the core beside the effective-area model."""
    with mpl.rc_context(STYLE):
        u = float(jn_zeros(0, 1)[0])
        xi = np.linspace(0.0, 1.0, 400)
        intensity = j0(u * xi) ** 2
        # A_eff = (int |e|^2)^2 / int |e|^4 over the disc, in units of a^2.
        quad = np.linspace(0.0, 1.0, 20001)
        two = np.trapezoid(j0(u * quad) ** 2 * 2 * np.pi * quad, quad)
        four = np.trapezoid(j0(u * quad) ** 4 * 2 * np.pi * quad, quad)
        area = two**2 / four
        mean = two / area  # P / A_eff relative to the on-axis intensity
        disc = np.sqrt(area / np.pi)

        figure, axis = plt.subplots(figsize=(5.6, 3.1), layout="constrained")
        axis.plot(xi, intensity, lw=2.0, label=r"HE$_{11}$: $J_0^2(u_{01}r/a)$")
        axis.plot(
            [0, disc, disc],
            [mean, mean, 0],
            lw=1.6,
            ls="--",
            label=rf"effective-area disc, $A_{{\rm eff}}={area:.3f}\,a^2$",
        )
        axis.axhline(mean, color="0.5", lw=0.6)
        axis.annotate(
            rf"$P/A_{{\rm eff}}$ = {mean:.3f} of the axis ($\eta_e^2$)",
            xy=(0.03, mean),
            xytext=(0.22, 0.72),
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
            fontsize=8.5,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("radius $r/a$")
        axis.set_ylabel("intensity / on-axis intensity")
        axis.legend(loc="upper right", fontsize=8.5)
        axis.set_title(f"on axis the mode is {1 / mean:.3f} times $P/A_{{\\rm eff}}$")
    _save(figure, "field-scale", "closed-form J0 profile and its effective area")


def raman_silica() -> None:
    """Draw fused silica's two Raman response shapes and their gain spectra."""
    with mpl.rc_context(STYLE):
        t = np.linspace(0.0, 2e-12, 40001)
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(7.6, 3.0), layout="constrained"
        )
        for name, tensor in (
            ("default (Hollenbeck-Cantrell)", False),
            ("tensor (Lin-Agrawal)", True),
        ):
            bundle = SilicaBundle.resolve(tensor=tensor)
            h = np.asarray(bundle.h_r(t))
            label = rf"{name}, $f_R={bundle.raman_fraction:.4g}$"
            left.plot(
                t[t < 0.4e-12] * 1e15, h[t < 0.4e-12] * 1e-12, lw=1.5, label=label
            )
            # Optics convention: H(Omega) = int h exp(+i Omega tau) d tau.
            # Im H > 0 is then gain on the Stokes side.
            shift = np.linspace(0.0, 2 * np.pi * 40e12, 800)
            dt = t[1] - t[0]
            transfer = np.array(
                [np.sum(h * np.exp(1j * s * t)) * dt for s in shift], dtype=complex
            )
            right.plot(
                shift / (2 * np.pi * 1e12), np.imag(transfer), lw=1.5, label=name
            )
        left.set_xlabel("delay (fs)")
        left.set_ylabel(r"$h_R(t)$ (ps$^{-1}$)")
        left.legend(fontsize=7.5)
        left.set_title("unit-integral response shapes")
        right.set_xlabel("frequency shift (THz)")
        right.set_ylabel(r"$\mathrm{Im}\,\tilde H_R(\Omega)$")
        right.set_title(r"Stokes gain, optics convention")
    _save(
        figure, "raman-silica", "solprop SilicaBundle h_r, default and tensor bundles"
    )


def ionisation_rates() -> None:
    """Draw optmat's recommended instantaneous rates for three noble gases."""
    with mpl.rc_context(STYLE):
        omega = 2 * np.pi * C / 800e-9
        figure, axis = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
        for colour, gas in zip(COLOURS, ("He", "Ar", "Xe"), strict=False):
            rate = OptmatRate(gas, 800e-9)
            field = np.geomspace(rate.threshold, rate.maximum_field, 300)
            axis.loglog(field, rate(field), color=colour, lw=1.8, label=gas)
            ip = rate.ionisation_potential
            keldysh = omega * np.sqrt(2 * M_E * ip) / E_CHARGE
            barrier = (ip / AU_ENERGY) ** 2 / 4 * AU_EFIELD
            axis.axvline(keldysh, color=colour, lw=0.8, ls=":")
            axis.axvline(barrier, color=colour, lw=0.8, ls="--")
        axis.set_xlim(1e9, 3e11)
        axis.set_ylim(1e0, 1e17)
        axis.set_xlabel("field amplitude (V/m)")
        axis.set_ylabel("rate $W$ (s$^{-1}$)")
        axis.legend(title="800 nm", fontsize=8.5, title_fontsize=8.5, loc="upper left")
        axis.set_title(
            r"dotted: Keldysh $\gamma=1$; dashed: barrier suppression $I_p^2/4Z$"
        )

        top = axis.secondary_xaxis(
            "top",
            functions=(
                lambda e: 0.5 * EPS0 * C * np.square(e) * 1e-4,
                lambda i: np.sqrt(np.maximum(i, 1e-300) * 1e4 / (0.5 * EPS0 * C)),
            ),
        )
        top.set_xlabel("intensity (W/cm²)")
    _save(
        figure,
        "ionisation-rates",
        "solprop OptmatRate, optmat's recommended rates at 800 nm",
    )


def trapezoid() -> None:
    """Draw the trapezoid rule's integration multiplier across a default field grid."""
    with mpl.rc_context(STYLE):
        ffac = 6.0
        x = np.linspace(0.02, 1.2, 400)  # omega / omega_max
        # theta = omega * dt_fine, with dt_fine = 2 pi / (ffac omega_max).
        theta = x * 2 * np.pi / ffac
        rho = (theta / 2) / np.tan(theta / 2)
        figure, axis = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
        axis.plot(
            x,
            rho,
            lw=1.8,
            label=r"one integral: $\rho(\theta)=\frac{\theta}{2}\cot\frac{\theta}{2}$",
        )
        axis.plot(x, rho**2, lw=1.8, label=r"Drude double integral: $\rho^2$")
        axis.axhline(
            1.0, color="0.3", lw=1.0, ls="--", label="exact, and the band-exact rule"
        )
        for edge, text in ((1.0, r"$\omega_{\max}$"), (1.1, r"$1.1\,\omega_{\max}$")):
            axis.axvline(edge, color="0.5", lw=0.8)
            axis.text(
                edge, 0.705, text, rotation=90, ha="right", va="bottom", fontsize=8.5
            )
        axis.set_xlim(0, 1.2)
        axis.set_ylim(0.7, 1.02)
        axis.set_xlabel(r"frequency / $\omega_{\max}$")
        axis.set_ylabel("multiplier on $i/\\omega$")
        axis.legend(loc="lower left", fontsize=8.5)
        axis.set_title(r"cumulative trapezoid at $\mathtt{ffac}=6$")
    _save(
        figure, "trapezoid", "closed-form trapezoid multiplier on a default field grid"
    )


FIGURES: dict[str, Callable[[], None]] = {
    "representations": representations,
    "capillary-dispersion": capillary_dispersion,
    "field-scale": field_scale,
    "raman-silica": raman_silica,
    "ionisation-rates": ionisation_rates,
    "trapezoid": trapezoid,
}
"""Every computed figure, by the name of the SVG it writes."""


def main() -> None:
    """Write every figure, or the ones named with ``--only``."""
    parser = argparse.ArgumentParser(
        description="Write the computed figures of the primer."
    )
    parser.add_argument("--only", action="append", choices=sorted(FIGURES))
    arguments = parser.parse_args()
    names: list[str] = arguments.only or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
