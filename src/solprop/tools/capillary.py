# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Soliton parameters of a mode or a gas-filled capillary, and capillary designs.

The full set of design numbers for a pulse in a mode (:func:`soliton_params`) or a
capillary (:func:`capillary_params`), the capillary that gives a soliton order
(:func:`capillary_design`) and the gas mixing ratio of a bulk dispersion
(:func:`gas_ratio`).
"""

import math
from dataclasses import dataclass
from typing import cast, get_args

import numpy as np
from jaxtyping import Float

from solprop._display import Displayed, line
from solprop._validation import choice, positive
from solprop.constants import ROOM_TEMPERATURE, C
from solprop.exceptions import (
    ModeError,
    ToolsError,
)
from solprop.modes import base
from solprop.modes.capillary import LossSpec, MarcatiliMode, MarcatiliModel, ModeKind
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import BASIS_NORMALISATION, mode_matrix, on_axis_rule
from solprop.physdata import (
    Gas,
    Mixture,
)
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.selection import MaterialSelection
from solprop.tools.beam import self_focusing_power
from solprop.tools.design import pressure_zdw
from solprop.tools.pulse import (
    PulseShape,
    _omega,
    pulse_energy,
    pulse_peak_power,
    pulse_width,
)
from solprop.tools.soliton import (
    SolitonContent,
    _area_per_order,
    _gaussian_order,
    dispersion_length,
    fission_length,
    nonlinear_coefficient,
    nonlinear_length,
    soliton_content,
    soliton_order,
    soliton_peak_power,
)
from solprop.tools.waveguide import (
    Fill,
    _check_carrier,
    _default_cladding,
    _fill_chi3,
    _fill_n2,
    _resolve_fill,
    _zdw_near,
)

__all__ = [
    "GasRatio",
    "SolitonParams",
    "capillary_design",
    "capillary_params",
    "gas_ratio",
    "soliton_params",
]


@dataclass(frozen=True)
class GasRatio:
    """Ratios of the bulk dispersion and the Kerr response of two gases.

    Attributes
    ----------
    beta2 : float
        Ratio of the bulk group-velocity dispersions, first gas over second.
    chi3 : float
        Ratio of the third-order susceptibilities, first gas over second.
    """

    beta2: float
    chi3: float


@dataclass(frozen=True, eq=False)
class SolitonParams(Displayed):
    """The design numbers of one pulse in one mode and one fill.

    Attributes
    ----------
    energy : float
        Pulse energy in J.
    fwhm : float
        Full width at half maximum of the power in s.
    width : float
        Natural width ``tau0`` of the shape in s
        (:func:`~solprop.tools.pulse.pulse_width`).
    omega : float
        Angular frequency of the pump in rad/s.
    wavelength : float
        Vacuum wavelength of the pump in m.
    gas : str
        The fill's species labels, joined by ``+``.
    pressure : float
        Total pressure at the position in bar; NaN for a mixture of independent
        populations, which has no single pressure.
    temperature : float
        Fill temperature in K.
    shape : {'sech', 'gauss'}
        ``"sech"`` or ``"gauss"``.
    peak_power : float
        Peak power in W.
    beta2 : float
        Group-velocity dispersion of the mode at the pump in s^2/m.
    density : float
        Total number density of the fill in m^-3.
    index : float
        Linear refractive index of the fill at the pump.
    n2 : float
        Nonlinear index of the fill in m^2/W.
    gamma : float
        Nonlinear coefficient in rad/(W m).
    soliton_order : float
        ``sqrt(L_D / L_NL)``.
    soliton_content : SolitonContent or None
        Unchirped NLSE constituent count and largest soliton; None for normal GVD.
    intensity : float
        Mode-averaged intensity ``peak_power / effective_area`` in W/m^2.
    peak_intensity : float
        On-axis intensity of the mode at the peak power in W/m^2: what a run's
        ``peak_intensity`` statistic reports at its first plane. For the fundamental
        of a capillary it is 1.77 times `intensity`. NaN for a mode without a
        transverse profile (a dispersion-specified one).
    critical_power : float
        Self-focusing power in W (:func:`~solprop.tools.beam.self_focusing_power`).
    dispersion_length : float
        In m.
    nonlinear_length : float
        In m.
    fission_length : float
        In m.
    zdw : float
        Zero-dispersion wavelength in m within 100 nm to 3 um, or NaN when the
        dispersion does not change sign there. No second, guess-seeded search is
        tried.
    loss_length : float
        The 1/e **power** loss length of the mode at the pump in m.
    effective_area : float
        In m^2.
    mode : Mode
        The mode.
    """

    energy: float
    fwhm: float
    width: float
    omega: float
    wavelength: float
    gas: str
    pressure: float
    temperature: float
    shape: PulseShape
    peak_power: float
    beta2: float
    density: float
    index: float
    n2: float
    gamma: float
    soliton_order: float
    soliton_content: SolitonContent | None
    intensity: float
    peak_intensity: float
    critical_power: float
    dispersion_length: float
    nonlinear_length: float
    fission_length: float
    zdw: float
    loss_length: float
    effective_area: float
    mode: Mode

    def __repr__(self) -> str:
        """Return a one-line description: the pulse, the fill and the soliton order.

        Returns
        -------
        str
            For example ``SolitonParams(300 uJ, 10 fs at 800 nm, He 0.4 bar, N=2.23)``.
        """
        return line(
            "SolitonParams",
            [
                ("", f"{self.energy * 1e6:.4g} uJ"),
                ("", f"{self.fwhm * 1e15:.4g} fs at {self.wavelength * 1e9:.4g} nm"),
                ("", f"{self.gas} {self.pressure:.4g} bar"),
                ("N", f"{self.soliton_order:.3g}"),
            ],
        )

    def __str__(self) -> str:
        """Return the parameter summary, section by section.

        Returns
        -------
        str
            MODE, WAVEGUIDE, FILL, PULSE, DISPERSION, POWER, INTENSITY and SOLITON,
            in the customary units: ``n2`` in cm^2/W, intensities in W/cm^2.
        """
        zdw = (
            "none in 100-3000 nm"
            if math.isnan(self.zdw)
            else f"{self.zdw * 1e9:.1f} nm"
        )
        ratio = self.peak_power / self.critical_power
        sections = (
            ("MODE", repr(self.mode)),
            (
                "WAVEGUIDE",
                f"Aeff = {self.effective_area:.2e} m^2, "
                f"Lloss = {self.loss_length:.2e} m (1/e power)",
            ),
            (
                "FILL",
                f"{self.pressure:.3g} bar {self.gas} at {self.temperature:g} K, "
                f"Pcr = {self.critical_power:.2e} W,\n  "
                f"gamma = {self.gamma:.2e} /(W m), n2 = {self.n2 * 1e4:.2e} cm^2/W",
            ),
            (
                "PULSE",
                f"{self.energy:.3e} J, {self.fwhm:.3e} s @ "
                f"{self.wavelength * 1e9:.1f} nm (shape: {self.shape})",
            ),
            (
                "DISPERSION",
                f"{self.beta2:.3e} s^2/m @ {self.wavelength * 1e9:.1f} nm, ZDW = {zdw}",
            ),
            ("POWER", f"{self.peak_power:.2e} W ({ratio:.4f} of Pcr)"),
            (
                "INTENSITY",
                f"{self.intensity * 1e-4:.2e} W/cm^2 mode-averaged, "
                f"{self.peak_intensity * 1e-4:.2e} W/cm^2 on axis",
            ),
            (
                "SOLITON",
                f"Ld = {self.dispersion_length:.2e} m, Lnl = "
                f"{self.nonlinear_length:.2e} m, Lfiss = {self.fission_length:.2e} m, "
                f"N = {self.soliton_order:.2f}",
            ),
        )
        content = self.soliton_content
        if content is None:
            constituents = "unavailable (normal dispersion)"
        elif content.count == 0:
            constituents = "0 discrete solitons"
        else:
            constituents = (
                f"{content.count} discrete solitons, "
                f"area/pi = {content.area_order:.3g}\n  "
                f"largest P = {content.largest_peak_power:.3e} W, "
                f"T = {content.largest_width:.3e} s, "
                f"kNL = {content.largest_nonlinear_wavenumber:.3e} 1/m"
            )
        return "\n".join(f"{name}:\n  {text}" for name, text in sections) + (
            f"\nCONSTITUENTS (unchirped NLSE):\n  {constituents}"
        )


def _on_axis_intensity(mode: Mode, peak_power: float, z: float) -> float:
    """Return the on-axis intensity of a mode carrying a given power.

    - ``peak_power``: power in W.
    - ``z``: position along the waveguide, m.
    - Returns: intensity on the axis in W/m^2, both transverse components included; NaN
      for a mode that declares no transverse profile.
    """
    # The same synthesis the run's peak_intensity statistic uses: the normalised mode
    # field at the origin in V/m per sqrt(W), so that eps0 c |e|^2 / 2 is the
    # intensity per watt.
    try:
        on_axis = mode_matrix(
            (mode,), on_axis_rule(mode.dimlimits(z)[0]), z, components=("x", "y")
        )
    except ModeError:
        # A dispersion-specified mode has an area and no transverse profile, so it has
        # a mode-averaged intensity and no on-axis one.
        return math.nan
    per_watt = BASIS_NORMALISATION * float(np.sum(np.abs(on_axis.synthesis) ** 2))
    return per_watt * peak_power


def soliton_params(
    mode: Mode,
    gas: Fill,
    pressure: float | None = None,
    *,
    energy: float,
    fwhm: float,
    wavelength: float,
    shape: PulseShape = "sech",
    temperature: float | None = None,
    z: float = 0.0,
    materials: MaterialSelection | None = None,
) -> SolitonParams:
    """Return the design numbers of a pulse in any mode.

    The mode supplies the dispersion, the area and the loss; the fill supplies the
    nonlinear index. They are separate arguments because a mode need not know its
    fill's Kerr response (a tabulated mode does not), and the caller is trusted to
    give the fill the mode was built with -- :func:`capillary_params` does both.

    Parameters
    ----------
    mode : Mode
        Any mode: a capillary, an antiresonant or tabulated fibre, a rectangular
        guide.
    gas : Gas, str, Mapping or Mixture
        The fill: a registry species, ``"air"``, mole fractions, or a mixture. A
        mixture of species selected from optmat takes their Kerr coefficients at
        `wavelength`, the carrier.
    pressure : float or None, optional
        Total pressure in **bar**; ``None`` is one bar, and is required beside a
        `Mixture`.
    energy : float
        Pulse energy in J, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used consistently for the width and the peak power.
    temperature : float or None, optional
        Fill temperature in K; ``None`` is room temperature, or the mixture's own
        beside a `Mixture`, which refuses any other.
    z : float, optional
        Position along the waveguide at which everything is evaluated, m.
    materials : MaterialSelection or None, optional
        Resolve a named fill from optmat under this selection, as a run's
        ``materials=`` does: its index, density and Kerr coefficient at
        `wavelength`. ``None`` is ``MaterialSelection()``; a `Mixture` is taken as
        it is. The mode is the caller's either way.

    Returns
    -------
    SolitonParams
        The numbers, with the mode.

    Raises
    ------
    ToolsError
        If a quantity is not positive, the shape is unknown, the mode has no
        dispersion at the pump, a component of the fill has no Kerr data, the
        selection names a carrier other than `wavelength`, or a pressure or a
        temperature other than its own is given beside a `Mixture`.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import soliton_params
    >>> mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced", cladding=1.45)
    >>> p = soliton_params(mode, "He", 0.4, energy=300e-6, fwhm=10e-15,
    ...                    wavelength=800e-9)
    >>> f"{p.soliton_order:.2f} {p.fission_length:.2f}"
    '2.26 1.75'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    omega = _omega(wavelength)
    here = np.array([float(z)])
    n2, index = _fill_n2(mixture, wavelength, float(z))
    peak_power = pulse_peak_power(energy, fwhm, shape)
    beta2 = float(base.beta2(mode, np.array([omega]), z)[0])
    gamma = nonlinear_coefficient(mode, n2, wavelength, z)
    area = base.aeff(mode, z)
    zdw = _zdw_near(mode, omega, float(z))
    shared = mixture.shared
    return SolitonParams(
        energy=float(energy),
        fwhm=float(fwhm),
        width=pulse_width(fwhm, shape),
        omega=omega,
        wavelength=float(wavelength),
        gas="+".join(mixture.labels),
        pressure=math.nan if shared is None else float(shared(here)[0]),
        # _resolve_fill refused a temperature that varies along z.
        temperature=cast("float", mixture.temperature),
        shape=shape,
        peak_power=peak_power,
        beta2=beta2,
        density=float(mixture.total_density(here)[0]),
        index=index,
        n2=n2,
        gamma=gamma,
        soliton_order=soliton_order(peak_power, fwhm, gamma, beta2, shape),
        soliton_content=(
            soliton_content(peak_power, fwhm, gamma, beta2, shape)
            if beta2 < 0
            else None
        ),
        intensity=peak_power / area,
        peak_intensity=_on_axis_intensity(mode, peak_power, float(z)),
        critical_power=self_focusing_power(wavelength, index, n2),
        dispersion_length=dispersion_length(fwhm, beta2, shape),
        nonlinear_length=nonlinear_length(peak_power, gamma),
        fission_length=fission_length(peak_power, fwhm, gamma, beta2, shape),
        zdw=zdw,
        loss_length=float(base.loss_length(mode, np.array([omega]), z)[0]),
        effective_area=area,
        mode=mode,
    )


def _capillary(
    radius: float,
    mixture: Mixture,
    *,
    n: int,
    m: int,
    kind: ModeKind,
    model: MarcatiliModel,
    loss: LossSpec,
    cladding: GlassIndex | complex | None,
) -> MarcatiliMode:
    """Build the HE, TE or TM capillary the capillary design tools report on.

    - ``radius``: core radius in m, > 0.
    - ``m``: radial mode index, >= 1.
    - ``cladding``: cladding index; ``None`` is fused silica.
    """
    return MarcatiliMode(
        positive(radius, "radius", ToolsError),
        mixture=mixture,
        n=n,
        m=m,
        kind=kind,
        model=model,
        loss=loss,
        cladding=_default_cladding(cladding),
    )


def capillary_params(
    radius: float,
    gas: Fill,
    pressure: float | None = None,
    *,
    energy: float,
    fwhm: float,
    wavelength: float,
    shape: PulseShape = "sech",
    temperature: float | None = None,
    n: int = 1,
    m: int = 1,
    kind: ModeKind = "HE",
    model: MarcatiliModel = "full",
    loss: LossSpec = True,
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> SolitonParams:
    """Return the design numbers of a pulse in a gas-filled capillary.

    Builds the mode as :func:`solprop.driver.interface.prop_capillary` does for the same
    arguments -- a :class:`~solprop.modes.capillary.MarcatiliMode` over the fill's
    mixture -- and hands it to :func:`soliton_params`.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas, str, Mapping or Mixture
        The fill, as `prop_capillary` takes it.
    pressure : float or None, optional
        Total pressure in **bar**; ``None`` is one bar. For a gradient, evaluate at
        the pressure of the position of interest.
    energy : float
        Pulse energy in J, > 0.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape.
    temperature : float or None, optional
        Fill temperature in K; ``None`` is room temperature, or the mixture's own
        beside a `Mixture`, which refuses any other.
    n : int, optional
        Azimuthal mode index.
    m : int, optional
        Radial mode index, >= 1.
    kind : {'HE', 'TE', 'TM'}, optional
        Mode family.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    loss : bool, float or callable, optional
        Loss policy of the mode.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Build the capillary as `prop_capillary` does under the same ``materials=``:
        the fill from optmat (index, density and Kerr coefficient at `wavelength`)
        and ``cladding=None`` as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    SolitonParams
        The numbers, with the mode that was built.

    Examples
    --------
    >>> from solprop.tools import capillary_params
    >>> p = capillary_params(125e-6, "He", 0.4, energy=300e-6, fwhm=10e-15,
    ...                      wavelength=800e-9, model="reduced", cladding=1.45)
    >>> f"{p.gamma:.3e} {p.intensity:.3e}"
    '4.884e-11 1.130e+18'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    mode = _capillary(
        radius,
        mixture,
        n=n,
        m=m,
        kind=kind,
        model=model,
        loss=loss,
        cladding=cladding,
    )
    return soliton_params(
        mode, mixture, energy=energy, fwhm=fwhm, wavelength=wavelength, shape=shape
    )


def _target_order(
    order: float | None,
    count: int | None,
    eigenvalue: float | None,
    shape: PulseShape,
) -> float:
    """Return the conventional soliton order that meets exactly one design target.

    - ``order``: a conventional soliton order, > 0.
    - ``count``: a number of solitons, >= 1.
    - ``eigenvalue``: the largest Zakharov-Shabat eigenvalue, > 0.
    - ``shape``: the pulse shape.
    """
    given = [
        name
        for name, value in (
            ("soliton_order", order),
            ("soliton_count", count),
            ("largest_eigenvalue", eigenvalue),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ToolsError(
            "give exactly one of soliton_order, soliton_count and largest_eigenvalue, "
            f"got {', '.join(given) or 'none'}"
        )
    if order is not None:
        return positive(order, "soliton_order", ToolsError)
    if count is not None:
        if count < 1:
            raise ToolsError(f"soliton_count must be at least 1, got {count}")
        # The count is M = ceil(S - 1/2) for an area S (soliton_content), so the
        # pulses with M solitons have S in (M - 1/2, M + 1/2]. The design takes the
        # middle, S = M: the pure M-soliton for sech, the farthest from either
        # threshold for a Gaussian.
        return count / _area_per_order(shape)
    eta = positive(cast("float", eigenvalue), "largest_eigenvalue", ToolsError)
    return eta + 0.5 if shape == "sech" else _gaussian_order(eta)


def capillary_design(
    radius: float,
    gas: Gas | str,
    *,
    zdw_wavelength: float,
    fwhm: float,
    wavelength: float,
    soliton_order: float | None = None,
    soliton_count: int | None = None,
    largest_eigenvalue: float | None = None,
    shape: PulseShape = "sech",
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    loss: LossSpec = True,
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
    pressure_max: float = 100.0,
) -> SolitonParams:
    r"""Return the pressure and energy that give a capillary a soliton design.

    The inverse of :func:`capillary_params`: the pressure puts the zero-dispersion point
    at `zdw_wavelength` (:func:`~solprop.tools.design.pressure_zdw`), and the energy
    then gives the pulse one soliton target. At that pressure :math:`\beta_2` and
    :math:`\gamma` do not depend on the energy and
    :math:`N^2 = \gamma P_0\tau_0^2/|\beta_2|`, so the peak power is
    :math:`P_0 = N^2|\beta_2|/(\gamma\tau_0^2)`. Give exactly one target:

    - `soliton_order`, the conventional :math:`N`;
    - `soliton_count`, a number :math:`M` of solitons in the scattering content
      (:func:`~solprop.tools.soliton.soliton_content`): the pulse area is set to
      :math:`M\pi`, the middle of the band of areas with :math:`M` solitons, which is
      :math:`N = M` for sech and :math:`N = M\sqrt{\pi/2}` for a Gaussian;
    - `largest_eigenvalue`, the :math:`\eta` of the largest constituent, which
      fixes its width :math:`\tau_0/(2\eta)`: :math:`N = \eta + 1/2` for sech, and
      a root of the Gaussian scattering problem otherwise.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas, as :func:`~solprop.tools.design.pressure_zdw` takes it.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    fwhm : float
        Full width at half maximum of the power in s, > 0.
    wavelength : float
        Pump wavelength in m, > 0.
    soliton_order : float or None, optional
        Target conventional soliton order, > 0.
    soliton_count : int or None, optional
        Target number of solitons, >= 1.
    largest_eigenvalue : float or None, optional
        Target imaginary part of the largest Zakharov-Shabat eigenvalue,
        dimensionless with time in units of the shape's natural width, > 0.
    shape : {'sech', 'gauss'}, optional
        The pulse shape, used for the width, the peak power and the content.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    loss : bool, float or callable, optional
        Loss policy of the mode in the returned numbers.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design under this selection, as :func:`capillary_params` and
        :func:`~solprop.tools.design.pressure_zdw` do. ``None`` is
        ``MaterialSelection()``.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**.

    Returns
    -------
    SolitonParams
        The numbers at the designed pressure and energy, with the mode; the
        pressure is ``.pressure`` (bar) and the energy ``.energy`` (J).

    Raises
    ------
    ToolsError
        If not exactly one target is given or it is out of range, no pressure in the
        bracket gives the zero-dispersion point, or a content target is asked of a
        pump in normal dispersion.

    Examples
    --------
    >>> from solprop.tools import capillary_design
    >>> p = capillary_design(125e-6, "Ar", zdw_wavelength=500e-9, fwhm=10e-15,
    ...                      wavelength=800e-9, soliton_order=3.0, cladding=1.45)
    >>> f"{p.pressure:.4f} bar {p.energy * 1e6:.1f} uJ N = {p.soliton_order:.3f}"
    '0.0664 bar 100.5 uJ N = 3.000'
    >>> g = capillary_design(125e-6, "Ar", zdw_wavelength=500e-9, fwhm=10e-15,
    ...                      wavelength=800e-9, soliton_count=2, shape="gauss",
    ...                      cladding=1.45)
    >>> f"{g.soliton_order:.4f} {g.soliton_content.count}"
    '2.5066 2'
    """
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    choice(shape, "shape", get_args(PulseShape), ToolsError)
    order = _target_order(soliton_order, soliton_count, largest_eigenvalue, shape)
    pressure = pressure_zdw(
        radius,
        gas,
        zdw_wavelength,
        pressure_max=pressure_max,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        materials=materials,
    )
    mixture = _resolve_fill(gas, pressure, temperature, materials)
    mode = _capillary(
        radius,
        mixture,
        n=1,
        m=m,
        kind="HE",
        model=model,
        loss=loss,
        cladding=cladding,
    )
    beta2 = float(base.beta2(mode, np.array([_omega(wavelength)]))[0])
    if soliton_order is None and beta2 >= 0.0:
        raise ToolsError(
            f"a soliton count or eigenvalue needs anomalous dispersion at the pump, "
            f"but beta2 = {beta2:.4g} s^2/m at {wavelength * 1e9:.4g} nm with the "
            f"zero-dispersion point at {zdw_wavelength * 1e9:.4g} nm"
        )
    n2, _ = _fill_n2(mixture, wavelength, 0.0)
    gamma = nonlinear_coefficient(mode, n2, wavelength)
    peak_power = soliton_peak_power(order, fwhm, gamma, beta2, shape)
    energy = pulse_energy(peak_power, fwhm, shape)
    return soliton_params(
        mode, mixture, energy=energy, fwhm=fwhm, wavelength=wavelength, shape=shape
    )


def gas_ratio(
    gas_a: Gas | str,
    gas_b: Gas | str,
    wavelength: float,
    *,
    pressure: float = 1.0,
    temperature: float = ROOM_TEMPERATURE,
    materials: MaterialSelection | None = None,
) -> GasRatio:
    """Return the ratios of bulk dispersion and Kerr response of two gases.

    The bulk (unguided) dispersion of each gas, not a mode's: it answers "how much
    more pressure of B matches A", which is a property of the gases alone.

    Parameters
    ----------
    gas_a : Gas or str
        The numerator's species.
    gas_b : Gas or str
        The denominator's species.
    wavelength : float
        Wavelength in m at which the dispersions are compared, > 0.
    pressure : float, optional
        Pressure of both gases in **bar**, > 0.
    temperature : float, optional
        Temperature of both in K.
    materials : MaterialSelection or None, optional
        Take both gases from optmat under this selection, as a run's ``materials=``
        does: their index, their density and their Kerr coefficient at
        `wavelength`; ``"air"`` is nitrogen and oxygen. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    GasRatio
        Named ``beta2`` and ``chi3`` ratios.

    Raises
    ------
    ToolsError
        If the selection names a carrier other than `wavelength`, or a gas has no
        Kerr data under it.

    Examples
    --------
    >>> from solprop.tools import gas_ratio
    >>> ratio = gas_ratio("Ar", "He", 800e-9)
    >>> f"{ratio.beta2:.1f} {ratio.chi3:.1f}"
    '18.6 28.4'
    """
    omega = _omega(wavelength)
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    dispersions = []
    susceptibilities = []
    for gas in (gas_a, gas_b):
        fill = _resolve_fill(
            gas, positive(pressure, "pressure", ToolsError), temperature, materials
        )
        dispersions.append(_bulk_beta2(_BulkGas(fill), omega))
        susceptibilities.append(_fill_chi3(fill, wavelength))
    return GasRatio(
        beta2=dispersions[0] / dispersions[1],
        chi3=susceptibilities[0] / susceptibilities[1],
    )


@dataclass(frozen=True)
class _BulkGas:
    """A uniform gas as an index model, for bulk dispersion.

    Attributes
    ----------
    fill : Mixture
        The gas resolved under a material selection, whose index is taken.
    """

    fill: Mixture

    def excess_beta(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the gas's part of the bulk propagation constant.

        The vacuum's ``omega / c`` has no second derivative, and leaving it out of a
        difference quotient removes its rounding, which at a gas's ``n - 1`` of 1e-5 is
        the larger part of the quotient.

        - ``omega``: angular frequencies in rad/s.
        - Returns: ``omega (n - 1) / c`` in 1/m, with ``n - 1`` formed without
          cancellation.
        """
        chi = self.fill.susceptibility(omega, np.zeros(1))[0]
        excess = chi / (1.0 + np.sqrt(1.0 + chi))
        return np.asarray(omega * np.real(excess) / C)


def _bulk_beta2(bulk: _BulkGas, omega: float) -> float:
    """Return the bulk group-velocity dispersion by a Richardson central difference.

    - ``omega``: angular frequency in rad/s.
    - Returns: ``d^2 beta / d omega^2`` in s^2/m.
    """
    # A second difference of the gas's part of beta: only a ratio of two such numbers is
    # used, so the few parts in 1e5 this reaches at the step modes.base chose are ample.
    step = base.BETA2_RELATIVE_STEP * omega

    def second(h: float) -> float:
        """Return the three-point second difference at step ``h``.

        - ``h``: step in rad/s.
        - Returns: the difference quotient in s^2/m.
        """
        values = bulk.excess_beta(np.array([omega - h, omega, omega + h]))
        return float(values[0] - 2.0 * values[1] + values[2]) / h**2

    return (4.0 * second(step / 2.0) - second(step)) / 3.0
