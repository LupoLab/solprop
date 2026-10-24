# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Describe bulk media and exact longitudinal propagation of spatial fields.

Spatial fields are in V/m. Energy, fluence and intensity are the axial Poynting flux
of the medium at the measured plane (:class:`SpatialFlux`, SEP 34). On a
line uniform in y (:class:`solprop.grid.LineGrid`) every transverse integral is per
metre of y: energy in J/m and power in W/m, while intensity and fluence stay local.
Transverse wave numbers are angular frequencies in rad/m. The propagation sign
and co-moving frame are those of ``docs/dev/conventions.md``.
"""

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike

from solprop import fourier
from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.exceptions import GridError, MaterialError, OutputError
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.modes.slab import PlanarGrid
from solprop.physdata import Gas, as_gas
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.physdata.species import fill_species
from solprop.profiles import Constant, DensityPath, Sum, ZProfile, as_profile

if TYPE_CHECKING:
    from solprop.translate.spatial import SeparableOperator

__all__ = [
    "ABSORBED_CHANNELS",
    "APERTURE_PROFILES",
    "EDGE_MOMENTS",
    "EVANESCENT_RATE",
    "FAR_FIELD_STATISTICS",
    "OPERATOR_STATISTICS",
    "ApertureProfile",
    "ApertureTables",
    "AxialFactors",
    "FluxSide",
    "SpatialDispersion",
    "SpatialFlux",
    "SpatialMedium",
    "aperture_profile",
    "aperture_scale",
    "aperture_statistics",
    "attenuation_extremes",
    "beam_radius",
    "cartesian_fluence_widths",
    "check_aperture_profile",
    "critical_power",
    "fwhm_r",
    "mask_plane_radius",
    "on_axis_intensity",
    "on_axis_spectrum",
    "peak_field",
    "physical_evanescent_rate",
    "planar_fluence",
    "planar_intensity",
    "planar_mode_energy",
    "plane_sample",
    "power_trace",
    "quadrature_half_width",
    "radial_fluence_widths",
    "spatial_energy",
    "spatial_fluence",
    "spatial_intensity",
    "spatial_power",
]

EVANESCENT_RATE = 200.0
"""Evanescent field attenuation cap in m^-1.

This is a declared numerical model of the nonpropagating band, not a material
absorption coefficient. Propagating material loss and gain are not clamped by it.
"""


@dataclass(frozen=True, eq=False)
class SpatialMedium:
    """Describe a gas, a glass or a constant complex bulk refractive index.

    Attributes
    ----------
    material : Gas, str, GlassIndex, complex or Mixture
        Gas name, existing glass model, or constant index. Default 1 is vacuum.
        Positive imaginary index is loss; negative imaginary index is gain.
        A mixture supplies its own density paths, temperature and optical models.
    pressure : float or solprop.profiles.ZProfile
        Absolute gas pressure in bar, nonnegative, optionally varying along z.
        Only gas media use pressure; other materials require the default 1.
    temperature : float or None
        Gas temperature in K, finite and positive; ``None`` (the default) is room
        temperature for a gas and the mixture's own beside a mixture, which is what
        the attribute holds after construction. A temperature other than the
        mixture's beside a mixture, and a temperature path along z, here or in a
        mixture, are refused by name: the mixture carries its temperature, and the
        free-space routes do not carry a path.
    susceptibility : solprop.physdata.sellmeier.GasSusceptibility or None
        Per-particle gas susceptibility; None selects the built-in gas's, optmat's
        (``solprop.physdata.species.default_species``). Not accepted for a glass or
        constant-index medium.
    density : solprop.profiles.ZProfile
        Gas number density in m^-3; dimensionless 1 for nongas media whose nonlinear
        susceptibilities are absolute. Derived from optmat's equation of state of
        the gas.

    Raises
    ------
    MaterialError
        If the material, pressure, temperature or susceptibility is inconsistent.
    """

    material: Gas | str | GlassIndex | complex | Mixture = 1.0
    pressure: float | ZProfile = 1.0
    temperature: float | None = None
    susceptibility: GasSusceptibility | None = None
    density: ZProfile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the material and its scalar density path."""
        self._resolve_temperature()
        if isinstance(self.material, Mixture):
            if self.pressure != 1.0 or self.susceptibility is not None:
                raise MaterialError("mixture carries its pressure and susceptibility")
            paths = tuple(c.density for c in self.material.components)
            object.__setattr__(
                self, "density", paths[0] if len(paths) == 1 else Sum(paths)
            )
        elif isinstance(self.material, str | Gas):
            self._resolve_gas()
        else:
            if self.pressure != 1.0 or self.susceptibility is not None:
                msg = "pressure and susceptibility apply only to gas media"
                raise MaterialError(msg)
            if not isinstance(self.material, GlassIndex):
                index = complex(self.material)
                if (
                    not math.isfinite(index.real)
                    or not math.isfinite(index.imag)
                    or index.real <= 0
                ):
                    msg = (
                        "constant material index must be finite with positive real part"
                    )
                    raise MaterialError(msg)
            object.__setattr__(self, "density", Constant(1.0))

    def _resolve_temperature(self) -> None:
        """Resolve the one temperature: the given one, the mixture's, or room's.

        Raises
        ------
        MaterialError
            If a temperature is a path along z, is not finite and positive, or
            differs from the mixture's beside a mixture.
        """
        own = self.material.temperature if isinstance(self.material, Mixture) else None
        given = self.temperature
        if isinstance(given, ZProfile) or isinstance(own, ZProfile):
            # A shared fill's pressure parameter and the Raman populations assume one
            # temperature; the free-space routes carry none along z.
            msg = (
                "a spatial medium takes one temperature in K, and a temperature path "
                "along z was given: the free-space routes do not carry one. Give a "
                "number, or propagate in a hollow guide (prop_capillary, "
                "prop_waveguide), which does"
            )
            raise MaterialError(msg)
        if given is not None and not (math.isfinite(given) and given > 0):
            msg = "spatial medium temperature must be finite and positive"
            raise MaterialError(msg)
        if own is None:
            object.__setattr__(
                self, "temperature", ROOM_TEMPERATURE if given is None else given
            )
            return
        if given is not None and float(given) != float(own):
            # As the guided modes and the run interfaces: the mixture's densities
            # were set at its own temperature, so another one cannot be honoured.
            msg = (
                f"the mixture carries its own temperature ({float(own):g} K), and "
                f"temperature={given!r} was given beside it: omit temperature=, or "
                "build the Mixture at the temperature you mean"
            )
            raise MaterialError(msg)
        object.__setattr__(self, "temperature", own)

    def _resolve_gas(self) -> None:
        """Resolve a named gas's susceptibility and density path.

        The built-in gas is optmat's: its law and its equation of state.

        Raises
        ------
        MaterialError
            If the pressure is negative.
        """
        gas = as_gas(cast("str | Gas", self.material))
        pressure = as_profile(self.pressure)
        if float(pressure(np.array([0.0]))[0]) < 0:
            msg = "spatial medium pressure must be nonnegative"
            raise MaterialError(msg)
        object.__setattr__(self, "material", gas)
        # The fill's single species: a given optmat law of the gas is carried in it
        # (SEP 3, Q27), so its density path names the law that acts.
        material = fill_species(gas, self.susceptibility)
        object.__setattr__(
            self,
            "density",
            # _resolve_temperature has made it a number.
            DensityPath(material, pressure, cast("float", self.temperature)),
        )
        if self.susceptibility is None:
            object.__setattr__(self, "susceptibility", material.susceptibility)

    def index(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Evaluate the material refractive index on a physical frequency band.

        Parameters
        ----------
        omega : numpy.ndarray
            Finite positive angular frequencies in rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position in m, finite; profile endpoint continuation applies.

        Returns
        -------
        numpy.ndarray
            Complex refractive index, dimensionless, shape ``(n_omega,)``.

        Raises
        ------
        MaterialError
            If a frequency, position or computed index is invalid.
        """
        _check_evaluation(omega, z)
        if isinstance(self.material, Mixture):
            susceptibility = self.material.susceptibility(omega, np.array([z]))[0]
            values = np.sqrt(1 + susceptibility)
        elif isinstance(self.material, GlassIndex):
            values = self.material.index(omega)
        elif self.susceptibility is not None:
            density = float(self.density(np.array([z]))[0])
            values = np.sqrt(1 + density * self.susceptibility.susceptibility(omega))
        else:
            values = np.full(omega.shape, complex(self.material), dtype=np.complex128)
        result = np.asarray(values, dtype=np.complex128)
        if not np.all(np.isfinite(result)):
            msg = "spatial medium returned a nonfinite refractive index"
            raise MaterialError(msg)
        return result

    def dindex_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Evaluate the analytic frequency derivative of the material index.

        Parameters
        ----------
        omega : numpy.ndarray
            Finite positive angular frequencies in rad/s, shape ``(n_omega,)``.
        z : float, optional
            Finite position in m.

        Returns
        -------
        numpy.ndarray
            Derivative in s/rad, shape ``(n_omega,)``.

        Raises
        ------
        MaterialError
            If the evaluation point or result is invalid.
        """
        _check_evaluation(omega, z)
        if isinstance(self.material, Mixture):
            derivative = self.material.dsusceptibility_domega(omega, np.array([z]))[0]
            return np.asarray(
                derivative / (2 * self.index(omega, z)), dtype=np.complex128
            )
        if isinstance(self.material, GlassIndex):
            return self.material.dindex_domega(omega)
        if self.susceptibility is None:
            return np.zeros(omega.shape, dtype=np.complex128)
        density = float(self.density(np.array([z]))[0])
        derivative = density * self.susceptibility.dsusceptibility_domega(omega)
        return np.asarray(derivative / (2 * self.index(omega, z)), dtype=np.complex128)

    def at_density(self, density: float) -> "SpatialMedium":  # noqa: UP037 - bound after the class body
        """Freeze the scalar density for a parameter-expansion sample.

        Parameters
        ----------
        density : float
            Finite nonnegative number density in m^-3 for a gas. Nongas media
            require their dimensionless density of 1.

        Returns
        -------
        SpatialMedium
            The same material with a constant density path.

        Raises
        ------
        MaterialError
            If density is invalid or changes the nongas normalisation.
        """
        if not math.isfinite(density) or density < 0:
            msg = "spatial density must be finite and nonnegative"
            raise MaterialError(msg)
        if isinstance(self.material, Mixture):
            if len(self.material.components) != 1:
                raise MaterialError(
                    "mixture density sampling requires labelled parameters"
                )
            return self.at_parameters({"density": density})
        if self.susceptibility is None and density != 1:
            msg = "nongas spatial media require dimensionless density 1"
            raise MaterialError(msg)
        result = copy.copy(self)
        object.__setattr__(result, "density", Constant(density))
        return result

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return every material path that changes the spatial operator.

        Returns
        -------
        collections.abc.Mapping
            A shared pressure in bar or independent labelled densities in m^-3.
            A single population retains the original density parameter.
        """
        if not isinstance(self.material, Mixture) or len(self.material.components) == 1:
            return {"density": self.density}
        if self.material.shared is not None:
            return {"pressure": self.material.shared}
        return {f"density:{c.label}": c.density for c in self.material.components}

    def at_parameters(self, values: Mapping[str, float]) -> "SpatialMedium":  # noqa: UP037 - bound after class creation
        """Freeze the named material parameters for an operator sample.

        Parameters
        ----------
        values : collections.abc.Mapping
            Values of the declared pressure or density parameters, in bar or m^-3.

        Returns
        -------
        SpatialMedium
            A sampled medium with unchanged species data.

        Raises
        ------
        MaterialError
            If a name is not declared or a value is invalid.
        """
        if set(values) - self.parameter_paths.keys():
            raise MaterialError(f"unknown spatial material parameters {tuple(values)}")
        if not isinstance(self.material, Mixture):
            return self.at_density(
                values.get("density", float(self.density(np.array([0.0]))[0]))
            )
        material = self.material
        if "density" in values:
            mixture = material.with_densities({material.labels[0]: values["density"]})
        else:
            mixture = material.at_parameters(values)
        return SpatialMedium(mixture)


@dataclass(frozen=True, eq=False)
class SpatialDispersion:
    """Evaluate exact longitudinal propagation in the existing temporal frame.

    Attributes
    ----------
    medium : SpatialMedium
        Bulk material index and its scalar density dependence.
    grid : solprop.grid.TimeGrid
        Temporal frequency grid; only its active band samples the material.
    frame : solprop.linear.Frame
        Frame at zero transverse frequency, shared by all spatial columns.
    evanescent_rate : float
        Maximum evanescent field attenuation in m^-1, finite and positive.
        Default 200.

    Raises
    ------
    MaterialError
        If the attenuation cap or frame reference is invalid.
    """

    medium: SpatialMedium
    grid: TimeGrid
    frame: Frame
    evanescent_rate: float = EVANESCENT_RATE

    def __post_init__(self) -> None:
        """Check the attenuation cap and the temporal reference."""
        if not math.isfinite(self.evanescent_rate) or self.evanescent_rate <= 0:
            msg = "evanescent_rate must be finite and positive"
            raise MaterialError(msg)
        if not self.grid.omega.min() <= self.frame.omega0 <= self.grid.omega.max():
            msg = "spatial frame reference must lie inside the temporal frequency grid"
            raise MaterialError(msg)

    def frame_quantities(self, z: float = 0.0) -> tuple[float, float]:
        """Evaluate the carrier and group-delay terms at zero transverse frequency.

        Parameters
        ----------
        z : float, optional
            Finite position in m.

        Returns
        -------
        tuple of float
            Removed ``beta0`` in m^-1 and ``beta1`` in s/m, following
            :class:`solprop.linear.Frame`, including its fixed overrides.
        """
        reference = np.array([self.frame.omega0])
        index = float(np.real(self.medium.index(reference, z))[0])
        derivative = float(np.real(self.medium.dindex_domega(reference, z))[0])
        beta0 = self.frame.beta0
        beta1 = self.frame.beta1
        if beta0 is None:
            beta0 = self.frame.omega0 * index / C
        if beta1 is None:
            beta1 = (index + self.frame.omega0 * derivative) / C
        return (beta0 if self.frame.subtract_beta0 else 0.0), beta1

    def shared_values(
        self, z: float = 0.0
    ) -> tuple[
        Complex[np.ndarray, " n_omega"],
        Float[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
    ]:
        """Evaluate frequency-only material and normalisation coefficients.

        Parameters
        ----------
        z : float, optional
            Finite position in m.

        Returns
        -------
        tuple of numpy.ndarray
            Wave-number square in m^-2, frame subtraction in m^-1, numerator
            ``-i*mu0*omega**2/2``, and clamped prefactor ``-i*omega/2``. All have
            shape ``(n_omega,)``; physical coefficients vanish outside the active
            band. Temporal transform scalings are not included.
        """
        grid = self.grid
        active = grid.active_mask
        omega = grid.omega[active]
        square = np.zeros(grid.n_omega, dtype=np.complex128)
        square[active] = (omega * self.medium.index(omega, z) / C) ** 2
        beta0, beta1 = self.frame_quantities(z)
        origin = self.frame.omega0 if grid.kind == "envelope" else 0.0
        frame = beta0 + beta1 * (grid.omega - origin)
        numerator = np.zeros(grid.n_omega, dtype=np.complex128)
        clamped = np.zeros_like(numerator)
        # Use the same epsilon_0 as modal UPPE normalisation. Independently rounded
        # CODATA mu_0 and epsilon_0 violate mu_0*epsilon_0*c^2=1 at 1.2e-12 on
        # the qualification environment, beyond the unchanged roundoff contract.
        numerator[active] = -0.5j * omega**2 / (EPS0 * C**2)
        clamped[active] = -0.5j * omega
        return square, frame, numerator, clamped

    def cartesian_squares(
        self, space: FreeGrid, z: float = 0.0
    ) -> tuple[
        Complex[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
        Float[np.ndarray, " n_modes"],
    ]:
        """Split Cartesian wave-number squares without losing cutoff information.

        Parameters
        ----------
        space : solprop.grid.FreeGrid
            Stored unsquared transverse axes in rad/m, with x fastest.
        z : float, optional
            Finite material sampling position in m.

        Returns
        -------
        tuple of numpy.ndarray
            Material high and low parts in m^-2, each complex and frequency-sized,
            and the real residual of ``space.kperp_sq`` in m^-2. Residuals are
            rounded from exact rational arithmetic on the stored binary inputs.
            Inactive material bins are zero. No spatial-frequency table is built.
        """
        square, low = self.material_squares(z)
        x_squared = [Fraction(float(value)) ** 2 for value in space.k_x]
        y_squared = [Fraction(float(value)) ** 2 for value in space.k_y]
        transverse_low = np.array(
            [
                float(x + y - Fraction(float(high)))
                for (x, y), high in zip(
                    ((x, y) for y in y_squared for x in x_squared),
                    space.kperp_sq,
                    strict=True,
                )
            ]
        )
        return square, low, transverse_low

    def material_squares(
        self, z: float = 0.0
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Split the material wave-number square into its rounded value and residual.

        Parameters
        ----------
        z : float, optional
            Finite material sampling position in m.

        Returns
        -------
        tuple of numpy.ndarray
            ``(omega n / c)**2`` in m^-2 and its residual, the exact rational square
            of the stored binary inputs minus the rounded one, each complex and
            frequency-sized. Inactive bins are zero.
        """
        active = self.grid.active_mask
        omega = self.grid.omega[active]
        index = self.medium.index(omega, z)
        square = np.zeros(self.grid.n_omega, complex)
        square[active] = (omega * index / C) ** 2
        low = np.zeros_like(square)
        for bin_index, frequency, material in zip(
            np.flatnonzero(active), omega, index, strict=True
        ):
            scale = (Fraction(float(frequency)) / Fraction(C)) ** 2
            real = Fraction(float(material.real))
            imag = Fraction(float(material.imag))
            value = square[bin_index]
            low[bin_index] = complex(
                float(scale * (real**2 - imag**2) - Fraction(float(value.real))),
                float(2 * scale * real * imag - Fraction(float(value.imag))),
            )
        return square, low

    def _longitudinal_square(
        self, transverse: FreeGrid | Float[np.ndarray, " n_modes"], z: float
    ) -> Complex[np.ndarray, "n_modes n_omega"]:
        """Preserve unsquared Cartesian inputs when evaluating the launch operator.

        Parameters
        ----------
        transverse : solprop.grid.FreeGrid or numpy.ndarray
            Cartesian axes or already squared transverse frequencies in m^-2.
        z : float
            Material sampling position in m.

        Returns
        -------
        numpy.ndarray
            Complex longitudinal squares in m^-2, one row per transverse column.
        """
        if isinstance(transverse, FreeGrid):
            square, low, transverse_low = self.cartesian_squares(transverse, z)
            return (square[None, :] - transverse.kperp_sq[:, None]) + (
                low[None, :] - transverse_low[:, None]
            )
        _check_transverse(transverse)
        square, _, _, _ = self.shared_values(z)
        return square[None, :] - transverse[:, None]

    def operator(
        self, kperp_sq: FreeGrid | Float[np.ndarray, " n_modes"], z: float = 0.0
    ) -> Complex[np.ndarray, "n_modes n_omega"]:
        """Evaluate the exact longitudinal rate before spectral boundary loss.

        Parameters
        ----------
        kperp_sq : numpy.ndarray or solprop.grid.FreeGrid
            Finite nonnegative transverse wave-number squares in m^-2, or a
            Cartesian grid whose unsquared axes preserve cutoff precision.
        z : float, optional
            Finite position in m.

        Returns
        -------
        numpy.ndarray
            Rate in m^-1, shape ``(n_modes, n_omega)``. The positive-real square
            root propagates with material loss/gain. Nonpositive real longitudinal
            square attenuates with the declared evanescent cap.

        Raises
        ------
        MaterialError
            If a transverse frequency is negative or nonfinite.
        """
        _, frame, _, _ = self.shared_values(z)
        longitudinal_square = self._longitudinal_square(kperp_sq, z)
        root = np.sqrt(longitudinal_square)
        loss = np.imag(root).copy()
        evanescent = np.real(longitudinal_square) <= 0
        loss[evanescent] = np.minimum(np.abs(loss[evanescent]), self.evanescent_rate)
        return (-loss - 1j * (np.real(root) - frame)) * self.grid.active_mask

    def prefactor(
        self, kperp_sq: FreeGrid | Float[np.ndarray, " n_modes"], z: float = 0.0
    ) -> Complex[np.ndarray, "n_modes n_omega"]:
        """Evaluate the spatial UPPE polarisation prefactor before temporal scaling.

        Parameters
        ----------
        kperp_sq : numpy.ndarray or solprop.grid.FreeGrid
            Finite nonnegative transverse wave-number squares in m^-2, or a
            Cartesian grid whose unsquared axes preserve cutoff precision.
        z : float, optional
            Finite position in m.

        Returns
        -------
        numpy.ndarray
            Purely imaginary ``-i*mu0*omega**2/(2*Re(k_z))``, shape
            ``(n_modes, n_omega)``. The literal evanescent norm of 1 gives
            ``-i*omega/2`` in that band; inactive frequencies are zero.

        Raises
        ------
        MaterialError
            If a transverse frequency is negative or nonfinite.
        """
        _, _, numerator, clamped = self.shared_values(z)
        longitudinal_square = self._longitudinal_square(kperp_sq, z)
        propagating = np.real(longitudinal_square) > 0
        # The clamped branch never divides: a unit stands in for its denominator.
        denominator = np.where(propagating, np.real(np.sqrt(longitudinal_square)), 1.0)
        result = np.where(propagating, numerator / denominator, clamped)
        return result * self.grid.active_mask


def _check_evaluation(omega: np.ndarray, z: float) -> None:
    """Reject nonphysical frequency samples and nonfinite positions.

    Parameters
    ----------
    omega : numpy.ndarray
        Physical angular frequencies in rad/s.
    z : float
        Propagation position in m.
    """
    if not np.all(np.isfinite(omega)) or np.any(omega <= 0) or not math.isfinite(z):
        msg = "spatial index needs finite positive frequencies and a finite position"
        raise MaterialError(msg)


def _check_transverse(kperp_sq: np.ndarray) -> None:
    """Reject nonfinite or negative squared transverse wave numbers.

    Parameters
    ----------
    kperp_sq : numpy.ndarray
        Transverse wave-number squares in m^-2.
    """
    if not np.all(np.isfinite(kperp_sq)) or np.any(kperp_sq < 0):
        msg = "transverse wave-number squares must be finite and nonnegative"
        raise MaterialError(msg)


def _check_field(
    grid: TimeGrid, space: RadialGrid | FreeGrid | PlanarGrid, spectrum: np.ndarray
) -> None:
    """Validate the dimensions and finiteness of a spatial spectrum.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid specifying the frequency count.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        A :class:`solprop.modes.slab.PlanarGrid` is refused by name: its columns
        are slab amplitudes, not field samples.
        Transverse geometry specifying the column count.
    spectrum : numpy.ndarray
        Internal spatial spectrum, shape (n_modes, n_pol, n_omega).
    """
    if isinstance(space, PlanarGrid):
        # A planar column is a slab amplitude per y frequency, not a field sample:
        # these helpers would read it as the Cartesian grid's DFT and be wrong.
        msg = (
            "this spatial helper reads a radial or Cartesian field spectrum, and a "
            "PlanarGrid's columns are slab amplitudes per y frequency: use "
            "planar_mode_energy for the energy and planar_fluence for the fluence"
        )
        raise GridError(msg)
    space.validate()
    if (
        spectrum.ndim != 3
        or spectrum.shape[0] != space.n_columns
        or spectrum.shape[1] not in (1, 2)
        or spectrum.shape[2] != grid.n_omega
        or not np.all(np.isfinite(spectrum))
    ):
        msg = "spatial spectrum must be finite with shape (n_modes, n_pol, n_omega)"
        raise GridError(msg)


def _analytic(grid: TimeGrid, spectrum: np.ndarray) -> np.ndarray:
    """Reconstruct the analytic field with the common temporal FFT convention.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal transform layout and sampling.
    spectrum : numpy.ndarray
        Internal temporal spectrum with frequency on the final axis.

    Returns
    -------
    numpy.ndarray
        Complex analytic samples with time on the final axis.
    """
    samples = fourier.inverse(spectrum, kind=grid.kind)
    return fourier.analytic_signal(samples) if grid.kind == "field" else samples


FluxSide = Literal["before", "after"]
"""The side of a declared index jump a flux plane is evaluated on (SEP 34 §4.4)."""

_FLUX_CHUNK = 2**21
"""Complex elements per column chunk of an integrated flux sum (32 MiB)."""


def plane_sample(medium: SpatialMedium, z: float, side: FluxSide | None) -> float:
    """Return where to evaluate the medium for a plane, checking its jump side.

    A one-sided limit at a declared index jump is the material one rounding unit to
    that side; away from a jump the plane's own position.

    Parameters
    ----------
    medium : SpatialMedium
        The bulk medium.
    z : float
        Finite position of the plane in m.
    side : {'before', 'after'} or None
        Side of a declared jump at ``z``: required there and refused elsewhere.

    Returns
    -------
    float
        The position at which to evaluate the medium, in m.

    Raises
    ------
    MaterialError
        If the position is not finite, or the side is missing, misplaced or invalid.
    """
    if not math.isfinite(z):
        msg = "a flux plane needs a finite position"
        raise MaterialError(msg)
    if side not in (None, "before", "after"):
        msg = "side must be 'before', 'after' or None"
        raise MaterialError(msg)
    jump = any(
        np.any(np.asarray(path.jumps) == z) for path in medium.parameter_paths.values()
    )
    if jump and side is None:
        msg = (
            f"the medium jumps at z = {z!r} m: a flux plane there must say which "
            "side it measures, side='before' or side='after'"
        )
        raise MaterialError(msg)
    if side is not None and not jump:
        msg = (
            f"side={side!r} names a side of an index jump, and the medium has no "
            f"declared jump at z = {z!r} m"
        )
        raise MaterialError(msg)
    if side is None:
        return z
    return float(np.nextafter(z, -np.inf if side == "before" else np.inf))


@dataclass(frozen=True, eq=False)
class SpatialFlux:
    r"""The axial-flux measure of one physical plane (SEP 34).

    Binds the temporal and transverse representation, the medium and the plane's
    position once. The scalar forward equation's axial admittance
    :math:`\mathcal Y_c=\varepsilon_0c^2k_z/\omega` of the medium at that plane
    measures the field: its real part :math:`Y_z` weighs integrated energy, and the
    complex value reconstructs the local magnetic closure :math:`G=\mathcal Y_cE`
    for signed local flux :math:`S_z=\tfrac12\operatorname{Re}(E_aG_a^*)`. Only
    active propagating columns at positive frequency are forward power ports; every
    other bin has zero admittance. Build one with :meth:`resolve`; a different plane
    needs a new record (``docs/guide/free-space/flux.md``).

    Attributes
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid of the field.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse geometry of the field.
    medium : SpatialMedium
        The bulk medium; for a planar grid, its slab modes' own fill.
    z : float
        Position of the plane in m.
    side : {'before', 'after'} or None
        Side of a declared index jump at ``z``; None away from one.
    coefficients : object
        The plane's separable coefficients
        (:class:`solprop.translate.spatial.SeparableOperator`), resolved once.
    """

    grid: TimeGrid
    space: RadialGrid | FreeGrid | PlanarGrid
    medium: SpatialMedium
    z: float
    side: FluxSide | None
    coefficients: object

    @classmethod
    def resolve(
        cls,
        grid: TimeGrid,
        space: RadialGrid | FreeGrid | PlanarGrid,
        medium: SpatialMedium,
        *,
        z: float = 0.0,
        side: FluxSide | None = None,
        evanescent_rate: float = EVANESCENT_RATE,
    ) -> "SpatialFlux":  # noqa: UP037 - beartype evaluates the class body
        """Resolve the measure of the plane at ``z`` in ``medium``.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Temporal grid of the field.
        space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
            Transverse geometry of the field.
        medium : SpatialMedium
            The bulk medium at the plane; for a planar grid, the modes' own fill.
        z : float, optional
            Finite position of the plane in m. Default 0.
        side : {'before', 'after'} or None, optional
            The side of a declared index jump at ``z``. Required there, and refused
            away from one.
        evanescent_rate : float, optional
            Evanescent attenuation cap of the run, in m^-1.

        Returns
        -------
        SpatialFlux
            The resolved, immutable measure.

        Raises
        ------
        MaterialError
            If the position is not finite, or a jump side is missing or misplaced.
        """
        from solprop.translate.flux import plane_coefficients  # noqa: PLC0415

        sample = plane_sample(medium, z, side)
        coefficients = plane_coefficients(
            space, medium, grid, sample, evanescent_rate=evanescent_rate
        )
        return cls(grid, space, medium, z, side, coefficients)

    @property
    def energy_scale(self) -> float:
        """Return the factor turning weighted ``|X|^2`` sums into joules.

        Returns
        -------
        float
            ``eps0 c / 2`` for a field in V/m; one for planar slab amplitudes, whose
            mode functions already carry it.
        """
        return 1.0 if isinstance(self.space, PlanarGrid) else 0.5 * EPS0 * C

    def admittance(
        self, columns: slice = slice(None)
    ) -> Complex[np.ndarray, "n_chunk n_omega"]:
        """Return the dimensionless forward admittance ``c k_z / omega`` of columns.

        Parameters
        ----------
        columns : slice, optional
            Transverse columns; all by default.

        Returns
        -------
        numpy.ndarray
            ``Y_c / (eps0 c)`` in the optics convention (positive imaginary part for
            loss), zero on nonpropagating, inactive and nonpositive-frequency bins,
            shape ``(n_chunk, n_omega)``.
        """
        from solprop.translate.spatial import axial_admittance  # noqa: PLC0415

        return axial_admittance(
            cast("SeparableOperator", self.coefficients),
            self.grid.omega,
            columns,
        )

    def chunks(self) -> list[slice]:
        """Return column slices bounding an integrated sum's working memory.

        Returns
        -------
        list of slice
            Consecutive slices covering every transverse column.
        """
        n_columns = self.space.spectral_weights.size
        size = max(1, _FLUX_CHUNK // self.grid.n_omega)
        return [
            slice(start, min(start + size, n_columns))
            for start in range(0, n_columns, size)
        ]


def _check_flux(
    spectrum: np.ndarray, flux: SpatialFlux, *, planar: bool | None = None
) -> None:
    """Validate a spectrum against its flux record.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.
    planar : bool or None, optional
        True requires a planar grid, False refuses one, None admits either.

    Raises
    ------
    GridError
        If the shape or finiteness is invalid or the grid kind is refused.
    """
    space = flux.space
    if isinstance(space, PlanarGrid):
        if planar is False:
            msg = (
                "this spatial helper reads a radial or Cartesian field spectrum, and a "
                "PlanarGrid's columns are slab amplitudes per y frequency: use "
                "planar_fluence or planar_intensity"
            )
            raise GridError(msg)
        _check_planar(flux.grid, space, spectrum)
        return
    if planar:
        msg = "this helper reads a PlanarGrid's slab amplitudes"
        raise GridError(msg)
    _check_field(flux.grid, space, spectrum)


def _column_power(
    spectrum: np.ndarray, flux: SpatialFlux
) -> Float[np.ndarray, " n_modes"]:
    """Return each column's weighted axial energy, summed over polarisation.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        ``scale * v_j * sum_k e_k Re(Y_jk) sum_p |X_jpk|^2`` in J (J/m on a line),
        one per column.
    """
    grid = flux.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    power = np.empty(spectrum.shape[0])
    for columns in flux.chunks():
        real = np.real(flux.admittance(columns))
        density = np.sum(np.abs(spectrum[columns]) ** 2, axis=1)
        power[columns] = np.sum(weights * real * density, axis=-1)
    return flux.energy_scale * flux.space.spectral_weights * power


def spatial_energy(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> float:
    r"""Integrate the physical energy crossing the plane, in joules.

    :math:`U=\frac{\Delta t}{2N_t}\sum_{j,p,k}v_j\gamma_kY_{z,jk}|\hat E_{jpk}|^2`
    (SEP 34 §4.3): the axial Poynting flux of the medium at the plane, integrated
    over the plane and time. Only the real admittance enters.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m, Cartesian spectrum in V/m or planar slab
        amplitudes, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure (:meth:`SpatialFlux.resolve`).

    Returns
    -------
    float
        Energy in J, summed over the beam and polarisations; in J per metre of y
        on a :class:`solprop.grid.LineGrid`.

    Raises
    ------
    GridError
        If the spectrum's shape or finiteness is invalid.
    """
    _check_flux(spectrum, flux)
    return float(np.sum(_column_power(spectrum, flux)))


def planar_mode_energy(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_p"]:
    r"""Integrate a planar beam's physical energy in each slab mode.

    The explicit slab metric of SEP 34 §4.3:
    :math:`U_p=\frac{\Delta t}{N_t}\frac{\Delta y}{N_y}\sum_{\ell,k}\gamma_kw_{p\ell k}
    |\hat A_{p\ell k}|^2` with :math:`w=c\operatorname{Re}q/\omega` and
    :math:`q=\sqrt{(\omega n_{{\rm eff},p}/c)^2-k_y^2}`. At ``k_y = 0`` the weight is
    the mode's real effective index, not the bulk index.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal ``(p, k_y)`` spectrum of slab amplitudes in sqrt(W/m), shape
        ``(n_p*n_y, 1, n_omega)``, the channel fastest.
    flux : SpatialFlux
        The plane's measure on a :class:`solprop.modes.slab.PlanarGrid`.

    Returns
    -------
    numpy.ndarray
        Energy in J per mode, shape ``(n_p,)``, in the modes' order. Their sum is
        :func:`spatial_energy`.

    Raises
    ------
    GridError
        If the spectrum is invalid or the grid is not planar.
    """
    _check_flux(spectrum, flux, planar=True)
    space = cast("PlanarGrid", flux.space)
    return np.sum(_column_power(spectrum, flux).reshape(space.n_y, space.n_p), axis=0)


def power_trace(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
    admittance: Complex[np.ndarray, "n_modes n_omega"],
    weights: Float[np.ndarray, " n_modes"],
    grid: TimeGrid,
    scale: float,
) -> Float[np.ndarray, " n_t"]:
    r"""Return the axial power through a plane in time from column spectra.

    Transverse Parseval turns the area integral of
    :math:`\tfrac12\operatorname{Re}(E_aG_a^*)` into a weighted sum over spectral
    columns of their temporal analytic signals, so no transverse synthesis is needed.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal column spectra, shape ``(n_modes, n_pol, n_omega)``.
    admittance : numpy.ndarray
        Dimensionless optics-convention admittance ``c k_z / omega`` per column and
        bin, shape ``(n_modes, n_omega)``.
    weights : numpy.ndarray
        Transverse Parseval weight of each column.
    grid : solprop.grid.TimeGrid
        Temporal grid.
    scale : float
        ``eps0 c / 2`` for fields in V/m, one for slab amplitudes.

    Returns
    -------
    numpy.ndarray
        Power in W (W/m on a line), shape ``(n_t,)``, signed and not clipped.
    """
    field = _analytic(grid, spectrum)
    closure = _analytic(grid, np.conj(admittance)[:, None, :] * spectrum)
    product = np.real(field * np.conj(closure))
    return scale * np.einsum("j,jpt->t", weights, product)


def spatial_power(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_t"]:
    """Return the axial power crossing the plane at every time sample.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m, Cartesian spectrum in V/m or planar slab
        amplitudes, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Cycle-averaged power in W (W/m on a line), shape ``(n_t,)``. Its time
        integral is :func:`spatial_energy`.

    Raises
    ------
    GridError
        If the spectrum's shape or finiteness is invalid.
    """
    _check_flux(spectrum, flux)
    total = np.zeros(flux.grid.n_t)
    for columns in flux.chunks():
        total += power_trace(
            spectrum[columns],
            flux.admittance(columns),
            flux.space.spectral_weights[columns],
            flux.grid,
            flux.energy_scale,
        )
    return total


def _synthesise(
    space: RadialGrid | FreeGrid | PlanarGrid, spectrum: np.ndarray
) -> np.ndarray:
    """Return a spectrum's physical field in V/m at the transverse nodes.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse geometry.
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        Frequency-domain field at the nodes, shape ``(n_nodes, n_pol, n_omega)``;
        planar nodes are flattened ``ix + n_x*iy``.
    """
    if isinstance(space, RadialGrid):
        return np.moveaxis(space.hankel.inverse(np.moveaxis(spectrum, 0, -1)), -1, 0)
    if isinstance(space, PlanarGrid):
        rows = np.fft.ifft(spectrum[:, 0].reshape(space.n_y, space.n_p, -1), axis=0)
        physical = np.einsum("xp,ypw->yxw", space.synthesis, rows)
        return physical.reshape(-1, 1, spectrum.shape[-1])
    return np.fft.ifft2(
        spectrum.reshape(space.n_y, space.n_x, *spectrum.shape[1:]), axes=(0, 1)
    ).reshape(spectrum.shape)


def _closure(spectrum: np.ndarray, flux: SpatialFlux) -> np.ndarray:
    """Return the internal spectrum of the magnetic closure ``G / (eps0 c)``.

    In the internal transform convention the stored spectrum is the complex
    conjugate of the optics one (``docs/dev/conventions.md`` §2), so the closure
    applies the conjugate admittance.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Spectrum of the same shape.
    """
    closure = np.empty_like(spectrum, dtype=np.complex128)
    for columns in flux.chunks():
        closure[columns] = (
            np.conj(flux.admittance(columns))[:, None, :] * (spectrum[columns])
        )
    return closure


def _local_pair(
    spectrum: np.ndarray, flux: SpatialFlux
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise E and its closure at the nodes, in the frequency domain.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    tuple of numpy.ndarray
        ``E`` in V/m and ``G / (eps0 c)`` in V/m, each ``(n_nodes, n_pol, n_omega)``.
    """
    return _synthesise(flux.space, spectrum), _synthesise(
        flux.space, _closure(spectrum, flux)
    )


def _signed_fluence(
    spectrum: np.ndarray, flux: SpatialFlux
) -> tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_nodes"]]:
    """Return the signed fluence and its rounding budget at the nodes.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    tuple of numpy.ndarray
        Fluence in J/m^2 at the nodes and a bound on its accumulated rounding,
        the same at every node.
    """
    grid = flux.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    field, closure = _local_pair(spectrum, flux)
    scale = 0.5 * EPS0 * C
    fluence = scale * np.sum(weights * np.real(field * np.conj(closure)), axis=(1, 2))
    magnitude = scale * np.sum(weights * np.abs(field) * np.abs(closure), axis=(1, 2))
    # The transverse syntheses sum every column, so each node's value carries
    # rounding relative to the plane's largest product, not its own: where the
    # beam has decayed, E and G are uncorrelated rounding residue. A few units per
    # term of the plane's scale bound the error of every node.
    budget = (
        8
        * (spectrum.shape[1] * grid.n_omega + spectrum.shape[0])
        * np.finfo(float).eps
        * np.max(magnitude, initial=0.0)
    )
    return fluence, np.full(fluence.shape, budget)


def spatial_fluence(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_nodes"]:
    r"""Return the signed time-integrated axial flux at the spatial nodes.

    :math:`F=\tfrac12\int\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*)\,dt`, with
    the complex admittance applied before the transverse synthesis and temporal
    Parseval for the time integral. Broadband or absorbing fields can give a
    negative local value; it is not clipped.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m, Cartesian spectrum in V/m or planar slab
        amplitudes, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Fluence in J/m^2 at the spatial nodes (planar nodes flattened
        ``ix + n_x*iy``).

    Raises
    ------
    GridError
        If the spectrum's shape or finiteness is invalid.
    """
    _check_flux(spectrum, flux, planar=False)
    return _signed_fluence(spectrum, flux)[0]


def spatial_intensity(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, "n_nodes n_t"]:
    r"""Return the signed local axial flux at every spatial node and time sample.

    :math:`S_z=\tfrac12\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*)` in W/m^2,
    with ``G`` reconstructed from the complex admittance in space and time. It is
    the axial Poynting flux of the declared scalar forward model, not
    :math:`|E_a|^2`, and it is not clipped.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m, Cartesian spectrum in V/m or planar slab
        amplitudes, shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Flux in W/m^2, shape ``(n_nodes, n_t)``.

    Raises
    ------
    GridError
        If the spectrum's shape or finiteness is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid, RadialGrid
    >>> from solprop.spatial import SpatialFlux, SpatialMedium, spatial_intensity
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 600e-15)
    >>> space = RadialGrid(1e-3, 16)
    >>> flux = SpatialFlux.resolve(grid, space, SpatialMedium("Ar", pressure=1.0))
    >>> spectrum = np.zeros((16, 1, grid.n_omega), complex)
    >>> spatial_intensity(spectrum, flux).shape == (16, grid.n_t)
    True
    """
    _check_flux(spectrum, flux, planar=False)
    return _local_intensity(spectrum, flux)


def _local_intensity(spectrum: np.ndarray, flux: SpatialFlux) -> np.ndarray:
    """Return the signed local axial flux at the nodes in time.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``, already validated.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Flux in W/m^2, shape ``(n_nodes, n_t)``.
    """
    field, closure = _local_pair(spectrum, flux)
    grid = flux.grid
    product = _analytic(grid, field) * np.conj(_analytic(grid, closure))
    return 0.5 * EPS0 * C * np.sum(np.real(product), axis=1)


def planar_fluence(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_nodes"]:
    """Return a planar beam's signed time-integrated axial flux at its nodes.

    The coherent synthesis of SEP 34 §4.3: ``E`` and ``G`` are both built from the
    slab mode functions, ``G`` with each column's complex admittance
    ``eps0 c**2 q / omega`` applied first, so the modes' local interference remains.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal ``(p, k_y)`` spectrum of slab amplitudes in sqrt(W/m), shape
        ``(n_p*n_y, 1, n_omega)``.
    flux : SpatialFlux
        The plane's measure on a :class:`solprop.modes.slab.PlanarGrid`.

    Returns
    -------
    numpy.ndarray
        Fluence in J/m^2 at the nodes ``(x[ix], y[iy])``, flattened
        ``ix + n_x*iy``, shape ``(n_x*n_y,)``.

    Raises
    ------
    GridError
        If the spectrum is invalid or the grid is not planar.
    """
    _check_flux(spectrum, flux, planar=True)
    return _signed_fluence(spectrum, flux)[0]


def planar_intensity(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, "n_nodes n_t"]:
    """Return a planar beam's signed local axial flux at its nodes in time.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal ``(p, k_y)`` spectrum of slab amplitudes in sqrt(W/m), shape
        ``(n_p*n_y, 1, n_omega)``.
    flux : SpatialFlux
        The plane's measure on a :class:`solprop.modes.slab.PlanarGrid`.

    Returns
    -------
    numpy.ndarray
        Flux in W/m^2 at the nodes ``(x[ix], y[iy])``, flattened ``ix + n_x*iy``,
        shape ``(n_x*n_y, n_t)``.

    Raises
    ------
    GridError
        If the spectrum is invalid or the grid is not planar.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.modes import PlanarGrid, SlabMode
    >>> from solprop.spatial import SpatialFlux, SpatialMedium, planar_intensity
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 600e-15)
    >>> space = PlanarGrid([SlabMode(63.5e-6)], 1e-3, 8)
    >>> flux = SpatialFlux.resolve(grid, space, SpatialMedium(1.0))
    >>> spectrum = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    >>> planar_intensity(spectrum, flux).shape == (space.x.size * 8, grid.n_t)
    True
    """
    _check_flux(spectrum, flux, planar=True)
    return _local_intensity(spectrum, flux)


def peak_field(
    grid: TimeGrid,
    space: RadialGrid | FreeGrid | PlanarGrid,
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> float:
    """Return the largest physical analytic field magnitude at any node and time.

    A named electric-field diagnostic, ``max sqrt(sum_pol |E_a|^2)`` in V/m: the
    physical field that drives ionisation rates, not an axial flux (SEP 34).

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse geometry.
    spectrum : numpy.ndarray
        Internal radial spectrum in V m, Cartesian spectrum in V/m or planar slab
        amplitudes, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    float
        Peak field amplitude in V/m.

    Raises
    ------
    GridError
        If the spectrum's shape or finiteness is invalid.
    """
    if isinstance(space, PlanarGrid):
        _check_planar(grid, space, spectrum)
    else:
        _check_field(grid, space, spectrum)
    analytic = _analytic(grid, _synthesise(space, spectrum))
    return float(np.sqrt(np.max(np.sum(np.abs(analytic) ** 2, axis=1))))


def _check_planar(
    grid: TimeGrid, space: PlanarGrid, spectrum: np.ndarray
) -> np.ndarray:
    """Validate a planar ``(p, k_y)`` spectrum and return it as y rows.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid specifying the frequency count.
    space : solprop.modes.slab.PlanarGrid
        The slab channels and the y axis.
    spectrum : numpy.ndarray
        Internal spectrum, shape ``(n_p*n_y, 1, n_omega)``, the channel fastest.

    Returns
    -------
    numpy.ndarray
        The same state as slab amplitudes in sqrt(W/m) at each y sample, shape
        ``(n_y, n_p, n_omega)``.
    """
    if spectrum.shape != (space.n_columns, 1, grid.n_omega) or not np.all(
        np.isfinite(spectrum)
    ):
        msg = (
            "a planar spectrum must be finite with shape (n_p*n_y, 1, n_omega) = "
            f"{(space.n_columns, 1, grid.n_omega)}"
        )
        raise GridError(msg)
    return np.fft.ifft(spectrum[:, 0].reshape(space.n_y, space.n_p, -1), axis=0)


def _on_axis_pair(
    spectrum: np.ndarray, flux: SpatialFlux
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct E and its closure at the geometric axis.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    tuple of numpy.ndarray
        On-axis temporal spectra of ``E`` and ``G / (eps0 c)`` in V/m, each
        ``(n_pol, n_omega)``.
    """
    _check_flux(spectrum, flux)
    space = flux.space
    # J0(0)=1. The inverse Hankel measure is k dk, whereas Parseval's stored
    # spectral measure carries 2 pi, so reconstruction divides it out.
    weights = (
        space.spectral_weights / (2 * np.pi)
        if isinstance(space, RadialGrid)
        else space.on_axis
    )
    return np.einsum("r,rpw->pw", weights, spectrum), np.einsum(
        "r,rpw->pw", weights, _closure(spectrum, flux)
    )


def on_axis_intensity(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> float:
    """Return the largest signed axial flux in time on the beam axis.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    float
        Peak on-axis axial flux in W/m^2, summed over polarisation.

    Raises
    ------
    GridError
        If the spectrum is invalid.
    """
    field, closure = _on_axis_pair(spectrum, flux)
    grid = flux.grid
    product = np.real(_analytic(grid, field) * np.conj(_analytic(grid, closure)))
    return float(0.5 * EPS0 * C * np.max(np.sum(product, axis=0)))


def on_axis_spectrum(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_omega"]:
    """Return the on-axis spectral fluence density summed over polarisation.

    The same-frequency real pairing of the coherent on-axis ``E`` and ``G``, not a
    squared coherent field sum: each transverse column carries its own admittance.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Real spectral fluence in J s/m^2 per angular frequency bin, shape
        ``(n_omega,)``. Its sum times ``grid.d_omega`` is the on-axis fluence.
        It is a density, so changing Fourier convention leaves it unchanged.

    Raises
    ------
    GridError
        If the spectrum is invalid.
    """
    field, closure = _on_axis_pair(spectrum, flux)
    grid = flux.grid
    factor = fourier.spectral_density(
        np.ones(grid.n_omega, complex), grid.dt, kind=grid.kind
    )
    return 0.5 * EPS0 * C * factor * np.sum(np.real(field * np.conj(closure)), axis=0)


# ----------------------------------------------- far-field apertures (SEP 23)

ApertureProfile = Literal["hard", "tanh", "rcos", "supergauss"]
"""The edge profile of a far-field aperture (SEP 23 section 4.4)."""

APERTURE_PROFILES: tuple[ApertureProfile, ...] = ("hard", "tanh", "rcos", "supergauss")
"""Every aperture edge profile, in the order of their integer profile codes."""

EDGE_MOMENTS: Mapping[str, float] = MappingProxyType(
    {"hard": 0.0, "tanh": math.pi**2 / 48, "rcos": 0.25 - 2 / math.pi**2}
)
r"""The first moment ``m_2`` of each symmetric edge's unit-width outer tail.

:math:`m_2=\int_0^\infty t\,W_1(t)\,dt`, a quarter of the edge kernel's variance in
units of the width squared: :math:`\pi^2/48` for ``tanh`` (the integral of
:math:`t/(1+e^{2t})`) and :math:`1/4-2/\pi^2` for ``rcos``, as SEP 23 section 4.5 and
``docs/guide/results/statistics.md`` (*Far-field apertures and angular regions*) derive
them; the tests check both against quadrature. ``hard`` needs no correction;
``supergauss`` has no symmetric edge and so no entry.
"""

FAR_FIELD_STATISTICS: tuple[str, ...] = (
    "aperture_spectrum",
    "aperture_on_axis_spectrum",
    "region_spectrum",
    "beam_spectrum",
    "absorbed_power",
)
"""The five statistics :func:`aperture_statistics` forms, in catalogue order."""

OPERATOR_STATISTICS: tuple[str, ...] = ("absorbed_power",)
"""The far-field statistic that needs the run's linear operator.

A saved plane does not carry the operator's absorption rates, so it is a backend
statistic and :func:`solprop.stats.collect` refuses it by name.
"""

ABSORBED_CHANNELS: tuple[str, ...] = (
    "spectral",
    "transverse",
    "evanescent",
    "material",
)
"""The channels of ``absorbed_power``, in the order of its four entries."""

AxialFactors = Callable[[slice], tuple[np.ndarray, np.ndarray | None]]
"""Per-chunk admittance and absorption rates of the plane's medium (SEP 34).

Called with a slice of transverse columns, it returns the dimensionless complex
admittance ``c k_z/omega``, shape ``(n_chunk, n_omega)``, zero on bins that are no
forward power port, and the four power rates of :data:`ABSORBED_CHANNELS` in 1/m,
shape ``(4, n_chunk, n_omega)``, or None where the run's operator is not available
(a saved plane). :func:`solprop.translate.spatial.axial_factors` is the in-run
implementation and :meth:`SpatialFlux.admittance` the saved-plane one.
"""

_FAR_FIELD_BUDGET = 64 * 2**20
"""Bytes of per-chunk temporaries :func:`aperture_statistics` holds when not told."""


def check_aperture_profile(kind: object) -> None:
    """Refuse, by name, anything that is not one of the aperture edge profiles.

    The one check behind :func:`aperture_profile`, :class:`ApertureTables` and
    :class:`solprop.stats.ApertureSet`. It takes any object, because a caller
    outside the test session's runtime type checker can pass anything.

    Parameters
    ----------
    kind : object
        The profile asked for.

    Raises
    ------
    OutputError
        If `kind` is not in :data:`APERTURE_PROFILES`.

    Examples
    --------
    >>> from solprop.spatial import check_aperture_profile
    >>> check_aperture_profile("rcos") is None
    True
    """
    if kind not in APERTURE_PROFILES:
        msg = f"unknown aperture profile {kind!r}; choose one of {APERTURE_PROFILES}"
        raise OutputError(msg)


def aperture_profile(
    radius: ArrayLike | float,
    half_width: float,
    edge: float,
    kind: ApertureProfile,
) -> Float[np.ndarray, "*shape"]:
    r"""Return an aperture's profile ``W`` at distances from its centre.

    How ``W`` enters depends on the aperture set's weighting (SEP 23 section 4.6):
    under ``"amplitude"`` weighting (the default) it is the field transmission, and
    ``aperture_spectrum`` weighs the collected power by ``W**2``; under
    ``"quadrature"`` weighting it weighs the power itself, once. The on-axis
    amplitude behind the aperture applies ``W`` once under both weightings.

    With ``r`` the half-width and ``t = (rho - r)/w``, the four profiles of SEP 23
    section 4.4 are: ``hard``, one for ``rho <= r`` and zero beyond; ``tanh``,
    :math:`\tfrac12[1-\tanh t]`; ``rcos``, :math:`\tfrac12[1-\sin(\pi t/2)]` with
    ``t`` clipped to ``[-1, 1]``, exactly one inside ``r - w`` and exactly zero
    outside ``r + w``; and ``supergauss``, :math:`\exp[-(\rho/r)^n]`. ``tanh`` and
    ``rcos`` satisfy the edge symmetry :math:`W(r+u)+W(r-u)=1`. The argument is
    formed as ``(rho - r) * (1/w)``, the product with the reciprocal a compiled pass
    stores, so every backend evaluates the same argument.

    Parameters
    ----------
    radius : array_like or float
        Distances :math:`\rho` from the aperture's centre in m, nonnegative.
    half_width : float
        The profile's half-width ``r`` in m, finite and positive: ``D/2`` under
        amplitude weighting, the corrected ``D_q/2`` under quadrature weighting
        (:func:`quadrature_half_width`).
    edge : float
        The edge width ``w`` in m for ``tanh`` and ``rcos`` (finite and positive),
        the order ``n`` for ``supergauss`` (finite, at least 2); not used for
        ``hard``.
    kind : {'hard', 'tanh', 'rcos', 'supergauss'}
        The edge profile.

    Returns
    -------
    numpy.ndarray
        The dimensionless profile ``W(rho)`` in ``[0, 1]``, the shape of `radius`.

    Raises
    ------
    OutputError
        If the profile is unknown, the half-width is not finite and positive, or the
        edge is not valid for the profile.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial import aperture_profile
    >>> aperture_profile(np.array([0.0, 1.0, 1.5]), 1.0, 0.0, "hard").tolist()
    [1.0, 1.0, 0.0]
    >>> float(aperture_profile(1.0, 1.0, 0.1, "tanh"))
    0.5
    >>> aperture_profile(np.array([0.85, 1.0, 1.15]), 1.0, 0.1, "rcos").tolist()
    [1.0, 0.5, 0.0]
    """
    check_aperture_profile(kind)
    if not math.isfinite(half_width) or half_width <= 0:
        msg = f"an aperture half-width must be finite and positive, got {half_width}"
        raise OutputError(msg)
    rho = np.asarray(radius, dtype=np.float64)
    if kind == "hard":
        return np.asarray(np.where(rho <= half_width, 1.0, 0.0))
    if kind == "supergauss":
        if not math.isfinite(edge) or edge < 2:
            msg = (
                f"a supergauss aperture needs a finite order of at least 2, got {edge}"
            )
            raise OutputError(msg)
        return np.asarray(np.exp(-((rho / half_width) ** edge)))
    if not math.isfinite(edge) or edge <= 0:
        msg = f"a {kind} aperture needs a finite positive edge width, got {edge}"
        raise OutputError(msg)
    argument = (rho - half_width) * (1.0 / edge)
    if kind == "tanh":
        return np.asarray(0.5 * (1.0 - np.tanh(argument)))
    clipped = np.clip(argument, -1.0, 1.0)
    # Exactly one and zero beyond the edge, whatever sin(+-pi/2) rounds to: the
    # compact support is a structural property a pass may rely on (the early-out).
    edge_values = 0.5 * (1.0 - np.sin((0.5 * math.pi) * clipped))
    return np.asarray(
        np.where(argument <= -1.0, 1.0, np.where(argument >= 1.0, 0.0, edge_values))
    )


def quadrature_half_width(
    half_width: ArrayLike | float, width: ArrayLike | float, kind: ApertureProfile
) -> Float[np.ndarray, "*shape"]:
    r"""Return the half-width at which a profile used once integrates a hard hole.

    Equation (6) of SEP 23: :math:`D_q/2=\sqrt{(D/2)^2-4m_2w^2}`, with :math:`m_2` from
    :data:`EDGE_MOMENTS`. Evaluated there, a symmetric edge's two-dimensional area
    :math:`\pi(r_q^2+\sigma^2)` is the hard hole's :math:`\pi r^2` exactly (for
    ``tanh`` up to its exponential tail). On a line the edges are straight and the
    caller passes the half-width unchanged.

    Parameters
    ----------
    half_width : array_like or float
        The hard hole's half-width ``D/2`` in m, positive.
    width : array_like or float
        The edge width ``w`` in m, positive; not used for ``hard``.
    kind : {'hard', 'tanh', 'rcos'}
        The edge profile; ``supergauss`` has no symmetric edge.

    Returns
    -------
    numpy.ndarray
        ``D_q/2`` in m, the broadcast shape of the inputs.

    Raises
    ------
    OutputError
        For ``supergauss``, or where the radicand is not positive.

    Examples
    --------
    >>> import math
    >>> from solprop.spatial import quadrature_half_width
    >>> value = float(quadrature_half_width(1.0, 0.1, "rcos"))
    >>> math.isclose(value, math.sqrt(1 - 4 * (0.25 - 2 / math.pi**2) * 0.01))
    True
    >>> float(quadrature_half_width(1.0, 0.1, "hard"))
    1.0
    """
    if kind not in EDGE_MOMENTS:
        msg = (
            f"quadrature weighting needs a symmetric edge, and the {kind!r} profile "
            "has none; use amplitude weighting"
        )
        raise OutputError(msg)
    radius = np.asarray(half_width, dtype=np.float64)
    edge = np.asarray(width, dtype=np.float64)
    if kind == "hard":
        return np.broadcast_to(radius, np.broadcast(radius, edge).shape).copy()
    radicand = radius * radius - (4.0 * EDGE_MOMENTS[kind]) * (edge * edge)
    if np.any(radicand <= 0):
        msg = "the quadrature correction needs (D/2)^2 > 4 m_2 w^2 for every aperture"
        raise OutputError(msg)
    return np.asarray(np.sqrt(radicand))


def aperture_scale(
    distances: Float[np.ndarray, " n_apertures"],
    omega: Float[np.ndarray, " n_omega"],
) -> Float[np.ndarray, "n_apertures n_omega"]:
    """Return the paraxial mask-plane scale ``s(omega) = z c / omega`` of each aperture.

    Equation (3) of SEP 23: a plane-wave component of transverse wavevector ``kappa``
    reaches the mask plane at ``x = s kappa``. The table is formed as
    ``(z * c) / omega`` and is exactly zero at and below zero frequency, where the
    mapping does not exist and every aperture weight is zero.

    Parameters
    ----------
    distances : numpy.ndarray
        Exit face to mask plane (or a lens's focal length) of each aperture, in m,
        shape ``(n_apertures,)``.
    omega : numpy.ndarray
        Physical angular frequencies in rad/s, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The scale in m^2, shape ``(n_apertures, n_omega)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import C
    >>> from solprop.spatial import aperture_scale
    >>> table = aperture_scale(np.array([0.5]), np.array([-1.0, 0.0, C]))
    >>> table.tolist()
    [[0.0, 0.0, 0.5]]
    """
    numerator = np.asarray(distances, dtype=np.float64)[:, None] * C
    frequencies = np.asarray(omega, dtype=np.float64)
    scale = np.zeros((numerator.shape[0], frequencies.size))
    np.divide(numerator, frequencies, out=scale, where=frequencies > 0)
    return scale


def mask_plane_radius(
    scale: Float[np.ndarray, " n_omega"],
    k_x: Float[np.ndarray, " n_chunk"],
    k_y: Float[np.ndarray, " n_chunk"],
    centre: Sequence[float] | np.ndarray,
) -> Float[np.ndarray, "n_chunk n_omega"]:
    r"""Return how far each column's mask-plane image lies from an aperture's centre.

    Equation (4) of SEP 23,
    :math:`\rho=\sqrt{(s k^{(s)}_x+X)^2+(s k^{(s)}_y+Y)^2}`, with the **stored**
    column axes :math:`k^{(s)}` (the physical wavevector is their negation, section
    4.1) and the physical centre ``(X, Y)``. It is evaluated as written -- products,
    sums, two squares, a sum and a correctly rounded square root, never ``hypot`` --
    so a hard edge decides identically on every backend that does the same.

    Parameters
    ----------
    scale : numpy.ndarray
        The aperture's row of :func:`aperture_scale`, m^2, shape ``(n_omega,)``.
    k_x, k_y : numpy.ndarray
        Each column's stored transverse wavevector in rad/m, shape ``(n_chunk,)``.
    centre : sequence of float or numpy.ndarray
        The aperture centre ``(X, Y)`` in the mask plane, physical axes, in m.

    Returns
    -------
    numpy.ndarray
        Distances in m, shape ``(n_chunk, n_omega)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial import mask_plane_radius
    >>> scale, k_x, k_y = np.array([1.0]), np.array([-3.0]), np.zeros(1)
    >>> radius = mask_plane_radius(scale, k_x, k_y, (3.0, 4.0))
    >>> radius.tolist()
    [[4.0]]
    """
    across = scale[None, :] * k_x[:, None] + float(centre[0])
    along = scale[None, :] * k_y[:, None] + float(centre[1])
    return np.sqrt(across * across + along * along)


@dataclass(frozen=True, eq=False)
class ApertureTables:
    """Hold the constants the far-field statistics read, formed once (SEP 23 §6).

    :func:`solprop.translate.spatial.aperture_tables` builds it from a statistics
    request, a frequency axis and a Cartesian geometry; every backend and
    :func:`solprop.stats.collect` then read the same values, the scale table among
    them to the bit. Equality is by identity.

    Attributes
    ----------
    k_x, k_y : numpy.ndarray
        The stored transverse axes in rad/m, FFT order, shapes ``(n_x,)`` and
        ``(n_y,)``; column ``ix + n_x*iy`` carries ``(k_x[ix], k_y[iy])``.
    weights : numpy.ndarray
        The geometry's Parseval spectral weights, m^2 (m on a line), shape
        ``(n_x*n_y,)``.
    on_axis : numpy.ndarray
        The geometry's complex on-axis reconstruction weights, shape ``(n_x*n_y,)``.
    regions : numpy.ndarray
        Boxes of physical transverse wavevector in rad/m, rows ``(kx_lo, kx_hi,
        ky_lo, ky_hi)``, shape ``(n_regions, 4)``; empty when none was asked for.
    centres : numpy.ndarray
        Aperture centres in the mask plane, physical axes, in m, shape
        ``(n_apertures, 2)``.
    half_widths : numpy.ndarray
        The profiles' half-widths in m, ``D/2`` or ``D_q/2``, shape
        ``(n_apertures,)``.
    edges : numpy.ndarray
        Edge widths in m (``tanh``, ``rcos``), orders (``supergauss``), or zeros
        (``hard``), shape ``(n_apertures,)``.
    distances : numpy.ndarray
        Exit face to mask plane in m, shape ``(n_apertures,)``.
    scale : numpy.ndarray
        :func:`aperture_scale` of the distances, m^2, shape ``(n_apertures,
        n_omega)``, zero at and below zero frequency.
    profile : {'hard', 'tanh', 'rcos', 'supergauss'}
        The edge profile the set shares.
    exponent : int
        The power of the weight in ``aperture_spectrum``: 2 for amplitude
        weighting, 1 for quadrature weighting.

    Raises
    ------
    OutputError
        If the arrays disagree in shape, the profile is unknown or the exponent is
        not 1 or 2.
    """

    k_x: Float[np.ndarray, " n_kx"]
    k_y: Float[np.ndarray, " n_ky"]
    weights: Float[np.ndarray, " n_weights"]
    on_axis: Complex[np.ndarray, " n_axis_weights"]
    regions: Float[np.ndarray, "n_region_rows 4"]
    centres: Float[np.ndarray, "n_centre_rows 2"]
    half_widths: Float[np.ndarray, " n_half_widths"]
    edges: Float[np.ndarray, " n_edge_values"]
    distances: Float[np.ndarray, " n_distances"]
    scale: Float[np.ndarray, "n_scale_rows n_scale_omega"]
    profile: ApertureProfile = "hard"
    exponent: int = 2

    def __post_init__(self) -> None:
        """Own read-only copies and check that the tables agree with each other."""
        for name in (
            "k_x",
            "k_y",
            "weights",
            "on_axis",
            "regions",
            "centres",
            "half_widths",
            "edges",
            "distances",
            "scale",
        ):
            values = np.array(
                getattr(self, name),
                dtype=np.complex128 if name == "on_axis" else np.float64,
                order="C",
                copy=True,
            )
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        n_apertures = self.half_widths.size
        if (
            self.weights.shape != (self.k_x.size * self.k_y.size,)
            or self.on_axis.shape != self.weights.shape
            or self.regions.ndim != 2
            or self.regions.shape[1] != 4
            or self.centres.shape != (n_apertures, 2)
            or self.edges.shape != (n_apertures,)
            or self.distances.shape != (n_apertures,)
            or self.scale.ndim != 2
            or self.scale.shape[0] != n_apertures
        ):
            msg = "aperture tables disagree in shape with each other or the columns"
            raise OutputError(msg)
        check_aperture_profile(self.profile)
        if self.exponent not in (1, 2):
            msg = f"aperture tables need an exponent of 1 or 2, got {self.exponent!r}"
            raise OutputError(msg)

    @property
    def n_columns(self) -> int:
        """Return the number of transverse columns the tables describe.

        Returns
        -------
        int
            ``n_x*n_y``.
        """
        return int(self.weights.size)

    @property
    def n_apertures(self) -> int:
        """Return the number of apertures.

        Returns
        -------
        int
            The rows of the aperture tables, zero when none was asked for.
        """
        return int(self.half_widths.size)

    @property
    def n_regions(self) -> int:
        """Return the number of regions of transverse wavevector.

        Returns
        -------
        int
            The rows of `regions`.
        """
        return int(self.regions.shape[0])

    @property
    def profile_code(self) -> int:
        """Return the profile's integer code, its index in :data:`APERTURE_PROFILES`.

        Returns
        -------
        int
            0 for ``hard``, 1 for ``tanh``, 2 for ``rcos``, 3 for ``supergauss``.
        """
        return APERTURE_PROFILES.index(self.profile)

    @property
    def inverse_widths(self) -> Float[np.ndarray, " n_apertures"]:
        """Return the reciprocal edge widths a compiled pass multiplies by.

        Returns
        -------
        numpy.ndarray
            ``1/w`` in 1/m for ``tanh`` and ``rcos``, zeros otherwise, shape
            ``(n_apertures,)``: the factor :func:`aperture_profile` forms itself.
        """
        if self.profile in ("tanh", "rcos"):
            return 1.0 / self.edges
        return np.zeros(self.n_apertures)


def _aperture_weight(
    tables: ApertureTables,
    index: int,
    k_x: Float[np.ndarray, " n_chunk"],
    k_y: Float[np.ndarray, " n_chunk"],
) -> Float[np.ndarray, "n_chunk n_omega"]:
    """Return one aperture's weight on a chunk of columns at every frequency.

    Parameters
    ----------
    tables : ApertureTables
        The far-field tables.
    index : int
        The aperture.
    k_x, k_y : numpy.ndarray
        The chunk's stored transverse wavevectors in rad/m, shape ``(n_chunk,)``.

    Returns
    -------
    numpy.ndarray
        ``W_a(rho_a(c, k))``, exactly zero at and below zero frequency and wherever
        the column's image lies at or beyond the mask plane's distance from the
        origin (SEP 23 sections 4.2 and 4.10), shape ``(n_chunk, n_omega)``.
    """
    scale = tables.scale[index]
    radius = mask_plane_radius(scale, k_x, k_y, tables.centres[index])
    weight = aperture_profile(
        radius,
        float(tables.half_widths[index]),
        float(tables.edges[index]),
        tables.profile,
    )
    # |kappa| >= omega/c lands at or beyond z_m and cannot leave the exit face; the
    # unbounded tanh and supergauss tails would otherwise reach those columns.
    origin = mask_plane_radius(scale, k_x, k_y, (0.0, 0.0))
    outside = (origin >= tables.distances[index]) | (scale <= 0)[None, :]
    return np.where(outside, 0.0, weight)


def aperture_statistics(
    names: Sequence[str],
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    tables: ApertureTables,
    *,
    axial: AxialFactors,
    chunk: int | None = None,
) -> dict[str, np.ndarray]:
    r"""Form the far-field statistics of one Cartesian or line spectrum (SEP 23).

    The reference pass of SEP 23's equations, measured by the axial flux of the
    medium (SEP 34): the spectrum collected through each aperture,
    :math:`S_a=\tfrac{\varepsilon_0c}{2}\sum_cw_cW_a^p\eta_c\sum_q
    \operatorname{sd}(E)` with the real admittance
    :math:`\eta=c\operatorname{Re}k_z/\omega`; the on-axis spectrum behind each
    aperture in a Fourier plane, the same-frequency pairing of the coherent
    :math:`A_{a,q}=\sum_c\alpha_cW_aE_{c,q}` with its magnetic closure; the spectrum
    in each box of physical transverse wavevector; the beam spectrum; and the power
    the spectral, transverse, evanescent and material channels of the linear
    operator remove, :math:`P_j=\tfrac{\varepsilon_0c}{2}\sum_cw_c\sum_{q,k}
    \nu_k\eta r_j|E|^2`. It walks the columns in chunks, forms the
    polarisation-summed spectral density once per chunk and adds the chunk partials
    in ascending order.

    Parameters
    ----------
    names : collections.abc.Sequence of str
        The statistics wanted, a subset of :data:`FAR_FIELD_STATISTICS`.
    grid : solprop.grid.TimeGrid
        The temporal grid: its spacing and kind fix the spectral density and the
        energy weights.
    field : numpy.ndarray
        One internal Cartesian spectrum in V/m (raw transverse DFT), shape
        ``(n_x*n_y, n_pol, n_omega)``.
    tables : ApertureTables
        The far-field tables of the request and the geometry.
    axial : callable
        The plane's per-chunk admittance and the operator's rates
        (:data:`AxialFactors`); the rates are needed by ``absorbed_power``.
    chunk : int or None, optional
        Columns per chunk, at least 1; ``None`` bounds the chunk's temporaries at
        64 MiB.

    Returns
    -------
    dict
        Each requested statistic: ``aperture_spectrum`` and ``region_spectrum`` in
        J/(rad/s), shapes ``(n_apertures, n_omega)`` and ``(n_regions, n_omega)``;
        ``aperture_on_axis_spectrum`` in J s/m^2, ``(n_apertures, n_omega)``;
        ``beam_spectrum`` in J/(rad/s), ``(n_omega,)``;
        ``absorbed_power`` in J/m, ``(4,)``, in the order of
        :data:`ABSORBED_CHANNELS`. On a line the spectra and powers are per metre
        of y, and the on-axis spectra stay local.

    Raises
    ------
    OutputError
        If a name is not a far-field statistic, the field does not match the
        tables, a requested statistic needs a table the request does not carry, or
        an operator statistic is asked for without the operator's rates.
    """
    wanted = _far_field_request(names, grid, field, tables)
    n_pol = field.shape[1]
    step = (
        max(1, _FAR_FIELD_BUDGET // (16 * (n_pol + 4) * grid.n_omega))
        if chunk is None
        else chunk
    )
    if step < 1:
        msg = f"a far-field chunk must hold at least one column, got {step}"
        raise OutputError(msg)
    sums = _FarFieldSums(wanted, grid, tables, n_pol)
    for first in range(0, tables.n_columns, step):
        stop = min(first + step, tables.n_columns)
        sums.add(first, field[first:stop], axial)
    return sums.values()


def _far_field_request(
    names: Sequence[str],
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    tables: ApertureTables,
) -> set[str]:
    """Check a far-field request against the field and its tables.

    Parameters
    ----------
    names : collections.abc.Sequence of str
        The statistics wanted.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    field : numpy.ndarray
        The spectrum, shape ``(n_x*n_y, n_pol, n_omega)``.
    tables : ApertureTables
        The far-field tables.

    Returns
    -------
    set of str
        The names, checked.

    Raises
    ------
    OutputError
        As :func:`aperture_statistics` documents.
    """
    wanted = set(names)
    unknown = sorted(wanted - set(FAR_FIELD_STATISTICS))
    if unknown:
        msg = f"{unknown} are not far-field statistics; they are {FAR_FIELD_STATISTICS}"
        raise OutputError(msg)
    n_omega = grid.n_omega
    if (
        field.ndim != 3
        or field.shape[0] != tables.n_columns
        or field.shape[1] not in (1, 2)
        or field.shape[2] != n_omega
        or (tables.n_apertures and tables.scale.shape[1] != n_omega)
    ):
        msg = (
            f"a far-field statistic needs a spectrum of shape ({tables.n_columns}, "
            f"n_pol, {n_omega}) matching its tables, got {field.shape}"
        )
        raise OutputError(msg)
    through = wanted & _THROUGH_APERTURES
    if through and tables.n_apertures == 0:
        msg = f"{sorted(through)} need apertures, and the request carries none"
        raise OutputError(msg)
    if "region_spectrum" in wanted and tables.n_regions == 0:
        msg = "region_spectrum needs spectral_regions, and the request carries none"
        raise OutputError(msg)
    return wanted


_THROUGH_APERTURES = frozenset({"aperture_spectrum", "aperture_on_axis_spectrum"})
"""The far-field statistics that read the aperture tables."""


class _FarFieldSums:
    """Accumulate the far-field partials chunk by chunk, in ascending order.

    Parameters
    ----------
    wanted : set of str
        The far-field statistics asked for.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    tables : ApertureTables
        The far-field tables.
    n_pol : int
        The spectrum's polarisation components.
    """

    def __init__(
        self, wanted: set[str], grid: TimeGrid, tables: ApertureTables, n_pol: int
    ) -> None:
        """Allocate one accumulator per requested quantity."""
        n_omega = grid.n_omega
        self._wanted = wanted
        self._grid = grid
        self._tables = tables
        self._beam = np.zeros(n_omega)
        self._absorbed = np.zeros(len(ABSORBED_CHANNELS))
        self._regions = np.zeros((tables.n_regions, n_omega))
        self._collected = np.zeros((tables.n_apertures, n_omega))
        self._amplitude = np.zeros(
            (tables.n_apertures, n_pol, n_omega), dtype=np.complex128
        )
        self._closure = np.zeros_like(self._amplitude)
        self._energy_weights = fourier.energy_weights(n_omega, grid.dt, kind=grid.kind)

    def add(
        self,
        first: int,
        spectrum: Complex[np.ndarray, "n_chunk n_pol n_omega"],
        axial: AxialFactors,
    ) -> None:
        """Add one chunk of columns, starting at column `first`.

        Parameters
        ----------
        first : int
            The chunk's first column.
        spectrum : numpy.ndarray
            The chunk of the spectrum, shape ``(n_chunk, n_pol, n_omega)``.
        axial : callable
            The plane's per-chunk admittance and rates.

        Raises
        ------
        OutputError
            If ``absorbed_power`` is wanted and the rates are not available.
        """
        tables, grid, wanted = self._tables, self._grid, self._wanted
        stop = first + spectrum.shape[0]
        columns = np.arange(first, stop)
        admittance, rates = axial(slice(first, stop))
        real = np.real(admittance)
        density = np.sum(fourier.spectral_density(spectrum, grid.dt, kind=grid.kind), 1)
        weighted = tables.weights[first:stop, None] * real * density
        k_x = tables.k_x[columns % tables.k_x.size]
        k_y = tables.k_y[columns // tables.k_x.size]
        if "beam_spectrum" in wanted:
            self._beam += np.sum(weighted, axis=0)
        if "region_spectrum" in wanted:
            self._add_regions(weighted, k_x, k_y)
        if wanted & _THROUGH_APERTURES:
            self._add_apertures(first, spectrum, admittance, weighted, k_x, k_y)
        if "absorbed_power" in wanted:
            if rates is None:
                msg = (
                    "absorbed_power needs the run's linear operator rates, which a "
                    "saved plane does not carry; a backend forms it during the run"
                )
                raise OutputError(msg)
            squared = np.sum(np.abs(spectrum) ** 2, axis=1)
            measured = (
                tables.weights[first:stop, None] * real * squared
            ) * self._energy_weights
            self._absorbed += np.sum(rates * measured[None], axis=(1, 2))

    def _add_regions(
        self,
        weighted: Float[np.ndarray, "n_chunk n_omega"],
        k_x: Float[np.ndarray, " n_chunk"],
        k_y: Float[np.ndarray, " n_chunk"],
    ) -> None:
        """Add a chunk's weighted density to every region its columns belong to.

        Parameters
        ----------
        weighted : numpy.ndarray
            ``w_c`` times the polarisation-summed spectral density.
        k_x, k_y : numpy.ndarray
            The chunk's stored transverse wavevectors in rad/m.
        """
        # Membership compares the physical wavevector, an exact negation of the
        # stored axes, with half-open edges, so regions that tile the plane tile the
        # columns (SEP 23 section 4.7).
        kappa_x, kappa_y = -k_x, -k_y
        for row, (x_lo, x_hi, y_lo, y_hi) in enumerate(self._tables.regions):
            member = (
                (x_lo <= kappa_x)
                & (kappa_x < x_hi)
                & (y_lo <= kappa_y)
                & (kappa_y < y_hi)
            )
            self._regions[row] += np.sum(weighted[member], axis=0)

    def _add_apertures(
        self,
        first: int,
        spectrum: Complex[np.ndarray, "n_chunk n_pol n_omega"],
        admittance: Complex[np.ndarray, "n_chunk n_omega"],
        weighted: Float[np.ndarray, "n_chunk n_omega"],
        k_x: Float[np.ndarray, " n_chunk"],
        k_y: Float[np.ndarray, " n_chunk"],
    ) -> None:
        """Add a chunk to every aperture's collected spectrum and on-axis amplitude.

        Parameters
        ----------
        first : int
            The chunk's first column.
        spectrum : numpy.ndarray
            The chunk of the spectrum.
        admittance : numpy.ndarray
            The chunk's complex admittance.
        weighted : numpy.ndarray
            ``w_c`` times the polarisation-summed spectral density.
        k_x, k_y : numpy.ndarray
            The chunk's stored transverse wavevectors in rad/m.
        """
        tables = self._tables
        stop = first + spectrum.shape[0]
        for index in range(tables.n_apertures):
            weight = _aperture_weight(tables, index, k_x, k_y)
            if "aperture_spectrum" in self._wanted:
                power = weight if tables.exponent == 1 else weight * weight
                self._collected[index] += np.sum(weighted * power, axis=0)
            if "aperture_on_axis_spectrum" in self._wanted:
                # The weight enters the amplitude once under both weightings; the
                # complex on-axis vector is multiplied as stored (section 4.6).
                through = weight[:, None, :] * spectrum
                self._amplitude[index] += np.einsum(
                    "r,rpw->pw", tables.on_axis[first:stop], through
                )
                # The same aperture on the magnetic closure: the internal spectrum
                # carries the conjugate admittance (SpatialFlux).
                self._closure[index] += np.einsum(
                    "r,rpw->pw",
                    tables.on_axis[first:stop],
                    np.conj(admittance)[:, None, :] * through,
                )

    def values(self) -> dict[str, np.ndarray]:
        """Return the requested statistics from the accumulated partials.

        Returns
        -------
        dict
            The statistics :func:`aperture_statistics` documents.
        """
        conversion = 0.5 * EPS0 * C
        grid = self._grid
        factor = fourier.spectral_density(
            np.ones(grid.n_omega, complex), grid.dt, kind=grid.kind
        )
        on_axis = factor * np.sum(
            np.real(self._amplitude * np.conj(self._closure)), axis=1
        )
        everything = {
            "aperture_spectrum": self._collected,
            "aperture_on_axis_spectrum": on_axis,
            "region_spectrum": self._regions,
            "beam_spectrum": self._beam,
            "absorbed_power": self._absorbed,
        }
        return {
            name: conversion * value
            for name, value in everything.items()
            if name in self._wanted
        }


def radial_fluence_widths(
    radius: ArrayLike,
    fluence: ArrayLike,
    weights: ArrayLike,
    axis_fluence: float,
) -> tuple[float, float, float]:
    """Return second-moment radius, first 1/e^2 radius and central FWHM.

    Parameters
    ----------
    radius : array_like
        Increasing positive radial samples in m.
    fluence : array_like
        Time-integrated intensity in J/m^2 at each radius.
    weights : array_like
        Positive transverse integration weights in m^2.
    axis_fluence : float
        Independently reconstructed axial fluence in J/m^2.

    Returns
    -------
    tuple of float
        Second-moment radius, central 1/e^2 radius and FWHM in m. The threshold
        widths are the first outward crossing of a fraction of the peak fluence,
        the larger of the axis value and the samples; they are NaN when the axis
        fluence lies below that threshold (a hollow profile at that level) or when
        the trace never crosses it, as the statistics catalogue specifies.

    Raises
    ------
    GridError
        If samples disagree, radii or weights are invalid, fluence is negative,
        or the fluence moments cannot be represented by finite floats.
    """
    if any(np.iscomplexobj(one) for one in (radius, fluence, weights)):
        msg = "radial fluence samples must be real numbers"
        raise GridError(msg)
    try:
        radius, fluence, weights = (
            np.asarray(one, dtype=np.float64) for one in (radius, fluence, weights)
        )
    except (TypeError, ValueError) as error:
        msg = "radial fluence samples must be real numbers"
        raise GridError(msg) from error
    if (
        radius.ndim != 1
        or radius.size < 2
        or fluence.shape != radius.shape
        or weights.shape != radius.shape
        or not all(np.all(np.isfinite(one)) for one in (radius, fluence, weights))
        or not math.isfinite(axis_fluence)
        or axis_fluence < 0
        or np.any(radius <= 0)
        or np.any(np.diff(radius) <= 0)
        or np.any(weights <= 0)
        or np.any(fluence < 0)
    ):
        msg = (
            "radial fluence widths require matching finite samples, increasing "
            "positive radii, positive weights and nonnegative fluence"
        )
        raise GridError(msg)
    try:
        with np.errstate(over="raise", invalid="raise"):
            energy = float(np.sum(weights * fluence))
            moment = float(np.sum(weights * fluence * radius**2))
    except FloatingPointError as error:
        msg = "radial fluence moments are not representable"
        raise GridError(msg) from error
    if energy <= 0:
        return math.nan, math.nan, math.nan
    second = math.sqrt(2.0) * math.sqrt(moment / energy)
    # The thresholds are fractions of the peak. A flat-top or a truncated series
    # puts its first samples a ripple above the separately reconstructed axis
    # value, so the axis is not required to be the maximum; a profile is hollow
    # at a level only when its axis lies below that level, and then the width at
    # that level is undefined rather than the diameter of a ring.
    peak = max(axis_fluence, float(np.max(fluence)))
    values = np.concatenate(([axis_fluence], fluence))
    positions = np.concatenate(([0.0], radius))

    def crossing(fraction: float) -> float:
        """Interpolate the first outward threshold crossing.

        Parameters
        ----------
        fraction : float
            Threshold divided by the central fluence, between zero and one.

        Returns
        -------
        float
            Radius in m, or NaN when the axis lies below the threshold or the
            trace never crosses it.
        """
        level = fraction * peak
        if axis_fluence < level:
            return math.nan
        below = np.flatnonzero(values < level)
        if below.size == 0:
            return math.nan
        index = int(below[0])
        left, right = values[index - 1], values[index]
        return float(
            positions[index - 1]
            + (left - level)
            / (left - right)
            * (positions[index] - positions[index - 1])
        )

    return second, crossing(math.exp(-2)), 2 * crossing(0.5)


def cartesian_fluence_widths(
    x: ArrayLike,
    y: ArrayLike,
    fluence: ArrayLike,
) -> tuple[float, float, float]:
    """Measure centroid-based moments and radially averaged Cartesian widths.

    Parameters
    ----------
    x, y : array_like
        Uniform increasing physical axes in m. ``x`` has at least two samples, and
        so has ``y`` except on a line uniform in y (:class:`solprop.grid.LineGrid`),
        where ``y`` is its one sample and the widths are one-dimensional.
    fluence : array_like
        Finite nonnegative time-integrated intensity in J/m^2, flattened with
        x fastest, shape ``(n_x*n_y,)``.

    Returns
    -------
    tuple of float
        ``sqrt(2*mean(r**2))`` about the fluence centroid, the first 1/e^2
        radius, and FWHM in m. Empty beams return three NaNs. Ring profiles or
        thresholds outside the largest inscribed sampled circle have NaN
        interpolated widths. On a line the moment is ``2*sqrt(mean(x**2))``
        about the centroid, which is the 1/e^2 half-width of a Gaussian fluence
        as the plane's is its 1/e^2 radius, and the threshold widths average the
        two sides of the centroid.

    Raises
    ------
    GridError
        If axes, samples or finite moments are invalid.

    Notes
    -----
    Radial averages sample the bilinear interpolant at half the finest pixel
    spacing in radius and at an angular spacing no larger than that arc length
    on the largest ring. The manual derives the associated interpolation error;
    scalar widths do not silently select the x or y line through the peak.
    """
    if any(np.iscomplexobj(one) for one in (x, y, fluence)):
        msg = "Cartesian fluence samples must be real"
        raise GridError(msg)
    try:
        x, y, values = (np.asarray(one, dtype=np.float64) for one in (x, y, fluence))
    except (TypeError, ValueError) as error:
        msg = "Cartesian fluence samples must be real"
        raise GridError(msg) from error
    if (
        x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 1
        or values.shape != (x.size * y.size,)
        or not all(np.all(np.isfinite(one)) for one in (x, y, values))
        or np.any(values < 0)
    ):
        msg = (
            "Cartesian fluence widths require finite axes "
            "and matching nonnegative samples"
        )
        raise GridError(msg)
    for axis in (x, y) if y.size > 1 else (x,):
        spacing = (axis[-1] - axis[0]) / (axis.size - 1)
        if spacing <= 0 or np.any(
            np.abs(np.diff(axis) - spacing)
            > 32 * np.finfo(float).eps * max(float(np.max(np.abs(axis))), spacing)
        ):
            msg = "Cartesian fluence axes must be uniformly increasing"
            raise GridError(msg)
    maximum = float(np.max(values))
    if maximum == 0:
        return math.nan, math.nan, math.nan
    if y.size == 1:
        return _line_fluence_widths(x, values / maximum)
    # Scaling cancels in centroid and widths, and avoids overflow in bright beams.
    samples = (values / maximum).reshape(y.size, x.size)
    total = float(np.sum(samples))
    center_x = float(np.sum(np.sum(samples, axis=0) * x) / total)
    center_y = float(np.sum(np.sum(samples, axis=1) * y) / total)
    radius = np.hypot(x[None, :] - center_x, y[:, None] - center_y)
    extent = float(np.max(radius))
    # Distinct validated axes guarantee a positive extent, even for a point beam.
    second = extent * math.sqrt(
        2 * float(np.sum(samples * (radius / extent) ** 2)) / total
    )
    if not math.isfinite(second):
        msg = "Cartesian fluence moments are not representable"
        raise GridError(msg)
    limit = min(center_x - x[0], x[-1] - center_x, center_y - y[0], y[-1] - center_y)
    spacing = min(x[1] - x[0], y[1] - y[0]) / 2
    if limit < 2 * spacing:
        return second, math.nan, math.nan
    # Anchor the rule to the grid, not the rounded centroid: otherwise a
    # one-ulp change in the limiting radius can change every interpolation
    # point through ceil/linspace, amplifying roundoff into sampling error.
    radii = spacing * np.arange(math.floor(limit / spacing) + 1)
    largest_circle = min(x[-1] - x[0], y[-1] - y[0]) / 2
    n_angles = max(8, math.ceil(2 * np.pi * largest_circle / spacing))
    angles = 2 * np.pi * np.arange(n_angles) / n_angles
    averaged = np.empty(radii.size)
    for index, distance in enumerate(radii):
        horizontal = np.clip(
            (center_x + distance * np.cos(angles) - x[0]) / (x[1] - x[0]), 0, x.size - 1
        )
        vertical = np.clip(
            (center_y + distance * np.sin(angles) - y[0]) / (y[1] - y[0]), 0, y.size - 1
        )
        ix = np.minimum(horizontal.astype(np.intp), x.size - 2)
        iy = np.minimum(vertical.astype(np.intp), y.size - 2)
        fx, fy = horizontal - ix, vertical - iy
        interpolated = (1 - fy) * (
            (1 - fx) * samples[iy, ix] + fx * samples[iy, ix + 1]
        ) + fy * ((1 - fx) * samples[iy + 1, ix] + fx * samples[iy + 1, ix + 1])
        averaged[index] = np.mean(interpolated)
    _, radius_e2, width = radial_fluence_widths(
        radii[1:], averaged[1:], np.ones(radii.size - 1), float(averaged[0])
    )
    return second, radius_e2, width


def _line_fluence_widths(
    x: Float[np.ndarray, " n_x"], samples: Float[np.ndarray, " n_x"]
) -> tuple[float, float, float]:
    """Measure the one-dimensional widths of a line's fluence about its centroid.

    Parameters
    ----------
    x : numpy.ndarray
        Validated uniform increasing axis in m.
    samples : numpy.ndarray
        Fluence divided by its maximum, nonnegative and not all zero.

    Returns
    -------
    tuple of float
        ``2*sqrt(mean((x - x_c)**2))``, the first 1/e^2 half-width and the FWHM in
        m, the last two NaN when the centroid lies within two half-spacings of an
        edge or the profile does not cross the level.

    Raises
    ------
    GridError
        If the moment is not representable.
    """
    total = float(np.sum(samples))
    center = float(np.sum(samples * x) / total)
    offset = x - center
    extent = float(np.max(np.abs(offset)))
    # A Gaussian fluence exp(-2 x^2/w^2) has mean(x^2) = w^2/4: twice the root
    # of the moment is its 1/e^2 half-width w, as sqrt(2 mean(r^2)) is the
    # plane's 1/e^2 radius.
    second = (
        2 * extent * math.sqrt(float(np.sum(samples * (offset / extent) ** 2)) / total)
    )
    if not math.isfinite(second):
        msg = "line fluence moments are not representable"
        raise GridError(msg)
    spacing = (x[1] - x[0]) / 2
    limit = min(center - x[0], x[-1] - center)
    if limit < 2 * spacing:
        return second, math.nan, math.nan
    # The plane's azimuthal average has two directions on a line: sample the
    # linear interpolant at the same half-pixel radii on both sides.
    radii = spacing * np.arange(math.floor(limit / spacing) + 1)
    averaged = 0.5 * (
        np.interp(center - radii, x, samples) + np.interp(center + radii, x, samples)
    )
    _, radius_e2, width = radial_fluence_widths(
        radii[1:], averaged[1:], np.ones(radii.size - 1), float(averaged[0])
    )
    return second, radius_e2, width


def _beam_widths(spectrum: np.ndarray, flux: SpatialFlux) -> tuple[float, float, float]:
    """Reduce the signed fluence profile with its spatial measure.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure on a radial or Cartesian grid.

    Returns
    -------
    tuple of float
        Second-moment radius, central 1/e^2 radius and FWHM in m.

    Raises
    ------
    GridError
        If the fluence is negative anywhere beyond its rounding budget: a width
        needs a nonnegative profile, and a signed axial flux is not clipped.
    """
    _check_flux(spectrum, flux, planar=False)
    fluence, budget = _signed_fluence(spectrum, flux)
    if np.any(fluence < -budget):
        worst = int(np.argmin(fluence + budget))
        msg = (
            "a beam width needs a nonnegative fluence profile, and this field's "
            f"signed axial flux is {fluence[worst]:.3e} J/m^2 at node {worst}, "
            "beyond its rounding budget; the width is undefined for it"
        )
        raise GridError(msg)
    fluence = np.maximum(fluence, 0.0)
    space = flux.space
    if isinstance(space, FreeGrid):
        return cartesian_fluence_widths(space.x, space.y, fluence)
    axis_fluence = float(np.sum(on_axis_spectrum(spectrum, flux)) * flux.grid.d_omega)
    radial = cast("RadialGrid", space)
    return radial_fluence_widths(radial.r, fluence, radial.radial_weights, axis_fluence)


def beam_radius(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
    flux: SpatialFlux,
    *,
    method: Literal["second_moment", "1/e2"] = "second_moment",
) -> float:
    """Measure a beam radius from its time-integrated axial flux.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure on a radial or Cartesian grid.
    method : {'second_moment', '1/e2'}, optional
        ``sqrt(2 integral r^2 F dA / integral F dA)`` or the first linearly
        interpolated crossing of the central fluence divided by e^2.

    Returns
    -------
    float
        Radius in m. Empty beams return NaN; interpolated radii also return NaN
        for a profile whose axis fluence lies below the threshold (hollow at that
        level) or that does not cross it within the grid.

    Raises
    ------
    GridError
        If the method or spectrum is invalid, the grid is planar, or the fluence is
        negative beyond its rounding budget.
    """
    if method not in ("second_moment", "1/e2"):
        msg = "beam radius method must be second_moment or 1/e2"
        raise GridError(msg)
    values = _beam_widths(spectrum, flux)
    return values[0 if method == "second_moment" else 1]


def fwhm_r(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> float:
    """Return the central fluence profile's full width at half maximum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Internal radial spectrum in V m or Cartesian spectrum in V/m,
        shape ``(n_modes, n_pol, n_omega)``.
    flux : SpatialFlux
        The plane's measure on a radial or Cartesian grid.

    Returns
    -------
    float
        Twice the interpolated half-maximum radius in m, or NaN for an empty
        profile, one whose axis fluence lies below half the peak (hollow) or one
        that never falls below it.

    Raises
    ------
    GridError
        If the spectrum is invalid, the grid is planar, or the fluence is negative
        beyond its rounding budget.
    """
    return _beam_widths(spectrum, flux)[2]


def attenuation_extremes(
    square: Complex[np.ndarray, " n_omega"],
    kperp_sq: Float[np.ndarray, " n_modes"],
    *,
    square_low: Complex[np.ndarray, " n_omega"] | None = None,
    kperp_sq_low: Float[np.ndarray, " n_modes"] | None = None,
) -> tuple[
    Float[np.ndarray, " n_omega"],
    Float[np.ndarray, " n_omega"],
    Float[np.ndarray, " n_omega"],
]:
    r"""Return each frequency's extreme field attenuations over the columns.

    SEP 22 section 4.4. For :math:`q = k^2 - k_\perp^2` and the principal root
    :math:`s = \sqrt q`, :math:`\operatorname{Im}\sqrt z =
    \operatorname{sign}(\operatorname{Im}z)\sqrt{(|z| - \operatorname{Re}z)/2}` is
    monotone in :math:`k_\perp^2` at fixed frequency, so the extremes over the
    propagating columns (:math:`\operatorname{Re}q > 0`) lie at the smallest
    :math:`k_\perp^2` and the largest one below :math:`\operatorname{Re}k^2`, and the
    largest :math:`|\operatorname{Im}s|` over the evanescent columns at the largest
    :math:`k_\perp^2`. The columns are sorted once and searched per frequency, in
    :math:`O(n_\omega \log n_{\mathrm{modes}})`, and the candidates beside each
    cutoff are classified with the full split arithmetic.

    Parameters
    ----------
    square : numpy.ndarray
        The material wave-number square :math:`k^2(\omega)` in m^-2, complex, shape
        ``(n_omega,)``.
    kperp_sq : numpy.ndarray
        The columns' transverse squares in m^-2, nonnegative, shape ``(n_modes,)``.
    square_low, kperp_sq_low : numpy.ndarray or None, optional
        Their residuals in m^-2, as :class:`solprop.translate.spatial.SeparableOperator`
        stores them; ``None`` is zero.

    Returns
    -------
    tuple of numpy.ndarray
        Per frequency, each of shape ``(n_omega,)`` and in m^-1: the largest and the
        smallest :math:`\operatorname{Im}s` over the propagating columns (``-inf``
        and ``+inf`` where none propagates), and the largest uncapped
        :math:`|\operatorname{Im}s|` over the evanescent columns (zero where none is
        evanescent).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.spatial import attenuation_extremes
    >>> square = np.array([4.0 + 1e-3j])
    >>> largest, smallest, evanescent = attenuation_extremes(
    ...     square, np.array([0.0, 3.0, 5.0])
    ... )
    >>> float(evanescent[0]) == float(abs(np.sqrt(square[0] - 5.0).imag))
    True
    """
    square = np.asarray(square, dtype=np.complex128)
    kperp_sq = np.asarray(kperp_sq, dtype=np.float64)
    low = np.zeros_like(square) if square_low is None else np.asarray(square_low)
    kperp_low = (
        np.zeros_like(kperp_sq) if kperp_sq_low is None else np.asarray(kperp_sq_low)
    )
    order = np.lexsort((kperp_low, kperp_sq))
    sorted_high = kperp_sq[order]
    cutoff = np.searchsorted(sorted_high, np.real(square), side="left")
    # The columns beside each cutoff (rounding can move it by one), and both ends.
    offsets = np.array([-2, -1, 0, 1])
    candidates = np.concatenate(
        (
            np.clip(cutoff[:, None] + offsets[None, :], 0, kperp_sq.size - 1),
            np.zeros((square.size, 1), dtype=np.intp),
            np.full((square.size, 1), kperp_sq.size - 1, dtype=np.intp),
        ),
        axis=1,
    )
    columns = order[candidates]
    longitudinal = (square[:, None] - kperp_sq[columns]) + (
        low[:, None] - kperp_low[columns]
    )
    root_imag = np.imag(np.sqrt(longitudinal))
    propagating = np.real(longitudinal) > 0
    largest = np.max(np.where(propagating, root_imag, -np.inf), axis=1)
    smallest = np.min(np.where(propagating, root_imag, np.inf), axis=1)
    evanescent = np.max(np.where(propagating, 0.0, np.abs(root_imag)), axis=1)
    return largest, smallest, evanescent


def physical_evanescent_rate(
    space: FreeGrid | RadialGrid | PlanarGrid,
    medium: SpatialMedium,
    grid: TimeGrid,
    *,
    z: float = 0.0,
) -> float:
    r"""Return the largest physical evanescent field attenuation the grid holds.

    The largest :math:`|\operatorname{Im}k_z|` over the active frequencies and the
    columns whose :math:`\operatorname{Re}(k^2 - k_\perp^2) \le 0` (SEP 22 section
    4.4), from :func:`attenuation_extremes`: the rate at which a non-propagating
    component decays physically, which the declared model caps at
    :data:`EVANESCENT_RATE`. Pass it as ``SimulationSettings(evanescent_rate=...)``
    for a thin sample whose evanescent content must decay as it does; the exponent
    budget then bounds the replayed step
    (``docs/guide/free-space/model.md``).

    Parameters
    ----------
    space : solprop.grid.FreeGrid or solprop.grid.RadialGrid
        The transverse grid; a :class:`solprop.grid.LineGrid` is a ``FreeGrid``. A
        :class:`solprop.modes.slab.PlanarGrid`, whose columns are slab channels, is
        refused.
    medium : SpatialMedium
        The bulk medium.
    grid : solprop.grid.TimeGrid
        The temporal grid; its active band is searched.
    z : float, optional
        Where the medium is sampled, m. Default 0.

    Returns
    -------
    float
        The rate in m^-1. A grid without an evanescent pair returns
        :data:`EVANESCENT_RATE`, since the cap is then inert.

    Raises
    ------
    GridError
        If `space` is neither Cartesian nor radial.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid, FreeGrid
    >>> from solprop.spatial import SpatialMedium, physical_evanescent_rate
    >>> grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (700e-9, 900e-9), 200e-15)
    >>> space = FreeGrid(radius_x=20e-6, n_x=64, radius_y=20e-6, n_y=64)
    >>> rate = physical_evanescent_rate(space, SpatialMedium(1.45 + 0j), grid)
    >>> rate > 1e6
    True
    """
    if isinstance(space, FreeGrid):
        transverse = space.kperp_sq
    elif isinstance(space, RadialGrid):
        transverse = space.k_r**2
    else:
        msg = (
            "physical_evanescent_rate takes a FreeGrid, LineGrid or RadialGrid, not "
            f"{type(space).__name__}"
        )
        raise GridError(msg)
    active = grid.active_mask
    omega = grid.omega[active]
    square = (omega * medium.index(omega, z) / C) ** 2
    _, _, evanescent = attenuation_extremes(square, transverse)
    largest = float(np.max(evanescent, initial=0.0))
    return largest if largest > 0 else EVANESCENT_RATE


def critical_power(
    wavelength: float,
    chi3: float,
    *,
    index: float,
    space: RadialGrid | FreeGrid | PlanarGrid | None = None,
) -> float:
    r"""Return the physical Townes critical-power baseline.

    This is the scalar, monochromatic, paraxial Kerr lower threshold. It is a
    diagnostic reference for an exact-UPPE pulse, whose dispersion, bandwidth,
    evanescent model and losses can change the focusing dynamics; it is not a
    broadband collapse prediction. Power is the axial Poynting flux
    ``n0*eps0*c*integral(|E_a|^2 dS)/2`` of the paraxial beam in the medium, as
    every spatial energy statistic measures it (SEP 34).

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in m, finite and positive.
    chi3 : float
        Absolute positive Kerr susceptibility in m^2/V^2, including number density.
    index : float
        Real refractive index ``n0`` of the medium at the carrier, finite and
        positive.
    space : RadialGrid, FreeGrid, PlanarGrid or None, optional
        The grid the power is to be compared on, when there is one. A
        :class:`solprop.grid.LineGrid` is refused: a beam uniform in y has one
        free transverse dimension, where Kerr self-focusing does not collapse. A
        :class:`solprop.modes.slab.PlanarGrid` is admitted: its field has both
        transverse dimensions, and the value is the material scale its focusing
        and modulational-instability lengths are quoted against (Notes).

    Returns
    -------
    float
        Critical-power baseline in W. The numerical Townes mass is quoted to
        six significant figures; no accuracy beyond that is implied.

    Raises
    ------
    MaterialError
        If the wavelength, susceptibility or index is nonpositive or nonfinite, or
        they yield a power outside the finite positive floating-point range.
    GridError
        If `space` is a :class:`solprop.grid.LineGrid`.

    Notes
    -----
    The fundamental scalar response gives ``g=3*k*chi3/(8*n**2)`` multiplying
    ``|A|^2*A``. Rescaling the paraxial equation gives
    ``Q''+Q'/r-Q+Q**3=0``. Its radial mass ``integral(Q**2 r dr)=1.86226``
    gives a threshold in ``integral(|E_a|^2 dS)`` in which the index cancels;
    the paraxial power ``n0*eps0*c/2`` times it is
    ``Pcrit=1.86226*n0*eps0*c*wavelength**2/(3*pi*chi3)``. This mass is
    rounded from the independent boundary-value calculation; the approximate
    1.86225 printed in [1]_ is lower by 5.52049e-6.

    A Gaussian's negative-Hamiltonian sufficient threshold has radial mass 2,
    above the Townes baseline. Twice this baseline exceeds that Gaussian
    sufficient threshold, while the actual collapse distance remains a diagnostic.

    On a line uniform in y the (1+1)-D equation has no collapse and no critical
    power: its stationary beam is the stable soliton ``sech(x/x0)``, whose power
    per metre of y is ``8*n0*eps0*c/(3*k0**2*chi3*x0)`` for every width ``x0``
    (``docs/guide/free-space/model.md``). The refusal says so rather than returning a
    number in W that a power in W/m cannot be compared with.

    On a planar grid resolved in y the value is the same material quantity, and it
    is the scale Arnold et al. (2009, section 2.1) quote a planar compressor
    against: with ``Pc = 2 pi n0 / (k0**2 n2)``, "the critical power for 2D
    collapse", they write the whole-beam self-focusing length and the length of
    the modulational instability in y in terms of ``P/Pc`` of the beam's power in
    the guide. The plates bound x, so the whole beam does not collapse as it would
    in free space, but a multimode x basis can still focus in x, and the ratio is
    the one their stability criterion uses. With ``n2 = 3 chi3 / (4 eps0 c n0**2)``
    their ``Pc`` is ``2 n0**2 / 1.86226`` times this value: the Gaussian mass 2
    against the Townes mass, and the ``n0**2`` of that ``n2`` convention, one to
    about 1e-3 in a gas. Convert when comparing with their numbers.

    References
    ----------
    .. [1] G. Fibich and A. L. Gaeta, Opt. Lett. 25, 335-337 (2000),
       doi:10.1364/OL.25.000335, equations (6)-(8).
    .. [2] C. L. Arnold et al., Opt. Express 17, 11122-11129 (2009),
       doi:10.1364/OE.17.011122, section 2.1.
    """
    if isinstance(space, LineGrid):
        msg = (
            "critical_power does not apply to a LineGrid: a beam uniform in y has one "
            "free transverse dimension, where Kerr self-focusing does not collapse; "
            "its stationary beam is the (1+1)-D sech soliton, whose power per metre "
            "of y depends on its width (docs/guide/free-space/model.md)"
        )
        raise GridError(msg)
    if any(
        not math.isfinite(value) or value <= 0 for value in (wavelength, chi3, index)
    ):
        msg = "critical power requires finite positive wavelength, chi3 and index"
        raise MaterialError(msg)
    # Evaluate products in logarithms so valid extreme inputs do not overflow
    # at an intermediate wavelength square before the chi3 division rescales it.
    logarithm = math.log(1.86226 * EPS0 * C / (3 * math.pi))
    logarithm += 2 * math.log(wavelength) - math.log(chi3) + math.log(index)
    if (
        not math.log(np.nextafter(0.0, 1.0))
        <= logarithm
        <= math.log(np.finfo(float).max)
    ):
        msg = "critical power lies outside the finite positive floating-point range"
        raise MaterialError(msg)
    return math.exp(logarithm)
