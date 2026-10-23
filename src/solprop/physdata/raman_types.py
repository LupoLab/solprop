# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parameter records of a supplied Raman response.

The records a caller fills from their own data to build a Raman response with
:mod:`solprop.nonlinear.raman`: a molecule's rotational and vibrational constants
(:class:`MolecularRaman`) with their density-dependent linewidths, an amorphous
solid's intermediate-broadening line set (:class:`IntermediateBroadeningRaman`) and
the Lin--Agrawal two-part silica response (:class:`LinAgrawalRaman`), with the
dephasing and option types the response builders take. They carry no material data:
since SEP 3 every built-in gas's and fused silica's response is optmat's
(:mod:`solprop.nonlinear.optmat_raman`,
:class:`~solprop.physdata.glasses.SilicaBundle`).
"""

import logging
import math
from dataclasses import dataclass, replace

import numpy as np
from jaxtyping import Float

from solprop.constants import AMAGAT
from solprop.exceptions import MaterialError
from solprop.physdata.gases import Gas

__all__ = [
    "Dephasing",
    "IntermediateBroadeningRaman",
    "LinAgrawalRaman",
    "Linewidth",
    "MolecularOptions",
    "MolecularRaman",
    "RotationalRaman",
    "VibrationalRaman",
    "linewidth_from_dephasing_time",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Linewidth:
    """Density-dependent dephasing rate of a Raman transition.

    The full width at half maximum (FWHM) in hertz is
    ``constant + inverse_density / n + density * n`` with ``n`` the number density in
    amagat, so the dephasing time is ``tau2 = 1 / (pi * fwhm)``. The middle term is
    Dicke narrowing, which makes lines *narrower* as the density rises until collisional
    broadening takes over; a gas with only a measured dephasing time at one density is
    represented by the constant term alone.

    Attributes
    ----------
    constant : float
        Density-independent term in Hz.
    inverse_density : float
        Dicke-narrowing coefficient in Hz amagat.
    density : float
        Collisional broadening coefficient in Hz/amagat.
    """

    constant: float = 0.0
    inverse_density: float = 0.0
    density: float = 0.0

    def dephasing_time(self, density_amagat: float) -> float:
        """Return the dephasing time at a given density.

        Parameters
        ----------
        density_amagat : float
            Number density in amagat (multiples of the Loschmidt constant,
            :data:`solprop.constants.AMAGAT`), > 0.

        Returns
        -------
        float
            Dephasing time ``tau2`` in s.

        Raises
        ------
        MaterialError
            If the density is not positive, where the Dicke term diverges.
        """
        if not density_amagat > 0.0:
            raise MaterialError(
                f"density must be positive to evaluate a Raman linewidth, got "
                f"{density_amagat} amagat"
            )
        rate = (
            self.constant
            + self.inverse_density / density_amagat
            + self.density * density_amagat
        )
        return 1.0 / (math.pi * rate)


def linewidth_from_dephasing_time(dephasing_time: float) -> Linewidth:
    """Build a density-independent linewidth from a measured dephasing time.

    A module-level function rather than a classmethod: the runtime type checking
    installed in the test session resolves annotations while the class body is still
    executing, so a method cannot name its own class in its return annotation.

    Parameters
    ----------
    dephasing_time : float
        Dephasing time ``tau2`` in s, > 0.

    Returns
    -------
    Linewidth
        A linewidth whose only term is ``1 / (pi tau2)``.

    Raises
    ------
    MaterialError
        If the dephasing time is not positive.

    Examples
    --------
    >>> from solprop.physdata import linewidth_from_dephasing_time
    >>> f"{linewidth_from_dephasing_time(6e-12).dephasing_time(1.0):.3e}"
    '6.000e-12'
    """
    if not dephasing_time > 0.0:
        raise MaterialError(f"dephasing time must be positive, got {dephasing_time} s")
    return Linewidth(constant=1.0 / (math.pi * dephasing_time))


@dataclass(frozen=True)
class RotationalRaman:
    """Constants of the rotational Raman spectrum of a linear molecule.

    Attributes
    ----------
    b : float
        Rotational constant in m^-1.
    d : float
        Centrifugal distortion constant in m^-1.
    degeneracy_odd : int
        Nuclear spin statistical weight of odd rotational states.
    degeneracy_even : int
        Nuclear spin statistical weight of even rotational states.
    polarisability_anisotropy : float
        Anisotropy of the molecular volume polarisability in m^3; multiply by
        ``4 pi epsilon_0`` to obtain the SI polarisability anisotropy.
    linewidth : Linewidth
        Dephasing rate of the rotational coherences.
    citation : str
        Where each constant comes from.
    """

    b: float
    d: float
    degeneracy_odd: int
    degeneracy_even: int
    polarisability_anisotropy: float
    linewidth: Linewidth
    citation: str


@dataclass(frozen=True)
class VibrationalRaman:
    """Constants of one vibrational Raman mode.

    Attributes
    ----------
    frequency : float
        Angular frequency of the mode in rad/s.
    polarisability_derivative : float
        Derivative of the mean volume polarisability with respect to a displacement
        coordinate, m^2. For a diatomic molecule the coordinate is bond length,
        paired with its reduced mass. A mass-weighted derivative uses a different
        convention and must first be converted consistently.
    reduced_mass : float
        Effective mass of that displacement coordinate in kg; the reduced mass
        for a diatomic bond coordinate.
    linewidth : Linewidth
        Dephasing rate of the vibrational coherence.
    citation : str
        Where each constant comes from.
    """

    frequency: float
    polarisability_derivative: float
    reduced_mass: float
    linewidth: Linewidth
    citation: str


@dataclass(frozen=True)
class MolecularRaman:
    """The Raman data of one gas: a rotational part, a vibrational part, or both.

    Attributes
    ----------
    gas : Gas or str
        The species.
    rotational : RotationalRaman or None
        The rotational spectrum, or ``None`` for a spherical top, which has none.
    vibrational : VibrationalRaman or None
        The vibrational mode, or ``None`` if there is no usable data.
    missing : str, optional
        Explanation of unavailable terms; no physical constants are inferred from it.
    """

    gas: Gas | str
    rotational: RotationalRaman | None
    vibrational: VibrationalRaman | None
    missing: str = ""


@dataclass(frozen=True)
class IntermediateBroadeningRaman:
    """Raman lines of an amorphous solid in the intermediate-broadening model.

    Each line contributes ``A exp(-gamma t) exp(-Gamma^2 t^2 / 4) sin(omega t)`` to the
    response: a damped sinusoid whose Lorentzian width ``gamma`` is the homogeneous
    dephasing and whose Gaussian width ``Gamma`` is the inhomogeneous spread of the
    disordered network. Neither width depends on density -- a glass has no density path
    -- so this model carries its own damping, unlike the molecular ones.

    The response is a *shape*: its amplitudes are relative, and the caller fixes the
    magnitude, normally as a Raman fraction of the medium's third-order susceptibility.

    Attributes
    ----------
    frequencies : tuple of float
        Line centres in rad/s.
    amplitudes : tuple of float
        Relative line strengths, dimensionless.
    gaussian_widths : tuple of float
        Inhomogeneous (Gaussian) widths in rad/s.
    lorentzian_widths : tuple of float
        Homogeneous (Lorentzian) widths in rad/s.
    citation : str
        Where the line set comes from.

    Raises
    ------
    MaterialError
        If the four sequences do not have the same length, or there are no lines.

    References
    ----------
    D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for fiber-optic
    Raman gain spectrum and response function", J. Opt. Soc. Am. B 19, 2886 (2002),
    doi:10.1364/JOSAB.19.002886.

    Examples
    --------
    >>> from solprop.physdata import IntermediateBroadeningRaman
    >>> lines = IntermediateBroadeningRaman(
    ...     frequencies=(8.2e13,),
    ...     amplitudes=(1.0,),
    ...     gaussian_widths=(9.6e12,),
    ...     lorentzian_widths=(3.0e12,),
    ...     citation="one illustrative line",
    ... )
    >>> lines.n_lines
    1
    """

    frequencies: tuple[float, ...]
    amplitudes: tuple[float, ...]
    gaussian_widths: tuple[float, ...]
    lorentzian_widths: tuple[float, ...]
    citation: str

    def __post_init__(self) -> None:
        """Reject a line set whose four sequences disagree.

        Raises
        ------
        MaterialError
            If the lengths differ or the set is empty.
        """
        lengths = {
            len(self.frequencies),
            len(self.amplitudes),
            len(self.gaussian_widths),
            len(self.lorentzian_widths),
        }
        if len(lengths) != 1:
            raise MaterialError(
                f"the four line sequences must have the same length, got "
                f"{sorted(lengths)}"
            )
        if not self.frequencies:
            raise MaterialError("a Raman line set must have at least one line")

    @property
    def n_lines(self) -> int:
        """Return the number of lines.

        Returns
        -------
        int
            The common length of the four sequences.
        """
        return len(self.frequencies)


@dataclass(frozen=True)
class LinAgrawalRaman:
    r"""The two-part Raman response of fused silica, in the Lin--Agrawal form.

    The delayed nuclear response of Lin and Agrawal's vector theory, whose
    isotropic and anisotropic parts are separate functions of delay:

    .. math::

        h_a(t) &= \frac{\tau_1^2 + \tau_2^2}{\tau_1\tau_2^2}
                  e^{-t/\tau_2}\sin(t/\tau_1), \\
        h_b(t) &= \frac{2\tau_b - t}{\tau_b^2} e^{-t/\tau_b},

    each integrating to one over the positive axis: :math:`h_a` is the damped
    vibrational response of the Blow--Wood model and :math:`h_b` the boson peak of
    the disordered network. The paper's two response functions are then

    .. math::

        a(\tau) = w_a h_a(\tau), \qquad
        b(\tau) = w_b h_b(\tau) + w_{ab} h_a(\tau),

    with :math:`\int a = w_a` and :math:`\int b = w_b + w_{ab}`, and the pair
    multiplies :math:`f_R\chi^{(3)}` once. Like
    :class:`IntermediateBroadeningRaman` this record is **data**: the response that
    evaluates it is :class:`solprop.nonlinear.raman.SilicaTensorResponse`.

    The built-in silica responses are optmat's bundles
    (:class:`~solprop.physdata.glasses.SilicaBundle`); this record is for a caller's own
    parameters.

    Attributes
    ----------
    tau1 : float
        Inverse of the vibrational frequency of :math:`h_a`, s, positive.
    tau2 : float
        Damping time of :math:`h_a`, s, positive.
    tau_b : float
        Time constant of the boson peak :math:`h_b`, s, positive.
    isotropic_weight : float
        :math:`w_a`, the weight of :math:`h_a` in the isotropic response
        :math:`a`, non-negative.
    boson_weight : float
        :math:`w_b`, the weight of :math:`h_b` in the anisotropic response
        :math:`b`, non-negative.
    anisotropic_weight : float
        :math:`w_{ab}`, the weight of :math:`h_a` in :math:`b`, non-negative.
    fraction : float
        The Raman fraction :math:`f_R` of the medium's third-order
        susceptibility, in ``(0, 1]``.
    citation : str
        Where the numbers come from.

    Raises
    ------
    MaterialError
        If a time constant is not positive and finite, a weight is negative, the
        three weights do not sum to one, or the fraction is outside ``(0, 1]``.

    References
    ----------
    Q. Lin and G. P. Agrawal, "Vector theory of stimulated Raman scattering and its
    application to fiber-based Raman amplifiers", J. Opt. Soc. Am. B 20, 1616
    (2003), doi:10.1364/JOSAB.20.001616, Eqs. (2.2) and (2.5), for the tensor
    structure; Q. Lin and G. P. Agrawal, "Raman response function for silica
    fibers", Opt. Lett. 31, 3086 (2006), doi:10.1364/OL.31.003086, for these
    parameters.

    Examples
    --------
    The 2006 letter's parameters, as a caller would supply them:

    >>> from solprop.physdata import LinAgrawalRaman
    >>> silica = LinAgrawalRaman(
    ...     tau1=12.2e-15,
    ...     tau2=32.0e-15,
    ...     tau_b=96.0e-15,
    ...     isotropic_weight=0.75,
    ...     boson_weight=0.21,
    ...     anisotropic_weight=0.04,
    ...     fraction=0.245,
    ...     citation="Lin and Agrawal, Opt. Lett. 31, 3086 (2006)",
    ... )
    >>> round(silica.isotropic_weight + silica.boson_weight
    ...       + silica.anisotropic_weight, 12)
    1.0
    """

    tau1: float
    tau2: float
    tau_b: float
    isotropic_weight: float
    boson_weight: float
    anisotropic_weight: float
    fraction: float
    citation: str

    def __post_init__(self) -> None:
        """Reject an unusable time constant, weight or fraction.

        Raises
        ------
        MaterialError
            Naming what is out of range.
        """
        times = (self.tau1, self.tau2, self.tau_b)
        if not all(math.isfinite(one) and one > 0.0 for one in times):
            raise MaterialError(
                f"a Lin-Agrawal silica response needs positive, finite time "
                f"constants, got {times}"
            )
        weights = (
            self.isotropic_weight,
            self.boson_weight,
            self.anisotropic_weight,
        )
        if not all(math.isfinite(one) and one >= 0.0 for one in weights):
            raise MaterialError(
                f"a Lin-Agrawal silica response needs non-negative weights, got "
                f"{weights}"
            )
        if abs(sum(weights) - 1.0) > 1e-12:
            raise MaterialError(
                f"the isotropic, boson and anisotropic weights share the Raman "
                f"fraction between them and must sum to one, got {sum(weights)!r}"
            )
        if not 0.0 < self.fraction <= 1.0:
            raise MaterialError(
                f"a Raman fraction is a share of the nonlinearity and must lie in "
                f"(0, 1], got {self.fraction}"
            )


@dataclass(frozen=True)
class Dephasing:
    """The decay of a Raman coherence, as a function of the local number density.

    A wrapper over :class:`solprop.physdata.raman_types.Linewidth` that turns a
    full width at half maximum in hertz into the decay factor ``exp(-t / tau2)`` a
    response is damped by, and that is defined at **every** density including zero.

    The zero-density limit has three cases, and which one applies is fixed by the
    linewidth's own coefficients:

    * a gas with Dicke narrowing (``inverse_density > 0``) has a diverging linewidth as
      the density falls, so ``tau2 -> 0``: no coherence at all;
    * a gas with only collisional broadening (``constant`` and ``inverse_density`` both
      zero) has a vanishing linewidth, so ``tau2 -> inf``: an undamped coherence;
    * a gas with a density-independent term keeps ``tau2 = 1 / (pi * constant)``.

    All three are finite answers, for the vibrational model as for the rotational one,
    so a fill that starts from vacuum gives no NaN polarisation. The density multiplies
    the polarisation anyway, so the choice is invisible in the answer and visible only
    in whether the answer is a number.

    Attributes
    ----------
    linewidth : Linewidth
        The dephasing rate of the transition, in Hz per its own density terms.

    Examples
    --------
    >>> from solprop.constants import AMAGAT
    >>> from solprop.nonlinear.raman import Dephasing
    >>> from solprop.physdata import Linewidth
    >>> dephasing = Dephasing(Linewidth(density=3.3e9))
    >>> f"{dephasing.dephasing_time(AMAGAT):.4e}"
    '9.6458e-11'
    >>> dephasing.dephasing_time(0.0)
    inf
    """

    linewidth: Linewidth

    def decay_rate(self, density: float) -> float:
        """Return the coherence decay rate at a number density.

        The reciprocal of :meth:`dephasing_time`, and the form the translation layer
        prefers: zero is the undamped limit and is an ordinary float, where the
        dephasing time it corresponds to is infinite.

        Parameters
        ----------
        density : float
            Number density in m^-3, non-negative.

        Returns
        -------
        float
            ``1 / tau2`` in s^-1. Zero for an undamped coherence and ``inf`` where the
            Dicke term diverges at zero density.

        Raises
        ------
        MaterialError
            If the density is negative.

        Examples
        --------
        >>> from solprop.nonlinear.raman import Dephasing
        >>> from solprop.physdata import Linewidth
        >>> Dephasing(Linewidth(inverse_density=6.15e6)).decay_rate(0.0)
        inf
        """
        if density < 0.0:
            raise MaterialError(
                f"a Raman dephasing needs a non-negative number density, got "
                f"{density} m^-3"
            )
        line = self.linewidth
        if density == 0.0:
            # The Dicke term is the only one that diverges; the other two are finite
            # there, so the limit is the constant term alone when it is absent.
            fwhm = math.inf if line.inverse_density > 0.0 else line.constant
        else:
            amagat = density / AMAGAT
            fwhm = line.constant + line.inverse_density / amagat + line.density * amagat
        return math.pi * fwhm

    def dephasing_time(self, density: float) -> float:
        """Return the coherence dephasing time at a number density.

        Parameters
        ----------
        density : float
            Number density in m^-3, non-negative.

        Returns
        -------
        float
            ``tau2`` in s. ``inf`` for an undamped coherence and ``0.0`` where the Dicke
            term diverges at zero density.

        Raises
        ------
        MaterialError
            If the density is negative.

        Examples
        --------
        >>> from solprop.nonlinear.raman import Dephasing
        >>> from solprop.physdata import Linewidth
        >>> Dephasing(Linewidth(constant=8220e6)).dephasing_time(0.0)
        3.872...e-11
        """
        rate = self.decay_rate(density)
        if rate == 0.0:
            return math.inf
        if math.isinf(rate):
            return 0.0
        return 1.0 / rate

    def damping(
        self, t: Float[np.ndarray, " n_to"], density: float
    ) -> Float[np.ndarray, " n_to"]:
        """Return the decay factor ``exp(-t / tau2)`` on a time axis.

        Parameters
        ----------
        t : numpy.ndarray
            Times since the drive, s, shape ``(n_to,)``, non-negative.
        density : float
            Number density in m^-3, non-negative.

        Returns
        -------
        numpy.ndarray
            The factor in ``[0, 1]``, shape ``(n_to,)``. All ones for an undamped
            coherence and all zeros where the coherence has collapsed, rather than an
            ``exp(-0/0)`` NaN.

        Raises
        ------
        MaterialError
            If the density is negative.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.raman import Dephasing
        >>> from solprop.physdata import Linewidth
        >>> dephasing = Dephasing(Linewidth(inverse_density=6.15e6))
        >>> dephasing.damping(np.array([0.0, 1e-12]), 0.0).tolist()
        [0.0, 0.0]
        """
        times = np.asarray(t, dtype=float)
        rate = self.decay_rate(density)
        if rate == 0.0:
            return np.ones_like(times)
        if math.isinf(rate):
            # A collapsed coherence contributes nothing at any delay, including t = 0,
            # where the models themselves already vanish.
            return np.zeros_like(times)
        return np.exp(-rate * times)


@dataclass(frozen=True)
class MolecularOptions:
    """Which parts of a molecular gas's Raman spectrum to build, and at what state.

    Grouped because :func:`solprop.nonlinear.raman.response_for` would otherwise take
    more arguments than the project allows loose (``AGENTS.md``,
    "Functions and modules").

    Each term flag has three states: ``None`` (the default) takes the term when the
    species has a model for it and silently goes without otherwise, ``True`` requires
    it and is refused by species name when no model exists, ``False`` leaves it out.
    So ``MolecularOptions()`` means every term the species has, and only a term the
    caller named can fail. The temperature likewise defaults to the run's or the
    mixture's own, and to :data:`solprop.constants.ROOM_TEMPERATURE` for a bare
    :func:`solprop.nonlinear.raman.response_for` call.

    Attributes
    ----------
    temperature : float or None, optional
        Gas temperature in K, positive; sets the rotational populations. ``None``
        takes the temperature of the run or mixture the options are used in.
    rotation : bool or None, optional
        Whether to include the rotational ladder. ``None`` takes it when available.
    vibration : bool or None, optional
        Whether to include the vibrational mode. ``None`` takes it when available.
    min_j : int, optional
        Lowest rotational quantum number, non-negative. Default 0.
    max_j : int, optional
        Highest rotational quantum number before truncation. Default 50.

    Examples
    --------
    >>> from solprop.nonlinear.raman import MolecularOptions
    >>> MolecularOptions(rotation=False).vibration is None
    True
    >>> MolecularOptions().resolved(300.0).temperature
    300.0
    """

    temperature: float | None = None
    rotation: bool | None = None
    vibration: bool | None = None
    min_j: int = 0
    max_j: int = 50

    def resolved(self, temperature: float) -> "MolecularOptions":  # noqa: UP037 - bound after the class body
        """Return the options with an unset temperature taken from the caller.

        Parameters
        ----------
        temperature : float
            The run's or mixture's temperature in K.

        Returns
        -------
        MolecularOptions
            The same options, with the temperature filled in when it was ``None``.
        """
        if self.temperature is not None:
            return self
        return replace(self, temperature=temperature)
