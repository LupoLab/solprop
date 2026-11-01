# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-element mathematics, written once and compiled for each backend family.

The ``define_bodies(decorate)`` factory of ``docs/dev/backends.md`` section 3.2: every
body is defined inside a function of the compiler decorator and bound to the others
through the closure, never through a module global, so one family can never reach
another's dispatcher. Two families exist:

``PLAIN``
    the identity, which is what the coverage lane executes with ``NUMBA_DISABLE_JIT=1``
    and what makes the wiring testable without a compiler.
``HOST``
    ``njit(inline="always")``, built by ``solprop.backends.numba_cpu.kernels`` from
    this factory and inlined into its fused kernels.

The formulas are read off :mod:`solprop.backends.numpy_ref.passes` and
:class:`solprop.translate.ionisation.RateTable` rather than re-derived, and the **order
of operations is preserved**: cross-backend agreement at 1e-16 rather than 1e-13 rests
on performing the same products in the same order. The floating-point licence is SEP
25's: the host family's bodies are inlined and take their kernel's flags (contraction by
default), and the protected bodies (``define_bodies``'s ``protect``) are compiled
strict and called.

Plasma bodies describe one channel's fraction and loss. Mixtures loop over independent
channel states outside these bodies and weight their currents by partial density;
neither the body signatures nor their operation order depend on composition.

**Why every body carries** ``@no_type_check``. The test session installs jaxtyping's
import hook over the whole package (``tests/conftest.py``), and the hook appends its
checker to the *end* of a function's decorator list -- that is, innermost -- so a
compiler decorator above it would compile the checker's wrapper rather than the body.
An annotated body fails with ``UnsupportedBytecodeError: Use of unsupported opcode
(DICT_MERGE)`` raised from the wrapper. ``@no_type_check`` is what the hook itself
looks for and skips, so the annotations stay readable and statically checked while the
compiler sees the function that was written.

Examples
--------
>>> from solprop.backends.bodies import PLAIN
>>> PLAIN.kerr_real(2.0, 4.0, 0.5)
4.0
>>> PLAIN.propagator_pair(0.0, 0.0)
(1.0, 0.0, 1.0, -0.0)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, no_type_check

import numpy as np

__all__ = [
    "FOLD_COMPLETE",
    "FOLD_EXCHANGE",
    "FOLD_SECOND",
    "GENERAL_PAIRS",
    "GENERAL_PAIR_ENTRY",
    "PLAIN",
    "Bodies",
    "RateArrays",
    "define_bodies",
]


FOLD_COMPLETE = 0
"""A low-storage pass's only evaluation, folded whole (SEP 24 section 4.3)."""

FOLD_EXCHANGE = 1
"""The first of a pass's two evaluations, the material response, at the exchange."""

FOLD_SECOND = 2
"""The second of a pass's two evaluations, the in-stage absorber's term."""


class RateArrays(NamedTuple):
    """The compiled ionisation rate as the flat arrays a kernel can take.

    :class:`solprop.translate.ionisation.RateTable` is a frozen dataclass a compiled
    kernel cannot accept; these are the same numbers, in the order the interpolant uses
    them.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Cubic spline coefficients, shape ``(4, n_intervals)``, highest power first.
    log_field_min : float
        Natural logarithm of the first knot's field amplitude, the field in V/m.
    d_log_field : float
        Knot spacing in ``log E``, positive.
    field_min : float
        Field amplitude below which the rate is exactly zero, V/m.
    field_max : float
        Field amplitude above which the rate is held at ``rate_max``, V/m.
    rate_max : float
        The rate at ``field_max``, s^-1.
    """

    coefficients: Any
    log_field_min: float
    d_log_field: float
    field_min: float
    field_max: float
    rate_max: float

    @classmethod
    def of(cls, table: Any) -> RateArrays:
        """Bundle a compiled rate table into the arrays a kernel takes.

        Parameters
        ----------
        table : solprop.translate.ionisation.RateTable
            The table the kernel problem carries.

        Returns
        -------
        RateArrays
            The same numbers, in the layout a kernel accepts.
        """
        return cls(
            coefficients=np.ascontiguousarray(table.coefficients),
            log_field_min=float(table.log_field_min),
            d_log_field=float(table.d_log_field),
            field_min=float(table.field_min),
            field_max=float(table.field_max),
            rate_max=float(table.rate_max),
        )


@dataclass(frozen=True)
class Bodies:
    """The per-element bodies of one compiler family.

    Attributes
    ----------
    spatial_element : collections.abc.Callable
        Exact longitudinal rate and prefactor from shared spatial coefficients.
    slab_element : collections.abc.Callable
        The same for a hybrid planar column, its channel's wall term subtracted last.
    longitudinal_rate : collections.abc.Callable
        The rate and prefactor of one longitudinal square, which both elements share.
    longitudinal_root : collections.abc.Callable
        The principal root of one longitudinal square, its propagating test and its
        field attenuation, which the rate and the far-field statistics share.
    mask_plane_radius : collections.abc.Callable
        A column's image's distance from an aperture's centre in the mask plane,
        evaluated strictly (SEP 23 equation (4)); protected.
    aperture_weight : collections.abc.Callable
        One aperture's transmission at one column and bin, with its exact zeros.
    region_member : collections.abc.Callable
        Whether a column lies in a half-open box of physical transverse wavevector.
    propagator_pair : collections.abc.Callable
        Both signs of ``exp(dPhi)`` from one real ``exp``, one ``sincos`` and one
        reciprocal.
    rate_at : collections.abc.Callable
        The tabulated ionisation rate at one field amplitude, s^-1.
    magnitude_pair : collections.abc.Callable
        Scaled Euclidean magnitude of two finite real field components.
    kerr_real : collections.abc.Callable
        The real cubic ``c rho (sum_q E_q^2) E_p`` of a field grid.
    kerr_analytic : collections.abc.Callable
        The analytic cubic of an envelope, as a real and imaginary pair.
    dispersive_isotropic : collections.abc.Callable
        The isotropic Kleinman contraction of three two-component fields.
    dispersive_conserving : collections.abc.Callable
        The number-conserving group of three two-component analytic fields.
    dispersive_harmonic : collections.abc.Callable
        The two harmonic groups of three two-component envelopes.
    dispersive_isotropic3 : collections.abc.Callable
        The isotropic Kleinman contraction of three three-component fields.
    dispersive_conserving3 : collections.abc.Callable
        The number-conserving group of three three-component analytic fields.
    dispersive_real_self : collections.abc.Callable
        The full cubic ``x^3`` of one real component.
    dispersive_real_pair : collections.abc.Callable
        The product ``x^2 f`` of one real component.
    dispersive_real_cross : collections.abc.Callable
        The product ``x f^2`` of one real component.
    dispersive_conserving_self : collections.abc.Callable
        The conserving group of ``(u, u, u)``, ``3 |u|^2 u``.
    dispersive_conserving_pair : collections.abc.Callable
        The conserving group of ``(u, u, f)``.
    dispersive_conserving_cross : collections.abc.Callable
        The conserving group of ``(u, f, f)``.
    dispersive_harmonic_self : collections.abc.Callable
        The harmonic groups of ``(u, u, u)``.
    dispersive_harmonic_pair : collections.abc.Callable
        The harmonic groups of ``(u, u, f)``.
    dispersive_harmonic_cross : collections.abc.Callable
        The harmonic groups of ``(u, f, f)``.
    held_plasma_source : collections.abc.Callable
        Combine the two held-current spectra in polarisation units on an active bin.
    loss_scale : collections.abc.Callable
        The guarded coefficient of the ionisation-loss current.
    trapezoid : collections.abc.Callable
        One increment of a cumulative trapezoid.
    fraction_at : collections.abc.Callable
        The ionised fraction at one sample of the exposure.
    accumulate : collections.abc.Callable
        One weighted stage accumulation.
    low_storage_fold : collections.abc.Callable
        One element of a low-storage RK4IP pass's finish: a source folded into the
        three registers.
    raman_drive : collections.abc.Callable
        Scalar field-square or cycle-averaged envelope drive.
    raman_tensor_drive : collections.abc.Callable
        Trace, difference and twice-coherence drives of a Cartesian field pair.
    raman_damped : collections.abc.Callable
        One undamped Raman sample times its exponential decay.
    raman_finish : collections.abc.Callable
        Delayed polarisation addition followed by the temporal collar.
    raman_tensor_finish : collections.abc.Callable
        One component of the delayed tensor polarisation and temporal collar.
    absorber_sample : collections.abc.Callable
        Temporal absorption as an in-stage rate or an accepted-step exponential.
    envelope_population : collections.abc.Callable
        Return stable ionised fraction and surviving neutral fraction from exposure.
    envelope_axes : collections.abc.Callable
        Scaled real Gram decomposition of a complex Cartesian envelope.
    envelope_moments : collections.abc.Callable
        Refined positive directional moments and a convergence status.
    envelope_loss : collections.abc.Callable
        Two complex current components from the directional moments.
    general_kerr_component : collections.abc.Callable
        One component of the instantaneous tensor contraction from a pair history.
    general_kerr : collections.abc.Callable
        A general modal guide's instantaneous tensor cubic at one node and sample,
        densely, on the analytic signal.
    general_symmetric_row : collections.abc.Callable
        One component's coefficients of the real cubic's symmetric contraction.
    general_symmetric_sum : collections.abc.Callable
        One component of the symmetric contraction from its coefficients and the
        monomials.
    general_real_kerr : collections.abc.Callable
        The real cubic by its symmetric contraction at one node and sample.
    general_pair_drive : collections.abc.Callable
        One entry of the real symmetric pair history a delayed tensor response is
        driven by.
    general_delayed : collections.abc.Callable
        One component of the delayed tensor contraction at one node and sample.
    """

    spatial_element: Any
    slab_element: Any
    longitudinal_rate: Any
    longitudinal_root: Any
    mask_plane_radius: Any
    aperture_weight: Any
    region_member: Any
    propagator_pair: Any
    rate_at: Any
    magnitude_pair: Any
    kerr_real: Any
    kerr_analytic: Any
    dispersive_isotropic: Any
    dispersive_conserving: Any
    dispersive_harmonic: Any
    dispersive_isotropic3: Any
    dispersive_conserving3: Any
    dispersive_real_self: Any
    dispersive_real_pair: Any
    dispersive_real_cross: Any
    dispersive_conserving_self: Any
    dispersive_conserving_pair: Any
    dispersive_conserving_cross: Any
    dispersive_harmonic_self: Any
    dispersive_harmonic_pair: Any
    dispersive_harmonic_cross: Any
    held_plasma_source: Any
    loss_scale: Any
    trapezoid: Any
    fraction_at: Any
    accumulate: Any
    low_storage_fold: Any
    raman_drive: Any
    raman_tensor_drive: Any
    raman_damped: Any
    raman_finish: Any
    raman_tensor_finish: Any
    absorber_sample: Any
    envelope_population: Any
    envelope_axes: Any
    envelope_moments: Any
    envelope_loss: Any
    general_kerr_component: Any
    general_kerr: Any
    general_symmetric_row: Any
    general_symmetric_sum: Any
    general_real_kerr: Any
    general_pair_drive: Any
    general_delayed: Any


GENERAL_PAIR_ENTRY = ((0, 3, 4), (3, 1, 5), (4, 5, 2))
"""Which stored pair history holds ``Q_kl``, for ``k`` and ``l`` in ``0, 1, 2``.

The six independent entries of a real symmetric three-by-three history are stored in
the order ``xx, yy, zz, xy, xz, yz``, the order of
:data:`solprop.backends.numpy_ref.general_modal.PAIR_HISTORIES`' documentation.
"""

GENERAL_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
"""The component pair of each stored history entry, the inverse of the table above."""


def define_bodies(  # noqa: PLR0915 - bodies share one compiler-bound closure
    decorate: Callable[[Any], Any],
    protect: Callable[[Any], Any] | None = None,
) -> Bodies:
    """Return the per-element bodies, compiled by ``decorate``.

    Parameters
    ----------
    decorate : collections.abc.Callable
        The compiler decorator: ``njit(inline="always")`` for the host family and the
        identity for the plain family.
    protect : collections.abc.Callable, optional
        The decorator of the **protected** bodies, which form a small difference or
        rely on an exact cancellation (SEP 25 section 4.2): the population maps
        ``fraction_at`` and ``envelope_population``, and the ellipse resolution and
        moments of ``envelope_axes`` and ``envelope_moments``, whose determinant
        ``xr yi - xi yr`` must vanish for a linear field; and the far-field
        statistics' ``mask_plane_radius``, whose hard-edge decisions must be the
        reference's exactly (SEP 23 section 6). The host family passes
        ``njit(strict=True)``: compiled on their own without fastmath flags and
        called, never inlined, since an inlined body takes its caller's flags.
        ``None`` uses `decorate`.

    Returns
    -------
    Bodies
        The family, whose members are bound to each other through this closure.

    Examples
    --------
    >>> from solprop.backends.bodies import define_bodies
    >>> plain = define_bodies(lambda function: function)
    >>> plain.trapezoid(2.0, 4.0, 0.5)
    1.5
    """
    protect = decorate if protect is None else protect

    @decorate
    @no_type_check
    def longitudinal_root(
        longitudinal: complex, evanescent_rate: float
    ) -> tuple[complex, bool, float]:
        """Return the principal root of a longitudinal square, its test and its rate.

        The element form of :func:`solprop.translate.spatial.longitudinal_root`: the
        root ``k_z`` the stages use, the test ``Re q > 0`` that separates propagating
        from evanescent columns, and the field attenuation, ``Im k_z`` where
        propagating and ``min(|Im k_z|, evanescent_rate)`` where not. The stage
        elements and the far-field statistics read it, so both see the same root
        (SEP 23 section 6).

        Parameters
        ----------
        longitudinal : complex
            ``k_z**2`` in m^-2.
        evanescent_rate : float
            Maximum evanescent field attenuation in m^-1.

        Returns
        -------
        tuple
            ``k_z`` in m^-1 (complex), whether the column propagates (bool) and the
            field attenuation in m^-1 (float).
        """
        root = np.sqrt(longitudinal)
        propagating = longitudinal.real > 0
        attenuation = root.imag if propagating else min(abs(root.imag), evanescent_rate)
        return root, propagating, attenuation

    @protect
    @no_type_check
    def mask_plane_radius(
        scale: float, k_x: float, k_y: float, centre_x: float, centre_y: float
    ) -> float:
        """Return how far a column's mask-plane image lies from an aperture's centre.

        Equation (4) of SEP 23, the element form of
        :func:`solprop.spatial.mask_plane_radius`:
        ``sqrt((s k_x + X)**2 + (s k_y + Y)**2)`` with the **stored** column axes and
        the physical centre. A protected body: evaluated as written, products, sums,
        two squares, a sum and a correctly rounded square root, never contracted and
        never ``hypot``, so a hard edge decides as the reference decides, on every
        input (an exact hard-mask contract, SEP 25).

        Parameters
        ----------
        scale : float
            The aperture's mask-plane scale ``z c / omega`` at this bin, m^2.
        k_x, k_y : float
            The column's stored transverse wavevector, rad/m.
        centre_x, centre_y : float
            The aperture's centre, physical axes, m; zero for the distance from the
            origin.

        Returns
        -------
        float
            The distance in m.
        """
        across = scale * k_x + centre_x
        along = scale * k_y + centre_y
        return math.sqrt(across * across + along * along)

    @decorate
    @no_type_check
    def aperture_weight(  # noqa: PLR0911, PLR0913, PLR0917 - one return per profile
        radius: float,
        origin: float,
        scale: float,
        half_width: float,
        inverse_width: float,
        edge: float,
        distance: float,
        code: int,
    ) -> float:
        """Return one aperture's amplitude transmission at one column and bin.

        The element form of :func:`solprop.spatial.aperture_profile` and the zeros of
        SEP 23 sections 4.2 and 4.10: exactly zero where the scale is zero (at and
        below zero frequency) or the column's image lies at or beyond the mask
        plane's distance from the origin; otherwise ``hard`` (code 0), one for
        ``radius <= half_width``; ``tanh`` (1), ``(1 - tanh t)/2``; ``rcos`` (2),
        ``(1 - sin(pi t/2))/2``, exactly one for ``t <= -1`` and exactly zero for
        ``t >= 1``, so the sine is skipped outside the edge; and ``supergauss`` (3),
        ``exp(-(radius/half_width)**edge)``; with ``t = (radius - half_width) *
        inverse_width``, the argument the reference forms.

        Parameters
        ----------
        radius : float
            The image's distance from the aperture's centre, m
            (:func:`mask_plane_radius`).
        origin : float
            The image's distance from the origin of the mask plane, m.
        scale : float
            The aperture's mask-plane scale at this bin, m^2.
        half_width : float
            ``D/2`` or the corrected ``D_q/2``, m.
        inverse_width : float
            ``1/w`` in 1/m for ``tanh`` and ``rcos``, unused otherwise.
        edge : float
            The order of ``supergauss``, unused otherwise.
        distance : float
            Exit face to mask plane, m.
        code : int
            The profile's index in :data:`solprop.spatial.APERTURE_PROFILES`.

        Returns
        -------
        float
            The dimensionless transmission in ``[0, 1]``.
        """
        if scale <= 0.0 or origin >= distance:
            return 0.0
        if code == 0:
            return 1.0 if radius <= half_width else 0.0
        if code == 3:
            return math.exp(-((radius / half_width) ** edge))
        argument = (radius - half_width) * inverse_width
        if code == 1:
            return 0.5 * (1.0 - math.tanh(argument))
        if argument <= -1.0:
            return 1.0
        if argument >= 1.0:
            return 0.0
        return 0.5 * (1.0 - math.sin((0.5 * math.pi) * argument))

    @decorate
    @no_type_check
    def region_member(
        k_x: float,
        k_y: float,
        kappa_x_low: float,
        kappa_x_high: float,
        kappa_y_low: float,
        kappa_y_high: float,
    ) -> bool:
        """Return whether a column belongs to a half-open box of transverse wavevector.

        SEP 23 section 4.7: the physical wavevector is the exact negation of the
        stored axes, compared with half-open edges ``[low, high)`` on both axes, so
        boxes that tile the plane tile the columns.

        Parameters
        ----------
        k_x, k_y : float
            The column's stored transverse wavevector, rad/m.
        kappa_x_low, kappa_x_high, kappa_y_low, kappa_y_high : float
            The box's physical edges in rad/m, infinite edges allowed.

        Returns
        -------
        bool
            Whether the column lies in the box.
        """
        kappa_x = -k_x
        kappa_y = -k_y
        return (
            kappa_x_low <= kappa_x < kappa_x_high
            and kappa_y_low <= kappa_y < kappa_y_high
        )

    @decorate
    @no_type_check
    def longitudinal_rate(  # noqa: PLR0913, PLR0917 - the element's operands
        longitudinal: complex,
        frame: float,
        spectral_rate: float,
        numerator: complex,
        clamped: complex,
        active: bool,
        evanescent_rate: float,
        transverse_rate: float,
    ) -> tuple[complex, complex]:
        """Return the rate and prefactor of one longitudinal wave-number square.

        What the free-space and the hybrid planar elements share: the principal root,
        the evanescent model and its clamp, the frame and the loss. The transverse
        collar's rate joins the attenuation inside the active band (SEP 22, Q6); a
        zero rate leaves the element as it was, signed zeros included.

        Parameters
        ----------
        longitudinal : complex
            ``k_z**2`` in m^-2.
        frame : float
            Frame subtraction in m^-1.
        spectral_rate : float
            Spectral power loss rate in m^-1.
        numerator, clamped : complex
            Nonlinear numerator and evanescent prefactor, including crop scaling.
        active : bool
            Whether this frequency belongs to the physical band.
        evanescent_rate : float
            Maximum evanescent field attenuation in m^-1.
        transverse_rate : float
            The column's transverse-wavevector power rate in m^-1, nonnegative.

        Returns
        -------
        tuple of complex
            Linear field rate in m^-1 and the cropped nonlinear prefactor.
        """
        root, propagating, attenuation = longitudinal_root(
            longitudinal, evanescent_rate
        )
        rate = (
            complex(-attenuation - 0.5 * transverse_rate, -(root.real - frame))
            if active
            else 0j
        )
        prefactor = numerator / root.real if propagating else clamped
        return rate - 0.5 * spectral_rate, prefactor if active else 0j

    @decorate
    @no_type_check
    def spatial_element(  # noqa: PLR0913, PLR0917 - one element of the frozen spatial record
        square: complex,
        transverse: float,
        square_low: complex,
        transverse_low: float,
        frame: float,
        spectral_rate: float,
        numerator: complex,
        clamped: complex,
        active: bool,
        evanescent_rate: float,
        transverse_rate: float,
    ) -> tuple[complex, complex]:
        """Evaluate one exact longitudinal rate and nonlinear prefactor.

        Parameters
        ----------
        square : complex
            Shared material wave-number square in m^-2.
        transverse : float
            Transverse wave-number square in m^-2.
        square_low : complex
            Material square residual in m^-2.
        transverse_low : float
            Transverse square residual in m^-2.
        frame : float
            Frame subtraction in m^-1.
        spectral_rate : float
            Spectral power loss rate in m^-1.
        numerator, clamped : complex
            Nonlinear numerator and evanescent prefactor, including crop scaling.
        active : bool
            Whether this frequency belongs to the physical band.
        evanescent_rate : float
            Maximum evanescent field attenuation in m^-1.
        transverse_rate : float
            The column's transverse-wavevector power rate in m^-1 (kernel schema 31).

        Returns
        -------
        tuple of complex
            Linear field rate in m^-1 and the cropped nonlinear prefactor.
        """
        longitudinal = (square - transverse) + (square_low - transverse_low)
        return longitudinal_rate(
            longitudinal,
            frame,
            spectral_rate,
            numerator,
            clamped,
            active,
            evanescent_rate,
            transverse_rate,
        )

    @decorate
    @no_type_check
    def slab_element(  # noqa: PLR0913, PLR0917 - one element of the frozen spatial record
        square: complex,
        transverse: float,
        square_low: complex,
        transverse_low: float,
        wall: complex,
        frame: float,
        spectral_rate: float,
        numerator: complex,
        clamped: complex,
        active: bool,
        evanescent_rate: float,
        transverse_rate: float,
    ) -> tuple[complex, complex]:
        """Evaluate one hybrid planar column's exact longitudinal rate and prefactor.

        Kernel schema 15's ``"slab"`` kind: the free-space element with the column's
        slab channel's complex wall term ``kappa_p(omega)**2`` subtracted **last**,
        ``q = ((k^2 - k_y^2) + (k_low^2 - k_y,low^2)) - kappa_p^2``, the order in which
        :func:`solprop.translate.spatial.separable_values` forms it.

        Parameters
        ----------
        square : complex
            Shared material wave-number square in m^-2.
        transverse : float
            The column's ``k_y**2`` in m^-2.
        square_low : complex
            Material square residual in m^-2.
        transverse_low : float
            Transverse square residual in m^-2.
        wall : complex
            The column's channel's ``kappa_sq`` at this frequency, in m^-2.
        frame : float
            Frame subtraction in m^-1.
        spectral_rate : float
            Spectral power loss rate in m^-1.
        numerator, clamped : complex
            Nonlinear numerator and evanescent prefactor, including crop scaling.
        active : bool
            Whether this frequency belongs to the physical band.
        evanescent_rate : float
            Maximum evanescent field attenuation in m^-1.
        transverse_rate : float
            The column's safety-ramp power rate on ``k_y``, m^-1 (SEP 40).

        Returns
        -------
        tuple of complex
            Linear field rate in m^-1 and the cropped nonlinear prefactor.
        """
        longitudinal = ((square - transverse) + (square_low - transverse_low)) - wall
        return longitudinal_rate(
            longitudinal,
            frame,
            spectral_rate,
            numerator,
            clamped,
            active,
            evanescent_rate,
            transverse_rate,
        )

    @decorate
    @no_type_check
    def propagator_pair(real: float, imag: float) -> tuple[float, float, float, float]:
        """Return both signs of ``exp(z)`` from one ``exp`` and one ``sincos``.

        Both signs are needed at every stage -- forward into the lab frame and back into
        the interaction picture -- and they share their trigonometry:
        ``exp(-a - ib) = (1/exp(a))(cos b - i sin b)``. A complex exponential per sign,
        which is what the reference evaluates, costs the transcendental twice for
        numbers that share all of it.

        Parameters
        ----------
        real : float
            Real part of the exponent, the accumulated loss.
        imag : float
            Imaginary part of the exponent, the accumulated phase, radians.

        Returns
        -------
        tuple of float
            ``(fwd_re, fwd_im, inv_re, inv_im)``: ``exp(z)`` then ``exp(-z)``.
        """
        scale = math.exp(real)
        cosine = math.cos(imag)
        sine = math.sin(imag)
        reciprocal = 1.0 / scale
        return scale * cosine, scale * sine, reciprocal * cosine, -reciprocal * sine

    @decorate
    @no_type_check
    def magnitude_pair(first: float, second: float) -> float:
        """Return a finite pair's Euclidean magnitude without squaring its scale.

        Parameters
        ----------
        first, second : float
            Finite Cartesian field components in V/m.

        Returns
        -------
        float
            Non-negative magnitude in V/m; overflow only if the result cannot fit.

        Notes
        -----
        A device compiler has been observed to lower hypot in a form that underflows
        when squaring normal small components. Dividing by the larger component keeps
        the square in [0, 1], preserving the existing magnitude error envelope.
        """
        larger, smaller = max(abs(first), abs(second)), min(abs(first), abs(second))
        if larger == 0.0:
            return 0.0
        ratio = smaller / larger
        return larger * math.sqrt(1.0 + ratio * ratio)

    @decorate
    @no_type_check
    def rate_at(magnitude: float, table: RateArrays) -> float:
        """Return the tabulated ionisation rate at one field amplitude.

        The knots are uniform in ``log E``, so locating one is arithmetic rather than a
        search. The three branches are :meth:`RateTable.__call__`'s: exactly zero below
        ``field_min``, ``rate_max`` above ``field_max``, and the cubic between.

        Only samples inside the table pay the logarithm, the cubic and the exponential,
        exactly as the reference's compressed evaluation does. That branch is why the
        cost of the response column -- and so the comparison against an array compiler,
        which must evaluate both sides of a select -- depends on the state it is
        measured from.

        Parameters
        ----------
        magnitude : float
            The field amplitude ``|E|``, V/m, non-negative.
        table : RateArrays
            The compiled rate.

        Returns
        -------
        float
            The rate, s^-1.
        """
        if magnitude > table.field_max:
            return table.rate_max
        if magnitude < table.field_min:
            return 0.0
        log_field = math.log(magnitude)
        index = int((log_field - table.log_field_min) / table.d_log_field)
        last = table.coefficients.shape[1] - 1
        if index < 0:
            index = 0
        elif index > last:
            index = last
        local = log_field - (table.log_field_min + index * table.d_log_field)
        c = table.coefficients
        cubic = ((c[0, index] * local + c[1, index]) * local + c[2, index]) * local + c[
            3, index
        ]
        return math.exp(cubic)

    @protect
    @no_type_check
    def envelope_population(exposure: float, initial: float) -> Any:
        """Return stable population and survival without subtracting saturation.

        Parameters
        ----------
        exposure : float
            Non-negative integrated cycle rate, dimensionless.
        initial : float
            Initial ionised fraction in [0, 1].

        Returns
        -------
        tuple of float
            Ionised fraction and surviving neutral fraction.
        """
        return (
            initial + (1 - initial) * -math.expm1(-exposure),
            (1 - initial) * math.exp(-exposure),
        )

    @protect
    @no_type_check
    def envelope_axes(first: complex, second: complex) -> Any:
        """Resolve an ellipse without squaring the physical field scale.

        Parameters
        ----------
        first, second : complex
            Physical Cartesian field components in V/m; second is zero for scalar.

        Returns
        -------
        tuple
            Major amplitude, minor/major ratio, cosine and sine of the real Gram
            eigenvector, two scaled major-vector components, two scaled minor
            components, physical field scale and dimensionless major square.
        """
        # Native MLIR lowers only two-argument max; preserve left-to-right ties.
        scale = max(abs(first.real), abs(first.imag))
        scale = max(scale, abs(second.real))
        scale = max(scale, abs(second.imag))
        if scale == 0.0:
            return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        xr, xi, yr, yi = (
            first.real / scale,
            first.imag / scale,
            second.real / scale,
            second.imag / scale,
        )
        xx, yy, xy = xr * xr + yr * yr, xi * xi + yi * yi, xr * xi + yr * yi
        major_square = 0.5 * (xx + yy + magnitude_pair(xx - yy, 2 * xy))
        determinant = xr * yi - xi * yr
        ratio = min(1.0, abs(determinant) / major_square)
        angle = 0.5 * math.atan2(2 * xy, xx - yy)
        cosine, sine = math.cos(angle), math.sin(angle)
        ux, uy = xr * cosine + xi * sine, yr * cosine + yi * sine
        # Recover the small axis from the determinant, not a subtraction of
        # nearly equal eigenvalues or the cancellation B @ null_eigenvector.
        signed_ratio = determinant / major_square
        vx, vy = -signed_ratio * uy, signed_ratio * ux
        return (
            scale * math.sqrt(major_square),
            ratio,
            cosine,
            sine,
            ux,
            uy,
            vx,
            vy,
            scale,
            major_square,
        )

    @protect
    @no_type_check
    def envelope_moments(
        major: float, ratio: float, table: RateArrays, nodes: Any, weights: Any
    ) -> Any:
        """Refine split quadrant integrals, returning failure at the accepted cap.

        Parameters
        ----------
        major : float
            Major-axis field in V/m, non-negative.
        ratio : float
            Minor/major amplitude ratio in [0, 1].
        table : RateArrays
            Validated instantaneous magnitude table.
        nodes, weights : numpy.ndarray
            Packed read-only rules, shape (4, 512), with orders 64/128/256/512.

        Returns
        -------
        tuple
            Cosine moment, sine moment in s^-1 and status: zero on convergence,
            one when either moment exceeds its 2e-6 budget at order 512.
        """
        if major < table.field_min:
            return 0.0, 0.0, 0
        if ratio == 1.0:
            rate = 0.5 * rate_at(major, table)
            return rate, rate, 0
        if not math.isfinite(major) or not math.isfinite(ratio):
            return 0.0, 0.0, 1
        # High-field clipping is encountered first as theta increases; the
        # lower cutoff follows. Empty panels cost no rate evaluations.
        high = table.field_max / major
        low = table.field_min / major
        high_edge = (
            0.0
            if high >= 1.0
            else math.pi / 2
            if high <= ratio
            else math.acos(
                math.sqrt((high * high - ratio * ratio) / (1 - ratio * ratio))
            )
        )
        low_edge = (
            0.0
            if low >= 1.0
            else math.pi / 2
            if low <= ratio
            else math.acos(math.sqrt((low * low - ratio * ratio) / (1 - ratio * ratio)))
        )
        previous_c, previous_s = 0.0, 0.0
        for level in range(4):
            order = 64 << level
            mc, ms = 0.0, 0.0
            for panel in range(3):
                lower = 0.0 if panel == 0 else high_edge if panel == 1 else low_edge
                upper = (
                    high_edge if panel == 0 else low_edge if panel == 1 else math.pi / 2
                )
                width = upper - lower
                if width == 0.0:
                    continue
                for index in range(order):
                    theta = lower + (nodes[level, index] + 1) * (0.5 * width)
                    cosine, sine = math.cos(theta), math.sin(theta)
                    c2, s2 = cosine * cosine, sine * sine
                    denominator = c2 + ratio * ratio * s2
                    rate = rate_at(major * math.sqrt(denominator), table) / denominator
                    weight = (width / math.pi) * weights[level, index]
                    mc += weight * rate * c2
                    ms += weight * rate * s2
            if (
                level > 0
                and math.isfinite(mc)
                and math.isfinite(ms)
                and abs(mc - previous_c) <= 2e-6 * max(mc, 1e-30)
                and abs(ms - previous_s) <= 2e-6 * max(ms, 1e-30)
            ):
                return mc, ms, 0
            previous_c, previous_s = mc, ms
        return previous_c, previous_s, 1

    @decorate
    @no_type_check
    def envelope_loss(axes: Any, mc: float, ms: float, work: float) -> Any:
        """Reconstruct both current components through their principal vectors.

        Parameters
        ----------
        axes : tuple
            Scaled decomposition from envelope_axes.
        mc, ms : float
            Positive directional moments in s^-1.
        work : float
            Ionisation potential times density and surviving fraction, J/m^3.

        Returns
        -------
        tuple of complex
            Two Cartesian current components in A/m^2.
        """
        if work == 0.0 or (mc == 0.0 and ms == 0.0):
            return 0j, 0j
        major_phase = complex(axes[2], axes[3])
        minor_phase = complex(-axes[3], axes[2])
        # Divide the moments by the physical scale before multiplying. A
        # standalone 1/scale can overflow for a finite, nearly zero field.
        major_moment, minor_moment = mc / axes[8], ms / axes[8]
        factor = 2 * work / axes[9]
        first = factor * (
            major_moment * axes[4] * major_phase + minor_moment * axes[6] * minor_phase
        )
        second = factor * (
            major_moment * axes[5] * major_phase + minor_moment * axes[7] * minor_phase
        )
        return first, second

    @decorate
    @no_type_check
    def kerr_real(field: float, summed_square: float, factor: float) -> float:
        """Return ``(factor * sum_q E_q^2) * E_p``, the full cubic of a real field.

        It already carries the third harmonic and the counter-rotating terms, which is
        why a field grid that radiates the harmonic needs no separate harmonic term.
        The summed square arrives rather than being formed here because the sum runs
        over the **polarisation** axis of one node, which the caller holds.

        The association is the reference's -- ``factor * squares`` first, then the field
        -- and not ``factor * E * E * E``. The two differ in the last bit, and matching
        :func:`solprop.backends.numpy_ref.passes.kerr_polarisation` to the last bit is
        what makes a disagreement between the backends mean something.

        Parameters
        ----------
        field : float
            The component ``E_p`` at one sample, V/m.
        summed_square : float
            ``sum_q E_q^2`` over the polarisation axis at this sample, V^2/m^2.
        factor : float
            ``coefficient * density``.

        Returns
        -------
        float
            The polarisation at that sample, C/m^2.
        """
        return factor * summed_square * field

    @decorate
    @no_type_check
    def kerr_analytic(  # noqa: PLR0913, PLR0917 - the analytic cubic's own operands
        real: float,
        imag: float,
        modulus: float,
        square_real: float,
        square_imag: float,
        factor: float,
        self_factor: float,
        conjugate_factor: float,
    ) -> tuple[float, float]:
        """Return the analytic cubic at one sample, real and imaginary.

        ``c rho [s (sum_q |a_q|^2) a_p + k (sum_q a_q^2) conj(a_p)]``, which is
        :class:`solprop.translate.responses.KerrCoefficients`' analytic form. The sums
        run over the **polarisation** axis of one node, so the caller passes them in
        rather than forming them here: two components of the field at one point of the
        cross-section, never two different points. The harmonic term, when an envelope
        grid radiates it, is applied by the caller with the carrier it holds.

        Parameters
        ----------
        real : float
            Real part of ``a_p`` at this sample, V/m.
        imag : float
            Imaginary part of ``a_p``.
        modulus : float
            ``sum_q |a_q|^2`` over the polarisation axis at this sample.
        square_real : float
            Real part of ``sum_q a_q^2``.
        square_imag : float
            Imaginary part of ``sum_q a_q^2``.
        factor : float
            ``coefficient * density``.
        self_factor : float
            Weight of the self term, ``KerrCoefficients.self_factor``.
        conjugate_factor : float
            Weight of the conjugate term, ``KerrCoefficients.conjugate_factor``.

        Returns
        -------
        tuple of float
            The polarisation at that sample, C/m^2, real then imaginary.
        """
        # Scaled first and multiplied out afterwards, which is the order the reference's
        # `conjugate_factor * squares * conj(values)` performs and therefore the only
        # order that reproduces it to the last bit. The complex product is written out
        # rather than formed with a complex multiply, which a device family lacks.
        weighted_real = conjugate_factor * square_real
        weighted_imag = conjugate_factor * square_imag
        weighted_self = self_factor * modulus
        return (
            factor
            * (weighted_self * real + (weighted_real * real + weighted_imag * imag)),
            factor
            * (weighted_self * imag + (weighted_imag * real - weighted_real * imag)),
        )

    # SEP 6's dispersive cubic (`nonlinear.responses.dispersive_cubic`) at one sample.
    # A product of three fields x, y, z is the isotropic Kleinman contraction
    # T_ijkl x_j y_k z_l; on analytic signals or envelopes a sector keeps some of its
    # conjugate placements: the number-conserving group x y z* + x y* z + x* y z and,
    # for the full cubic on an envelope grid, the harmonic groups x y z (carried by
    # exp(2 i omega0 t)) and x y* z* + x* y z* + x* y* z (by its conjugate). The
    # scalar bodies are the closed forms of the three products the evaluators form --
    # (u, u, u), (u, u, f) and (u, f, f) -- because the general form costs several
    # times the multiplications at every sample of every node; the vector bodies are
    # the general contraction, whose one-component case the closed forms are.

    @decorate
    @no_type_check
    def dispersive_isotropic(
        x0: complex, x1: complex, y0: complex, y1: complex, z0: complex, z1: complex
    ) -> tuple[complex, complex]:
        """Return the isotropic contraction of three two-component fields.

        ``[(x . y) z + (x . z) y + (y . z) x] / 3`` without conjugation, grouped as
        the reference's ``_isotropic`` groups it.

        Parameters
        ----------
        x0, x1, y0, y1, z0, z1 : complex
            The two components of each field at one sample, real or complex, V/m.

        Returns
        -------
        tuple
            The two components of the contraction, V^3/m^3.
        """
        xy = x0 * y0 + x1 * y1
        xz = x0 * z0 + x1 * z1
        yz = y0 * z0 + y1 * z1
        return (
            (xy * z0 + xz * y0 + yz * x0) / 3.0,
            (xy * z1 + xz * y1 + yz * x1) / 3.0,
        )

    @decorate
    @no_type_check
    def dispersive_conserving(
        x0: complex, x1: complex, y0: complex, y1: complex, z0: complex, z1: complex
    ) -> tuple[complex, complex]:
        """Return the number-conserving group of three two-component analytic fields.

        ``T(x, y, z*) + T(x, y*, z) + T(x*, y, z)``, ``T`` the isotropic contraction.

        Parameters
        ----------
        x0, x1, y0, y1, z0, z1 : complex
            The two components of each analytic field or envelope at one sample, V/m.

        Returns
        -------
        tuple
            The group's two components, V^3/m^3.
        """
        a0, a1 = dispersive_isotropic(x0, x1, y0, y1, z0.conjugate(), z1.conjugate())
        b0, b1 = dispersive_isotropic(x0, x1, y0.conjugate(), y1.conjugate(), z0, z1)
        c0, c1 = dispersive_isotropic(x0.conjugate(), x1.conjugate(), y0, y1, z0, z1)
        return a0 + b0 + c0, a1 + b1 + c1

    @decorate
    @no_type_check
    def dispersive_harmonic(  # noqa: PLR0913, PLR0917 - three fields and the carrier
        x0: complex,
        x1: complex,
        y0: complex,
        y1: complex,
        z0: complex,
        z1: complex,
        carrier: complex,
    ) -> tuple[complex, complex]:
        """Return the two harmonic groups of three two-component envelopes.

        ``c T(x, y, z) + c* [T(x, y*, z*) + T(x*, y, z*) + T(x*, y*, z)]`` with
        ``c = exp(2 i omega0 t)``: the full cubic's quartets with three and with one
        positive signed leg, which the number-conserving group lacks.

        Parameters
        ----------
        x0, x1, y0, y1, z0, z1 : complex
            The two components of each envelope at one sample, V/m.
        carrier : complex
            ``exp(2 i omega0 t)`` at the sample.

        Returns
        -------
        tuple
            The groups' two components, V^3/m^3.
        """
        u0, u1 = dispersive_isotropic(x0, x1, y0, y1, z0, z1)
        x0c, x1c = x0.conjugate(), x1.conjugate()
        y0c, y1c = y0.conjugate(), y1.conjugate()
        z0c, z1c = z0.conjugate(), z1.conjugate()
        a0, a1 = dispersive_isotropic(x0, x1, y0c, y1c, z0c, z1c)
        b0, b1 = dispersive_isotropic(x0c, x1c, y0, y1, z0c, z1c)
        c0, c1 = dispersive_isotropic(x0c, x1c, y0c, y1c, z0, z1)
        down = carrier.conjugate()
        return (
            carrier * u0 + down * (a0 + b0 + c0),
            carrier * u1 + down * (a1 + b1 + c1),
        )

    @decorate
    @no_type_check
    def dispersive_isotropic3(  # noqa: PLR0913, PLR0917 - three fields of three components
        x0: complex,
        x1: complex,
        x2: complex,
        y0: complex,
        y1: complex,
        y2: complex,
        z0: complex,
        z1: complex,
        z2: complex,
    ) -> tuple[complex, complex, complex]:
        """Return the isotropic contraction of three three-component fields.

        The general modal route's full vector field (SEP 6, decision Q3):
        ``[(x . y) z + (x . z) y + (y . z) x] / 3`` without conjugation, each dot
        product summed over the components in order, as the reference's
        ``_isotropic`` sums them.

        Parameters
        ----------
        x0, x1, x2, y0, y1, y2, z0, z1, z2 : complex
            The three components of each field at one sample, real or complex, V/m.

        Returns
        -------
        tuple
            The three components of the contraction, V^3/m^3.
        """
        xy = x0 * y0 + x1 * y1 + x2 * y2
        xz = x0 * z0 + x1 * z1 + x2 * z2
        yz = y0 * z0 + y1 * z1 + y2 * z2
        return (
            (xy * z0 + xz * y0 + yz * x0) / 3.0,
            (xy * z1 + xz * y1 + yz * x1) / 3.0,
            (xy * z2 + xz * y2 + yz * x2) / 3.0,
        )

    @decorate
    @no_type_check
    def dispersive_conserving3(  # noqa: PLR0913, PLR0917 - three fields of three components
        x0: complex,
        x1: complex,
        x2: complex,
        y0: complex,
        y1: complex,
        y2: complex,
        z0: complex,
        z1: complex,
        z2: complex,
    ) -> tuple[complex, complex, complex]:
        """Return the number-conserving group of three three-component analytic fields.

        ``T(x, y, z*) + T(x, y*, z) + T(x*, y, z)``, ``T`` the isotropic contraction.

        Parameters
        ----------
        x0, x1, x2, y0, y1, y2, z0, z1, z2 : complex
            The three components of each analytic field at one sample, V/m.

        Returns
        -------
        tuple
            The group's three components, V^3/m^3.
        """
        a0, a1, a2 = dispersive_isotropic3(
            x0, x1, x2, y0, y1, y2, z0.conjugate(), z1.conjugate(), z2.conjugate()
        )
        b0, b1, b2 = dispersive_isotropic3(
            x0, x1, x2, y0.conjugate(), y1.conjugate(), y2.conjugate(), z0, z1, z2
        )
        c0, c1, c2 = dispersive_isotropic3(
            x0.conjugate(), x1.conjugate(), x2.conjugate(), y0, y1, y2, z0, z1, z2
        )
        return a0 + b0 + c0, a1 + b1 + c1, a2 + b2 + c2

    @decorate
    @no_type_check
    def dispersive_real_self(x: float) -> float:
        """Return ``x^3``, the full cubic of one real field component.

        Parameters
        ----------
        x : float
            The field at one sample, V/m.

        Returns
        -------
        float
            ``(x x) x``, the reference's grouping, V^3/m^3.
        """
        return x * x * x

    @decorate
    @no_type_check
    def dispersive_real_pair(x: float, f: float) -> float:
        """Return ``x^2 f``, the product ``(E, E, F)`` of one real component.

        Parameters
        ----------
        x, f : float
            The two fields at one sample.

        Returns
        -------
        float
            ``(x x) f``, the reference's grouping.
        """
        return x * x * f

    @decorate
    @no_type_check
    def dispersive_real_cross(x: float, f: float) -> float:
        """Return ``x f^2``, the product ``(E, F, F)`` of one real component.

        Parameters
        ----------
        x, f : float
            The two fields at one sample.

        Returns
        -------
        float
            ``(x f) f``, the reference's grouping.
        """
        return x * f * f

    @decorate
    @no_type_check
    def dispersive_conserving_self(u: complex) -> complex:
        """Return ``3 |u|^2 u``, the number-conserving group of ``(u, u, u)``.

        Parameters
        ----------
        u : complex
            The analytic signal or envelope at one sample, V/m.

        Returns
        -------
        complex
            The group, V^3/m^3.
        """
        weight = 3.0 * (u.real * u.real + u.imag * u.imag)
        return complex(weight * u.real, weight * u.imag)

    @decorate
    @no_type_check
    def dispersive_conserving_pair(u: complex, f: complex) -> complex:
        """Return ``u^2 f* + 2 |u|^2 f``, the conserving group of ``(u, u, f)``.

        Parameters
        ----------
        u, f : complex
            The two analytic signals or envelopes at one sample.

        Returns
        -------
        complex
            The group.
        """
        modulus = u.real * u.real + u.imag * u.imag
        return u * u * f.conjugate() + (2.0 * modulus) * f

    @decorate
    @no_type_check
    def dispersive_conserving_cross(u: complex, f: complex) -> complex:
        """Return ``2 |f|^2 u + f^2 u*``, the conserving group of ``(u, f, f)``.

        Parameters
        ----------
        u, f : complex
            The two analytic signals or envelopes at one sample.

        Returns
        -------
        complex
            The group.
        """
        modulus = f.real * f.real + f.imag * f.imag
        return (2.0 * modulus) * u + f * f * u.conjugate()

    @decorate
    @no_type_check
    def dispersive_harmonic_self(u: complex, carrier: complex) -> complex:
        """Return ``c u^3 + c* 3 |u|^2 u*``, the harmonic groups of ``(u, u, u)``.

        Parameters
        ----------
        u : complex
            The envelope at one sample, V/m.
        carrier : complex
            ``exp(2 i omega0 t)`` at the sample.

        Returns
        -------
        complex
            The groups, V^3/m^3.
        """
        weight = 3.0 * (u.real * u.real + u.imag * u.imag)
        return carrier * (u * u * u) + carrier.conjugate() * (weight * u.conjugate())

    @decorate
    @no_type_check
    def dispersive_harmonic_pair(u: complex, f: complex, carrier: complex) -> complex:
        """Return ``c u^2 f + c* (2 |u|^2 f* + u*^2 f)``, the groups of ``(u, u, f)``.

        Parameters
        ----------
        u, f : complex
            The two envelopes at one sample.
        carrier : complex
            ``exp(2 i omega0 t)`` at the sample.

        Returns
        -------
        complex
            The groups.
        """
        modulus = u.real * u.real + u.imag * u.imag
        conjugate = u.conjugate()
        return carrier * (u * u * f) + carrier.conjugate() * (
            (2.0 * modulus) * f.conjugate() + conjugate * conjugate * f
        )

    @decorate
    @no_type_check
    def dispersive_harmonic_cross(u: complex, f: complex, carrier: complex) -> complex:
        """Return ``c u f^2 + c* (u f*^2 + 2 |f|^2 u*)``, the groups of ``(u, f, f)``.

        Parameters
        ----------
        u, f : complex
            The two envelopes at one sample.
        carrier : complex
            ``exp(2 i omega0 t)`` at the sample.

        Returns
        -------
        complex
            The groups.
        """
        modulus = f.real * f.real + f.imag * f.imag
        conjugate = f.conjugate()
        return carrier * (u * f * f) + carrier.conjugate() * (
            u * conjugate * conjugate + (2.0 * modulus) * u.conjugate()
        )

    @decorate
    @no_type_check
    def held_plasma_source(current: complex, edge: complex, inverse: float) -> complex:
        """Combine X/q - Z/q², given 1/omega at a positive active frequency.

        Parameters
        ----------
        current, edge : complex
            DFT[gJ] and DFT[g'J] in the internal FFT convention.
        inverse : float
            Reciprocal positive angular frequency, s; zero bins never reach here.

        Returns
        -------
        complex
            Plasma source in polarisation units.
        """
        return complex(0.0, -inverse) * current + (inverse * inverse) * edge

    @decorate
    @no_type_check
    def loss_scale(
        rate: float, fraction: float, squared: float, ionisation_potential: float
    ) -> float:
        """Return ``Ip W (1 - f) / |E|^2``, or zero where the divisor is not positive.

        The guard is the reference's ``where=(rate > 0) & (squared > 0)``: ``squared``
        underflows to zero below about 1e-162 V/m, and the quotient would be a NaN that
        poisons the rest of the column.

        Parameters
        ----------
        rate : float
            The ionisation rate at this sample, s^-1.
        fraction : float
            The ionised fraction at this sample, in ``[0, 1]``.
        squared : float
            ``|E|^2`` at this sample, V^2/m^2.
        ionisation_potential : float
            The potential the freed electron is charged, J.

        Returns
        -------
        float
            The coefficient the field is multiplied by, or zero.
        """
        if rate > 0.0 and squared > 0.0:
            return ionisation_potential * rate * (1.0 - fraction) / squared
        return 0.0

    @decorate
    @no_type_check
    def trapezoid(current: float, previous: float, dt: float) -> float:
        """Return one increment of a cumulative trapezoid.

        ``dt * (y_k + y_{k-1}) / 2`` in that order, which is the order
        :func:`scipy.integrate.cumulative_trapezoid` uses, so the sequential running sum
        of the plasma column reproduces the reference's multipass scan rather than
        merely agreeing with it to ``n eps``.

        Parameters
        ----------
        current : float
            The integrand at this sample.
        previous : float
            The integrand at the sample before.
        dt : float
            Sample spacing, s.

        Returns
        -------
        float
            The increment to add to the running sum.
        """
        return dt * (current + previous) / 2.0

    @protect
    @no_type_check
    def fraction_at(exposure: float, preionisation: float) -> float:
        """Return the ionised fraction at one sample of the exposure.

        ``1 - (1 - p) exp(-W_int)``: the exposure integral turned back into a
        fraction, in the reference's own order.

        Parameters
        ----------
        exposure : float
            The running integral of the rate up to this sample, dimensionless.
        preionisation : float
            The fraction already ionised at this position, in ``[0, 1]``.

        Returns
        -------
        float
            The fraction ionised, non-decreasing in time and in
            ``[preionisation, 1]``.
        """
        return 1.0 - (1.0 - preionisation) * math.exp(-exposure)

    @decorate
    @no_type_check
    def accumulate(total: float, value: float, weight: float) -> float:
        """Return one weighted stage accumulation.

        ``total + weight * value``, which is
        :func:`solprop.backends.numpy_ref.passes.accumulate` for a single element. The
        step's factor ``h`` is applied in the stage input and at acceptance, never here.

        Parameters
        ----------
        total : float
            The accumulator so far.
        value : float
            The stage's value at this element.
        weight : float
            The tableau weight.

        Returns
        -------
        float
            The updated accumulator.
        """
        return total + weight * value

    @decorate
    @no_type_check
    def low_storage_fold(  # noqa: PLR0911, PLR0913, PLR0917 - one return per table cell
        stage: int,
        mode: int,
        step: float,
        source: complex,
        state: complex,
        accumulator: complex,
        register: complex,
        first: complex,
        second: complex,
    ) -> tuple[complex, complex, complex]:
        """Fold one evaluation's source into the low-storage registers at one element.

        The finish column of SEP 24 section 4.3's flow table, ``A``, ``B`` and ``C``
        being the state, the accumulator and the stage register, ``P`` and ``Q`` the
        half-step flows and ``h`` the step:

        ====  ===============================================================
        pass  finish
        ====  ===============================================================
        1     ``A <- P A``; ``B <- A + h/6 P S``; ``C <- A + h/2 P S``
        2     ``B <- B + h/3 S``; ``C <- A + h/2 S``
        3     ``B <- B + h/3 S``; ``C <- Q (A + h S)``
        4     ``A <- Q B + h/6 S``
        ====  ===============================================================

        A pass that evaluates twice -- the material response, then the in-stage
        absorber's term, from the same input -- folds the first at the exchange and
        the second after it (section 4.3, "Two evaluations per pass"): the exchange
        is the table with pass 3's ``C <- A + h S`` left unflowed, and the second
        fold adds its source to what the exchange left, pass 3's ``C <- Q (C + h S)``
        taking the flow. The two folds' sum is the pass's.

        Parameters
        ----------
        stage : int
            The pass, 0 to 3.
        mode : int
            :data:`FOLD_COMPLETE`, :data:`FOLD_EXCHANGE` or :data:`FOLD_SECOND`.
        step : float
            The step ``h``, m.
        source : complex
            The evaluation's source at this element, the state's units per metre.
        state, accumulator, register : complex
            ``A``, ``B`` and ``C`` at this element.
        first, second : complex
            ``P`` and ``Q`` at this element; pass 1 reads ``P`` and passes 3 and 4
            ``Q``.

        Returns
        -------
        tuple of complex
            ``A``, ``B`` and ``C`` after the fold.
        """
        if stage == 0:
            slope = first * source
            if mode == 2:
                return (
                    state,
                    accumulator + (step / 6.0) * slope,
                    register + (0.5 * step) * slope,
                )
            moved = first * state
            return moved, moved + (step / 6.0) * slope, moved + (0.5 * step) * slope
        if stage == 1:
            summed = accumulator + (step / 3.0) * source
            if mode == 2:
                return state, summed, register + (0.5 * step) * source
            return state, summed, state + (0.5 * step) * source
        if stage == 2:
            summed = accumulator + (step / 3.0) * source
            if mode == 0:
                return state, summed, second * (state + step * source)
            if mode == 1:
                return state, summed, state + step * source
            return state, summed, second * (register + step * source)
        if mode == 2:
            return state + (step / 6.0) * source, accumulator, register
        return second * accumulator + (step / 6.0) * source, accumulator, register

    @decorate
    @no_type_check
    def raman_drive(real: float, imag: float, real_field: bool) -> float:
        """Return the scalar quadratic drive in V^2/m^2.

        Parameters
        ----------
        real, imag : float
            Real and imaginary field components in V/m.
        real_field : bool
            Keep the harmonic of a real field; otherwise use half the modulus square.

        Returns
        -------
        float
            Instantaneous or cycle-averaged scalar drive.
        """
        if real_field:
            return real * real
        return 0.5 * (real * real + imag * imag)

    @decorate
    @no_type_check
    def raman_tensor_drive(
        x_real: float, x_imag: float, y_real: float, y_imag: float, real_field: bool
    ) -> tuple[float, float, float]:
        """Return the three real Cartesian quadratic drives in V^2/m^2.

        Parameters
        ----------
        x_real, x_imag, y_real, y_imag : float
            Cartesian field components in V/m.
        real_field : bool
            Use the instantaneous real field; otherwise optical-cycle averaging.

        Returns
        -------
        tuple of float
            ``Qxx + Qyy``, ``Qxx - Qyy`` and ``2 Qxy``.
        """
        xx = raman_drive(x_real, x_imag, real_field)
        yy = raman_drive(y_real, y_imag, real_field)
        cross = (
            2.0 * x_real * y_real if real_field else x_real * y_real + x_imag * y_imag
        )
        return xx + yy, xx - yy, cross

    @decorate
    @no_type_check
    def raman_tensor_finish(  # noqa: PLR0913, PLR0917 - scalar kernel operands
        total: complex,
        field: complex,
        other: complex,
        diagonal: float,
        off_diagonal: float,
        factor: float,
        window: float,
    ) -> complex:
        """Add one component of the tensor polarisation and apply its collar.

        Parameters
        ----------
        total : complex
            Existing polarisation in C/m^2.
        field, other : complex
            This and the other Cartesian field component in V/m.
        diagonal, off_diagonal : float
            Convolved tensor components including the integration spacing.
        factor : float
            Vacuum permittivity times the molecular amplitude density, SI.
        window : float
            Dimensionless temporal collar in [0, 1].

        Returns
        -------
        complex
            Updated and windowed Cartesian polarisation in C/m^2.
        """
        return (total + factor * (diagonal * field + off_diagonal * other)) * window

    @decorate
    @no_type_check
    def general_kerr_component(
        tensor: Any, component: int, pairs: Any, fields: Any
    ) -> Any:
        """Return ``sum_j (sum_kl A_ijkl Q_kl) E_j`` for one component ``i``.

        The reduction order of
        :func:`solprop.backends.numpy_ref.general_modal.general_source`: the tensor
        is contracted with the pair history over ``k`` and then ``l``, and the result
        against the field over ``j``, each sum sequential from zero.

        Parameters
        ----------
        tensor : numpy.ndarray
            The node's summed ``chi3 T``, shape ``(3, 3, 3, 3)``, m²/V².
        component : int
            Which Cartesian component ``i`` to return.
        pairs : tuple
            ``Q_kl = E_k conj(E_l)``, three rows of three, V²/m².
        fields : tuple
            ``E_j``, three values, V/m.

        Returns
        -------
        complex or float
            The component, in V/m, the type of `fields`.
        """
        total = 0.0 * fields[0]
        for j in range(3):
            weighted = 0.0 * fields[0]
            for k in range(3):
                for m in range(3):
                    weighted = weighted + tensor[component, j, k, m] * pairs[k][m]
            total = total + weighted * fields[j]
        return total

    @decorate
    @no_type_check
    def general_kerr(tensor: Any, samples: Any, base: int, time: int) -> Any:
        """Return a general modal guide's instantaneous cubic at one node and sample.

        ``sum_j (sum_kl A_ijkl E_k conj(E_l)) E_j`` for ``i = x, y, z``, before the
        ``eps0``: the analytic form's dense contraction. The pair history is formed
        once and shared by the three components, as the reference's ``pair`` buffer
        is. The real form is :func:`general_real_kerr`'s.

        Parameters
        ----------
        tensor : numpy.ndarray
            The node's summed ``chi3 T``, shape ``(3, 3, 3, 3)``, m²/V².
        samples : numpy.ndarray
            The chunk's field on the fine time grid, rows ``3 q + c``, V/m.
        base : int
            The node's first row, ``3 q``.
        time : int
            The sample index.

        Returns
        -------
        tuple
            The three components, the type of `samples`.
        """
        first = samples[base, time]
        second = samples[base + 1, time]
        third = samples[base + 2, time]
        fields = (first, second, third)
        pairs = (
            (first * np.conj(first), first * np.conj(second), first * np.conj(third)),
            (
                second * np.conj(first),
                second * np.conj(second),
                second * np.conj(third),
            ),
            (third * np.conj(first), third * np.conj(second), third * np.conj(third)),
        )
        return (
            general_kerr_component(tensor, 0, pairs, fields),
            general_kerr_component(tensor, 1, pairs, fields),
            general_kerr_component(tensor, 2, pairs, fields),
        )

    @decorate
    @no_type_check
    def general_symmetric_row(tensor: Any, component: int) -> Any:
        """Return one component's coefficients of the real cubic's symmetric form.

        ``C_im``, the sum of ``A_ijkl`` over the orderings of each monomial
        ``j <= k <= l`` (``xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz``), each
        sum from zero over its orderings in lexicographic order: the order of
        :func:`solprop.backends.numpy_ref.general_modal.symmetric_kerr_coefficients`,
        so the coefficients are that function's bit for bit.

        Parameters
        ----------
        tensor : numpy.ndarray
            The node's summed ``chi3 T``, shape ``(3, 3, 3, 3)``, m²/V².
        component : int
            Which Cartesian component ``i``.

        Returns
        -------
        tuple
            The ten coefficients, m²/V².
        """
        i = component
        return (
            0.0 + tensor[i, 0, 0, 0],
            0.0 + tensor[i, 0, 0, 1] + tensor[i, 0, 1, 0] + tensor[i, 1, 0, 0],
            0.0 + tensor[i, 0, 0, 2] + tensor[i, 0, 2, 0] + tensor[i, 2, 0, 0],
            0.0 + tensor[i, 0, 1, 1] + tensor[i, 1, 0, 1] + tensor[i, 1, 1, 0],
            0.0
            + tensor[i, 0, 1, 2]
            + tensor[i, 0, 2, 1]
            + tensor[i, 1, 0, 2]
            + tensor[i, 1, 2, 0]
            + tensor[i, 2, 0, 1]
            + tensor[i, 2, 1, 0],
            0.0 + tensor[i, 0, 2, 2] + tensor[i, 2, 0, 2] + tensor[i, 2, 2, 0],
            0.0 + tensor[i, 1, 1, 1],
            0.0 + tensor[i, 1, 1, 2] + tensor[i, 1, 2, 1] + tensor[i, 2, 1, 1],
            0.0 + tensor[i, 1, 2, 2] + tensor[i, 2, 1, 2] + tensor[i, 2, 2, 1],
            0.0 + tensor[i, 2, 2, 2],
        )

    @decorate
    @no_type_check
    def general_symmetric_sum(row: Any, monomials: Any) -> float:
        """Return ``sum_m C_im monomial_m`` for one component, in order from zero.

        Parameters
        ----------
        row : tuple
            One component's ten coefficients, m²/V².
        monomials : tuple
            The ten monomials ``(E_j E_k) E_l``, V³/m³.

        Returns
        -------
        float
            The component, V/m.
        """
        total = 0.0
        for monomial in range(10):
            total = total + row[monomial] * monomials[monomial]
        return total

    @decorate
    @no_type_check
    def general_real_kerr(rows: Any, samples: Any, base: int, time: int) -> Any:
        """Return the real cubic ``sum_jkl A_ijkl E_j E_k E_l`` at one node and sample.

        The symmetric contraction of
        :func:`solprop.backends.numpy_ref.general_modal.symmetric_cubic`, in its
        order: the six pair products, each monomial ``(E_j E_k) E_l``, and each
        component the sum of its coefficients times the monomials, in order from
        zero. Before the ``eps0``.

        Parameters
        ----------
        rows : tuple
            :func:`general_symmetric_row` of the node's tensor for ``i = x, y, z``.
        samples : numpy.ndarray
            The chunk's real field on the fine time grid, rows ``3 q + c``, V/m.
        base : int
            The node's first row, ``3 q``.
        time : int
            The sample index.

        Returns
        -------
        tuple
            The three components, V/m.
        """
        first = samples[base, time]
        second = samples[base + 1, time]
        third = samples[base + 2, time]
        xx = first * first
        xy = first * second
        xz = first * third
        yy = second * second
        yz = second * third
        zz = third * third
        monomials = (
            xx * first,
            xx * second,
            xx * third,
            xy * second,
            xy * third,
            xz * third,
            yy * second,
            yy * third,
            yz * third,
            zz * third,
        )
        return (
            general_symmetric_sum(rows[0], monomials),
            general_symmetric_sum(rows[1], monomials),
            general_symmetric_sum(rows[2], monomials),
        )

    @decorate
    @no_type_check
    def general_pair_drive(first: Any, second: Any, analytic: bool) -> float:
        """Return one entry of the delayed response's real pair history, V²/m².

        ``E_k E_l`` of the real field, which carries the ``2 omega`` beat, or the
        beat-free ``Re(E_k conj(E_l)) / 2`` of the analytic signal: the reference's
        ``real(pair)`` and ``0.5 * real(pair)`` of the same Kerr pair product.

        Parameters
        ----------
        first, second : complex or float
            The components ``E_k`` and ``E_l`` at one sample, V/m.
        analytic : bool
            Whether the samples are the analytic signal.

        Returns
        -------
        float
            The drive entry.
        """
        if analytic:
            return 0.5 * (first * np.conj(second)).real
        return (first * second).real

    @decorate
    @no_type_check
    def general_delayed(  # noqa: PLR0913, PLR0917 - one sample's operands
        tensor: Any,
        component: int,
        history: Any,
        history_base: int,
        field: Any,
        field_base: int,
        time: int,
        scale: float,
    ) -> float:
        """Return ``sum_jkl D_ijkl Q_kl E_j`` at one node and sample.

        The delayed tensor contraction of
        :func:`solprop.backends.numpy_ref.general_modal.delayed_polarisation`, over
        the real field, with ``Q_kl`` read from the six stored convolved histories
        and multiplied by the inverse transform's own ``1 / (2 n_to)``, which an
        unnormalised plan does not apply.

        Parameters
        ----------
        tensor : numpy.ndarray
            One response and channel's ``chi3 D``, shape ``(3, 3, 3, 3)``, m²/V².
        component : int
            Which Cartesian component ``i`` to return.
        history : numpy.ndarray
            The convolved histories on the doubled axis, rows ``6 q + e``.
        history_base : int
            The node's first history row, ``6 q``.
        field : numpy.ndarray
            The chunk's field samples, rows ``3 q + c``; only the real part is read.
        field_base : int
            The node's first field row, ``3 q``.
        time : int
            The sample index.
        scale : float
            The inverse transform's normalisation.

        Returns
        -------
        float
            The component, before ``eps0``.
        """
        total = 0.0
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    stored = history[history_base + GENERAL_PAIR_ENTRY[k][m], time]
                    total = (
                        total
                        + tensor[component, j, k, m]
                        * (stored * scale)
                        * field[field_base + j, time].real
                    )
        return total

    @decorate
    @no_type_check
    def raman_damped(value: float, rate: float, delay: float) -> float:
        """Damp one sampled Raman row without changing its normalization.

        Parameters
        ----------
        value : float
            Undamped response sample in m^5 V^-2 s^-1.
        rate : float
            Non-negative decay rate in s^-1.
        delay : float
            Non-negative causal delay in s.

        Returns
        -------
        float
            Damped response in the input sample's units.
        """
        return value * math.exp(-rate * delay)

    @decorate
    @no_type_check
    def raman_finish(
        total: complex, field: complex, convolved: float, factor: float, window: float
    ) -> complex:
        """Add the delayed scalar polarisation before applying the temporal collar.

        Parameters
        ----------
        total : complex
            Existing real-field or complex-envelope polarisation, C/m^2.
        field : complex
            Real-field or complex-envelope sample in V/m.
        convolved : float
            Convolved drive, with the kernel's dt scaling already included.
        factor : float
            Vacuum permittivity times the response's number density, SI.
        window : float
            Dimensionless temporal collar in [0, 1].

        Returns
        -------
        complex
            Windowed polarisation. A real input family retains real scalar outputs.
        """
        delayed = factor * field * convolved
        return (total + delayed) * window

    @decorate
    @no_type_check
    def absorber_sample(
        field: complex, rate: float, distance: float, split_step: bool
    ) -> complex:
        """Apply the existing temporal-absorber formula to one real or complex sample.

        Parameters
        ----------
        field : complex
            Real field or complex envelope sample, in the caller's amplitude units.
        rate : float
            Non-negative power absorption coefficient in m^-1.
        distance : float
            Accepted step length in m, used only by the split-step scheme.
        split_step : bool
            Apply the accepted-step exponential rather than the in-stage rate.

        Returns
        -------
        complex
            Absorbed field for split-step, or field derivative per m otherwise.
            A real input family retains real outputs.
        """
        factor = -0.5 * rate
        if split_step:
            factor = math.exp(factor * distance)
        return factor * field

    return Bodies(
        spatial_element=spatial_element,
        slab_element=slab_element,
        longitudinal_rate=longitudinal_rate,
        longitudinal_root=longitudinal_root,
        mask_plane_radius=mask_plane_radius,
        aperture_weight=aperture_weight,
        region_member=region_member,
        propagator_pair=propagator_pair,
        rate_at=rate_at,
        magnitude_pair=magnitude_pair,
        kerr_real=kerr_real,
        kerr_analytic=kerr_analytic,
        dispersive_isotropic=dispersive_isotropic,
        dispersive_conserving=dispersive_conserving,
        dispersive_harmonic=dispersive_harmonic,
        dispersive_isotropic3=dispersive_isotropic3,
        dispersive_conserving3=dispersive_conserving3,
        dispersive_real_self=dispersive_real_self,
        dispersive_real_pair=dispersive_real_pair,
        dispersive_real_cross=dispersive_real_cross,
        dispersive_conserving_self=dispersive_conserving_self,
        dispersive_conserving_pair=dispersive_conserving_pair,
        dispersive_conserving_cross=dispersive_conserving_cross,
        dispersive_harmonic_self=dispersive_harmonic_self,
        dispersive_harmonic_pair=dispersive_harmonic_pair,
        dispersive_harmonic_cross=dispersive_harmonic_cross,
        held_plasma_source=held_plasma_source,
        loss_scale=loss_scale,
        trapezoid=trapezoid,
        fraction_at=fraction_at,
        accumulate=accumulate,
        low_storage_fold=low_storage_fold,
        raman_drive=raman_drive,
        raman_tensor_drive=raman_tensor_drive,
        raman_damped=raman_damped,
        raman_finish=raman_finish,
        raman_tensor_finish=raman_tensor_finish,
        absorber_sample=absorber_sample,
        envelope_population=envelope_population,
        envelope_axes=envelope_axes,
        envelope_moments=envelope_moments,
        envelope_loss=envelope_loss,
        general_kerr_component=general_kerr_component,
        general_kerr=general_kerr,
        general_symmetric_row=general_symmetric_row,
        general_symmetric_sum=general_symmetric_sum,
        general_real_kerr=general_real_kerr,
        general_pair_drive=general_pair_drive,
        general_delayed=general_delayed,
    )


PLAIN = define_bodies(lambda function: function)
"""The interpreted family: what the coverage lane executes and what tests the wiring."""
