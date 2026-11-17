# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plasma passes: ionisation rates, populations and the plasma currents.

The field grid's plasma, the ``ionisation_rate``, ``plasma_scan`` and
``plasma_current`` stages of ``responses.response_order``: :func:`ionisation_rate` looks
the rate up on the field's magnitude, :func:`plasma_scan` integrates the ionised
fraction and the free current along t, and :func:`plasma_total_current` adds the
loss current; :func:`plasma_channel_fractions` and :func:`plasma_channels_current` are
the same for independent channels. :func:`held_plasma_spectrum` turns a current held
at the temporal collar into its polarisation spectrum (SEP 27).
:func:`plasma_current` and :func:`plasma_channels_polarisation` integrate to the
time-domain polarisation, the oracle form, and
:func:`plasma_integration_diagnostics` measures the running integral (SEP 18).
The envelope grid's plasma is cycle-averaged: :func:`envelope_sources` and
:func:`envelope_populations` split each sample's complex envelope into its
polarisation ellipse by an SVD, read the cycle moments of the rate
(:func:`solprop.translate.envelope_plasma.cycle_moments`) and integrate the exposure
by the trapezoid.
"""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop.exceptions import BackendError
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.translate.envelope_plasma import (
    CycleIntegralData,
    EnvelopePlasmaAssembly,
    cycle_moments,
)
from solprop.translate.ionisation import RateTable
from solprop.translate.responses import (
    PlasmaChannel,
    PlasmaCoefficients,
)
from solprop.translate.stage import StageScalars

__all__ = [
    "envelope_populations",
    "envelope_sources",
    "held_plasma_spectrum",
    "ionisation_rate",
    "plasma_channel_fractions",
    "plasma_channels_current",
    "plasma_channels_polarisation",
    "plasma_current",
    "plasma_integration_diagnostics",
    "plasma_integration_statistics",
    "plasma_scan",
    "plasma_total_current",
]


def ionisation_rate(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    table: RateTable,
) -> Float[np.ndarray, "n_nodes_chunk n_to"]:
    r"""Look the ionisation rate up on the magnitude of the field.

    The ``intensity`` and ``ionisation_rate`` stages of the fused pass, together: the
    magnitude is what the intensity stage forms and the only thing the rate needs. For
    one component that is ``|E|``; for two it is ``hypot(E_x, E_y)`` rather than
    ``sqrt(E_x^2 + E_y^2)``, because the tunnelling exponent is tens to hundreds and
    turns one unit in the last place of the magnitude into 1e-13 of the rate
    (:meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate` takes the same route).

    The table is exactly zero below its ``field_min`` and holds its last value above
    ``field_max`` rather than extrapolating, which is what keeps the loss current of
    :func:`plasma_current` finite and a diverged step from becoming an infinity.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    table : solprop.translate.ionisation.RateTable
        The compiled rate, ``responses.plasma_coefficients.rate``.

    Returns
    -------
    numpy.ndarray
        The rate per atom in s^-1, shape ``(n_nodes_chunk, n_to)``, finite and
        non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.plasma import ionisation_rate
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.translate.ionisation import build_rate_table
    >>> table = build_rate_table(OptmatRate("He", 800e-9), (2e9, 1e11), rtol=1e-4)
    >>> field = np.array([[[5e10, 1e6]]])
    >>> np.round(np.log10(ionisation_rate(field, table)[0, 0]), 3).item()
    11.207
    """
    return table(_magnitude(field_t))


def plasma_scan(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    coefficients: PlasmaCoefficients,
    *,
    preionisation: float,
    dt: float,
    integral: RunningIntegral | None = None,
) -> tuple[
    Float[np.ndarray, "n_nodes_chunk n_to"],
    Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
]:
    r"""Integrate the ionised fraction and the free-electron current along t.

    The ``plasma_scan`` stage: two of the three cumulative integrals the response
    declares, which is why its domain is ``memory`` -- the value at one sample depends
    on every earlier sample and on none of the later ones.

    ``f = 1 - (1 - f_0) exp(-int W dt)``, with the pre-ionised fraction **inside** the
    exponential. Writing ``f_0 + 1 - exp(-int W dt)`` instead would let ``f`` exceed
    one, make ``1 - f`` negative and stop ``df/dt`` being ``W (1 - f)``, so the loss
    term would no longer pay exactly one ionisation potential per electron. Then
    ``J_free = (e^2/m_e) int f E dt``, the current of electrons already freed and still
    being accelerated: what lowers the refractive index and blue-shifts the pulse.

    Both integrals are the problem's running-integral rule (SEP 18), applied in the
    argument order of :meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate`, each
    a vectorised cumulative sum (``scipy.integrate.cumulative_trapezoid``); this pass
    and the oracle are the same computation, and other backends' scans are held to the
    scan's rounding bound (SEP 25), not to its order. Under the band
    rule the exposure is floored at zero, so the fraction stays in
    ``[preionisation, 1)`` and is non-decreasing to within the filter's undershoot. Here
    the multipass form **is** the reference, and fusing the three sums into one
    sequential sweep belongs to the native backend
    (``docs/guide/media/nonlinear.md``).

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    rate : numpy.ndarray
        The ionisation rate from :func:`ionisation_rate`, s^-1, shape
        ``(n_nodes_chunk, n_to)``.
    coefficients : solprop.translate.responses.PlasmaCoefficients
        The compiled plasma group; only ``charge_mass_ratio`` is read here.
    preionisation : float
        The pre-ionised fraction at this position,
        ``coefficients.preionisation(z)``, in ``[0, 1]``.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule, ``responses.plasma_integral``; ``None`` is the
        trapezoid.

    Returns
    -------
    fraction : numpy.ndarray
        The fraction ionised, shape ``(n_nodes_chunk, n_to)``, in
        ``[preionisation, 1]`` and non-decreasing in time (under the band rule, to
        within the filter's undershoot).
    free_current : numpy.ndarray
        ``J_free`` per unit density, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        C/(m^2 s) per m^-3.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.plasma import plasma_scan
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> compiled = plasma_coefficients(
    ...     PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)), (1e9, 1e11), z_max=1.0
    ... )
    >>> field = np.zeros((1, 1, 4))
    >>> fraction, current = plasma_scan(
    ...     field, np.zeros((1, 4)), compiled, preionisation=0.25, dt=1e-16
    ... )
    >>> fraction.tolist(), bool(np.all(current == 0.0))
    ([[0.25, 0.25, 0.25, 0.25]], True)
    """
    rule = _rule(integral)
    exposure = rule.floor(rule.integrate(rate, dt))
    fraction = 1.0 - (1.0 - preionisation) * np.exp(-exposure)
    integrand = fraction[..., np.newaxis, :] * coefficients.charge_mass_ratio * field_t
    free_current = rule.integrate(integrand, dt)
    return fraction, free_current


def _rule(integral: RunningIntegral | None) -> RunningIntegral:
    """Return the running-integral rule a plasma pass uses.

    - ``integral``: the problem's rule, or ``None`` for the trapezoid.
    """
    return RunningIntegral.trapezoid() if integral is None else integral


def plasma_current(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    fraction: Float[np.ndarray, "n_nodes_chunk n_to"],
    free_current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    coefficients: PlasmaCoefficients,
    *,
    density: float,
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Add the ionisation loss to the free current and integrate to a polarisation.

    The ``plasma_current`` stage, and the third cumulative integral.
    ``J_loss = I_p W (1 - f) E / |E|^2`` is added pointwise; its form is fixed by
    ``E . J_loss = I_p W (1 - f) = I_p df/dt``, so the field pays exactly one ionisation
    potential for every electron it frees. For one component it is ``I_p W (1-f)/E``,
    sign and all; for two it is that magnitude **directed along the field**, which is
    what makes a linearly polarised pulse give the scalar answer rotated. The
    polarisation is ``P = rho int (J_free + J_loss) dt``.

    The division is written with a ``where`` mask rather than a branch, as the oracle
    writes it, so it never happens at a sample where the rate or the field vanishes: the
    term is finite only because the rate is **exactly** zero below threshold, which is
    what :class:`solprop.translate.ionisation.RateTable` guarantees below its
    ``field_min``; a clipped but non-zero rate would divide by a tiny ``|E|^2``.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in
        V/m.
    rate : numpy.ndarray
        The ionisation rate from :func:`ionisation_rate`, s^-1, shape
        ``(n_nodes_chunk, n_to)``.
    fraction : numpy.ndarray
        The fraction ionised from :func:`plasma_scan`, shape
        ``(n_nodes_chunk, n_to)``.
    free_current : numpy.ndarray
        ``J_free`` from :func:`plasma_scan`, shape ``(n_nodes_chunk, n_pol, n_to)``.
    coefficients : solprop.translate.responses.PlasmaCoefficients
        The compiled plasma group; only ``ionisation_potential`` is read here.
    density : float
        The density path at this position, ``responses.density(z)``, in m^-3,
        non-negative. The currents are per atom; this is what turns them into a
        polarisation.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule, ``responses.plasma_integral``; ``None`` is the
        trapezoid.

    Returns
    -------
    numpy.ndarray
        The plasma polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.plasma import plasma_current
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> from solprop.translate.responses import plasma_coefficients
    >>> compiled = plasma_coefficients(
    ...     PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)), (1e9, 1e11), z_max=1.0
    ... )
    >>> zeros = np.zeros((1, 1, 4))
    >>> flat = np.zeros((1, 4))
    >>> plasma_current(
    ...     zeros, flat, flat, zeros, compiled, density=1e25, dt=1e-16
    ... ).tolist()
    [[[0.0, 0.0, 0.0, 0.0]]]
    """
    current = plasma_total_current(field_t, rate, fraction, free_current, coefficients)
    return density * _rule(integral).integrate(current, dt)


def plasma_total_current(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    rate: Float[np.ndarray, "n_nodes_chunk n_to"],
    fraction: Float[np.ndarray, "n_nodes_chunk n_to"],
    free_current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    coefficients: PlasmaCoefficients,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    """Add the guarded loss current to the free current, per unit density.

    ``J_free + I_p W (1 - f) E / |E|^2``, the integrand of :func:`plasma_current` and
    what a stage holds at the collar (SEP 27); the guard is the one
    :func:`plasma_current` describes.

    Parameters
    ----------
    field_t : numpy.ndarray
        The real field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``,
        in V/m.
    rate : numpy.ndarray
        The ionisation rate from :func:`ionisation_rate`, s^-1, shape
        ``(n_nodes_chunk, n_to)``.
    fraction : numpy.ndarray
        The fraction ionised from :func:`plasma_scan`, shape ``(n_nodes_chunk, n_to)``.
    free_current : numpy.ndarray
        ``J_free`` from :func:`plasma_scan`, shape ``(n_nodes_chunk, n_pol, n_to)``.
    coefficients : solprop.translate.responses.PlasmaCoefficients
        The compiled plasma group; only ``ionisation_potential`` is read here.

    Returns
    -------
    numpy.ndarray
        The total current per unit number density, in A m, with the field's shape.
    """
    # The squared magnitude rather than the field itself: it is the divisor the guard
    # has to test, and it underflows to zero below about 1e-162 V/m.
    squared = _magnitude(field_t) ** 2
    scale = np.zeros_like(squared)
    np.divide(
        coefficients.ionisation_potential * rate * (1.0 - fraction),
        squared,
        out=scale,
        where=(rate > 0.0) & (squared > 0.0),
    )
    return free_current + scale[..., np.newaxis, :] * field_t


def plasma_channel_fractions(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> tuple[
    Float[np.ndarray, "n_channels n_nodes_chunk n_to"],
    Float[np.ndarray, "n_channels n_nodes_chunk n_to"],
]:
    """Evaluate independent exposure scans in channel order.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape (n_nodes_chunk, n_pol, n_to).
    channels : tuple of PlasmaChannel
        The independent rate tables and population metadata.
    preionisation : tuple of float
        Initial fractions, in channel order, in [0, 1].
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule; ``None`` is the trapezoid.

    Returns
    -------
    rates : numpy.ndarray
        Instantaneous rates in s^-1, shape (n_channels, n_nodes_chunk, n_to).
    fractions : numpy.ndarray
        Ionised fractions with the same shape, dimensionless.

    Raises
    ------
    BackendError
        If the channel scalars do not match their declared count.
    """
    if not channels or len(channels) != len(preionisation):
        raise BackendError("plasma channels require one initial fraction each")
    magnitude = _magnitude(field_t)
    rates = np.stack([channel.rate(magnitude) for channel in channels])
    fractions = np.empty_like(rates)
    rule = _rule(integral)
    for index, initial in enumerate(preionisation):
        exposure = rule.floor(rule.integrate(rates[index], dt))
        fractions[index] = 1.0 - (1.0 - initial) * np.exp(-exposure)
    return rates, fractions


def plasma_channels_polarisation(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    densities: tuple[float, ...],
    preionisation: tuple[float, ...],
    dt: float,
    *,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Sum channel weights before the two shared current integrals.

    Each fraction has its own exposure scan (:func:`plasma_channel_fractions`). The
    free-current weight is ``sum rho_c f_c e_c^2/m_c`` (times each channel's electrons
    per event); the loss numerator is ``sum rho_c I_p,c W_c (1-f_c)``, divided by
    ``|E|^2`` and directed along the field as in :func:`plasma_current`. The densities
    are inside the sums, where the one-channel pass keeps its density outside the
    integrals.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape (n_nodes_chunk, n_pol, n_to).
    channels : tuple of PlasmaChannel
        Ordered independent channel coefficients.
    densities : tuple of float
        Number density of each population in m^-3.
    preionisation : tuple of float
        Initial fraction of each population, dimensionless in [0, 1].
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule shared by every channel; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        Plasma polarisation in C/m², with the field shape.

    Raises
    ------
    BackendError
        If there is not one density per channel.
    """
    return _rule(integral).integrate(
        plasma_channels_current(
            field_t, channels, densities, preionisation, dt, integral
        ),
        dt,
    )


def plasma_channels_current(  # noqa: PLR0917 - the scan's positional signature
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: tuple[PlasmaChannel, ...],
    densities: tuple[float, ...],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Return every channel's free and loss current, summed, before the last integral.

    The integrand of :func:`plasma_channels_polarisation`, and what a stage holds at
    the collar: the free current is the one shared integral of the weighted field, the
    loss current is pointwise.

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape ``(n_nodes_chunk, n_pol, n_to)``.
    channels : tuple of PlasmaChannel
        Ordered independent channel coefficients.
    densities : tuple of float
        Number density of each population in m^-3.
    preionisation : tuple of float
        Initial fraction of each population, dimensionless in ``[0, 1]``.
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule shared by every channel; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        The total plasma current density in A/m^2, with the field's shape.

    Raises
    ------
    BackendError
        If there is not one density per channel.
    """
    if len(densities) != len(channels):
        raise BackendError("plasma channels require one density each")
    rule = _rule(integral)
    rates, fractions = plasma_channel_fractions(
        field_t, channels, preionisation, dt, rule
    )
    weight = np.zeros_like(rates[0])
    work = np.zeros_like(weight)
    for channel, density, rate, fraction in zip(
        channels, densities, rates, fractions, strict=True
    ):
        weight += (
            density * fraction * channel.charge_mass_ratio * channel.electrons_per_event
        )
        work += density * channel.ionisation_potential * rate * (1.0 - fraction)
    free = rule.integrate(weight[:, None, :] * field_t, dt)
    squared = _magnitude(field_t) ** 2
    guard = np.zeros_like(work)
    np.divide(work, squared, out=guard, where=squared > 0.0)
    loss = guard[:, None, :] * field_t
    return free + loss


def plasma_integration_diagnostics(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
    channels: Sequence[PlasmaCoefficients],
    preionisation: tuple[float, ...],
    dt: float,
    integral: RunningIntegral | None = None,
) -> Float[np.ndarray, " 6"]:
    """Return the parts of the column's integration diagnostics (SEP 18).

    For every channel and node: the exposure ``S`` as the rule forms it, before the
    floor, and its increments; the fraction; the work the loss current pays per
    ionisation potential, ``int W (1 - f) dt`` by the trapezoid of the raw rate; the
    electrons that freed, ``f_end - f_0``; and the free-current integrand ``|f E|``.
    Every part is a maximum over nodes and channels, so chunks combine by taking the
    largest of each, and each ratio is normalised by the most-ionised node rather than
    node by node: a node the field barely ionises, whose rate is one sampled spike,
    would otherwise report the filter's full undershoot on a rate at the level of
    rounding (:func:`plasma_integration_statistics` forms the ratios).

    Parameters
    ----------
    field_t : numpy.ndarray
        Real node fields in V/m, shape ``(n_nodes_chunk, n_pol, n_to)``.
    channels : sequence of PlasmaCoefficients
        The channels' compiled coefficients, each with its rate table.
    preionisation : tuple of float
        Each channel's initial fraction at this plane.
    dt : float
        Fine-grid sample spacing in s.
    integral : solprop.nonlinear.ionisation.RunningIntegral or None, optional
        The running-integral rule; ``None`` is the trapezoid.

    Returns
    -------
    numpy.ndarray
        The largest exposure dip ``-min dS`` and rise ``max dS``; the largest
        ``|paid - freed|`` and ``freed``; the largest ``|f E|`` in the first ``2K``
        samples and anywhere. Dimensionless but for the last two, in V/m.
    """
    rule = _rule(integral)
    reach = 2 * rule.half_width
    magnitude = _magnitude(field_t)
    parts = np.zeros(6)
    for channel, initial in zip(channels, preionisation, strict=True):
        rate = channel.rate(magnitude)
        exposure = rule.integrate(rate, dt)
        steps = np.diff(exposure, axis=-1)
        fraction = 1.0 - (1.0 - initial) * np.exp(-rule.floor(exposure))
        paid = np.trapezoid(rate * (1.0 - fraction), dx=dt, axis=-1)
        freed = fraction[..., -1] - fraction[..., 0]
        integrand = np.abs(fraction[..., np.newaxis, :] * field_t)
        early = float(np.max(integrand[..., :reach])) if reach else 0.0
        parts = np.maximum(
            parts,
            [
                max(-float(np.min(steps)), 0.0),
                float(np.max(steps)),
                float(np.max(np.abs(paid - freed))),
                float(np.max(freed)),
                early,
                float(np.max(integrand)),
            ],
        )
    return parts


def plasma_integration_statistics(
    parts: Float[np.ndarray, " 6"],
) -> tuple[float, float, float]:
    """Form the three diagnostics from :func:`plasma_integration_diagnostics`'s parts.

    Parameters
    ----------
    parts : numpy.ndarray
        The six maxima, combined over every chunk of nodes.

    Returns
    -------
    tuple of float
        The undershoot ``max(-dS) / max(dS)``, the work defect
        ``max |paid - freed| / max freed`` and the boundary level, each zero where
        its denominator is.
    """
    dip, rise, defect, freed, early, peak = (float(part) for part in parts)
    return (
        dip / rise if rise > 0.0 else 0.0,
        defect / freed if freed > 0.0 else 0.0,
        early / peak if peak > 0.0 else 0.0,
    )


def held_plasma_spectrum(
    gated: Float[np.ndarray, "... n_to"],
    edge: Float[np.ndarray, "... n_to"],
    omega: Float[np.ndarray, " n_omega"],
    crop_index: Int[np.ndarray, " n_omega"],
    active: Bool[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "... n_omega"]:
    r"""Transform a current held at the temporal collar (SEP 27 section 4.4).

    Form ``DFT[g J]/(i omega) - DFT[g' J]/(i omega)^2`` only on active bins.
    Projection can precede or follow this linear map; crop and amplitude factors
    belong to the route. No running integral or numerical derivative enters it.

    Parameters
    ----------
    gated, edge : numpy.ndarray
        Sampled ``g J`` and ``g' J``, in A/m^2 and A/(m^2 s), with time last.
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape ``(n_omega,)``.
    crop_index : numpy.ndarray
        Coarse positions in the fine half-spectrum, shape ``(n_omega,)``.
    active : numpy.ndarray
        Active frequency mask, shape ``(n_omega,)``; all selected frequencies positive.

    Returns
    -------
    numpy.ndarray
        Polarisation-equivalent spectrum in raw DFT units, with inactive bins zero.
    """
    current = np.fft.rfft(gated, axis=-1)[..., crop_index]
    derivative = np.fft.rfft(edge, axis=-1)[..., crop_index]
    result = np.zeros_like(current)
    q = 1j * omega[active]
    result[..., active] = current[..., active] / q - derivative[..., active] / q**2
    return result


def _magnitude(
    field_t: Float[np.ndarray, "n_nodes_chunk n_pol n_to"],
) -> Float[np.ndarray, "n_nodes_chunk n_to"]:
    """Return the field magnitude the ionisation rate and the loss term take.

    - Returns: ``|E|`` in V/m, shape ``(n_nodes_chunk, n_to)``.
    """
    if field_t.shape[-2] == 1:
        return np.abs(field_t[..., 0, :])
    return np.hypot(field_t[..., 0, :], field_t[..., 1, :])


def envelope_sources(
    field_t: Complex[np.ndarray, "n_nodes n_pol n_to"],
    assembly: EnvelopePlasmaAssembly,
    scalars: StageScalars,
    dt: float,
    window: Float[np.ndarray, " n_to"],
) -> tuple[
    Complex[np.ndarray, "n_nodes n_pol n_to"],
    Complex[np.ndarray, "n_nodes n_pol n_to"],
]:
    """Form two windowed sources from independently recomputed channel populations.

    Parameters
    ----------
    field_t : numpy.ndarray
        Physical complex V/m envelope, shape ``(n_nodes, n_pol, n_to)``.
    assembly : EnvelopePlasmaAssembly
        Labelled channels and validated cycle data.
    scalars : StageScalars
        Density and pre-ionisation at this RHS position in channel order.
    dt : float
        Fine time spacing in s.
    window : numpy.ndarray
        Material-source collar, shape ``(n_to,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Drude drive in A m^-2 s^-1 and loss current in A m^-2, each with the
        field's shape. Projection and temporal FFT follow this function.
    """
    drive = np.zeros_like(field_t)
    loss = np.zeros_like(field_t)
    for channel, density, initial in zip(
        assembly.channels,
        scalars.plasma_densities,
        scalars.plasma_preionisation,
        strict=True,
    ):
        for node, field in enumerate(field_t):
            rate, current = _cycle_quantities(field, channel.cycle, loss=True)
            fraction, survival = _population(rate, initial, dt)
            drive[node] += (channel.charge_mass_ratio * density * fraction) * field
            loss[node] += (channel.ionisation_potential * density * survival) * current
    return drive * window, loss * window


def _cycle_quantities(
    field: Complex[np.ndarray, "n_pol n_to"], cycle: CycleIntegralData, *, loss: bool
) -> tuple[Float[np.ndarray, " n_to"], Complex[np.ndarray, "n_pol n_to"]]:
    """Evaluate rate and optional directional loss by independent SVD axes.

    - ``field``: physical V/m envelope, shape (n_pol, n_to).
    - Returns: rate in s^-1 and loss moment per unit ionisation work, with shapes
      (n_to,) and (n_pol, n_to).
    """
    rate = np.zeros(field.shape[1])
    current = np.zeros_like(field)
    for sample in range(field.shape[1]):
        a = field[:, sample]
        scale = float(np.max(np.abs(a)))
        if scale == 0:
            continue
        matrix = np.column_stack((np.real(a) / scale, np.imag(a) / scale))
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        major = float(singular[0]) * scale
        ratio = (
            0.0 if len(singular) == 1 else min(1.0, float(singular[1] / singular[0]))
        )
        moments = cycle_moments(major, ratio, cycle)
        rate[sample] = moments[0] + ratio**2 * moments[1]
        if not loss:
            continue
        current[:, sample] = (
            (2 * (moments[0] / major)) * left[:, 0] * (right[0, 0] + 1j * right[0, 1])
        )
        if len(singular) == 2:
            current[:, sample] += (
                (2 * (ratio * moments[1] / major))
                * left[:, 1]
                * (right[1, 0] + 1j * right[1, 1])
            )
    return rate, current


def _population(
    rate: Float[np.ndarray, " n_to"], initial: float, dt: float
) -> tuple[Float[np.ndarray, " n_to"], Float[np.ndarray, " n_to"]]:
    """Retain survival separately from a cancellation-free population scan.

    - ``rate``: Cycle-averaged rates in s^-1, shape (n_to,).
    - ``initial``: initial ionised fraction in [0, 1].
    - ``dt``: fine time spacing in s.
    - Returns: ionised fraction and surviving neutral fraction, each shape (n_to,).
    """
    with np.errstate(over="ignore"):
        increments = (0.5 * rate[:-1] + 0.5 * rate[1:]) * dt
        exposure = np.concatenate((np.zeros(1), np.cumsum(increments)))
    survival = (1 - initial) * np.exp(-exposure)
    fraction = initial + (1 - initial) * -np.expm1(-exposure)
    return fraction, survival


def envelope_populations(
    field_t: Complex[np.ndarray, "n_nodes n_pol n_to"],
    assembly: EnvelopePlasmaAssembly,
    initials: tuple[float, ...],
    dt: float,
) -> tuple[
    Float[np.ndarray, "n_channels n_nodes"], Float[np.ndarray, "n_channels n_nodes"]
]:
    """Recompute each node's final population and peak cycle-averaged rate.

    Parameters
    ----------
    field_t : numpy.ndarray
        Physical V/m envelopes, shape (n_nodes, n_pol, n_to).
    assembly : EnvelopePlasmaAssembly
        Compiled channels in population order.
    initials : tuple of float
        Initial fractions at this position in channel order.
    dt : float
        Fine time spacing in s.

    Returns
    -------
    tuple of numpy.ndarray
        Final fractions and peak cycle-averaged rates in s^-1, both shaped
        (n_channels, n_nodes). Every call scans the supplied field afresh.
    """
    fractions = np.empty((len(assembly.channels), len(field_t)))
    peaks = np.empty_like(fractions)
    for index, (channel, initial) in enumerate(
        zip(assembly.channels, initials, strict=True)
    ):
        for node, field in enumerate(field_t):
            rate, _ = _cycle_quantities(field, channel.cycle, loss=False)
            fraction, _ = _population(rate, initial, dt)
            fractions[index, node] = fraction[-1]
            peaks[index, node] = np.max(rate)
    return fractions, peaks
