# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The response passes: the polarisation of the field at a chunk of nodes.

Pass 4 of ``docs/dev/architecture.md`` section 8. :func:`nonlinear_sources` walks
``responses.response_order`` stage by stage and returns :class:`NonlinearSources`, the
collared local polarisation and, separately, the ungated plasma current, which a stage
assembles in frequency
(:func:`~solprop.backends.numpy_ref.passes.plasma.held_plasma_spectrum`).
Its stages are the named functions here and in
:mod:`~solprop.backends.numpy_ref.passes.plasma`: :func:`kerr_polarisation` (the
instantaneous cubic), :func:`dispersive_kerr_polarisation` (SEP 6),
:func:`raman_polarisation` (the delayed term on the doubled grid, scalar or vector)
with the kernel spectra it convolves with (:func:`raman_spectrum`,
:func:`mixture_raman_spectrum`, :func:`tensor_raman_spectrum`).
:func:`nonlinear_polarisation` is the time-domain total, the physical-polarisation
oracle that tests hold the responses to.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Num

from solprop import fourier
from solprop.backends.numpy_ref.passes.plasma import (
    ionisation_rate,
    plasma_channels_current,
    plasma_scan,
    plasma_total_current,
)
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.nonlinear.raman import raman_drive
from solprop.nonlinear.responses import dispersive_cubic
from solprop.translate.ionisation import RateTable
from solprop.translate.raman import kernel_spectrum
from solprop.translate.responses import (
    CompiledResponses,
    DispersiveKerrTerms,
    KerrCoefficients,
    PlasmaCoefficients,
    RamanCoefficients,
    ResponseStage,
)
from solprop.translate.stage import StageScalars

__all__ = [
    "NonlinearSources",
    "dispersive_kerr_polarisation",
    "kerr_polarisation",
    "mixture_raman_spectrum",
    "nonlinear_polarisation",
    "nonlinear_sources",
    "raman_polarisation",
    "raman_spectrum",
    "tensor_raman_spectrum",
]


def kerr_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    kerr: KerrCoefficients,
    density: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate the instantaneous Kerr polarisation at every node of a chunk.

    The ``kerr`` stage of the fused pass. The formulas are the two cubic forms of
    :class:`solprop.translate.responses.KerrCoefficients`, which were read off
    :meth:`solprop.nonlinear.responses.KerrResponse.evaluate` rather than re-derived:

    ``real``
        ``P_p = c rho (sum_q E_q^2) E_p`` -- the full cubic of a real field, which
        already carries the third harmonic and the counter-rotating terms.
    ``analytic``
        ``P_p = c rho [s (sum_q |a_q|^2) a_p + k (sum_q a_q^2) conj(a_p)]``, plus
        ``c rho m exp(2 i omega_0 t) (sum_q a_q^2) a_p`` when the envelope grid radiates
        the harmonic, with ``s``, ``k`` and ``m`` the self, conjugate and harmonic
        factors. For one component the first collapses to ``(3/4)|a|^2 a``.

    The sums run over the **polarisation** axis of one node, which is what the
    vector-isotropic coupling means: two components of the field at one point of the
    cross-section, never two different points.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid.
    kerr : solprop.translate.responses.KerrCoefficients
        The compiled coefficients, whose ``coefficient`` is ``eps0 chi3`` per unit of
        the density path.
    density : float
        The density path at this position, ``responses.density(z)``: a number density in
        m^-3 for a gas, 1.0 for a medium whose ``chi3`` is already absolute.
        Non-negative.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, shape ``(n_to,)``. Required when
        ``kerr.harmonic_factor`` is not zero; the harmonic term is added whenever a
        carrier is given, and ``None`` selects the fundamental cubic alone.

    Returns
    -------
    numpy.ndarray
        The polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.responses import kerr_polarisation
    >>> from solprop.translate.responses import KerrCoefficients
    >>> coefficients = KerrCoefficients(1.0, "analytic", 0.5, 0.25, 0.0, False, False)
    >>> field = np.array([[[2.0 + 0.0j]]])
    >>> polarisation = kerr_polarisation(field, coefficients, 1.0)
    >>> bool(np.allclose(polarisation, 0.75 * abs(2.0) ** 2 * 2.0))
    True
    """
    values = fourier.analytic_signal(field_t) if kerr.analytic_signal else field_t
    factor = kerr.coefficient * density
    if kerr.cubic_form == "real":
        polarisation = factor * np.sum(values**2, axis=-2, keepdims=True) * values
    else:
        squares = np.sum(values**2, axis=-2, keepdims=True)
        moduli = np.sum(np.abs(values) ** 2, axis=-2, keepdims=True)
        polarisation = factor * (
            kerr.self_factor * moduli * values
            + kerr.conjugate_factor * squares * np.conj(values)
        )
        if carrier is not None:
            polarisation = polarisation + (
                factor * kerr.harmonic_factor * carrier * squares * values
            )
    return polarisation.real if kerr.take_real else polarisation


def dispersive_kerr_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    terms: DispersiveKerrTerms,
    densities: Sequence[float],
    *,
    kind: GridKind,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    window: Float[np.ndarray, " n_to_window"] | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate SEP 6's dispersive electronic polarisation at every node of a chunk.

    The ``dispersive_kerr`` stage of the fused pass: per channel, the transforms of
    the field the evaluator's forms need, the products of the filtered fields in the
    leaf's sector, the temporal collar on each product, the output multipliers and one
    inverse transform, by :func:`solprop.nonlinear.responses.dispersive_cubic`, which
    is the arithmetic of the reference
    :meth:`~solprop.nonlinear.responses.DispersiveKerrResponse.evaluate` on the leaf's
    sampled multipliers. The polynomial's channels add, ``P = sum_s c_s rho_s
    cubic_s(E)``; the symmetric evaluator's mixture takes one filter built from its
    species' densities at this stage, ``P = F G_mix[w (G_mix E)^3]``
    (:meth:`~solprop.translate.responses.DispersiveKerrTerms.evaluations`). It reads the
    bare field and touches no other response's source.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid.
    terms : solprop.translate.responses.DispersiveKerrTerms
        The compiled leaf.
    densities : sequence of float
        Each channel's number density at this position, m^-3, in the leaf's order.
    kind : {'field', 'envelope'}
        The spectral layout.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, shape ``(n_to,)``; required when
        the leaf's ``carrier`` is set.
    window : numpy.ndarray or None, optional
        The temporal collar, shape ``(n_to,)``, applied to each product before the
        output multipliers, ``G[w (G E)^3]``; ``None`` is a unit collar.

    Returns
    -------
    numpy.ndarray
        The polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid, the collar already inside it.

    Raises
    ------
    BackendError
        If the densities do not match the channels, or the carrier is missing.
    """
    if len(densities) != terms.n_channels:
        msg = (
            f"the dispersive Kerr leaf has {terms.n_channels} channels and the stage "
            f"carries {len(densities)} densities"
        )
        raise BackendError(msg)
    if terms.carrier and carrier is None:
        raise BackendError(
            "the dispersive Kerr leaf keeps the full cubic on an envelope grid and no "
            "harmonic carrier was given"
        )
    total = np.zeros_like(field_t)
    for _, factor, weights in terms.evaluations(densities):
        total = total + factor * dispersive_cubic(
            field_t,
            weights,
            kind=kind,
            sector=terms.sector,
            carrier=carrier if terms.carrier else None,
            window=window,
        )
    return total


def raman_spectrum(
    undamped: Float[np.ndarray, "n_terms n_to_doubled"],
    rates: Sequence[float] | np.ndarray,
    fraction: float,
    dt: float,
) -> Complex[np.ndarray, " n_omega_doubled"]:
    r"""Damp each undamped Raman row at its own rate, sum, scale and transform.

    The half of the kernel that moves along z. ``raman_kernel_undamped`` holds one
    causal, tapered row per independently damped term and ``raman_decay_rates`` one
    scalar path per row (``kernel-contract.md`` section 5); this forms
    ``f_R sum_k u_k(t) exp(-gamma_k t)`` on the doubled grid and hands it to
    :func:`solprop.translate.raman.kernel_spectrum`, which folds in the sample spacing
    the discrete sum needs to become an integral. There is one statement of that
    normalisation and this is not it.

    Damping each term separately is the whole reason the contract carries rows rather
    than one array: a molecular gas's rotational and vibrational coherences decay 3.5
    times (H2) to 17 times (N2) apart at one bar, so damping their sum once would be
    wrong for every such gas (``docs/guide/media/raman-numerics.md``).

    The terms are summed in the order of the rows and the fraction multiplies the sum,
    which is the arithmetic of :func:`solprop.translate.raman.sample_kernel` operation
    for operation, so the two spectra agree to the last bit. The padding half of each
    row is zero, so damping it costs nothing and changes nothing.

    Parameters
    ----------
    undamped : numpy.ndarray
        ``responses.raman_kernel_undamped``, float64, shape ``(n_terms, 2 n_to)``, in
        m^5 V^-2 s^-1.
    rates : array_like
        The decay rate of each term at this position, s^-1, non-negative, one per row:
        ``[table(z) for table in responses.raman_decay_rates]``.
    fraction : float
        ``responses.raman_fraction``, in ``(0, 1]``; 1 for an absolute molecular
        response.
    dt : float
        Sample spacing of the fine time grid, s, positive.

    Returns
    -------
    numpy.ndarray
        The damped kernel spectrum with the ``dt`` scaling folded in, complex128, shape
        ``(n_to + 1,)``, in the **internal** convention.

    Raises
    ------
    BackendError
        If there is not one rate per row.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.responses import raman_spectrum
    >>> rows = np.zeros((1, 8))
    >>> rows[0, 1] = 1.0
    >>> spectrum = raman_spectrum(rows, [0.0], 1.0, 2.0)
    >>> spectrum.shape, bool(np.isclose(spectrum[0], 2.0))
    ((5,), True)
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size != undamped.shape[0]:
        msg = (
            f"the Raman kernel has {undamped.shape[0]} row(s) and {values.size} decay "
            "rate(s) were given; a backend damps row k at rate k"
        )
        raise BackendError(msg)
    delays = np.arange(undamped.shape[1]) * dt
    total = np.zeros(undamped.shape[1])
    for row, rate in zip(undamped, values, strict=True):
        total += row * np.exp(-rate * delays)
    return kernel_spectrum(fraction * total, dt)


def mixture_raman_spectrum(
    components: tuple[RamanCoefficients, ...],
    rates: tuple[float, ...],
    densities: tuple[float, ...],
    dt: float,
) -> Complex[np.ndarray, " n_omega_doubled"]:
    """Form one amplitude-weighted kernel spectrum for all Raman components.

    Parameters
    ----------
    components : tuple of RamanCoefficients
        Raman-active populations in mixture order.
    rates : tuple of float
        Decay rates in s^-1, flattened by component then kernel row.
    densities : tuple of float
        Component amplitude densities in m^-3.
    dt : float
        Fine-grid sample spacing in s.

    Returns
    -------
    numpy.ndarray
        One damped kernel spectrum including densities and dt.

    Raises
    ------
    BackendError
        If the amplitude or decay-rate counts disagree with the components.
    """
    if (
        not components
        or len(densities) != len(components)
        or len(rates) != sum(c.n_terms for c in components)
    ):
        raise BackendError("Raman component amplitudes and rates must match their rows")
    delays = np.arange(components[0].undamped.shape[1]) * dt
    total = np.zeros_like(delays)
    offset = 0
    for component, density in zip(components, densities, strict=True):
        amplitude = np.zeros_like(total)
        for row in component.undamped:
            amplitude += row * np.exp(-rates[offset] * delays)
            offset += 1
        total += density * component.fraction * amplitude
    return kernel_spectrum(total, dt)


def tensor_raman_spectrum(
    components: tuple[RamanCoefficients, ...],
    rates: tuple[float, ...],
    densities: tuple[float, ...],
    dt: float,
) -> Complex[np.ndarray, "n_raman_kernels n_omega_doubled"]:
    """Form effective trace and traceless spectra from independently damped terms.

    Parameters
    ----------
    components : tuple of RamanCoefficients
        Tensor-capable components in mixture order, with per-row numeric weights.
    rates : tuple of float
        Decay rates in s^-1, flattened by component then kernel row.
    densities : tuple of float
        Amplitude densities in m^-3. Supply ``(1.0,)`` when a single component's
        density is multiplied outside the convolution.
    dt : float
        Fine-grid sample spacing in s.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_raman_kernels, n_to + 1)`` with dt included. One trace kernel
        for isotropic-only declarations; otherwise trace and traceless rows.

    Raises
    ------
    BackendError
        If counts disagree or any term lacks a vector contraction.
    """
    if (
        not components
        or len(densities) != len(components)
        or len(rates) != sum(c.n_terms for c in components)
        or not all(c.tensor_capable for c in components)
    ):
        raise BackendError("vector Raman requires matching tensor-capable components")
    n_kernels = 2 if any(np.any(c.tensor_weights[:, 1]) for c in components) else 1
    delays = np.arange(components[0].undamped.shape[1]) * dt
    total = np.zeros((n_kernels, delays.size))
    offset = 0
    for component, density in zip(components, densities, strict=True):
        amplitude = np.zeros_like(total)
        for row, weights in zip(
            component.undamped, component.tensor_weights, strict=True
        ):
            damped = row * np.exp(-rates[offset] * delays)
            for channel in range(n_kernels):
                amplitude[channel] += weights[channel] * damped
            offset += 1
        total += density * component.fraction * amplitude
    return np.fft.rfft(total, axis=-1) * dt


def raman_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    spectrum: Complex[np.ndarray, "... n_omega_doubled"],
    *,
    kind: GridKind,
    thg: bool,
    density: float,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Evaluate the delayed Raman polarisation at every node of a chunk.

    The scalar reduction preserves the formula of
    :meth:`solprop.nonlinear.raman.RamanResponse.evaluate` in its original order of
    operations:

    .. math::

        P(t) = \varepsilon_0\,\rho\,E(t) \int h(t - t')\,D(t')\,\mathrm{d}t' ,

    with ``D`` the real drive :func:`solprop.nonlinear.raman.raman_drive` chooses for
    the grid -- ``E^2`` on a field grid that keeps the harmonic, ``|E_a|^2/2`` on one
    that does not, ``|A|^2/2`` on an envelope grid.

    The convolution runs on a **doubled, zero-padded** grid, because a transform product
    is a circular convolution and a rotational coherence outlives the window by orders
    of magnitude: hydrogen's lives about 2.9 ns at one bar against a window of tens of
    picoseconds, so the wrapped tail would be almost the whole response. The drive and
    the kernel occupy the first half of an array of twice the length, which makes the
    product the linear convolution exactly (``docs/guide/media/raman-numerics.md``, "The
    doubled grid").

    For two Cartesian components, convolve the trace, difference and twice the
    off-diagonal entry of the real quadratic tensor, then apply the resulting
    symmetric matrix to the field. Isotropic-only declarations need just the trace
    history. Each history is real on both grids and uses one real transform pair.
    ``numpy.fft`` preserves the scalar oracle's transform arithmetic.

    Parameters
    ----------
    field_t : numpy.ndarray
        Field at the chunk's nodes, shape ``(n_nodes_chunk, n_pol, n_to)``, in V/m:
        real on a field grid, complex on an envelope grid. One or two physical
        Cartesian components.
    spectrum : numpy.ndarray
        Damped complex128 kernels with ``dt`` included. Scalar fields use shape
        ``(n_to + 1,)`` from :func:`raman_spectrum`; vector fields use shape
        ``(n_raman_kernels, n_to + 1)`` from :func:`tensor_raman_spectrum`, with one
        trace kernel or the trace/traceless pair.
    kind : {'field', 'envelope'}
        The spectral layout the samples belong to.
    thg : bool
        Whether a field grid's drive keeps the ``2 omega`` beat, and so radiates a third
        harmonic through the later multiplication by the field. Ignored on an envelope
        grid, which cannot carry it in a quadratic drive.
    density : float
        The density path at this position, ``responses.density(z)``, in m^-3 for a gas
        and 1.0 for a medium whose kernel is already absolute. Non-negative.

    Returns
    -------
    numpy.ndarray
        Raman polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on a
        field grid and complex on an envelope grid.

    Raises
    ------
    BackendError
        If the component count or effective kernel shape is incompatible with the
        doubled temporal grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.responses import raman_polarisation
    >>> field = np.ones((1, 1, 4))
    >>> flat = np.zeros(5, dtype=complex)
    >>> polarisation = raman_polarisation(
    ...     field, flat, kind="field", thg=True, density=1.0
    ... )
    >>> bool(np.all(polarisation == 0.0))
    True
    """
    if field_t.shape[-2] == 2:
        return _vector_raman_polarisation(
            field_t, spectrum, kind=kind, thg=thg, density=density
        )
    if field_t.shape[-2] != 1:
        raise BackendError("Raman needs one or two physical components")
    n_to = field_t.shape[-1]
    if spectrum.shape != (n_to + 1,):
        msg = (
            f"the Raman kernel spectrum has shape {spectrum.shape}, expected "
            f"{(n_to + 1,)}: the one-sided spectrum of the doubled grid"
        )
        raise BackendError(msg)
    drive = np.zeros((field_t.shape[0], 2 * n_to))
    # One column at a time, as the oracle takes them: the drive of a field grid without
    # the harmonic is an analytic signal, and forming it per column is what makes this
    # pass reproduce the oracle rather than merely agree with it.
    for node in range(field_t.shape[0]):
        drive[node, :n_to] = raman_drive(field_t[node], kind=kind, thg=thg)
    convolved = np.fft.irfft(
        spectrum * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    return (EPS0 * density) * field_t * convolved[:, np.newaxis, :]


def _vector_raman_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk 2 n_to"],
    spectrum: Complex[np.ndarray, "... n_omega_doubled"],
    *,
    kind: GridKind,
    thg: bool,
    density: float,
) -> Num[np.ndarray, "n_nodes_chunk 2 n_to"]:
    """Convolve trace, difference and coherence drives with compiled kernels.

    - ``field_t``: Cartesian field in V/m, shape ``(n_nodes_chunk, 2, n_to)``.
    - ``thg``: keep the real-field harmonic drive; ignored for envelopes.
    - ``density``: outside amplitude density in m^-3, or one for a density-weighted
      mixture.
    - Returns: Cartesian polarisation in C/m^2 with the field's shape.
    """
    n_to = field_t.shape[-1]
    if spectrum.ndim != 2 or spectrum.shape not in ((1, n_to + 1), (2, n_to + 1)):
        raise BackendError("vector Raman needs one or two tensor kernel spectra")
    values = field_t
    if kind == "field" and not thg:
        values = np.array(
            [
                [fourier.analytic_signal(np.real(row)) for row in node]
                for node in field_t
            ]
        )
    x, y = values[:, 0], values[:, 1]
    half = 0.5 if kind == "envelope" or not thg else 1.0
    xx = half * (np.real(x) ** 2 + np.imag(x) ** 2)
    yy = half * (np.real(y) ** 2 + np.imag(y) ** 2)
    n_drives = 1 if spectrum.shape[0] == 1 else 3
    drive = np.zeros((field_t.shape[0], n_drives, 2 * n_to))
    drive[:, 0, :n_to] = xx + yy
    if n_drives == 3:
        drive[:, 1, :n_to] = xx - yy
        drive[:, 2, :n_to] = (
            2 * half * (np.real(x) * np.real(y) + np.imag(x) * np.imag(y))
        )
    kernels = spectrum if n_drives == 1 else spectrum[[0, 1, 1]]
    convolved = np.fft.irfft(
        kernels * np.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    if n_drives == 1:
        return (EPS0 * density) * field_t * convolved
    u, v, w = convolved[:, 0], convolved[:, 1], convolved[:, 2]
    result = np.empty_like(field_t)
    result[:, 0] = (u + v) * field_t[:, 0] + w * field_t[:, 1]
    result[:, 1] = w * field_t[:, 0] + (u - v) * field_t[:, 1]
    return (EPS0 * density) * result


def nonlinear_polarisation(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    responses: CompiledResponses,
    t_window_fine: Float[np.ndarray, " n_to_window"],
    *,
    kind: GridKind,
    density: float,
    preionisation: float,
    dt: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    raman_spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None = None,
    scalars: StageScalars | None = None,
) -> Num[np.ndarray, "n_nodes_chunk n_pol n_to"]:
    r"""Return the total physical polarisation of a chunk of nodes, windowed.

    :func:`nonlinear_sources`' collared local polarisation plus the plasma current
    integrated along t by the problem's running-integral rule and collared: the
    time-domain form the oracles of
    :meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate` compare against. A
    stage never takes it: the stage assembles the current in frequency instead
    (:func:`~solprop.backends.numpy_ref.passes.plasma.held_plasma_spectrum`, SEP 27).

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes from
        :func:`~solprop.backends.numpy_ref.passes.transforms.synthesise`, shape
        ``(n_nodes_chunk, n_pol, n_to)``, in V/m.
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the kernel contract: the order, the coefficients and the tables.
    t_window_fine : numpy.ndarray
        The temporal collar profile in ``[0, 1]``, float64, shape ``(n_to,)``.
    kind : {'field', 'envelope'}
        The spectral layout, which the Raman drive depends on.
    density : float
        The density path at this position, ``responses.density(z)``, non-negative.
    preionisation : float
        The pre-ionised fraction at this position, in ``[0, 1]``. Ignored when no
        plasma is compiled.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, for an envelope-grid harmonic
        term.
    raman_spectrum : numpy.ndarray or None, optional
        The damped Raman kernel spectrum at this position from :func:`raman_spectrum`,
        shape ``(n_to + 1,)``. Required when the order names ``raman_drive``; the caller
        forms it once per stage rather than once per chunk, because it does not depend
        on the nodes.
    scalars : StageScalars or None, optional
        Labelled component densities and preionisations at the stage. Required for
        multiple plasma channels; None retains the standalone single-response call.

    Returns
    -------
    numpy.ndarray
        The total polarisation in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``, real on
        a field grid and complex on an envelope grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.numpy_ref.passes.responses import nonlinear_polarisation
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.nonlinear.responses import KerrResponse
    >>> from solprop.profiles import Constant
    >>> from solprop.translate.responses import compile_responses
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 400e-15)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0], omega_ref=grid.omega0, aeff=1e-10
    ... )
    >>> compiled = compile_responses(
    ...     [KerrResponse(chi3=1.0)],
    ...     grid,
    ...     mode,
    ...     density=Constant(1.0),
    ...     z_max=1.0,
    ...     prefactor_index=1.0,
    ... )
    >>> field = np.full((1, 1, 4), 2.0)
    >>> out = nonlinear_polarisation(
    ...     field, compiled, np.ones(4), kind="field", density=1.0, preionisation=0.0,
    ...     dt=grid.dt_fine,
    ... )
    >>> bool(np.allclose(out, compiled.kerr_coefficients.coefficient * 8.0))
    True
    """
    sources = nonlinear_sources(
        field_t,
        responses,
        t_window_fine,
        kind=kind,
        density=density,
        preionisation=preionisation,
        dt=dt,
        carrier=carrier,
        raman_spectrum=raman_spectrum,
        scalars=scalars,
    )
    if sources.current is None:
        return sources.polarisation
    return sources.polarisation + t_window_fine * responses.plasma_integral.integrate(
        sources.current, dt
    )


@dataclass(frozen=True)
class NonlinearSources:
    """Carry the separate local polarisation and field-plasma current.

    Attributes
    ----------
    polarisation : numpy.ndarray
        Collared Kerr and Raman polarisation, including the separately collared
        dispersive Kerr, in C/m^2, shape ``(n_nodes_chunk, n_pol, n_to)``.
    current : numpy.ndarray or None
        Total ungated plasma current density in A/m^2 with the same shape; ``None``
        when no field plasma is compiled. Every channel and both currents enter once.
    """

    polarisation: Num[np.ndarray, "n_nodes_chunk n_pol n_to"]
    current: Float[np.ndarray, "n_nodes_chunk n_pol n_to"] | None


def nonlinear_sources(
    field_t: Num[np.ndarray, "n_nodes_chunk n_pol n_to"],
    responses: CompiledResponses,
    t_window_fine: Float[np.ndarray, " n_to_window"],
    *,
    kind: GridKind,
    density: float,
    preionisation: float,
    dt: float,
    carrier: Complex[np.ndarray, " n_to_carrier"] | None = None,
    raman_spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None = None,
    scalars: StageScalars | None = None,
) -> NonlinearSources:
    r"""Separate the collared local polarisation and the ungated plasma current.

    Pass 4 of ``architecture.md`` section 8: the traversal ``responses.response_order``
    names, term by term, accumulating ``P(t)`` and applying the temporal collar at the
    end -- but for SEP 6's dispersive response, whose collar acts on its products
    before its output multipliers (:func:`dispersive_kerr_polarisation`) and which is
    added after. Each stage is one of the named functions of this module, so the order
    the translation layer compiled is the order that runs, and a term this backend
    cannot execute is refused at construction rather than dropped here.

    Parameters
    ----------
    field_t : numpy.ndarray
        The field at the chunk's nodes from
        :func:`~solprop.backends.numpy_ref.passes.transforms.synthesise`, shape
        ``(n_nodes_chunk, n_pol, n_to)``, in V/m.
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the kernel contract: the order, the coefficients and the tables.
    t_window_fine : numpy.ndarray
        The temporal collar profile in ``[0, 1]``, float64, shape ``(n_to,)``.
    kind : {'field', 'envelope'}
        The spectral layout, which the Raman drive depends on.
    density : float
        The density path at this position, ``responses.density(z)``, non-negative.
    preionisation : float
        The pre-ionised fraction at this position, in ``[0, 1]``. Ignored when no
        plasma is compiled.
    dt : float
        Sample spacing of the fine time grid, s, positive.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega_0 t)`` on the fine time grid, for an envelope-grid harmonic
        term.
    raman_spectrum : numpy.ndarray or None, optional
        The damped Raman kernel spectrum at this position from :func:`raman_spectrum`,
        shape ``(n_to + 1,)``. Required when the order names ``raman_drive``; the caller
        forms it once per stage rather than once per chunk, because it does not depend
        on the nodes.
    scalars : StageScalars or None, optional
        Labelled component densities and preionisations at the stage. Required for
        multiple plasma channels; None retains the standalone single-response call.

    Returns
    -------
    NonlinearSources
        The collared non-plasma polarisation and total ungated current density.
    """
    total = np.zeros_like(field_t)
    current = None
    dispersive = None
    multiple = len(responses.plasma_channels) > 1
    if multiple and scalars is None:
        raise BackendError("multiple plasma channels require labelled stage scalars")
    plasma_density = (
        density
        if scalars is None or not scalars.plasma_densities
        else scalars.plasma_densities[0]
    )
    kerr_density = density if scalars is None else scalars.kerr_density
    raman_density = (
        1.0
        if len(responses.raman_terms) > 1
        else density
        if scalars is None or not scalars.raman_densities
        else scalars.raman_densities[0]
    )
    rate = None
    scan = None
    for stage in responses.response_order:
        if stage is ResponseStage.IONISATION_RATE and not multiple:
            rate = ionisation_rate(field_t, _plasma_rate(responses))
        elif stage is ResponseStage.PLASMA_SCAN and not multiple:
            scan = plasma_scan(
                field_t,
                _checked(rate),
                _plasma(responses),
                preionisation=preionisation,
                dt=dt,
                integral=responses.plasma_integral,
            )
        elif stage is ResponseStage.PLASMA_CURRENT and multiple:
            assert scalars is not None
            current = plasma_channels_current(
                field_t,
                responses.plasma_channels,
                scalars.plasma_densities,
                scalars.plasma_preionisation,
                dt,
                responses.plasma_integral,
            )
        elif stage is ResponseStage.PLASMA_CURRENT:
            fraction, free_current = _checked(scan)
            current = plasma_density * plasma_total_current(
                field_t,
                _checked(rate),
                fraction,
                free_current,
                _plasma(responses),
            )
        elif stage is ResponseStage.KERR:
            total = total + kerr_polarisation(
                field_t, _kerr(responses), kerr_density, carrier
            )
        elif stage is ResponseStage.DISPERSIVE_KERR:
            if scalars is None:
                raise BackendError(
                    "the dispersive Kerr stage reads each channel's density from the "
                    "stage scalars"
                )
            # The collar is inside: it acts on the products, before G.
            dispersive = dispersive_kerr_polarisation(
                field_t,
                _checked(responses.dispersive_kerr),
                scalars.dispersive_kerr_densities,
                kind=kind,
                carrier=carrier,
                window=t_window_fine,
            )
        elif stage is ResponseStage.RAMAN_DRIVE:
            total = total + raman_polarisation(
                field_t,
                _raman(raman_spectrum),
                kind=kind,
                thg=_raman_thg(responses),
                density=raman_density,
            )
    windowed = total * t_window_fine
    return NonlinearSources(
        windowed if dispersive is None else windowed + dispersive, current
    )


def _kerr(responses: CompiledResponses) -> KerrCoefficients:
    """Return the compiled Kerr coefficients, which the Kerr stage implies."""
    assert responses.kerr_coefficients is not None
    return responses.kerr_coefficients


def _plasma(responses: CompiledResponses) -> PlasmaCoefficients:
    """Return the compiled plasma group, which a plasma stage implies."""
    plasma = responses.plasma_coefficients
    assert plasma is not None
    return plasma


def _raman_thg(responses: CompiledResponses) -> bool:
    """Return whether a field grid's Raman drive keeps the ``2 omega`` beat.

    The drive form is a property of the problem, carried on the Raman leaf; the
    translation layer has already checked that the declared Kerr and Raman responses
    agree about it. An envelope grid's drive is ``|A|^2/2`` whatever it says, which is
    why the value is not checked there.

    - Returns: True when the drive is ``E^2`` rather than ``|E_a|^2/2``.
    """
    return responses.raman_terms[0].thg


def _raman(
    spectrum: Complex[np.ndarray, "... n_omega_doubled"] | None,
) -> Complex[np.ndarray, "... n_omega_doubled"]:
    """Return the Raman kernel spectrum a declared Raman stage promised."""
    if spectrum is None:
        msg = (
            "response_order names the Raman stage but no kernel spectrum was given; "
            "form it once per stage with raman_spectrum and pass it in"
        )
        raise BackendError(msg)
    return spectrum


def _plasma_rate(responses: CompiledResponses) -> RateTable:
    """Return the rate table the ionisation stage reads."""
    return responses.ionisation_table[0]


def _checked[T](value: T | None) -> T:
    """Return a stage result the derived order guarantees is present."""
    assert value is not None
    return value
