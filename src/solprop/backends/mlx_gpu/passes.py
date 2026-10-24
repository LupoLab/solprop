# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The passes of one stage on the device, in binary32: the oracle's, in MLX.

Each function follows the single-precision oracle's pass of the same name
(:mod:`solprop.backends.numpy_ref.single` and :mod:`solprop.backends.numpy_ref.passes`)
and is compared with it, class R, by ``tests/backends/test_mlx.py``. The running
integrals use the oracle's doubling order exactly; the transforms are MLX's FFT, the
matrix products MLX's float32 GEMM (the backend has probed it), and the remaining
element-wise work MLX's own operations. Every array is float32 or complex64.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from solprop.backends.mlx_gpu.arithmetic import population
from solprop.fourier import GridKind
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.translate.ionisation import RateTable
from solprop.translate.responses import KerrCoefficients

__all__ = [
    "DeviceRateTable",
    "analytic",
    "kerr",
    "plasma_current",
    "project",
    "raman",
    "running_integral",
    "synthesise",
    "to_frequency",
    "to_time",
]


def _scatter(values: mx.array, index: mx.array, width: int) -> mx.array:
    """Place the columns of `values` at `index` of a zero row of `width`, in complex64.

    The GPU's scatter takes real arrays only, so the two parts are placed apart.

    Parameters
    ----------
    values : mlx.core.array
        Complex64, ``(rows, n_index)``.
    index : mlx.core.array
        Integer positions, ``(n_index,)``.
    width : int
        The output's last dimension.

    Returns
    -------
    mlx.core.array
        Complex64, ``(rows, width)``.
    """
    rows = values.shape[0]
    real = mx.zeros((rows, width), dtype=mx.float32)
    imag = mx.zeros((rows, width), dtype=mx.float32)
    real[:, index] = mx.real(values)
    imag[:, index] = mx.imag(values)
    return real + 1j * imag


def to_time(
    spectrum: mx.array, crop: mx.array, n_to: int, *, kind: GridKind
) -> mx.array:
    """Pad onto the fine grid and transform every modal column to time.

    Parameters
    ----------
    spectrum : mlx.core.array
        ``(n_modes, n_pol, n_omega)`` complex64.
    crop : mlx.core.array
        Where each coarse bin sits in the fine spectrum.
    n_to : int
        Fine time samples.
    kind : {'field', 'envelope'}
        The spectral layout.

    Returns
    -------
    mlx.core.array
        ``(n_modes, n_pol, n_to)``, float32 on a field grid, complex64 otherwise.
    """
    n_modes, n_pol = spectrum.shape[:2]
    width = n_to // 2 + 1 if kind == "field" else n_to
    fine = _scatter(spectrum.reshape(n_modes * n_pol, -1), crop, width)
    if kind == "field":
        time = mx.fft.irfft(fine, n=n_to, axis=-1)
    else:
        time = mx.fft.ifft(fine, axis=-1)
    return time.reshape(n_modes, n_pol, n_to)


def to_frequency(values: mx.array, crop: mx.array, *, kind: GridKind) -> mx.array:
    """Transform to frequency and gather the coarse bins.

    Parameters
    ----------
    values : mlx.core.array
        ``(n_modes, n_pol, n_to)``.
    crop : mlx.core.array
        The coarse bins' positions.
    kind : {'field', 'envelope'}
        The spectral layout.

    Returns
    -------
    mlx.core.array
        ``(n_modes, n_pol, n_omega)`` complex64.
    """
    n_modes, n_pol = values.shape[:2]
    flat = values.reshape(n_modes * n_pol, -1)
    spectrum = (
        mx.fft.rfft(flat, axis=-1) if kind == "field" else mx.fft.fft(flat, axis=-1)
    )
    return mx.take(spectrum, crop, axis=-1).reshape(n_modes, n_pol, -1)


def _product(matrix: mx.array, values: mx.array) -> mx.array:
    """Multiply a real float32 matrix into the leading axis of values.

    Complex values are multiplied part by part, so that every product is a real
    float32 GEMM of the dispatch family the backend probed.

    Parameters
    ----------
    matrix : mlx.core.array
        ``(rows, k)`` float32.
    values : mlx.core.array
        ``(k, ...)``.

    Returns
    -------
    mlx.core.array
        ``(rows, ...)``.
    """
    k = values.shape[0]
    flat = values.reshape(k, -1)
    if values.dtype == mx.complex64:
        result = (matrix @ mx.real(flat)) + 1j * (matrix @ mx.imag(flat))
    else:
        result = matrix @ flat
    return result.reshape(matrix.shape[0], *values.shape[1:])


def synthesise(modal_t: mx.array, rows: mx.array, taper: float, n_pol: int) -> mx.array:
    """Synthesise the field on a chunk of nodes, ``tau S a``.

    Parameters
    ----------
    modal_t : mlx.core.array
        ``(n_modes, n_pol_state, n_to)``.
    rows : mlx.core.array
        The chunk's rows of the narrowed synthesis matrix.
    taper : float
        The taper scale at this position.
    n_pol : int
        Components the transform resolves at each node.

    Returns
    -------
    mlx.core.array
        ``(n_nodes_chunk, n_pol * n_pol_state, n_to)``.
    """
    nodes = _product(rows * taper, modal_t)
    n_rows, n_pol_state, n_to = nodes.shape
    return nodes.reshape(n_rows // n_pol, n_pol * n_pol_state, n_to)


def project(values: mx.array, columns: mx.array, taper: float, n_pol: int) -> mx.array:
    """Project node values back onto the modes, ``P / tau``.

    Parameters
    ----------
    values : mlx.core.array
        ``(n_nodes_chunk, n_pol_node, n_to)``.
    columns : mlx.core.array
        The chunk's columns of the narrowed projection matrix.
    taper : float
        The taper scale.
    n_pol : int
        Components the transform resolves at each node.

    Returns
    -------
    mlx.core.array
        ``(n_modes, n_pol_state, n_to)``.
    """
    n_chunk, n_pol_node, n_to = values.shape
    rows = values.reshape(n_chunk * n_pol, n_pol_node // n_pol, n_to)
    return _product(columns * (1.0 / taper), rows)


def analytic(values: mx.array) -> mx.array:
    """Return the analytic signal of a real array along its last axis, in complex64.

    Parameters
    ----------
    values : mlx.core.array
        Float32 values.

    Returns
    -------
    mlx.core.array
        Complex64.
    """
    n = values.shape[-1]
    weights = np.zeros(n, dtype=np.float32)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[n // 2] = 1.0
        weights[1 : n // 2] = 2.0
    else:
        weights[1 : (n + 1) // 2] = 2.0
    return mx.fft.ifft(mx.fft.fft(values, axis=-1) * mx.array(weights), axis=-1)


def kerr(
    field: mx.array,
    coefficients: KerrCoefficients,
    factor: float,
    carrier: mx.array | None,
) -> mx.array:
    """Evaluate the Kerr polarisation, the oracle's ``kerr_single``.

    Parameters
    ----------
    field : mlx.core.array
        The scaled field at the nodes.
    coefficients : solprop.translate.responses.KerrCoefficients
        The compiled coefficients.
    factor : float
        ``coefficient * density * s**2``, rounded once.
    carrier : mlx.core.array or None
        The harmonic carrier on an envelope grid that radiates it.

    Returns
    -------
    mlx.core.array
        The polarisation divided by ``s``.
    """
    values = analytic(field) if coefficients.analytic_signal else field
    if coefficients.cubic_form == "real":
        polarisation = factor * mx.sum(values * values, axis=-2, keepdims=True) * values
    else:
        squares = mx.sum(values * values, axis=-2, keepdims=True)
        re, im = mx.real(values), mx.imag(values)
        moduli = mx.sum(re * re + im * im, axis=-2, keepdims=True)
        polarisation = factor * (
            coefficients.self_factor * moduli * values
            + coefficients.conjugate_factor * squares * mx.conj(values)
        )
        if carrier is not None:
            polarisation = polarisation + (
                (factor * coefficients.harmonic_factor) * carrier * squares * values
            )
    return mx.real(polarisation) if coefficients.take_real else polarisation


def _drive(column: mx.array, *, kind: GridKind, thg: bool) -> mx.array:
    """Return one component's Raman drive along the last axis.

    Parameters
    ----------
    column : mlx.core.array
        ``(n_nodes, n_to)``.
    kind : {'field', 'envelope'}
        The spectral layout.
    thg : bool
        Whether the drive keeps the harmonic.

    Returns
    -------
    mlx.core.array
        Float32 drive.
    """
    if kind == "envelope":
        re, im = mx.real(column), mx.imag(column)
        return 0.5 * (re * re + im * im)
    if thg:
        return column * column
    signal = analytic(column)
    re, im = mx.real(signal), mx.imag(signal)
    return 0.5 * (re * re + im * im)


def raman(
    field: mx.array,
    spectrum: mx.array,
    factor: float,
    *,
    kind: GridKind,
    thg: bool,
) -> mx.array:
    """Evaluate the delayed Raman polarisation, the oracle's ``raman_single``.

    Parameters
    ----------
    field : mlx.core.array
        The scaled field, ``(n_nodes, n_pol, n_to)``.
    spectrum : mlx.core.array
        The narrowed kernel spectrum: ``(n_to + 1,)`` scalar, ``(1 | 2, n_to + 1)``
        tensor.
    factor : float
        The combined coefficient, rounded once.
    kind : {'field', 'envelope'}
        The spectral layout.
    thg : bool
        Whether the drive keeps the harmonic.

    Returns
    -------
    mlx.core.array
        The polarisation divided by ``s``.
    """
    n_nodes, n_pol, n_to = field.shape
    pad = mx.zeros((n_nodes, n_to), dtype=mx.float32)
    if n_pol == 1:
        drive = mx.concatenate([_drive(field[:, 0], kind=kind, thg=thg), pad], axis=-1)
        convolved = mx.fft.irfft(
            spectrum * mx.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
        )[..., :n_to]
        return factor * field * convolved[:, None, :]
    values = field
    if kind == "field" and not thg:
        values = analytic(field)
    x, y = values[:, 0], values[:, 1]
    half = 0.5 if kind == "envelope" or not thg else 1.0
    xr, xi, yr, yi = mx.real(x), mx.imag(x), mx.real(y), mx.imag(y)
    xx = half * (xr * xr + xi * xi)
    yy = half * (yr * yr + yi * yi)
    drives = [xx + yy]
    if spectrum.shape[0] == 2:
        drives += [xx - yy, 2 * half * (xr * yr + xi * yi)]
    drive = mx.concatenate(
        [mx.stack(drives, axis=1), mx.stack([pad] * len(drives), axis=1)], axis=-1
    )
    kernels = (
        spectrum
        if len(drives) == 1
        else mx.stack([spectrum[0], spectrum[1], spectrum[1]])
    )
    convolved = mx.fft.irfft(
        kernels * mx.fft.rfft(drive, axis=-1), n=2 * n_to, axis=-1
    )[..., :n_to]
    if len(drives) == 1:
        return factor * field * convolved
    u, v, w = convolved[:, 0], convolved[:, 1], convolved[:, 2]
    first = (u + v) * field[:, 0] + w * field[:, 1]
    second = w * field[:, 0] + (u - v) * field[:, 1]
    return factor * mx.stack([first, second], axis=1)


def _prefix(values: mx.array) -> mx.array:
    """Return the doubling scan along the last axis, the oracle's order exactly.

    Parameters
    ----------
    values : mlx.core.array
        The increments.

    Returns
    -------
    mlx.core.array
        The running sums.
    """
    scan = values
    n = values.shape[-1]
    widths = [(0, 0)] * (values.ndim - 1)
    shift = 1
    while shift < n:
        # One addition of the level's shifted copy: adding zero to the first `shift`
        # entries is exact, so this is the oracle's in-place update exactly, and each
        # level is one array that the next can release.
        scan = scan + mx.pad(scan[..., :-shift], [*widths, (shift, 0)])
        shift *= 2
    return scan


def running_integral(values: mx.array, dt: float, rule: RunningIntegral) -> mx.array:
    """Return a running integral, the oracle's ``running_integral``.

    Parameters
    ----------
    values : mlx.core.array
        The integrand along the last axis, float32.
    dt : float
        The sample spacing, s.
    rule : solprop.nonlinear.ionisation.RunningIntegral
        The problem's rule.

    Returns
    -------
    mlx.core.array
        The running integral, zero at the first sample.
    """
    filtered = values
    half = rule.half_width
    if half:
        n = values.shape[-1]
        reach = np.arange(half, 0, -1)
        head = 2 * values[..., :1] - values[..., mx.array(reach)]
        tail = 2 * values[..., -1:] - values[..., mx.array(n - 1 - reach[::-1])]
        extended = mx.concatenate([head, values, tail], axis=-1)
        taps = rule.taps.astype(np.float32)
        filtered = float(taps[0]) * values
        for k in range(1, half + 1):
            filtered = filtered + float(taps[k]) * (
                extended[..., half - k : half - k + n]
                + extended[..., half + k : half + k + n]
            )
    increments = (filtered[..., :-1] + filtered[..., 1:]) * float(np.float32(0.5 * dt))
    zero = mx.zeros((*values.shape[:-1], 1), dtype=mx.float32)
    result = mx.concatenate([zero, _prefix(increments)], axis=-1)
    # Evaluated here: a scan's levels would otherwise stay live behind the rest of the
    # stage's graph, about log2(n) arrays of the node chunk each.
    mx.eval(result)
    return result


def _split(a: mx.array) -> tuple[mx.array, mx.array]:
    """Split binary32 values into halves of 12 bits (Veltkamp).

    Parameters
    ----------
    a : mlx.core.array
        Float32 values, well inside binary32's range.

    Returns
    -------
    tuple of mlx.core.array
        ``hi + lo == a`` exactly.
    """
    c = 4097.0 * a
    hi = c - (c - a)
    return hi, a - hi


def _two_product(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """Return ``a b`` and its rounding error exactly (Dekker), without a fused add.

    Parameters
    ----------
    a, b : mlx.core.array
        Float32 values.

    Returns
    -------
    tuple of mlx.core.array
        ``p = fl(a b)`` and ``e`` with ``p + e == a b``.
    """
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@dataclass(frozen=True)
class DeviceRateTable:
    """An ionisation rate table on the device.

    The spline of :class:`solprop.translate.ionisation.RateTable` in ``ln E``, with its
    local coordinate formed as ``log1p(E / E_i - 1)`` against the narrowed left edge
    ``E_i`` of its interval: a binary32 ``ln E`` would carry an error of ``|ln E| u``,
    which the rate's steep slope multiplies. The polynomial's constant term, ``ln W`` at
    the interval's edge (of order 70), is held as a double-float pair and added to the
    small remainder by an exact two-sum, and the rate is ``exp(hi) (1 + lo)``: a
    binary32 constant term would cost ``|ln W| u`` relative in the rate.

    Attributes
    ----------
    coefficients : mlx.core.array
        ``(4, n_intervals)`` float32; the last row is the constant term's head.
    constant_tail : mlx.core.array
        The constant term's tail, float32.
    inverse_edges : mlx.core.array
        ``1 / E_i`` per interval, float32.
    inverse_tail : mlx.core.array
        The tail of ``1 / E_i`` as a double-float pair, float32.
    log_field_min : float
        ``ln E_0``.
    d_log_field : float
        The interval width in ``ln E``.
    field_min : float
        Below it the rate is zero, V/m.
    field_max : float
        Above it the rate holds ``rate_max``, V/m.
    rate_max : float
        s^-1.
    """

    coefficients: mx.array
    constant_tail: mx.array
    inverse_edges: mx.array
    inverse_tail: mx.array
    log_field_min: float
    d_log_field: float
    field_min: float
    field_max: float
    rate_max: float

    @classmethod
    def upload(cls, table: RateTable) -> "DeviceRateTable":  # noqa: UP037 - the runtime type hook resolves this while the class body runs
        """Narrow a rate table onto the device.

        Parameters
        ----------
        table : solprop.translate.ionisation.RateTable
            The compiled table.

        Returns
        -------
        DeviceRateTable
            The device table.
        """
        n = table.coefficients.shape[1]
        edges = np.exp(table.log_field_min + np.arange(n) * table.d_log_field)
        head = table.coefficients.astype(np.float32)
        tail = (table.coefficients[3] - head[3].astype(np.float64)).astype(np.float32)
        return cls(
            mx.array(head),
            mx.array(tail),
            mx.array((1.0 / edges).astype(np.float32)),
            mx.array(
                (1.0 / edges - (1.0 / edges).astype(np.float32)).astype(np.float32)
            ),
            float(table.log_field_min),
            float(table.d_log_field),
            float(table.field_min),
            float(table.field_max),
            float(table.rate_max),
        )

    def __call__(self, magnitude: mx.array) -> mx.array:
        """Return the rate at field magnitudes, float32.

        Parameters
        ----------
        magnitude : mlx.core.array
            ``|E|``, V/m, float32.

        Returns
        -------
        mlx.core.array
            The rate, s^-1.
        """
        n = self.coefficients.shape[1]
        safe = mx.maximum(magnitude, float(np.float32(self.field_min)))
        index = mx.clip(
            ((mx.log(safe) - self.log_field_min) * (1.0 / self.d_log_field)).astype(
                mx.int32
            ),
            0,
            n - 1,
        )
        # E / E_i as a double-float product, so that the local coordinate carries
        # binary32's rounding relative to itself, not relative to one.
        head, tail = _two_product(safe, mx.take(self.inverse_edges, index))
        tail = tail + safe * mx.take(self.inverse_tail, index)
        local = mx.log1p((head - 1.0) + tail)
        c = [mx.take(self.coefficients[k], index) for k in range(4)]
        remainder = ((c[0] * local + c[1]) * local + c[2]) * local
        head = c[3] + remainder
        shifted = head - c[3]
        error = (c[3] - (head - shifted)) + (remainder - shifted)
        tail = error + mx.take(self.constant_tail, index)
        rate = mx.exp(head) * (1.0 + tail)
        ceiling = float(np.float32(self.rate_max))
        rate = mx.where(magnitude > self.field_max, ceiling, rate)
        return mx.where(magnitude >= self.field_min, rate, 0.0)


def plasma_current(
    field: mx.array,
    tables: tuple[DeviceRateTable, ...],
    constants: tuple[tuple[float, float], ...],
    initials: tuple[float, ...],
    dt: float,
    rule: RunningIntegral,
) -> mx.array:
    """Return the field-grid plasma's ungated current, the oracle's.

    Parameters
    ----------
    field : mlx.core.array
        The physical field ``s E``, float32, ``(n_nodes, n_pol, n_to)``.
    tables : tuple of DeviceRateTable
        One per channel.
    constants : tuple of tuple of float
        Per channel, ``density (e^2/m) n_c`` and ``density I_p``.
    initials : tuple of float
        Per channel, the pre-ionised fraction.
    dt : float
        The sample spacing, s.
    rule : solprop.nonlinear.ionisation.RunningIntegral
        The running-integral rule.

    Returns
    -------
    mlx.core.array
        The current density, A/m^2.
    """
    if field.shape[-2] == 1:
        magnitude = mx.abs(field[..., 0, :])
    else:
        magnitude = mx.sqrt(field[..., 0, :] ** 2 + field[..., 1, :] ** 2)
    weight = mx.zeros(magnitude.shape, dtype=mx.float32)
    work = mx.zeros(magnitude.shape, dtype=mx.float32)
    for table, (drude, potential), initial in zip(
        tables, constants, initials, strict=True
    ):
        rate = table(magnitude)
        exposure = running_integral(rate, dt, rule)
        if rule.policy != "trapezoid":
            exposure = mx.maximum(exposure, 0.0)
        fraction, survival = population(exposure, initial)
        weight = weight + float(np.float32(drude)) * fraction
        work = work + float(np.float32(potential)) * (rate * survival)
    free = running_integral(weight[..., None, :] * field, dt, rule)
    squared = magnitude * magnitude
    guard = mx.where(
        (work > 0) & (squared > 0), work / mx.where(squared > 0, squared, 1.0), 0.0
    )
    return free + guard[..., None, :] * field
