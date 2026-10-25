# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The envelope-grid plasma on the Metal device (SEP 33 section 6.3, phase 4).

The oracle (:mod:`solprop.backends.numpy_ref.envelope_plasma`) decomposes every
sample's two-component envelope by an SVD and refines its cycle moments with host
decisions. On the device each sample is one thread of one custom kernel:

- the polarisation ellipse in closed form, native's ``envelope_axes``: the scaled
  Gram matrix's eigenvalues by a scaled norm, the minor axis from the determinant, so
  that circular, nearly linear and degenerate (scalar) fields keep their limits;
- the cycle moments on the accepted rules of orders 64, 128, 256 and 512, split at the
  angles where the field crosses the table's cutoffs -- the panel past the low cut
  contributes nothing, the panel before the high cut the ceiling rate, and the field
  between them is held inside the table's range, so that binary32's rounding of a cut
  angle cannot put a node on the wrong side of a cutoff -- each level's sums
  compensated, and refined until two successive levels agree within the
  representation's frozen budget (``2e-6`` relative, floor ``1e-30`` s^-1);
- the rate from the table's spline as in :func:`~solprop.backends.mlx_gpu.arithmetic.
  rate_kernel` (an exact ``fma`` local coordinate, a double-float constant term);
- a failure flag per sample where the 512-point rule does not converge, which the
  backend reads with the step's norm and turns into the oracle's ``ConvergenceError``.

Its discretisation (one refinement per sample, binary32 nodes and arithmetic) differs
from the oracle's binary64 adaptive evaluation, so it is qualified as a discretisation,
class T, not as rounding.
"""

import math
from dataclasses import dataclass
from functools import cache

import mlx.core as mx
import numpy as np

from solprop.backends.mlx_gpu.passes import DeviceRateTable
from solprop.translate.envelope_plasma import CycleIntegralData

__all__ = ["DeviceCycle", "envelope_cycle"]

_LEVELS = 4
_MAX_ORDER = 512

_HEADER = """
inline float rate_at(
    float e,
    const device float* coefficients,
    const device float* constant_tail,
    const device float* inverse_edges,
    const device float* inverse_tail,
    const device float* constants,
    int intervals
) {
    float log_min = constants[0];
    float inverse_width = constants[1];
    float field_min = constants[2];
    float field_max = constants[3];
    if (e < field_min) { return 0.0f; }
    if (e > field_max) { return constants[4]; }
    int k = int((metal::precise::log(e) - log_min) * inverse_width);
    k = metal::clamp(k, 0, intervals - 1);
    float p = e * inverse_edges[k];
    float error = metal::fma(e, inverse_edges[k], -p) + e * inverse_tail[k];
    float x = (p - 1.0f) + error;
    float u = 1.0f + x;
    float local = u == 1.0f ? x : metal::precise::log(u) * x / (u - 1.0f);
    float c0 = coefficients[k];
    float c1 = coefficients[intervals + k];
    float c2 = coefficients[2 * intervals + k];
    float c3 = coefficients[3 * intervals + k];
    float remainder = ((c0 * local + c1) * local + c2) * local;
    float head = c3 + remainder;
    float shifted = head - c3;
    float tail = (c3 - (head - shifted)) + (remainder - shifted) + constant_tail[k];
    return metal::precise::exp(head) * (1.0f + tail);
}

inline float edge_angle(float relative, float ratio) {
    const float half_pi = 1.57079632679489662f;
    if (relative >= 1.0f) { return 0.0f; }
    if (relative <= ratio) { return half_pi; }
    return metal::precise::acos(metal::precise::sqrt(
        (relative * relative - ratio * ratio) / (1.0f - ratio * ratio)));
}

// Kahan's compensated addition of a term into (sum, compensation).
inline void kahan(thread float& sum, thread float& carry, float term) {
    float y = term - carry;
    float t = sum + y;
    carry = (t - sum) - y;
    sum = t;
}
"""

_CYCLE = """
    uint i = thread_position_in_grid.x;
    if (i >= N) { return; }
    uint node = i / NT;
    uint t = i % NT;
    float factor = constants[5];
    float xr = re[(node * NPOL) * NT + t] * factor;
    float xi = im[(node * NPOL) * NT + t] * factor;
    float yr = NPOL == 2 ? re[(node * NPOL + 1) * NT + t] * factor : 0.0f;
    float yi = NPOL == 2 ? im[(node * NPOL + 1) * NT + t] * factor : 0.0f;
    rate[i] = 0.0f;
    failed[i] = 0;
    for (uint p = 0; p < NPOL; ++p) {
        current_re[(node * NPOL + p) * NT + t] = 0.0f;
        current_im[(node * NPOL + p) * NT + t] = 0.0f;
    }
    float scale = metal::max(metal::max(metal::abs(xr), metal::abs(xi)),
                             metal::max(metal::abs(yr), metal::abs(yi)));
    if (scale == 0.0f) { return; }
    xr /= scale; xi /= scale; yr /= scale; yi /= scale;
    // native's envelope_axes: the scaled Gram matrix of the real and imaginary parts.
    float xx = xr * xr + yr * yr;
    float yy = xi * xi + yi * yi;
    float xy = xr * xi + yr * yi;
    float larger = metal::max(metal::abs(xx - yy), metal::abs(2.0f * xy));
    float smaller = metal::min(metal::abs(xx - yy), metal::abs(2.0f * xy));
    float norm_ratio = larger == 0.0f ? 0.0f : smaller / larger;
    float norm = larger * metal::precise::sqrt(1.0f + norm_ratio * norm_ratio);
    float major_square = 0.5f * (xx + yy + norm);
    float determinant = metal::fma(xr, yi, -xi * yr);
    float ratio = metal::min(1.0f, metal::abs(determinant) / major_square);
    float angle = 0.5f * metal::precise::atan2(2.0f * xy, xx - yy);
    float c = metal::precise::cos(angle);
    float s = metal::precise::sin(angle);
    float ux = xr * c + xi * s;
    float uy = yr * c + yi * s;
    float signed_ratio = determinant / major_square;
    float vx = -signed_ratio * uy;
    float vy = signed_ratio * ux;
    float major = scale * metal::precise::sqrt(major_square);
    int intervals = INTERVALS;
    float field_min = constants[2];
    float field_max = constants[3];
    if (major < field_min) { return; }
    float mc = 0.0f;
    float ms = 0.0f;
    if (ratio == 1.0f) {
        float w = 0.5f * rate_at(major, coefficients, constant_tail, inverse_edges,
                                 inverse_tail, constants, intervals);
        mc = w;
        ms = w;
    } else {
        const float half_pi = 1.57079632679489662f;
        const float inverse_pi = 0.318309886183790672f;
        float high_edge = edge_angle(field_max / major, ratio);
        float low_edge = edge_angle(field_min / major, ratio);
        float previous_c = 0.0f;
        float previous_s = 0.0f;
        bool converged = false;
        for (int level = 0; level < LEVELS; ++level) {
            int order = 64 << level;
            float sum_c = 0.0f, carry_c = 0.0f, sum_s = 0.0f, carry_s = 0.0f;
            // The split's own meaning: past the low cut the field is below the
            // table and the rate is zero; before the high cut it is above it and the
            // rate is the ceiling; between them the field is held inside the table's
            // range, so that binary32's rounding of the cut angle cannot put a node on
            // the wrong side of a cutoff and make the levels disagree.
            for (int panel = 0; panel < 2; ++panel) {
                float lower = panel == 0 ? 0.0f : high_edge;
                float upper = panel == 0 ? high_edge : low_edge;
                float width = upper - lower;
                if (width <= 0.0f) { continue; }
                for (int k = 0; k < order; ++k) {
                    float theta = lower + (nodes[level * MAXORDER + k] + 1.0f)
                        * (0.5f * width);
                    float co = metal::precise::cos(theta);
                    float si = metal::precise::sin(theta);
                    float c2 = co * co;
                    float s2 = si * si;
                    float denominator = metal::fma(ratio * ratio, s2, c2);
                    float e = metal::clamp(
                        major * metal::precise::sqrt(denominator), field_min,
                        field_max);
                    float w = (panel == 0 ? constants[4]
                               : rate_at(e, coefficients, constant_tail, inverse_edges,
                                         inverse_tail, constants, intervals))
                        / denominator;
                    float weight = (width * inverse_pi) * weights[level * MAXORDER + k];
                    kahan(sum_c, carry_c, weight * w * c2);
                    kahan(sum_s, carry_s, weight * w * s2);
                }
            }
            if (level > 0 && metal::isfinite(sum_c) && metal::isfinite(sum_s)
                && metal::abs(sum_c - previous_c) <= constants[6]
                    * metal::max(sum_c, constants[7])
                && metal::abs(sum_s - previous_s) <= constants[6]
                    * metal::max(sum_s, constants[7])) {
                mc = sum_c;
                ms = sum_s;
                converged = true;
                break;
            }
            previous_c = sum_c;
            previous_s = sum_s;
        }
        if (!converged) {
            failed[i] = 1;
            mc = previous_c;
            ms = previous_s;
        }
    }
    rate[i] = mc + ratio * ratio * ms;
    if (mc == 0.0f && ms == 0.0f) { return; }
    // native's envelope_loss per unit work: the moments divided by the scale first,
    // so that no standalone inverse of a tiny scale can overflow.
    float major_moment = mc / scale;
    float minor_moment = ms / scale;
    float f = 2.0f / major_square;
    float ar = major_moment * c;
    float ai = major_moment * s;
    float br = -minor_moment * s;
    float bi = minor_moment * c;
    current_re[(node * NPOL) * NT + t] = f * (ux * ar + vx * br);
    current_im[(node * NPOL) * NT + t] = f * (ux * ai + vx * bi);
    if (NPOL == 2) {
        current_re[(node * NPOL + 1) * NT + t] = f * (uy * ar + vy * br);
        current_im[(node * NPOL + 1) * NT + t] = f * (uy * ai + vy * bi);
    }
"""


@cache
def _cycle_kernel() -> object:
    """Return the compiled cycle-moment kernel.

    Returns
    -------
    object
        The ``mlx.core.fast.metal_kernel`` callable.
    """
    return mx.fast.metal_kernel(
        name="solprop_envelope_cycle",
        input_names=[
            "re",
            "im",
            "coefficients",
            "constant_tail",
            "inverse_edges",
            "inverse_tail",
            "constants",
            "nodes",
            "weights",
        ],
        output_names=["rate", "current_re", "current_im", "failed"],
        source=_CYCLE,
        header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


@dataclass(frozen=True)
class DeviceCycle:
    """A channel's cycle representation on the device.

    Attributes
    ----------
    table : DeviceRateTable
        The instantaneous rate table, narrowed.
    nodes : mlx.core.array
        The four rules' nodes on ``[-1, 1]``, ``(4, 512)`` float32, zero-padded.
    weights : mlx.core.array
        Their weights, likewise.
    moment_rtol : float
        The refinement budget.
    negligible_rate : float
        The comparison floor, s^-1.
    """

    table: DeviceRateTable
    nodes: mx.array
    weights: mx.array
    moment_rtol: float
    negligible_rate: float

    @classmethod
    def upload(cls, cycle: CycleIntegralData) -> "DeviceCycle":  # noqa: UP037 - the runtime type hook resolves this while the class body runs
        """Narrow a validated cycle representation onto the device.

        Parameters
        ----------
        cycle : solprop.translate.envelope_plasma.CycleIntegralData
            The channel's rate table and rules.

        Returns
        -------
        DeviceCycle
            The device copy.
        """
        nodes = np.zeros((_LEVELS, _MAX_ORDER), dtype=np.float32)
        weights = np.zeros_like(nodes)
        for level, rule in enumerate(cycle.rules):
            nodes[level, : rule.nodes.size] = rule.nodes
            weights[level, : rule.weights.size] = rule.weights
        return cls(
            DeviceRateTable.upload(cycle.rate),
            mx.array(nodes),
            mx.array(weights),
            cycle.moment_rtol,
            cycle.negligible_rate,
        )


def envelope_cycle(
    field: mx.array, cycle: DeviceCycle, exponent: int
) -> tuple[mx.array, mx.array, mx.array]:
    """Return a channel's cycle-averaged rate, loss current per unit work and failures.

    Parameters
    ----------
    field : mlx.core.array
        The scaled envelope ``E``, complex64, ``(n_nodes, n_pol, n_to)``; the physical
        field is ``2**exponent E``.
    cycle : DeviceCycle
        The channel's device representation.
    exponent : int
        The state scale's exponent.

    Returns
    -------
    tuple of mlx.core.array
        The rate, s^-1, ``(n_nodes, n_to)``; the loss current per unit ionisation work,
        complex64, of the field's shape, for the physical field; and the number of
        samples whose refinement failed, int32, a scalar.
    """
    n_nodes, n_pol, n_t = field.shape
    n = n_nodes * n_t
    table = cycle.table
    constants = np.array(
        [
            table.log_field_min,
            1.0 / table.d_log_field,
            table.field_min,
            table.field_max,
            table.rate_max,
            math.ldexp(1.0, exponent),
            cycle.moment_rtol,
            cycle.negligible_rate,
        ],
        dtype=np.float32,
    )
    rate, current_re, current_im, failed = _cycle_kernel()(  # pyright: ignore[reportCallIssue]
        inputs=[
            mx.real(field),
            mx.imag(field),
            table.coefficients.reshape(-1),
            table.constant_tail,
            table.inverse_edges,
            table.inverse_tail,
            mx.array(constants),
            cycle.nodes,
            cycle.weights,
        ],
        template=[
            ("N", n),
            ("NT", n_t),
            ("NPOL", n_pol),
            ("INTERVALS", table.constant_tail.size),
            ("LEVELS", _LEVELS),
            ("MAXORDER", _MAX_ORDER),
        ],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n_nodes, n_t), field.shape, field.shape, (n,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.int32],
    )
    current = current_re + 1j * current_im
    return rate, current.astype(mx.complex64), mx.sum(failed)
