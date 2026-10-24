# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The protected operations of the ``mlx`` backend, and the probes that establish them.

SEP 33 section 6.3's arithmetic policy. Each protected operation has a mechanism and its
evidence:

* **Matrix products** are float32 only with ``MLX_ENABLE_TF32=0``, which the package
  sets when it is imported and MLX has not been initialised; :func:`gemm_probe` measures
  every dispatch family a run uses against a discriminating bound (``2**-17``) and
  refuses a device that computes them in TF32.
* **The stage propagator and every combination of stage terms** are formed in
  double-float arithmetic from binary64 terms held as pairs of binary32 numbers
  (:class:`DoubleFloatTerms`), then rounded once: on the Metal device by one custom
  kernel compiled in ``"safe"`` math mode, whose two-sum and two-product cannot be
  reassociated; on MLX's CPU device, which continuous integration uses, in binary64 on
  the host with NumPy (the memory is shared). The phase is range-reduced in double-float
  before its binary32 cosine and sine, and the exponential of the real part is
  ``exp(hi) (1 + lo)``.

Nothing outside :mod:`solprop.backends.mlx_gpu` imports MLX.
"""

import math
from dataclasses import dataclass
from functools import cache

import mlx.core as mx
import numpy as np

from solprop.exceptions import BackendError

__all__ = [
    "GEMM_PROBE_BOUND",
    "DoubleFloatTerms",
    "gemm_probe",
    "on_gpu",
    "population",
]

GEMM_PROBE_BOUND = 2.0**-17
"""The normwise relative error a float32 product must stay below (SEP 33 7.1).

Float32 products of inner dimension up to ``10**3`` are expected near
``sqrt(n) u <= 2e-6`` and TF32 near its own unit roundoff ``2**-11``: the bound
separates the two by more than a factor of four on each side.
"""

# 2 pi as a double-float pair, and its reciprocal: the reduction of a phase of up to
# 1e7 rad keeps the residual to about 2**-48 of the phase.
_TWO_PI_HI = float(np.float32(2 * math.pi))
_TWO_PI_LO = float(np.float32(2 * math.pi - _TWO_PI_HI))

_HEADER = """
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return float2(s, e);
}
inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return float2(s, e);
}
inline float2 two_prod(float a, float b) {
    float p = a * b;
    float e = metal::fma(a, b, -p);
    return float2(p, e);
}
inline float2 df_add(float2 x, float2 y) {
    float2 s = two_sum(x.x, y.x);
    float e = s.y + x.y + y.y;
    return quick_two_sum(s.x, e);
}
inline float2 df_mul(float2 x, float2 y) {
    float2 p = two_prod(x.x, y.x);
    p.y = metal::fma(x.x, y.y, metal::fma(x.y, y.x, p.y));
    return quick_two_sum(p.x, p.y);
}
"""

_COMBINE = """
    uint i = thread_position_in_grid.x;
    if (i >= N) { return; }
    float2 re = float2(0.0f, 0.0f);
    float2 im = float2(0.0f, 0.0f);
    for (int t = 0; t < TERMS; ++t) {
        float2 w = float2(w_hi[t], w_lo[t]);
        uint k = t * N + i;
        re = df_add(re, df_mul(w, float2(re_hi[k], re_lo[k])));
        im = df_add(im, df_mul(w, float2(im_hi[k], im_lo[k])));
    }
    if (EXPONENTIAL == 0) {
        out_re[i] = re.x + re.y;
        out_im[i] = im.x + im.y;
        return;
    }
    float sign = SIGN > 0 ? 1.0f : -1.0f;
    float turns = metal::rint(im.x * INV_TWO_PI);
    float2 period = float2(TWO_PI_HI, TWO_PI_LO);
    float2 phase = df_add(im, df_mul(float2(-turns, 0.0f), period));
    float angle = sign * (phase.x + phase.y);
    // exp(hi + lo) = exp(hi) (1 + lo) to binary64-level accuracy, lo being below
    // hi's unit in the last place.
    float magnitude = metal::precise::exp(sign * re.x) * (1.0f + sign * re.y);
    out_re[i] = magnitude * metal::precise::cos(angle);
    out_im[i] = magnitude * metal::precise::sin(angle);
"""


@cache
def _kernel() -> object:
    """Return the compiled double-float combination kernel.

    Returns
    -------
    object
        The ``mlx.core.fast.metal_kernel`` callable.
    """
    source = (
        _COMBINE.replace("INV_TWO_PI", f"{1 / (2 * math.pi)!r}f")
        .replace("TWO_PI_HI", f"{_TWO_PI_HI!r}f")
        .replace("TWO_PI_LO", f"{_TWO_PI_LO!r}f")
    )
    return mx.fast.metal_kernel(
        name="solprop_double_float_combine",
        input_names=["re_hi", "re_lo", "im_hi", "im_lo", "w_hi", "w_lo"],
        output_names=["out_re", "out_im"],
        source=source,
        header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def on_gpu() -> bool:
    """Return whether MLX's default device is the Metal GPU.

    Returns
    -------
    bool
        True on the GPU, False on MLX's CPU device.
    """
    return mx.default_device() == mx.gpu


def _split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split binary64 values into a binary32 head and tail.

    Parameters
    ----------
    values : numpy.ndarray
        Real binary64 values.

    Returns
    -------
    tuple of numpy.ndarray
        ``hi = float32(v)`` and ``lo = float32(v - hi)``; ``hi + lo`` holds ``v`` to
        about 48 bits.
    """
    hi = values.astype(np.float32)
    lo = (values - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


@dataclass(frozen=True)
class DoubleFloatTerms:
    """Complex binary64 terms held on the device for double-float combination.

    On the Metal device each term is four float32 arrays (the real and imaginary heads
    and tails); on MLX's CPU device it is one complex128-equivalent pair of float64
    arrays. :meth:`combine` returns ``sum_t w_t T_t`` rounded once to complex64, and
    :meth:`exponential` ``exp(sign sum_t w_t T_t)`` with the phase range-reduced in
    double-float before the binary32 exponential, cosine and sine (SEP 33 4.2).

    Attributes
    ----------
    shape : tuple of int
        The shape of one term.
    n_terms : int
        The number of terms.
    parts : tuple of mlx.core.array
        The device arrays, each of shape ``(n_terms * prod(shape),)``; empty on MLX's
        CPU device.
    host : numpy.ndarray or None
        The binary64 terms, kept on MLX's CPU device only.
    """

    shape: tuple[int, ...]
    n_terms: int
    parts: tuple[mx.array, ...]
    host: np.ndarray | None = None

    @classmethod
    def upload(cls, terms: np.ndarray) -> "DoubleFloatTerms":  # noqa: UP037 - the runtime type hook resolves this while the class body runs
        """Upload binary64 terms of shape ``(n_terms, *shape)``.

        Parameters
        ----------
        terms : numpy.ndarray
            Complex binary64 terms, finite.

        Returns
        -------
        DoubleFloatTerms
            The terms on the device.

        Raises
        ------
        BackendError
            If a term is not finite.
        """
        values = np.ascontiguousarray(terms, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise BackendError("a stage term must be finite to reach the device")
        shape = values.shape[1:]
        flat = values.reshape(-1)
        if on_gpu():
            re_hi, re_lo = _split(flat.real)
            im_hi, im_lo = _split(flat.imag)
            parts = tuple(mx.array(part) for part in (re_hi, re_lo, im_hi, im_lo))
            return cls(shape, int(values.shape[0]), parts)
        return cls(shape, int(values.shape[0]), (), values)

    @property
    def nbytes(self) -> int:
        """Return the device bytes the terms hold.

        Returns
        -------
        int
            Their size.
        """
        host = 0 if self.host is None else int(self.host.nbytes)
        return int(sum(part.nbytes for part in self.parts)) + host

    def _evaluate(
        self, weights: np.ndarray, *, exponential: int, sign: int
    ) -> mx.array:
        """Combine the terms over binary64 weights on the device.

        Parameters
        ----------
        weights : numpy.ndarray
            One binary64 weight per term.
        exponential : int
            1 to return the exponential, 0 the combination.
        sign : int
            The exponential's sign, +1 or -1.

        Returns
        -------
        mlx.core.array
            Complex64 of shape :attr:`shape`.

        Raises
        ------
        BackendError
            If the weights do not match the terms.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != self.n_terms:
            msg = f"{w.size} weights for {self.n_terms} stage terms"
            raise BackendError(msg)
        if self.host is not None:
            total = np.tensordot(w, self.host, axes=1)
            if exponential:
                total = np.exp(sign * total)
            return mx.array(total.astype(np.complex64))
        n = math.prod(self.shape)
        w_hi, w_lo = _split(w)
        out_re, out_im = _kernel()(  # pyright: ignore[reportCallIssue]
            inputs=[*self.parts, mx.array(w_hi), mx.array(w_lo)],
            template=[
                ("N", n),
                ("TERMS", self.n_terms),
                ("EXPONENTIAL", exponential),
                ("SIGN", sign),
            ],
            grid=(n, 1, 1),
            threadgroup=(min(256, n), 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return (out_re + 1j * out_im).astype(mx.complex64).reshape(self.shape)

    def combine(self, weights: np.ndarray) -> mx.array:
        """Return ``sum_t w_t T_t`` rounded once to complex64.

        Parameters
        ----------
        weights : numpy.ndarray
            One binary64 weight per term.

        Returns
        -------
        mlx.core.array
            The combination.
        """
        return self._evaluate(weights, exponential=0, sign=1)

    def exponential(self, weights: np.ndarray, sign: int = 1) -> mx.array:
        """Return ``exp(sign sum_t w_t T_t)`` to binary64 accuracy, rounded once.

        Parameters
        ----------
        weights : numpy.ndarray
            One binary64 weight per term.
        sign : int, optional
            +1 for the propagator, -1 for its inverse.

        Returns
        -------
        mlx.core.array
            The complex64 propagator.
        """
        return self._evaluate(weights, exponential=1, sign=sign)


_POPULATION = """
    uint i = thread_position_in_grid.x;
    if (i >= N) { return; }
    float x = exposure[i];
    float keep = constants[0];
    float initial = constants[1];
    // Kahan's expm1(-x) from the precise exponential and logarithm: a few ulp at every
    // argument, where MLX's own GPU expm1 loses up to 1e-4 relative at small ones.
    float u = metal::precise::exp(-x);
    float m;
    if (u == 1.0f) {
        m = -x;
    } else if (u - 1.0f == -1.0f) {
        m = -1.0f;
    } else {
        m = (u - 1.0f) * (-x) / metal::precise::log(u);
    }
    fraction[i] = initial + keep * (-m);
    survival[i] = keep * u;
"""


@cache
def _population_kernel() -> object:
    """Return the compiled population kernel.

    Returns
    -------
    object
        The ``mlx.core.fast.metal_kernel`` callable.
    """
    return mx.fast.metal_kernel(
        name="solprop_stable_population",
        input_names=["exposure", "constants"],
        output_names=["fraction", "survival"],
        source=_POPULATION,
        compile_options={"math_mode": "safe"},
    )


def population(exposure: mx.array, preionisation: float) -> tuple[mx.array, mx.array]:
    """Return the ionised fraction and the neutral survival in their stable forms.

    The oracle's ``stable_population`` (SEP 33 section 4.4): ``f = p + (1 - p)
    (-expm1(-X))`` and ``sigma = (1 - p) exp(-X)``. On the Metal device it is a custom
    kernel in ``"safe"`` math mode with Kahan's ``expm1``; on MLX's CPU device MLX's own
    ``expm1``, which is accurate there.

    Parameters
    ----------
    exposure : mlx.core.array
        ``X``, float32, non-negative.
    preionisation : float
        ``p`` in ``[0, 1]``.

    Returns
    -------
    tuple of mlx.core.array
        The fraction and the survival, float32.
    """
    keep = float(np.float32(1.0 - preionisation))
    initial = float(np.float32(preionisation))
    if not on_gpu():
        return initial + keep * -mx.expm1(-exposure), keep * mx.exp(-exposure)
    flat = exposure.reshape(-1)
    n = flat.size
    fraction, survival = _population_kernel()(  # pyright: ignore[reportCallIssue]
        inputs=[flat, mx.array(np.array([keep, initial], dtype=np.float32))],
        template=[("N", n)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return fraction.reshape(exposure.shape), survival.reshape(exposure.shape)


def gemm_probe(shapes: list[tuple[int, int, int]]) -> float:
    """Measure float32 matrix products of each dispatch family a run uses.

    Gaussian random factors of each ``(m, k, n)`` are multiplied on the device and
    compared with binary64 (SEP 33 6.3, 7.1).

    Parameters
    ----------
    shapes : list of tuple of int
        The products' shapes.

    Returns
    -------
    float
        The largest normwise relative error.

    Raises
    ------
    BackendError
        If any product exceeds :data:`GEMM_PROBE_BOUND`: the device computes in TF32
        or lower, which SEP 25 forbids.
    """
    rng = np.random.default_rng(33)
    worst = 0.0
    for m, k, n in shapes:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        product = np.asarray(mx.array(a) @ mx.array(b), dtype=np.float64)
        exact = a.astype(np.float64) @ b.astype(np.float64)
        error = float(np.linalg.norm(product - exact) / np.linalg.norm(exact))
        worst = max(worst, error)
        if error > GEMM_PROBE_BOUND:
            msg = (
                f"MLX computed a {m}x{k} by {k}x{n} float32 product to {error:.2e}, "
                f"beyond {GEMM_PROBE_BOUND:.1e}: its matrix products run in TF32. Set "
                "MLX_ENABLE_TF32=0 before MLX is first used in the process (in the "
                "shell, or before any import of mlx)"
            )
            raise BackendError(msg)
    return worst
