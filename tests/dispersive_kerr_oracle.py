# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test-only reference forms for SEP 6's frequency-dependent electronic cubic.

Two things live here, written apart from any production code (SEP 6, Phase 0):

- :func:`direct_sum`, the independent oracle: the third-order polarisation of a
  band-limited periodic real field as an explicit sum over signed input triplets,
  ``O(K**3)`` in the number of occupied bins, with the kernel evaluated leg by leg;
- the FFT forms Phase 0 derived for both evaluators (the grouped gas polynomial and
  the symmetric factorisation), each with the per-leg bound of SEP 6 section 4.10, for
  the full cubic and for the number-conserving sector, on the field frame and on an
  envelope frame.

Phase 1 compares the package's implementation with :func:`direct_sum`; these FFT forms
are the record of the derivation it starts from.

Conventions. The grid is one period of ``n`` samples on ``t in [0, 2 pi)``, so the
angular frequency of FFT bin ``k`` is the integer ``k``. NumPy's forward FFT is the
internal FFT convention of ``docs/dev/conventions.md`` section 2: the analytic signal
``u`` of a real field ``E = Re u`` occupies the positive bins. The kernel depends on
each leg only through the weight ``d(k) = min((k / k_scale)**2, d_cap)``, the squared
signed wavenumber of SEP 6 section 4.5 in units of ``k_scale``, held at ``d_cap`` beyond
the bound of section 4.10. Fields carry no DC or Nyquist content, so every leg is
strictly positive or strictly negative.

A product of three real fields ``X Y Z`` with analytic signals ``x, y, z`` is
``(1/8)`` of the eight sign choices ``x^(*) y^(*) z^(*)``. The **number-conserving
sector** keeps the quartets with two positive and two negative signed legs, the output
counted with a minus sign: inputs ``(+, +, -)`` into a positive output and their
conjugates. Its real polarisation is ``Re[(1/4) Pi_+ (x y z* + x y* z + x* y z)]``.
The **full cubic** adds ``x y z`` (three photons into one) and the positive-output
part of ``x y* z*`` (one photon into three).
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

type Trilinear = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
type Weight = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Polynomial:
    """The gas polynomial ``1 + A S2 + B S2**2 + B' S4``, per unit ``gamma0``."""

    a: float
    b: float
    b_prime: float
    k_scale: float = 10.0
    d_cap: float = math.inf

    def weight(self, k: np.ndarray) -> np.ndarray:
        """Return the bounded leg weight ``d`` of signed integer frequencies ``k``."""
        return np.minimum((np.asarray(k, float) / self.k_scale) ** 2, self.d_cap)

    def kernel(self, legs: np.ndarray) -> np.ndarray:
        """Return the kernel of quartets ``legs[..., 4]`` (output first)."""
        d = self.weight(legs)
        s2 = d.sum(-1)
        s4 = (d**2).sum(-1)
        return 1.0 + self.a * s2 + self.b * s2**2 + self.b_prime * s4

    def kerr(self, d: np.ndarray) -> np.ndarray:
        """Return the Kerr diagonal at leg weight ``d``: ``S2 = 4d``, ``S4 = 4d**2``."""
        return 1.0 + 4 * self.a * d + (16 * self.b + 4 * self.b_prime) * d**2


@dataclass(frozen=True)
class Symmetric:
    """The symmetric factorisation of the same Kerr diagonal, ``prod_j g(k_j)``."""

    polynomial: Polynomial

    def weight(self, k: np.ndarray) -> np.ndarray:
        """Return the leg weight of the underlying polynomial."""
        return self.polynomial.weight(k)

    def g(self, k: np.ndarray) -> np.ndarray:
        """Return the fourth root of the Kerr diagonal relative to its static value."""
        return self.polynomial.kerr(self.weight(k)) ** 0.25

    def kernel(self, legs: np.ndarray) -> np.ndarray:
        """Return the product of the four legs' factors."""
        return np.prod(self.g(legs), axis=-1)


def scalar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the scalar trilinear form ``x y z``."""
    return x * y * z


def isotropic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the symmetric isotropic contraction ``T_ijkl x_j y_k z_l``.

    ``T = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 3``, the polarisation axis first; for
    ``x = y = z = E`` it is ``(E . E) E``.
    """
    return ((x * y).sum(0) * z + (x * z).sum(0) * y + (y * z).sum(0) * x) / 3.0


def analytic(field: np.ndarray) -> np.ndarray:
    """Return the analytic signal ``u`` of a real field, ``field = Re u``."""
    spectrum = np.fft.fft(field, axis=-1)
    k = np.fft.fftfreq(field.shape[-1], 1.0 / field.shape[-1])
    return np.fft.ifft(np.where(k > 0, 2.0 * spectrum, 0.0), axis=-1)


def positive(series: np.ndarray, offset: int = 0) -> np.ndarray:
    """Project onto absolute frequencies above zero; ``offset`` is a carrier bin."""
    spectrum = np.fft.fft(series, axis=-1)
    k = np.fft.fftfreq(series.shape[-1], 1.0 / series.shape[-1]) + offset
    return np.fft.ifft(np.where(k > 0, spectrum, 0.0), axis=-1)


def multiply(series: np.ndarray, weight: Weight, offset: int = 0) -> np.ndarray:
    """Apply a spectral multiplier, a function of the absolute bin."""
    spectrum = np.fft.fft(series, axis=-1)
    k = np.fft.fftfreq(series.shape[-1], 1.0 / series.shape[-1]) + offset
    out = np.fft.ifft(spectrum * weight(k), axis=-1)
    return np.real(out) if np.isrealobj(series) else out


def full_product(form: Trilinear, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Return the real cubic of three real fields."""
    return form(x, y, z)


def conserving_product(form: Trilinear, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Return the number-conserving part of a product of three real fields."""
    a, b, c = analytic(x), analytic(y), analytic(z)
    inner = form(a, b, c.conj()) + form(a, b.conj(), c) + form(a.conj(), b, c)
    return (0.25 * positive(inner)).real


def polynomial_cubic(
    field: np.ndarray, model: Polynomial, *, form: Trilinear = scalar, sector="full"
) -> np.ndarray:
    """Return ``6 P / (N gamma0)`` by the grouped form of SEP 6 section 4.5."""
    product = full_product if sector == "full" else conserving_product
    d = model.weight
    e0 = field
    e1 = multiply(e0, d)
    e2 = multiply(e1, d)
    a, b, bp = model.a, model.b, model.b_prime
    terms = (
        (lambda k: 1 + a * d(k) + (b + bp) * d(k) ** 2, product(form, e0, e0, e0)),
        (lambda k: 3 * a + 6 * b * d(k), product(form, e0, e0, e1)),
        (lambda k: 3 * (b + bp) + 0 * k, product(form, e0, e0, e2)),
        (lambda k: 6 * b + 0 * k, product(form, e0, e1, e1)),
    )
    total = np.zeros_like(field)
    for weight, source in terms:
        total = total + multiply(source, weight)
    return total


def symmetric_cubic(
    field: np.ndarray, model: Symmetric, *, form: Trilinear = scalar, sector="full"
) -> np.ndarray:
    """Return ``6 P / (N gamma0)`` as ``G[(G E)^3]``."""
    product = full_product if sector == "full" else conserving_product
    filtered = multiply(field, model.g)
    return multiply(product(form, filtered, filtered, filtered), model.g)


def envelope_cubic(
    envelope: np.ndarray, offset: int, cubic_of_weights, sector="full"
) -> np.ndarray:
    """Evaluate the analytic polarisation on an envelope frame ``A = u exp(-i m0 t)``.

    ``cubic_of_weights(filter_fields, products, output)`` builds the same grouped sum
    as the field forms, but with every multiplier evaluated at the absolute bin
    ``k + m0`` and the products written in the frame: ``x y z`` carries
    ``exp(2 i m0 t)`` and ``x y* z*`` carries ``exp(-2 i m0 t)``, as the solprop
    envelope harmonic does.
    """
    n = envelope.shape[-1]
    t = 2 * np.pi * np.arange(n) / n

    def filt(series, weight):
        return multiply(series, weight, offset)

    def product(form, x, y, z):
        conserving = form(x, y, z.conj()) + form(x, y.conj(), z) + form(x.conj(), y, z)
        if sector == "conserving":
            inner = conserving
        else:
            up = form(x, y, z) * np.exp(2j * offset * t)
            down = (
                form(x, y.conj(), z.conj())
                + form(x.conj(), y, z.conj())
                + form(x.conj(), y.conj(), z)
            ) * np.exp(-2j * offset * t)
            inner = up + conserving + down
        return 0.25 * positive(inner, offset)

    def output(series, weight):
        return multiply(series, weight, offset)

    return cubic_of_weights(filt, product, output)


def polynomial_envelope(
    envelope: np.ndarray, offset: int, model: Polynomial, *, form=scalar, sector="full"
) -> np.ndarray:
    """Return the analytic ``6 P / (N gamma0)`` of the polynomial, envelope frame."""
    d = model.weight
    a, b, bp = model.a, model.b, model.b_prime

    def build(filt, product, output):
        e0 = envelope
        e1 = filt(e0, d)
        e2 = filt(e1, d)
        return (
            output(
                product(form, e0, e0, e0), lambda k: 1 + a * d(k) + (b + bp) * d(k) ** 2
            )
            + output(product(form, e0, e0, e1), lambda k: 3 * a + 6 * b * d(k))
            + 3 * (b + bp) * product(form, e0, e0, e2)
            + 6 * b * product(form, e0, e1, e1)
        )

    return envelope_cubic(envelope, offset, build, sector)


def symmetric_envelope(
    envelope: np.ndarray, offset: int, model: Symmetric, *, form=scalar, sector="full"
) -> np.ndarray:
    """Return the analytic ``6 P / (N gamma0)`` of the factorisation, envelope frame."""

    def build(filt, product, output):
        filtered = filt(envelope, model.g)
        return output(product(form, filtered, filtered, filtered), model.g)

    return envelope_cubic(envelope, offset, build, sector)


def direct_sum(
    field: np.ndarray, kernel: Callable[[np.ndarray], np.ndarray], *, sector="full"
) -> np.ndarray:
    """Return ``6 P / (N gamma0)`` of a real field by the explicit triplet sum.

    ``field`` has shape ``(n,)`` (scalar) or ``(n_pol, n)``, in which case the isotropic
    tensor contracts the three inputs. Every signed input triplet of occupied bins is
    summed with the kernel of its quartet ``(k1 + k2 + k3; k1, k2, k3)``; the output
    must stay below the Nyquist bin, which the caller ensures by its grid.
    """
    vector = field.ndim == 2
    fields = field if vector else field[None, :]
    n = fields.shape[-1]
    spectra = np.fft.fft(fields, axis=-1)
    k_all = np.fft.fftfreq(n, 1.0 / n).astype(int)
    occupied = k_all[np.any(np.abs(spectra) > 1e-12 * np.abs(spectra).max(), axis=0)]
    if np.any(occupied == 0) or np.any(np.abs(occupied) >= n // 2):
        msg = "the oracle needs a field without DC or Nyquist content"
        raise ValueError(msg)
    triplets = np.array(list(itertools.product(occupied, repeat=3)))
    out_k = triplets.sum(1)
    if np.any(np.abs(out_k) >= n // 2):
        msg = "the grid is too short for the cubic's support"
        raise ValueError(msg)
    if sector == "conserving":
        signs = np.sign(np.column_stack([-out_k, triplets]))
        # A zero output leg (rectification) belongs to neither sign class.
        keep = ((signs > 0).sum(1) == 2) & ((signs < 0).sum(1) == 2)
        triplets, out_k = triplets[keep], out_k[keep]
    weights = kernel(np.column_stack([out_k, triplets]))
    s1, s2, s3 = (spectra[:, triplets[:, i] % n] for i in range(3))
    if vector:
        delta = np.eye(fields.shape[0])
        tensor = (
            np.einsum("ij,kl->ijkl", delta, delta)
            + np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        ) / 3.0
        values = np.einsum("ijkl,jm,km,lm->im", tensor, s1, s2, s3)
    else:
        values = s1 * s2 * s3
    result = np.zeros_like(spectra)
    for component in range(fields.shape[0]):
        np.add.at(result[component], out_k % n, weights * values[component])
    result /= n**2
    series = np.fft.ifft(result, axis=-1).real
    return series if vector else series[0]
