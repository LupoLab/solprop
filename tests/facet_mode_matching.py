# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""An independent full-vector mode-matching solution of a step-index fibre facet.

The oracle of the vector coupling model's recorded envelope
(:data:`solprop.coupling.envelopes.VECTOR_STEP_INDEX_ENVELOPE`). A beam in a
homogeneous medium of permittivity :math:`\varepsilon_{\rm in}` meets the planar
entrance of a step-index fibre (core :math:`\varepsilon_{\rm co}` inside radius a,
cladding :math:`\varepsilon_{\rm cl}` beyond) at z = 0. Tangential E and H are
continuous there, the input side carries the incident and a reflected (backward)
field, and the fibre side its forward guided and radiation fields. ``vector_paraxial``
keeps only the incident field in the flux pairing; this module solves the whole
interface.

**Formulation** (optics convention, :math:`e^{i(\beta z - \omega t)}`, SI). With the
transverse fields :math:`\mathbf u = \mathbf E_t` and
:math:`\mathbf v = -\hat z\times\mathbf H_t` (so a forward plane wave has
:math:`\mathbf v = Y\mathbf u`), Maxwell's equations in a z-invariant medium give
:math:`\beta\mathbf u = A\mathbf v` and :math:`\beta\mathbf v = P\mathbf u` with

.. math::

    A\mathbf v = \omega\mu_0\mathbf v + \frac{1}{\omega\varepsilon_0}
                 \nabla_t\bigl(\varepsilon^{-1}\nabla_t\cdot\mathbf v\bigr),\qquad
    P\mathbf u = \omega\varepsilon_0\varepsilon\mathbf u - \frac{1}{\omega\mu_0}
                 \hat z\times\nabla_t(\hat z\cdot\nabla_t\times\mathbf u),

both symmetric. Forward modes solve :math:`AP\mathbf u = \beta^2\mathbf u` with
:math:`\operatorname{Re}\beta > 0` or :math:`\operatorname{Im}\beta > 0`, and the
forward admittance operator is :math:`Y = P(AP)^{-1/2}`. Continuity at the facet,
:math:`\mathbf u_i + \mathbf u_r = \mathbf u_t` and
:math:`Y_0\mathbf u_i - Y_0\mathbf u_r = Y_f\mathbf u_t`, gives the transmitted field
:math:`\mathbf u_t = 2(Y_0 + Y_f)^{-1}Y_0\mathbf u_i` exactly. Its guided amplitude is
the flux pairing of :math:`(\mathbf u_t, Y_f\mathbf u_t)` with the unit-power mode;
the fibre's forward modes are flux-orthogonal to every other forward and backward
mode, so nothing else contributes. The difference from ``vector_paraxial``'s
amplitude with the same incident H is, by the algebra of the two expressions,

.. math::

    a_{\rm exact} - a_{\rm model} = -\tfrac14\langle\mathbf u_i,
        D(Y_0 + Y_f)^{-1}D\,\mathbf e_m\rangle,\qquad D = Y_f - Y_0,

second order in the index step, and :math:`D\mathbf e_m = \mathbf h_m -
Y_0\mathbf e_m` is the mode's own closure defect.

**Discretisation.** The fibre and the input medium are enclosed in a perfectly
conducting cylinder of radius R much larger than a and the beam, and each azimuthal
order :math:`e^{in\phi}` is solved separately (the structure is rotationally
symmetric, so orders do not mix). The basis is the cylinder's own orthonormal TM
(:math:`\nabla_t J_n(k r)e^{in\phi}`, :math:`J_n(kR) = 0`) and TE
(:math:`\hat z\times\nabla_t J_n(kr)e^{in\phi}`, :math:`J_n'(kR) = 0`) transverse
fields, which carry the conductor's boundary condition for both u and v. In it
:math:`Y_0` is diagonal (:math:`\beta/(\omega\mu_0)` for TE,
:math:`\omega\varepsilon_0\varepsilon_{\rm in}/\beta` for TM) and A, P are Galerkin
matrices of the weak forms above, integrated by Gauss--Legendre panels broken at the
core radius. The fibre's modes are the eigenvectors of AP. Nothing here uses the
package's step-index solver: the guided eigenvalue converges to the exact vector
eigenvalue equation (Snyder and Love, *Optical Waveguide Theory*, 1983, eq. 12-11),
which :func:`hybrid_eigenvalue` solves directly, and the fields converge to
:meth:`solprop.modes.StepIndexMode.electromagnetic_field`.

**Convergence.** The basis is smooth and the fields' radial component jumps at the
core, so the eigenvalues and shares converge as 1/N in the basis size N (the change
from N to 2N estimates the error); the facet correction, a difference of two
amplitudes formed from one discretisation, converges much faster.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jnp_zeros, jv, jvp, kv, kvp

from solprop.constants import EPS0, MU0, C

__all__ = [
    "Beam",
    "FacetShares",
    "StepIndexFacet",
    "hybrid_eigenvalue",
    "ring_rule",
]

Y_VACUUM = float(np.sqrt(EPS0 / MU0))
"""The vacuum admittance, A/V."""


def ring_rule(
    breaks: tuple[float, ...], panel: float, n_theta: int = 96, nodes: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a polar rule: Gauss--Legendre panels in r, uniform in theta.

    The radial panels break at every entry of ``breaks`` (the first is the axis,
    the last the outer radius) and are at most ``panel`` long; the weights are
    plain dr weights, the Jacobian is applied by the caller.
    """
    radii, weights = [], []
    for low, high in pairwise(breaks):
        count = max(1, int(np.ceil((high - low) / panel)))
        edges = np.linspace(low, high, count + 1)
        x, w = np.polynomial.legendre.leggauss(nodes)
        for left, right in pairwise(edges):
            radii.append(0.5 * (right - left) * x + 0.5 * (right + left))
            weights.append(0.5 * (right - left) * w)
    theta = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    return np.concatenate(radii), np.concatenate(weights), theta


class _DiscBasis:
    """The orthonormal TM and TE transverse fields of a conducting disc, order n."""

    def __init__(self, order: int, outer: float, count: int) -> None:
        self.order = order
        m = abs(order)
        self.k_tm = jn_zeros(m, count) / outer
        # J_0' = -J_1, whose zero at the origin is not a mode.
        self.k_te = (jnp_zeros(m, count) if m > 0 else jn_zeros(1, count)) / outer
        self.k = np.concatenate((self.k_tm, self.k_te))
        self.is_te = np.arange(2 * count) >= count
        # Closed-form norms: the Dirichlet and Neumann Bessel integrals.
        self._norm_tm = (
            self.k_tm * np.sqrt(np.pi) * outer * np.abs(jv(m + 1, self.k_tm * outer))
        )
        kr = self.k_te * outer
        self._norm_te = (
            self.k_te
            * np.sqrt(np.pi)
            * outer
            * np.sqrt(1 - m**2 / kr**2)
            * np.abs(jv(m, kr))
        )

    def fields(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return e_r, e_phi and div e of every basis field at radii r (> 0)."""
        n, m = self.order, abs(self.order)
        r = np.asarray(r, dtype=float)[:, None]
        x = self.k_tm[None, :] * r
        tm = self._norm_tm[None, :]
        er_tm = self.k_tm * jvp(m, x) / tm
        ep_tm = 1j * n * jv(m, x) / r / tm
        div_tm = -(self.k_tm**2) * jv(m, x) / tm
        x = self.k_te[None, :] * r
        te = self._norm_te[None, :]
        er_te = -1j * n * jv(m, x) / r / te
        ep_te = self.k_te * jvp(m, x) / te
        return (
            np.concatenate((er_tm, er_te), axis=1),
            np.concatenate((ep_tm, ep_te), axis=1),
            np.concatenate((div_tm, np.zeros_like(er_te)), axis=1),
        )

    def grams(self, r: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the field and divergence Grams over a radial rule (r dr weights)."""
        er, ep, div = self.fields(r)
        weight = 2 * np.pi * r * w
        field = (er.conj().T * weight) @ er + (ep.conj().T * weight) @ ep
        return field, (div.conj().T * weight) @ div

    def project(
        self,
        radii: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        radial: np.ndarray,
        azimuthal: np.ndarray,
    ) -> np.ndarray:
        """Return the coefficients of a field sampled on rings, shape (n_basis,)."""
        phase = np.exp(-1j * self.order * theta)
        radial_n = radial @ phase / theta.size
        azimuthal_n = azimuthal @ phase / theta.size
        er, ep, _ = self.fields(radii)
        weight = 2 * np.pi * radii * weights
        return (er.conj().T * weight) @ radial_n + (ep.conj().T * weight) @ azimuthal_n


@dataclass(frozen=True)
class FacetShares:
    """One mode's captured share of an incident beam, three ways.

    Attributes
    ----------
    exact : float
        The mode-matching solution's guided power over the incident power.
    model_exact_h : float
        ``vector_paraxial``'s flux pairing with the incident beam's exact H.
    model : float
        ``vector_paraxial``'s flux pairing with the closure
        :math:`H_t = Y_0\\hat z\\times E_t`, as the package computes it.
    """

    exact: float
    model_exact_h: float
    model: float

    @property
    def facet(self) -> float:
        """Return what the model omits at the interface: exact minus model."""
        return self.exact - self.model_exact_h


class StepIndexFacet:
    """The facet of one azimuthal order: fibre modes, admittances, matching.

    Parameters
    ----------
    order : int
        Azimuthal order n of :math:`e^{in\\phi}`.
    radius : float
        Core radius a, m.
    indices : tuple of float
        Input medium, core and cladding indices.
    omega : float
        Angular frequency, rad/s.
    outer : float
        Conducting cylinder's radius R, m.
    count : int
        TM and TE basis fields each.
    """

    def __init__(
        self,
        order: int,
        radius: float,
        indices: tuple[float, float, float],
        omega: float,
        outer: float,
        count: int,
    ) -> None:
        n_in, n_co, n_cl = indices
        self.radius, self.indices, self.omega = radius, indices, omega
        self.k0 = omega / C
        basis = self.basis = _DiscBasis(order, outer, count)
        size = 2 * count
        # Panels short enough for the fastest basis oscillation, broken at a.
        panel = np.pi / (1.5 * max(float(basis.k.max()), 1.0 / radius))
        core_r, core_w, _ = ring_rule((0.0, radius), panel, n_theta=1)
        clad_r, clad_w, _ = ring_rule((radius, outer), panel, n_theta=1)
        core_g, core_d = basis.grams(core_r, core_w)
        clad_g, clad_d = basis.grams(clad_r, clad_w)
        self.orthonormality = float(np.max(np.abs(core_g + clad_g - np.eye(size))))
        self.cladding_gram = clad_g
        te = basis.is_te
        p = omega * EPS0 * (n_co**2 * core_g + n_cl**2 * clad_g) - np.diag(
            np.where(te, basis.k**2, 0.0)
        ) / (omega * MU0)
        a = omega * MU0 * np.eye(size) - (core_d / n_co**2 + clad_d / n_cl**2) / (
            omega * EPS0
        )
        beta_sq, vectors = np.linalg.eig(a @ p)
        beta = np.sqrt(beta_sq.astype(complex))
        # The forward branch: propagating towards +z, or decaying there. A real
        # beta^2 carries a rounding-level imaginary part from the eigensolver, whose
        # sign must not pick the branch.
        propagating = np.abs(np.imag(beta)) <= 1e-9 * np.abs(beta)
        beta = np.where(
            propagating,
            np.abs(np.real(beta)) + 1j * np.abs(np.imag(beta)),
            np.where(np.imag(beta) < 0, -beta, beta),
        )
        self.beta, self.vectors, self.p = beta, vectors, p
        self.y_fibre = p @ vectors @ np.diag(1 / beta) @ np.linalg.inv(vectors)
        beta_in = np.sqrt((self.k0**2 * n_in**2 - basis.k**2).astype(complex))
        beta_in = np.where(np.imag(beta_in) < 0, -beta_in, beta_in)
        self.y_input = np.diag(
            np.where(te, beta_in / (omega * MU0), omega * EPS0 * n_in**2 / beta_in)
        )

    def guided(self) -> np.ndarray:
        """Return the guided modes' indices, highest effective index first."""
        n_cl = self.indices[2]
        neff = np.real(self.beta) / self.k0
        bound = (np.abs(np.imag(self.beta)) < 1e-9 * self.k0) & (neff > n_cl)
        found = np.flatnonzero(bound)
        return found[np.argsort(-neff[found])]

    def unit_mode(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a mode's u and v coefficients normalised to unit power."""
        u = self.vectors[:, index]
        power = 0.5 * np.real(u.conj() @ self.p @ u / self.beta[index])
        u = u / np.sqrt(power)
        return u, self.p @ u / self.beta[index]

    def amplitudes(
        self, coefficients: np.ndarray, index: int
    ) -> tuple[complex, complex, complex, tuple[float, float, float]]:
        """Return the exact, model (exact H) and model (closure) amplitudes.

        Also the incident, reflected and transmitted powers, W, whose balance
        checks the matching.
        """
        u, v = self.unit_mode(index)
        admittance = self.indices[0] * Y_VACUUM
        transmitted = 2 * np.linalg.solve(
            self.y_input + self.y_fibre, self.y_input @ coefficients
        )
        exact = 0.25 * (
            v.conj() @ transmitted + u.conj() @ (self.y_fibre @ transmitted)
        )
        model_exact_h = 0.25 * (
            v.conj() @ coefficients + u.conj() @ (self.y_input @ coefficients)
        )
        model = 0.25 * (
            v.conj() @ coefficients + admittance * (u.conj() @ coefficients)
        )
        reflected = transmitted - coefficients
        powers = (
            0.5 * float(np.real(coefficients.conj() @ self.y_input @ coefficients)),
            0.5 * float(np.real(reflected.conj() @ self.y_input @ reflected)),
            0.5 * float(np.real(transmitted.conj() @ self.y_fibre @ transmitted)),
        )
        return complex(exact), complex(model_exact_h), complex(model), powers

    def worst_field(self, sine: float) -> float:
        r"""Return the largest |omitted facet share| over every field in a cone.

        Each guided mode's exact amplitude is a linear form of the incident
        coefficients, :math:`\tfrac14(\mathbf v_m^\dagger + \mathbf u_m^\dagger
        Y_f)T` with :math:`T = 2(Y_0 + Y_f)^{-1}Y_0`, and the model's with the
        same incident H is :math:`\tfrac14(\mathbf v_m^\dagger + \mathbf
        u_m^\dagger Y_0)`. Summed over the order's guided modes, the difference of
        the two rank-one capture forms, over the incident measure M0
        (:math:`\tfrac12 n_{\rm in}Y_{\rm vac}|\mathbf c|^2` in the orthonormal
        basis), restricted to the basis fields whose transverse wavenumber lies
        inside the cone ``sine`` (of the polar angle in the input medium), is a
        Hermitian form: its largest eigenvalue modulus bounds the summed facet
        share of every incident field of this order in the cone, whatever its
        shape or polarisation. Orders do not mix, so the largest over orders
        bounds every field.
        """
        transfer = 2 * np.linalg.solve(self.y_input + self.y_fibre, self.y_input)
        live = self.basis.k < sine * self.k0 * self.indices[0]
        measure = 0.5 * self.indices[0] * Y_VACUUM
        form = np.zeros((int(np.sum(live)),) * 2, dtype=complex)
        for index in self.guided():
            u, v = self.unit_mode(int(index))
            exact = (0.25 * (v.conj() + u.conj() @ self.y_fibre) @ transfer)[live]
            model = (0.25 * (v.conj() + u.conj() @ self.y_input))[live]
            form += np.outer(exact.conj(), exact) - np.outer(model.conj(), model)
        values = np.linalg.eigvalsh(0.5 * (form + form.conj().T) / measure)
        return float(np.max(np.abs(values))) if values.size else 0.0


@dataclass(frozen=True)
class Beam:
    """An x-polarised Gaussian beam at the facet, in units of the core radius.

    Attributes
    ----------
    waist : float
        1/e field radius over a.
    shift : tuple of float
        Displacement (x, y) over a.
    tilt : float
        Transverse wavenumber along x over :math:`k_0\\,\\mathrm{NA}`.
    """

    waist: float
    shift: tuple[float, float] = (0.0, 0.0)
    tilt: float = 0.0

    def field(
        self, x: np.ndarray, y: np.ndarray, radius: float, kappa_unit: float
    ) -> np.ndarray:
        """Return E_x (optics convention) at points, for a core radius and k0 NA."""
        dx, dy = self.shift
        envelope = np.exp(
            -((x - dx * radius) ** 2 + (y - dy * radius) ** 2)
            / (self.waist * radius) ** 2
        )
        return envelope * np.exp(1j * self.tilt * kappa_unit * x)


def mode_shares(  # noqa: PLR0913 - the fibre, the beam and the discretisation
    beam: Beam,
    radius: float,
    indices: tuple[float, float, float],
    omega: float,
    orders: tuple[int, ...] = (1, -1),
    which: int | None = 0,
    *,
    outer: float = 8.0,
    count: int = 100,
) -> FacetShares:
    """Return one guided family's shares of a beam, summed over the given orders.

    ``which`` picks the guided mode of each order by descending effective index
    (0 is HE11 in orders 1 and -1), or None for every guided mode of the orders;
    the ``n`` and ``-n`` orders together are the two parities of one mode.
    ``outer`` is R/a.
    """
    n_in, n_co, n_cl = indices
    radii, weights, theta = ring_rule((0.0, radius, outer * radius), radius / 4)
    r, t = radii[:, None], theta[None, :]
    x, y = r * np.cos(t), r * np.sin(t)
    ex = beam.field(x, y, radius, omega / C * np.sqrt(n_co**2 - n_cl**2))
    incident = (
        (0.5 * n_in * Y_VACUUM * np.sum((radii * weights)[:, None] * np.abs(ex) ** 2))
        * 2
        * np.pi
        / theta.size
    )
    totals = np.zeros(3)
    for order in orders:
        facet = StepIndexFacet(order, radius, indices, omega, outer * radius, count)
        coefficients = facet.basis.project(
            radii, weights, theta, ex * np.cos(t), -ex * np.sin(t)
        )
        guided = facet.guided()
        for index in guided if which is None else guided[which : which + 1]:
            amplitudes = facet.amplitudes(coefficients, int(index))[:3]
            totals += np.abs(np.asarray(amplitudes)) ** 2 / incident
    return FacetShares(*(float(one) for one in totals))


def hybrid_eigenvalue(order: int, v: float, n_co: float, n_cl: float) -> float:
    """Return the HE_{n1} core parameter U of the exact vector eigenvalue equation.

    Snyder and Love, *Optical Waveguide Theory* (1983), eq. 12-11 with
    :math:`W = \\sqrt{V^2 - U^2}`; the HE_{11} root is the smallest positive one,
    below the first zero of J_0 for order 1.
    """
    ratio = n_cl**2 / n_co**2

    def residual(u: float) -> float:
        w = np.sqrt(v**2 - u**2)
        j = jvp(order, u) / (u * jv(order, u))
        k = kvp(order, w) / (w * kv(order, w))
        return float(
            (j + k) * (j + ratio * k)
            - order**2 * (1 / u**2 + 1 / w**2) * (1 / u**2 + ratio / w**2)
        )

    grid = np.linspace(1e-3, min(v, 2.404825557695773) - 1e-9, 4001)
    values = np.array([residual(u) for u in grid])
    change = int(np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0])
    return float(brentq(residual, grid[change], grid[change + 1]))
