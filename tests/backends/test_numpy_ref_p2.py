# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""P2 of the verification ladder: z-varying loss *and* dispersion, still exact.

``docs/dev/stepper.md`` section 11, rung P2. For

.. math::

   i \partial_Z A + \tfrac{d(Z)}{2}\partial_{\tau\tau}A + \gamma(Z)|A|^2A
       + \tfrac{i}{2}\alpha(Z) A = 0,

the substitution :math:`A = e^{-\Gamma/2}B` with :math:`\Gamma = \int_0^Z\alpha` removes
the loss exactly, and :math:`\zeta(Z) = \int_0^Z d` turns what is left into the standard
nonlinear Schroedinger equation **provided** :math:`\gamma e^{-\Gamma}/d = 1`. That is
one closure condition on three functions, so ``alpha`` and ``d`` may be chosen
completely independently and :math:`\gamma = d\,e^{\Gamma}` follows. Then

.. math::

   A(Z, \tau) = e^{-\Gamma(Z)/2}\,\psi\bigl(\zeta(Z), \tau\bigr)

is exact for any solution :math:`\psi` of the standard equation: arbitrary z-varying
loss, arbitrary z-varying dispersion, an analytic answer and no reference solution.

The parameters are ``alpha(Z) = 0.7 + 0.3 sin 2Z`` and ``d(Z)
= 1 + 0.5 sin 3Z``, whose recurrence sits at ``Z = 1.2748`` with ``Gamma = 1.167`` -- 44
% of the amplitude lost, so this is a genuinely dissipative test that still has an
analytic answer. Two traps that module records are respected here: ``gamma`` is not
fixed at 1 (which would make the recurrence unreachable for ``alpha >= 2/pi``), and
``Gamma(L)`` is kept near 1 so that the nonlinear phase per step, rather than the scheme
under test, is not what limits the step.

The problem reaches the backend as a substituted kernel problem: the closed-form ``Phi``
becomes the linear operator's table and ``gamma(Z)`` the contract's density path, which
is how a synthetic fibre crosses the seam without the physics layer having to be able to
express it.
"""

import dataclasses
import math

import numpy as np
import pytest

from solprop import fourier
from solprop.stepper import PIController
from solprop.translate.linop import tabulate
from solprop.translate.problem import assemble

# The dimensionless problem, mapped onto the fibre by one time and one length scale.
TAU0 = 100e-15  # s
Z0 = 1.0  # m
GAMMA_REF = 0.11  # 1/(W m), the assembled fibre's coefficient at unit density
PEAK_POWER = 1.0 / (GAMMA_REF * Z0)  # W, the amplitude unit of the dimensionless field
BETA2_UNIT = TAU0**2 / Z0  # s^2/m
AMPLITUDE = 1.3  # the boosted soliton's eta
VELOCITY = 0.4  # its v, in units of 1/tau
RECURRENCE = math.pi / 2  # zeta at the recurrence
STEP_POSITION = 0.6  # Z of the P2b loss step


def smooth_loss(z):
    """``alpha(Z) = 0.7 + 0.3 sin 2Z``."""
    return 0.7 + 0.3 * np.sin(2.0 * z)


def smooth_loss_integral(z):
    """``Gamma(Z)``, in closed form so the reference carries no quadrature error."""
    return 0.7 * z + 0.15 * (1.0 - np.cos(2.0 * z))


def step_loss_integral(z):
    """``Gamma(Z)`` for the P2b variant, whose ``alpha`` jumps at ``STEP_POSITION``."""
    before, after = 0.4, 1.2
    return np.where(
        z < STEP_POSITION,
        before * z,
        before * STEP_POSITION + after * (z - STEP_POSITION),
    )


def dispersion_integral(z):
    """``zeta(Z)`` for ``d(Z) = 1 + 0.5 sin 3Z``, in closed form."""
    return z + (0.5 / 3.0) * (1.0 - np.cos(3.0 * z))


def kerr_path(z, loss_integral):
    """``gamma(Z) = d(Z) exp(Gamma(Z))``, the closure condition the exactness needs."""
    dispersion = 1.0 + 0.5 * np.sin(3.0 * z)
    return dispersion * np.exp(loss_integral(z))


def recurrence_position(tolerance=1e-13):
    """Return the ``Z`` at which ``zeta(Z) = pi/2``, by bisection."""
    low, high = 0.0, 3.0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if dispersion_integral(middle) < RECURRENCE:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def seed(zeta, tau):
    """A boosted fundamental soliton of the standard nonlinear Schroedinger equation."""
    argument = AMPLITUDE * (tau - VELOCITY * zeta)
    envelope = AMPLITUDE / np.cosh(argument)
    phase = VELOCITY * tau + (AMPLITUDE**2 - VELOCITY**2) * zeta / 2.0
    return envelope * np.exp(1j * phase)


def exact_spectrum(grid, z, loss_integral):
    """Return the analytic solution at ``z``, as a stored internal-convention spectrum.

    The closed form is written in the optics convention, in which the equation above is
    stated, and conjugated once on the way in; no other conversion happens
    anywhere in this module.
    """
    scale = z / Z0
    damping = math.exp(-0.5 * float(loss_integral(np.asarray(scale))))
    optics = (
        math.sqrt(PEAK_POWER)
        * damping
        * seed(float(dispersion_integral(np.asarray(scale))), grid.t / TAU0)
    )
    return fourier.forward(fourier.to_internal(optics), kind="envelope")


@pytest.fixture
def dissipative(
    soliton_grid,
    gnlse,
    gnlse_settings,
    substitute,
    synthetic_expansion,
    expanded_prefactor,
):
    """Return ``(*, stepped) -> (problem, distance, Gamma)`` for the P2 fibre.

    The operator is the closed-form ``Phi`` as the separable sum it is -- one array
    times the dispersion's integral plus another times the loss's -- and ``gamma(Z)``
    rides the contract's density path, which multiplies the compiled Kerr coefficient
    at every position. The prefactor is read from an expansion rather than from the
    constant fast path, because a fibre whose operator varies is what a real gradient
    looks like.
    """

    def build(*, stepped):
        loss_integral = step_loss_integral if stepped else smooth_loss_integral
        distance = recurrence_position() * Z0
        start = exact_spectrum(soliton_grid, 0.0, loss_integral)
        base = assemble(
            soliton_grid,
            gnlse.mode,
            [gnlse.kerr],
            start[np.newaxis, np.newaxis],
            controller=PIController(rtol=1e-11, atol=1e-20),
            save_planes=[distance],
            dz_max=distance / 40,
            dz_init=distance / 4000,
            **gnlse_settings,
        )
        detuning = base.grids.omega - base.grids.omega0
        active = base.grids.active_mask
        kinks = (STEP_POSITION * Z0,) if stepped else ()

        terms = [
            0.5j * BETA2_UNIT * Z0 * detuning**2 * active,
            -0.5 * active,
        ]

        def weights_of(z):
            """The two integrals of ``Phi = i T0^2 zeta Omega^2 / 2 - Gamma / 2``."""
            scale = np.asarray(z, dtype=float) / Z0
            return np.stack([dispersion_integral(scale), loss_integral(scale)], axis=-1)

        span = base.linear.z_span
        density = tabulate(
            lambda z: kerr_path(np.asarray(z) / Z0, loss_integral),
            span[0],
            span[1],
            kinks=kinks,
            tol=1e-13,
        )
        kerr = base.responses.kerr_coefficients
        assert kerr is not None
        problem = substitute(
            base,
            linear=synthetic_expansion(base, terms, weights_of, kinks=kinks),
            responses=dataclasses.replace(
                expanded_prefactor(base.responses),
                density=density,
                # Schema 6 binds Kerr to its own density leaf. The imposed P2
                # closure must replace that leaf as well as total density.
                kerr_coefficients=dataclasses.replace(kerr, density=density),
            ),
        )
        return problem, distance, loss_integral

    return build


def relative(got, want):
    """Return the deviation as a fraction of the reference's norm."""
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.mark.parametrize("stepped", [False, True])
def test_the_dissipative_solution_is_reproduced(
    backend_class, dissipative, soliton_grid, stepped
):
    """P2 and P2b: a smooth loss, and one that jumps at a declared hard kink."""
    problem, distance, loss_integral = dissipative(stepped=stepped)
    backend = backend_class(problem, grid=soliton_grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(distance)
    plane = backend.observe()
    backend.close()
    reference = exact_spectrum(soliton_grid, distance, loss_integral)
    reference = reference * problem.grids.active_mask
    # Measured 1.01e-13 for the smooth loss and 9.89e-14 for the stepped one, over 395
    # adaptive steps at rtol 1e-11. What is left is the grid's own representation of
    # the soliton -- its sech tails reach exp(-26) at the window edge -- and the
    # round-off of 395 propagator applications, not the scheme: the reference carries
    # no quadrature error, because Gamma and zeta are both closed form.
    assert relative(plane.field[0, 0, 0], reference) < 1e-11
    assert summary.met_tolerance
    if stepped:
        assert problem.linear.hard_kinks.tolist() == [STEP_POSITION * Z0]
        assert STEP_POSITION * Z0 in plane.step_history.z.tolist()


def test_the_parameters_are_the_ones_the_design_note_records():
    """The recurrence at Z = 1.2748 with Gamma = 1.167."""
    position = recurrence_position()
    assert position == pytest.approx(1.2748, abs=5e-5)
    assert float(smooth_loss_integral(np.asarray(position))) == pytest.approx(
        1.167, abs=1e-3
    )
    # 44 % of the amplitude lost: dissipative enough to be worth calling a test.
    assert math.exp(-0.5 * 1.167) == pytest.approx(0.558, abs=1e-3)
    # The closure condition is what makes the answer exact; check it on the paths.
    probe = np.linspace(0.0, position, 17)
    dispersion = 1.0 + 0.5 * np.sin(3.0 * probe)
    ratio = (
        kerr_path(probe, smooth_loss_integral)
        * np.exp(-smooth_loss_integral(probe))
        / dispersion
    )
    assert np.allclose(ratio, 1.0)
    # And alpha, whose integral the reference uses, is the one that was written down.
    step = 1e-6
    numerical = (
        smooth_loss_integral(probe + step) - smooth_loss_integral(probe)
    ) / step
    assert np.allclose(numerical, smooth_loss(probe), atol=1e-5)
