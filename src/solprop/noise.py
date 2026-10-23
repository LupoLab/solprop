# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shot-noise realisations: one photon per mode per frequency bin.

Two noise models share one draw. The **input** model of the supercontinuum literature
[2]_ adds a realisation to the initial field once, at ``z = 0``, and lets the
propagation carry it; the **modified** model of Chen and Wise [1]_ keeps it as a source
term injected into the nonlinear operator at every step, which seeds spontaneous
processes without the artificial four-wave-mixing phase matching that a propagated
background produces. The modified field is advanced by the unitary part of the linear
operator rather than held constant in z: holding it constant is exact only in the
dispersion-free gain model it came from (see ``docs/dev/decisions.md``).

The amplitude is fixed by ``docs/dev/conventions.md`` §4 and pairs with the spectral
density of :func:`solprop.fourier.spectral_density`, so that every bin of the active
band carries exactly ``hbar omega``: one photon. The phase is uniform on ``[0, 2 pi)``
and is the only random quantity. Both grids are masked to the active band, so no noise
sits in the zero-frequency bin or above the window edge of a field grid.

Randomness is explicit: the caller passes a :class:`numpy.random.Generator`, and the
entropy it was created with is recorded on the realisation so that a run can be
reproduced from its output file.

The manual page is ``docs/guide/media/noise.md``.

References
----------
.. [1] Y. Chen and F. W. Wise, "A simple accurate way to model noise-seeded ultrafast
       nonlinear processes", arXiv:2410.20567 (2024).
.. [2] J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic
       crystal fiber", Rev. Mod. Phys. 78, 1135 (2006). doi:10.1103/RevModPhys.78.1135
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.constants import EPS0, HBAR, C
from solprop.exceptions import NoiseError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.modes.slab import PlanarGrid

__all__ = [
    "NoiseModel",
    "NoiseRealisation",
    "input_noise",
    "modified_noise",
    "photons_per_bin",
    "spatial_noise",
    "transformed_noise",
]

logger = logging.getLogger(__name__)

NoiseModel = Literal["input", "modified"]
"""Which of the two shot-noise models a realisation belongs to."""


@dataclass(frozen=True, eq=False)
class NoiseRealisation:
    """One drawn shot-noise field, with the seed that produced it.

    Equality is by identity (``eq=False``) because ``spectrum`` is an array; compare the
    seeds instead, which is what reproducibility actually means.

    Attributes
    ----------
    spectrum : numpy.ndarray
        Stored noise spectrum, shape ``(n_modes, n_pol, n_omega)``, complex, in the
        square root of watts for modes, or the spatial transform's field units for a
        spatial run, in the internal FFT convention. Zero outside the active band.
    seed : int or None
        Entropy of the generator the realisation was drawn from, or ``None`` when it
        could not be recovered. Recorded so that a run is reproducible and resumable.
    model : {'input', 'modified'}
        Which model the realisation is for. The draw is the same; the difference is
        entirely in how the driver consumes it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import input_noise
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> realisation = input_noise(grid, np.random.default_rng(11))
    >>> realisation.model, realisation.seed
    ('input', 11)
    """

    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"]
    seed: int | None
    model: NoiseModel

    def __post_init__(self) -> None:
        """Adopt a read-only copy of the spectrum, so a realisation cannot drift."""
        frozen = np.array(self.spectrum, dtype=np.complex128, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "spectrum", frozen)


def spatial_noise(
    grid: TimeGrid,
    space: RadialGrid | FreeGrid | PlanarGrid,
    rng: np.random.Generator,
    *,
    model: NoiseModel = "input",
    n_pol: int = 1,
) -> NoiseRealisation:
    """Draw one photon per transverse column, polarisation and active frequency bin.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid with angular frequencies in rad/s.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse grid; a :class:`solprop.modes.slab.PlanarGrid` is refused by
        name. Radial spectral weights have units m^-2, while
        Cartesian raw-DFT weights have units m^2, and m on a
        :class:`solprop.grid.LineGrid`.
    rng : numpy.random.Generator
        Explicit source of random phases.
    model : {'input', 'modified'}, optional
        How the driver consumes the draw; both use the same normalization.
    n_pol : int, optional
        Independent polarisation components, 1 or 2.

    Returns
    -------
    NoiseRealisation
        Internal radial spectrum in V m or Cartesian spectrum in V/m, retaining
        the original draw's seed and model.

    Raises
    ------
    NoiseError
        If the model or polarisation count is invalid, or the grid is not a radial
        or Cartesian one (a planar grid has no noise draw yet).

    Notes
    -----
    A column's energy is ``epsilon_0 c w_k / 2`` times the temporal spectral
    energy functional. Dividing the existing modal draw by the square root of
    that factor therefore retains exactly one photon per bin in both models.

    On a line uniform in y the same draw is one photon per column and bin **per
    metre of y**: the line's one ``k_y = 0`` column stands for a unit period in y,
    so a column's energy is in J/m. This is a declared convention, not a
    derivation of the vacuum noise of a beam of finite height.
    """
    if model not in ("input", "modified") or n_pol not in (1, 2):
        msg = "spatial noise requires input or modified model and n_pol of 1 or 2"
        raise NoiseError(msg)
    if not isinstance(space, RadialGrid | FreeGrid):
        # A PlanarGrid's columns are slab amplitudes, whose photon measure this draw
        # does not declare, so planar noise is not supported yet.
        msg = (
            f"spatial noise draws on a RadialGrid or a FreeGrid; a "
            f"{type(space).__name__} has no declared photon measure, and a planar "
            "run carries no noise yet"
        )
        raise NoiseError(msg)
    space.validate()
    draw = _draw(grid, rng, model, space.n_columns, n_pol)
    scale = np.sqrt(0.5 * EPS0 * C * space.spectral_weights)
    return NoiseRealisation(draw.spectrum / scale[:, None, None], draw.seed, model)


def _seed_of(rng: np.random.Generator) -> int | None:
    """Return the entropy a generator was created with, when it can be recovered.

    A generator built by :func:`numpy.random.default_rng` carries its seed sequence, and
    an unseeded one carries the operating system entropy it drew, so in both cases the
    realisation can be reproduced from the recorded value. A generator built directly
    from a bit generator need not carry one, and a seed given as a sequence of integers
    is not recorded.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator to interrogate.

    Returns
    -------
    int or None
        The entropy, or ``None`` when it is absent or not a single integer.
    """
    # getattr rather than attribute access: the annotated type of ``seed_seq`` is the
    # ISeedSequence protocol, which does not declare ``entropy``.
    seed_seq = getattr(rng.bit_generator, "seed_seq", None)
    entropy = getattr(seed_seq, "entropy", None)
    return entropy if isinstance(entropy, int) else None


def _amplitude(grid: TimeGrid) -> Float[np.ndarray, " n_omega"]:
    """Return the one-photon-per-bin spectral amplitude on the grid's active band.

    ``|E_k| = sqrt(hbar omega_k / d_omega) sqrt(2 pi) / (2 dt)`` on a field grid and the
    same without the factor of two on an envelope grid
    (``docs/dev/conventions.md`` §4). The two forms differ because the field grid stores
    a one-sided spectrum whose analytic signal doubles every interior bin, and they pair
    exactly with the two prefactors of
    :func:`solprop.fourier.spectral_density`, so that ``S(omega_k) d_omega = hbar
    omega_k``.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid to draw on.

    Returns
    -------
    numpy.ndarray
        Amplitude per bin, shape ``(n_omega,)``, zero outside the active band.
    """
    active = grid.active_mask
    amplitude = np.zeros(grid.n_omega)
    scale = np.sqrt(2.0 * np.pi) / grid.dt
    if grid.kind == "field":
        scale = 0.5 * scale
    amplitude[active] = scale * np.sqrt(HBAR * grid.omega[active] / grid.d_omega)
    return amplitude


def _draw(
    grid: TimeGrid,
    rng: np.random.Generator,
    model: NoiseModel,
    n_modes: int,
    n_pol: int,
) -> NoiseRealisation:
    """Draw a shot-noise realisation; the one path both models share.

    The phase is drawn over the whole spectral axis rather than over the active band
    alone, so that widening or narrowing the band does not shift the random stream and a
    seed keeps meaning the same realisation on the same grid size.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid to draw on.
    rng : numpy.random.Generator
        Source of the random phase.
    model : {'input', 'modified'}
        Label to record on the realisation.
    n_modes : int
        Number of modes, at least 1. Each gets an independent realisation.
    n_pol : int
        Number of polarisation components, 1 or 2. Each gets an independent realisation,
        because the two transverse components are separate field modes.

    Returns
    -------
    NoiseRealisation
        The realisation, with its seed and model recorded.

    Raises
    ------
    NoiseError
        If ``n_modes`` or ``n_pol`` is smaller than one.
    """
    for name, count in (("n_modes", n_modes), ("n_pol", n_pol)):
        if count < 1:
            msg = f"{name} must be a positive count, got {count!r}"
            raise NoiseError(msg)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_modes, n_pol, grid.n_omega))
    spectrum = _amplitude(grid) * np.exp(1j * phase)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s shot noise: %d modes x %d polarisations, %d of %d bins active",
            model,
            n_modes,
            n_pol,
            int(np.count_nonzero(grid.active_mask)),
            grid.n_omega,
        )
    return NoiseRealisation(spectrum=spectrum, seed=_seed_of(rng), model=model)


def input_noise(
    grid: TimeGrid, rng: np.random.Generator, *, n_modes: int = 1, n_pol: int = 1
) -> NoiseRealisation:
    """Draw the traditional input-noise realisation, added to the field at ``z = 0``.

    This is the model of the supercontinuum literature: one photon per mode per bin with
    a random phase, superposed on the input pulse and then propagated like any other
    field. It seeds modulation instability and spontaneous processes correctly wherever
    the seeding happens near the input, and it costs nothing during the run; its
    weakness is that dispersion phase-matches the propagated background into four-wave
    mixing that should not be there.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid to draw on; its ``kind`` selects the amplitude and its active mask decides
        which bins are populated.
    rng : numpy.random.Generator
        Source of the random phase. Build it with :func:`numpy.random.default_rng` so
        that its seed is recorded on the realisation.
    n_modes : int, optional
        Number of modes, at least 1, each with an independent realisation.
    n_pol : int, optional
        Number of polarisation components, 1 or 2, each with an independent realisation.

    Returns
    -------
    NoiseRealisation
        The realisation, with ``model = 'input'``.

    Raises
    ------
    NoiseError
        If ``n_modes`` or ``n_pol`` is smaller than one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.noise import input_noise, photons_per_bin
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    >>> realisation = input_noise(grid, np.random.default_rng(3))
    >>> photons = photons_per_bin(grid, realisation.spectrum)[0, 0]
    >>> bool(np.allclose(photons[grid.active_mask], 1.0, rtol=1e-12, atol=0.0))
    True
    """
    return _draw(grid, rng, "input", n_modes, n_pol)


def modified_noise(
    grid: TimeGrid, rng: np.random.Generator, *, n_modes: int = 1, n_pol: int = 1
) -> NoiseRealisation:
    """Draw the modified-model realisation, injected as a source at every step.

    The draw is identical to :func:`input_noise` — the same amplitudes, the same mask,
    the same stream for the same seed — and the two differ only in how the driver uses
    the result. Here the realisation is a source term added inside the nonlinear
    operator at every stage, advanced by the unitary part of the cumulative linear
    operator so that it disperses but neither grows nor decays (see
    ``docs/dev/decisions.md``); holding it constant in z would be exact only in the
    dispersion-free gain model it came from. Expect roughly eight times as many
    integrator steps as the input model, because the source is broadband and never
    decays.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid to draw on.
    rng : numpy.random.Generator
        Source of the random phase.
    n_modes : int, optional
        Number of modes, at least 1, each with an independent realisation.
    n_pol : int, optional
        Number of polarisation components, 1 or 2, each with an independent realisation.

    Returns
    -------
    NoiseRealisation
        The realisation, with ``model = 'modified'``.

    Raises
    ------
    NoiseError
        If ``n_modes`` or ``n_pol`` is smaller than one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import input_noise, modified_noise
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> one = input_noise(grid, np.random.default_rng(5))
    >>> other = modified_noise(grid, np.random.default_rng(5))
    >>> bool(np.array_equal(one.spectrum, other.spectrum)), other.model
    (True, 'modified')
    """
    return _draw(grid, rng, "modified", n_modes, n_pol)


def transformed_noise(
    realisation: NoiseRealisation,
    transform: Complex[np.ndarray, "n_modes_out n_modes_in"],
) -> NoiseRealisation:
    r"""Carry one draw into a declared change of channel basis, covariantly.

    The shot-noise draw is one photon per channel per bin with a fixed modulus and a
    random phase, and that is a statement about an **orthonormal true-power** basis:
    it is where "one photon in this channel" means an energy of ``hbar omega``. A run
    in a declared unitary basis therefore draws **once, in the reference basis, with
    the same seed and the same stream**, and this function carries that one
    realisation into the run's coordinates. It is not a second draw and does not
    touch the generator.

    **The rule, and why it is the inverse.** With ``U`` the declared transform, whose
    columns are the new basis vectors written in the old ones
    (:class:`solprop.modes.guide.ChannelBasis`), a physical field with
    reference-basis amplitudes ``c`` has declared-basis amplitudes ``d = U.H c``: the
    fields carry ``U`` and the amplitudes carry its inverse, which for a unitary
    ``U`` is ``U.H``. So the realisation is multiplied on its channel axis by ``U.H``.

    **What is conserved and what is not.** ``sum_m |d_m|^2 = c.H U U.H c = sum_n
    |c_n|^2`` for a unitary ``U``, so the **total** photon number in each bin is
    exactly what it was; the per-channel numbers are redistributed, and the moduli
    are no longer all equal. That is the physics the A2 companion names -- "fixed
    modulus random phases are not invariant in distribution under arbitrary unitary
    mixing" -- and it is why the draw is made once, in the basis it is defined in,
    rather than again in each basis a run chooses.

    Parameters
    ----------
    realisation : NoiseRealisation
        A draw made in the orthonormal true-power reference basis, whose spectrum
        has one row per reference channel.
    transform : numpy.ndarray
        The declared unitary ``U``, shape ``(n_modes, n_modes)``, dimensionless.

    Returns
    -------
    NoiseRealisation
        The same realisation in the declared coordinates, with its seed and model
        unchanged, so that reproducibility and the resume key are untouched.

    Raises
    ------
    NoiseError
        If the transform is not a square matrix of the realisation's own channel
        count.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.noise import input_noise, photons_per_bin, transformed_noise
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
    >>> drawn = input_noise(grid, np.random.default_rng(3), n_modes=2)
    >>> turned = transformed_noise(drawn, np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2))
    >>> before = photons_per_bin(grid, drawn.spectrum).sum(axis=(0, 1))
    >>> after = photons_per_bin(grid, turned.spectrum).sum(axis=(0, 1))
    >>> bool(np.allclose(before, after, rtol=1e-12, atol=0.0)), turned.seed
    (True, 3)
    """
    matrix = np.asarray(transform)
    modes = realisation.spectrum.shape[0]
    if matrix.ndim != 2 or matrix.shape != (modes, modes):
        msg = (
            f"a noise realisation of {modes} channel(s) is carried by a square "
            f"transform of that width, got shape {matrix.shape}"
        )
        raise NoiseError(msg)
    turned = np.einsum("mn,npk->mpk", matrix.conj().T, realisation.spectrum)
    logger.debug(
        "carried a %s shot-noise draw of %d channel(s) into a declared basis",
        realisation.model,
        modes,
    )
    return NoiseRealisation(turned, realisation.seed, realisation.model)


def photons_per_bin(
    grid: TimeGrid, spectrum: Complex[np.ndarray, "*batch n_omega"]
) -> Float[np.ndarray, "*batch n_omega"]:
    """Return the photon number in each frequency bin of a stored spectrum.

    ``n_k = S(omega_k) d_omega / (hbar omega_k)`` with ``S`` the spectral energy density
    of :func:`solprop.fourier.spectral_density`. For a shot-noise realisation this is 1
    in every active bin, which is the definition the amplitudes come from; for a pulse
    it says how far above the quantum floor the field sits, bin by bin.

    Bins at or below zero frequency have no photon number and are reported as zero.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid the spectrum belongs to.
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(..., n_omega)``, in the square root of watts.

    Returns
    -------
    numpy.ndarray
        Photon number per bin, dimensionless, of the same shape as ``spectrum``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import input_noise, photons_per_bin
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> realisation = input_noise(grid, np.random.default_rng(2))
    >>> photons = photons_per_bin(grid, realisation.spectrum)
    >>> float(round(photons[0, 0][grid.active_mask].mean(), 12))
    1.0
    """
    density = fourier.spectral_density(spectrum, grid.dt, kind=grid.kind)
    photons = np.zeros_like(density)
    np.divide(
        density * grid.d_omega,
        HBAR * grid.omega,
        out=photons,
        where=grid.omega > 0.0,
    )
    return photons
