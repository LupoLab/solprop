# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Adapt optmat's gas Raman responses onto solprop's existing Raman terms (SEP 3).

optmat gives a molecular gas's, or a gas mixture's, delayed response as a sum of
lines,

.. math::

    h(\tau; N) = \Theta(\tau) \sum_k A_k \sin(\Omega_k(N)\tau)\, e^{-\Gamma_k(N)\tau},
    \qquad \Gamma_k = \pi\, \Delta\nu_k(N),

for one composition, one temperature and one pump wavelength, with the amplitudes in
m^5 V^-2 s^-1 per molecule of the **whole** composition: a constituent's lines carry
its mole fraction, so a mixture's response is driven by its total density. That is
solprop's own convention (:mod:`solprop.nonlinear.raman`), so no unit changes here.

Lines that share a width law and a shift form one **damping group** (optmat's
``groups()``). Each group and tensor family becomes one existing
:class:`~solprop.nonlinear.raman.RamanTensorTerm`: an undamped row
``u(t) = sum_k A_k sin(Omega_k t)`` (:class:`RamanGroupLines`) and the group's decay
rate as a function of density (:class:`GroupDephasing`), which the translation layer
composes with the density path. The isotropic amplitudes become an isotropic term and
the anisotropic ones a rank-two term, so a copolarised probe sees ``a + b`` and a
cross-polarised one ``a`` (Hellwarth's functions). The mapping is exact for every gas
whose lines do not shift, at any density path. H2's Q lines shift with density: at a
constant density their row is built at that density, which is exact; along a
gradient they are refused unless the selection names the approximation
``raman_shift="reference-density"``, whose phase bound is recorded.

A mixture's response is built once, for the composition of all its labelled
populations, and is driven, and damped, by the **summed** density of those labels
(:attr:`OptmatRamanResponse.labels`). optmat builds a mixture's response only from the
collision partners whose broadening it has curated, and a mixture with any other
partner is refused here by name, naming the pair (SEP 3, D16). The derivation and the
limits are on the manual page ``docs/guide/media/raman-data.md``, "Raman responses from
optmat".
"""

import functools
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

import numpy as np
from jaxtyping import Float

from solprop.constants import AMAGAT, ROOM_TEMPERATURE
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.nonlinear.raman import RamanResponse, RamanTensorTerm
from solprop.physdata import _optmat
from solprop.physdata.mixtures import Mixture
from solprop.physdata.raman_types import Dephasing, Linewidth, MolecularOptions
from solprop.physdata.selection import (
    RAMAN_POLICIES,
    MaterialRecord,
    MaterialSelection,
)
from solprop.physdata.species import built_in_population
from solprop.profiles import Sum, ZProfile, _probe_positions

__all__ = [
    "COMPOSITION_TOLERANCE",
    "GroupDephasing",
    "OptmatRamanResponse",
    "RamanAssembly",
    "RamanGroupLines",
    "RamanShift",
    "gas_raman",
    "mixture_raman",
    "raman_status",
]

logger = logging.getLogger(__name__)

COMPOSITION_TOLERANCE: Final = 1e-2
"""Largest relative change of a constituent's mole fraction along the run.

A response is built for one composition. Populations whose densities are proportional
keep it; the partial-pressure rule (SEP 3, D17) makes them proportional only to the
gases' differing non-ideality, which moves argon's fraction of dry air by
1.6 x 10^-3 of itself between 10 bar and vacuum at 293.15 K (nitrogen's by 8 x 10^-5).
Beyond this tolerance the composition varies along z and the response is refused by
name rather than built at a composition the run does not have.
"""

_PROBES: Final = 65
"""Evenly spaced positions over the run, beside the breakpoints, at which the
composition, the densities checked against the response's range and the shift's
bound are sampled."""

_Share = Literal["isotropic", "anisotropic"]
"""The tensor family of a row: the lines' isotropic or anisotropic amplitudes."""


@dataclass(frozen=True)
class GroupDephasing(Dephasing):
    r"""The decay of one optmat damping group's coherence with the total density.

    The rate is optmat's group law,
    :math:`\Gamma(N) = \gamma_c + \min(\gamma_d / N, \gamma_D) + \gamma_\beta N`,
    evaluated by optmat itself (``RamanGroup.decay_rate``). It is defined for every
    :math:`N \ge 0`: the Dicke term is capped at the forward-scattering Doppler width,
    so at vacuum it is :math:`\pi(c + \Delta\nu_D)` with a Dicke term and
    :math:`\pi c` without one -- zero for nitrogen's and oxygen's lines, whose widths
    are pure broadening. No floor or clamp is applied here.

    Attributes
    ----------
    linewidth : Linewidth
        The group's width law ``c + d/n + beta n`` in Hz as optmat curates it, without
        the Doppler cap, for the record.
    doppler_width : float
        The forward-scattering Doppler width in Hz that caps the Dicke term; zero
        without one.
    group : optmat.gas.raman_response.RamanGroup
        optmat's group, which evaluates the law. A live handle, not recorded.

    Examples
    --------
    >>> from optmat.gas.raman_response import raman_response
    >>> from solprop.nonlinear.optmat_raman import GroupDephasing
    >>> group = raman_response("H2", 293.15).groups()[0]
    >>> dephasing = GroupDephasing.of(group, raman_response("H2", 293.15))
    >>> dephasing.decay_rate(0.0) == float(group.decay_rate(0.0)) > 0.0
    True
    """

    doppler_width: float
    group: _optmat.Untyped = field(
        compare=False, repr=False, metadata={"record": False}
    )

    @classmethod
    def of(cls, group: _optmat.Untyped, response: _optmat.Untyped) -> "GroupDephasing":  # noqa: UP037 - the hook resolves class names eagerly
        """Return the dephasing of one group of an optmat response.

        Parameters
        ----------
        group : optmat.gas.raman_response.RamanGroup
            The group.
        response : optmat.gas.raman_response.GasRamanResponse
            The response it belongs to, whose line arrays carry the law in Hz.

        Returns
        -------
        GroupDephasing
            The group's decay law.
        """
        first = int(group.lines[0])
        return cls(
            Linewidth(
                constant=float(response.width_constant[first]),
                inverse_density=float(response.width_dicke[first]),
                density=float(response.width_broadening[first]),
            ),
            float(response.doppler_width[first]),
            group,
        )

    def decay_rate(self, density: float) -> float:
        """Return the group's coherence decay rate at a total number density.

        Parameters
        ----------
        density : float
            Total number density of the composition in m^-3, finite and >= 0.

        Returns
        -------
        float
            ``Gamma = pi FWHM`` in s^-1, finite at every density, vacuum included.

        Raises
        ------
        MaterialError
            If the density is negative or not finite.
        """
        if not (math.isfinite(density) and density >= 0.0):
            msg = (
                f"a Raman dephasing needs a finite, non-negative number density, got "
                f"{density} m^-3"
            )
            raise MaterialError(msg)
        return float(self.group.decay_rate(float(density)))


@dataclass(frozen=True)
class RamanGroupLines:
    """The undamped row of one damping group and tensor family.

    ``u(t) = sum_k A_k sin(Omega_k t)`` for ``t > 0`` and zero otherwise, in
    m^5 V^-2 s^-1: a :class:`~solprop.nonlinear.raman.RamanModel` whose dephasing is
    the group's (:class:`GroupDephasing`), which the translation layer applies per
    density. The translation's causal taper multiplies the row; nothing here does.

    Attributes
    ----------
    frequencies : numpy.ndarray
        The lines' angular frequencies in rad/s, shape ``(n_lines,)``: at zero
        density for an unshifted group, and at the density the shift is held at for
        a shifted one (:class:`RamanShift`).
    amplitudes : numpy.ndarray
        The lines' amplitudes of this tensor family in m^5 V^-2 s^-1, per molecule of
        the composition, shape ``(n_lines,)``.
    dephasing : GroupDephasing
        The group's decay law.
    line_ids : tuple of str
        optmat's stable identifier of each line, ``species/band/branch/J``.
    share : {'isotropic', 'anisotropic'}
        Which of the lines' two amplitude shares the row carries.

    Raises
    ------
    MaterialError
        If the arrays disagree in length, a frequency is not positive and finite, or
        an amplitude is not finite.
    """

    # Independent axis names, so that __post_init__ enforces the relation where no
    # runtime type checker is installed (AGENTS.md, "Types and arrays").
    frequencies: Float[np.ndarray, " n_lines"]
    amplitudes: Float[np.ndarray, " n_amplitudes"]
    dephasing: GroupDephasing
    line_ids: tuple[str, ...]
    share: _Share

    def __post_init__(self) -> None:
        """Check the row's arrays and freeze them.

        Raises
        ------
        MaterialError
            If the arrays disagree in length or hold invalid values.
        """
        frequencies = np.array(self.frequencies, dtype=float, copy=True)
        amplitudes = np.array(self.amplitudes, dtype=float, copy=True)
        if not frequencies.size == amplitudes.size == len(self.line_ids) > 0:
            msg = (
                f"a Raman group row needs one frequency, amplitude and identifier per "
                f"line, got {frequencies.size}, {amplitudes.size} and "
                f"{len(self.line_ids)}"
            )
            raise MaterialError(msg)
        if not np.all(np.isfinite(frequencies) & (frequencies > 0.0)):
            msg = f"Raman line frequencies must be positive and finite: {self.line_ids}"
            raise MaterialError(msg)
        if not np.all(np.isfinite(amplitudes)):
            msg = f"Raman line amplitudes must be finite: {self.line_ids}"
            raise MaterialError(msg)
        for name, array in (("frequencies", frequencies), ("amplitudes", amplitudes)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "line_ids", tuple(str(i) for i in self.line_ids))

    def undamped(self, t: Float[np.ndarray, " n_to"]) -> Float[np.ndarray, " n_to"]:
        """Return the undamped row on a time axis, in m^5 V^-2 s^-1.

        Parameters
        ----------
        t : numpy.ndarray
            Delays in s, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            ``sum_k A_k sin(Omega_k t)`` where ``t > 0`` and zero elsewhere, shape
            ``(n_to,)``.
        """
        times = np.asarray(t, dtype=float)
        total = np.zeros_like(times)
        # One line at a time, as for the rotational ladder: an (n_lines, n_to)
        # intermediate on a 2^17-sample fine grid would cost megabytes for nothing.
        for amplitude, frequency in zip(self.amplitudes, self.frequencies, strict=True):
            total += amplitude * np.sin(frequency * times)
        return np.where(times > 0.0, total, 0.0)


@dataclass(frozen=True)
class RamanShift:
    r"""How a density-dependent line shift is held in a response's rows.

    Attributes
    ----------
    policy : {'constant-density', 'reference-density'}
        ``constant-density``: the density is constant, and the rows are built at it,
        exactly. ``reference-density``: the selection's named approximation, the
        rows held at the density-weighted reference density along a gradient.
    density : float
        The total number density in m^-3 the shifted rows are built at.
    frequency_error : float
        :math:`\max_g |\sigma_g| \max_z |N(z) - N_\mathrm{ref}|` in rad/s, the largest
        error of a line's angular frequency over the run; zero at constant density.
        Over a window :math:`T_w` the phase error is bounded by its product with
        :math:`T_w` (:meth:`phase_bound`).
    """

    policy: Literal["constant-density", "reference-density"]
    density: float
    frequency_error: float

    def phase_bound(self, window: float) -> float:
        """Return the largest phase error of a shifted line over a time window.

        Parameters
        ----------
        window : float
            The time window in s, positive.

        Returns
        -------
        float
            ``frequency_error * window`` in rad.

        Examples
        --------
        >>> from solprop.nonlinear.optmat_raman import RamanShift
        >>> RamanShift("reference-density", 1e25, 2e9).phase_bound(1e-12)
        0.002
        """
        return self.frequency_error * window

    def to_dict(self) -> dict[str, object]:
        """Return the record as plain values.

        Returns
        -------
        dict
            ``policy``, ``density_m-3`` and ``frequency_error_rad_s``.
        """
        return {
            "policy": self.policy,
            "density_m-3": self.density,
            "frequency_error_rad_s": self.frequency_error,
        }


@dataclass(frozen=True)
class RamanAssembly:
    """Record how an optmat Raman response was mapped onto solprop's terms.

    Attributes
    ----------
    material : str
        What optmat's response is of: a gas, or the species of a mixture joined by
        ``+``.
    composition : tuple of tuple of (str, float)
        The species and the mole fractions the response was built for.
    temperature : float
        The reference temperature in K: the populations and the widths' temperature
        laws.
    wavelength : float
        The pump wavelength in m at which the vibrational strengths are dispersed.
    groups : int
        The damping groups the response keeps.
    terms : int
        Its rows after the tensor split, all-zero rows dropped.
    shift : RamanShift or None
        How a line shift is held, or ``None`` when no kept group shifts.
    densities : tuple of float or None
        The smallest and largest non-zero total densities in m^-3 declared to the
        response's range check; the smallest is 0.0 when the path falls to vacuum,
        whose every density above zero is then declared. ``None`` when none was
        declared.
    composition_spread : float
        The largest relative departure of a constituent's mole fraction from the
        composition along the run; zero for a pure gas.
    """

    material: str
    composition: tuple[tuple[str, float], ...]
    temperature: float
    wavelength: float
    groups: int
    terms: int
    shift: RamanShift | None = None
    densities: tuple[float, float] | None = None
    composition_spread: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return the record as plain values.

        Returns
        -------
        dict
            Every field, the shift as its own record.
        """
        return {
            "material": self.material,
            "composition": [[name, fraction] for name, fraction in self.composition],
            "temperature_K": self.temperature,
            "wavelength_m": self.wavelength,
            "groups": self.groups,
            "terms": self.terms,
            "shift": None if self.shift is None else self.shift.to_dict(),
            "densities_m-3": None if self.densities is None else list(self.densities),
            "composition_spread": self.composition_spread,
        }


@dataclass(frozen=True, init=False)
class OptmatRamanResponse(RamanResponse):
    """A Raman response adapted from optmat, driven by its composition's density.

    A :class:`~solprop.nonlinear.raman.RamanResponse` whose terms are optmat's damping
    groups, one per group and tensor family, and which records how it was built.
    Build one with :func:`mixture_raman` or :func:`gas_raman`.

    Parameters
    ----------
    models : None, optional
        Refused: the terms are declared by `tensor_terms`.
    fraction : None, optional
        Refused: the response is absolute.
    thg : bool, optional
        Whether a field grid keeps the third harmonic. Default ``True``.
    species : str or None, optional
        The leaf's label.
    tensor_terms : tuple of RamanTensorTerm
        The group rows.
    labels : tuple of str, optional
        The composition's labels.
    assembly : RamanAssembly or None, optional
        How the response was mapped onto terms.
    source : optmat.gas.raman_response.GasRamanResponse or None, optional
        The response optmat built.

    Attributes
    ----------
    tensor_terms : tuple of RamanTensorTerm
        Isotropic and rank-two terms, one per damping group and tensor family.
    fraction : None
        Always ``None``: the response is absolute.
    thg : bool
        Whether a field grid keeps the third harmonic of the drive.
    species : str or None
        The label the compiled leaf carries: the first Raman-active label of the
        composition, or ``None`` for a single population.
    labels : tuple of str
        The labelled populations of the composition, whose **summed** density drives
        the response and sets its decay rates; empty for a response not bound to a
        mixture, which takes the run's one density.
    assembly : RamanAssembly
        How the response was mapped onto terms.
    source : optmat.gas.raman_response.GasRamanResponse
        optmat's response. A live handle, recorded through :attr:`material_records`.

    Raises
    ------
    MaterialError
        If a fraction is given, or a term's model is not an optmat group row.
    """

    labels: tuple[str, ...] = ()
    assembly: RamanAssembly | None = None
    source: _optmat.Untyped = field(
        default=None, compare=False, repr=False, metadata={"record": False}
    )

    def __init__(  # noqa: PLR0913 - the base class's fields and this class's three
        self,
        models: tuple[object, ...] | None = None,
        fraction: float | None = None,
        thg: bool = True,
        species: str | None = None,
        *,
        tensor_terms: tuple[RamanTensorTerm, ...] | None = None,
        labels: tuple[str, ...] = (),
        assembly: RamanAssembly | None = None,
        source: _optmat.Untyped = None,
    ) -> None:
        """Build the base response and attach the composition and its record."""
        if models is not None or fraction is not None:
            msg = (
                "an optmat Raman response is built from tensor terms and is absolute; "
                "it takes neither legacy models nor a fraction"
            )
            raise MaterialError(msg)
        super().__init__(None, None, thg, species, tensor_terms=tensor_terms)
        if not all(isinstance(t.model, RamanGroupLines) for t in self.tensor_terms):
            msg = "an optmat Raman response's terms are optmat group rows"
            raise MaterialError(msg)
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "assembly", assembly)
        object.__setattr__(self, "source", source)

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the record of optmat's response, for the run's manifest.

        Returns
        -------
        tuple of MaterialRecord
            The response's identity, or none for a response without its source.
        """
        return () if self.source is None else (_optmat.record(self.source),)


@dataclass(frozen=True)
class _Request:
    """One response to assemble: the composition, its state and what to keep.

    Attributes
    ----------
    fractions : Mapping of str to float
        Species to mole fraction, each non-zero.
    temperature : float
        Reference temperature in K.
    wavelength : float
        Pump wavelength in m.
    selection : MaterialSelection
        The run's selection.
    keep : Mapping of str to MolecularOptions
        The species whose lines are kept, with their term flags.
    dephasing : Mapping of str to Dephasing
        Supplied width laws by species.
    densities : numpy.ndarray or None
        The total density at the probes, m^-3, or ``None`` when not known.
    constant : float or None
        The total density in m^-3 when it is constant along the run.
    reference : float or None
        The density-weighted reference density in m^-3 of a varying path.
    spread : float
        The composition's relative spread along the run.
    where : str
        What the response is for, in messages.
    """

    fractions: Mapping[str, float]
    temperature: float
    wavelength: float
    selection: MaterialSelection
    keep: Mapping[str, MolecularOptions]
    dephasing: Mapping[str, Dephasing]
    densities: Float[np.ndarray, " n_probes"] | None
    constant: float | None
    reference: float | None
    spread: float
    where: str


def _gas_kind(name: str) -> Literal["atom", "molecule"] | None:
    """Return whether optmat knows a species as an atom or a molecule.

    Parameters
    ----------
    name : str
        A species name.

    Returns
    -------
    {'atom', 'molecule'} or None
        The kind, or ``None`` for a name optmat's gas registry does not curate.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.gases import as_gas, info  # noqa: PLC0415 - as above

    try:
        return "atom" if info(as_gas(name)).atomic else "molecule"
    except OptmatError:
        return None


@functools.cache
def raman_status(gas: str) -> str:
    """Return the status of optmat's curated Raman response of a built-in gas.

    The status is a property of the curated data, not of the state: it is read from
    optmat's curated choices (``raman_response_status``) without building the lines.
    The capillary driver's ``raman=None`` takes it to decide whether a fill carries a
    Raman response by default.

    Parameters
    ----------
    gas : str
        A built-in gas's name.

    Returns
    -------
    str
        ``"atom"`` for a species optmat knows as an atom (no Raman-active mode);
        optmat's status of the response otherwise (``"validated"``, or
        ``"provisional"`` for N2O's bands, SEP 3, D13); ``"none"`` for a molecule
        optmat has no response of.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_raman import raman_status
    >>> raman_status("N2"), raman_status("N2O"), raman_status("He")
    ('validated', 'provisional', 'atom')
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.raman_response import (  # noqa: PLC0415 - as above
        raman_response_status,
    )

    if _gas_kind(gas) != "molecule":
        return "atom"
    try:
        status = raman_response_status(gas)
    except OptmatError:
        return "none"
    return "none" if status is None else str(status[0])


def _build(request: _Request) -> tuple[_optmat.Untyped, list[warnings.WarningMessage]]:
    """Return optmat's response for a request, with the warnings its build raised.

    Parameters
    ----------
    request : _Request
        The composition, state and selection.

    Returns
    -------
    tuple
        The response optmat built, and the warnings it raised while being built.

    Raises
    ------
    MaterialError
        If optmat refuses the composition: naming each Raman-active constituent and
        partner whose broadening it has not curated, or with optmat's reason.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.raman_response import raman_response  # noqa: PLC0415 - as above

    chosen = request.selection.raman
    if not isinstance(chosen, str):
        return _supplied(chosen, request), []
    if chosen not in RAMAN_POLICIES:
        msg = (
            f"{request.where}: MaterialSelection.raman={chosen!r} names no gas Raman "
            "response. On a gas run 'recommended' is optmat's curated response and "
            "'provisional' also admits its provisional bands; a silica bundle name "
            "(such as 'tensor') applies to a silica fibre's GNLSE only, and optmat "
            "names a gas response by its gas or composition, which the run supplies. "
            "Select 'recommended', 'provisional' or an optmat response object"
        )
        raise MaterialError(msg)
    composition = dict(request.fractions)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            response = raman_response(
                composition if len(composition) > 1 else next(iter(composition)),
                request.temperature,
                wavelength=request.wavelength,
            )
        except OptmatError as error:
            _refuse_partners(request, error)
    return response, list(caught)


def _refuse_partners(request: _Request, error: Exception) -> NoReturn:
    """Refuse a composition optmat cannot build, naming the pairs it lacks.

    Each Raman-active constituent is tried against each partner on its own, so that
    the message names the pair optmat has no broadening for (SEP 3, D16).

    Parameters
    ----------
    request : _Request
        The refused composition.
    error : Exception
        The refusal of the whole composition by optmat.

    Raises
    ------
    MaterialError
        Always.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.raman_response import raman_response  # noqa: PLC0415 - as above

    active = [name for name in request.fractions if _gas_kind(name) == "molecule"]
    pairs = []
    for name in active:
        for partner in request.fractions:
            if partner == name:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    raman_response(
                        {name: 0.5, partner: 0.5},
                        request.temperature,
                        wavelength=request.wavelength,
                    )
                except OptmatError:
                    pairs.append(f"{name} by {partner}")
    if pairs:
        msg = (
            f"{request.where}: optmat has curated no Raman broadening of "
            f"{', '.join(pairs)}, so solprop builds no Raman response for this "
            "mixture and does not substitute self-broadening for the missing partner "
            "(SEP 3, D16). Pass raman=False, or a mixture of partners optmat curates"
        )
        raise MaterialError(msg) from error
    msg = f"{request.where}: optmat refuses the Raman response: {error}"
    raise MaterialError(msg) from error


def _supplied(response: object, request: _Request) -> _optmat.Untyped:
    """Check a caller's optmat response against the run it is used in.

    A response fixes its composition, temperature and pump wavelength; one built for
    another state is refused rather than used where it does not apply.

    Parameters
    ----------
    response : object
        ``MaterialSelection.raman``: an optmat object with an identity.
    request : _Request
        The run's composition and state.

    Returns
    -------
    optmat.gas.raman_response.GasRamanResponse
        The response, as it is.

    Raises
    ------
    MaterialError
        If it is not a gas Raman response, or its composition, temperature or pump
        wavelength is not the run's.
    """
    model = _optmat.untyped(response)
    if _optmat.record(response).kind != "raman_response":
        msg = (
            f"{request.where}: MaterialSelection.raman is an optmat "
            f"{_optmat.record(response).kind} model, not a gas Raman response; a "
            "silica bundle applies to a silica fibre's GNLSE only"
        )
        raise MaterialError(msg)
    given = dict(model.composition)
    wanted = dict(request.fractions)
    same_composition = set(given) == set(wanted) and all(
        math.isclose(given[name], wanted[name], rel_tol=1e-9) for name in wanted
    )
    pump = model.wavelength
    if (
        not same_composition
        or not math.isclose(model.temperature, request.temperature, rel_tol=1e-12)
        or pump is None
        or not math.isclose(pump, request.wavelength, rel_tol=1e-12)
    ):
        msg = (
            f"{request.where}: the supplied Raman response is for {given} at "
            f"{model.temperature:g} K and a pump of {pump} m; the run needs {wanted} "
            f"at {request.temperature:g} K and {request.wavelength:g} m. Build it with "
            "raman_response(composition, temperature, wavelength=carrier)"
        )
        raise MaterialError(msg)
    return model


def _with_dephasing(response: _optmat.Untyped, request: _Request) -> _optmat.Untyped:
    """Apply supplied width laws through optmat's own override.

    A component's :class:`~solprop.physdata.raman_types.Dephasing` sets the width law
    ``c + beta n`` of every band of its species (optmat's ``width_law``), so that the
    response's identity records the change (status ``user``). A law with a Dicke term
    is refused: optmat caps that term at a Doppler width the legacy law does not have.

    Parameters
    ----------
    response : optmat.gas.raman_response.GasRamanResponse
        The response.
    request : _Request
        The supplied laws by species.

    Returns
    -------
    optmat.gas.raman_response.GasRamanResponse
        The response with the laws applied, or as it was without any.

    Raises
    ------
    MaterialError
        If a supplied law has a Dicke term.
    """
    laws: dict[str, tuple[float, float, float, float]] = {}
    for name, dephasing in request.dephasing.items():
        line = dephasing.linewidth
        if line.inverse_density > 0.0:
            msg = (
                f"{request.where}: {name}'s supplied linewidth has a Dicke term, which "
                "optmat caps at a Doppler width the supplied law does not state; pass "
                "MaterialSelection(raman=response.with_parameters(width_law=...)) with "
                "(c, d, beta, cap) instead"
            )
            raise MaterialError(msg)
        bands = sorted(
            {
                b
                for s, b in zip(response.species, response.band, strict=True)
                if s == name
            }
        )
        for band in bands:
            laws[f"{name} {band}"] = (line.constant, 0.0, line.density, 0.0)
    if not laws:
        return response
    return response.with_parameters(width_law=laws)


def _reemit(
    caught: Sequence[warnings.WarningMessage],
    response: _optmat.Untyped,
    *,
    kind: str,
    where: str,
    part: str,
) -> None:
    """Raise optmat's range warnings once per run, and record them.

    Parameters
    ----------
    caught : sequence of warnings.WarningMessage
        What optmat raised.
    response : optmat.gas.raman_response.GasRamanResponse
        The response the warnings are about.
    kind : str
        ``"temperature"`` or ``"density"``: one warning of each per response and run.
    where : str
        What the response is for.
    part : str
        The state declared, for the record.
    """
    from optmat.exceptions import ModelRangeWarning  # noqa: PLC0415 - optmat is lazy

    entry = _optmat.record(response)
    messages = []
    for item in caught:
        if issubclass(item.category, ModelRangeWarning):
            messages.append(str(item.message))
        else:
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )
    if not messages:
        return
    _optmat.record_outside(response, where, tuple(f"{part}: {m}" for m in messages))
    _optmat.warn_once(
        (f"raman-{kind}", entry.digest),
        f"{where}: optmat's Raman response {entry.name} is used outside its "
        f"validated {kind} range ({part}). optmat: {'; '.join(messages)}",
        MaterialRangeWarning,
    )


def _check_range(
    response: _optmat.Untyped, request: _Request
) -> tuple[float, float] | None:
    """Declare the run's non-zero densities to the response's range check.

    The polarisation is exactly zero in vacuum, so only the non-zero densities are
    declared (SEP 3, section 4.4). A path that falls to vacuum between two probes
    covers every density between zero and its last positive probe by continuity, so
    the smallest positive double is declared for that interval too, and the record's
    lower bound is zero: "to vacuum".

    Parameters
    ----------
    response : optmat.gas.raman_response.GasRamanResponse
        The response.
    request : _Request
        The run's densities, at its probes in z order.

    Returns
    -------
    tuple of float or None
        The smallest and largest declared densities in m^-3, the smallest 0.0 for a
        path that reaches vacuum, or ``None`` when none is known.
    """
    if request.densities is None:
        return None
    values = request.densities
    positive = values > 0.0
    # A request with densities has a non-zero one: a fill that is vacuum along the
    # whole run has no composition, and builds no response.
    declared = values[positive]
    high = float(declared.max())
    to_vacuum = bool(np.any(positive[1:] != positive[:-1]))
    low = 0.0 if to_vacuum else float(declared.min())
    if to_vacuum:
        declared = np.append(declared, np.finfo(float).tiny)
    lower = "vacuum" if to_vacuum else f"{low / AMAGAT:.4g}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response.check_densities(declared)
    _reemit(
        caught,
        response,
        kind="density",
        where=request.where,
        part=f"{lower} to {high / AMAGAT:.4g} amagat",
    )
    return low, high


def _kept_groups(response: _optmat.Untyped, request: _Request) -> list[_optmat.Untyped]:
    """Return the damping groups the requested species and term flags keep.

    Parameters
    ----------
    response : optmat.gas.raman_response.GasRamanResponse
        The response.
    request : _Request
        The species to keep and their flags.

    Returns
    -------
    list of optmat.gas.raman_response.RamanGroup
        The kept groups in optmat's order.

    Raises
    ------
    MaterialError
        If a term required by ``True`` is absent, or nothing is left.
    """
    kept = []
    for name, options in request.keep.items():
        bands = {
            b for s, b in zip(response.species, response.band, strict=True) if s == name
        }
        for flag, rotational, what in (
            (options.rotation, True, "rotational"),
            (options.vibration, False, "vibrational"),
        ):
            present = any((band == "rotation") == rotational for band in bands)
            if flag is True and not present:
                msg = (
                    f"{request.where}: optmat's Raman response of {name} has no "
                    f"{what} lines, which MolecularOptions requires"
                )
                raise MaterialError(msg)
    for group in response.groups():
        first = int(group.lines[0])
        name = response.species[first]
        options = request.keep.get(name)
        if options is None:
            continue
        rotational = response.band[first] == "rotation"
        flag = options.rotation if rotational else options.vibration
        if flag is not False:
            kept.append(group)
    if not kept:
        msg = (
            f"{request.where}: no Raman lines are left after the term flags "
            f"{dict(request.keep)}"
        )
        raise MaterialError(msg)
    return kept


def _shift(groups: Sequence[_optmat.Untyped], request: _Request) -> RamanShift | None:
    """Decide the density at which the shifted groups' rows are built.

    Parameters
    ----------
    groups : sequence of optmat.gas.raman_response.RamanGroup
        The kept groups.
    request : _Request
        The run's density path and selection.

    Returns
    -------
    RamanShift or None
        ``None`` when no kept group shifts; the constant density; or, under
        ``raman_shift="reference-density"``, the reference density and its bound.

    Raises
    ------
    MaterialError
        If a group shifts on a varying density without the named approximation, or
        on a density the run does not state.
    """
    shifted = [group for group in groups if group.shifted]
    if not shifted:
        return None
    lines = ", ".join(sorted({i.split("/Q/")[0] for g in shifted for i in g.line_ids}))
    if request.constant is not None:
        return RamanShift("constant-density", request.constant, 0.0)
    if request.reference is None or request.densities is None:
        msg = (
            f"{request.where}: the lines {lines} shift with density, so their rows "
            "need the density they are built at; pass density= for a constant fill, "
            "or build the response from the run's mixture"
        )
        raise MaterialError(msg)
    if request.selection.raman_shift != "reference-density":
        msg = (
            f"{request.where}: the lines {lines} shift with density, and the density "
            "varies along the run, which the existing Raman seam cannot follow (SEP 10 "
            "removes this). Select MaterialSelection(raman_shift='reference-density') "
            "to hold the shift at the density-weighted reference density, with its "
            "phase bound recorded"
        )
        raise MaterialError(msg)
    deviation = float(np.max(np.abs(request.densities - request.reference)))
    error = max(abs(float(group.shift)) for group in shifted) * deviation
    logger.info(
        "%s: the Raman line shift is held at the reference density %.6g m^-3; the "
        "largest line-frequency error along the run is %.3g rad/s",
        request.where,
        request.reference,
        error,
    )
    return RamanShift("reference-density", request.reference, error)


def _terms(
    groups: Sequence[_optmat.Untyped],
    response: _optmat.Untyped,
    shift: RamanShift | None,
) -> tuple[RamanTensorTerm, ...]:
    """Map the kept groups onto one isotropic and one rank-two term each.

    Hellwarth's decomposition gives a copolarised probe ``a + b = iso + aniso`` and
    a cross-polarised one ``a = iso - aniso/2``; solprop's isotropic term contributes
    ``h_v`` to both and its rank-two term ``h_r`` to the first and ``-h_r/2`` to the
    second, so the isotropic amplitudes go to the isotropic term and the
    anisotropic ones to the rank-two term unchanged. An all-zero row is dropped.

    Parameters
    ----------
    groups : sequence of optmat.gas.raman_response.RamanGroup
        The kept groups.
    response : optmat.gas.raman_response.GasRamanResponse
        Their response, for the width laws in Hz.
    shift : RamanShift or None
        The density a shifted group's rows are built at.

    Returns
    -------
    tuple of RamanTensorTerm
        In group order, isotropic before rank-two within a group.
    """
    terms = []
    for group in groups:
        frequencies = np.asarray(group.frequency, dtype=float)
        if group.shifted and shift is not None:
            frequencies = frequencies + float(group.shift) * shift.density
        dephasing = GroupDephasing.of(group, response)
        families: tuple[tuple[_Share, Literal["isotropic", "rank2"]], ...] = (
            ("isotropic", "isotropic"),
            ("anisotropic", "rank2"),
        )
        for share, contraction in families:
            amplitudes = np.asarray(getattr(group, share), dtype=float)
            if not np.any(amplitudes):
                continue
            row = RamanGroupLines(
                frequencies, amplitudes, dephasing, tuple(group.line_ids), share
            )
            terms.append(RamanTensorTerm(row, contraction))
    return tuple(terms)


def _assemble(
    request: _Request, *, thg: bool, labels: tuple[str, ...], leaf: str | None
) -> OptmatRamanResponse:
    """Build, check and map one optmat response onto solprop's terms.

    Parameters
    ----------
    request : _Request
        The composition, state and what to keep.
    thg : bool
        Whether a field grid keeps the third harmonic of the drive.
    labels : tuple of str
        The labels whose summed density drives the response.
    leaf : str or None
        The label the compiled leaf carries.

    Returns
    -------
    OptmatRamanResponse
        The response.

    Raises
    ------
    MaterialError
        If optmat refuses the composition, a provisional response is not admitted,
        or a shift or a term flag cannot be honoured.

    Warns
    -----
    MaterialRangeWarning
        Once per run each, if the temperature or a declared density is outside the
        response's validated range.
    MaterialStatusWarning
        Once per run, for an admitted provisional response.
    """
    built, caught = _build(request)
    response = _with_dephasing(built, request)
    entry = _optmat.record(response)
    if entry.status == "provisional" and request.selection.raman != "provisional":
        msg = (
            f"{request.where}: optmat's Raman response {entry.name} is provisional "
            f"({entry.status_reason}). It is admitted only with "
            "MaterialSelection(raman='provisional'), which warns once and is recorded "
            "(SEP 3, D13)"
        )
        raise MaterialError(msg)
    _optmat.check_status(response, request.where)
    _reemit(
        caught,
        response,
        kind="temperature",
        where=request.where,
        part=f"{request.temperature:.6g} K",
    )
    densities = _check_range(response, request)
    groups = _kept_groups(response, request)
    shift = _shift(groups, request)
    terms = _terms(groups, response, shift)
    assembly = RamanAssembly(
        material=entry.material,
        composition=tuple((str(s), float(x)) for s, x in response.composition),
        temperature=float(request.temperature),
        wavelength=float(request.wavelength),
        groups=len(groups),
        terms=len(terms),
        shift=shift,
        densities=densities,
        composition_spread=request.spread,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s: optmat Raman response %s, %d damping groups as %d terms",
            request.where,
            entry.name,
            assembly.groups,
            assembly.terms,
        )
    return OptmatRamanResponse(
        thg=thg,
        species=leaf,
        tensor_terms=terms,
        labels=labels,
        assembly=assembly,
        source=response,
    )


def _carrier(
    selection: MaterialSelection, wavelength: float | None, where: str
) -> float:
    """Return the pump wavelength the vibrational strengths are dispersed at.

    Parameters
    ----------
    selection : MaterialSelection
        The run's selection; an explicit carrier is its deliberate choice.
    wavelength : float or None
        The run's grid reference wavelength in m.
    where : str
        What the response is for.

    Returns
    -------
    float
        ``selection.carrier_wavelength``, else `wavelength`, in m.

    Raises
    ------
    MaterialError
        If neither is given.
    """
    carrier = selection.carrier_wavelength
    if carrier is None:
        carrier = wavelength
    if carrier is None:
        msg = (
            f"{where}: optmat disperses the vibrational Raman strengths to the pump "
            "wavelength, so the response needs the carrier: pass the run's wavelength, "
            "or MaterialSelection(carrier_wavelength=...)"
        )
        raise MaterialError(msg)
    return float(carrier)


def gas_raman(
    gas: str,
    options: MolecularOptions | None = None,
    *,
    selection: MaterialSelection,
    thg: bool = True,
    density: float | None = None,
    wavelength: float | None = None,
) -> OptmatRamanResponse:
    """Build a pure gas's Raman response from optmat, not bound to a mixture.

    Parameters
    ----------
    gas : str
        The species, a molecule optmat curates Raman lines for.
    options : MolecularOptions or None, optional
        The term flags and the temperature in K; ``None`` keeps every term at
        :data:`solprop.constants.ROOM_TEMPERATURE`. ``min_j`` and ``max_j`` must keep
        their defaults: optmat keeps every level above its population floor.
    selection : MaterialSelection
        The selection: ``raman`` and the carrier.
    thg : bool, optional
        Whether a field grid keeps the third harmonic. Default ``True``.
    density : float or None, optional
        The constant number density in m^-3 of the fill, > 0: declared to the
        response's range check, and the density H2's shifted lines are built at.
        ``None`` declares none, and refuses a gas whose lines shift.
    wavelength : float or None, optional
        The run's reference wavelength in m, used when the selection names no
        carrier.

    Returns
    -------
    OptmatRamanResponse
        The response, taking the run's one density.

    Raises
    ------
    MaterialError
        If optmat has no Raman lines for the gas, the options cannot be honoured, the
        lines shift and no density is given, or the carrier is unknown.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_raman import gas_raman
    >>> from solprop.physdata.selection import MaterialSelection
    >>> response = gas_raman("N2", selection=MaterialSelection(carrier_wavelength=8e-7))
    >>> response.assembly.groups, response.assembly.terms
    (43, 72)
    """
    options = options or MolecularOptions()
    _check_levels(options, gas)
    kind = _gas_kind(gas)
    if kind != "molecule":
        what = "an atom" if kind == "atom" else "not a gas optmat curates"
        msg = f"{gas} has no molecular Raman response: it is {what}"
        raise MaterialError(msg)
    if density is not None and not (math.isfinite(density) and density > 0.0):
        msg = f"{gas}: a Raman fill density must be positive and finite, got {density}"
        raise MaterialError(msg)
    temperature = (
        ROOM_TEMPERATURE if options.temperature is None else options.temperature
    )
    request = _Request(
        fractions={gas: 1.0},
        temperature=float(temperature),
        wavelength=_carrier(selection, wavelength, gas),
        selection=selection,
        keep={gas: options},
        dephasing={},
        densities=None if density is None else np.array([float(density)]),
        constant=density,
        reference=None,
        spread=0.0,
        where=gas,
    )
    return _assemble(request, thg=thg, labels=(), leaf=None)


def _check_levels(options: MolecularOptions, where: str) -> None:
    """Refuse a level truncation, which optmat's response does not take.

    Parameters
    ----------
    options : MolecularOptions
        The options.
    where : str
        What they are for.

    Raises
    ------
    MaterialError
        If ``min_j`` or ``max_j`` differs from its default.
    """
    defaults = MolecularOptions()
    if (options.min_j, options.max_j) != (defaults.min_j, defaults.max_j):
        msg = (
            f"{where}: MolecularOptions(min_j=, max_j=) truncates solprop's own "
            "rotational ladder; optmat's response keeps every level above its "
            "population floor and takes no truncation"
        )
        raise MaterialError(msg)


@dataclass(frozen=True)
class _Path:
    """The composition's labelled densities, sampled over the run.

    Attributes
    ----------
    z : numpy.ndarray
        The probe positions in m.
    densities : numpy.ndarray
        Each label's density in m^-3 at the probes, shape ``(n_labels, n_probes)``.
    total : ZProfile
        The summed density path in m^-3.
    span : tuple of float
        The run's span in m, finite or not.
    """

    z: Float[np.ndarray, " n_probes"]
    densities: Float[np.ndarray, "n_labels n_samples"]
    total: ZProfile
    span: tuple[float, float]


def _sampled(mixture: Mixture, length: float | None) -> _Path:
    """Sample every population's density over the run.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    length : float or None
        The run's length in m, or ``None`` for the densities' own span.

    Returns
    -------
    _Path
        The probes, the densities and the summed path.
    """
    paths = tuple(component.density for component in mixture.components)
    total = paths[0] if len(paths) == 1 else Sum(paths)
    span = (0.0, float(length)) if length is not None else total.z_span
    if all(math.isfinite(end) for end in span) and span[1] > span[0]:
        kinks = np.asarray(total.breakpoints, dtype=float)
        z = np.unique(
            np.concatenate(
                [
                    np.linspace(span[0], span[1], _PROBES),
                    kinks[(kinks > span[0]) & (kinks < span[1])],
                ]
            )
        )
    else:
        z = _probe_positions(total)
    return _Path(z, mixture.density(z), total, span)


def _composition(
    mixture: Mixture, path: _Path, where: str
) -> tuple[dict[str, float], float]:
    """Return the composition's mole fractions and their spread along the run.

    Duplicate labels of one species are merged; a label whose density is zero over
    the whole run takes no part.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    path : _Path
        Its sampled densities.
    where : str
        What the response is for.

    Returns
    -------
    tuple
        Species to mole fraction, and the largest relative spread of a fraction.

    Raises
    ------
    MaterialError
        If the fractions vary along the run beyond :data:`COMPOSITION_TOLERANCE`.
    """
    names = [component.species.name for component in mixture.components]
    by_species: dict[str, Float[np.ndarray, " n_probes"]] = {}
    for name, row in zip(names, path.densities, strict=True):
        by_species[name] = by_species.get(name, np.zeros_like(row)) + row
    total = np.sum(np.stack(list(by_species.values())), axis=0)
    filled = total > 0.0
    fractions: dict[str, float] = {}
    spread = 0.0
    for name, row in by_species.items():
        if not np.any(row > 0.0):
            continue
        mean = float(np.sum(row[filled]) / np.sum(total[filled]))
        local = row[filled] / total[filled]
        spread = max(spread, float(np.max(np.abs(local - mean))) / mean)
        fractions[name] = mean
    if spread > COMPOSITION_TOLERANCE:
        msg = (
            f"{where}: the populations' densities are not proportional along the run "
            f"(a mole fraction varies by {spread:.3g} of itself, beyond "
            f"{COMPOSITION_TOLERANCE:g}), so the Raman composition varies along z, "
            "which one response cannot represent; it is refused rather than built at "
            "one composition (SEP 3, section 4.4)"
        )
        raise MaterialError(msg)
    return fractions, spread


def _choices(
    mixture: Mixture,
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions],
) -> dict[str, MolecularOptions]:
    """Return each selected label's term flags.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    raman : bool, sequence of str or Mapping
        :func:`solprop.nonlinear.mixtures.responses_for`'s selection.

    Returns
    -------
    dict
        Label to options, for the labels selected.
    """
    chosen: dict[str, MolecularOptions] = {}
    for component in mixture.components:
        label = component.label
        choice: bool | MolecularOptions = (
            raman.get(label, False)
            if isinstance(raman, Mapping)
            else raman
            if isinstance(raman, bool)
            else label in raman
        )
        if choice is False:
            continue
        options = choice if isinstance(choice, MolecularOptions) else component.raman
        chosen[label] = options or MolecularOptions()
    return chosen


def _keep(
    mixture: Mixture, chosen: Mapping[str, MolecularOptions], *, explicit: bool
) -> tuple[dict[str, MolecularOptions], dict[str, Dephasing]]:
    """Return the species whose lines are kept, their flags and supplied laws.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    chosen : Mapping of str to MolecularOptions
        The selected labels' options.
    explicit : bool
        Whether the labels were named, so that one without a response is refused.

    Returns
    -------
    tuple
        Species to options, and species to a supplied dephasing.

    Raises
    ------
    MaterialError
        If a named label has no molecular response, or two labels of one species
        disagree about their options or linewidths.
    """
    keep: dict[str, MolecularOptions] = {}
    laws: dict[str, Dephasing] = {}
    for component in mixture.components:
        label, name = component.label, component.species.name
        if label not in chosen:
            continue
        _check_levels(chosen[label], label)
        kind = _gas_kind(name)
        if kind != "molecule":
            if explicit:
                what = "an atom" if kind == "atom" else "not a gas optmat curates"
                msg = f"{label} ({name}) has no molecular Raman response: it is {what}"
                raise MaterialError(msg, rule="raman-needs-molecule")
            continue
        options = chosen[label]
        if keep.setdefault(name, options) != options:
            msg = (
                f"two populations of {name} ask for different Raman terms; the "
                "species' lines are built once for the composition"
            )
            raise MaterialError(msg)
        if (
            component.dephasing is not None
            and laws.setdefault(name, component.dephasing) != component.dephasing
        ):
            msg = f"two populations of {name} supply different Raman linewidths"
            raise MaterialError(msg)
    return keep, laws


def _requested_temperature(options: Sequence[MolecularOptions]) -> float | None:
    """Return the one explicit reference temperature the options name, if any.

    Parameters
    ----------
    options : sequence of MolecularOptions
        The kept species' options.

    Returns
    -------
    float or None
        The temperature in K, or ``None`` when none names one.

    Raises
    ------
    MaterialError
        If two name different temperatures: a response has one.
    """
    named = {float(o.temperature) for o in options if o.temperature is not None}
    if len(named) > 1:
        msg = (
            f"the Raman options name temperatures {sorted(named)} K, and a "
            "composition's response is built at one"
        )
        raise MaterialError(msg)
    return named.pop() if named else None


def mixture_raman(  # noqa: PLR0913 - the mixture, its selection and the response's switches
    mixture: Mixture,
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions],
    *,
    selection: MaterialSelection,
    thg: bool = True,
    wavelength: float | None = None,
    length: float | None = None,
    explicit: bool = False,
) -> OptmatRamanResponse | None:
    """Build the Raman response of a mixture's whole composition from optmat.

    Every labelled population is a member of the composition, Raman-active or not:
    the response is optmat's for those mole fractions, at the reference temperature
    of :func:`solprop.nonlinear.mixtures.raman_temperature` (over the summed density),
    dispersed to the carrier, and driven and damped by the summed density of the
    labels. The selected labels choose whose lines are kept, and their
    :class:`~solprop.physdata.raman_types.MolecularOptions` which bands.

    Parameters
    ----------
    mixture : Mixture
        The fill; every population optmat-resolved.
    raman : bool, sequence of str or Mapping
        ``True`` keeps every Raman-active species; a sequence or mapping selects
        labels, and a mapping's options select bands.
    selection : MaterialSelection
        The selection: ``raman`` (``"recommended"``, ``"provisional"`` or an optmat
        response object), ``raman_shift`` and the carrier.
    thg : bool, optional
        Whether a field grid keeps the third harmonic. Default ``True``.
    wavelength : float or None, optional
        The run's reference wavelength in m, used when the selection names no
        carrier.
    length : float or None, optional
        The run's length in m, over which the composition, the reference
        temperature and density, and the densities checked are taken; ``None``
        takes the densities' own span.
    explicit : bool, optional
        Whether `raman` named labels, so that one without a response is refused.

    Returns
    -------
    OptmatRamanResponse or None
        The response, or ``None`` when no selected population is a molecule, or the
        fill is vacuum along the whole run.

    Raises
    ------
    MaterialError
        If a population is not optmat-resolved beside a Raman-active one, optmat has
        no broadening for a pair of constituents (SEP 3, D16), the composition
        varies along z, a provisional response is not admitted, a shift is not
        admitted on a gradient, or the options cannot be honoured.

    Warns
    -----
    MaterialRangeWarning
        Once per run each, outside the response's temperature or density range.
    MaterialStatusWarning
        Once per run, for an admitted provisional response.
    RamanTemperatureWarning
        For a temperature path held at its density-weighted reference.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_raman import mixture_raman
    >>> from solprop.physdata import Mixture
    >>> from solprop.physdata.selection import MaterialSelection
    >>> selection = MaterialSelection(carrier_wavelength=800e-9)
    >>> air = Mixture.air(1.0, selection=selection)
    >>> response = mixture_raman(air, True, selection=selection)
    >>> response.labels, response.assembly.groups
    (('N2', 'O2', 'Ar'), 82)
    """
    from solprop.nonlinear.mixtures import raman_temperature  # noqa: PLC0415 - a cycle

    chosen = _choices(mixture, raman)
    keep, laws = _keep(mixture, chosen, explicit=explicit)
    if not keep:
        logger.info(
            "%s: no selected population has a molecular Raman response", mixture
        )
        return None
    where = "+".join(mixture.labels)
    strangers = [
        f"{c.label} ({c.species.name})"
        for c in mixture.components
        if not built_in_population(c.species) or _gas_kind(c.species.name) is None
    ]
    if strangers:
        msg = (
            f"{where}: {', '.join(strangers)} is not a gas optmat resolves, so optmat "
            "has no Raman broadening of the mixture's molecules by it (SEP 3, D16); "
            "build the fill with Mixture.fill(..., selection=) or pass raman=False"
        )
        raise MaterialError(msg)
    path = _sampled(mixture, length)
    fractions, spread = _composition(mixture, path, where)
    keep = {name: options for name, options in keep.items() if name in fractions}
    if not keep:
        logger.info("%s: the Raman-active populations are vacuum along the run", where)
        return None
    total = np.sum(path.densities, axis=0)
    temperature = raman_temperature(
        mixture,
        mixture.labels,
        _requested_temperature(list(keep.values())),
        length=length,
        widths=True,
    )
    constant = (
        float(total[0])
        if all(c.density.is_constant for c in mixture.components)
        else None
    )
    request = _Request(
        fractions=fractions,
        temperature=temperature,
        wavelength=_carrier(selection, wavelength, where),
        selection=selection,
        keep=keep,
        dephasing={name: law for name, law in laws.items() if name in keep},
        densities=total,
        constant=constant,
        reference=None if constant is not None else _reference_density(path),
        spread=spread,
        where=where,
    )
    leaf = next(
        c.label
        for c in mixture.components
        if c.species.name in keep and c.label in chosen
    )
    return _assemble(
        request,
        thg=thg,
        labels=mixture.labels,
        leaf=None if len(mixture.components) == 1 else leaf,
    )


def _reference_density(path: _Path) -> float | None:
    r"""Return the density-weighted mean density over a finite run.

    The analogue of the reference temperature:
    :math:`N_\mathrm{ref} = \int N^2\,dz / \int N\,dz`, by quadrature across the
    path's breakpoints.

    Parameters
    ----------
    path : _Path
        The run's summed density.

    Returns
    -------
    float or None
        The reference density in m^-3, or ``None`` over an unbounded run.
    """
    from solprop.nonlinear.mixtures import _mean_temperature  # noqa: PLC0415 - a cycle

    start, end = path.span
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        return None
    return _mean_temperature(path.total, path.total, start, end, path.z)
