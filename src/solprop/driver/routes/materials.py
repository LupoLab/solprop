# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve a route's built-in materials under a material selection (SEP 3).

Also the policy checks a route applies to its selection: the carrier, and the
electronic policy its responses can honour.
"""

import dataclasses
from collections.abc import Sequence

from solprop.exceptions import DriverError
from solprop.modes.capillary import GlassIndex
from solprop.modes.graded_index import GradedIndexProvider
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.material import MaterialRegion
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import fused_silica
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.nonlinear_data import carriers_agree
from solprop.physdata.selection import DISPERSIVE_POLICIES, MaterialSelection

__all__ = []


def _refuse_dispersive_policy(materials: MaterialSelection | None, route: str) -> None:
    """Refuse a dispersive electronic policy on a route whose response is constant."""
    if materials is None or materials.electronic not in DISPERSIVE_POLICIES:
        return
    msg = (
        f"the electronic policy {materials.electronic!r} evaluates optmat's "
        f"dispersive hyperpolarizability model, and {route} is a constant coefficient "
        "this route keeps: select 'carrier-kerr' (or 'recommended') here"
    )
    raise DriverError(msg, parameter="materials.electronic", rule="gnlse-carrier-kerr")


def _check_electronic_policy(
    materials: MaterialSelection | None, responses: Sequence[object]
) -> None:
    """Refuse a constant Kerr response under a dispersive electronic policy.

    A caller's own responses are taken as they are, but a constant
    :class:`~solprop.nonlinear.responses.KerrResponse` beside a selection that names a
    dispersive policy would record one law and propagate another.
    """
    if materials is None or materials.electronic not in DISPERSIVE_POLICIES:
        return
    if any(isinstance(response, KerrResponse) for response in responses):
        msg = (
            f"the selection's electronic policy is {materials.electronic!r} and the "
            "run declares a constant KerrResponse; build the electronic response as "
            "a DispersiveKerrResponse (solprop.nonlinear.mixtures."
            "electronic_response), or select 'carrier-kerr'"
        )
        raise DriverError(msg)


def _check_selection_carrier(
    materials: MaterialSelection | None,
    reference: float,
    wavelength: float | None = None,
) -> None:
    """Refuse a carrier that is not the run's: one carrier sets the Kerr coefficient.

    The electronic coefficient, the Raman response and the rate are all evaluated at
    the carrier.

    - ``reference``: the grid's reference wavelength in m, the run's carrier.
    - ``wavelength``: the driver's `wavelength` in m, at which the responses are built.
    """
    if materials is None:
        return
    # Without the driver's wavelength the reference is a grid given whole.
    carrier = (
        "materials.carrier_wavelength",
        "grid" if wavelength is None else "wavelength",
    )
    for name, value, parameter in (
        ("the selection's carrier_wavelength", materials.carrier_wavelength, carrier),
        ("wavelength", wavelength, ("wavelength", "grid")),
    ):
        if value is not None and not carriers_agree(float(value), reference):
            msg = (
                f"{name} is {value:g} m and the grid's reference wavelength "
                f"{reference:g} m: under a material selection one carrier sets the "
                "Kerr coefficient of every optmat species (SEP 3, section 4.3), and "
                "it is the grid's. Build the grid at that wavelength, or leave the "
                "selection's carrier_wavelength unset"
            )
            raise DriverError(msg, parameter=parameter)


def _default_cladding(cladding: GlassIndex | complex | None) -> GlassIndex | complex:
    """Return the cladding a numbered capillary set takes.

    - ``cladding``: the cladding given; ``None`` is the package default.
    """
    return fused_silica() if cladding is None else cladding


def _selected_guide(guide: ModalGuide, materials: MaterialSelection) -> ModalGuide:
    """Resolve a guide's default delayed silica response under a material selection.

    A region's :class:`~solprop.nonlinear.raman.SilicaTensorResponse` with the
    default parameters, optmat's recommended tensor bundle, meant the built-in silica
    tensor response; under the selection it becomes the tensor bundle its ``raman``
    names (:meth:`~solprop.physdata.glasses.SilicaBundle.resolve` with
    ``tensor=True``). A response declared with any other parameters, and a provider
    of another kind, are taken as they are.

    - Returns: the guide to propagate on: the one given when nothing resolves, otherwise
      the same provider, channels and sampling with the regions' delayed responses
      resolved.
    """
    default = SilicaTensorResponse()
    if not any(region.raman == default for region in guide.regions):
        return guide
    provider = guide.field_provider
    if not isinstance(provider, StepIndexProvider | GradedIndexProvider):
        return guide
    resolved = SilicaTensorResponse(
        parameters=SilicaBundle.resolve(materials.raman, tensor=True)
    )

    def selected(region: MaterialRegion) -> MaterialRegion:
        """Return a region with its default delayed response resolved."""
        return dataclasses.replace(
            region,
            responses=tuple(
                resolved if one == default else one for one in region.responses
            ),
        )

    rebuilt = (
        dataclasses.replace(
            provider, regions=tuple(selected(one) for one in provider.regions)
        )
        if isinstance(provider, StepIndexProvider)
        else dataclasses.replace(
            provider,
            core_responses=selected(provider.regions[0]).responses,
        )
    )
    return dataclasses.replace(
        guide, field_provider=rebuilt, regions=tuple(rebuilt.regions)
    )
