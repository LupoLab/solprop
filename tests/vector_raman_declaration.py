# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prototype the proposed immutable constructor and metadata contract."""

from dataclasses import dataclass

from solprop.output.schema import serialise


@dataclass(frozen=True)
class Term:
    """Keep a kernel's identity beside its explicit contraction."""

    model: object
    contraction: str

    def __post_init__(self):
        if self.contraction not in ("scalar_only", "isotropic", "rank2"):
            raise ValueError("unknown contraction")


@dataclass(frozen=True, init=False)
class Declaration:
    """Exercise canonical dataclass fields and keyword-only tensor construction."""

    tensor_terms: tuple[Term, ...]
    fraction: float | None
    thg: bool
    species: str | None

    def __init__(
        self, models=None, fraction=None, thg=True, species=None, *, tensor_terms=None
    ):
        if (models is None) == (tensor_terms is None):
            raise ValueError("exactly one input route is required")
        if models is not None:
            terms = tuple(Term(model, "scalar_only") for model in models)
        else:
            assert tensor_terms is not None
            terms = tuple(tensor_terms)
        if not terms:
            raise ValueError("empty response")
        for name, value in (
            ("tensor_terms", terms),
            ("fraction", fraction),
            ("thg", thg),
            ("species", species),
        ):
            object.__setattr__(self, name, value)

    @property
    def models(self):
        return tuple(term.model for term in self.tensor_terms)

    def record(self):
        """Propose one authoritative metadata route: the tensor terms, always."""
        record = {
            "__class__": "solprop.nonlinear.raman.RamanResponse",
            "fraction": self.fraction,
            "thg": self.thg,
            "species": self.species,
        }
        record["tensor_terms"] = [
            {
                "__class__": "solprop.nonlinear.raman.RamanTensorTerm",
                "model": serialise(term.model),
                "contraction": term.contraction,
            }
            for term in self.tensor_terms
        ]
        return record
