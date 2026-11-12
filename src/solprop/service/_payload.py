# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A request's payload, and the recipe it carries."""

from collections.abc import Mapping
from pathlib import Path

from solprop.recipe import Recipe

__all__ = ["Payload", "base_of", "read_recipe"]

type Payload = Mapping[str, object]
"""What a request carries: plain data, keyed by name."""


def base_of(payload: Payload) -> Path | None:
    """Return the directory a payload's recipe reads its data files from, if any."""
    base = payload.get("base")
    return Path(str(base)) if base else None


def read_recipe(payload: Payload) -> Recipe:
    """Read the recipe a payload carries: ``text``, its TOML, and ``base`` if any."""
    return Recipe.loads(str(payload["text"]), base=base_of(payload))
