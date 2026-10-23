# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The metals: optmat's tables by default and by name, and a foreign model refused.

Every built-in metal is optmat's table (``table/Ag``, ``table/Al``; derivatives and
ranges in test_optmat_solids.py).
"""

import pytest

from solprop.exceptions import MaterialError
from solprop.physdata import aluminium, silver


@pytest.mark.parametrize(("factory", "material"), [(silver, "Ag"), (aluminium, "Al")])
def test_a_metal_is_optmats_table_by_default_and_by_name(factory, material):
    assert factory().source == f"table/{material}"
    assert factory(model=f"table/{material}") == factory()


@pytest.mark.parametrize(("factory", "other"), [(silver, "Al"), (aluminium, "Ag")])
def test_another_metals_table_is_refused(factory, other):
    with pytest.raises(MaterialError):
        factory(model=f"table/{other}")
