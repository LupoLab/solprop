# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: what import says about the thread variables."""
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import solprop
print("finished warnings", [str(x.message)[:120] for x in caught if x.category is RuntimeWarning], flush=True)
