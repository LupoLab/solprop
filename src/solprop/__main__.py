# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``python -m solprop``: the ``solprop`` command line (:mod:`solprop.cli`)."""

from solprop.cli import main

__all__: list[str] = []

if __name__ == "__main__":
    raise SystemExit(main())
