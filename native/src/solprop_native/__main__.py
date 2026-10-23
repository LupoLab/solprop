# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Print the installed solprop-native build record as JSON."""

from __future__ import annotations

import dataclasses
import json

from . import build_info

print(json.dumps(dataclasses.asdict(build_info()), sort_keys=True))
