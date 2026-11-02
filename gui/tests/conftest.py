# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test configuration: widgets are drawn without a display."""

import os

# Before any Qt application exists: the tests run headless on every platform, and a
# developer's display is never used. An explicit setting (a debugging session) wins.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
