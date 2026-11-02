# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service between the window and solprop: messages, kernel, workers, transports.

This layer imports no Qt. The window talks to it through one set of request and event
messages; the kernel process (validation, previews, design tools) and one worker
process per run import solprop, and the window's process never imports its heavy parts.
"""

__all__: list[str] = []
