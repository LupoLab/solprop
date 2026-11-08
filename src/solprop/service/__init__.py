# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service between a front end and solprop: messages, kernel, workers, transports.

A front end (the desktop window of ``solprop-gui``) talks to this layer through one set
of request and event messages. The kernel process answers checks, previews, design
tools and readings of a results file as plain data, and one worker process per run
streams reduced rows while it writes the file. The window's process imports only the
light modules here (the messages, the clients, the reductions' descriptions); the heavy
ones run in the children. Nothing here imports Qt, and nothing else in solprop imports
this package.

This is front-end support, not part of the user interface: it changes without notice.
"""

__all__: list[str] = []
