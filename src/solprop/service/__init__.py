# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service between a front end and solprop: messages, kernel, runs, machines.

A front end (the desktop window of ``solprop-gui``) talks to this layer through one set
of request and event messages (:mod:`~solprop.service.messages`). The window's process
imports only the light modules; the heavy ones run in its children. Nothing here imports
Qt, and nothing else in solprop imports this package.

- The kernel, a warm solprop answering checks, previews, design tools and readings as
  plain data: :mod:`~solprop.service.client` and :mod:`~solprop.service.kernel` (the
  process) and :mod:`~solprop.service.handlers` (the request table), drawing on
  ``design``, ``explore``, ``sheet``, ``waveguide``, ``results``, ``radial_pages``,
  ``postprocess`` and ``figures``; ``beam``, ``reduce``, ``scan_reductions`` and
  ``recipes`` are shared with the runs and the window.
- Runs, each a job of its own that writes its record: :mod:`~solprop.service.worker`
  (the spec), :mod:`~solprop.service.run` and :mod:`~solprop.service.job` (running it),
  :mod:`~solprop.service.record` and :mod:`~solprop.service.frames` (its record),
  :mod:`~solprop.service.runs` and :mod:`~solprop.service.states` (their states).
- Other machines: :mod:`~solprop.service.machines`, :mod:`~solprop.service.agent`,
  :mod:`~solprop.service.launch` and :mod:`~solprop.service.install`.

This is front-end support, not part of the user interface: it changes without notice.
The design is ``docs/dev/gui/design.md`` (13.3, 13.4).
"""

__all__: list[str] = []
