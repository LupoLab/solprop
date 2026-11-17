# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's marches, one module each, and what they share (SEP 47 phase 2).

- :mod:`~solprop.backends.numpy_ref.marches.host`: what every march holds
  (:class:`~solprop.backends.numpy_ref.marches.host.MarchState`) and what it asks of
  the session that schedules it
  (:class:`~solprop.backends.numpy_ref.marches.host.MarchHost`);
- :mod:`~solprop.backends.numpy_ref.marches.runge_kutta`: the embedded Runge-Kutta
  march, adaptive or replayed;
- :mod:`~solprop.backends.numpy_ref.marches.adams`: the Adams march, adaptive or
  replayed;
- :mod:`~solprop.backends.numpy_ref.marches.low_storage`: the low-storage replay.

Only the reference runs them; the native session's compiled marches are traced against
them (``docs/dev/backends.md`` section 1).
"""
