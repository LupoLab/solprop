# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's own host execution machinery (SEP 47 phase 2).

The stateful control the reference session runs on: the step driver
(:mod:`solprop.backends.host.driver`), the Adams march and its general path
(:mod:`~solprop.backends.host.adams`, :mod:`~solprop.backends.host.general`), the
low-storage march (:mod:`~solprop.backends.host.low_storage`), the fixed-schedule group
(:mod:`~solprop.backends.host.group`), the reference's run session
(:mod:`~solprop.backends.host.session`) and its group session
(:mod:`~solprop.backends.host.group_session`). The reference shares stateless numerical
modules with the native bridge -- the controller cores, tableaux, stage tables and the
exponential coefficients -- and the parts every Python session composes
(:mod:`solprop.backends.session_core`), and never imports the inherited driver
(:mod:`solprop.backends.driver`) that bridge still runs on, or its transitional
sessions (:mod:`solprop.backends.driver_session`) (SEP 47, the maintainer's decision at
acceptance on the transition).

**Recorded exceptions**, transitional: the Adams host engine
:class:`~solprop.backends.multistep.HostHistory`, whose arithmetic the bridge's march
also calls where native keeps no resident history, and the low-storage host engine
:class:`~solprop.backends.low_storage.HostLowStorage`, the reference's alone but still
housed beside the bridge's march. They leave the shared modules when the bridge's
marches are deleted (SEP 47 phase 5).

The package imports nothing at its top level, so that a backend module can import one
part without the others.
"""
