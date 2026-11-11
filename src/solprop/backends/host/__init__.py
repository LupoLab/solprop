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
(:mod:`~solprop.backends.host.group_session`). The reference shares only stateless
numerical modules with the native backend -- the controller cores, tableaux, stage
tables, the exponential coefficients, the Adams and low-storage host engines
(:class:`~solprop.backends.multistep.HostHistory`,
:class:`~solprop.backends.low_storage.HostLowStorage`) and the call ledger -- and the
parts every Python session composes (:mod:`solprop.backends.session_core`), as the
maintainer decided at SEP 47's acceptance; the native backend runs in its own session
and inherits nothing of this package.

The package imports nothing at its top level, so that a backend module can import one
part without the others.
"""
