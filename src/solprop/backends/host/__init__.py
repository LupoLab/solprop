# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's own host execution machinery (SEP 47 phase 2).

The stateful control of the reference's runs: the reference's run session
(:mod:`~solprop.backends.host.session`), which schedules a run and drives its march,
and the marches it drives -- the Runge-Kutta march
(:mod:`~solprop.backends.host.runge_kutta`), the Adams march
(:mod:`~solprop.backends.host.adams`) and the low-storage march
(:mod:`~solprop.backends.host.low_storage`) -- with what every march holds and asks
(:mod:`~solprop.backends.host.marches`); the engines they drive are the workspace's
(:mod:`solprop.backends.numpy_ref.engines`,
:class:`~solprop.backends.numpy_ref.multistep.HostHistory`). The reference shares only
stateless numerical modules with the native backend -- the controller cores,
tableaux, stage tables, the exponential coefficients and the call ledger -- and the
parts every Python session composes (:mod:`solprop.backends.session_core`), as the
maintainer decided at SEP 47's acceptance; the native backend runs in its own session
and inherits nothing of this package.

The package imports nothing at its top level, so that a backend module can import one
part without the others.
"""
