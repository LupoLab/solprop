# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy reference's own host execution machinery (SEP 47 phase 2).

The stateful control the reference session runs on: the step driver
(:mod:`solprop.backends.host.driver`), the Adams march and its general path
(:mod:`~solprop.backends.host.adams`, :mod:`~solprop.backends.host.general`), the
low-storage march (:mod:`~solprop.backends.host.low_storage`), the fixed-schedule group
(:mod:`~solprop.backends.host.group`) and the reference's run session
(:mod:`~solprop.backends.host.session`). The reference shares only stateless numerical
modules with the native bridge -- the controller cores, tableaux, stage tables, the
exponential coefficients and the Adams and low-storage engines -- and never imports the
inherited driver (:mod:`solprop.backends.driver`) that bridge still runs on (SEP 47,
the maintainer's decision at acceptance on the transition).

The package imports nothing at its top level, so that a backend module can import one
part without the others.
"""
