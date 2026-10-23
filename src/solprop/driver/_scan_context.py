# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share the CLI execution policy between imported and module entry points."""

import contextvars

from solprop.driver.executors import Executor

_EXECUTOR: contextvars.ContextVar[Executor | None] = contextvars.ContextVar(
    "scan_executor", default=None
)


def current_executor() -> Executor | None:
    """Return the CLI policy within the currently executing script.

    Returns
    -------
    Executor or None
        Default executor; explicit run_scan arguments always take precedence.
    """
    return _EXECUTOR.get()
