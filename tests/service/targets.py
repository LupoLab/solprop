# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Child entry points for the client tests; importable by a spawned process."""

import os
import time
from multiprocessing.connection import Connection

from solprop.service import kernel


def echo_kernel(connection: Connection) -> None:
    """A kernel that answers ``echo`` with its ``x``, slowly enough to supersede."""

    def echo(payload: object) -> object:
        time.sleep(0.2)
        return payload["x"]  # pyright: ignore[reportIndexIssue]

    kernel.serve(connection, {"echo": echo})


def ending_kernel(connection: Connection) -> None:
    """A kernel that ends at once."""
    connection.close()


def deaf_kernel(connection: Connection) -> None:
    """A kernel that ignores the pipe's closing."""
    del connection
    time.sleep(60)


def environment_kernel(connection: Connection) -> None:
    """A kernel that answers ``environment`` with one of its variables."""

    def environment(payload: object) -> object:
        return os.environ.get(payload["name"])  # pyright: ignore[reportIndexIssue]

    kernel.serve(connection, {"environment": environment})
