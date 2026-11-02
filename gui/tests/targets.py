# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Child entry points for the client tests; importable by a spawned process."""

import time
from multiprocessing.connection import Connection

from solprop_gui.service import kernel
from solprop_gui.service.worker import RunSpec


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


def silent_worker(spec: RunSpec, events: Connection, control: Connection) -> None:
    """A worker that ends without a terminal event."""
    del spec, events, control


def deaf_worker(spec: RunSpec, events: Connection, control: Connection) -> None:
    """A worker that never answers."""
    del spec, events, control
    time.sleep(60)
