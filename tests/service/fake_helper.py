# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A helper that misbehaves on purpose, for the connection's tests.

Run as ``python fake_helper.py MODE``:

``protocol``   says hello in another protocol.
``chatty``     says hello, then a frame that is no dict, a reply nobody asked for, and
               an event, then answers one request.
``silent``     says hello and never answers.
``drop``       says hello, reads one request and leaves without answering.
``stubborn``   says hello and ignores its input closing for a minute.
"""

import sys
import time

from solprop.service.agent import hello
from solprop.service.frames import read_frame, write_frame
from solprop.service.messages import PROTOCOL


def main() -> None:
    mode = sys.argv[1]
    out = sys.stdout.buffer
    greeting = hello()
    if mode == "protocol":
        greeting["protocol"] = PROTOCOL + 1
    write_frame(out, greeting)
    if mode == "chatty":
        write_frame(out, [1, 2])
        write_frame(out, {"kind": "reply", "id": 999, "value": None})
        write_frame(out, {"kind": "note", "text": "hi"})
        request = read_frame(sys.stdin.buffer)
        write_frame(out, {"kind": "reply", "id": request["id"], "value": "ok"})  # type: ignore[index]
        read_frame(sys.stdin.buffer)
    elif mode == "silent":
        while True:
            read_frame(sys.stdin.buffer)
    elif mode == "drop":
        read_frame(sys.stdin.buffer)
    elif mode == "stubborn":
        time.sleep(60)


if __name__ == "__main__":
    main()
