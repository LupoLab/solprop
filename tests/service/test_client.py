# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The window's clients, over real child processes."""

import threading

from service import targets
from solprop.service.client import KernelClient, ReplyFilter
from solprop.service.messages import Reply, Request


def test_the_filter_drops_stale_and_superseded_replies():
    wanted = ReplyFilter()
    wanted.sent(Request("check", {}, 1, revision=1))
    wanted.sent(Request("examples", {}, 2))
    wanted.revision = 2
    assert not wanted.accepts(Reply("check", 1, 1))  # an older revision
    assert wanted.accepts(Reply("examples", 2, 0))  # about no recipe
    assert not wanted.accepts(Reply("examples", 2, 0))  # answered already
    wanted.sent(Request("x", {}, 3))
    assert not wanted.accepts(Reply("x", 3, 0, superseded=True))


def test_the_kernel_client_delivers_the_latest_reply():
    replies: list[Reply] = []
    done = threading.Event()

    def on_reply(reply: Reply) -> None:
        replies.append(reply)
        if reply.value == 3:
            done.set()

    client = KernelClient(on_reply, target=targets.echo_kernel)
    client.revision = 1
    assert client.revision == 1
    for x in (1, 2, 3):
        client.request("echo", {"x": x}, revision=1)
    assert done.wait(60)
    assert client.alive
    client.close()
    assert [reply.value for reply in replies][-1] == 3
    assert all(reply.value != 2 for reply in replies)  # superseded or stale


def test_a_reply_that_arrives_while_closing_is_not_delivered():
    replies: list[Reply] = []
    client = KernelClient(replies.append, target=targets.echo_kernel)
    client._closing = True  # as close() sets it, before the pipe is shut
    client.request("echo", {"x": 1})
    client._reader.join(60)
    assert replies == [] and not client._reader.is_alive()
    client.close()


def test_a_reply_to_a_deleted_receiver_ends_the_reader_quietly():
    lost = threading.Event()

    def on_reply(_reply: Reply) -> None:
        raise RuntimeError("Signal source has been deleted")  # as Qt says

    client = KernelClient(on_reply, on_lost=lost.set, target=targets.echo_kernel)
    client.request("echo", {"x": 1})
    client._reader.join(60)
    assert not client._reader.is_alive() and not lost.is_set()
    client.close()


def test_the_reader_may_close_its_own_client():
    closed = threading.Event()
    holder: list[KernelClient] = []

    def on_reply(_reply: Reply) -> None:
        holder[0].close()  # from the reader thread: it does not wait for itself
        closed.set()

    holder.append(KernelClient(on_reply, target=targets.echo_kernel))
    holder[0].request("echo", {"x": 1})
    assert closed.wait(60) and not holder[0].alive


def test_a_kernel_that_ends_is_reported_once():
    lost = threading.Event()
    client = KernelClient(
        lambda _reply: None, on_lost=lost.set, target=targets.ending_kernel
    )
    assert lost.wait(60)
    client.close()


def test_a_kernel_that_will_not_end_is_terminated():
    client = KernelClient(lambda _reply: None, target=targets.deaf_kernel)
    client.close(timeout=0.5)
    assert not client.alive
