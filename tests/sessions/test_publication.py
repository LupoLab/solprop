# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep session receipts transactional when an event payload cannot be allocated."""

import pytest

from solprop.backends.events import EventBatch, Outcome
from solprop.backends.session import SessionError, SessionLedger, SessionState


@pytest.mark.parametrize("outcome", [None, Outcome.FAILED])
def test_payload_failure_keeps_the_prior_receipt_and_does_not_consume_sequence(outcome):
    ledger = SessionLedger()
    prior = ledger.issue()
    offered = []

    def fail(receipt):
        offered.append(receipt)
        assert ledger.ticket_valid(prior)
        assert not ledger.ticket_valid(receipt)
        raise MemoryError("event allocation failed")

    for _ in range(2):
        with pytest.raises(MemoryError, match="event allocation failed"):
            ledger.publish(fail, outcome=outcome)
        assert ledger.pending == prior and ledger.outcome is None
    assert offered[0] == offered[1]
    result = ledger.publish(EventBatch, outcome=outcome)
    assert result.receipt == offered[0] and ledger.pending == result.receipt
    assert not ledger.ticket_valid(prior)
    assert ledger.outcome is outcome
    ledger.acknowledge(result.receipt)
    assert ledger.state is (
        SessionState.READY if outcome is None else SessionState.FINISHED
    )


def test_closed_ledger_refuses_before_running_the_payload_factory():
    ledger = SessionLedger()
    ledger.close()

    def never(_):
        pytest.fail("a closed ledger ran a publication factory")

    with pytest.raises(SessionError, match="closed"):
        ledger.publish(never)
