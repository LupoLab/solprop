Keep Julia dense-output periodic rows in the same delivery chunk as their accepted
attempt. Endpoint statistics are sampled before an interior plane can drain that
attempt; owed planes still precede snapshots and further trials (#687).
