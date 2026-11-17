Fixed a native Adams trial whose source evaluation failed leaving its prediction in the
state, so that the failed run's terminal checkpoint and march snapshot held the
prediction instead of the last accepted step (#129).
