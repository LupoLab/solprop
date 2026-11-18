Dropped optmat's internal tracker ids, references users cannot look up, from every
optmat text solprop passes on: its range and absorption warnings (`MaterialRangeWarning`),
the status reason of each material record (`MaterialRecord.status_reason`, and so the
run's material manifest and `solprop.capabilities.species`) and the
`MaterialStatusWarning` built from it, which also no longer doubles the reason's full
stop. The rest of each text is unchanged
([#166](https://github.com/LupoLab/solprop/issues/166)).
