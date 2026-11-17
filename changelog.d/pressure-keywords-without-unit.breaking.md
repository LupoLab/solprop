The steady-flow and pressure-profile functions name their pressures without the unit,
as `physdata` does; the pressures stay in bar. `gasflow.throughput` and
`PressureGradient.two_point` rename `start_bar` and `end_bar` to `start_pressure` and
`end_pressure`, `gasflow.choked_pressure` renames `high_bar` to `high_pressure`,
`gasflow.steady_fill` renames `pressures_bar` to `pressures` (as
`PressureGradient.from_stages` already spells it), `StageFlow` renames its fields
`start_bar` and `end_bar` to `start_pressure` and `end_pressure`, and `FlowReport`
renames `choked_start_bar` and `choked_end_bar` to `choked_start_pressure` and
`choked_end_pressure`. A recipe's `PressureGradient.two_point` table takes the new
keys. Positional calls are unchanged; rename the keywords, recipe keys and attributes.
