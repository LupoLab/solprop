`StatDefinition.implemented` is removed: every statistic in `solprop.stats.CATALOGUE`
is implemented, and the reserved-name refusal it served had no entries. Each
`StatDefinition` now also says how a statistic is formed and treated (`evaluate`,
`metric`, `weight`, `profile`, `plasma`, `general`, `advice`), and the name sets such as
`METRIC_WEIGHTED_STATISTICS` and `GENERAL_MODAL_ACCESSORS` are derived from those
fields, with unchanged members.
