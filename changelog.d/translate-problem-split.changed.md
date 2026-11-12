`solprop.translate.problem` now holds the kernel contract alone; translation is split
into modules by concern: `solprop.translate.assembly` (`assemble`,
`MAX_GENERAL_FLOW_EXPONENT`), `solprop.translate.walls` (a capillary's coherent wall and
taper coupling) and `solprop.translate.validation` (`validate_coupled_flow`,
`DEFAULT_FLOW_CONDITION_LIMIT`, `MOVING_FLOW_CERTIFICATE`, `TAPER_FLOW_CERTIFICATE`).
Every name `solprop.translate.problem` exported still imports from it, and
`solprop.translate.assemble` is unchanged.
