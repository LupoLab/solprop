`solprop.output.schema` is split into five modules: `solprop.output.serialisation`
(`serialise`, `ArrayDigest`, `StatedInput`, `recorded_input`), `solprop.output.provenance`
(`BackendRecord`, `GitProvenance`, `package_versions`), `solprop.output.records`
(`GridRecord`, `ModeRecord`, `StepperRecord`, `SavedPlane`, `RunOutcome` and the other
record types), `solprop.output.metadata` (`RunMetadata`, `SCHEMA_VERSION`,
`resolve_metadata_write`) and `solprop.output.store` (`Output`, `PlaneStore`,
`FieldRetention` and the other save protocols). Import these names from
`solprop.output`, which still exports the public ones, or from their new module. A run
record names its record types by module (`solprop.output.records.GridRecord`), so every
configuration hash differs from the one an earlier version wrote: a file or a scan
written before cannot be resumed or continued, and is read as it was.
