`solprop.output.schema` is split into five modules: `solprop.output.serialisation`
(`serialise`, `ArrayDigest`, `StatedInput`, `recorded_input`), `solprop.output.provenance`
(`BackendRecord`, `GitProvenance`, `package_versions`), `solprop.output.records`
(`GridRecord`, `ModeRecord`, `StepperRecord`, `SavedPlane`, `RunOutcome` and the other
record types), `solprop.output.metadata` (`RunMetadata`, `SCHEMA_VERSION`,
`resolve_metadata_write`) and `solprop.output.store` (`Output`, `PlaneStore`,
`FieldRetention` and the other save protocols). Import these names from
`solprop.output`, which still exports the public ones, or from their new module. A run
whose input array was recorded by its digest names the digest's class by its new module,
so its configuration hash differs from one written before.
