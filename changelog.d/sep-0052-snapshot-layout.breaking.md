The output schema is 3 (SEP 52): a file's march snapshot (`/march_snapshot`) now stores
its `state` as `(n_modes, n_pol, n_omega)` and its engine `rows` as `(n_slots, n_modes,
n_pol, n_omega)`, without the leading axis of one they kept before. A snapshot written in
the earlier layout is refused by name rather than squeezed, and a file of output schema 2,
which 0.2.0 wrote, is refused as any file of another schema is: resume or continue it with
the solprop version that wrote it. Saved fields, statistics and the run record are
otherwise unchanged.
