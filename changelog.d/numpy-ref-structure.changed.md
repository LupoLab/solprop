The `numpy_ref` backend's internals are reorganised for reading. The passes are now
the `numpy_ref.passes` subpackage (`transforms`, `responses`, `plasma`, `stepping`),
and the workspace's source evaluation, statistics and coupled flow have modules of their
own (`source`, `statistics`, `coupled_flow`). The Adams engine, kernels and tables are
`multistep`, `adams_kernels` and `adams_tables`. The low-storage engine is
`numpy_ref.engines.NumpyLowStorage`. The package no longer re-exports the passes.
Selecting `backend="numpy_ref"` is unchanged.
