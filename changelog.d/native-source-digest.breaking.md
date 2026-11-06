`KERNEL_SCHEMA_VERSION` and `KernelProblem.schema_version` are gone, and with them the
kernel schema number, and
`solprop_native.BuildInfo.schema_version` is now `source_digest`, the SHA-256 of the C++
sources and `CMakeLists.txt` the extension was compiled from: a native run's record names it
in `backend.settings` in place of the number. Neither enters a configuration hash or a resume
check. `solprop_native.check_build()` takes no problem, and in an editable checkout refuses
an extension built from other sources than `native/`, naming `uv sync --reinstall-package
solprop-native` (ADR-068). To migrate, read `build_info().source_digest` where code read
`schema_version`, call `check_build()` with no argument, and drop
`solprop.translate.KERNEL_SCHEMA_VERSION` from imports and checks: solprop-native is pinned
at solprop's own version, which is now the whole contract.
