`KERNEL_SCHEMA_VERSION`, which ended at 45, and `KernelProblem.schema_version` are gone, and
`solprop_native.BuildInfo.schema_version` is now `source_digest`, the SHA-256 of the C++
sources and `CMakeLists.txt` the extension was compiled from: a native run's record names it
in `backend.settings` in place of the number. Neither enters a configuration hash or a resume
check. `solprop_native.check_build()` takes no problem, and in an editable checkout refuses
an extension built from other sources than `native/`, naming `uv sync --reinstall-package
solprop-native` (ADR-068).
