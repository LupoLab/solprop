Selecting the native backend with a malformed process-start setting, such as a
non-numeric `SOLPROP_THREADS`, now raises a `BackendError` carrying the setting's
message instead of a bare `ValueError` or `RuntimeError` from the package import, and
`solprop.backends.available()` is documented as listing the installed backends.
