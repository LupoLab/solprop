A published-install check after each PyPI release, on all four wheel platforms:
fresh CPython 3.14 environments resolve the native extra with pip and verify its
version, dependencies and default backend. The same check can be run by hand for a
named release, with an optional run of its tagged README example and retained logs.
