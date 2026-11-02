Guide and incident-plane construction, kernel translation and coupling now limit
NumPy's and SciPy's controllable BLAS pools to one thread, or `SOLPROP_BLAS_THREADS`
within the process budget, and restore their previous limits afterwards. This
prevents the thousands of small graded-index eigensolves from occupying every CPU
before a backend starts.
Vendor thread defaults are overridden only during construction; the native
backend's own BLAS keeps its fitted width.
