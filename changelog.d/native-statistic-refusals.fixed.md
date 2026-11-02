A boundary monitor built for another grid is now refused by name on every backend, instead
of a bare `ValueError` on `native` and an accident in `matmul` on `numpy_ref`. A direct call
to `native`'s `stats` now checks every route for names the route doesn't define, and refuses
a general flux statistic under a weight other than one rather than ignoring the weight.
