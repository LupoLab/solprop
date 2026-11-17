The `ModeCollection` factories share one layout: the geometry, then `gas` and
`pressure` (optional, as on the modes), then keyword-only options, with `temperature=None`
(room temperature, or the mixture's own) and `mixture=` on all of them. `he1m` and
`vincetti` take `count=` by keyword (default one), `slab` takes `orders=` by keyword
(default `(1,)`) after the fill, and `marcatili` takes `kinds=`, `n=` and `m=` by keyword:
write `ModeCollection.slab(63.5e-6, "Ar", 1.0, orders=(1, 3, 5))` for
`ModeCollection.slab(63.5e-6, (1, 3, 5), "Ar", 1.0)` and
`ModeCollection.he1m(125e-6, "He", 0.8, count=4)` for
`ModeCollection.he1m(125e-6, "He", 0.8, 4)`.
