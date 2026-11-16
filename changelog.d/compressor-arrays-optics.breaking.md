The array form of the compressors (`grid=`, `spectrum=` of `compress_taylor`,
`compress_material`, `compress_mirrors`, `compress_gratings`, `compress_prisms`,
`transform_limit` and `optimise_cep`) now takes the spectrum in the optics convention, the
one `output["field"]` and `plane_field` return, so `compress_taylor(grid=grid,
spectrum=run.plane_field(-1))` finds what `compress_taylor(run)` finds. A spectrum from a
pulse's `build(grid)` is internal: pass `fourier.spectrum_to_optics(spectrum,
kind=grid.kind)`.
