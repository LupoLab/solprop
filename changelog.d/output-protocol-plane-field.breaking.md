The `Output` protocol now includes `plane_field(index, convention="optics")`, which
`solprop.processing` uses to read one saved plane: a handler of your own implements it
(returning shape `(n_modes, n_pol, n_omega)`) beside `fields`. `CallbackOutput` refuses it
by name. `processing.axes.layout` reads the shape from the store without reading the
field, and counts every saved plane.
