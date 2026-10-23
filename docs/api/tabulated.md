# Tabulated modes

{mod}`solprop.modes.tabulated` builds waveguide modes from arrays of effective index,
for a guide solved elsewhere (a finite-element solver, a measurement) or sampled from
one of solprop's own modes. Its public names are:

- {class}`~solprop.modes.tabulated.IndexTable`: a passive mode's complex effective index,
  and optionally its effective area, tabulated against frequency and interpolated;
- {class}`~solprop.modes.tabulated.TabulatedHollowMode`: an evacuated hollow guide's table
  with a gas fill added at a fixed transverse wavenumber;
- {class}`~solprop.modes.tabulated.TabulatedMode`: a filled table's whole linear
  response, with Marcatili transverse fields;
- {func}`~solprop.modes.tabulated.sample_index_table`: sample an existing mode into a
  dense table;
- {data}`~solprop.modes.tabulated.MIN_TABLE_POINTS`: the fewest samples (four) a table
  may have, which the cubic interpolant needs. {mod}`solprop.nonlinear.gnlse` uses the
  same interpolant for its tabulated dispersion and re-exports the constant.

Use the [tabulated-mode guide](../guide/waveguides/tabulated-modes.md) to choose
between an evacuated and a filled table, inspect interpolation error, and ensure
the full active band lies within the data. The private shared base is not a public
constructor.
