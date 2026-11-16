Every route now follows one rule for the saved planes and the end of a run
([SEP 51](docs/dev/seps/archive/sep-0051-save-planes.md)). A run marches from `initial_z` to its
length, `length` or `grid.length`, and always records both ends. `n_save=N` gives `N`
evenly spaced observations over that interval, both ends included, and
`save_planes=[...]` lists positions inside it. `SimulationSettings` gains `n_save`, its
`save_planes` becomes optional (given neither, a run records 201), and a recipe's
object-route `[build.settings]` takes `n_save` too. Migration:
`save_planes=80` is refused with "pass `n_save=80`" in Python and "write `n_save = 80`"
in a recipe; `save_planes=0.25` must be `[0.25]`; `n_save` and `save_planes` together,
`n_save` below 2, and positions before the entrance or beyond the end are refused; a
position within eight ulps of an endpoint is that endpoint. Positions short of the length
no longer end the run: it continues to the length and records it, so indices counted
from the end (`field_planes=[-1, …]`) may name other planes, and a short window of a
longer setup is now a shorter grid, with `boundary_length` pinned to keep the absorber.
On the keyword routes `length` may be `None` beside `grid=`, which then supplies it, as
in `prop_capillary(radius, None, "Ar", grid=grid, ...)`; it is still passed, so `gas`
stays required. Given beside `grid=`, the two must agree, and a call with neither is
refused; a recipe may leave `length` out beside a grid. `n_save` beside an interior
launch counts over
`[initial_z, length]`. A replayed `FixedSchedule` must start at the entrance and end at
the length, within the rounding its step sum accumulates, or it is refused before the
run. The resolved recipe writes the run's positions in `[build.settings]` on an object
route and drops `n_save`.
In the front end, every sheet's *Saved planes* row offers *Evenly spaced* (`n_save`) and
*Explicit positions* (`save_planes`), and typing into one form removes the other in the
same undoable edit. A setup opens on the form it holds; the object sheets no longer turn
a count into a list of positions, nor rewrite the positions when the length changes. An
object-sheet count of 80 planes, which left out the entrance, is `n_save = 81`.
