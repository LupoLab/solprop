In the desktop front end, every sheet's *Saved planes* row offers *Evenly spaced*
(`n_save`) and *Explicit positions* (`save_planes`), and typing into one form removes
the other in the same undoable edit. A setup opens on the form it holds; the object
sheets no longer turn a count into a list of positions, nor rewrite the positions when
the length changes. An object-sheet count of 80 planes, which left out the entrance, is
`n_save = 81`.
