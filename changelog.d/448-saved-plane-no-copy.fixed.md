A run hands its output each plane without copying the field. `SavedPlane` copied an
observation's field, and again when the driver added the plane's statistics, so a handler
that only reduces a plane paid two field-sized copies and a peak of two volumes per plane.
`plane.field` is read-only and nothing changes it afterwards, so a handler can still keep
it; a `SavedPlane` constructed by hand copies, as before (#448).
