A finished file run is readable through the handler it returned: `prop_*(...,
output="run.h5")` and `Simulation.run` hand back a closed `HDF5Output` whose reads
(`run["field"]`, `fields()`, `plane_field()`, processing functions and plots) open the file
read-only for the one read and close it again, so no handle is held between reads. A
suspended writer still refuses, and the refusal now says why. Once another run has written or
continued a file at that path, the handler's read is refused, naming the file, and
`open_file` reads what the file now holds.
