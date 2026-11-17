A `SimulationRunner`, and the backend a scan keeps between points, is no longer reused
for settings that name another backend or node budget: a scan point builds a backend of
its own, and a direct `runner.run` after such a change raises a `DriverError` before it
reads the input ([#705](https://github.com/LupoLab/solprop/issues/705)).
