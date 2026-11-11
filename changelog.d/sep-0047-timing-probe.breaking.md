The backend seam's `timed_step()` is replaced by an explicit timing probe
([SEP 47](docs/dev/seps/archive/sep-0047-run-ownership.md) section 6.6):
`Backend.timing_probe(start)` loads a `solprop.backends.session.Start` and returns a
`solprop.backends.protocol.TimingProbe`, whose `trial(z, dz)` takes one trial step that
commits nothing and whose `field()` reads that trial's end state back on the host.
`Simulation.estimate` and the SEP 46 calibration time and compare trials through it,
instead of loading the backend themselves and reaching private driver members; the
calibration's plan comparison on the native backend now compares the trial fields,
where it read the unchanged launch field before.
