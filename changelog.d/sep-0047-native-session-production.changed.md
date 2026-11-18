The native backend runs every run it admits in its own compiled session, with no Python
step loop
([SEP 45](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0045-native-rk-march.md),
[SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)).
Beside the Adams march it already had, it now steps the adaptive Runge–Kutta loop of
`RungeKutta()` and the low-storage replay (`LowStorageRungeKutta`) natively, and the
session also runs resumed runs (carrying the Adams history and the call ledgers, and
exporting the Adams history beside each saved plane and checkpoint), the input and
modified noise models (also on a propagator stored as a z table), dense output
(`interpolate_planes`, interpolated where the state lives for Adams and the embedded
Runge–Kutta tableaux alike), the temporal boundary monitor and a free-space run's
transverse edge (launch judgement, read spacing, reads, a requested collar's current and
return, warnings, stops and records), the Adams carrier frame
(`Multistep(carrier_frame="scalar")`) on an operator stored as a z table (#255), and
coupled flows wider than 16 channels or with bins off their eigenbasis route, which 0.2.0
stepped with solprop's Python march. Such a bin takes its matrix functions from the
scaled Taylor series and squaring of the augmented exponential rather than SciPy's
`expm`, and a step whose exponent there would need more than 30 squarings is refused by
name. A run's backend metadata says which loop stepped it (`adams_march`, `rk_march`),
and no built-in configuration issues `HostMarchWarning` any more; the class stays
importable for warning filters. Some results move at the rounding level: the
Runge–Kutta error norm, weighted or not, and the split collar's crossing limit now sum in
one sequential order on both backends (in a near-tie a split-scheme run's collar bucket
can move one place), and the transverse edge's safety-ramp shares are summed natively in
another order, within a rounding bound that accumulates across reads, so a warning or
stop may differ from `numpy_ref`'s for a share within that bound of its threshold. A
signal during a run is handled once the native call returns, within 256 attempts or
0.1 s, and the run's checkpoint includes every attempt made; under warning filters that
raise a boundary warning, the run fails at that endpoint with its field available for
the checkpoint, and a failure partway through a low-storage step is recorded against the
latest running checkpoint.
