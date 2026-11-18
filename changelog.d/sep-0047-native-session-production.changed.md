The native backend runs every run it admits in its own compiled session, with no Python
step loop
([SEP 45](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0045-native-rk-march.md),
[SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)).
Beside the Adams march it already had, it now steps the adaptive Runge–Kutta loop of
`RungeKutta()` and the low-storage replay (`LowStorageRungeKutta`) natively, and the
session also runs resumed runs, the noise models, dense output (`interpolate_planes`),
the temporal boundary monitor, a free-space run's transverse edge, the Adams carrier
frame on an operator stored as a z table (#255), and coupled flows wider than 16
channels or with bins off their eigenbasis route, which 0.2.0 stepped with solprop's
Python march; such a flow's step whose exponent would need more than 30 squarings is
refused by name. A run's backend metadata says which loop stepped it (`adams_march`,
`rk_march`), and no built-in configuration issues `HostMarchWarning` any more; the class
stays importable for warning filters. Some results move at the rounding level, because
the Runge–Kutta error norm, the split collar's crossing limit and the transverse edge's
safety-ramp shares are summed in another order: in a near-tie a split-scheme run's
collar bucket can move one place, and an edge warning or stop may differ from
`numpy_ref`'s for a share within a rounding bound of its threshold.
