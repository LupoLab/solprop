On a GPU build (CUDA or HIP) of the native backend, runs take the same compiled session
as on a CPU build
([SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)):
Adams and the embedded Runge–Kutta tableaux, adaptive or replayed, the low-storage
replay, dense output (`interpolate_planes`), the modal, general and spatial routes
(radial, Cartesian, line and planar, with their statistics and fluence widths), saved
planes, periodic statistics rows, step planes, the temporal boundary monitor and a
free-space run's transverse edge with a requested collar's current, the input and
modified noise models and resumed Adams history. A coupled flow wider than 16 channels
or with bins off its eigenbasis runs on the native Adams march, rather than solprop's
Python march with a `HostMarchWarning`, so its fields differ at the rounding level, and a
step whose exponent would need more than 30 squarings is refused by name, as on a CPU
build. On an NVIDIA A40 (CUDA) the session takes the same steps, statistics and fields as
the backend's previous march on the same device (dense Runge–Kutta planes and the
transverse edge's ramp sums within rounding bounds); device builds are not yet qualified
on either vendor. A saved plane's field crosses from the device once, into its
destination, with no extra host copy. Device builds refuse the Adams carrier frame by
name.
