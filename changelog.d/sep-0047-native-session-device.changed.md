On a GPU build (CUDA or HIP) of the native backend, runs take the compiled session of a
CPU build, with its coverage and its refusals
([SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)),
and device builds also refuse the Adams carrier frame by name. On an NVIDIA A40 (CUDA)
the session takes the same steps, statistics and fields as the backend's previous march
on the same device (dense Runge–Kutta planes and the transverse edge's ramp sums within
rounding bounds); device builds are not yet qualified on either vendor. A saved plane's
field crosses from the device once, into its destination, with no extra host copy.
