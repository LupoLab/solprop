Smaller speed-ups, each with the same results. Building a graded-index guide evaluates
its material once per mode solve rather than once per finite element, and a large
free-space launch forms each distinct column's admittance once, sums its column energies
and copies its admittance in threads (the `grin_nonlinear` build and translation from
24 s to 16 s on one core, a 384-node Kerr beam's launch from 12.6 s to 7.1 s). A run's
fixed host cost is smaller: the first statistics no longer import `scipy.signal`
(0.7 s), opening a run's output no longer serialises and digests its configuration a
second time, and package versions are read once per process
([#347](https://github.com/LupoLab/solprop/issues/347)). Host passes too small to
thread no longer read the CPU topology first (#556). On the native backend, free space's
separable stage propagators are formed a tile at a time through its vector maths rather
than one libm `sincos` per element (`line` 0.89 of its time at one thread, 0.93 at 16;
[#338](https://github.com/LupoLab/solprop/issues/338)); the transverse-edge reads on
radial grids synthesise their rows in one matrix product (the 256- and 512-node radial
filaments 0.83 and 0.75 of their one-thread windows on an Apple M5 Max); single-mode
problems run their synthesis and projection on the response team with Apple's
Accelerate as with OpenBLAS; and on Arm and other non-x86-64 hosts the plasma column's
band filter sums each sample's taps in one unrolled pass (the plasma cases' windows 1-6 %
faster on an Apple M5 Max, #560).
