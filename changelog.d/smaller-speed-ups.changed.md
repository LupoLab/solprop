Smaller speed-ups, each with the same results. Building a graded-index guide evaluates
its material once per mode solve rather than once per finite element (the
`grin_nonlinear` build and translation from 24 s to 16 s on one core), and a large
free-space launch forms each distinct column's admittance once, in threads (a 384-node
Kerr beam's launch from 12.6 s to 7.1 s). A run's fixed host cost is smaller: the first
statistics no longer import `scipy.signal` (0.7 s), opening a run's output no longer
digests its configuration a second time, and package versions are read once per process
([#347](https://github.com/LupoLab/solprop/issues/347)). Host passes too small to thread
no longer read the CPU topology first (#556). On the native backend, free space's
separable stage propagators go through its vector maths (`line` 0.89 of its time at one
thread; [#338](https://github.com/LupoLab/solprop/issues/338)); radial transverse-edge
reads synthesise their rows in one matrix product (radial filaments down to 0.75 of
their one-thread windows on an Apple M5 Max); single-mode problems use Apple's
Accelerate as they use OpenBLAS; and on Arm hosts the plasma column's band filter sums
its taps in one pass (the plasma cases 1-6 % faster on an Apple M5 Max, #560).
