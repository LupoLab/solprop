`estimate_memory` and the admission now count the native backend's own buffers on the
Cartesian and line routes (`solprop.backends.memory.free_space_memory`): its statistics and
absorber buffers per chunk, a local flux statistic's magnetic closure, and the Runge-Kutta
march's rows. The reference's count fell
short by up to 46 % on an envelope grid and 80 % on a field grid when one chunk held the
whole grid, so a native run could be admitted and then exceed its budget. A constructed
Kerr beam's ledger and estimate now agree to the byte, with `default_spec("free")` or no
statistics; a boundary monitor's weights, bound when a run starts, are not counted (#7).
