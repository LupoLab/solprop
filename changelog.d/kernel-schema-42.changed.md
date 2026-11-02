Kernel schema 42: `KernelProblem.initial.field0` may be a zero-stride broadcast view, and
every backend's `set_initial`, solprop-native's included, accepts a field of any strides
and copies it contiguous. No field or signature moves (#5).
