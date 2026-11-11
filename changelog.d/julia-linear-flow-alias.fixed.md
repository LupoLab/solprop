The optional Julia engine now refuses linear-flow and propagator buffers that overlap
through foreign array headers or alias retained preparation storage, before modifying
either input or output. Independent in-place and strided outputs remain supported
([#691](https://github.com/LupoLab/solprop/issues/691)).
