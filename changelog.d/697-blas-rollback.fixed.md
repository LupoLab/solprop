Preserved the original error and retained a retry owner when acquiring host BLAS
limits and rolling them back both fail. The next acquisition retries the rollback
before applying new limits, so a failed request no longer leaves a BLAS pool at its
width indefinitely ([#697](https://github.com/LupoLab/solprop/issues/697)).
