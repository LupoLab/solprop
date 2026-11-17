The batch script of a Slurm scan (`solprop.driver.executors.Slurm`) now sets
`OPENBLAS_THREAD_TIMEOUT=20` unless the job's environment sets it, as the front end's
Slurm runs already did, so the scan controller's BLAS threads sleep instead of spinning
between calls.
