Correct the 0.1.0 limitation statement about `mlx`: its custom Metal kernels already
run in continuous integration's macOS default lane. That lane now requires Metal,
so losing the device fails the job rather than silently skipping the kernel tests.
MLX coverage, runtime type checks and an explicit CPU-device pass remain absent.
