# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the private bridge in a fresh process before numerical qualification."""

import gc
import os

import numpy as np

from solprop.backends.julia._runtime import call_julia, runtime
from solprop.exceptions import BackendError


def main() -> None:
    """Check no-GIL threads, owning complex transfers, FFTs and exception causes."""
    runtime()
    import juliacall  # noqa: PLC0415 - after the isolated runtime is configured

    module = juliacall.newmodule("InteropSmoke")
    module.seval("""
        using FFTW, AcceleratedKernels
        own(x) = Matrix{ComplexF64}(x)
        function threaded(x)
            y = similar(x)
            AcceleratedKernels.foreachindex(x; max_tasks=Threads.nthreads()) do i
                y[i] = 2x[i]
            end
            y
        end
        function roundtrip(x)
            y = plan_fft(view(x,:,:),1)*x
            plan_ifft(view(y,:,:),1)*y
        end
        fail() = throw(ArgumentError("intentional bridge smoke test"))
    """)
    original = np.arange(24).reshape(6, 4).astype(complex) * (1 + 0.5j)
    owned = module.own(original)
    expected = original.copy()
    original[:] = 0
    gc.collect()
    module.seval("GC.gc()")
    np.testing.assert_array_equal(np.asarray(owned), expected)
    np.testing.assert_array_equal(
        np.asarray(call_julia(module.threaded, owned, nogil=True)), 2 * expected
    )
    transformed = np.asarray(call_julia(module.roundtrip, owned, nogil=True))
    # Two length-six transforms: normwise backward error scales with log N.
    bound = 32 * np.finfo(float).eps * np.log2(6) * np.linalg.norm(expected)
    assert np.linalg.norm(transformed - expected) <= bound
    try:
        call_julia(module.fail)
    except BackendError as error:
        assert error.__cause__ is not None and "intentional" in str(error)
    else:
        raise AssertionError("Julia exception was not translated")
    assert os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "yes"


if __name__ == "__main__":
    main()
