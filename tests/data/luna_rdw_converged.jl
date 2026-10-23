# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_rdw_converged.py. That script prepends a
# block of `const` definitions and runs the result under `julia --startup-file=no`; every
# capitalised name below is defined there. Read-only against the Luna checkout: this
# program calls Luna, writes one HDF5 file at the path it is given, and changes nothing
# in the checkout. It is never run by the test suite.
#
# The full M2 case at a step tolerance two and a half decades tighter than the one every
# published Luna result carries. It is driven through `prop_capillary_args` and
# `Luna.run` rather than `prop_capillary`, because Luna's interface cannot set a
# tolerance at all -- the same path the switches generator's tolerance
# ladder uses.
#
# Why: the M2 regression's higher-mode metrics compare solprop converged against Luna
# *unconverged*, so a disagreement there has never been separable into "the two codes
# differ" and "Luna has not converged". This run is the other half.
using Luna
import Luna: Interface, PhysData
import LinearAlgebra: BLAS

emit(tag, values) = println(tag, "\t", join(string.(values), ","))

emit("CONSTANTS", [PhysData.c, PhysData.ε_0, PhysData.roomtemp])
emit("THREADS", [Threads.nthreads(), BLAS.get_num_threads()])

keywords() = (; λ0 = LAMBDA0, τfwhm = TAU_FWHM, energy = ENERGY,
              modes = MODES, trange = TRANGE,
              λlims = (LAMBDA_MIN, LAMBDA_MAX), temperature = TEMPERATURE,
              saveN = N_SAVE, shotnoise = false, filepath = OUTPUT_PATH)

started = time()
Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(
    RADIUS, LENGTH, GAS, (P_IN, P_OUT); keywords()...)
Luna.run(Eω, grid, linop, transform, FT, output; rtol = RTOL, atol = ATOL)
emit("RUNTIME", [time() - started])
# The accepted-step count is read from the file afterwards, not from the output object:
# `Luna.run` hands back an `Output.HDF5Output` whose contents are reached by indexing
# rather than by a `.data` field, and getting that wrong here would throw away the whole
# run at its last line -- which is exactly what it did once.
