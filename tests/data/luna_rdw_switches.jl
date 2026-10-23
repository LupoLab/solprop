# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_rdw_switches.py. That script prepends a block
# of `const` definitions and runs the result under `julia --startup-file=no`; every name in
# capitals below is defined there. Read-only against the Luna checkout: this program calls
# Luna, writes nothing outside its own standard output, and changes nothing in the
# checkout. It is never run by the test suite.
#
# The full M2 case's per-mode energy along z, at every save plane, for three ladders:
#
#   * the plasma and the waveguide loss, switched on and off independently, which is what
#     says which term carries the two codes' disagreement in the higher modes;
#   * the transverse quadrature order, which says whether either code's rule is what does;
#   * the step tolerance over two decades, which says whether either code's stepping is.
#
# The reference row is the same run `luna_rdw.jl` makes, so its energies can be checked
# against the ones already stored in `luna_rdw_full.h5` (whose statistics were never
# trimmed) before any other row is believed.
using Luna
import Luna: Grid, Fields, Interface, PhysData
import LinearAlgebra: BLAS

emit(tag, values) = println(tag, "\t", join(string.(values), ","))

emit("CONSTANTS", [PhysData.c, PhysData.ε_0, PhysData.roomtemp])
# The Julia, BLAS and FFTW thread counts this run actually used, read here
# rather than assumed -- every `runtime_s` this script has ever recorded was taken
# with this same launch (`julia --startup-file=no`, no `-t`). The pool is encoded
# numerically since `emit` is numbers-only: 0 is "none" (a single Julia thread never
# raises the threading question, so `Luna.Utils._native_fftw_threads[]` stays
# `nothing`), 1 is libfftw3's native pthreads, 2 is FFTW.jl's Julia-task (partr)
# callback, 3 is "decided `false` for another reason".
let pool = if Threads.nthreads() == 1
        0
    elseif Luna.Utils._native_fftw_threads[] === true
        1
    elseif Luna.Utils._native_fftw_threads[] === false
        2
    else
        3
    end
    emit("THREADS", [Threads.nthreads(:default), BLAS.get_num_threads(),
                     Luna.Utils.FFTWthreads(), pool])
end

thegrid() = Grid.RealGrid(LENGTH, LAMBDA0, (LAMBDA_MIN, LAMBDA_MAX), TRANGE)

keywords(; kwargs...) = (; λ0 = LAMBDA0, τfwhm = TAU_FWHM, energy = ENERGY,
                         modes = MODES, trange = TRANGE,
                         λlims = (LAMBDA_MIN, LAMBDA_MAX), temperature = TEMPERATURE,
                         saveN = N_SAVE, shotnoise = false, kwargs...)

"""Emit one run's per-mode energy along z, its cost and the rule it used.

The energy is evaluated from the saved spectra with Luna's own `energyfuncs`, which is
the functional its `Stats.energy` uses, so the reference row is comparable bin for bin
with the statistics in the stored witness.
"""
function emit_run(tag, output, runtime)
    _, energy_ω = Fields.energyfuncs(thegrid())
    Eω = output.data["Eω"]
    z = output.data["z"]
    stats = output.data["stats"]
    emit(tag * "_Z", z)
    for m = 1:MODES
        emit("$(tag)_E_$(m)", [energy_ω(Eω[:, m, i]) for i in eachindex(z)])
    end
    emit(tag * "_RUNTIME", [runtime])
    emit(tag * "_STEPS", [length(stats["z"])])
    # the realised node count, so that "the matching order" is a measurement rather than
    # an assumption about two codes' rounding of nr to a Kronrod extension
    emit(tag * "_POINTS", [stats["transverse_points"][1]])
end

"Run the full case with one pair of switches."
function switch_case(tag, plasma, loss)
    started = time()
    output = prop_capillary(RADIUS, LENGTH, GAS, (P_IN, P_OUT);
                            keywords(plasma = plasma, loss = loss)...)
    emit_run(tag, output, time() - started)
end

"Run the reference case at one transverse quadrature order."
function quadrature_case(tag, nr)
    started = time()
    output = prop_capillary(RADIUS, LENGTH, GAS, (P_IN, P_OUT); keywords(nr = nr)...)
    emit_run(tag, output, time() - started)
end

"""Run the reference case at one step tolerance.

Driven through `prop_capillary_args` and `Luna.run` rather than `prop_capillary`, because
the interface cannot set a tolerance at all -- the same path the witness
generator's tolerance measurement used.
"""
function tolerance_case(tag, rtol, plasma)
    started = time()
    Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(
        RADIUS, LENGTH, GAS, (P_IN, P_OUT); keywords(plasma = plasma)...)
    Luna.run(Eω, grid, linop, transform, FT, output; rtol = rtol, atol = ATOL)
    emit_run(tag, output, time() - started)
end

switch_case("REFERENCE", true, true)
switch_case("NOPLASMA", false, true)
switch_case("NOLOSS", true, false)
switch_case("NEITHER", false, false)
for nr in QUADRATURE_ORDERS
    quadrature_case("NR$(nr)", nr)
end
for exponent in TOLERANCE_EXPONENTS
    tolerance_case("RTOL$(exponent)", 10.0^(-exponent), true)
end
# The control that separates a step-count-bound mechanism from the plasma's own
# stiffness: the same tolerance ladder with the plasma off. Luna's right-endpoint
# linear operator and its per-step re-apodisation both scale with the accepted-step
# count and are both live on this branch -- it carries no `Boundaries.jl`, so `stepfun`
# multiplies `ωwin` and `twin` into the state after every accepted step -- so if either
# is what moves the higher modes, it moves them here too.
for exponent in TOLERANCE_EXPONENTS
    tolerance_case("NOPLASMA_RTOL$(exponent)", 10.0^(-exponent), false)
end
