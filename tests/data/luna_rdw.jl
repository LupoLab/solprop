# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_rdw.py. That script prepends a block of
# `const` definitions and runs the result under `julia --startup-file=no`; every name in
# capitals below is defined there. Read-only against the Luna checkout: this program
# calls Luna, writes its output files where it is told to, and changes nothing in the
# checkout. It is never run by the test suite.
using Luna
import Luna: Grid, Capillary, PhysData, Output
import Luna.PhysData: wlfreq
import LinearAlgebra: BLAS

emit(tag, values) = println(tag, "\t", join(string.(values), ","))

# Both codes take their constants from a CODATA release, and the two releases differ;
# emitting Luna's lets the comparison say which of the two it is looking at rather than
# widening a tolerance to cover both.
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
# The Ermolov helium expansion is referenced to 1 atm and Luna divides it by the 1 bar
# density. Emitting both densities lets the Python side reconstruct Luna's variant
# exactly, as the reference state of the same fit, instead of tabulating it.
emit("HE_DENSITIES", [PhysData.dens_1bar_0degC[:He], PhysData.dens_1atm_0degC[:He]])

"Run one case and emit the grid and the media Luna used, sampled on that grid."
function case(tag, radius, flength, pressure, modes, λlims, trange, n_save, path)
    grid = Grid.RealGrid(flength, LAMBDA0, λlims, trange)
    emit(tag * "_OMEGA", grid.ω)
    emit(tag * "_T", grid.t)
    emit(tag * "_GRID", [length(grid.t), length(grid.to), length(grid.ω), grid.ω[2]])

    # The two media the comparison replaces on the solprop side, sampled on the run's
    # own frequency axis. The DC bin is skipped: a wavelength is not defined there and
    # neither model is evaluated there by the propagation either (Luna's `sidx`).
    λ = [w > 0 ? wlfreq(w) : NaN for w in grid.ω]
    γ = PhysData.sellmeier_gas(GAS)
    emit(tag * "_GAMMA", [isnan(l) ? 0.0 : γ(l * 1e6) for l in λ])
    cladding = PhysData.ref_index_fun(:SiO2)
    clad = [isnan(l) ? complex(1.0) : complex(cladding(l)) for l in λ]
    emit(tag * "_CLAD_RE", real.(clad))
    emit(tag * "_CLAD_IM", imag.(clad))

    # The density path the modes and the responses both see, at the saved planes, so
    # that the two equations of state can be compared rather than assumed equal.
    _, dens = Capillary.gradient(GAS, flength, pressure...; T = TEMPERATURE)
    z = collect(range(0, flength, length = n_save))
    emit(tag * "_DENSITY", [dens(zi) for zi in z])
    emit(tag * "_Z", z)

    started = time()
    prop_capillary(radius, flength, GAS, pressure;
                   λ0 = LAMBDA0, τfwhm = TAU_FWHM, energy = ENERGY,
                   modes = modes, trange = trange, λlims = λlims,
                   temperature = TEMPERATURE, saveN = n_save,
                   shotnoise = false, filepath = path)
    emit(tag * "_RUNTIME", [time() - started])
    emit(tag * "_AEFF", [Modes.Aeff(Capillary.MarcatiliMode(radius, GAS, pressure[1];
                                                            T = TEMPERATURE, m = m))
                         for m = 1:modes])
end

case("REDUCED", REDUCED_RADIUS, REDUCED_LENGTH, (REDUCED_P_IN, REDUCED_P_OUT),
     REDUCED_MODES, (REDUCED_LAMBDA_MIN, REDUCED_LAMBDA_MAX), REDUCED_TRANGE,
     REDUCED_N_SAVE, REDUCED_PATH)
case("FULL", FULL_RADIUS, FULL_LENGTH, (FULL_P_IN, FULL_P_OUT),
     FULL_MODES, (FULL_LAMBDA_MIN, FULL_LAMBDA_MAX), FULL_TRANGE, FULL_N_SAVE,
     FULL_PATH)
