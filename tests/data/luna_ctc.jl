# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_ctc.py. That script prepends a block of
# `const` definitions and runs the result under `julia --startup-file=no`; every name in
# capitals below is defined there. Read-only against the Luna checkout: this program
# calls Luna, writes its output files where it is told to, and changes nothing in the
# checkout. It is never run by the test suite.
using Luna
import Luna: Grid, Capillary, PhysData, Modes, Ionisation, Raman
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
# The reference densities of the hydrogen susceptibility, so that the Python side can
# tell a normalisation apart from a different fit, as for helium, where the Ermolov
# expansion was referenced to 1 atm and divided by the 1 bar density.
emit("H2_DENSITIES", [PhysData.dens_1bar_0degC[GAS], PhysData.dens_1atm_0degC[GAS]])
# Hydrogen's third-order hyperpolarisability, per particle: Luna's χ³ is this times the
# density, so one number fixes the whole nonlinearity.
emit("GAMMA3", [PhysData.γ3_gas(GAS)])

# Luna's molecular ionisation rate, the one a hydrogen run actually uses:
# `IonRateADK(gas)` takes the potential alone, so Z = 1 and l = 0, with an occupancy of
# two and a threshold. Sampled log-spaced across the field range the CtC case reaches,
# which straddles both the threshold and hydrogen's barrier-suppression field.
let ionpot = PhysData.ionisation_potential(GAS)
    rate = Ionisation.IonRateADK(GAS)
    field = 10 .^ range(log10(ADK_FIELD_MIN), log10(ADK_FIELD_MAX), length = ADK_SAMPLES)
    emit("ADK_IONPOT", [ionpot])
    emit("ADK_THRESHOLD", [Ionisation.ADK_threshold(ionpot)])
    emit("ADK_FIELD", field)
    emit("ADK_RATE", [rate(E) for E in field])
end

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

    # A uniform fill, so the density path is one number repeated; emitted at the saved
    # planes anyway, so that the two equations of state are compared rather than assumed
    # equal and so that the reader of the file cannot mistake which form it is.
    ρ = PhysData.density(GAS, pressure, TEMPERATURE)
    z = collect(range(0, flength, length = n_save))
    emit(tag * "_DENSITY", fill(ρ, length(z)))
    emit(tag * "_Z", z)

    # Luna's own Raman response, on the fine grid of this run and at its density. Only
    # the reduced case emits it: it is n_to samples long and the comparison that needs
    # it runs on the reduced case alone.
    if tag == "REDUCED"
        response = Raman.raman_response(grid.to, GAS; temp = TEMPERATURE)
        ht = zeros(length(grid.to))
        response(ht, ρ)
        emit(tag * "_RAMAN", ht)
    end

    started = time()
    prop_capillary(radius, flength, GAS, pressure;
                   λ0 = LAMBDA0, τfwhm = TAU_FWHM, energy = ENERGY,
                   modes = modes, trange = trange, λlims = λlims,
                   temperature = TEMPERATURE, saveN = n_save, loss = false,
                   shotnoise = false, filepath = path)
    emit(tag * "_RUNTIME", [time() - started])
    emit(tag * "_AEFF", [Modes.Aeff(Capillary.MarcatiliMode(radius, GAS, pressure;
                                                            T = TEMPERATURE, m = m))
                         for m = 1:modes])
end

case("REDUCED", REDUCED_RADIUS, REDUCED_LENGTH, REDUCED_PRESSURE, REDUCED_MODES,
     (REDUCED_LAMBDA_MIN, REDUCED_LAMBDA_MAX), REDUCED_TRANGE, REDUCED_N_SAVE,
     REDUCED_PATH)
case("FULL", FULL_RADIUS, FULL_LENGTH, FULL_PRESSURE, FULL_MODES,
     (FULL_LAMBDA_MIN, FULL_LAMBDA_MAX), FULL_TRANGE, FULL_N_SAVE, FULL_PATH)
