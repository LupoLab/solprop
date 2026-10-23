# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Read-only Luna witness, launched only by the adjacent Python generator.
using Luna
import Luna: Grid, Capillary, PhysData, Modes, Nonlinear, NonlinearRHS, LinearOps, Fields, Output
import LinearAlgebra: norm

emit(tag, values) = println(tag, "\t", join(string.(values), ","))
complex_emit(tag, values) = (emit(tag * "_RE", real.(vec(values))); emit(tag * "_IM", imag.(vec(values))))
emit("CONSTANTS", [PhysData.c, PhysData.ε_0, PhysData.e_ratio, NonlinearRHS.nlscale])
grid = Grid.RealGrid(LENGTH, WAVELENGTH, (LAMBDA_MIN, LAMBDA_MAX), TRANGE; ffac=6)
emit("T", grid.t); emit("TO", grid.to); emit("OMEGA", grid.ω)
emit("OMEGA_WINDOW", grid.ωwin); emit("T_WINDOW", grid.twin); emit("TO_WINDOW", grid.towin)
emit("ACTIVE", Int.(grid.sidx))
inputs = Fields.GaussField(λ0=WAVELENGTH, τfwhm=FWHM, energy=ENERGY)

function analytic_rate(scale)
    function rate!(out, field)
        out .= scale .* (abs.(field) ./ RATE_REFERENCE).^2
    end
end

function case(tag, gases, pressures, plasma)
    mode = Capillary.MarcatiliMode(RADIUS, gases, pressures; loss=false, T=TEMPERATURE)
    aeff(z) = Modes.Aeff(mode; z=z)
    rho = gases isa Symbol ? PhysData.density(gases, pressures, TEMPERATURE) :
        [PhysData.density(gas, pressure, TEMPERATURE) for (gas, pressure) in zip(gases, pressures)]
    densityfun(z) = rho
    if gases isa Symbol
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gases)),)
        emit(tag * "_DENSITY", [rho]); emit(tag * "_GAMMA3", [PhysData.γ3_gas(gases)])
    else
        responses = Tuple(plasma ?
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, analytic_rate(RATE_SCALES[index]), POTENTIALS[index])) :
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            for (index, gas) in enumerate(gases))
        emit(tag * "_DENSITY", rho); emit(tag * "_GAMMA3", [PhysData.γ3_gas(gas) for gas in gases])
    end
    linop, beta!, beta1, alpha = LinearOps.make_const_linop(grid, mode, WAVELENGTH)
    initial, transform, FT = Luna.setup(grid, densityfun, responses, inputs, beta!, aeff)
    complex_emit(tag * "_INITIAL", initial)
    complex_emit(tag * "_LINEAR", linop)
    prefactor = ones(ComplexF64, length(grid.ω))
    transform.norm!(prefactor, 0.0)
    prefactor[.!grid.sidx] .= 0
    complex_emit(tag * "_PREFACTOR", prefactor)
    emit(tag * "_AEFF", [aeff(0.0)])
    emit(tag * "_BETA1", [beta1])
    finals = Vector{Vector{ComplexF64}}()
    for steps in STEPS
        counter = Ref(0)
        statsfun(field, z, dz) = (counter[] += 1; nothing)
        output = Output.MemoryOutput(0, LENGTH, 2, statsfun)
        started = time()
        Luna.run(initial, grid, linop, transform, FT, output; init_dz=LENGTH/steps,
                 min_dz=LENGTH/steps, max_dz=LENGTH/steps, rtol=RTOL, atol=ATOL,
                 status_period=60)
        final = vec(output.data["Eω"][:, end])
        complex_emit(tag * "_FINAL_" * string(steps), final)
        emit(tag * "_RUN_" * string(steps), [counter[], time() - started])
        push!(finals, final)
    end
    return finals
end
# Scratch's public override keeps FFT wisdom and lock files in the witness directory.
parentmodule(Luna.Utils.get_scratch!).with_scratch_directory(CACHE_PATH) do
rho = PhysData.density(:Ar, 5.0, TEMPERATURE)
halfpressure = PhysData.pressure(:Ar, rho / 2, TEMPERATURE)
single = case("SINGLE", :Ar, 5.0, false)
split = case("SPLIT", (:Ar, :Ar), (halfpressure, halfpressure), false)
emit("SPLIT_DIFFERENCE", [norm(split[i] - single[i]) / norm(single[i]) for i in eachindex(single)])
case("PLASMA", (:Ar, :Ne), (1.0, 2.0), true)

end
