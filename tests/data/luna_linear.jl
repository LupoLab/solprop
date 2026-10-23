# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_linear.py. That script prepends a
# block of `const` definitions and runs the result under `julia --startup-file=no`;
# every name in capitals below is defined there. Read-only against the Luna
# checkout: this program calls Luna and prints, and changes nothing.
using Luna
import Luna: Grid, Capillary, Modes, LinearOps, NonlinearRHS, Nonlinear, SimpleFibre
import Luna: PhysData

emit(tag, values) = println(tag, "\t", join(string.(values), ","))
emitc(tag, values) = (emit(tag * "_RE", real.(values)); emit(tag * "_IM", imag.(values)))

# Luna takes its constants from PhysicalConstants.CODATA2014; solprop takes them
# from scipy.constants, which is CODATA 2022. Emitting them lets the comparison
# account for the difference instead of loosening a tolerance over it.
emit("CONSTANTS", [PhysData.ε_0, PhysData.c, PhysData.μ_0])

# ---------------------------------------------------------------- capillary linop
capillary = Capillary.MarcatiliMode(CORE_RADIUS, GAS, PRESSURE; model=:full, loss=true)
emit("CAP_AEFF", [Modes.Aeff(capillary)])

grids = Dict(
    "REAL" => Grid.RealGrid(CAP_LENGTH, CAP_LAMBDA0, CAP_LIMS, CAP_TRANGE),
    "ENV" => Grid.EnvGrid(CAP_LENGTH, CAP_LAMBDA0, CAP_LIMS, CAP_TRANGE),
)
for tag in ("REAL", "ENV")
    grid = grids[tag]
    linop, betafun!, beta1, alphafun! = LinearOps.make_const_linop(
        grid, capillary, CAP_LAMBDA0)
    beta = zero(grid.ω); betafun!(beta, 0.0)
    alpha = zero(grid.ω); alphafun!(alpha, 0.0)
    emit("OMEGA_" * tag, grid.ω)
    emit("SIDX_" * tag, Int.(grid.sidx))
    emit("BETA_" * tag, beta)
    emit("ALPHA_" * tag, alpha)
    emit("BETA1_" * tag, [beta1])
    emit("BETA0_" * tag, [Modes.β(capillary, PhysData.wlfreq(CAP_LAMBDA0))])
    emitc("LINOP_" * tag, linop)
    norm = ones(ComplexF64, length(grid.ω))
    NonlinearRHS.norm_mode_average(
        grid, betafun!, z -> Modes.Aeff(capillary); shock=true)(norm, 0.0)
    emitc("NORM_" * tag, norm)
end

# ------------------------------------------------------------------- GNLSE fibre
simple = SimpleFibre.SimpleMode(PhysData.wlfreq(GN_LAMBDA0), BETAS;
                                Aeff=GN_AEFF, loss=GN_LOSS_DB)
gngrid = Grid.EnvGrid(GN_LENGTH, GN_LAMBDA0, GN_LIMS, GN_TRANGE)
gnlinop, gnbeta!, gnbeta1, gnalpha! = LinearOps.make_const_linop(
    gngrid, simple, GN_LAMBDA0)
emit("GN_OMEGA", gngrid.ω)
emit("GN_SIDX", Int.(gngrid.sidx))
emit("GN_BETA", [Modes.β(simple, w) for w in gngrid.ω])
emit("GN_ALPHA", [Modes.α(simple, w) for w in gngrid.ω])
emit("GN_BETA1", [gnbeta1])
emit("GN_ALPHA_SCALAR", [simple.α])
emitc("GN_LINOP", gnlinop)
for (suffix, shock) in (("", true), ("_NOSHOCK", false))
    norm = ones(ComplexF64, length(gngrid.ω))
    NonlinearRHS.norm_mode_average_gnlse(gngrid, z -> GN_AEFF; shock=shock)(norm, 0.0)
    emitc("GN_NORM" * suffix, norm)
end

# ------------------------------------------------------------ coefficient mapping
for i = 1:3
    gamma, aeff, fr = MAPPINGS[3i-2], MAPPINGS[3i-1], MAPPINGS[3i]
    k0 = 2pi / GN_LAMBDA0
    n2 = gamma / k0 * aeff
    chi3 = 4 / 3 * (1 - fr) * n2 * (PhysData.ε_0 * PhysData.c)
    emit("MAP_$i", [gamma, aeff, fr, n2, chi3])
end

# ------------------------------------------------------------------ Kerr responses
n = length(TAXIS)
scalar_field = reshape(copy(FIELD_X), :, 1)
out = zeros(size(scalar_field))
Nonlinear.Kerr_field(GAMMA3)(out, scalar_field, DENSITY)
emit("KERR_FIELD", vec(out))

out = zeros(size(scalar_field))
Nonlinear.KerrFieldNoTHG(GAMMA3)(out, scalar_field, DENSITY)
emit("KERR_FIELD_NOTHG", vec(out))

vector_field = hcat(FIELD_X, FIELD_Y)
out = zeros(size(vector_field))
Nonlinear.Kerr_field(GAMMA3)(out, vector_field, DENSITY)
emit("KERR_FIELD_VEC_X", out[:, 1])
emit("KERR_FIELD_VEC_Y", out[:, 2])

scalar_env = reshape(copy(ENV_X), :, 1)
out = zeros(ComplexF64, size(scalar_env))
Nonlinear.Kerr_env(GAMMA3)(out, scalar_env, DENSITY)
emitc("KERR_ENV", vec(out))

vector_env = hcat(ENV_X, ENV_Y)
out = zeros(ComplexF64, size(vector_env))
Nonlinear.Kerr_env(GAMMA3)(out, vector_env, DENSITY)
emitc("KERR_ENV_VEC_X", out[:, 1])
emitc("KERR_ENV_VEC_Y", out[:, 2])

out = zeros(ComplexF64, n)
Nonlinear.Kerr_env_thg(GAMMA3, KERR_OMEGA0, TAXIS)(out, copy(ENV_X), DENSITY)
emitc("KERR_ENV_THG", out)
