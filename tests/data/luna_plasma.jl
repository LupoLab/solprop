# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_plasma.py. That script prepends a block of
# `const` definitions and runs the result under `julia --startup-file=no`; every name in
# capitals below is defined there. Read-only against the Luna checkout: this program
# calls Luna and prints, and changes nothing.
using Luna
import Luna: Nonlinear, Ionisation, PhysData

emit(tag, values) = println(tag, "\t", join(string.(values), ","))

# Luna takes its constants from PhysicalConstants.CODATA2014; solprop takes them from
# scipy.constants, which is CODATA 2022. The plasma carries no eps_0: the free-electron
# current is linear in e^2/m_e and the loss term is linear in the ionisation potential,
# which this program is *given* rather than looking up, so only e_ratio needs accounting
# for. Emitting it lets the comparison rescale instead of loosening a tolerance.
emit("CONSTANTS", [PhysData.e_ratio, PhysData.electron, PhysData.m_e])

# The analytic rate, spelled to match the Python side operation for operation: an eighth
# power with a hard threshold, carrying no atomic units at all, so a case that uses it
# compares the plasma arithmetic and nothing else.
function analytic!(out, E)
    m = abs.(E)
    out .= ifelse.(m .>= RATE_THRESHOLD, RATE_SCALE .* (m ./ RATE_REFERENCE) .^ 8, 0.0)
end

# The real thing, for one case: Luna's direct PPT with its own defaults. `modal-fixed`
# has no barrier-suppression correction, so the Python side asks for none.
function ppt!(out, E)
    out .= Ionisation.ionrate_PPT(GAS, WAVELENGTH, E)
end

nt = length(TAXIS)

"Emit every stage of the scalar kernel, and check the fused form against it."
function scalar_case(tag, E, ratefunc, ionpot, preionfrac)
    P = zeros(nt); J = zeros(nt); phase = zeros(nt)
    rate = zeros(nt); fraction = zeros(nt)
    Nonlinear._plasma_scalar_multipass!(P, J, phase, rate, fraction, E, ratefunc,
                                        ionpot, preionfrac, DT)
    emit(tag * "_RATE", rate)
    emit(tag * "_FRACTION", fraction)
    emit(tag * "_J", J)
    emit(tag * "_P", P)
    # The fused two-pass kernel the propagation actually runs claims to be bit-identical
    # to the reference above; emit it so that the claim is witnessed, not assumed.
    fused = zeros(nt); ratef = zeros(nt)
    Nonlinear._plasma_scalar!(fused, ratef, E, ratefunc, ionpot, preionfrac, DT)
    emit(tag * "_P_FUSED", fused)
end

"Emit the same for the vector kernel, component by component."
function vector_case(tag, Ex, Ey, ratefunc, ionpot, preionfrac)
    Em = hypot.(Ex, Ey)
    E = hcat(Ex, Ey)
    P = zeros(nt, 2); J = zeros(nt, 2); phase = zeros(nt, 2)
    rate = zeros(nt); fraction = zeros(nt)
    Nonlinear._plasma_vector_multipass!(P, J, phase, rate, fraction, E, Ex, Ey, Em,
                                        ratefunc, ionpot, preionfrac, DT)
    emit(tag * "_RATE", rate)
    emit(tag * "_FRACTION", fraction)
    emit(tag * "_J_X", J[:, 1])
    emit(tag * "_J_Y", J[:, 2])
    emit(tag * "_P_X", P[:, 1])
    emit(tag * "_P_Y", P[:, 2])
    fused = zeros(nt, 2); ratef = zeros(nt)
    Nonlinear._plasma_vector!(fused, ratef, Ex, Ey, Em, ratefunc, ionpot, preionfrac, DT)
    emit(tag * "_P_FUSED_X", fused[:, 1])
    emit(tag * "_P_FUSED_Y", fused[:, 2])
end

# Each case is run twice, the second time with a zero ionisation potential. That run is
# exactly the free-electron current, which is linear in e_ratio; subtracting it leaves
# the ionisation-loss part, which carries no constant at all once the potential is given
# rather than looked up. Each half can then be rescaled on its own.
scalar_case("SCALAR", FIELD, analytic!, IONPOT, 0.0)
scalar_case("SCALAR_FREE", FIELD, analytic!, 0.0, 0.0)
scalar_case("PREIONISED", FIELD, analytic!, IONPOT, PREIONFRAC)
scalar_case("PREIONISED_FREE", FIELD, analytic!, 0.0, PREIONFRAC)
scalar_case("ZEROS", ZEROED, analytic!, IONPOT, 0.0)
scalar_case("ZEROS_FREE", ZEROED, analytic!, 0.0, 0.0)
scalar_case("PPT", FIELD, ppt!, IONPOT, 0.0)
scalar_case("PPT_FREE", FIELD, ppt!, 0.0, 0.0)
vector_case("VECTOR", FIELD_X, FIELD_Y, analytic!, IONPOT, 0.0)
vector_case("VECTOR_FREE", FIELD_X, FIELD_Y, analytic!, 0.0, 0.0)
