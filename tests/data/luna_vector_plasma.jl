# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Needs a Luna.jl checkout: julia --project=/path/to/Luna tests/data/luna_vector_plasma.jl reduced OUTDIR
include(joinpath(@__DIR__, "luna_polarised_common.jl"))

for state in ("aligned", "45deg", "circular", "scalar")
    circular = state == "circular"
    scalar = state == "scalar"
    modes = circular ? (he(1, 1), he(1, 1; phi=π / 2)) :
        (he(1, 1; phi=state == "45deg" ? π / 4 : 0.0),)
    modal_witness((
        name="plasma_$(state)", family="field",
        source=scalar ? "polarisation/modal_nonvector_plasma.jl" :
            "polarisation/modal_vector_plasma" * (circular ? "_CP" : state == "45deg" ? "_45deg" : "") * ".jl",
        radius=9e-6, gas=:Ar, pressure=5.0, distance=0.25,
        wavelength=1500e-9, limits=(200e-9, 3000e-9), time_window=2e-12,
        fwhm=30e-15, energy=1.7e-6, modes=modes,
        weights=circular ? ComplexF64[1, im] / sqrt(2) : ComplexF64[1],
        components=scalar ? :y : :xy, full=false, plasma=true, loss=false,
        corrections="Construct linop before statistics; retain source physics; plotting omitted.",
    ), @__FILE__)
end
