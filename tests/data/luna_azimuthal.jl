# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Needs a Luna.jl checkout: julia --project=/path/to/Luna tests/data/luna_azimuthal.jl reduced OUTDIR
include(joinpath(@__DIR__, "luna_polarised_common.jl"))

for (name, modes, plasma) in (
    ("basic_modal", (he(1, 1), he(1, 2)), true),
    ("he_selection", (he(1, 1), he(1, 2), he(2, 1)), false),
    ("tm_selection", (transverse(:TM), he(1, 1)), false),
    ("te", (transverse(:TE),), false),
    ("tm", (transverse(:TM),), false),
)
    weights = zeros(ComplexF64, length(modes))
    weights[1] = 1
    modal_witness((
        name=name, family="field", source="full_modal/basic_modal_full_bothpolarisations.jl",
        radius=13e-6, gas=:Ar, pressure=5.0, distance=0.15,
        wavelength=800e-9, limits=(160e-9, 3000e-9), time_window=1e-12,
        fwhm=30e-15, energy=1e-6, modes=modes, weights=weights,
        components=:xy, full=true, plasma=plasma, loss=false,
        corrections="Construct linop before statistics; allocate a two-component plasma " *
            "scratch for xy. The basic source contains HE11 and HE12 with the same phi; " *
            "selection and transverse-family cases are explicitly additional witnesses.",
    ), @__FILE__)
end
