# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Needs a Luna.jl checkout: julia --project=/path/to/Luna tests/data/luna_polarised_kerr.jl reduced OUTDIR
include(joinpath(@__DIR__, "luna_polarised_common.jl"))

for family in ("field", "envelope")
    field = family == "field"
    rotation = field ? π / 4 - deg2rad(10.0) : π / 6 + π / 8
    weights = Polarisation.rotate(Polarisation.WP(π / 2), rotation) * ComplexF64[1, 0]
    modal_witness((
        name="elliptical_$(family)", family=family,
        source=field ? "polarisation/elliptical.jl" : "polarisation/elliptical_env.jl",
        radius=field ? 225e-6 : 50e-6, gas=:Ar, pressure=field ? 0.4 : 5.0,
        distance=field ? 2.5 : 0.1, wavelength=field ? 1800e-9 : 400e-9,
        limits=field ? (200e-9, 3000e-9) : (160e-9, 3000e-9),
        time_window=1e-12, fwhm=30e-15, energy=field ? 500e-6 : 30e-6,
        modes=(he(1, 1), he(1, 1; phi=π / 2)), weights=weights,
        components=:xy, full=false, plasma=false, loss=true,
        corrections=field ? "None; plotting omitted." :
            "Use tau_fwhm instead of undefined tau; use current Luna.setup signature; " *
            "construct the same Gaussian through Fields.GaussField; plotting omitted.",
    ), @__FILE__)
end
