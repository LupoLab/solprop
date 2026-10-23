# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Read-only Luna witness, run only by generate_luna_mixture.py from solprop.
using Luna
import Luna: PhysData

emit(tag, values) = println(tag, "\t", join(string.(values), ","))
wavelengths = [500e-9, 600e-9, 800e-9, 1000e-9]
gases = (:N2, :O2, :Ar)
emit("WAVELENGTH_M", wavelengths)
emit("CONSTANTS", [PhysData.c, PhysData.ε_0, PhysData.e_ratio])
for gas in gases
    model = PhysData.sellmeier_gas(gas)
    emit("GAMMA_$(gas)", model.(wavelengths .* 1e6))
end
for (index, partials) in enumerate(((0.7808, 0.2095, 0.0093), (6.0, 3.0, 1.0), (0.0, 0.4, 0.6)))
    index_fun = PhysData.ref_index_fun(gases, partials, 300.0)
    emit("PRESSURE_$(index)", partials)
    emit("DENSITY_$(index)", [p == 0 ? 0.0 : PhysData.density(gas, p, 300.0) for (gas, p) in zip(gases, partials)])
    emit("INDEX_$(index)", index_fun.(wavelengths))
end
