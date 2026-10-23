# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Read-only material witness for the spatial propagation comparisons.
# Run from solprop, never from the Luna checkout. Existing propagation files stay intact.
using Luna
using HDF5
using SHA

length(ARGS) == 2 || error("usage: luna_spatial_media.jl WITNESS_DIRECTORY OUTPUT_FILE")
isfile(ARGS[2]) && error("refusing to overwrite $(ARGS[2])")
h5open(ARGS[2], "w") do out
    out["luna_commit"] = readchomp(`git -C $(dirname(pathof(Luna))) rev-parse HEAD`)
    out["generator_sha256"] = bytes2hex(sha256(read(@__FILE__)))
    out["argon_reference_density"] = PhysData.dens_1bar_0degC[:Ar]
    for path in sort(filter(p -> endswith(p, ".h5"), readdir(ARGS[1]; join=true)))
        h5open(path, "r") do input
            haskey(input, "witness") || return
            p = input["witness"]
            omega = read(input["grid/ω"])
            group = create_group(out, splitext(basename(path))[1])
            group["omega"] = omega
            # Evaluate only the positive active band; never query a model at DC.
            active = read(input["grid/ωwin"]) .> 0
            wavelengths = 2π * PhysData.c ./ omega[active]
            group["active"] = collect(active)
            n = PhysData.ref_index_fun(:Ar, read(p["pressure_bar"]), read(p["temperature"]))
            group["index"] = n.(wavelengths)
            group["cladding"] = PhysData.ref_index_fun(:SiO2).(wavelengths)
        end
    end
end
