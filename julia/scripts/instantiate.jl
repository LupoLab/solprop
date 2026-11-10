# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

include("artifact_inventory.jl")

length(ARGS) == 1 || throw(ArgumentError("usage: instantiate.jl <inventory-output.toml>"))
engine_root = normpath(joinpath(@__DIR__, ".."))
inventory = SetupInventory.build_inventory(engine_root)
SetupInventory.publish_inventory(only(ARGS), inventory)
