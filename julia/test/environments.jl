# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import Pkg
import TOML

function resolved_entry(entry, manifest)
    result = deepcopy(entry)
    for key in ("deps", "weakdeps")
        dependencies = get(result, key, nothing)
        if dependencies isa Vector
            result[key] = Dict(
                name => only(manifest[name])["uuid"] for name in dependencies
            )
        end
    end
    return result
end

@testset "Application manifests share one closed resolution" begin
    root = normpath(joinpath(@__DIR__, "..", "environments"))
    joint = TOML.parsefile(joinpath(root, "joint", "Manifest.toml"))["deps"]
    for name in ("cpu", "cuda", "rocm", "metal")
        directory = joinpath(root, name)
        project = TOML.parsefile(joinpath(directory, "Project.toml"))
        entries = TOML.parsefile(joinpath(directory, "Manifest.toml"))["deps"]
        @test all(haskey(entries, package) for package in keys(project["deps"]))
        for (package, alternatives) in entries
            # Pkg writes weak dependencies as names when present, and UUID mappings
            # when absent. Compare resolved identities, including every extension.
            @test [resolved_entry(entry, entries) for entry in alternatives] ==
                [resolved_entry(entry, joint) for entry in joint[package]]
            for entry in alternatives, dependency in get(entry, "deps", String[])
                @test haskey(entries, dependency)
            end
        end
        @test only(entries["SolpropJulia"])["path"] == "../.."
        environment = Pkg.Types.EnvCache(joinpath(directory, "Project.toml"))
        @test environment.manifest.other["project_hash"] ==
            Pkg.Types.project_resolve_hash(environment, environment.manifest.julia_version)
        for tool in ("Aqua", "JET", "Runic")
            @test !haskey(entries, tool)
        end
    end
    cpu = TOML.parsefile(joinpath(root, "cpu", "Manifest.toml"))["deps"]
    for device in ("CUDA", "CUDA_Runtime_jll", "CUDA_Compiler_jll", "AMDGPU", "Metal")
        @test !haskey(cpu, device)
    end
    @test only(cpu["PythonCall"])["version"] == "0.9.37"
    engine = TOML.parsefile(joinpath(root, "..", "Project.toml"))
    @test !haskey(engine["deps"], "PythonCall")
    @test engine["extensions"]["SolpropJuliaPythonCallExt"] == "PythonCall"
end
