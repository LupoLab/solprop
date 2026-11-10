# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

include(joinpath(@__DIR__, "..", "scripts", "artifact_inventory.jl"))

@testset "Setup cannot add an unselected active environment" begin
    root = normpath(joinpath(@__DIR__, ".."))
    cpu = joinpath(root, "environments", "cpu")
    metal = joinpath(root, "environments", "metal")
    runtime_stack = copy(LOAD_PATH)
    depots = copy(DEPOT_PATH)
    project = Base.active_project()
    mktempdir() do depot
        try
            empty!(DEPOT_PATH)
            push!(DEPOT_PATH, depot)
            Base.set_active_project(joinpath(metal, "Project.toml"))
            empty!(LOAD_PATH)
            append!(LOAD_PATH, [cpu, "@stdlib"])
            @test_throws "must include its active environment" begin
                SetupInventory.validate_context(root)
            end
            insert!(LOAD_PATH, 2, metal)
            _, actual_depot, actual_directory = SetupInventory.validate_context(root)
            @test actual_depot == realpath(depot)
            @test actual_directory == realpath(metal)
        finally
            Base.set_active_project(project)
            empty!(LOAD_PATH)
            append!(LOAD_PATH, runtime_stack)
            empty!(DEPOT_PATH)
            append!(DEPOT_PATH, depots)
        end
    end
end

@testset "Artifact inventories retain safe aliases and refuse escapes" begin
    mktempdir() do directory
        root = mkpath(joinpath(directory, "artifact"))
        content = mkpath(joinpath(root, "content"))
        write(joinpath(content, "data"), "retained artifact")
        symlink("content", joinpath(root, "alias"))
        symlink("content/data", joinpath(root, "link"))
        @test SetupInventory.inventory_files(root) == ["alias/data", "content/data", "link"]
        symlink(directory, joinpath(root, "escape"))
        @test_throws ArgumentError SetupInventory.inventory_files(root)
        rm(joinpath(root, "escape"))
        symlink(".", joinpath(root, "cycle"))
        @test_throws ArgumentError SetupInventory.inventory_files(root)
        rm(joinpath(root, "cycle"))
        @test SetupInventory.contained_path(root, directory) == "artifact"
        @test_throws ArgumentError SetupInventory.contained_path(directory, root)
    end
end

@testset "Preference guards include UUID-filtered tables" begin
    runtime_stack = copy(LOAD_PATH)
    mktempdir() do directory
        cpu, cuda, metal = [
            mkpath(joinpath(directory, name)) for name in ("cpu", "cuda", "metal")
        ]
        uuid = Base.UUID("36c3d7d3-259d-4cac-bc03-bbb9ac303aa9")
        write(joinpath(cpu, "Project.toml"), "[deps]\n")
        for project in (cuda, metal)
            write(joinpath(project, "Project.toml"), "[deps]\nExample = \"$uuid\"\n")
            manifest = Dict(
                "manifest_format" => "2.0", "julia_version" => string(VERSION),
                "deps" => Dict("Example" => [Dict("uuid" => string(uuid))]),
            )
            open(joinpath(project, "Manifest.toml"), "w") do io
                TOML.print(io, manifest)
            end
        end
        for (project, value) in ((cpu, "same"), (cuda, "different"), (metal, "same"))
            write(
                joinpath(project, "LocalPreferences.toml"),
                "[Example]\nvalue = \"$value\"\n",
            )
        end
        stack = [cpu, cuda, metal, "@stdlib"]
        try
            empty!(LOAD_PATH)
            append!(LOAD_PATH, stack)
            names = Base.get_preferences()
            @test Base.get_preferences(uuid)["value"] == "different"
            empty!(LOAD_PATH)
            append!(LOAD_PATH, [metal, cpu, cuda, "@stdlib"])
            @test Base.get_preferences() == names
            @test Base.get_preferences(uuid)["value"] == "same"
            empty!(LOAD_PATH)
            append!(LOAD_PATH, stack)
            @test_throws ArgumentError SetupInventory.with_precompile_stack(metal) do
                error("changed UUID preferences must refuse before precompilation")
            end
            @test LOAD_PATH == stack
        finally
            empty!(LOAD_PATH)
            append!(LOAD_PATH, runtime_stack)
        end
    end
end

@testset "Existing corrupt package trees and artifact overrides refuse" begin
    mktempdir() do directory
        write(joinpath(directory, "Project.toml"), "name = \"Example\"\n")
        expected_hash = bytes2hex(Pkg.GitTools.tree_hash(directory))
        @test SetupInventory.verify_package(directory, expected_hash) === nothing
        write(joinpath(directory, "modified"), "corrupt existing source")
        @test_throws ArgumentError SetupInventory.verify_package(directory, expected_hash)
        @test SetupInventory.refuse_overrides(directory) === nothing
        mkpath(joinpath(directory, "artifacts"))
        write(joinpath(directory, "artifacts", "Overrides.toml"), "")
        @test_throws ArgumentError SetupInventory.refuse_overrides(directory)
    end
end

@testset "Only the audited lazy-artifact selectors run" begin
    mktempdir() do directory
        uuid = Base.UUID("36c3d7d3-259d-4cac-bc03-bbb9ac303aa9")
        tree_hash = "1"^40
        @test SetupInventory.selector_kind(directory, uuid, tree_hash, Dict()) == "none"
        write(joinpath(directory, "Artifacts.toml"), "")
        @test SetupInventory.selector_kind(directory, uuid, tree_hash, Dict()) == "default"
        selector = joinpath(mkpath(joinpath(directory, ".pkg")), "select_artifacts.jl")
        write(selector, "error(\"an unaudited hook must never run\")\n")
        @test_throws ArgumentError SetupInventory.selector_kind(
            directory, uuid, tree_hash, Dict(),
        )
        policy = Dict(
            string(uuid) => Dict(
                "tree_hash" => tree_hash,
                "files" => Dict(
                    ".pkg/select_artifacts.jl" => SetupInventory.file_sha256(selector),
                ),
            )
        )
        @test SetupInventory.selector_kind(directory, uuid, tree_hash, policy) == "hook"
        @test_throws ArgumentError SetupInventory.selector_kind(
            directory, uuid, "2"^40, policy,
        )
        write(selector, "changed selector\n")
        @test_throws ArgumentError SetupInventory.selector_kind(
            directory, uuid, tree_hash, policy,
        )
    end
end

@testset "Inventory serialization is deterministic and never overwrites" begin
    mktempdir() do directory
        inventory = Dict("packages" => [], "artifacts" => [], "complete" => true)
        first = joinpath(directory, "first.toml")
        second = joinpath(directory, "second.toml")
        SetupInventory.publish_inventory(first, inventory)
        SetupInventory.publish_inventory(second, inventory)
        @test read(first) == read(second)
        @test TOML.parsefile(first) == inventory
        @test_throws ArgumentError SetupInventory.publish_inventory(first, inventory)
    end
end

@testset "Precompilation preserves runtime preferences and restores the stack" begin
    runtime_stack = copy(LOAD_PATH)
    mktempdir() do directory
        cpu = mkpath(joinpath(directory, "cpu"))
        metal = mkpath(joinpath(directory, "metal"))
        for project in (cpu, metal)
            write(joinpath(project, "Project.toml"), "[deps]\n")
        end
        preference = "[Example]\nbackend = \"same\"\n"
        write(joinpath(cpu, "LocalPreferences.toml"), preference)
        write(joinpath(metal, "LocalPreferences.toml"), preference)
        stack = [cpu, metal, "@stdlib"]
        try
            empty!(LOAD_PATH)
            append!(LOAD_PATH, stack)
            preferences = Base.get_preferences()
            @test SetupInventory.with_precompile_stack(metal) do
                @test LOAD_PATH == [metal, cpu, "@stdlib"]
                @test Base.get_preferences() == preferences
                :completed
            end == :completed
            @test LOAD_PATH == stack
            @test_throws ErrorException SetupInventory.with_precompile_stack(metal) do
                error("precompilation failed")
            end
            @test LOAD_PATH == stack
            write(
                joinpath(metal, "LocalPreferences.toml"),
                "[Example]\nbackend = \"other\"\n",
            )
            @test_throws ArgumentError SetupInventory.with_precompile_stack(metal) do
                error("changed preferences must refuse before precompilation")
            end
            @test LOAD_PATH == stack
        finally
            empty!(LOAD_PATH)
            append!(LOAD_PATH, runtime_stack)
        end
    end
end
