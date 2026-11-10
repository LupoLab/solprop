# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

include(joinpath(@__DIR__, "..", "scripts", "artifact_inventory.jl"))

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
