# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SetupInventory

import Pkg
import SHA
import TOML
import UUIDs: UUID

const ENGINE_UUID = UUID("bc9db228-6fd1-46b1-b175-748485922872")
const ENVIRONMENT_NAMES = ("cpu", "cuda", "rocm", "metal")
const CUDA_UUIDS = (
    UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
    UUID("d1e2174e-dfdc-576e-b43e-73b79eb1aca8"),
)

file_sha256(path) = bytes2hex(SHA.sha256(read(path)))
portable_path(path) = replace(path, '\\' => '/')

function contained_path(path, root)
    relative = relpath(realpath(path), realpath(root))
    if isabspath(relative) || first(splitpath(relative)) == ".."
        throw(ArgumentError("path escapes its owned directory: $path"))
    end
    return portable_path(relative)
end

function refuse_overrides(depot)
    overrides = joinpath(depot, "artifacts", "Overrides.toml")
    if ispath(overrides) || islink(overrides)
        throw(ArgumentError("artifact overrides are not supported: $overrides"))
    end
    return nothing
end

function inventory_files(root)
    files = String[]
    function visit(directory, ancestors)
        actual = realpath(directory)
        actual in ancestors && throw(ArgumentError("cyclic artifact symlink: $directory"))
        contained_path(directory, root)
        next_ancestors = union(ancestors, Set([actual]))
        for path in readdir(directory; join = true, sort = true)
            contained_path(path, root)
            if isdir(path)
                visit(path, next_ancestors)
            elseif isfile(path)
                # Record files through safe directory symlinks too: removing the alias
                # must make the Python-side offline inventory check fail.
                push!(files, portable_path(relpath(path, root)))
            else
                throw(ArgumentError("artifact contains a non-file entry: $path"))
            end
        end
        return nothing
    end
    visit(root, Set{String}())
    return sort!(files)
end

function verify_package(path, expected_hash)
    isfile(joinpath(path, "Project.toml")) || throw(
        ArgumentError(
            "installed package has no Project.toml: $path",
        )
    )
    actual_hash = bytes2hex(Pkg.GitTools.tree_hash(path))
    actual_hash == expected_hash || throw(
        ArgumentError(
            "installed package is corrupt: $path " *
                "(expected $expected_hash, found $actual_hash)",
        )
    )
    return nothing
end

function selector_kind(path, uuid, tree_hash, policy)
    artifact_file = findfirst(
        name -> isfile(joinpath(path, name)), Pkg.Artifacts.artifact_names,
    )
    selector = joinpath(path, ".pkg", "select_artifacts.jl")
    if isfile(selector)
        key = string(uuid)
        haskey(policy, key) || throw(ArgumentError("unaudited artifact selector: $key"))
        audit = policy[key]
        audit["tree_hash"] == tree_hash || throw(
            ArgumentError(
                "artifact selector package changed; audit required: $key",
            )
        )
        expected_files = audit["files"]
        for (relative, digest) in expected_files
            source = joinpath(path, relative)
            isfile(source) && file_sha256(source) == digest || throw(
                ArgumentError(
                    "artifact selector source changed; audit required: $source",
                )
            )
        end
        haskey(expected_files, ".pkg/select_artifacts.jl") || throw(
            ArgumentError(
                "artifact selector audit is incomplete: $key",
            )
        )
        artifact_file === nothing && throw(
            ArgumentError(
                "artifact selector has no artifact table: $key",
            )
        )
        return "hook"
    end
    return artifact_file === nothing ? "none" : "default"
end

function validate_cuda_preferences(dependencies)
    all(haskey(dependencies, uuid) for uuid in CUDA_UUIDS) || return nothing
    preferences = map(Base.get_preferences, CUDA_UUIDS)
    preferences[1] == preferences[2] || throw(
        ArgumentError(
            "CUDA Runtime and Compiler preferences must agree",
        )
    )
    preference = only(unique(preferences))
    local_runtime = get(preference, "local", false)
    local_runtime isa Bool || throw(ArgumentError("CUDA local preference must be boolean"))
    if local_runtime
        haskey(preference, "version") && throw(
            ArgumentError(
                "local CUDA selection must not also specify a runtime version",
            )
        )
    else
        version = get(preference, "version", nothing)
        version isa String && tryparse(VersionNumber, version) !== nothing ||
            throw(ArgumentError("CUDA setup requires an explicit runtime version"))
    end
    return nothing
end

function validate_context(engine_root)
    length(DEPOT_PATH) == 1 || throw(ArgumentError("setup requires one private depot"))
    depot = realpath(only(DEPOT_PATH))
    refuse_overrides(depot)
    Base.JLOptions().startupfile == 2 || throw(
        ArgumentError(
            "setup requires --startup-file=no",
        )
    )
    versions = TOML.parsefile(joinpath(engine_root, "environments", "versions.toml"))
    pinned_version = VersionNumber(versions["julia"])
    (VERSION.major, VERSION.minor) == (pinned_version.major, pinned_version.minor) ||
        throw(
        ArgumentError(
            "setup requires Julia $(pinned_version.major).$(pinned_version.minor)",
        )
    )
    project_file = Base.active_project()
    project_file === nothing && throw(ArgumentError("setup requires an explicit project"))
    environment_root = joinpath(engine_root, "environments")
    allowed = [
        realpath(joinpath(environment_root, name)) for name in ENVIRONMENT_NAMES
            if isfile(joinpath(environment_root, name, "Project.toml"))
    ]
    directory = realpath(dirname(project_file))
    directory in allowed || throw(ArgumentError("unsupported setup project: $directory"))
    length(LOAD_PATH) >= 2 && last(LOAD_PATH) == "@stdlib" || throw(
        ArgumentError(
            "setup LOAD_PATH must contain the CPU environment and end with @stdlib",
        )
    )
    selected_names = String[]
    for entry in LOAD_PATH[1:(end - 1)]
        selected = entry == "@" ? directory : entry
        isdir(selected) && realpath(selected) in allowed || throw(
            ArgumentError(
                "setup LOAD_PATH contains an unowned environment: $entry",
            )
        )
        contained_path(selected, engine_root)
        push!(selected_names, basename(realpath(selected)))
    end
    expected_names = [name for name in ENVIRONMENT_NAMES if name in selected_names]
    first(selected_names) == "cpu" && selected_names == expected_names ||
        throw(ArgumentError("setup LOAD_PATH must be a unique CPU-first device stack"))
    context = Pkg.Types.Context()
    manifest_file = context.env.manifest_file
    isfile(manifest_file) || throw(ArgumentError("missing setup manifest: $manifest_file"))
    expected_hash = Pkg.Types.project_resolve_hash(
        context.env, context.env.manifest.julia_version,
    )
    get(context.env.manifest.other, "project_hash", nothing) == expected_hash ||
        throw(ArgumentError("setup manifest does not match its project: $directory"))
    return context, depot, directory
end

function selected_artifacts(path, uuid, platform, local_cuda)
    selected = Pkg.Operations.collect_artifacts(path; platform, include_lazy = true)
    if uuid in CUDA_UUIDS && !local_cuda && (Sys.islinux() || Sys.iswindows())
        any(!isempty(metadata) for (_, metadata) in selected) || throw(
            ArgumentError(
                "the requested CUDA runtime has no selected artifact for $(basename(path))",
            )
        )
    end
    return selected
end

function build_inventory(engine_root)
    context, depot, directory = validate_context(engine_root)
    project_file = context.env.project_file
    manifest_file = context.env.manifest_file
    project_hash = file_sha256(project_file)
    manifest_hash = file_sha256(manifest_file)
    policy = TOML.parsefile(joinpath(@__DIR__, "artifact_selectors.toml"))["selectors"]
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    # PythonCall embeds Python during Julia precompilation. Keep Python bytecode out
    # of the immutable package source tree; JuliaCall's runtime direction differs.
    ENV["PYTHONDONTWRITEBYTECODE"] = "1"

    # Download source before invoking any selector, so an unaudited hook is refused
    # before Pkg.instantiate can execute it. Pkg APIs are pinned to versions.toml.
    if Pkg.Registry.download_default_registries(context.io)
        copy!(context.registries, Pkg.Registry.reachable_registries())
    end
    Pkg.Operations.download_source(context)
    dependencies = Pkg.dependencies()
    validate_cuda_preferences(dependencies)
    packages = Dict{String, Any}[]
    package_sources = Tuple{UUID, String, Dict{String, Any}}[]
    for (uuid, info) in sort!(collect(dependencies); by = pair -> pair.second.name)
        uuid == ENGINE_UUID && continue
        Pkg.Types.is_stdlib(uuid) && info.tree_hash === nothing && continue
        relative = contained_path(info.source, depot)
        info.tree_hash === nothing && throw(
            ArgumentError(
                "setup requires a registered package tree hash: $(info.name)",
            )
        )
        verify_package(info.source, info.tree_hash)
        kind = selector_kind(info.source, uuid, info.tree_hash, policy)
        entry = Dict{String, Any}(
            "name" => info.name, "uuid" => string(uuid), "path" => relative,
            "selector" => kind, "selected_artifacts" => String[],
        )
        push!(packages, entry)
        push!(package_sources, (uuid, info.source, entry))
    end
    refuse_overrides(depot)
    Pkg.instantiate(
        context; update_registry = false, allow_build = false,
        allow_autoprecomp = false, julia_version_strict = true,
    )

    platform = Base.BinaryPlatforms.HostPlatform()
    artifacts = Dict{String, Any}[]
    verified_trees = Dict{String, Vector{String}}()
    local_cuda = any(get(Base.get_preferences(uuid), "local", false) for uuid in CUDA_UUIDS)
    for (uuid, path, package) in package_sources
        selections = selected_artifacts(path, uuid, platform, local_cuda)
        for (artifact_file, selected) in selections
            for (name, metadata) in sort!(collect(selected); by = first)
                tree_hash = metadata["git-tree-sha1"]
                refuse_overrides(depot)
                Pkg.Artifacts.ensure_artifact_installed(
                    name, metadata, artifact_file; platform,
                )
                refuse_overrides(depot)
                files = get!(verified_trees, tree_hash) do
                    hash = Base.SHA1(tree_hash)
                    Pkg.Artifacts.verify_artifact(hash; honor_overrides = false) ||
                        throw(ArgumentError("corrupt artifact: $name/$tree_hash"))
                    root = joinpath(depot, "artifacts", tree_hash)
                    contained_path(root, depot) == "artifacts/$tree_hash" ||
                        throw(ArgumentError("artifact is outside its private depot: $name"))
                    inventory_files(root)
                end
                push!(package["selected_artifacts"], name)
                push!(
                    artifacts, Dict{String, Any}(
                        "name" => name, "package_uuid" => string(uuid),
                        "tree_hash" => tree_hash, "path" => "artifacts/$tree_hash",
                        "files" => files,
                    )
                )
            end
        end
        sort!(package["selected_artifacts"])
    end
    Pkg.precompile(; strict = true)
    refuse_overrides(depot)
    unchanged = file_sha256(project_file) == project_hash &&
        file_sha256(manifest_file) == manifest_hash
    unchanged ||
        throw(ArgumentError("setup changed its pinned project or manifest"))
    selection_count = sum(length(package["selected_artifacts"]) for package in packages)
    selection_count == length(artifacts) ||
        throw(ArgumentError("artifact inventory is incomplete"))
    return Dict{String, Any}(
        "schema_version" => 1, "complete" => true, "project" => basename(directory),
        "platform" => Base.BinaryPlatforms.triplet(platform),
        "project_sha256" => project_hash, "manifest_sha256" => manifest_hash,
        "package_count" => length(packages), "artifact_selection_count" => selection_count,
        "artifact_verified_count" => length(artifacts), "packages" => packages,
        "artifacts" => artifacts,
    )
end

function publish_inventory(path, inventory)
    ispath(path) && throw(ArgumentError("inventory output already exists: $path"))
    temporary, io = mktemp(dirname(abspath(path)))
    try
        TOML.print(io, inventory; sorted = true)
        close(io)
        mv(temporary, path)
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary)
    end
    return nothing
end

end
