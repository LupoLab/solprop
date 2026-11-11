# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct StorageTestSpace
    context::Int
end

mutable struct StorageTestArray{T} <: AbstractVector{T}
    owner::Vector{T}
    first::Int
    count::Int
    space::StorageTestSpace
    live::Bool
end
Base.size(array::StorageTestArray) = (array.count,)
Base.IndexStyle(::Type{<:StorageTestArray}) = IndexLinear()
Base.getindex(array::StorageTestArray, index::Int) = array.owner[array.first + index - 1]
Base.dataids(array::StorageTestArray) = Base.dataids(array.owner)
function SolpropJulia._transfer_storage_space(array::StorageTestArray)
    array.live || throw(ArgumentError("released test storage"))
    return array.space
end
function SolpropJulia._transfer_storage_span(array::StorageTestArray)
    isempty(array) && return nothing
    SolpropJulia._transfer_storage_space(array)
    first = UInt(pointer(array.owner, array.first))
    return first, first + UInt(array.count * sizeof(eltype(array)))
end
function SolpropJulia._memory_array_owner(array::StorageTestArray)
    SolpropJulia._transfer_storage_space(array)
    return array.owner, sizeof(array.owner)
end

struct StorageTestUnknown{A <: AbstractArray} <: AbstractVector{Int}
    values::A
end
Base.size(array::StorageTestUnknown) = (length(array.values),)
Base.dataids(array::StorageTestUnknown) = Base.dataids(array.values)

struct StorageTestOpaque{A, B}
    numerical::A
    unrelated::B
end
SolpropJulia._retained_storage(value::StorageTestOpaque) = (value.numerical,)

function storage_test_ledger(value)
    ledger = Dict{Symbol, Int}()
    SolpropJulia._memory_arrays!(ledger, IdDict{Any, Bool}(), "owner", value)
    return ledger
end

@testset "Storage spaces distinguish coordinates without weakening dependencies" begin
    owner = collect(1:32)
    space = StorageTestSpace(1)
    left = StorageTestArray(owner, 1, 8, space, true)
    same = StorageTestArray(owner, 4, 8, space, true)
    other = StorageTestArray(owner, 1, 8, StorageTestSpace(2), true)
    @test Base.mightalias(left, other)
    @test SolpropJulia._transfer_arrays_alias(left, same)
    @test SolpropJulia._transfer_arrays_alias(same, left)
    @test !SolpropJulia._transfer_arrays_alias(left, other)
    @test !SolpropJulia._transfer_arrays_alias(other, left)
    @test !SolpropJulia._transfer_arrays_alias(owner, left)
    @test !SolpropJulia._transfer_arrays_alias(left, owner)
    @test SolpropJulia._transfer_storage_span(view(left, 2:3)) ==
        SolpropJulia._transfer_storage_span(left)
    @test SolpropJulia._transfer_storage_space(view(left, 2:3)) == space

    indices = [1, 3, 5, 7]
    input = view(left, indices)
    for wrapped in (input, reshape(input, 2, 2), transpose(reshape(input, 2, 2)))
        @test SolpropJulia._transfer_arrays_alias(indices, wrapped)
        @test SolpropJulia._transfer_arrays_alias(wrapped, indices)
    end
    @test !SolpropJulia._transfer_arrays_alias(copy(indices), input)
    @test !SolpropJulia._transfer_arrays_alias(input, copy(indices))
    # Address spaces differ for the numerical data, but a mutable host gather
    # table still belongs to the host destination's address space.
    GC.@preserve indices begin
        foreign_indices = unsafe_wrap(Array, pointer(indices, 2), (2,); own = false)
        @test SolpropJulia._transfer_arrays_alias(foreign_indices, input)
        @test SolpropJulia._transfer_arrays_alias(input, foreign_indices)
    end

    other.live = false
    @test_throws ArgumentError SolpropJulia._transfer_arrays_alias(left, other)
    @test_throws ArgumentError SolpropJulia._transfer_arrays_alias(other, left)
    @test_throws ArgumentError SolpropJulia._transfer_arrays_alias(owner, other)
    @test_throws ArgumentError SolpropJulia._retained_alias(owner, (other,))
    unknown = StorageTestUnknown(owner)
    @test Base.mightalias(owner, unknown)
    @test_throws ArgumentError SolpropJulia._transfer_arrays_alias(owner, unknown)
    @test_throws ArgumentError SolpropJulia._transfer_arrays_alias(unknown, owner)
    @test !SolpropJulia._transfer_arrays_alias(Int[], other)
    @test !SolpropJulia._transfer_arrays_alias(StorageTestUnknown(Int[]), owner)
    @test owner == collect(1:32)
end

@testset "Allocation owners are independent of visible alias intervals" begin
    owner = zeros(Float64, 32)
    left = StorageTestArray(owner, 1, 4, StorageTestSpace(1), true)
    right = StorageTestArray(owner, 17, 4, StorageTestSpace(1), true)
    ledger = storage_test_ledger((left, right, view(left, 2:3), reshape(right, 2, 2)))
    @test length(ledger) == 1
    @test only(values(ledger)) == sizeof(owner)
    @test only(values(storage_test_ledger((owner, view(owner, 2:4))))) == sizeof(owner)
    @test sum(values(storage_test_ledger((zeros(2), zeros(3))))) == 5sizeof(Float64)
    right.live = false
    @test_throws ArgumentError storage_test_ledger(right)
end

@testset "Opaque retention exposes only numerical plan storage" begin
    numerical, metadata = zeros(8), zeros(9)
    plan = StorageTestOpaque(numerical, metadata)
    @test SolpropJulia._retained_alias(numerical, (plan,))
    @test !SolpropJulia._retained_alias(metadata, (plan,))
    @test !SolpropJulia._retained_alias(zeros(8), (plan,))
    GC.@preserve numerical begin
        interior = unsafe_wrap(Array, pointer(numerical, 2), (3,); own = false)
        @test SolpropJulia._retained_alias(interior, (plan,))
    end
    SolpropJulia._retained_alias(numerical, (plan,))
    @test @allocated(SolpropJulia._retained_alias(numerical, (plan,))) == 0
end
