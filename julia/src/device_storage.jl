# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct _HostStorageSpace end

_transfer_storage_space(::Array) = _HostStorageSpace()
function _transfer_storage_space(
        array::Union{
            SubArray, Base.ReshapedArray, Base.ReinterpretArray,
            LinearAlgebra.Transpose, LinearAlgebra.Adjoint, PermutedDimsArray,
        },
    )
    isempty(array) && return nothing
    return _transfer_storage_space(parent(array))
end
function _transfer_storage_space(array::AbstractArray)
    isempty(array) && return nothing
    throw(ArgumentError("transfer storage needs a qualified address space"))
end

# This identity describes retained allocation payload, not the interval a particular
# header addresses. Device extensions specialize it independently of alias spans.
function _memory_array_owner(array::AbstractArray)
    owner = parent(array)
    return owner === array ? (owner, sizeof(owner)) : _memory_array_owner(owner)
end

# Opaque vendor objects expose only their actual numerical buffers. The generic
# retained-owner traversal must not inspect handles, caches or unrelated runtime data.
_retained_storage(value) = ()
