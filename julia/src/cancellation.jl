# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    StopFlag()
    StopFlag(address::Integer)

Own an initially clear flag, or borrow an existing atomic byte at `address`.

The session retains the byte's allocation owner while reading it. A foreign listener
retains that allocation through every captured invocation, including invocations after
detachment. The flag is monotone: a new session receives a new flag instead of clearing
a live one.
The borrowed form requires the caller to retain the allocation through every engine
read and captured listener invocation. Its address must be nonzero, writable and aligned
for an atomic byte. Borrowing lets a Python listener own C storage without retaining a
Julia wrapper whose destruction would enter Julia on the stopping thread.
"""
struct StopFlag{Owner}
    storage::Owner
    address::Ptr{UInt8}
end

function StopFlag()
    storage = UInt8[0]
    return StopFlag(storage, pointer(storage))
end

function StopFlag(address::Integer)
    if !(0 < address <= typemax(UInt))
        throw(ArgumentError("the stop flag address must fit a nonzero machine pointer"))
    end
    return StopFlag(nothing, Ptr{UInt8}(UInt(address)))
end

"""
    stop_address(flag::StopFlag)

Return the flag's byte address for the foreign atomic setter.

The returned `UInt` is valid only while the allocation owner is retained. For a borrowed
flag, retaining `flag` alone is insufficient. This is not a Julia object address; the
caller must not free it or access it with non-atomic operations.
"""
stop_address(flag::StopFlag) = UInt(flag.address)

"""
    stop_requested(flag::StopFlag)

Read whether cancellation was requested, with acquire ordering and no allocation.
"""
function stop_requested(flag::StopFlag)
    GC.@preserve flag begin
        return unsafe_load(flag.address, :acquire) != 0
    end
end

"""
    stop_setter_address()

Return a compiled C `void (uint8_t *)` leaf that stores one atomically with release ordering.

Compile this function during preparation, never from the listener. Retain the engine
and flag owners for the listener's lifetime. The leaf makes no Julia runtime entry,
allocates nothing and acquires no lock, even on a previously unknown foreign thread.
"""
function stop_setter_address()
    # A Julia @cfunction wrapper adopts a foreign thread and can allocate or wait for
    # GC. Returning a raw leaf avoids that wrapper. Internal linkage isolates the
    # symbol in its JIT module, and the admitted platforms all have 64-bit pointers.
    return Base.llvmcall(
        (
            """
            define internal void @solprop_store_stop(ptr %address) {
                store atomic i8 1, ptr %address release, align 1
                ret void
            }
            define i64 @solprop_stop_setter_address() {
                %address = ptrtoint ptr @solprop_store_stop to i64
                ret i64 %address
            }
            """,
            "solprop_stop_setter_address",
        ),
        UInt,
        Tuple{},
    )
end
