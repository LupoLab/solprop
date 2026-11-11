# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import Libdl

public CPUBlasInfo, configure_cpu_blas!

"""
    CPUBlasInfo

Describe the verified process BLAS forwarding choice without retaining a thread lease.

`provider` is `:accelerate` or `:openblas`; `library` is the actual primary library.
`interface` is Julia's `:ilp64` or `:lp64` integer ABI. `fallback_library` names retained
OpenBLAS forwards when Accelerate is selected, or is empty for OpenBLAS alone.
`thread_scope` is `:calling_thread` for Accelerate or `:process` for OpenBLAS. `reason`
is empty for Accelerate, or names the platform/capability that retained OpenBLAS.
This immutable record reports a provider, never a promised automatic thread count.
"""
struct CPUBlasInfo
    provider::Symbol
    library::String
    interface::Symbol
    fallback_library::String
    thread_scope::Symbol
    reason::String
end

const _CPU_BLAS_INFO = Ref{Union{Nothing, CPUBlasInfo}}(nothing)
const _CPU_BLAS_LOCK = ReentrantLock()
const _CPU_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
const _CPU_BLAS_SYMBOLS = ("dgemm_", "zgemm_", "dgesv_", "zgesv_")

struct _AccelerateAPI
    handle::Ptr{Cvoid}
    getter::Ptr{Cvoid}
    setter::Ptr{Cvoid}
end

function _cpu_blas_main()
    Threads.threadid() == 1 ||
        throw(ArgumentError("CPU BLAS configuration requires Julia's main thread"))
    return nothing
end

_cpu_blas_interface() = sizeof(LinearAlgebra.BlasInt) == 8 ? :ilp64 : :lp64

function _cpu_blas_owner(symbol, interface; config = LinearAlgebra.BLAS.lbt_get_config())
    library = LinearAlgebra.BLAS.lbt_find_backing_library(symbol, interface; config)
    library === nothing && error("CPU BLAS has no backing library for $symbol")
    return String(library.libname)
end

function _cpu_blas_openblas(reason::String)
    interface = _cpu_blas_interface()
    config = LinearAlgebra.BLAS.lbt_get_config()
    path = _cpu_blas_owner("dgemm_", interface; config)
    occursin("openblas", lowercase(basename(path))) ||
        error("CPU BLAS initialization requires the owned OpenBLAS fallback")
    for symbol in _CPU_BLAS_SYMBOLS
        _cpu_blas_owner(symbol, interface; config) == path ||
            error("CPU BLAS initialization found inconsistent OpenBLAS forwards")
    end
    return CPUBlasInfo(:openblas, path, interface, "", :process, reason)
end

function _cpu_accelerate_api(path::String)
    handle = Libdl.dlopen_e(path)
    handle == C_NULL && return nothing, "accelerate-library-unavailable"
    retained = false
    try
        # Probe the documented new LAPACK/ILP64 family before changing any forward.
        # AppleAccelerate.jl uses this same public symbol family and clear=false:
        # https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl
        for stem in ("dgemm", "zgemm", "dgesv", "zgesv"),
                suffix in ("\$NEWLAPACK", "\$NEWLAPACK\$ILP64")
            Libdl.dlsym_e(handle, stem * suffix) != C_NULL ||
                return nothing, "accelerate-new-lapack-unavailable"
        end
        getter = Libdl.dlsym_e(handle, :BLASGetThreading)
        setter = Libdl.dlsym_e(handle, :BLASSetThreading)
        if getter == C_NULL || setter == C_NULL
            return nothing, "accelerate-thread-control-unavailable"
        end
        retained = true
        return _AccelerateAPI(handle, getter, setter), ""
    finally
        retained || Libdl.dlclose(handle)
    end
end

function _cpu_accelerate_mode(api::_AccelerateAPI)
    mode = ccall(api.getter, Cuint, ())
    mode in (Cuint(0), Cuint(1)) || error("Accelerate returned an unknown threading mode")
    return mode
end

function _cpu_accelerate_mode!(api::_AccelerateAPI, mode::Cuint)
    ccall(api.setter, Cint, (Cuint,), mode) == 0 ||
        error("Accelerate refused its calling-thread mode")
    _cpu_accelerate_mode(api) == mode ||
        error("Accelerate did not retain its calling-thread mode")
    return nothing
end

function _cpu_blas_forward(api::_AccelerateAPI, fallback::CPUBlasInfo)
    original = _cpu_accelerate_mode(api)
    try
        _cpu_accelerate_mode!(api, Cuint(1))
        # New LAPACK uses a suffix without the conventional Fortran underscore.
        # LBT's leading \x1a hint supplies that symbol spelling. Keeping previous
        # forwards preserves OpenBLAS operations absent from Accelerate.
        for suffix in ("\x1a\$NEWLAPACK", "\x1a\$NEWLAPACK\$ILP64")
            LinearAlgebra.BLAS.lbt_forward(
                _CPU_ACCELERATE; clear = false, suffix_hint = suffix,
            ) > 0 || error("Accelerate forwarding did not install any BLAS symbols")
        end
        config = LinearAlgebra.BLAS.lbt_get_config()
        for interface in (:lp64, :ilp64), symbol in _CPU_BLAS_SYMBOLS
            _cpu_blas_owner(symbol, interface; config) == _CPU_ACCELERATE ||
                error("Accelerate forwarding did not select the required BLAS/LAPACK ABI")
        end
    catch failure
        try
            _cpu_accelerate_mode!(api, original)
        catch cleanup
            # Julia's standard exception aggregate owns a Vector{Any}; this
            # preparation-only failure path preserves primary and cleanup causes.
            throw(CompositeException(Any[failure, cleanup]))
        end
        rethrow()
    end
    _cpu_accelerate_mode!(api, original)
    return CPUBlasInfo(
        :accelerate, _CPU_ACCELERATE, fallback.interface, fallback.library,
        :calling_thread, "",
    )
end

function _cpu_blas_validate(info::CPUBlasInfo)
    config = LinearAlgebra.BLAS.lbt_get_config()
    for symbol in _CPU_BLAS_SYMBOLS
        _cpu_blas_owner(symbol, info.interface; config) == info.library ||
            error("CPU BLAS forwarding changed after initialization")
    end
    return info
end

"""
    configure_cpu_blas!()::CPUBlasInfo

Select and verify the owned CPU BLAS provider once, before acquiring lifetime limits.

Call on Julia's main thread during runtime initialization, before any session or BLAS
lease exists. On macOS, prefer Accelerate only when new LAPACK/ILP64 and the documented
macOS 15 calling-thread controls exist. Forward both integer ABIs through Julia's LBT
with `clear=false`, retaining OpenBLAS for missing symbols. Other platforms and older
macOS retain the loaded OpenBLAS provider with an explicit reason. Repeated calls verify
the selected primary forwards and return the same immutable record.

This preparation briefly selects single mode on the calling thread during forwarding,
then restores its exact original enum. It installs no persistent thread limit. Python
must subsequently hold single mode for every Accelerate preparation lifetime. Internal
Julia threads do not inherit that thread-local setting: the current engine uses scalar
matrix maps and FFT plans, and admits no worker-thread BLAS calls. A future such call
needs an explicit scope on its own calling thread.

Throw `ArgumentError` for a non-main-thread call and `ErrorException` for inconsistent
provider/ABI state or failed mode control. A partially failed forwarding attempt must
poison runtime startup; it is not a recoverable per-session provider switch. If both
forwarding and mode restoration fail, retain both failures in a `CompositeException`.

# References
- Apple `BLASSetThreading`: documented thread-local mode, macOS 15 availability.
  https://developer.apple.com/documentation/accelerate/blassetthreading(_:)
- Apple `BLAS_THREADING`: single mode versus an implementation-selected count.
  https://developer.apple.com/documentation/accelerate/blas_threading
- Julia's LBT forwarding API, including `clear=false` and ABI suffix detection:
  https://github.com/JuliaLinearAlgebra/libblastrampoline/blob/v5.15.0/README.md
"""
function configure_cpu_blas!()
    _cpu_blas_main()
    return lock(_CPU_BLAS_LOCK) do
        existing = _CPU_BLAS_INFO[]
        existing === nothing || return _cpu_blas_validate(existing)
        fallback = _cpu_blas_openblas("not-macos")
        if !Sys.isapple()
            _CPU_BLAS_INFO[] = fallback
            return fallback
        end
        api, reason = _cpu_accelerate_api(_CPU_ACCELERATE)
        if api === nothing
            fallback = CPUBlasInfo(
                fallback.provider, fallback.library, fallback.interface, "",
                fallback.thread_scope, reason,
            )
            _CPU_BLAS_INFO[] = fallback
            return fallback
        end
        try
            chosen = _cpu_blas_forward(api, fallback)
            _CPU_BLAS_INFO[] = chosen
            return chosen
        finally
            Libdl.dlclose(api.handle)
        end
    end
end
