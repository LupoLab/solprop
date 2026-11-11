# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public FFTPlanning, plan_owned_fft!, fft_planning_record, fft_runtime_identity
public prepare_fft_planning, fft_planning_transfer, release_fft_planning!

"""
    prepare_fft_planning(transfer)

Copy detached cache offers and effort through the optional PythonCall extension.

The result is one start's private `FFTPlanning`. It borrows no Python data and
publishes nothing. Failed starts discard their context without publishing facts.
"""
function prepare_fft_planning end

"""
    fft_planning_transfer(context::FFTPlanning)

Return owned sorted tuple pairs and keys for guarded host transfer.

The same measured/applied/rejected/reused facts as `fft_planning_record` are packed
without dictionaries so conversion never depends on a foreign mapping protocol.
No cache file or numerical workspace is changed.
"""
function fft_planning_transfer end

"""
    FFTPlanning(; offered=Dict{String,String}(), effort=:measure, keep=true,
                  provider="fftw")

Own one preparation's isolated FFTW wisdom and planning facts (ADR-076).

`offered` maps transform keys to detached wisdom strings; it is copied. `effort` is
`:estimate`, `:measure` or `:patient`. `keep=false` suppresses newly measured records,
as required for precompile/calibration preparations. `provider` must name the loaded
FFTW provider; this leaf qualifies the `"fftw"` provider only. The context contains no
array workspace, runtime handle or filesystem policy. It is used synchronously during
preparation, never from a numerical kernel. Publication follows successful preparation.

Throw `ArgumentError` for unsupported effort/provider or non-string offered records.
"""
mutable struct FFTPlanning
    offered::Dict{String, String}
    measured::Dict{String, String}
    applied::Dict{String, String}
    reused::Set{String}
    rejected::Dict{String, String}
    effort::Symbol
    keep::Bool
    provider::String
end

function FFTPlanning(;
        offered = Dict{String, String}(), effort::Symbol = :measure,
        keep::Bool = true, provider::String = "fftw",
    )
    effort in (:estimate, :measure, :patient) ||
        throw(ArgumentError("FFT planning effort must be estimate, measure or patient"))
    provider == FFTW.fftw_provider == "fftw" ||
        throw(ArgumentError("FFT planning requires the loaded fftw provider"))
    copied = Dict{String, String}()
    for (key, wisdom) in offered
        key isa String && wisdom isa String ||
            throw(ArgumentError("FFT planning offers must contain string pairs"))
        copied[key] = wisdom
    end
    return FFTPlanning(
        copied, Dict{String, String}(), Dict{String, String}(), Set{String}(),
        Dict{String, String}(), effort, keep, provider,
    )
end

"""
    fft_planning_record(context::FFTPlanning)

Copy the preparation's measured/applied maps, reused keys and rejected-cache facts.

Return a detached named tuple of `measured`, `applied`, `reused` (sorted string vector)
and `rejected`. Only `measured` is published to the persistent cache; the other facts
describe actual preparation and permit host logging without Julia I/O. No returned
mutable container aliases the context.
"""
function fft_planning_record(context::FFTPlanning)
    return (
        measured = copy(context.measured), applied = copy(context.applied),
        reused = sort!(collect(context.reused)), rejected = copy(context.rejected),
    )
end

function fft_planning_transfer(context::FFTPlanning)
    packed(values) = Tuple((key, values[key]) for key in sort!(collect(keys(values))))
    return (
        measured = packed(context.measured), applied = packed(context.applied),
        rejected = packed(context.rejected), reused = Tuple(sort!(collect(context.reused))),
    )
end

"""
    release_fft_planning!(context::FFTPlanning)

Release preparation-only wisdom after the host has detached the planning facts.

Replace all five maps/sets so retained execution contexts hold no cache strings or
their former container capacity. This changes no FFT plan, array or global wisdom.
Call only after all numerical construction using this private context has finished.
Return `nothing`.
"""
function release_fft_planning!(context::FFTPlanning)
    context.offered = Dict{String, String}()
    context.measured = Dict{String, String}()
    context.applied = Dict{String, String}()
    context.rejected = Dict{String, String}()
    context.reused = Set{String}()
    return nothing
end

"""
    fft_runtime_identity()

Return pointer-free strings identifying Julia and its loaded FFTW provider/libraries.

The named tuple contains `julia_version`, `julia_machine`, `julia_cpu_name`, `provider`,
`fftw_package_version`, `fftw_library_version`, `library_f64` and `library_f32`. The
adapter must match/hash the actual library files against its verified setup inventory;
paths or version text alone are not a persistent-cache fingerprint.
"""
function fft_runtime_identity()
    return FFTW.@exclusive (
        julia_version = string(VERSION), julia_machine = Sys.MACHINE,
        julia_cpu_name = Sys.CPU_NAME, provider = FFTW.fftw_provider,
        fftw_package_version = string(Base.pkgversion(FFTW)),
        fftw_library_version = string(FFTW.get_version()),
        library_f64 = String(FFTW.libfftw3_no_init),
        library_f32 = String(FFTW.libfftw3f_no_init),
    )
end

_fft_precision(::Type{Float32}) = Float32
_fft_precision(::Type{ComplexF32}) = Float32
_fft_precision(::Type{Float64}) = Float64
_fft_precision(::Type{ComplexF64}) = Float64

# All these C functions are called only inside the complete fftwlock transaction.
# FFTW's string exporter specifies libc free, not fftw_free:
# https://www.fftw.org/fftw3_doc/Wisdom-Export.html
for (scalar, stem, library) in (
        (Float64, :fftw, :libfftw3), (Float32, :fftwf, :libfftw3f),
    )
    export_name = Symbol(stem, :_export_wisdom_to_string)
    import_name = Symbol(stem, :_import_wisdom_from_string)
    forget_name = Symbol(stem, :_forget_wisdom)
    get_threads = Symbol(stem, :_planner_nthreads)
    set_threads = Symbol(stem, :_plan_with_nthreads)
    @eval begin
        function _fft_export(::Type{$scalar})
            pointer = ccall(($(QuoteNode(export_name)), FFTW.$library), Ptr{UInt8}, ())
            pointer == C_NULL && throw(OutOfMemoryError())
            try
                return unsafe_string(pointer)
            finally
                Base.Libc.free(pointer)
            end
        end
        function _fft_import(::Type{$scalar}, wisdom::String)
            occursin('\0', wisdom) && return false
            return ccall(
                ($(QuoteNode(import_name)), FFTW.$library), Cint, (Cstring,), wisdom,
            ) != 0
        end
        _fft_forget(::Type{$scalar}) =
            ccall(($(QuoteNode(forget_name)), FFTW.$library), Cvoid, ())
        _fft_threads(::Type{$scalar}) =
            Int(ccall(($(QuoteNode(get_threads)), FFTW.$library), Cint, ()))
        _fft_threads!(::Type{$scalar}, threads::Int) = ccall(
            ($(QuoteNode(set_threads)), FFTW.$library), Cvoid, (Cint,), threads,
        )
    end
end

function _fft_storage(
        array::StridedArray{T, N},
    ) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}, N}
    width = sizeof(T)
    bytes = map(stride -> Base.checked_mul(stride, width), strides(array))
    _transfer_writable_strides(size(array), bytes, width)
    return nothing
end

function _fft_validate(input, output, axes::Tuple, kind::Symbol, threads::Int)
    input isa StridedArray && output isa StridedArray ||
        throw(ArgumentError("FFT planning requires supported strided arrays"))
    _transfer_writable_destination(input)
    _transfer_writable_destination(output)
    eltype(input) in (Float32, Float64, ComplexF32, ComplexF64) &&
        eltype(output) in (Float32, Float64, ComplexF32, ComplexF64) ||
        throw(ArgumentError("FFT planning requires binary32 or binary64 arrays"))
    # Keep each array's fixed scalar type through validation; real/complex pairs
    # need not share an element type, while their byte layouts must both be valid.
    _fft_storage(input)
    _fft_storage(output)
    _fft_precision(eltype(input)) === _fft_precision(eltype(output)) ||
        throw(ArgumentError("FFT planning precisions differ"))
    ndims(input) == ndims(output) && all(>(0), size(input)) &&
        all(>(0), size(output)) ||
        throw(ArgumentError("FFT planning needs matching ranks and nonempty extents"))
    !isempty(axes) && all(axis -> axis isa Int && 1 <= axis <= ndims(input), axes) ||
        throw(ArgumentError("FFT planning axes must be nonempty in-range integers"))
    length(unique(axes)) == length(axes) ||
        throw(ArgumentError("FFT planning axes must be distinct"))
    1 <= threads <= typemax(Cint) ||
        throw(ArgumentError("FFT planning threads must be positive C integers"))
    inplace = input === output
    !inplace && _transfer_arrays_alias(input, output) &&
        throw(ArgumentError("FFT planning input and output partially alias"))
    if kind in (:c2c_forward, :c2c_backward)
        eltype(input) <: Complex && eltype(output) === eltype(input) &&
            size(input) == size(output) ||
            throw(ArgumentError("complex FFT planning needs equal complex array shapes"))
    elseif kind in (:r2c, :c2r)
        length(axes) == 1 && !inplace ||
            throw(ArgumentError("real FFT planning requires one out-of-place axis"))
        real_array, complex_array = kind === :r2c ? (input, output) : (output, input)
        eltype(real_array) <: Real && eltype(complex_array) <: Complex ||
            throw(ArgumentError("real FFT planning has incompatible input/output types"))
        for dim in 1:ndims(input)
            extent = size(real_array, dim)
            required = dim == only(axes) ? fld(extent, 2) + 1 : extent
            size(complex_array, dim) == required ||
                throw(ArgumentError("real FFT planning has incompatible extents"))
        end
    else
        throw(ArgumentError("unknown FFT planning transform kind"))
    end
    return inplace ? UInt32(0) : FFTW.PRESERVE_INPUT
end

function _fft_key(input, output, axes::Tuple, kind::Symbol, threads::Int, restrictions)
    precision = _fft_precision(eltype(input)) === Float64 ? "f64" : "f32"
    return join(
        (
            "solprop-fft-1", precision, kind, size(input), size(output),
            strides(input), strides(output), axes, input === output ? "in" : "out",
            FFTW.alignment_of(input), FFTW.alignment_of(output), threads, restrictions,
        ), '|',
    )
end

function _fft_make_plan(
        input::StridedArray{T, N}, output::StridedArray{T, N}, axes, kind, flags,
    ) where {T <: Union{ComplexF32, ComplexF64}, N}
    if kind === :c2c_forward
        return FFTW.cFFTWPlan{T, FFTW.FORWARD, input === output, N}(
            input, output, axes, flags, FFTW.NO_TIMELIMIT,
        )
    else
        return FFTW.cFFTWPlan{T, FFTW.BACKWARD, input === output, N}(
            input, output, axes, flags, FFTW.NO_TIMELIMIT,
        )
    end
end

function _fft_make_plan(
        input::StridedArray{T, N}, output::StridedArray{Complex{T}, N}, axes, kind, flags,
    ) where {T <: Union{Float32, Float64}, N}
    return FFTW.rFFTWPlan{T, FFTW.FORWARD, false, N}(
        input, output, axes, flags, FFTW.NO_TIMELIMIT,
    )
end

function _fft_make_plan(
        input::StridedArray{Complex{T}, N}, output::StridedArray{T, N}, axes, kind, flags,
    ) where {T <: Union{Float32, Float64}, N}
    return FFTW.rFFTWPlan{Complex{T}, FFTW.BACKWARD, false, N}(
        input, output, axes, flags, FFTW.NO_TIMELIMIT,
    )
end

function _fft_cached_plan(input, output, axes, kind, flags)
    try
        return _fft_make_plan(input, output, axes, kind, flags | FFTW.WISDOM_ONLY)
    catch error
        # The pinned FFTW constructors use this exact error for C_NULL. Nothing
        # else is a cache miss: array/FFI/other preparation failures must propagate.
        error isa ErrorException && error.msg == "FFTW could not create plan" || rethrow()
        return nothing
    end
end

function _fft_transaction!(
        context::FFTPlanning, input::StridedArray{Ti, N}, output::StridedArray{To, N},
        axes::Tuple, kind::Symbol, threads::Int, restrictions::UInt32,
    ) where {
        Ti <: Union{Float32, Float64, ComplexF32, ComplexF64},
        To <: Union{Float32, Float64, ComplexF32, ComplexF64}, N,
    }
    precision = _fft_precision(eltype(input))
    saved_input = copy(input)
    saved_output = input === output ? nothing : copy(output)
    global_wisdom = _fft_export(precision)
    global_threads = _fft_threads(precision)
    key = _fft_key(input, output, axes, kind, threads, restrictions)
    flags = restrictions | (
        context.effort === :estimate ? FFTW.ESTIMATE :
            context.effort === :patient ? FFTW.PATIENT : FFTW.MEASURE
    )
    rejected = Dict{String, String}()
    reused = false
    local plan, applied
    try
        _fft_forget(precision)
        _fft_threads!(precision, threads)
        seed = get(context.measured, key, get(context.offered, key, nothing))
        if seed === nothing
            # Only this context's own earlier measurements may contribute shared
            # subplans for a new key. Never seed from ambient process wisdom.
            marker = precision === Float64 ? "solprop-fft-1|f64|" : "solprop-fft-1|f32|"
            for (prior, wisdom) in context.measured
                startswith(prior, marker) && _fft_import(precision, wisdom)
            end
        elseif !_fft_import(precision, seed)
            rejected[key] = "FFTW refused the cached wisdom string"
            _fft_forget(precision)
        end
        plan = seed === nothing || haskey(rejected, key) ? nothing :
            _fft_cached_plan(input, output, axes, kind, flags)
        if plan === nothing
            if seed !== nothing && !haskey(rejected, key)
                rejected[key] = "cached wisdom does not cover this transform and effort"
                _fft_forget(precision)
            end
            plan = _fft_make_plan(input, output, axes, kind, flags)
        else
            reused = true
        end
        FFTW.assert_applicable(plan, input, output)
        applied = _fft_export(precision)
    finally
        # Restore exact caller storage even when planning or export failed. The
        # nested finally blocks still restore FFTW state if an array copy fails.
        try
            copyto!(input, saved_input)
            saved_output === nothing || copyto!(output, saved_output)
        finally
            try
                _fft_forget(precision)
                _fft_import(precision, global_wisdom) ||
                    error("FFTW refused restoration of its own wisdom")
            finally
                _fft_threads!(precision, global_threads)
            end
        end
    end
    context.applied[key] = applied
    merge!(context.rejected, rejected)
    reused && push!(context.reused, key)
    if context.keep && !reused && context.effort !== :estimate
        context.measured[key] = applied
    end
    return plan
end

"""
    plan_owned_fft!(context, input, output, axes::Tuple, kind::Symbol; threads=1)

Plan against the actual owned input/output arrays and preserve their values exactly.

`kind` is `:c2c_forward`, `:c2c_backward`, `:r2c` or `:c2r`; transforms use the raw
FFTW sign/normalization (backward is unnormalized). Binary32/binary64 precision must
match. Complex transforms permit identical arrays for in-place execution; real
transforms require distinct arrays and one axis. Arrays must have supported writable,
injective strided layouts and cannot partially alias. `threads` is a positive C integer.

Return the concrete FFTW plan, checked against both supplied arrays. Planning may
allocate and overwrite arrays temporarily; a finally path restores their values and
unrelated FFTW wisdom/thread policy before publishing context facts. The complete
transaction holds FFTW's planner lock. The transform key includes all extents, strides,
axes, placement, actual alignment classes, precision, threads and execution flags;
planning effort is excluded. Out-of-place execution preserves input.

Throw `ArgumentError` on invalid admission or propagate a planner failure. Failed
planning publishes no context facts. Only `fftw` provider plans are qualified here.

FFTW's primary contract distinguishes planning writes from execution preservation:
https://www.fftw.org/fftw3_doc/Planner-Flags.html . New-array restrictions and serialized
planner access are specified in
https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html
and https://www.fftw.org/fftw3_doc/Thread-safety.html . No approximate FFT is introduced.
"""
function plan_owned_fft!(
        context::FFTPlanning, input, output, axes::Tuple, kind::Symbol; threads::Int = 1,
    )
    context.provider == FFTW.fftw_provider == "fftw" ||
        throw(ArgumentError("FFT planning provider changed"))
    context.effort in (:estimate, :measure, :patient) ||
        throw(ArgumentError("FFT planning effort must be estimate, measure or patient"))
    restrictions = _fft_validate(input, output, axes, kind, threads)
    return FFTW.@exclusive _fft_transaction!(
        context, input, output, axes, kind, threads, restrictions,
    )
end
