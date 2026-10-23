# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Radial free-space witnesses.
# From the solprop worktree: julia --startup-file=no tests/data/luna_radial.jl reduced|full OUTPUT_DIRECTORY
# These are the literal weak-energy examples, not a claim of self-focusing.
using Luna
using SHA
using Dates
import Luna: Hankel
import LinearAlgebra: BLAS

length(ARGS) == 2 || error("usage: luna_radial.jl reduced|full OUTPUT_DIRECTORY")
const TIER = ARGS[1]
TIER in ("reduced", "full") || error("tier must be reduced or full")
const DIRECTORY = abspath(ARGS[2])
const WITNESS_RTOL = parse(Float64, get(ENV, "SOLPROP_LUNA_RTOL", "1e-6"))
const WITNESS_ATOL = parse(Float64, get(ENV, "SOLPROP_LUNA_ATOL", "1e-10"))
0 < WITNESS_RTOL < 1 || error("SOLPROP_LUNA_RTOL must lie in (0,1)")
WITNESS_ATOL > 0 || error("SOLPROP_LUNA_ATOL must be positive")
mkpath(DIRECTORY)
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(parse(Int, get(ENV, "SOLPROP_LUNA_FFTW_THREADS", "1")))

function radial_witness(family)
    selected = get(ENV, "SOLPROP_LUNA_CASE", "all")
    selected in ("all", family) || return
    reduced = TIER == "reduced"
    distance = reduced ? 0.006 : 0.6
    radius = reduced ? 0.6e-3 : 4e-3
    nodes = reduced ? 128 : family == "field" ? 1024 : 512
    propz = reduced ? -0.003 : -0.3
    wavelength = 800e-9
    fwhm = 20e-15
    energy = 2e-9
    waist = 40e-6
    limits = (400e-9, 2000e-9)
    window = 0.2e-12
    gas, pressure = :Ar, 1.2
    grid = family == "field" ?
        Grid.RealGrid(distance, wavelength, limits, window) :
        Grid.EnvGrid(distance, wavelength, limits, window)
    q = Hankel.QDHT(radius, nodes, dim=2)
    density = PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = family == "field" ?
        (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),) :
        (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
    index = PhysData.ref_index_fun(gas, pressure)
    linear = LinearOps.make_const_linop(grid, q, index)
    normalisation = NonlinearRHS.const_norm_radial(grid, q, index)
    input = Fields.GaussGaussField(λ0=wavelength, τfwhm=fwhm,
                                 energy=energy, w0=waist, propz=propz)
    field, transform, fft = Luna.setup(grid, q, densityfun, normalisation, responses, input)
    filename = joinpath(DIRECTORY, "radial_$(family)_$(TIER).h5")
    isfile(filename) && error("refusing to overwrite witness $filename")
    output = Output.HDF5Output(filename, 0, distance, reduced ? 9 : 201)
    source = "examples/low_level_interface/freespace/" *
             (family == "field" ? "radial.jl" : "radial_env.jl")
    provenance = Dict(
        "name" => "literal_radial_$(family)", "family" => family, "tier" => TIER,
        "source_example" => source, "generator" => basename(@__FILE__),
        "generator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "luna_source" => pathof(Luna), "julia_version" => string(VERSION),
        "luna_commit" => readchomp(`git -C $(dirname(pathof(Luna))) rev-parse HEAD`),
        "luna_dirty" => !isempty(readchomp(`git -C $(dirname(pathof(Luna))) status --porcelain`)),
        "luna_version" => string(Base.pkgversion(Luna)), "created_utc" => string(now(UTC)),
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "fftw_threads" => Luna.Utils.FFTWthreads(), "fftw_planning" => "estimate",
        "blas_config" => string(BLAS.get_config()), "distance" => distance,
        "radius" => radius, "n_r" => nodes, "time_window" => window,
        "wavelength" => wavelength, "lambda_min" => limits[1], "lambda_max" => limits[2],
        "fwhm" => fwhm, "energy" => energy, "waist" => waist, "propz" => propz,
        "spatial_order" => 1, "temporal_order" => 1, "gas" => string(gas),
        "pressure_bar" => pressure, "density" => density, "temperature" => PhysData.roomtemp,
        "chi3_per_particle" => PhysData.γ3_gas(gas), "c" => PhysData.c,
        "epsilon0" => PhysData.ε_0, "rtol" => WITNESS_RTOL, "atol" => WITNESS_ATOL,
        "spatial_absorber" => false, "input_prepropagation_medium" => "vacuum (Fields.prop!)",
        "input_evanescent_model" => "kz squared clamped to zero; no evanescent attenuation",
        "source_corrections" => "Output redirected to HDF5; plotting and unused ionisation-table construction omitted. No response physics changed.",
        "reduced_changes" => reduced ? "L=0.006 m; propz=-0.003 m; R=0.6 mm; nr=128; saveN=9" : "none",
        "field_layout" => "omega,r,z; internal FFT convention",
    )
    output(provenance; group="witness")
    output("r", q.r; group="witness")
    output("k_r", q.k; group="witness")
    started = time_ns()
    Luna.run(field, grid, linear, transform, fft, output; rtol=WITNESS_RTOL, atol=WITNESS_ATOL)
    output("duration_seconds", (time_ns() - started) / 1e9; group="witness")
    println("WITNESS\t", filename)
end

radial_witness("field")
radial_witness("envelope")
