# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Cartesian free-space witnesses.
# From the solprop worktree: julia --startup-file=no tests/data/luna_free.jl reduced|full OUTPUT_DIRECTORY
# The literal field/envelope examples have DIFFERENT energies; retain both.
using Luna
using SHA
using Dates
import LinearAlgebra: BLAS

length(ARGS) == 2 || error("usage: luna_free.jl reduced|full OUTPUT_DIRECTORY")
const TIER = ARGS[1]
TIER in ("reduced", "full") || error("tier must be reduced or full")
const DIRECTORY = abspath(ARGS[2])
mkpath(DIRECTORY)
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(parse(Int, get(ENV, "SOLPROP_LUNA_FFTW_THREADS", "1")))

function free_witness(family)
    selected = get(ENV, "SOLPROP_LUNA_CASE", "all")
    selected in ("all", family) || return
    reduced = TIER == "reduced"
    distance = reduced ? 0.02 : 2.0
    radius = 6e-3
    nodes = reduced ? 32 : 128
    wavelength, fwhm, waist = 800e-9, 30e-15, 2e-3
    energy = family == "field" ? 1.5e-9 : 1.5e-3
    limits, window = (400e-9, 2000e-9), 0.2e-12
    gas, pressure = :Ar, 4.0
    initial_step = reduced ? 1e-3 : 1e-1
    grid = family == "field" ?
        Grid.RealGrid(distance, wavelength, limits, window) :
        Grid.EnvGrid(distance, wavelength, limits, window)
    space = Grid.FreeGrid(radius, nodes)
    density = PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = family == "field" ?
        (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),) :
        (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
    index = PhysData.ref_index_fun(gas, pressure)
    linear = LinearOps.make_const_linop(grid, space, index)
    normalisation = NonlinearRHS.const_norm_free(grid, space, index)
    input = Fields.GaussGaussField(λ0=wavelength, τfwhm=fwhm, energy=energy, w0=waist)
    field, transform, fft = Luna.setup(grid, space, densityfun, normalisation, responses, input)
    filename = joinpath(DIRECTORY, "free_$(family)_$(TIER).h5")
    isfile(filename) && error("refusing to overwrite witness $filename")
    output = Output.HDF5Output(filename, 0, distance, reduced ? 9 : 21)
    source = "examples/low_level_interface/freespace/" *
             (family == "field" ? "full3D.jl" : "full3D_env.jl")
    provenance = Dict(
        "name" => "literal_free_$(family)", "family" => family, "tier" => TIER,
        "source_example" => source, "generator" => basename(@__FILE__),
        "generator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "luna_source" => pathof(Luna), "julia_version" => string(VERSION),
        "luna_commit" => readchomp(`git -C $(dirname(pathof(Luna))) rev-parse HEAD`),
        "luna_dirty" => !isempty(readchomp(`git -C $(dirname(pathof(Luna))) status --porcelain`)),
        "luna_version" => string(Base.pkgversion(Luna)), "created_utc" => string(now(UTC)),
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "fftw_threads" => Luna.Utils.FFTWthreads(), "fftw_planning" => "estimate",
        "blas_config" => string(BLAS.get_config()),
        "distance" => distance, "radius_x" => radius, "radius_y" => radius,
        "n_x" => nodes, "n_y" => nodes, "time_window" => window,
        "wavelength" => wavelength, "lambda_min" => limits[1], "lambda_max" => limits[2],
        "fwhm" => fwhm, "energy" => energy, "waist" => waist, "propz" => 0.0,
        "literal_energy_difference" => "field=1.5e-9 J; envelope=1.5e-3 J; deliberately unmatched",
        "spatial_order" => 1, "temporal_order" => 1, "gas" => string(gas),
        "pressure_bar" => pressure, "density" => density, "temperature" => PhysData.roomtemp,
        "chi3_per_particle" => PhysData.γ3_gas(gas), "c" => PhysData.c,
        "epsilon0" => PhysData.ε_0, "rtol" => 1e-6, "atol" => 1e-10,
        "initial_step" => initial_step, "maximum_step" => "Inf",
        "spatial_absorber" => false, "grid_window" => "Luna FreeGrid default; saved xywin is not applied",
        "source_corrections" => "Output redirected to HDF5; plotting, unused NumericalIntegration import, unused ionisation construction and post-processing omitted. Input energies and response physics preserved.",
        "reduced_changes" => reduced ? "L=0.02 m; nx=ny=32; init_dz=0.001 m; saveN=9" : "none",
        "field_layout" => "omega,y,x,z; internal FFT convention; raw transverse DFT",
    )
    output(provenance; group="witness")
    for (name, values) in (("x", space.x), ("y", space.y), ("k_x", space.kx), ("k_y", space.ky), ("xywin", space.xywin))
        output(name, values; group="witness")
    end
    started = time_ns()
    Luna.run(field, grid, linear, transform, fft, output; rtol=1e-6, atol=1e-10,
             max_dz=Inf, init_dz=initial_step)
    output("duration_seconds", (time_ns()-started)/1e9; group="witness")
    println("WITNESS\t", filename)
end

free_witness("field")
free_witness("envelope")
