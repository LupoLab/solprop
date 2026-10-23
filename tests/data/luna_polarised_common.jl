# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared witness machinery for the polarised spatial witnesses. Launch from solprop;
# keep the Luna checkout unchanged.
using Luna
using SHA
using Dates
import LinearAlgebra: BLAS

length(ARGS) == 2 || error("usage: GENERATOR.jl reduced|full OUTPUT_DIRECTORY")
const WITNESS_TIER = ARGS[1]
WITNESS_TIER in ("reduced", "full") || error("tier must be reduced or full")
const WITNESS_DIRECTORY = abspath(ARGS[2])
const WITNESS_RTOL = parse(Float64, get(ENV, "SOLPROP_LUNA_RTOL", "1e-6"))
const WITNESS_ATOL = parse(Float64, get(ENV, "SOLPROP_LUNA_ATOL", "1e-10"))
0 < WITNESS_RTOL < 1 || error("SOLPROP_LUNA_RTOL must lie in (0,1)")
WITNESS_ATOL > 0 || error("SOLPROP_LUNA_ATOL must be positive")
mkpath(WITNESS_DIRECTORY)
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(parse(Int, get(ENV, "SOLPROP_LUNA_FFTW_THREADS", "1")))

function modal_witness(config, generator)
    selected = get(ENV, "SOLPROP_LUNA_CASE", "all")
    selected in ("all", config.name) || return
    reduced = WITNESS_TIER == "reduced"
    distance = reduced ? config.distance / 100 : config.distance
    time_window = reduced ? 400e-15 : config.time_window
    grid = config.family == "field" ?
        Grid.RealGrid(distance, config.wavelength, config.limits, time_window) :
        Grid.EnvGrid(distance, config.wavelength, config.limits, time_window)
    modes = Tuple(Capillary.MarcatiliMode(
        config.radius, config.gas, config.pressure; kind=entry.kind,
        n=entry.n, m=entry.m, ϕ=entry.phi, loss=config.loss) for entry in config.modes)
    density = PhysData.density(config.gas, config.pressure)
    densityfun(z) = density
    kerr = config.family == "field" ? Nonlinear.Kerr_field(PhysData.γ3_gas(config.gas)) :
        Nonlinear.Kerr_env(PhysData.γ3_gas(config.gas))
    responses = if config.plasma
        potential = PhysData.ionisation_potential(config.gas)
        rate = Ionisation.IonRateADK(potential)
        scratch = config.components == :xy ? zeros(length(grid.to), 2) : zeros(length(grid.to))
        (kerr, Nonlinear.PlasmaCumtrapz(grid.to, scratch, rate, potential))
    else
        (kerr,)
    end
    input = Fields.GaussField(λ0=config.wavelength, τfwhm=config.fwhm, energy=config.energy)
    field, transform, fft = Luna.setup(
        grid, densityfun, responses, input, modes, config.components;
        full=config.full, nr=reduced ? 24 : 64, nθ=16, kronrod=false)
    template = copy(field[:, 1])
    field .= template * transpose(config.weights)
    linear = LinearOps.make_const_linop(grid, modes, config.wavelength)
    stats = Stats.collect_stats(grid, field, Stats.ω0(grid))
    path = joinpath(WITNESS_DIRECTORY, "$(config.name)_$(WITNESS_TIER).h5")
    isfile(path) && error("refusing to overwrite witness $path")
    output = Output.HDF5Output(path, 0, distance, reduced ? 9 : 201, stats)
    provenance = Dict(
        "name" => config.name, "tier" => WITNESS_TIER, "family" => config.family,
        "source_example" => config.source, "generator" => basename(generator),
        "generator_sha256" => bytes2hex(sha256(read(generator))),
        "common_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "luna_source" => pathof(Luna), "julia_version" => string(VERSION),
        "luna_commit" => readchomp(`git -C $(dirname(pathof(Luna))) rev-parse HEAD`),
        "luna_dirty" => !isempty(readchomp(`git -C $(dirname(pathof(Luna))) status --porcelain`)),
        "luna_version" => string(Base.pkgversion(Luna)), "created_utc" => string(now(UTC)),
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "fftw_threads" => Luna.Utils.FFTWthreads(),
        "fftw_planning" => "estimate", "blas_config" => string(BLAS.get_config()),
        "radius" => config.radius, "pressure_bar" => config.pressure,
        "gas" => string(config.gas), "density" => density,
        "temperature" => PhysData.roomtemp, "wavelength" => config.wavelength,
        "fwhm" => config.fwhm, "energy" => config.energy, "distance" => distance,
        "time_window" => time_window, "lambda_min" => config.limits[1],
        "lambda_max" => config.limits[2], "rtol" => WITNESS_RTOL, "atol" => WITNESS_ATOL,
        "radius_nodes" => reduced ? 24 : 64, "theta_nodes" => 16,
        "components" => string(config.components), "full_rule" => config.full,
        "mode_kinds" => string.([entry.kind for entry in config.modes]),
        "mode_n" => [entry.n for entry in config.modes],
        "mode_m" => [entry.m for entry in config.modes],
        "mode_phi" => [entry.phi for entry in config.modes],
        "plasma" => config.plasma, "loss" => config.loss,
        "weights_real" => collect(real.(config.weights)),
        "weights_imag" => collect(imag.(config.weights)),
        "source_corrections" => config.corrections,
        "c" => PhysData.c, "epsilon0" => PhysData.ε_0,
        "chi3_per_particle" => PhysData.γ3_gas(config.gas),
    )
    output(provenance; group="witness")
    started = time_ns()
    Luna.run(field, grid, linear, transform, fft, output; rtol=WITNESS_RTOL, atol=WITNESS_ATOL)
    output("duration_seconds", (time_ns() - started) / 1e9; group="witness")
    println("WITNESS\t", path)
end

he(n, m; phi=0.0) = (kind=:HE, n=n, m=m, phi=phi)
transverse(kind, m=1) = (kind=kind, n=0, m=m, phi=0.0)
