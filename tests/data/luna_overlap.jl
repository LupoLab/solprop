# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The Luna side of tests/data/generate_luna_overlap.py. That script prepends a block of
# `const` definitions and runs the result under `julia --startup-file=no`; every name in
# capitals below is defined there. Read-only against the Luna checkout: this program
# calls Luna, writes its output files where it is told to, and changes nothing in the
# checkout. It is never run by the test suite.
using Luna
import Luna: Grid, Capillary, Fields, LinearOps, Modes, NonlinearRHS,
             Nonlinear, Output, PhysData
import LinearAlgebra: BLAS

emit(tag, values) = println(tag, "\t", join(string.(values), ","))
emitc(tag, values) = (emit(tag * "_RE", real.(values));
                      emit(tag * "_IM", imag.(values)))

# Both codes take their constants from a CODATA release, and the two releases differ;
# emitting Luna's lets the comparison say which of the two it is looking at.
emit("CONSTANTS", [PhysData.c, PhysData.ε_0, PhysData.roomtemp])
# The Julia, BLAS and FFTW thread counts this run actually used, read here
# rather than assumed -- every timing this script has ever recorded was taken with
# this same launch (`julia --startup-file=no`, no `-t`). The pool is encoded
# numerically since `emit` is numbers-only: 0 is "none" (a single Julia thread never
# raises the threading question, so `Luna.Utils._native_fftw_threads[]` stays
# `nothing`), 1 is libfftw3's native pthreads, 2 is FFTW.jl's Julia-task (partr)
# callback, 3 is "decided `false` for another reason".
let pool = if Threads.nthreads() == 1
        0
    elseif Luna.Utils._native_fftw_threads[] === true
        1
    elseif Luna.Utils._native_fftw_threads[] === false
        2
    else
        3
    end
    emit("THREADS", [Threads.nthreads(:default), BLAS.get_num_threads(),
                     Luna.Utils.FFTWthreads(), pool])
end

modeset(count) = [Capillary.MarcatiliMode(RADIUS, GAS, PRESSURE; T = TEMPERATURE, m = m)
                  for m = 1:count]

"Emit each mode's normalisation, effective area and profile, so they can be named."
function emit_modes(tag, ms)
    emit(tag * "_N", [Modes.N(m) for m in ms])
    emit(tag * "_AEFF", [Modes.Aeff(m) for m in ms])
    emit(tag * "_UNM", [m.unm for m in ms])
    emit(tag * "_RADII", RADIUS .* PROFILE_FRACTIONS)
    for (im, m) in enumerate(ms)
        # Exy is field/sqrt(N); for HE_1m at phi = 0 the x component vanishes exactly,
        # so the y component is the whole of the normalised profile -- in Luna's
        # convention; solprop's phi = 0 is x-polarised (ADR-031), so the comparison
        # reads solprop's component 0 against this column.
        emit("$(tag)_EXY_$(im)",
             [Modes.Exy(m, (RADIUS * f, 0.0))[2] for f in PROFILE_FRACTIONS])
    end
end

"""Emit the quartic overlap ratios Luna's own fixed modal transform realises.

A field that is the fundamental alone has a Kerr polarisation `ehat_1(r)^3 f(t)`, which
factorises, so the drive projected into mode p is `T_p f(t)` with
`T_p = int ehat_1^3 ehat_p dA`. Every later step of `_finish_modal!` -- the time window,
the transform, the spectral window and `norm_modal`'s `-i omega / 4` -- is the same for
every mode, so `nl[:, p] / nl[:, 1]` is exactly `T_p / T_1` at every frequency, whatever
the envelope or the response. That ratio is what the comparison checks.
"""
function projection_case(tag, nmodes, nr, kronrod)
    ms = modeset(nmodes)
    ts = Modes.ToSpace(ms, components = :y)
    grid = Grid.RealGrid(LENGTH, LAMBDA0, (LAMBDA_MIN, LAMBDA_MAX), TRANGE)
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(GAS)),)
    density = PhysData.density(GAS, PRESSURE, TEMPERATURE)
    transform = NonlinearRHS.TransModalFixed(
        grid, ts, resp, z -> density, NonlinearRHS.norm_modal(grid);
        full = false, nr = nr, kronrod = kronrod)
    key = "$(tag)_$(nr)_$(kronrod)"
    emit(key * "_POINTS", [transform.ncalls])

    # The discrete Gram matrix on Luna's own rule. The basis is orthonormal under
    # (1/2) eps0 c, so this is the identity unless the rule cannot integrate a *pair* of
    # profiles -- which would put mode 1 into mode p with no nonlinearity at all.
    emit(key * "_GRAM",
         vec(0.5 * PhysData.ε_0 * PhysData.c * (transform.S * transform.Wp)))

    Eω = zeros(ComplexF64, length(grid.ω), nmodes)
    @. Eω[:, 1] = exp(-((grid.ω - OMEGA0) / BANDWIDTH)^2)
    nl = similar(Eω)
    transform(nl, Eω, 0.0)
    # Only where the fundamental's own drive is well above rounding: elsewhere the ratio
    # is 0/0 and says nothing.
    band = abs.(nl[:, 1]) .> RATIO_FLOOR * maximum(abs.(nl[:, 1]))
    for p = 1:nmodes
        ratio = nl[band, p] ./ nl[band, 1]
        mean = sum(ratio) / length(ratio)
        # mean real part, mean imaginary part, and the spread across the band: the
        # ratio is exactly real and exactly constant, so the spread is the witness that
        # the factorisation argument above holds in the code as well as on paper.
        emit("$(key)_RATIO_$(p)",
             [real(mean), imag(mean), maximum(abs.(ratio .- mean)), sum(band)])
    end
end

"""Emit the same ratios from Luna's own mode matrices at a much higher radial order.

The internal control on `projection_case`: the same quantity, from the same profiles and
the same quadrature machinery, but at an order where the rule is exact to rounding and
without going through the transform at all.
"""
function reference_case(tag, nmodes, nr)
    ms = modeset(nmodes)
    ts = Modes.ToSpace(ms, components = :y)
    dl = Modes.dimlimits(ms[1], z = 0.0)
    quad = Modes.transverse_quadrature(dl, false; nr = nr, nθ = 1, kronrod = false)
    Ems = Modes.mode_matrix(ms, ts.indices, Modes.quadrature_nodes(quad, dl); z = 0.0)
    w = Modes.quadrature_weights(quad, dl)
    overlaps = [sum(w .* vec(Ems[1, 1, :]) .^ 3 .* vec(Ems[p, 1, :])) for p = 1:nmodes]
    emit("$(tag)_REFERENCE_$(nr)", overlaps ./ overlaps[1])
end

for (tag, nmodes) in (("REDUCED", REDUCED_MODES), ("FULL", FULL_MODES))
    emit_modes(tag, modeset(nmodes))
    reference_case(tag, nmodes, REFERENCE_NR)
    for nr in QUADRATURE_ORDERS, kronrod in (true, false)
        projection_case(tag, nmodes, nr, kronrod)
    end
end

# ------------------------------------------------------------ oracle 2: undepleted pump
#
# A weak pump in the fundamental, a weak seed in the second mode, a constant fill and no
# plasma, over a length short enough that the first-order solution of the coupled
# equations holds. Luna is run twice, with the Kerr term and with it switched off by a
# zero susceptibility, so the nonlinear change of each mode is a difference of two runs
# of the same code and no frame, transform or amplitude convention has to be modelled on
# the Python side. Driven through `Luna.setup`/`Luna.run` rather than `prop_capillary`
# because the interface cannot set a step tolerance at all.

"Emit the input and exit spectra of one undepleted-pump run."
function undepleted_case(tag, chi3_scale)
    grid = Grid.RealGrid(UNDEPLETED_LENGTH, LAMBDA0,
                         (UNDEPLETED_LAMBDA_MIN, UNDEPLETED_LAMBDA_MAX),
                         UNDEPLETED_TRANGE)
    ms = [Capillary.MarcatiliMode(RADIUS, GAS, UNDEPLETED_PRESSURE;
                                  T = TEMPERATURE, m = m) for m = 1:2]
    density = PhysData.density(GAS, UNDEPLETED_PRESSURE, TEMPERATURE)
    resp = (Nonlinear.Kerr_field(chi3_scale * PhysData.γ3_gas(GAS)),)
    # The seed is an amplitude fraction of the pump, so its energy is that fraction
    # squared; both are transform-limited Gaussians of the same width and carrier, so
    # the seed is exactly SEED times the pump, bin for bin.
    pump = Fields.GaussField(λ0 = LAMBDA0, τfwhm = UNDEPLETED_FWHM,
                             energy = UNDEPLETED_ENERGY)
    seed = Fields.GaussField(λ0 = LAMBDA0, τfwhm = UNDEPLETED_FWHM,
                             energy = UNDEPLETED_SEED^2 * UNDEPLETED_ENERGY)
    inputs = ((mode = 1, fields = (pump,)), (mode = 2, fields = (seed,)))
    # One frame for the set: `make_const_linop` takes beta_1 from `ref_mode` once and
    # gives it to every mode, so the modes stay simultaneous and their group-delay
    # walk-off is carried rather than transformed away.
    linop = LinearOps.make_const_linop(grid, ms, grid.referenceλ)
    Eω, transform, FT = Luna.setup(grid, z -> density, resp, inputs, ms, :y;
                                   full = false, modal_integral = :fixed,
                                   nr = UNDEPLETED_NR, kronrod = true)
    output = Output.MemoryOutput(0, grid.zmax, 2)
    Luna.run(Eω, grid, linop, transform, FT, output;
             rtol = UNDEPLETED_RTOL, atol = UNDEPLETED_ATOL)
    stored = output.data["Eω"]  # (nω, nmodes, nz)
    emit(tag * "_OMEGA", grid.ω)
    for (plane, name) in ((1, "IN"), (size(stored, 3), "OUT")), m = 1:2
        emitc("$(tag)_$(name)_$(m)", stored[:, m, plane])
    end
end

undepleted_case("UNDEPLETED_NL", 1.0)
undepleted_case("UNDEPLETED_LIN", 0.0)
