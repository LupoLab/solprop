# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
An interactive session: design, run, compress
=============================================

This page is written as you would type it at a terminal or in a notebook: one import,
then everything is an attribute of ``sol``. The case is the everyday one of a
hollow-fibre compressor. A 30 fs, 100 uJ pulse at 800 nm is spectrally broadened in a
metre of 125 um capillary filled with argon, and the positively chirped pulse that comes
out is compressed with a spectral phase, with glass, and with chirped mirrors and
wedges.

The import configures logging and silences the material-range notices, for convenience
in an
exploratory session. Restore those warnings when
assessing a material model outside its fitted range.
:doc:`/guide/workflows/interactive` says exactly what the import does and how to undo
it.
"""

# %%
# One import. The log level is lowered here only to keep this page short; at a prompt
# the default, ``INFO``, reports the set-up of each run and draws its progress bar.

import matplotlib.pyplot as plt

import solprop.sol as sol  # noqa: PLR0402 - the form the manual teaches

sol.set_log_level("WARNING")

# %%
# Design
# ------
# What will this pulse do in this fibre? ``capillary_params`` gives the numbers, and
# printing the result lays them out as one block: the dispersion at the pump, the
# characteristic lengths, the soliton order and the powers.

radius, gas, pressure = 125e-6, "Ar", 1.0
pulse = {"energy": 100e-6, "fwhm": 30e-15, "wavelength": 800e-9}

params = sol.capillary_params(radius, gas, pressure, shape="gauss", **pulse)
print(params)

# %%
# The dispersion is **normal** at the pump (the zero-dispersion wavelength is on its red
# side), the peak power is a quarter of the critical power, and the nonlinear length is
# 11 cm: a metre of this fibre gives a B-integral of about nine, which is spectral
# broadening by self-phase modulation with a positive chirp. That is a compressor.
#
# The same tools answer the opposite question. At what pressure would the pump sit at
# the zero-dispersion point, and if we went below it, into the soliton regime, where
# would the resonant dispersive wave come out?

at_zdw = sol.pressure_zdw(radius, gas, pulse["wavelength"])
print(f"the zero-dispersion wavelength reaches 800 nm at {at_zdw:.2f} bar")

thin = sol.MarcatiliMode(radius, gas, 0.2)
linear = sol.lambda_rdw(thin, pulse["wavelength"])
gamma = sol.nonlinear_coefficient(
    thin,
    sol.n2(gas, pressure=0.2, wavelength=pulse["wavelength"]),
    pulse["wavelength"],
)
kerr = sol.lambda_rdw(
    thin,
    pulse["wavelength"],
    peak_power=params.peak_power,
    gamma=gamma,
    fwhm=pulse["fwhm"],
    shape="gauss",
)
print(f"at 0.2 bar the dispersive wave is phase matched at {linear * 1e9:.0f} nm,")
print(f"or {kerr * 1e9:.0f} nm with the soliton's nonlinear wavenumber included;")
target = sol.pressure_rdw(radius, gas, 250e-9, pulse["wavelength"])
print(f"for a wave at 250 nm, fill to {target:.2f} bar")

# %%
# The same phase matching applies to an antiresonant fibre. Here d/D is the
# **inner** resonator diameter divided by the inscribed core diameter. The model's
# wall resonances can produce several roots; phase matching does not ensure low loss.

lattice = sol.TubeLattice.from_inner_diameter_ratio(20e-6, 300e-9, 0.6, 6)
antiresonant = sol.VincettiMode(20e-6, "He", 2.0, lattice=lattice)
antiresonant_gamma = sol.nonlinear_coefficient(
    antiresonant, sol.n2("He", pressure=2.0, wavelength=800e-9), 800e-9
)
roots = sol.lambda_rdw_all(
    antiresonant,
    800e-9,
    peak_power=1e9,
    gamma=antiresonant_gamma,
    fwhm=30e-15,
    shape="gauss",
    wavelength_limits=(210e-9, 700e-9),
)
print("Vincetti RDW roots (nm):", [round(root * 1e9, 1) for root in roots])

# %%
# Run
# ---
# One call. The pulse is a Gaussian, the default; the run takes a second or two.

run = sol.prop_capillary(
    radius,
    1.0,  # length, m
    gas,
    pressure,
    wavelength=pulse["wavelength"],
    wavelength_limits=(400e-9, 1.6e-6),
    trange=600e-15,
    fwhm=pulse["fwhm"],
    energy=pulse["energy"],
    n_save=41,
    progress="none",
    seed=20261027,
)
print(run)

# %%
# The propagation maps: the spectrum broadens symmetrically and the pulse stretches, as
# self-phase modulation in normal dispersion does.

figures = sol.prop_2d(run, "wavelength", spec_range=(500e-9, 1.2e-6), db_min=-30)

# %%
# Compress
# --------
# How short could the output be, and what gets it there? The transform limit first,
# then a pure group-delay dispersion, then dispersion and third order together.

limit_fwhm, limit_power = sol.transform_limit(run)
print(f"transform limit: {limit_fwhm * 1e15:.2f} fs, {limit_power * 1e-9:.1f} GW")

gdd = sol.compress_taylor(run)
gdd_tod = sol.compress_taylor(run, order=3)
print(gdd)
print(gdd_tod)

# %%
# About -40 fs\ :sup:`2` does most of the work, reaching 0.86 of the transform-limited
# peak power, and adding third order recovers much of what is left. Neither is a thing
# you can buy. Glass has the wrong sign for this pulse, and the search says so: the
# best it can do is *remove* glass that is not there.

print(sol.compress_material(run, medium="CaF2", thickness_range=(-5e-3, 5e-3)))

# %%
# What is on the table is a set of chirped mirrors, which over-compensate in whole
# bounces, and a pair of wedges to trim the rest. ``compress_mirrors`` tries every count
# and tunes the insertion for each.

mirrors = sol.compress_mirrors(
    run,
    mirror="PC70",
    max_reflections=6,
    medium="CaF2",
    thickness_range=(0.0, 3e-3),
)
print(mirrors)

# %%
# The standard plots take the element too: the pulse behind the mirrors and wedges, and
# the spectrum it was compressed from.

sol.time_1d(run, transfer=mirrors.transfer)
sol.spec_1d(run)

# %%
# Every result carries its transfer function, which the time-domain accessors and plots
# take directly: ``sol.time_1d(run, transfer=mirrors.transfer)`` draws the compressed
# pulse. Here the three are drawn together: the pulse as it leaves the fibre, after the
# mirrors and wedges, and at its transform limit.

figure, axis = plt.subplots(figsize=(7.0, 4.0))
cases = (
    ("as it leaves the fibre", {}, "C0"),
    ("mirrors and wedges", {"transfer": mirrors.transfer}, "C1"),
    ("transform limit", {"transform_limit": True}, "0.5"),
)
for label, keywords, colour in cases:
    t, power = sol.get_power(run, plane=-1, oversampling=4, **keywords)
    width = sol.fwhm(t, power) * 1e15
    print(f"{label}: {power.max() * 1e-9:.2f} GW, {width:.1f} fs")
    axis.plot(t * 1e15, power * 1e-9, color=colour, label=f"{label}, {width:.1f} fs")
axis.set_xlim(-80, 80)
axis.set_ylim(bottom=0.0)
axis.set_xlabel("Time (fs)")
axis.set_ylabel("Power (GW)")
axis.legend(frameon=False)
figure.tight_layout()

# %%
# A session ends when the interpreter does. To hand the process back to the library's
# defaults before that -- the range warnings visible, no log handler -- end it:

sol.end_session()

plt.show()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/workflows/interactive` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_02_soliton`.
