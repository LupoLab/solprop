# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the manual's short scalar Kerr capillary with the private Julia backend."""

import sys

from solprop.driver import Simulation
from solprop.stepper import RungeKutta


def main() -> None:
    """Propagate four modes to 0.1 mm and release the in-memory output."""
    simulation = Simulation.from_capillary(
        125e-6,
        0.01,
        "Ar",
        1.0,
        lambda0=800e-9,
        wavelength_limits=(300e-9, 2e-6),
        trange=100e-15,
        fwhm=15e-15,
        energy=1e-6,
        modes=4,
        plasma=False,
        raman=False,
        integrator=RungeKutta(),
        backend="julia",
        save_planes=[1e-4],
    )
    output = simulation.run(progress="none")
    output.close()
    sys.stdout.write("Julia four-mode Kerr example completed at z = 0.1 mm.\n")


if __name__ == "__main__":
    main()
