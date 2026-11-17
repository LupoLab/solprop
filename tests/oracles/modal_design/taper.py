# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The taper note's geometry: a capillary tapering between two straight ends.

The prototype's own modal and mapped-coordinate FEM solves of this taper were run by
hand; their receipt is stored in ``tests/data/taper_note.json``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Taper:
    length: float = 0.01
    smooth: bool = False
    loss: bool = False
    initial_radius: float = 125e-6
    final_radius: float = 62.5e-6
    straight: float = 0.005
    # The taper note's carrier. A field of the case rather than a literal so that the
    # production study can ask the same reference at a shorter wavelength; the default
    # keeps every recorded receipt's case exactly as it was.
    wavelength: float = 800e-9

    @property
    def events(self):
        return (
            0,
            self.straight,
            self.straight + self.length,
            2 * self.straight + self.length,
        )

    def geometry(self, z):
        fraction = np.clip((z - self.straight) / self.length, 0, 1)
        active = self.straight < z < self.straight + self.length
        shape = 3 * fraction**2 - 2 * fraction**3 if self.smooth else fraction
        slope = (
            (6 * fraction - 6 * fraction**2 if self.smooth else 1) / self.length
            if active
            else 0.0
        )
        difference = self.final_radius - self.initial_radius
        return self.initial_radius + difference * shape, difference * slope
