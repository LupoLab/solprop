`Simulation.run(..., resume=True)` no longer replaces the simulation's own noise
realisation with the one drawn from the resumed file's recorded seed. The redraw now
belongs to the resumed run alone, so a later fresh `run` of the same simulation is no
longer silently the resumed file's realisation; the resumed run itself is unchanged.
