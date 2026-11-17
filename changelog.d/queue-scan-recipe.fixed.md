A recipe scan run on `Queue` or `Slurm` now records its recipe in the collected scan
file, as on `Local` and `Pool`, so `Recipe.from_output` reopens it with its setup.
