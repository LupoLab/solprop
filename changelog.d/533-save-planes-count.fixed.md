An integer `save_planes` in a recipe's `SimulationSettings` table, on the radial, free
and planar routes, is now a count of planes, as `n_save` is on the keyword routes:
`save_planes = 80` saves 80 evenly spaced planes over `grid.length`, counting the
entrance and ending at `grid.length`, where it was read as one plane at z = 80 m. A
count below 2 or a boolean is refused; a list is still positions in metres. In Python,
`SimulationSettings` now refuses a bare integer `save_planes`, which also read as one
plane that many metres away: pass positions, or the new
`solprop.driver.settings.counted_save_planes(grid.length, count)`. `Recipe.check()` now
warns when an object route's last save plane is not `grid.length`
([#533](https://github.com/LupoLab/solprop/issues/533)).
