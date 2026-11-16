Fixed a recipe pinning the capillary's `pressure` beside a temperature path failing
with a `TypeError` inside the solver: `solprop.recipe.pins.solve_pin` now refuses it
with a `RecipeError` on the pin, since a pinned pressure needs one temperature.
