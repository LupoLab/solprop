`SimulationSettings` now refuses a `resume_history` that is not `True`, `False` or
`None`. A recipe's `[execution]` table that wrote `resume_history = "none"` built a
run, and a Python export, that kept the Adams history always, the opposite of what it
said; omit the key for the automatic choice
([#703](https://github.com/LupoLab/solprop/issues/703)).
