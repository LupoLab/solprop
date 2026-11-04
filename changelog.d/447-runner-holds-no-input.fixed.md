A `SimulationRunner` no longer holds an input through its march. An array recorded by its
digest is read in place rather than copied, and neither it, the launch converted from it
nor the derived simulation outlives the load, so a delay scan that forms each input in the
call holds it only until the backend has it (#447).
