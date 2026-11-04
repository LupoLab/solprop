`Simulation.with_input` and `SimulationRunner.run` take `input_identity=`, a string the
run record holds with the array's dtype and shape (`solprop.output.StatedInput`) in place
of the SHA-256 of an array input's bytes. Without it, every hash is unchanged. The
identity is not checked: two different arrays stated alike have one configuration hash,
so a wrong identity passes a resume and a scan's cache. A derived run no longer digests
its parent's input to check its `configuration=` keys either (#449).
