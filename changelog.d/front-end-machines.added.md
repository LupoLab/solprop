The desktop front end runs every simulation and scan as a process of its own, which outlives the window and is followed from its record, and runs them on other machines over SSH too: a
workstation, a Slurm cluster or a rented machine, picked on the Runs on line. Each run
keeps its record in its folder there, so it survives a closed window or a dropped network
and is found again on reconnecting; a Runs window lists every run on every machine, a
results file is analysed on its machine, and a machine can be set up with solprop from the
window. See [Running on other machines](docs/guide/front-end/machines.md).
