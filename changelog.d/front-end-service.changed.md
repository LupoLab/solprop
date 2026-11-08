The front end's service layer (its kernel process, run workers, live reductions and their
messages) moved from `solprop_gui.service` into solprop as `solprop.service`, so a machine
that only runs simulations for the desktop front end needs solprop alone, without Qt. It is
front-end support, not part of the user interface, and may change without notice.
