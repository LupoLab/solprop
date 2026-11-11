Share controllable host BLAS limits across Julia preparation and host-construction
lifetimes. Record actual widths and restore them only after the last owner releases
the library, including interrupted or abandoned Julia preparations.
