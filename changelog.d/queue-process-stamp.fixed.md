A Queue scan whose point owner exits while its process start is being read no longer fails with an unhandled `OSError`; the owner is treated as unidentified until the next check (#760).
