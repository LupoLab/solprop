The TDSE provider's worker process no longer receives the running script's own
directory on its import path, so a module there named like a standard-library one
(a script's `random.py`, say) is not imported in the worker in its place.
