A `Pool` scan point whose worker is interrupted, by Ctrl-C (which a terminal sends to
every worker) or SIGTERM, is now a stopped point, given back so that a rerun computes
it, instead of being committed as `failed` with a `KeyboardInterrupt` traceback.
