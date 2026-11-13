Fixed the Adams march dropping the attempt row of a rejected trial whose reduced
proposal fell below `dz_min`: the completed trial is now recorded, on both the NumPy
reference and the native backend, before the floor's `ConvergenceError` is raised; the
proposal, accepted state and call ledger are unchanged (#651).
