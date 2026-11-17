Fixed the execution record of a run on Windows storing `None` for the host's
`memory_bytes` and the run's `peak_rss_bytes`: they now hold the installed physical
memory and the process's peak working set, read as the scan executors already did.
