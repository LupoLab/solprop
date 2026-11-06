A run into `MemoryOutput`, the default handler, no longer computes the configuration hash
for a debug message that is switched off, which took seconds on a large grid.
