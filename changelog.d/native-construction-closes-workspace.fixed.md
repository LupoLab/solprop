A native backend whose construction fails after its native workspace is built, by a
refusal, an error or an interrupt, now closes the workspace before raising, rather than
leaving it, and on a GPU build its device memory, to the garbage collector.
