A translated problem no longer holds a full copy of a broadcast initial field (#5): a
`numpy.broadcast_to` placeholder, of any numeric dtype, costs only the block it repeats.
Every other initial field, noise realisation and batch parameter is now copied, so
changing the caller's array after translation can no longer change the problem, and the
caller's array is no longer made read-only; before, a contiguous complex128 field and a
noise realisation were shared with the caller.
