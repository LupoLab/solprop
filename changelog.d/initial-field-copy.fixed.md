A translated problem no longer holds a full copy of a broadcast or read-only initial
field (#5): a `numpy.broadcast_to` placeholder costs only the block it repeats, and a
read-only array is kept as it is. A writeable input is now always copied, so changing
it after translation cannot change the problem's initial state, which it could when the
array was already contiguous. `estimate_memory` and the memory admission count the
field the problem holds, as the resident group `initial_field` (#6).
