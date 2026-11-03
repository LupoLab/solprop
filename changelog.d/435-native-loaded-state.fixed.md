The native backend no longer keeps the last host state it loaded alive after
`set_initial`: a temporary, such as an input noise sum, a resumed plane or a delay
scan's input, was one host field volume held until the next load, outside
`memory_ledger()` and `estimate_memory` (#435).
