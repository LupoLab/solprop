A recipe's `save_planes = 80` in an object route's `[build.settings]` is no longer read
as one plane at 80 m, with the run marching until something unrelated stopped it: it is
refused with "write `n_save = 80`", and every run now ends at its grid's length (#533).
