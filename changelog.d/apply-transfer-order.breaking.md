`solprop.processing.compression.apply_transfer` takes the output first, like every other
processing function: `apply_transfer(output, transfer)` replaces
`apply_transfer(transfer, output)`. It and the compressors also take `z=` to select the
saved plane nearest a position.
