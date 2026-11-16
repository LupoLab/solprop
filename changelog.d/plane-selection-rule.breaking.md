One plane-selection rule across `solprop.processing`: every function takes `plane=None,
z=None`, and naming a plane both by index and by position raises `ProcessingError`
everywhere (it raised `OutputError` in the field and time accessors, and the beam readers
let `z` win silently). With neither, a function returning one plane
(`get_spectrogram`, the readers of `processing.beams`, the reconstructions of
`processing.modal`, the compressors, `apply_transfer`, `fluence_map`, `field_plane`)
reads the last plane, and a reader of every plane returns them all. The rule is stated in
the processing guide.
