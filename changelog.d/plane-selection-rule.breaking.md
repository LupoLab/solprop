One plane-selection rule across `solprop.processing`: a function returning one plane
(`get_spectrogram`, the readers of `processing.beams`, the reconstructions of
`processing.modal`, the compressors, `apply_transfer`, `fluence_map`, `field_plane`) takes
`plane=-1, z=None`, and naming a plane both by index and by position raises
`ProcessingError` everywhere (it raised `OutputError` in the field and time accessors, and
the beam readers let `z` win silently). The readers of every plane keep `plane=None`
meaning every plane. The rule is stated in the processing guide.
