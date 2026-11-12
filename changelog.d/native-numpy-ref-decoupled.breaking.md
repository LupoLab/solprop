The general modal route's and the coupled flow's admission is part of the problem's own
validation (`solprop.translate.admission.check_general_route`), raised once as a
`TranslationError` naming solprop rather than as a `BackendError` from each backend;
`solprop.backends.numpy_ref.admission` and `refuse_general_leaves` are removed. The
general route's derivations moved to `solprop.translate.general` (`node_kerr_tensor`,
`node_kerr_form`, `node_delayed_maps`) and its node chunk to `solprop.backends.node_chunk`
(`general_node_chunk`, `general_planned_chunk`); the quadrature alarm is
`solprop.translate.modal.QUADRATURE_ALARM`, logged on the `solprop.translate.modal`
logger; the Adams call ledger is `solprop.backends.ledger.CallLedger`, and the
reference's Adams module is `solprop.backends.numpy_ref.multistep`.
`CompiledResponses.response_order` is derived from the compiled leaves and is no longer a
constructor argument. `solprop_native` no longer exports the `*_FIELDS` seam tuples, and
`check_build` no longer looks seam members up.
