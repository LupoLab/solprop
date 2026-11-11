Preserve the initiating error and retain a retry owner when acquiring host BLAS limits
and rolling them back both fail. A later acquisition on the same thread restores the
pending request before applying new limits, preventing a permanently orphaned width
or Accelerate mode ([#697](https://github.com/LupoLab/solprop/issues/697)).
