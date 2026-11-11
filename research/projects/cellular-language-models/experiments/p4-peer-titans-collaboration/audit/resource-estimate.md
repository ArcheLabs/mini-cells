# Resource estimate

For dimension 32 and hidden width 64, a Cell holds 4,256 Fast Weight scalars plus 4,256 Momentum scalars, or 34,048 bytes at FP32. The largest input tensor has 2,048 scalars. One frozen control and 15 independent development episodes at 128 updates each require 2,048 inner updates total. No GPU allocation or outer optimizer is required. Measured timing and checkpoint bytes are reported separately.
