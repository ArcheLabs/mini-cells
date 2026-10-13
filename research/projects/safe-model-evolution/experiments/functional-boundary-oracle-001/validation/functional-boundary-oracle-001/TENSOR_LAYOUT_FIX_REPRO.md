Reproduction invariant from Kaggle seed 26090511:

`input_linear.weight = [32, 1024, 1024]`

`output_linear.weight = [32, 1024, 512]`

The formal runtime detector must infer expert intermediate width `512` from these shapes even when `model.config.intermediate_size == 1024`.
