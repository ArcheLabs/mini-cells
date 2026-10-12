"""Compatibility entrypoint for published Experiment 001 commands."""
from experiments.common.train import (checkpoint, evaluate, main, make_batch, prepare_data, train_one, write_json)

if __name__ == "__main__":
    main()
