import os
from src.training.train import train
import datetime
from src.paths import ROOTDIR
from pathlib import Path as P

if __name__ == "__main__":
    hyperparams = {
        "batch_size": 16,
        "epochs": 300,
        "lr": 0.00001,
        "dropout_rate": 0.1,
        "weight_decay": 0.001,
        "end_to_end": True,
    }

    timestamp = datetime.datetime.now().strftime("%d_%H:%M:%S")
    run_dir = P.joinpath(P(ROOTDIR), "runs", timestamp)
    os.mkdir(run_dir)

    train(hyperparams, run_dir)
