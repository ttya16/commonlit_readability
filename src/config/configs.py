from pathlib import Path

import torch

class Config:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_data_path = Path("../data")
    output_path = "./outputs"

    # training parameters
    num_folds = 5
    num_epochs = 2
    batch_size = 16
