import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setup

    torch.backends.cudnn.benchmark = True  # Allow for potential optimizations

    # Note: Some operations may still have nondeterministic behavior.
    print(f"Seed set to: {seed}.")
