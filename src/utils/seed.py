import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU is the supported execution environment.  Warn rather than abort if a
    # future optional accelerator path uses a nondeterministic operation.
    torch.use_deterministic_algorithms(True, warn_only=True)
