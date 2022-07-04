import torch
import numpy as np
import random
from experiments import *


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed()
    # independent_2d_rotations()

    set_seed()
    # independent_3d_rotations()

    set_seed()
    barbel_experiment()