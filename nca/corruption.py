import random
import torch

def corrupt(x):
    """
    Introduces noise or missing values and returns mask.
    """
    x = x.clone()
    mask = torch.ones_like(x)

    for i in range(x.shape[0]):
        if random.random() < 0.4:
            idx = random.randint(0, 2)

            if random.random() < 0.5:
                x[i, idx] = x[i, idx] * random.randint(5, 15)
            else:
                x[i, idx] = 0.0

            mask[i, idx] = 0

    return x, mask