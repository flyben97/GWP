import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move dataloaders to device
def move_to_device(data):
    if isinstance(data, (list, tuple)):
        return [move_to_device(d) for d in data]
    return data.to(device)

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)