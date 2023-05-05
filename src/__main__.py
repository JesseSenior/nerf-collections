import torch
import numpy as np

import config
import train

torch.set_default_tensor_type("torch.cuda.FloatTensor")
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


config.init()
train.train()
