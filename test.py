import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F
import math
import random



x=np.array([[1,2,3]])
print(x[0,:2])


print((np.random.random()-0.5)*0.08)