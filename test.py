import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F
import math



x=np.array([[1,2,3]])
print(x[0,:2])


a=np.random.random((1,2))*np.array([[1,0.15]])+np.array([[0,0.85]])
b=np.array([[1,2,3],[2,3,4]])
print(a)
print(b)
d=b.reshape((1,-1))
print(d)
c=np.append(a,d,axis=1)
print(c)

