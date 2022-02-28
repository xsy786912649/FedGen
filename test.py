import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.params = [
                    torch.Tensor(256, 6).uniform_(-1./math.sqrt(6), 1./math.sqrt(6)).requires_grad_(),
                    torch.Tensor(256).zero_().requires_grad_(),

                    torch.Tensor(256, 256).uniform_(-1./math.sqrt(256), 1./math.sqrt(256)).requires_grad_(),
                    torch.Tensor(256).zero_().requires_grad_(),

                    torch.Tensor(128, 256).uniform_(-1./math.sqrt(256), 1./math.sqrt(256)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(1, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(1).zero_().requires_grad_(),
                ]

    def dense(self, x, params):
        x = F.linear(x, params[0], params[1])
        x = F.relu(x)

        x = F.linear(x, params[2], params[3])
        x = F.relu(x)

        x = F.linear(x, params[4], params[5])
        x = F.relu(x)

        x = F.linear(x, params[6], params[7])
        x = (F.sigmoid(x)*2.0 - 1.0)*(1.0/math.sqrt(3))
        return x

    def input_process(self, x):
        x_position,x_theta=x.split([2,1],dim=1)
        x_sin=torch.sin(x_theta)
        x_cos=torch.cos(x_theta)
        x_sin_2=torch.sin(2*x_theta)
        x_cos_2=torch.cos(2*x_theta)
        return torch.cat((x_position,x_sin,x_cos,x_sin_2,x_cos_2), 1)

    def forward(self, x, params):
        return self.dense(self.input_process(x), params)

def predict(model, pre_x):
    outputs=[]
    pre=torch.tensor(pre_x).float().to(device)
    with torch.no_grad():
        inputs = pre.to(device)
        output = model.dense(model.input_process(inputs),model.params) 
        outputs = output.cpu().numpy()
    outputs=np.array(outputs)
    return outputs


x=np.array([[1,2,3]])
print(x[0,:2])
