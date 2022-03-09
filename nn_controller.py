from re import A
import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class nn_controller(torch.nn.Module):
    def __init__(self):
        super(nn_controller, self).__init__()
        self.obs_n=10
        self.goal_n=1
        self.params = [
                    torch.Tensor(128, 6+self.goal_n*2+self.obs_n*2).uniform_(-1./math.sqrt(6), 1./math.sqrt(6)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(128, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(64, 128).uniform_(-1./math.sqrt(128), 1./math.sqrt(128)).requires_grad_(),
                    torch.Tensor(128).zero_().requires_grad_(),

                    torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)).requires_grad_(),
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
        
        return x

    def input_process(self, x):
        x_position,x_theta,x_obs=x.split([2,1,self.goal_n*2+self.obs_n*2],dim=1)
        x_sin=torch.sin(x_theta)
        x_cos=torch.cos(x_theta)
        x_sin_2=torch.sin(2*x_theta)
        x_cos_2=torch.cos(2*x_theta)
        return torch.cat((x_position,x_sin,x_cos,x_sin_2,x_cos_2,x_obs), 1)

    def forward(self, x, params):
        x = self.dense(self.input_process(x), params)
        u = (F.sigmoid(x)*2.0 - 1.0)*(1.0/math.sqrt(3))
        return u

    def predict(self, pre_x):
        outputs=[]
        pre=torch.tensor(pre_x).float().to(device)
        with torch.no_grad():
            inputs = pre.to(device)
            output = self.forward(inputs, self.params) 
            outputs = output.cpu().numpy()
        outputs=np.array(outputs)
        return float(outputs)

    def save_model(self, id, iteration):
        return torch.save(self, './pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl') 

    def save_model_global(self,iteration):
        return torch.save(self, './pkl/robot_global'+'_'+'iteration'+str(iteration)+'.pkl') 


def load_model(id, iteration):
    return torch.load('./pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl')

def load_model_global(iteration):
    return torch.load('./pkl/robot_global'+'_'+'iteration'+str(iteration)+'.pkl')

if __name__=='__main__':
    a= nn_controller()
    b=a
    c=a
    b.obs_n=12
    print(a.obs_n)