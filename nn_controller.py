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
        self.obs_n=1
        self.goal_n=1
        self.params = [
                    torch.Tensor(64, 6+self.goal_n*2+self.obs_n*2).uniform_(-1./math.sqrt(6), 1./math.sqrt(6)).requires_grad_(),
                    torch.Tensor(64).zero_().requires_grad_(),

                    torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)).requires_grad_(),
                    torch.Tensor(64).zero_().requires_grad_(),

                    torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)).requires_grad_(),
                    torch.Tensor(64).zero_().requires_grad_(),

                    torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)).requires_grad_(),
                    torch.Tensor(1).zero_().requires_grad_(),
                ]
        self.optimizer = torch.optim.Adam(self.params,lr=0.001,weight_decay=0.0000)
        #self.optimizer = torch.optim.SGD(self.params,lr=0.03,weight_decay=0.0000)

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
        x=torch.tensor(x).float().to(device)
        x = self.dense(self.input_process(x), params)
        u = (torch.sigmoid(x)*2.0 - 1.0)*(1.0/math.sqrt(3))
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
        print('save'+'./pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl')
        return torch.save(self, './pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl') 

    def save_model_global(self,iteration):
        return torch.save(self, './pkl/robot_global'+'_'+'iteration'+str(iteration)+'.pkl') 

    def compute_gradient(self,inputs,deltas):
        #deltas.shape==(n,1)
        #inputs.shape==(n,12)
        u=self.forward(inputs, self.params)
        u_multi_deltas=torch.tensor(deltas)*u
        u_mean=torch.mean(u_multi_deltas)
        grads = torch.autograd.grad(u_mean, self.params, create_graph=True, retain_graph=True)
        return grads

    def update(self,grd,lr):
        params1 = [(self.params[i] - lr*grd[i]) for i in range(len(self.params))]
        self.params=params1
        return

    def update1(self,inputs,deltas,lr=0.001):
        self.optimizer.zero_grad()
        u=self.forward(inputs, self.params)
        u_multi_deltas=torch.tensor(deltas)*u
        u_mean=torch.mean(u_multi_deltas)
        u_mean.backward()
        self.optimizer.step()
        return
        
def load_model(id, iteration):
    print("load"+'./pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl')
    return torch.load('./pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl')

def load_model_global(iteration):
    return torch.load('./pkl/robot_global'+'_'+'iteration'+str(iteration)+'.pkl')

def tensor_norm(tensors):
    squra=0
    for tensor in tensors:
        b=torch.square(tensor)
        squra=squra+torch.sum(b).detach().numpy()
    return math.sqrt(squra)

if __name__=='__main__':
    #a= nn_controller()
    a = torch.randn(2, 2)
    b=[a,a]
    print(b)
    print(tensor_norm(b))