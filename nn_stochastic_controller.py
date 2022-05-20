import numpy as np
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class nn_stochastic_controller(torch.nn.Module):
    def __init__(self,numRays):
        super(nn_stochastic_controller, self).__init__()
        self.r = 0.1
        self.v0 = 2.5
        self.u_diff_max = 0.5*(self.v0/self.r) 
        self.numRays=numRays
        """
        self.params = [
                    torch.Tensor(20, self.numRays).uniform_(-0.1/5,0.1/5),
                    torch.Tensor(20).zero_(),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    torch.Tensor(1, 20).uniform_(-0.1/5,0.1/5),
                    torch.Tensor(1).zero_(),
                ]

        self.params2 = [
                    torch.Tensor(20, self.numRays).uniform_(0.08/5, 0.1/5),
                    torch.Tensor(20).uniform_(0.08/5, 0.1/5),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    torch.Tensor(1, 20).uniform_(0.08/5, 0.1/5),
                    torch.Tensor(1).uniform_(0.08/5, 0.1/5),
                ]
        """
        self.params = [
                    torch.Tensor(1, self.numRays).uniform_(-0.1,0.1),
                    torch.Tensor(1).zero_(),
                ]

        self.params2 = [
                    torch.Tensor(1, self.numRays).uniform_(0.08, 0.1),
                    torch.Tensor(1).uniform_(0.08, 0.1),
                ]
        
        self.random_parameter_posi=[
                ]

        self.random_parameter_neg=[
                ]

        self.normal_posi=[
                ]

        self.t=0

        self.m= [   torch.Tensor(1, self.numRays).zero_(),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    torch.Tensor(1).zero_(),]

        self.u=[    torch.Tensor(1, self.numRays).zero_(),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    torch.Tensor(1).zero_(),
                ]


        self.m2=[    torch.Tensor(1, self.numRays).zero_(),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    torch.Tensor(1).zero_(),
                ]

        self.u2=[    torch.Tensor(1, self.numRays).zero_(),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(64, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    #torch.Tensor(64).zero_(),

                    #torch.Tensor(1, 64).uniform_(-1./math.sqrt(64), 1./math.sqrt(64)),
                    torch.Tensor(1).zero_(),
                ]
                

    def dense(self, x, params):
        x = F.linear(x, params[0], params[1])
        #x = F.relu(x)

        #x = F.linear(x, params[2], params[3])
        #x = F.relu(x)

        #x = F.linear(x, params[2], params[3])

        return x

    def input_process(self, x):
        return 1.0/(x+0.1)

    def set_random_para(self,seed):
        torch.manual_seed(seed)
        param_sample = [(torch.normal(mean=torch.zeros_like(self.params[i]))) for i in range(len(self.params))] 
        self.normal_posi= param_sample
        self.random_parameter_posi=[self.normal_posi[i]*self.params2[i] + self.params[i] for i in range(len(self.params))] 
        self.random_parameter_neg=[-self.normal_posi[i]*self.params2[i] + self.params[i] for i in range(len(self.params))] 
        '''
        print("---------------------")
        print(self.params[0])
        print(self.params2[0])
        print("---------------------")
        '''
        return

    def forward(self, x , params):
        x=torch.tensor(x).float().to(device)
        x = self.dense(self.input_process(x), params)
        u1 = torch.clamp(x*self.u_diff_max,-self.u_diff_max,self.u_diff_max)
        if torch.any(torch.isnan(u1)):
            print("gg")
            input()
        return u1

    def predict(self, pre_x, mode):
        outputs1=[]
        pre=torch.tensor(pre_x).float().to(device)
        with torch.no_grad():
            inputs = pre.to(device)
            if mode==1:
                output1 = self.forward(inputs,self.random_parameter_posi)
            elif mode==2:
                output1 = self.forward(inputs,self.random_parameter_neg)
            elif mode==0:
                output1 = self.forward(inputs,self.params)
            outputs1 = output1.cpu().numpy()
        outputs1=float(np.array(outputs1))
        return outputs1

    def save_model(self, id, iteration):
        print('save'+'./pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl')
        return torch.save(self, './pkl/robot'+str(id)+'_'+'iteration'+str(iteration)+'.pkl') 

    def save_model_global(self,iteration):
        return torch.save(self, './pkl/robot_global'+'_'+'iteration'+str(iteration)+'.pkl') 

    def compute_gradient(self,Q_positive, Q_negtive):
        #Q==float
        grad1= [self.normal_posi[i]/self.params2[i]*(Q_positive-Q_negtive) for i in range(len(self.params))]
        grad2= [torch.clamp((self.normal_posi[i]*self.normal_posi[i]-1.0)/self.params2[i]*(Q_positive+Q_negtive),-1,1) for i in range(len(self.params2))]
        #print(grad1[0])
        #print(grad2[0])
        return grad1,grad2

    def update(self,grd1,grd2,lr=0.01,lr2=0.001):
        params_new = [(self.params[i] - lr*grd1[i]) for i in range(len(self.params))]
        self.params=params_new
        params2_new = [(self.params2[i] - lr2*grd2[i]) for i in range(len(self.params2))]
        self.params2=params2_new
        return

    def adam(self,grd1,grd2):
        self.t=self.t+1

        beta_1=0.9
        beta_2=0.999
        ep=0.000001
        lr= 0.001
        lr2= 0.0003

        self.m=[a_i*beta_1+(1-beta_1)*b_i for a_i, b_i in zip(self.m, grd1) ]

        self.u=[a_i*beta_2+(1-beta_2)*(b_i**2) for a_i, b_i in zip(self.u, grd1) ]
        
        hm=[a_i/(1-(beta_1**self.t)) for a_i in self.m]
        hu=[a_i/(1-(beta_2**self.t)) for a_i in self.u]
        
        dws_new=[lr * a_i /(ep + torch.sqrt(b_i)) for a_i, b_i in zip(hm, hu)]

        params = [a_i - b_i for a_i, b_i in zip(self.params, dws_new)]

        self.params=params

        self.m2=[a_i*beta_1+(1-beta_1)*b_i for a_i, b_i in zip(self.m2, grd2) ]
        self.u2=[a_i*beta_2+(1-beta_2)*(b_i**2) for a_i, b_i in zip(self.u2, grd2) ]

        hm=[a_i/(1-(beta_1**self.t)) for a_i in self.m2]
        hu=[a_i/(1-(beta_2**self.t)) for a_i in self.u2]

        dws_new=[lr2 * a_i /(ep + torch.sqrt(b_i)) for a_i, b_i in zip(hm, hu)]

        params2 = [a_i - b_i for a_i, b_i in zip(self.params2, dws_new)]
        self.params2=params2
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

def tensor_norm_more(tensor_list):
    squra=0
    for tensors in tensor_list:
        for tensor in tensors:
            b=torch.square(tensor)
            squra=squra+torch.sum(b).detach().numpy()
    return math.sqrt(squra)

def averge_gra(tensor_list):
    grad_final=[]
    for j in range(len(tensor_list[0])):
        grad=torch.zeros_like(tensor_list[0][j])
        for i in range(len(tensor_list)):
            grad+=tensor_list[i][j]
        grad=grad/float(len(tensor_list))
        grad_final.append(grad)
    return grad_final

if __name__=='__main__':
    #a= nn_stochastic_controller()
    a = torch.randn(2, 2)
    b=[a,a]
    print(b)
    print(tensor_norm(b))