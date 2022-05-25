import pybullet as pybullet
import numpy as np
import time
from utils_simulation import *
from nn_stochastic_controller import nn_stochastic_controller
from nn_stochastic_controller import *


def compute_gradient(n_E, params, husky, sphere,controller,random_seed=None,popula=30):
    np.random.seed(random_seed)
    index = np.random.randint(0,1000000,size=popula)
    index1= np.random.randint(0,10000)
    Q_positive=0
    Q_negtive=0
    grad1_list=[]
    grad2_list=[]
    #q=0.0
    for i in range(popula):
        seed_i=index[i]
        controller.set_random_para(seed_i)
        Q_positive,_=environment_costs(n_E, controller, params, husky, sphere, False, index1, 1)
        Q_negtive,_=environment_costs(n_E, controller, params, husky, sphere, False, index1, 2)
        grad1,grad2= controller.compute_gradient(Q_positive,Q_negtive)
        grad1_list.append(grad1)
        grad2_list.append(grad2)
        #q=q+Q_positive+Q_negtive
    
    grad1_fin=averge_gra(grad1_list)
    grad2_fin=averge_gra(grad2_list)
    norm=tensor_norm_more(grad1_fin)/2
    print("norm1: " +str(norm))
    #print("norm2: " +str(tensor_norm_more(grad2_fin)/2))
    '''
    print("-------------------------------------")
    print(grad1_fin)
    print(grad2_fin)
    print(controller.params)
    print(controller.params2)
    print("-------------------------------------")
    '''
    controller.update(grad1_fin,grad2_fin)
    #print(q/2.0/popula)
    return grad1_fin, norm 

if __name__=='__main__':
    a=1


