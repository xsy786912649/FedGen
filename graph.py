import pybullet as pybullet
import numpy as np
import time
from utils_simulation import *
from nn_stochastic_controller import nn_stochastic_controller
from gradient import *
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt

from math import sin,cos,sqrt,exp
import matplotlib

import pickle


start=time.time()
test_num=200
#setup configuration
#random_seed=36
#environment parameter
n_obs=1
n_E=10

#FedGen parameter
gamma=0.01
ell=0.06 #0.03 #Lipschitz constant

q=np.sqrt(2*np.log(2/gamma)/n_E)*ell
s=np.sqrt(np.log(2/gamma)/n_E/2)/10.0
zeta=1
K=200
print(q,s)

robo_network=[]
n_robot=8

file_global='./pkl/global_minimum'+str(n_obs)+'obs'+'.pkl'
Process=[]
lock = mp.Lock()

for id in range(n_robot):
    file='./pkl/robot'+str(id)+'_'+str(n_obs)+'obs'+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    f.close()
    
    Y=robo_data['testing_Y']
    tt=robo_data['t_theta']
    #tt=list(range(K))
    plt.plot(tt,Y,label='Robot '+str(id))
    switch=robo_data['Switch']
    y_swith=[]
    t_swith=[]
    for t in tt:
        if t in switch:
            y_swith.append(Y[t])
            t_swith.append(t)
    plt.plot(t_swith,y_swith,'o',color='r')
    plt.legend()
plt.show()

for id in range(n_robot):
    file='./pkl/robot'+str(id)+'_'+str(n_obs)+'obs'+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    f.close()
    COL=np.array(robo_data['testing_col'])
    tt=robo_data['t_theta']
    #tt=list(range(K))
    plt.plot(tt,COL,label='Robot '+str(id))
    switch=robo_data['Switch']
    y_swith=[]
    t_swith=[]
    for t in tt:
        if t in switch:
            y_swith.append(COL[t])
            t_swith.append(t)
    plt.plot(t_swith,y_swith,'o',color='r')
    plt.legend()
plt.show()


for id in range(n_robot):
    file='./pkl/robot'+str(id)+'_'+str(n_obs)+'obs'+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    f.close()
    RUN=np.array(robo_data['testing_runningtime'])
    tt=robo_data['t_theta']
    plt.plot(tt,RUN,label='Robot '+str(id))
    switch=robo_data['Switch']
    y_swith=[]
    t_swith=[]
    for t in tt:
        if t in switch:
            y_swith.append(RUN[t])
            t_swith.append(t)
    plt.plot(t_swith,y_swith,'o',color='r')
    plt.legend()
plt.show()