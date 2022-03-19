#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:40:01 2020

@author: zqy5086@AD.PSU.EDU
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:55:49 2019

@author: zqy5086@AD.PSU.EDU
"""

import numpy as np
import matplotlib.pyplot as plt

from math import sin,cos,sqrt,exp
import matplotlib

from scipy import spatial
import pickle
from sklearn.neighbors import BallTree as KDTree
from wind import wind
import time
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# import copy
import time
import random
np.random.seed(11)  
import multiprocessing as mp
import nn_controller
from nn_controller import load_model
from nn_controller import load_model_global
from nn_controller import tensor_norm
from nn_controller import nn_controller

import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms
from torch.nn import functional as F



def test_robot(robo,T,num,iteration_N,lock):
    file='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    tt=robo.testing_all(T,num,iteration_N)
    robo_data['testing_Y']=robo.testing_Y
    robo_data['testing_col']=robo.testing_col
    robo_data['t_theta']=tt
    
    if robo_data['local_converge']:
        controller=load_model(robo.id,robo_data['local_converge_t'+1])
        y,col=robo.testing_one(T,num,controller)
        robo_data['local_converge_test_y']=y
        robo_data['local_converge_test_col']=col
    
    
    fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl','wb')
    pickle.dump(robo_data,fw)
    fw.close()
    plt.plot(robo.testing_Y)
    plt.title('Robot '+str(robo.id))

class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,dynamics,goal_size,obs_size,n_E,n_init,n_obs,n_theta,q,s,robo_id,zeta,r,controller,alpha): 
        self.init=None
        self.x=None
        
        self.X=None
        self.goal_size=goal_size
        self.obs_size=obs_size
        self.n_obs=n_obs
        self.obs=None
        self.goal=None
        self.X_=None
        #self.theta=None
        self.dynamics=dynamics
        self.u=None
        self.n_E=n_E
        self.n_init=n_init
        self.Y=[]
        self.n_theta=n_theta
        self.q=q
        self.s=s
        self.id=robo_id
        self.zeta=zeta
        self.y=None
        self.z=None
        self.z_norm=None
        self.r=r
        self.converge=False
        self.controller=controller
        self.testing_Y=None
        self.testing_y=None
        self.testing_col=None
        self.alpha=alpha
        self.local_converge=False
        #self.local_converge_theta=None
        
        
        file='./pkl/robot'+str(self.id)+'_'+str(n_obs)+'obs'+'.pkl'
        f=open(file,'wb')
        robot_data=dict()
        robot_data['Y']=[]
        robot_data['y']=None
        robot_data['nE']=n_E
        robot_data['alpha']=alpha
        robot_data['ninit']=n_init
        robot_data['n_obs']=n_obs
        robot_data['s']=s
        robot_data['q']=q
        robot_data['converge']=False
        self.controller.save_model(self.id,0)
        robot_data['z_norm']=None
        robot_data['n_obs']=n_obs
        robot_data['r']=r
        robot_data['Switch']=[]
        robot_data['goal_size']=goal_size
        robot_data['obs_size']=obs_size
        robot_data['local_converge']=False
        #robot_data['local_converge_theta']=None
        robot_data['testing_Y']=None
        robot_data['testing_col']=None
        robot_data['local_converge_test_y']=None
        robot_data['local_converge_test_col']=None
        robot_data['zeta']=1
        pickle.dump(robot_data,f)
    
    def testing_one(self, T,num, controller):
        y=0
        col=0
        self.controller=controller
#        np.random.seed(10000)
        for E in range(num):
            goal, init,obs=env_config(self.n_obs,random_seed=E) 
            self.run_setup(goal,init,obs)
            eta=2
            for t in range(T):
                
                # print(self.goal)
                Collision,Goal,current_x=self.run(E,self.controller)
                if Collision:
                    eta=1+self.reward(current_x,t,False,True)
                    col=col+1
                    break
                if Goal:
                    eta=self.reward(current_x,t,True,False)
                    # print(eta)
                    break
                    
            if not Collision and not Goal:
                eta=self.reward(current_x,t,False,True)
                col=col+1
            y=y+eta
            # self.plot_config()    
        y=y/num
#        print(y)
        self.testing_y=y
        
        return y,col

    def testing_all(self,T,num,iteration_N):
        Y=[]
        COL=[]
        cnt=1
        tt=[]
        for iteration in range(iteration_N):
            controller=load_model(self.id,iteration)
            t=iteration
            y,col=self.testing_one(T,num,controller)
            Y.append(y)
            COL.append(col/num)
            tt.append(t)
            print('Testing...robot '+str(self.id)+' '+str(cnt)+'/'+str(iteration_N)+' Done!')
            cnt=cnt+1
        print('Robot '+str(self.id)+' completes testing !!!!')
        self.testing_Y=Y
        self.testing_col=COL
        
        return tt

    def kruzkov_transform(self,t):
        return 1-exp(-self.alpha*(t+1))

    def kruzkov_transform1(self,t):
        return t/200.0

    def reward(self,stop_state,t,is_goal=True, timeout=False): 
        reward0=1 
        if is_goal==True: 
            reward0=self.kruzkov_transform(t)-1
        elif timeout==True: 
            reward0= np.linalg.norm(stop_state[0,:2]-self.goal)*np.linalg.norm(stop_state[0,:2]-self.goal)/2.0 
        return reward0 
        
    def run_setup(self,goal,init,obs):
        self.init=init
        self.x=init
        self.goal=goal
        self.obs=obs

        self.X_=np.vstack((self.goal,self.obs))
        self.X=init
        
    def plot_config(self):
        plt.figure()
        plt.scatter(self.goal[0,0],self.goal[0,1])
        plt.scatter(self.init[0,0],self.init[0,1],marker='^',s=150,label='$x_{init}$')
        plt.scatter(self.obs[:,0],self.obs[:,1],marker='o',color='red',label='obstacle')
         
        circle8 = plt.Rectangle((self.goal[0,0]-self.goal_size, self.goal[0,1]-self.goal_size),2*self.goal_size,2*self.goal_size,fc='green',label='goal')#.Circle((goal[0,0], goal[0,1]), goal_size, color='green',label='goal')
        plt.gca().add_patch(circle8)
        for i in range(len(self.obs)):
            circle8 = plt.Circle((self.obs[i,0],self.obs[i,1]),self.obs_size, color='red')
            plt.gca().add_patch(circle8)
            if i==0:
                plt.legend()
        plt.plot(self.X[:,0],self.X[:,1])

    def in_scope(self,current_x):
        if current_x[0,0]>1 or current_x[0,0]<0 or current_x[0,1]>1 or current_x[0,1]<0:
            return False
        else:
            return True

    def run(self,E,controller):
        Collision=False
        Goal=False
#        print(self.x)
        if np.linalg.norm(self.x[0,:2]-self.goal,ord=1)<self.goal_size:
            Goal=True
        elif any(np.linalg.norm(self.x[0,:2]-self.obs,axis=1)<self.obs_size):
            Collision=True
        else:
            obs_flatten=self.obs.reshape((1,-1)) 

            input=np.hstack((self.x,self.goal,obs_flatten))
            self.u=controller.predict(input)

            self.x=self.dynamics(self.x,self.u,E)  

            self.X=np.vstack((self.X,self.x))

        return Collision, Goal, self.x

    def run_for_gradient(self, x, E, controller,t,T,td):
        Collision=False
        Goal=False
        eta=2
        for i in range(T-t):
            if np.linalg.norm(x[0,:2]-self.goal,ord=1)<self.goal_size:
                Goal=True
                break
            elif any(np.linalg.norm(x[0,:2]-self.obs,axis=1)<self.obs_size):
                Collision=True
                break
            else:
                obs_flatten=self.obs.reshape((1,-1)) 
                input=np.hstack((x,self.goal,obs_flatten))
                if i==0:
                    u=controller.predict(input)+td
                else:
                    u=controller.predict(input)
                x=self.dynamics(x,u,E)  
        Time_t=t+i
        if Goal:
            eta=self.reward(x,Time_t,is_goal=True,timeout=False)
        if Collision:
            eta=1+self.reward(x,Time_t,False,True)
        if not Collision and not Goal:
            eta=self.reward(x,Time_t,is_goal=False,timeout=True)
        return eta 

    def local_update(self,T,random_seed,Z,i,lock,test_seed):
        file='./pkl/robot'+str(self.id)+'_'+str(self.n_obs)+'obs'+'.pkl'

        lock.acquire()
        fr=open(file,'rb')
        robot_data=pickle.load(fr)
        fr.close()
        lock.release()

        self.converge=robot_data['converge']
        self.controller=load_model(self.id,i)
        if not self.converge:
            self.local_converge=robot_data['local_converge']
            #self.local_converge_theta=robot_data['local_converge_theta']
            self.collect_y(T,random_seed,test_seed,self.controller)
            robot_data['y']=self.y
            lock.acquire()
            print(str(self.id)+' local updating y = ', end='')
            print(self.y)
            lock.release()

            #if self.y<1.1: 
            print(str(self.id)+' local updating z ... ')
            self.z=self.collect_z(T,random_seed,Z,self.controller)
            robot_data['z_norm']=tensor_norm(self.z)
            print(self.id, 'z norm = ',robot_data['z_norm'], 'Theshold = ',2*np.sqrt(self.n_theta)*self.q)
            robot_data['Y'].append(self.y)
            if robot_data['z_norm']>= 2*np.sqrt(self.n_theta)*self.q: 
                #self.controller.update(self.z, self.r)
                self.Y.append(self.y)
                self.controller.save_model(self.id, i+1)
                print('theta updated_one_time')
            else:
                self.converge=True
                robot_data['converge']=True
                if not self.local_converge:
                    self.local_converge=True
                    #self.local_converge_theta=self.theta_
                    #robot_data['local_converge_theta']=self.theta_
                    self.controller.save_model(self.id, i+1)
                    robot_data['local_converge']=True
                    robot_data['local_converge_t']=i

            lock.acquire()
            fw=open(file,'wb')                
            pickle.dump(robot_data,fw)
            fw.close()
            lock.release()
            print(self.id, 'Done local update!  ', "convergence: ", str(self.converge))
        else:
            self.controller.save_model(self.id, i+1)
        return
                
    def collect_y(self,T,random_seed,test_seed,controller):
        y=0
        np.random.seed(random_seed)
        goal_number=0
        for E in range(self.n_E):
            goal, init, obs=env_config(self.n_obs,random_seed=test_seed+100*E)
            for j in range(self.n_init):
                _, init,_=env_config(self.n_obs,random_seed=j+test_seed+100*E)

                self.run_setup(goal,init,obs)
                eta=2

                current_x=self.x
                for t in range(T):                   
                    # print(self.goal)                  
                    Collision,Goal,current_x=self.run(E+random_seed*self.n_E,controller)
                    if Collision:
                        eta=1+self.reward(current_x,t,False,True)
                        break
                    if Goal:   
                        goal_number=goal_number+1
                        eta=self.reward(current_x,t,True,False)
                        break                      
                if not Collision and not Goal:
                    eta=self.reward(current_x,t,False,True)
                y=y+eta

                #print(eta)
                #print(current_x)
        print("goal: "+str(goal_number*1.0/self.n_E/self.n_init))
        y=y/self.n_E/self.n_init
        self.y=y
        return y

    def collect_z(self,T,random_seed,Z,controller):
        np.random.seed(random_seed)
        n_E=self.n_E
        n_init=self.n_init
        inputs=0
        deltas=0

        for E in range(n_E):
            goal, init, obs=env_config(self.n_obs,random_seed=None)
            for j in range(n_init):
                _, init,_=env_config(self.n_obs,random_seed=None)

                td_list=[]
                self.run_setup(goal,init,obs)
                eta=2
                current_x=self.x
                obs_flatten=self.obs.reshape((1,-1)) 

                time_list=np.append([0],np.random.randint(0, high=T, size=10))
                eta_list=[]
                input_list=[]

                for t in range(T):
                    if (t in time_list) and (self.in_scope(current_x)):
                        td=(np.random.random()-0.5)*0.04
                        td_list.append([2*td])
                        reward_t=self.run_for_gradient(current_x,E+random_seed*self.n_E,controller,t,T,td)
                        reward_t1=self.run_for_gradient(current_x,E+random_seed*self.n_E,controller,t,T,-td)
                        eta_list.append([reward_t-reward_t1])
                        input_list.append(np.hstack((current_x,self.goal,obs_flatten))[0])

                    Collision,Goal,current_x=self.run(E+random_seed*self.n_E,controller)
                    if Collision:
                        eta=1+self.reward(current_x,t,False,True)
                        break
                    if Goal:
                        eta=self.reward(current_x,t,True,False)
                        break
                if not Collision and not Goal:
                    eta=self.reward(current_x,t,False,True)
                
                delta_list=np.array(eta_list)/np.array(td_list)
                input_array=np.array(input_list)
                delta_array=np.array(delta_list)
                #print(input_array.shape)

                if isinstance(inputs,int):
                    inputs=input_array
                    deltas=delta_array
                else:
                    inputs=np.vstack((inputs, input_array))
                    deltas=np.vstack((deltas, delta_array))
        #print(inputs.shape)
        self.z= self.controller.compute_gradient(inputs,deltas)
        self.controller.update1(inputs,deltas)
        return self.z
        
#Environment configuration
def env_config(n_obs,random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    goal=np.random.random((1,2))*np.array([[1,0.15]])+np.array([[0,0.85]])
    #init=np.random.random((1,3))*np.array([[1,0.15,np.pi]]) # 2*np.pi
    init=np.random.random((1,3))*np.array([[1,1,2*np.pi]])
    obs=np.random.random((n_obs,2))*np.array([[1,0.5]])+np.array([[0,0.25]])

    goal=np.array([[0.5,0.9]])
    obs=np.array([[0.3,0.5]])
    return goal, init, obs

def single_integrator(x,u):
    time_interval=0.01
    return x+np.array([[1,1]])*u*time_interval

def dubin_car(x,u,E):
    time_interval=0.01
#    np.random.seed(E)
#    d=np.random.uniform(-0.2,0.2)
#    d=wind(x[0,0],x[0,1],E) #wind disturbance
    x_=np.zeros(x.shape)
    x_[0,0]=np.cos(x[0,2])#+d
    x_[0,1]=np.sin(x[0,2])
    x_[0,2]=u/0.03     #u=[-1/np.sqrt(3),1/np.sqrt(3)] (2*sigmoid(u)-1)/np.sqrt(3.0) -30~30

    x_go=x+time_interval*x_
    if x_go[0,2]<0:
        x_go[0,2]=x_go[0,2]+2*np.pi
    elif x_go[0,2]>2*np.pi:
        x_go[0,2]=x_go[0,2]-2*np.pi
    return x_go
    
def cloud_update(file_global,robo_network,iteration):
    for robo in robo_network:
        try:
            f=open(file_global,'rb')
            global_min=pickle.load(f)
            y_j=global_min['y']
            s_j=global_min['s']            
    
            file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
            fb=open(file_r,'rb')
            robo_data=pickle.load(fb)
            robo.y=robo_data['y']
            
            print(robo.y,robo.s,y_j,s_j)

            if robo.y+robo.s<y_j+s_j:
                global_min=dict()
                global_min['y']=robo.y
                global_min['s']=robo.s
                global_min['id']=robo.id
                controller_current=load_model(robo.id,iteration)
                controller_current.save_model_global(iteration)
                # global_min=np.array([robo.y,robo.s, robo.id])
                f=open(file_global,'wb')
                pickle.dump(global_min,f)
                f.close()
            
        except FileNotFoundError:
            global_min=dict()
            file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
            fr=open(file_r,'rb')
            robo_data=pickle.load(fr)
            global_min['y']=robo_data['y']
            global_min['s']=robo.s
            global_min['id']=robo.id
            controller_current=load_model(robo.id,iteration)
            controller_current.save_model_global(iteration)
            f=open(file_global,'wb')
            pickle.dump(global_min,f)

def learner_fusion(robo_network,iteration):
    for robo in robo_network:
        f=open(file_global,'rb')
        global_min=pickle.load(f)
        y_j=global_min['y']
        s_j=global_min['s']
        id_j=global_min['id']
        
        file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
        fr=open(file_r,'rb')
        robo_data=pickle.load(fr)
        robo.y=robo_data['y']
        robo.z_norm=robo_data['z_norm']
        #robo.theta_=robo_data['theta']
        robo.controller=load_model(robo.id,iteration+1)
        robo.zeta=robo_data['zeta']
        robo.converge=robo_data['converge']
        if robo.id != id_j and y_j+s_j< robo.zeta and  y_j+s_j< robo.y-robo.s and robo.z_norm<2*np.sqrt(robo.n_theta)*robo.q:
            robo.controller=load_model_global(iteration)
            #robo.theta_=global_min['theta']
            robo.zeta=y_j
            robo.converge=False
            robo_data['converge']=robo.converge
            #robo.controller.save_model(robo.id,iteration+1)
            #robo_data['theta']=robo.theta_
            robo_data['zeta']=y_j
            robo_data['switch']=True
            #robo_data['Switch'].append((iteration,robo.theta_))
            print('Robot ', robo.id, 'Switched to ',id_j,'!!!!')
        
        print('Robot '+str(robo.id)+' theta updated!')
        
        robo.controller.save_model(robo.id,iteration+1)
        fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl','wb')
        pickle.dump(robo_data,fw)
        fw.close()

if __name__=='__main__':
    start=time.time()
    test_num=200
    #setup configuration
    random_seed=36
    
    #environment parameter
    n_obs=1
    obs_size=0.03
    goal_size=0.05
    n_E=20
    n_init=10
    
    #optimization parameter
    Z=2 #number of samples for zero-th order gradient computation
    r=0.001  #step size for gradient descent   
    alpha=0.005   #Kruzkov transform 
    T=200 #maximium exploration step
    
    np.random.seed(10)  
    
    #FedGen parameter
    gamma=0.01
    ell=0.0003 #0.003 #Lipschitz constant
    
    q=np.sqrt(2*np.log(2/gamma)/n_E/n_init)*ell
    s=np.sqrt(np.log(2/gamma)/n_E/n_init/2)
    zeta=1
    K=100
    print(q,s)
    robo_network=[]
    n_robot=1
    
    #Initialization
    for i in range(n_robot):
        #initialize theta
        controller=nn_controller()
        theta_size=4
        #initialize robot
        robo=robot(dubin_car,goal_size,obs_size,n_E,n_init,n_obs,theta_size,q,s,i,zeta,r,controller,alpha) #(dynamics,goal_size,obs_size,n_E,n_init,n_obs,n_theta,q,s,robo_id,zeta,r,controller,alpha):
        robo_network.append(robo)
    
    bias=0
    file_global='./pkl/global_minimum'+str(n_obs)+'obs'+'.pkl'
    Process=[]
    lock = mp.Lock()
    #FedGen algorithm
    for i in range(K):
        print('Iteration '+str(i))
        for robo in robo_network:
            p=mp.Process(target=robo.local_update, args=(T,i+robo.id*20,Z,i,lock,500+i*1000,))
            Process.append(p)
            p.start()
        for p in Process:
            p.join()        
        cloud_update(file_global,robo_network,i)
        learner_fusion(robo_network,i)
    
#testing ..........................................
    print('testing............')
    Robot_Y=[]
    Robot_COL=[]
    
    P_test=[]
    for robo in robo_network:
        p=mp.Process(target=test_robot,args=(robo,T,test_num,K+1,lock,))
        p.start()
        P_test.append(p)
    for p in P_test:
        p.join()
    print('Done testing.')
    
    for robo in robo_network:
        file='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
        f=open(file,'rb')
        robo_data=pickle.load(f)
        
        Y=robo_data['testing_Y']
        COL=robo_data['testing_col']
        tt=robo_data['t_theta']
        plt.plot(tt,Y,label='Robot '+str(robo.id))
#        if robo_data['local_converge']:
#            y_converge=robo_data['local_converge_test_y']    
#            plt.plot(np.linspace(0,len(Y),100),y_converge*np.ones((100,)),'--')
        plt.legend()
#        print(robo.Y)

