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
np.random.seed(10)  
import multiprocessing as mp




def test_robot(robo,T,num,lock):
    file='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    robo.Theta=robo_data['Theta']
#        for theta in robo_data['Theta']:
    tt=robo.testing_all(T,num)
    robo_data['testing_Y']=robo.testing_Y
    robo_data['testing_col']=robo.testing_col
    robo_data['t_theta']=tt
    if robo_data['local_converge']:
        theta=robo_data['local_converge_theta']
        y,col=robo.testing_one(T,num,theta)
        robo_data['local_converge_test_y']=y
        robo_data['local_converge_test_col']=col
    
#        Y.append(y)
#        COL.append(col/num)
    fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl','wb')
    pickle.dump(robo_data,fw)
    fw.close()
    plt.plot(robo.testing_Y)
    plt.title('Robot '+str(robo.id))

class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,dynamics,goal_size,obs_size,n_E,n_init,n_obs,n_theta,q,s,robo_id,zeta,r,theta,m_g,m_obs,alpha):
        self.init=None
        self.x=None
        
        self.X=None
        self.goal_size=goal_size
        self.obs_size=obs_size
        self.n_obs=n_obs
        self.obs=None
        self.goal=None
        self.X_=None
        self.theta=None
        self.dynamics=dynamics
        self.u=None
        self.n_E=n_E
        self.n_init=n_init
        self.Y=[]
        self.Theta=[(-1,theta)]
        self.n_theta=n_theta
        self.q=q
        self.s=s
        self.id=robo_id
        self.zeta=zeta
        self.y=None
        self.z=None
        self.r=r
        self.theta=theta
        self.m_g=m_g
        self.m_obs=m_obs
        self.converge=False
        self.theta=theta
        self.theta_=theta
        self.testing_Y=None
        self.testing_y=None
        self.testing_col=None
        self.alpha=alpha
        self.local_converge=False
        self.local_converge_theta=None
        
        
        file='./pkl/robot'+str(self.id)+'_'+str(n_obs)+'.pkl'
        f=open(file,'wb')
        robot_data=dict()
        robot_data['Y']=[]
        robot_data['y']=None
        robot_data['Theta']=[(-1,theta)]
        robot_data['nE']=n_E
        robot_data['alpha']=alpha
        robot_data['ninit']=n_init
        robot_data['n_obs']=n_obs
        robot_data['s']=s
        robot_data['q']=q
        robot_data['converge']=False
        robot_data['theta']=theta
        robot_data['theta_0']=theta
        robot_data['z']=None
        robot_data['n_obs']=n_obs
        robot_data['r']=r
        robot_data['Switch']=[]
        robot_data['goal_size']=goal_size
        robot_data['obs_size']=obs_size
        robot_data['local_converge']=False
        robot_data['local_converge_theta']=None
        robot_data['testing_Y']=None
        robot_data['testing_col']=None
        robot_data['local_converge_test_y']=None
        robot_data['local_converge_test_col']=None
        robot_data['zeta']=1
        pickle.dump(robot_data,f)
    
    def testing_one(self, T,num,theta):
        y=0
        col=0
        self.theta=theta
#        np.random.seed(10000)
        for E in range(num):
            goal, init,obs=env_config(self.n_obs,random_seed=E) 
            self.run_setup(goal,init,obs)
            eta=0
            for t in range(T):
                
                # print(self.goal)
                Collision,Goal,current_x=self.run(E)
                if Collision:
                    eta=1
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

    def testing_all(self,T,num):
        Y=[]
        COL=[]
        cnt=1
        tt=[]
        for i_theta in self.Theta:
            theta=i_theta[1]
            t=i_theta[0]
            y,col=self.testing_one(T,num,theta)
            Y.append(y)
            COL.append(col/num)
            tt.append(t)
            print('Testing...robot '+str(self.id)+' '+str(cnt)+'/'+str(len(self.Theta))+' Done!')
            cnt=cnt+1
        print('Robot '+str(self.id)+' completes testing !!!!')
        self.testing_Y=Y
        self.testing_col=COL
        
        return tt
    
    def kruzkov_transform(self,t):
        return 1-exp(-self.alpha*(t+1))

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

    def controller(self):
        X=self.x[0,:2]-self.X_
        X_norm=np.linalg.norm(X,axis=1)
        # print(X_norm)
        X_norm_clip=np.where(X_norm[1:]-self.obs_size<0,0,X_norm[1:]-self.obs_size)
        # print(X_norm_clip)
        X[1:,:]=X[1:,:]/(X_norm_clip*np.ones((len(X)-1,2)).T).T/self.theta[1]
        X[0]=X[0]*self.theta[0]

        u=np.sum(X,axis=0)

        return u[0]/np.linalg.norm(u)

    def run(self,E):
        Collision=False
        Goal=False
#        print(self.x)
        if np.linalg.norm(self.x[0,:2]-self.goal,ord=1)<self.goal_size:
            Goal=True
        elif any(np.linalg.norm(self.x[0,:2]-self.obs,axis=1)<self.obs_size):
            Collision=True
        elif self.x[0,0]>1 or self.x[0,0]<0 or self.x[0,1]>1 or self.x[0,1]<0:
            Collision=True
        else:
            self.u=self.controller()
            # print(u.T)

            self.x=self.dynamics(self.x,self.u,E)  

            self.X=np.vstack((self.X,self.x))

        return Collision, Goal, self.x

    def reinitialize(self,random_seed):
        file='./pkl/robot'+str(self.id)+'_'+str(self.n_obs)+'.pkl'
        fr=open(file,'rb')
        robot_data=pickle.load(fr)
        np.random.seed(random_seed)
        theta_g=np.random.uniform(-1,0,(1,2))*self.m_g
        theta=np.random.uniform(0,1,(1,2))*self.m_obs
        theta=np.vstack((theta_g,theta))
#                theta_g=np.random.normal(-1,0,(2,1))*self.m_g
#                theta=np.random.normal(0,1,(2,1*n_obs))*self.m_obs
        self.theta_=theta

        robot_data['theta']=self.theta_
#                robot_data['y']=self.y
        robot_data['theta_0']=self.theta_
        robot_data['Theta']=[(-1,theta)]
        fw=open(file,'wb')                
        pickle.dump(robot_data,fw)
#            print(str(robo.id)+' data written!!')
        fw.close()

    def local_update(self,T,random_seed,Z,i,lock):
        file='./pkl/robot'+str(self.id)+'_'+str(self.n_obs)+'.pkl'

        lock.acquire()
        fr=open(file,'rb')
        robot_data=pickle.load(fr)
        fr.close()
        lock.release()

        self.converge=robot_data['converge']
        self.Theta=robot_data['Theta']
        if not self.converge:

            self.theta_=robot_data['theta']
            self.local_converge=robot_data['local_converge']
            self.local_converge_theta=robot_data['local_converge_theta']
            self.collect_y(T,random_seed)
            robot_data['y']=self.y
            lock.acquire()
            print(str(self.id)+' local updating y = ', end='')
            print(self.y)
            lock.release()

            if self.y<1: 
                print(str(self.id)+' local updating z ... ')
                self.collect_z(T,random_seed,Z)
                robot_data['z']=self.z
                print(self.id,'Theta length = ',len(robot_data['Theta']), 'z norm = ',np.linalg.norm(self.z), 'Theshold = ',2*np.sqrt(self.n_theta)*self.q)
#                self.Theta.append(self.theta_)
#                robot_data['Theta'].append(self.theta_)
                robot_data['Y'].append(self.y)
                if np.linalg.norm(self.z)>= 2*np.sqrt(self.n_theta)*self.q:  #??????????????????????
                    self.theta_=self.theta_-self.z*self.r#/(i+1)                    
                    self.Y.append(self.y)
#                    robot_data['Y'].append(self.y)
                    robot_data['theta']=self.theta_
                    print('theta updated_one_time')
                else:
                    self.converge=True
                    robot_data['converge']=True
                    if not self.local_converge:
                        self.local_converge=True
                        self.local_converge_theta=self.theta_
                        robot_data['local_converge_theta']=self.theta_
                        robot_data['local_converge']=True
                        robot_data['local_converge_t']=i

                lock.acquire()
                fw=open(file,'wb')                
                pickle.dump(robot_data,fw)
                fw.close()
                lock.release()
                
            
            elif len(self.Theta)==1:
                self.reinitialize(random_seed)
                self.local_update(T,random_seed+1,Z,i,lock)
                print('local updated')
            
        print(self.id, 'Done local update!  ', "convergence: ", str(self.converge))
                
    def collect_y(self,T,random_seed):
        y=0
        np.random.seed(random_seed)
        for E in range(self.n_E):
            goal, init, obs=env_config(self.n_obs,random_seed=None)
            for j in range(self.n_init):
                _, init,_=env_config(self.n_obs,random_seed=None)

                self.run_setup(goal,init,obs)
                eta=0
                self.theta=self.theta_

                current_x=self.x
                for t in range(T):                   
                    # print(self.goal)                  
                    Collision,Goal,current_x=self.run(E+random_seed*self.n_E)
                    if Collision:
                        eta=1
                        break
                    if Goal:   
                        print(t)
                        #eta=self.kruzkov_transform(t) 
                        eta=self.reward(current_x,t,True,False)
                        break                      
                if not Collision and not Goal:
                    eta=self.reward(current_x,t,False,True)
                y=y+eta
                
        y=y/self.n_E/self.n_init
        self.y=y
        return y

    def zeroth_gradient(self,T,goal,init,obs,eta,E,random_seed,Z):
        z=0
        for k in range(Z):
            delta=np.random.normal(0,1,self.theta.shape)      
            self.theta=self.theta_+delta  
            self.run_setup(goal,init,obs)
            eta_=1
            for t in range(T):
                Collision,Goal,current_x=self.run(E+random_seed*self.n_E)
                if Collision:
                    eta_=1
                    break
                if Goal:
                    eta_=self.reward(current_x,t,True,False)
                    break
            if not Collision and not Goal:
                eta_=self.reward(current_x,t,False,True)
            z=z+(eta_-eta)*delta 
        return z/Z

    def collect_z(self,T,random_seed,Z):
        z_k=0
        np.random.seed(random_seed)
        n_E=int(self.n_E/5)
        n_init=self.n_init
        for E in range(n_E):
            goal, init, obs=env_config(self.n_obs,random_seed=None)
            for j in range(n_init):
                _, init,_=env_config(self.n_obs,random_seed=None)

                self.run_setup(goal,init,obs)
                eta=0
                self.theta=self.theta_
                for t in range(T):
                    Collision,Goal,current_x=self.run(E+random_seed*self.n_E)
                    if Collision:
                        eta=1
                        break
                    if Goal:
                        eta=self.reward(current_x,t,True,False)
                        break
                if not Collision and not Goal:
                    eta=self.reward(current_x,t,True,False)

                z=self.zeroth_gradient(T,goal,init,obs,eta,E,random_seed,Z)
            z_k=z_k+z
            
        self.z=z_k/n_E/n_init       
        # return z_k/n_E/n_init
        

#Environment configuration
def env_config(n_obs,random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    goal=np.random.random((1,2))*np.array([[1,0.15]])+np.array([[0,0.85]])
    init=np.random.random((1,3))*np.array([[1,0.15,2*np.pi]])
    obs=np.random.random((n_obs,2))*np.array([[1,0.5]])+np.array([[0,0.25]])
    return goal, init, obs

def single_integrator(x,u):
    time_interval=0.01
    return x+np.array([[1,1]])*u*time_interval

def dubin_car(x,u,E):
    time_interval=0.01
#    np.random.seed(E)
#    d=np.random.uniform(-0.2,0.2)
    d=wind(x[0,0],x[0,1],E) #wind disturbance
    x_=np.zeros(x.shape)
    x_[0,0]=np.cos(x[0,2])+d
    x_[0,1]=np.sin(x[0,2])
    x_[0,2]=1/0.03*u     #u=[-1/np.sqrt(3),1/np.sqrt(3)] (2*sigmoid(u)-1)/np.sqrt(3.0) -30~30

    x_go=x+time_interval*x_
    if x_go[0,2]<0:
        x_go[0,2]=x_go[0,2]+2*np.pi
    elif x_go[0,2]>2*np.pi:
        x_go[0,2]=x_go[0,2]-2*np.pi
    return x_go
    
def cloud_update(file_global,robo_network):
    for robo in robo_network:
        try:
            f=open(file_global,'rb')
            global_min=pickle.load(f)
            y_j=global_min['y']
            s_j=global_min['s']            
    
            file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl'
            fb=open(file_r,'rb')
            robo_data=pickle.load(fb)
            robo.y=robo_data['y']
            
            print(robo.y,robo.s,y_j,s_j)

            if robo.y+robo.s<y_j+s_j:
                global_min=dict()
                global_min['y']=robo.y
                global_min['s']=robo.s
                global_min['id']=robo.id
                global_min['theta']=robo.theta
                # global_min=np.array([robo.y,robo.s, robo.id])
                f=open(file_global,'wb')
                pickle.dump(global_min,f)
                f.close()
            
        except FileNotFoundError:
            global_min=dict()
            file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl'
            fr=open(file_r,'rb')
            robo_data=pickle.load(fr)
            global_min['y']=robo_data['y']
            global_min['s']=robo.s
            global_min['id']=robo.id
            global_min['theta']=robo_data['theta']
            f=open(file_global,'wb')
            pickle.dump(global_min,f)

def learner_fusion(robo_network):
    for robo in robo_network:
        f=open(file_global,'rb')
        global_min=pickle.load(f)
        y_j=global_min['y']
        s_j=global_min['s']
        id_j=global_min['id']
        
        file_r='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl'
        fr=open(file_r,'rb')
        robo_data=pickle.load(fr)
        robo.y=robo_data['y']
        robo.z=robo_data['z']
        robo.theta_=robo_data['theta']
        robo.zeta=robo_data['zeta']
        robo.converge=robo_data['converge']
        if robo.id != id_j and y_j+s_j< robo.zeta and  y_j+s_j< robo.y-robo.s and np.linalg.norm(robo.z)<2*np.sqrt(robo.n_theta)*robo.q:
            robo.theta_=global_min['theta']
            robo.zeta=y_j
            robo.converge=False
            robo_data['converge']=robo.converge
            robo_data['theta']=robo.theta_
            robo_data['zeta']=y_j
            robo_data['switch']=True
            robo_data['Switch'].append((i,robo.theta_))
            print('Robot ', robo.id, 'Switched to ',id_j,'!!!!')
        if not robo.converge:
            robo.Theta.append((i,robo.theta_))
            robo_data['Theta'].append((i,robo.theta_))
            print('Robot '+str(robo.id)+' Theta updated!')
            fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl','wb')
            
            pickle.dump(robo_data,fw)
            
            fw.close()

        
if __name__=='__main__':
    start=time.time()
    test_num=200
    #setup configuration
    random_seed=36
    
    #environment parameter
    n_obs=10
    obs_size=0.03
    goal_size=0.05
    n_E=20
    n_init=5
    
    #optimization parameter
    Z=2 #number of samples for zero-th order gradient computation
    r=0.01  #step size for gradient descent   
    alpha=0.005   #Kruzkov transform 
    T=300 #maximium exploration step
    
    np.random.seed(10)  
    
    #FedGen parameter
    gamma=0.01
    ell=0.01 #Lipschitz constant
    
    q=np.sqrt(2*np.log(2/gamma)/n_E/n_init)*ell
    s=np.sqrt(np.log(2/gamma)/n_E/n_init/2)
    zeta=1
    K=20
    print(q,s)
    robo_network=[]
    n_robot=8
    
    #Initialization
    for i in range(n_robot):
        #initialize theta
        m_g=2
        m_obs=10+i*10
        theta_g=np.random.uniform(-1,0,(1,2))*m_g
        theta=np.random.uniform(0,1,(1,2))*m_obs
        theta=np.vstack((theta_g,theta))
        theta_size=theta.size
        #initialize robot
        robo=robot(dubin_car,goal_size,obs_size,n_E,n_init,n_obs,theta.size,q,s,i,zeta,r,theta,10+i*10,m_obs,alpha)
        robo_network.append(robo)
        
    bias=0
    file_global='./pkl/global_minimum'+str(n_obs)+'.pkl'
    Process=[]
    lock = mp.Lock()
    #FedGen algorithm
    for i in range(K):
        print('Iteration '+str(i))
        for robo in robo_network:
            p=mp.Process(target=robo.local_update, args=(T,i+robo.id*20,Z,i,lock,))
#            robo.local_update(T,i+robo.id*20,alpha,Z,i)
            Process.append(p)
            p.start()
        for p in Process:
            p.join()        
        cloud_update(file_global,robo_network)
        learner_fusion(robo_network)
                
#testing ..........................................
    print('testing............')
    Robot_Y=[]
    Robot_COL=[]
    
    P_test=[]
    for robo in robo_network:
        p=mp.Process(target=test_robot,args=(robo,T,test_num,lock,))
        p.start()
        P_test.append(p)
    for p in P_test:
        p.join()
    print('Done testing.')
    
    for robo in robo_network:
        file='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'.pkl'
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



