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

GUI = False
params = get_parameters()
husky, sphere, numRays, thetas_nominal,robotRadius=setup_pybullet(GUI, params)

class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,n_E,n_obs,n_theta,q,s,robo_id,zeta,controller): 
        self.x=None
        
        self.X=None
        self.n_obs=n_obs
        self.obs=None
        self.goal=None
        self.X_=None
        #self.theta=None
        self.u=None
        self.n_E=n_E
        self.Y=[]
        self.n_theta=n_theta
        self.q=q
        self.s=s
        self.id=robo_id
        self.zeta=zeta
        self.y=None
        self.z=None
        self.z_norm=None
        self.converge=False
        self.controller=controller
        self.testing_Y=None
        self.testing_y=None
        self.testing_col=None
        self.testing_runningtime=None
        self.local_converge=False
        #self.local_converge_theta=None
        
        
        file='./pkl/robot'+str(self.id)+'_'+str(n_obs)+'obs'+'.pkl'
        f=open(file,'wb')
        robot_data=dict()
        robot_data['Y']=[]
        robot_data['y']=None
        robot_data['nE']=n_E
        robot_data['n_obs']=n_obs
        robot_data['s']=s
        robot_data['q']=q
        robot_data['converge']=False
        self.controller.save_model(self.id,0)
        robot_data['z_norm']=None
        robot_data['n_obs']=n_obs
        robot_data['Switch']=[]
        robot_data['local_converge']=False
        #robot_data['local_converge_theta']=None
        robot_data['testing_Y']=None
        robot_data['testing_col']=None
        robot_data['testing_runningtime']=None
        robot_data['local_converge_test_y']=None
        robot_data['local_converge_test_col']=None
        robot_data['zeta']=1
        pickle.dump(robot_data,f)

    def testing_one(self, num, controller):
        y=0
        col=0
        y,col,runningtime=environment_runtime_cost(num, controller, params, husky, sphere, GUI, 1000)
        self.testing_y=y
        return y,col,runningtime

    def testing_all(self,num,iteration_N):
        Y=[]
        COL=[]
        Runningtime=[]
        cnt=1
        tt=[]
        for iteration in range(iteration_N):
            pybullet.disconnect()
            setup_pybullet(False, params)
            controller=load_model(self.id,iteration)
            t=iteration
            y,col,runningtime=self.testing_one(num,controller)
            Y.append(y)
            COL.append(col)
            Runningtime.append(runningtime)
            tt.append(t)
            print('Testing...robot '+str(self.id)+' '+str(cnt)+'/'+str(iteration_N)+' Done!')
            cnt=cnt+1
        print('Robot '+str(self.id)+' completes testing !!!!')
        self.testing_Y=Y
        self.testing_col=COL
        self.testing_runningtime=Runningtime
        return tt

    def local_update(self,i,lock,test_seed):
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
            self.y,_=environment_costs(self.n_E*20, self.controller, params, husky, sphere, GUI, test_seed)
            robot_data['y']=self.y
            lock.acquire()
            print(str(self.id)+' local updating y = ', end='')
            print(self.y)
            lock.release()

            print(str(self.id)+' local updating z ... ')
            self.z, robot_data['z_norm']=compute_gradient(self.n_E, params, husky, sphere, self.controller)
            print(self.id, 'z norm = ',robot_data['z_norm'], 'Theshold = ',2*np.sqrt(self.n_theta)*self.q)
            robot_data['Y'].append(self.y)
            if robot_data['z_norm']>= 2*np.sqrt(self.n_theta)*self.q:
                #self.controller.update(self.z)
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
                else:
                    self.controller.save_model(self.id, i+1)

            lock.acquire()
            fw=open(file,'wb')                
            pickle.dump(robot_data,fw)
            fw.close()
            lock.release()
            print(self.id, 'Done local update!  ', "convergence: ", str(self.converge))
        else:
            self.controller.save_model(self.id, i+1)
        return

def cloud_update(file_global,robo_network,iteration):
    if iteration>0:
        controller_current=load_model_global(iteration-1)
        controller_current.save_model_global(iteration)

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
                controller_current=load_model(robo.id,iteration+1)
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
        if robo.id != id_j and y_j+s_j< robo.zeta and  y_j+s_j< robo.y-robo.s and robo.converge==True:
            robo.controller=load_model_global(iteration)
            #robo.theta_=global_min['theta']
            robo.zeta=y_j
            robo.converge=False
            robo_data['converge']=False
            #robo.controller.save_model(robo.id,iteration+1)
            #robo_data['theta']=robo.theta_
            robo_data['zeta']=y_j
            robo_data['switch']=True
            robo_data['Switch'].append(iteration)
            print('Robot ', robo.id, 'Switched to ',id_j,'!!!!')
        
        print('Robot '+str(robo.id)+' theta updated!')
        
        robo.controller.save_model(robo.id,iteration+1)
        fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl','wb')
        pickle.dump(robo_data,fw)
        fw.close()

def test_robot(robo,num,iteration_N,lock):
    pybullet.disconnect()
    setup_pybullet(False, params)
    file='./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl'
    f=open(file,'rb')
    robo_data=pickle.load(f)
    f.close()
    tt=robo.testing_all(num,iteration_N)
    robo_data['testing_Y']=robo.testing_Y
    robo_data['testing_col']=robo.testing_col
    robo_data['testing_runningtime']=robo.testing_runningtime
    robo_data['t_theta']=tt
    
    if robo_data['local_converge']:
        controller=load_model(robo.id,robo_data['local_converge_t']+1)
        y,col=robo.testing_one(num,controller)
        robo_data['local_converge_test_y']=y
        robo_data['local_converge_test_col']=col
    
    fw=open('./pkl/robot'+str(robo.id)+'_'+str(robo.n_obs)+'obs'+'.pkl','wb')
    pickle.dump(robo_data,fw)
    fw.close()
    ##print(tt)
    #plt.plot(robo.testing_Y)
    #plt.title('Robot '+str(robo.id))
    

if __name__=='__main__':
    start=time.time()
    test_num=5000
    #setup configuration
    #random_seed=36
    #environment parameter
    n_obs=1
    n_E=10
    
    #FedGen parameter
    gamma=0.01
    ell=0.1 #0.03 #Lipschitz constant
    
    q=np.sqrt(2*np.log(2/gamma)/n_E)*ell
    s=np.sqrt(np.log(2/gamma)/n_E/2)/7.5
    zeta=1
    K=220
    theta_size=21*20
    print(q,s)
    print(2*np.sqrt(theta_size)*q)

    robo_network=[]
    n_robot=8

    file_global='./pkl/global_minimum'+str(n_obs)+'obs'+'.pkl'
    Process=[]
    lock = mp.Lock()
    
    #Initialization
    for i in range(n_robot):
        #initialize theta
        controller=nn_stochastic_controller(numRays)
        theta_size=21*20
        #initialize robot
        robo=robot(n_E,n_obs,theta_size,q,s,i,zeta,controller) #(self,n_E,n_obs,n_theta,q,s,robo_id,zeta,controller): 
        robo_network.append(robo)
    

    #FedGen algorithm
    for i in range(K):
        print('Iteration '+str(i))
        for robo in robo_network:

            if i>100:
                ell=0.05 #0.03 #Lipschitz constant
                robo.q=np.sqrt(2*np.log(2/gamma)/n_E)*ell
                robo.s=np.sqrt(np.log(2/gamma)/n_E/2)/100.0

            p=mp.Process(target=robo.local_update, args=(i,lock,10,))  #local_update(self,i,lock,test_seed):
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
        p=mp.Process(target=test_robot,args=(robo,test_num,K+1,lock,)) #test_robot(robo,num,iteration_N,lock):
        p.start()
        P_test.append(p)
    for p in P_test:
        p.join()
    print('Done testing.')


    for id in range(n_robot):
        file='./pkl/robot'+str(id)+'_'+str(n_obs)+'obs'+'.pkl'
        f=open(file,'rb')
        robo_data=pickle.load(f)
        f.close()
        Y=robo_data['testing_Y']
        tt=robo_data['t_theta']
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
        running=np.array(robo_data['testing_runningtime'])
        tt=robo_data['t_theta']
        plt.plot(tt,running,label='Robot '+str(id))
        switch=robo_data['Switch']
        y_swith=[]
        t_swith=[]
        for t in tt:
            if t in switch:
                y_swith.append(running[t])
                t_swith.append(t)
        plt.plot(t_swith,y_swith,'o',color='r')
        plt.legend()
    plt.show()