import pybullet as pybullet
import numpy as np
import time
from utils_simulation import *
from nn_stochastic_controller import nn_stochastic_controller
from gradient import *
import torch

GUI = False

# PAC parameters
env_batch_size = 10 # Number of environments to sample
env_validation_batch_size = 100
random_seed = 10
epoch= 200

params = get_parameters()
husky, sphere, numRays, thetas_nominal,robotRadius=setup_pybullet(GUI, params)


nn_controller=0
load_from_file=False

if load_from_file:
    nn_controller=torch.load("control1.pkl")
else:
    nn_controller=nn_stochastic_controller(numRays)

for i in range(epoch):
    print("epoch: ", i)
    norm=compute_gradient(env_batch_size,params, husky, sphere, nn_controller) 
    if norm<0.25:
        break
    if i%5==0 and i>0:
        print("-------------------------------------")
        nn_controller.set_random_para(i)
        cost, fail_rate=environment_costs(env_validation_batch_size, nn_controller, params, husky, sphere, GUI, random_seed)
        print("fail_rate: ", fail_rate)
        print("cost: ", cost)
        torch.save(nn_controller,"control1.pkl")
        print("-------------------------------------")

################################################################################
# Estimate true expected cost

numEnvs = 100 # 100000 were used in the paper (here, we're using few environments to speed things up) 
seed = 100 # Different random seed
cost, fail_rate_for_test = environment_costs(numEnvs, nn_controller, params, husky, sphere, GUI, seed)
print("fail_rate_for_test: ", fail_rate_for_test)
print("cost: ", cost)
print("Estimating test fail_rate based on ", numEnvs, " environments...")

pybullet.disconnect()

torch.save(nn_controller,"control1.pkl")

################################################################################
# Run optimized controller on some environments to visualize

GUI = True
random_seed = 25 #
numEnvs = 20 # Number of environments to show videos for
husky, sphere, numRays, thetas_nominal,robotRadius=setup_pybullet(GUI, params)
print("Simulating optimized controller in a few environments...")
simulate_controller(numEnvs, nn_controller, params, husky, sphere, GUI, random_seed)
pybullet.disconnect()
print("Done.")

