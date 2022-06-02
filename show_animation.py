import pybullet as pybullet
import numpy as np
import time
from utils_simulation import *
from nn_stochastic_controller import nn_stochastic_controller
from gradient import *
import torch

params = get_parameters()
model= torch.load("./pkl/robot1_iteration220.pkl")
#model= torch.load("./pkl/robot0_iteration110.pkl")
#model= torch.load("./pkl/robot0_iteration0.pkl")

GUI = True
random_seed = 2#
numEnvs = 10000 # Number of environments to show videos for
numEnvs1=1


husky, sphere, numRays, thetas_nominal,robotRadius=setup_pybullet(False, params)
cost, fail_rate=environment_costs(numEnvs, model, params, husky, sphere, False, random_seed)
print("----------------------------")
print("fail_rate: ", fail_rate)
print("cost: ", cost)
pybullet.disconnect()

husky, sphere, numRays, thetas_nominal,robotRadius=setup_pybullet(GUI, params)
print("Simulating optimized controller in a few environments...")
simulate_controller_write(numEnvs1, model, params, husky, sphere, GUI, random_seed)
pybullet.disconnect()
print("Done.")

