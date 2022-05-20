import numpy as np
import pybullet as pybullet
import time
from wind import *

# Helper functions for setting up obstacle environments, simulating dynamics, etc.

# Parameters
def get_parameters():
    
    params = {} # Initialize parameter dictionary
    params['numRays'] = 20 # number of rays for sensor measurements
    params['senseRadius'] = 5.0 # sensing radius
    params['robotRadius'] = 0.27 # radius of robot
    params['robotHeight'] = 0.15/2 # rough height of COM of robot
    params['th_min'] = -np.pi/3 # sensing angle minimum 
    params['th_max'] = np.pi/3 # sensing angle maximum
    params['T_horizon'] = 150 # time horizon over which to evaluate everything   
    
    # precompute vector of angles for sensor
    params['thetas_nominal'] = np.reshape(np.linspace(params['th_min'], params['th_max'], params['numRays']), (params['numRays'],1))
    
    return params 


# Robot dynamics
def robot_update_state(state, u_diff):
    
    # State: [x,y,theta]
    # x: horizontal position
    # y: vertical position
    # theta: angle from vertical (positive is anti-clockwise)
    
    # Dynamics:
    # xdot = -(r/2)*(ul + ur)*sin(theta)
    # ydot = (r/2)*(ul + ur)*cos(theta)
    # thetadot = (r/L)*(ur - ul)

    # Robot parameters
    r = 0.1; # Radius of robot wheel
    L = 0.5; # Length between wheels (i.e., width of base)
    
    dt = 0.05
    v0 = 2.5 # forward speed
    
    # Saturate udiff
    u_diff_max = 0.5*(v0/r) 
    u_diff_min = -u_diff_max
    u_diff = np.maximum(u_diff_min, u_diff)
    u_diff = np.minimum(u_diff_max, u_diff)
    
    ul = v0/r - u_diff
    ur = v0/r + u_diff

    #d=wind(state[0],state[1])
    d=np.random.uniform(-0.2,0.2)

    new_state = [0.0, 0.0, 0.0]
    new_state[0] = state[0] + dt*(-(r/2)*(ul + ur)*np.sin(state[2])+d) # x position
    new_state[1] = state[1] + dt*((r/2)*(ul + ur)*np.cos(state[2])) # y position
    new_state[2] = state[2] + dt*((r/L)*(ur - ul))
    
    return new_state

# Create some obstacles 
def generate_obstacles(p, heightObs, robotRadius):
    # First create bounding obstacles
    x_lim = [-5.0, 5.0]
    y_lim = [0.0, 10.0]
        
    numObs = 15+np.random.randint(0,11) # 30 
    # radiusObs = 0.15
    massObs = 0
    visualShapeId = -1
    linkMasses = [None]*(numObs+3) # +3 is because we have three bounding walls
    colIdxs = [None]*(numObs+3)
    visIdxs = [None]*(numObs+3)
    posObs = [None]*(numObs+3)
    orientObs = [None]*(numObs+3)
    parentIdxs = [None]*(numObs+3)
    
    linkInertialFramePositions = [None]*(numObs+3)
    linkInertialFrameOrientations = [None]*(numObs+3)
    linkJointTypes = [None]*(numObs+3)
    linkJointAxis = [None]*(numObs+3)

    for obs in range(numObs):
        
        linkMasses[obs] = 0.0
        visIdxs[obs] = -1 # p.createVisualShape(p.GEOM_CYLINDER,radiusObs,[1,1,1],heightObs,rgbaColor=[0,0,0,1])
        parentIdxs[obs] = 0
    
        linkInertialFramePositions[obs] = [0,0,0]
        linkInertialFrameOrientations[obs] = [0,0,0,1]
        linkJointTypes[obs] = p.JOINT_FIXED
        linkJointAxis[obs] = np.array([0,0,1]) # [None]*numObs
        
        posObs_obs = np.array([None]*3)
        posObs_obs[0] = x_lim[0] + (x_lim[1] - x_lim[0])*np.random.random_sample(1) 
        posObs_obs[1] = 2.0 + y_lim[0] + (y_lim[1] - y_lim[0] - 2.0)*np.random.random_sample(1) # Push up a bit
        posObs_obs[2] = 0 # set z at ground level
        posObs[obs] = posObs_obs # .tolist()
        orientObs[obs] = [0,0,0,1]
        colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=(0.20 - 0.1)*np.random.random_sample(1)+0.1,height=heightObs)
        # colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=radiusObs,height=heightObs)

    # Create bounding objects
    # Left wall
    linkMasses[numObs] = 0.0
    visIdxs[numObs] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0]) # -1
    parentIdxs[numObs] = 0
    linkInertialFramePositions[numObs] = [0,0,0]
    linkInertialFrameOrientations[numObs] = [0,0,0,1]
    linkJointTypes[numObs] = p.JOINT_FIXED
    linkJointAxis[numObs] = np.array([0,0,1]) 
    posObs[numObs] = [x_lim[0], (y_lim[0]+y_lim[1])/2.0, 0.0]
    orientObs[numObs] = [0,0,0,1]
    colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
    
    
    # Right wall
    linkMasses[numObs+1] = 0.0
    visIdxs[numObs+1] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0]) # -1
    parentIdxs[numObs+1] = 0
    linkInertialFramePositions[numObs+1] = [0,0,0]
    linkInertialFrameOrientations[numObs+1] = [0,0,0,1]
    linkJointTypes[numObs+1] = p.JOINT_FIXED
    linkJointAxis[numObs+1] = np.array([0,0,1]) 
    posObs[numObs+1] = [x_lim[1], (y_lim[0]+y_lim[1])/2.0, 0.0]
    orientObs[numObs+1] = [0,0,0,1]
    colIdxs[numObs+1] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])
    
    # Bottom wall
    linkMasses[numObs+2] = 0.0
    visIdxs[numObs+2] = -1 # p.createVisualShape(p.GEOM_BOX, halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2], rgbaColor=[0.8,0.1,0.1,1.0])
    parentIdxs[numObs+2] = 0
    linkInertialFramePositions[numObs+2] = [0,0,0]
    linkInertialFrameOrientations[numObs+2] = [0,0,0,1]
    linkJointTypes[numObs+2] = p.JOINT_FIXED
    linkJointAxis[numObs+2] = np.array([0,0,1]) 
    posObs[numObs+2] = [(x_lim[0]+x_lim[1])/2.0, y_lim[0], 0.0]
    orientObs[numObs+2] = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
    colIdxs[numObs+2] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2])        
        
    obsUid = p.createMultiBody(baseCollisionShapeIndex = -1, baseVisualShapeIndex = -1, basePosition = [0,0,0], baseOrientation = [0,0,0,1], baseInertialFramePosition = [0,0,0], baseInertialFrameOrientation = [0,0,0,1], linkMasses = linkMasses, linkCollisionShapeIndices = colIdxs, linkVisualShapeIndices = visIdxs, linkPositions = posObs, linkOrientations = orientObs, linkParentIndices = parentIdxs, linkInertialFramePositions = linkInertialFramePositions, linkInertialFrameOrientations = linkInertialFrameOrientations, linkJointTypes = linkJointTypes, linkJointAxis = linkJointAxis)
    
    return obsUid

# Simulate range sensor (get distances along rays)        
def getDistances(p, state, robotHeight, numRays, senseRadius, thetas_nominal): 

        # Get distances 
        # rays emanate from robot
        raysFrom = np.concatenate((state[0]*np.ones((numRays,1)), state[1]*np.ones((numRays,1)), robotHeight*np.ones((numRays,1))), 1)

        thetas = (-state[2]) + thetas_nominal # Note the minus sign: +ve direction for state[2] is anti-clockwise (right hand rule), but sensor rays go clockwise

        raysTo = np.concatenate((state[0]+senseRadius*np.sin(thetas), state[1]+senseRadius*np.cos(thetas), robotHeight*np.ones((numRays,1))), 1)

        coll = p.rayTestBatch(raysFrom, raysTo)

        dists = np.zeros((1,numRays))
        for i in range(numRays):
            dists[0][i] = senseRadius*coll[i][2]
            
                
        return dists   



# Precompute costs for different environments and controllers
def environment_fail_rate(numEnvs, controller, params, husky, sphere, GUI, seed, mode=0):
    
    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']   
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']
    
    # Fix random seed for consistency of results
    np.random.seed(seed)
    
    # Initialize costs for the different environments and different controllers
    costs = np.zeros((numEnvs))
    
    for env in range(0,numEnvs):
                
        # Print
        if (env%100 == 0) and (env>0):
            print(env, "out of", numEnvs)
    
        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, robotRadius)  

        # Initialize position of robot
        state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])        

        # Cost for this particular controller (lth controller) in this environment
        cost_env_l = 1.0

        for t in range(0, T_horizon):

            # Get sensor measurement
            y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

            # Compute control input
            u = controller.predict(y,mode)

            # Update state
            state = robot_update_state(state, u)

            # Update position of pybullet object
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])    

            if (GUI):
                pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                time.sleep(0.025) 


            # Check if the robot is in collision. If so, cost = 1.0.      
            # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
            closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)

            # See if the robot is in collision. If so, cost = 1.0. 
            if closestPoints: # Check if closestPoints is non-empty 
                cost_env_l = 1.0
                break 

            if state[1]>10.1:
                break
        
        # Check that cost is between 0 and 1 (for sanity)
        if (cost_env_l > 1.0):
            raise ValueError("Cost is greater than 1!")
            

        
        # Record cost for this environment and this controller
        costs[env] = cost_env_l
            
        # Remove obstacles
        pybullet.removeBody(obsUid)
    
    return np.average(costs,axis=None)


def environment_costs(numEnvs, controller, params, husky, sphere, GUI, seed, mode=0):
    
    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']   
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']
    
    # Fix random seed for consistency of results
    np.random.seed(seed)
    
    # Initialize costs for the different environments and different controllers
    costs = np.zeros((numEnvs))
    success= np.zeros((numEnvs))
    
    for env in range(0, numEnvs):
                
        if (env%100 == 0) and (env>0):
            print(env, "out of", numEnvs)
    
        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, robotRadius)  

        # Initialize position of robot
        state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])        

        # Cost for this particular controller (lth controller) in this environment
        cost_env_l = 1.0
        success_env= 0.0

        for t in range(0, T_horizon):

            # Get sensor measurement
            y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

            # Compute control input
            u = controller.predict(y,mode)
            #print(u)

            # Update state
            state = robot_update_state(state, u)

            # Update position of pybullet object
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])    

            if (GUI):
                pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                time.sleep(0.025) 


            # Check if the robot is in collision. If so, cost = 1.0.      
            # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
            closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)
            if (10.0-float(state[1]))/10.0< cost_env_l:
                cost_env_l = (10.0-float(state[1]))/10.0
            
            mindis_cost=(-np.min(np.array(y)[0])/5.0+1.0)/5.0
            #mindis_cost=0
            # See if the robot is in collision. If so, cost = 1.0. 
            if closestPoints: # Check if closestPoints is non-empty 
                success_env = 1.0
                cost_env_l+=0.1
                break

            if state[1]>10.1:
                break
        
        # Check that cost is between 0 and 1 (for sanity)
        if (cost_env_l > 2.0):
            raise ValueError("Cost is greater than 1!")
            
        # Record cost for this environment and this controller
        costs[env] = max(cost_env_l,0)
        success[env] = success_env
            
        # Remove obstacles
        pybullet.removeBody(obsUid)

    return np.average(costs,axis=None), np.average(success,axis=None)




# Compute expected cost (just a utility function)
def expected_cost(p, precomputed_costs):
    
    # Compute average cost
    costs_envs = np.matmul(precomputed_costs, p)
    cost_mean = np.mean(costs_envs)
    
    return cost_mean

def simulate_controller(numEnvs, controller, params, husky, sphere, GUI, seed, mode=0):
    
    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']   
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']
    
    # Fix random seed for consistency of results
    np.random.seed(seed)
    
    for env in range(0,numEnvs):
        print(env)
        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, robotRadius)
        
            
        # Initialize position of robot
        state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])        

        cost_env_l=1
        for t in range(0, T_horizon):

            # Get sensor measurement
            y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

            # Compute control input
            u = controller.predict(y,mode)
            
            # Update state
            state = robot_update_state(state, u)
            #print(state)

            # Update position of pybullet object
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])    

            if (GUI):
                pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                time.sleep(0.025) 


            # Check if the robot is in collision. If so, cost = 1.0.      
            # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
            closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)

            if (10.0-float(state[1]))/10.0< cost_env_l:
                cost_env_l = (10.0-float(state[1]))/10.0
            # See if the robot is in collision. If so, cost = 1.0. 
            if closestPoints: # Check if closestPoints is non-empty 
                print("collision")
                cost_env_l+=0.1
                break 
            if state[1]>10.1:
                break

        cost_env_l =max(cost_env_l,0)
        print(cost_env_l)
        # Remove obstacles
        pybullet.removeBody(obsUid)
    
    return True
       
    
def setup_pybullet(GUI, params):
    # pyBullet
    if (GUI):
        pybullet.connect(pybullet.GUI)
    else:
        pybullet.connect(pybullet.DIRECT) 

    # Get some robot parameters
    
    robotRadius = params['robotRadius']
    numRays = params['numRays']
    thetas_nominal = params['thetas_nominal']

    # Ground plane
    pybullet.loadURDF("./URDFs/plane.urdf")

    # Load robot from URDF
    husky = pybullet.loadURDF("./URDFs/husky.urdf", globalScaling=0.5)

    # Sphere
    colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robotRadius)
    mass = 0
    if (GUI):
        visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=robotRadius,rgbaColor=[0,0,0,0]) # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
    else:
        visualShapeId = -1

    sphere = pybullet.createMultiBody(mass,colSphereId,visualShapeId)

    return husky, sphere, numRays, thetas_nominal,robotRadius
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    