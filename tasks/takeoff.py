import numpy as np
from physics_sim import PhysicsSim

class Takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, runtime=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
                
        init_pose = np.array([0., 0., 0., 0., 0., 0.])   #set initial position at origin 
        init_velocities = np.array([0., 0., 0.])         # initial velocities at 0 for takeoff
        init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities at 0 for takeoff
        
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 250
        self.action_high = 750
        self.action_size = 4


    def get_reward(self):
        """Uses current pose of sim to return reward.
        
        Objective(takeoff): 
            starting from x=0,y=0,z=0, we want it to shoot up in the z axis, while remaining steady in the x & y
        """
        
        reward_pos = 0.
        penalty_pos = 0.
        reward_vel = 0.
        penalty_vel = 0.     
        penalty_tilt = 0.
        penalty_eulVel = 0.
        penalty_propDiff = 0.    
        reward = 0.
        
        # reward based on z (height) distance above 0 for takeoff, with penalties for x and y drift
        if self.sim.pose[2] != 0.:
            reward_pos = np.log(np.square(self.sim.pose[2]))
        if self.sim.pose[0] != 0. or self.sim.pose[1] != 0.:
            penalty_pos = np.log((0.3*(np.square(self.sim.pose[:2]).sum())))
        reward_pos -= penalty_pos           
        
        # reward based on z velocity (+/-), with penalties for x and y velocities
        if self.sim.v[2] > 0.:
            reward_vel = np.log(np.square(self.sim.v[2]))
        elif self.sim.v[2] < 0.:
            reward_vel = 0 - np.log(np.square(self.sim.v[2]))            
        if self.sim.v[0] != 0. or self.sim.pose[1] != 0.:
            penalty_vel = np.log((0.3*(np.square(self.sim.v[:2]).sum())))
        reward_vel -= penalty_vel 
        '''
        #penalize drift in tilt position from all axis
        if self.sim.pose[3:6].any() != 0.:
            penalty_tilt = 0. - (0.5*(np.log(np.square(self.sim.pose[3:6]).sum()))    
        ''''''
        #euler velocities should penalize drift in any axis
        if self.sim.angular_v.any() != 0.:
            penalty_eulVel = 0. - (0.5*(np.log(np.square(self.sim.angular_v[:3]).sum()))
        '''
        #penalty for difference between fastest/slowest prop, for gentle adjustments.
        propDiff = np.square(np.amax(self.sim.prop_wind_speed) - np.amin(self.sim.prop_wind_speed))              
        if propDiff > 0.:
            penalty_propDiff = 0. - (0.25*(np.log(np.square(propDiff))))       
        
        #combine
        reward = reward_pos + reward_vel + penalty_tilt  + penalty_eulVel + penalty_propDiff
                                   
        return reward
    
    
    
    
    
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state