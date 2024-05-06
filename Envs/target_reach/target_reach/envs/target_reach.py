from typing import Dict, Union
import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.spaces import Box
import os
import collections
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class target_reach(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(
        self,render_mode=None,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
    ):
        #self.physics_client = p.connect(p.GUI)
        self.physics_client = p.connect(p.DIRECT)


        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self.reward_run_time_weight = 1

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.81)     
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


        planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,1.2]
        startOrientation = p.getQuaternionFromEuler(np.array([math.pi,0,0]))
        robotPath= "/home/hoan/rl-baselines3-zoo/rl_zoo3/double_pendulum/urdf/double_pendulum.urdf"
        self.robotId = p.loadURDF(robotPath,startPos, startOrientation, useFixedBase=True)

        while True:
            #self.goal = self.np_random.uniform(low=-1, high=1, size=2)
            self.goal = np.array([random.uniform(-1,1),random.uniform(0.2,2.2)])
            if np.linalg.norm(self.goal-np.array([0,1.2])) < 0.98:
                break

        self.EOF_target = np.array([0,self.goal[0],self.goal[1]])
        info = {}
        self.num_step = 0
        self._env_step_counter=0;
        return self._get_obs(0), info 
    
       
    def step(self, action):
        

        tau_J1 = action[0]*40
        tau_J2 = action[1]*20

        p.setJointMotorControl2(self.robotId , 0, p.TORQUE_CONTROL,tau_J1)
        p.setJointMotorControl2(self.robotId , 1, p.TORQUE_CONTROL,tau_J2)
        p.stepSimulation()

        self.num_step += 1
        self.run_time= self.num_step/240

        reward, reward_info = self._get_rew(action)
        info = reward_info
        observation = self._get_obs(self.run_time)

        #print(np.round(reward,2), np.round(observation,2))
        if self.render_mode == "human":
            self.render()

        terminated = False
        if self.pos_err_norm < 0.01:
            terminated = True
            print("reach target")
        Truncated = False
        if self._env_step_counter >512:
            Truncated = True
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        self._env_step_counter+=1
        return observation, reward, terminated, Truncated, info

    def _get_rew(self, action):

        self.EOF_pos = p.getLinkState(self.robotId,2)[0]
        

        self.pos_err = self.EOF_pos - self.EOF_target
        self.pos_err_norm = np.linalg.norm(self.pos_err)
        
        if self._env_step_counter % 100000 == 0:
            print(
                "targer=",np.round(self.EOF_target,3),
                "current pos=",np.round(self.EOF_pos,3),
                "error=",np.round(self.pos_err_norm,4)
                )

        reward_dist = -self.pos_err_norm * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight
        
        reward_time = (np.exp(-0.01*self.run_time**2)-1) * self.reward_run_time_weight
        
        reward_action = -(action[0]**2 + action[1]**2)*0.1

        reward = reward_dist + reward_ctrl + 0*reward_time + reward_action*0

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_time": reward_time
        }

        return reward, reward_info


    def _get_obs(self, time):

        run_time = time
        self.theta_j1_pos = p.getJointState(self.robotId,0)[0]
        self.theta_j2_pos = p.getJointState(self.robotId,1)[0]
        self.theta_j1_vel = p.getJointState(self.robotId,0)[1]
        self.theta_j2_vel = p.getJointState(self.robotId,1)[1]
        self.EOF_pos = p.getLinkState(self.robotId,2)[0]
        self.pos_err = self.EOF_pos - self.EOF_target
        self.pos_err_norm = np.linalg.norm(self.pos_err)
        
        return np.array(
            [
                np.cos(self.theta_j1_pos),
                np.cos(self.theta_j2_pos),
                np.sin(self.theta_j1_pos),
                np.sin(self.theta_j2_pos),
                self.EOF_target[1],
                self.EOF_target[2],
                self.theta_j1_vel,
                self.theta_j2_vel,
                self.pos_err[1],
                self.pos_err[2],
                run_time
            ]
        )
