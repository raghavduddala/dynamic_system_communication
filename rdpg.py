

import os
import numpy as np 
import random
import gym
import cartpole_random

from cartpole_random_env import random_cartpole_env

experiment = "Cartpole_rand-v0"

# env = gym.make(experiment)
"""
Checking the randomized environment implementation
"""
par_ranges = np.array([[0,1], [0,0.75], [1,2], [0,1]])

randomized_env = random_cartpole_env(experiment,par_ranges)

randomized_env.sample_env()

env,env_params = randomized_env.get_sampled_env()
print("Array of Environment Parmaters:[Pole length, Pole Mass, Cart Mass, Friction]",env_params)

state_array = env.reset()
print("State Array:[cart position, cart velocity, pole angle, pole length]",state_array)

# for _ in range(100000):
#     env.render()
#     env.step(env.action_space.sample())

# env.close()

