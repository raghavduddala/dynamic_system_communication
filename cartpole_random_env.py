import gym
import numpy as np
import random

import cartpole_random
# from cartpole_random import cartpole_random

env_to_random = "CartPole-v0"
experiment = "Cartpole_rand-v0"
# from __future__ import collections,print_function
"""
Adapted from : https://github.com/little-nem/dynamic_randomization/blob/master/environment.py

1.Using the cartpole env directly from the gym environments without any modifications to the goal position(i.e to stabilize the pole at 90deg)
  from arbitary start points(i.e Learning a Cartpole Swing Policy with RDPG and dynamics Randomization)
2. Modifying the cartpole env specifically for different goal postions( such that the pole stays stable at a constant angle from vertical)

"""
"""
Default values currently for the following parameters:
1. Gravity = 9.82
2. Mass of the pole = 0.5
3. Mass of the cart = 0.5
4. Length of the pole = 0.5
5. Friction Coefficient = 0.1            1,5 are environment dynamics paramters and 2,3,4 are agent's dynamic parameters
"""

#Let us  make a child class and see if we can take the inputs in the constructors for sampling and returing those parameter values
# For now, Consdering the range of agent dynamic paramters for randomizations to be same and for the friction it is a bit different
#dyn_par_ranges size = (4,2) # rows for paramteres and columns for upper and lower bounds

class random_cartpole_env:
    def __init__(self,env_to_random, dyn_par_ranges):
        self._env_to_random = env_to_random
        self._dyn_par_ranges = dyn_par_ranges
        self._randomized_par = []

    def sample_env(self):
        lower_bound_mass_pole = self._dyn_par_ranges[0][0]
        upper_bound_mass_pole = self._dyn_par_ranges[0][1]
        lower_bound_length_pole = self._dyn_par_ranges[1][0]
        upper_bound_length_pole = self._dyn_par_ranges[1][1]
        lower_bound_mass_cart = self._dyn_par_ranges[2][0]
        upper_bound_mass_cart = self._dyn_par_ranges[2][1]
        lower_bound_friction = self._dyn_par_ranges[3][0]
        upper_bound_friction = self._dyn_par_ranges[3][1]
        random_mass_pole = lower_bound_mass_pole + (upper_bound_mass_pole - lower_bound_mass_pole)*random.random()
        random_length_pole = lower_bound_length_pole + (upper_bound_length_pole - lower_bound_length_pole)*random.random()
        random_mass_cart = lower_bound_mass_cart + (upper_bound_mass_cart - lower_bound_mass_cart)*random.random()
        random_friction = lower_bound_friction + (upper_bound_friction - lower_bound_friction)*random.random()
        self._env = gym.make(self._env_to_random)
        #Seting the random_values for dynamic parameters by calling the function
        self._env.set_dynamic_parameters(random_mass_pole,random_length_pole,random_mass_cart,random_friction)
        self._randomized_par = np.array([random_mass_pole,
                                         random_length_pole,
                                         random_mass_cart,
                                         random_friction])


    def get_sampled_env(self):
        """
        Should return botth the sampled environment and the vector of the associated dynamic parameters
        """
        return self._env, self._randomized_par

    def close_sampled_env(self):
        self._env.close()



