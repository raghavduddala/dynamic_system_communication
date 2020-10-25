from collections import deque
import random
import numpy as np 

"""
deque = double ended queue, have to go through this!!
History has to be changed such that it just takes the value of previous action and feeds that into network and is updated incrementally
Current State is already given to the model 
"""


class EpisodeMemory:
    def __init__(self,env, max_steps_episode):
        self._states = []
        self._actions = []
        self._rewards = []
        self._history = []
        self._env = env
        self._max_steps_episode = max_steps_episode


    def episode_step(self, state, action, reward, terminal=False):
        self._actions.append(action)
        self._states.append(state)
        self._rewards.append(reward)






class ReplayMemory:
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._buffer = deque()

    def add_episode(self,trajectory):
        self._buffer.append(trajectory)


