from collections import deque
import random 
import numpy as np 


class MemoryBuffer:
    def __init__(self,memory_buffer_size):
        self.max_size = memory_buffer_size
        self.memory_buffer = deque(maxlen = self.max_size)

    def add_episode_step(self,state,action,reward,obs,done):
        experience_step = (state,action, reward, obs, done)
        self.memory_buffer.append(experience_step)

    def sample_experience_batch(self,batch_size):
        state_batch = []
        action_batch = []
        reward_batch = [] 
        obs_batch = []
        eps_end_batch = []

        batch = random.sample(self.memory_buffer,batch_size)

        for exp_step in batch:
            state,action,reward,obs,done = exp_step
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            obs_batch.append(obs)
            eps_end_batch.append(done)


        return state_batch, action_batch, reward_batch, obs_batch, eps_end_batch

    
    def __len__(self):
        return len(self.memory_buffer)

