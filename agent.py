import numpy as np 
import gym
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from model import Actor, Critic
from random_process_action import OURandomNoiseAction

"""
Code Inspired from : 

"""
# Has to be reimplemented

class RDPGAgent(object):
    def __init__(self,env,gamma,par_ranges):
        self._env = gym.make(env)
        self.state_dim = self._env.state_space.shape[0]
        self.action_dim = self._env.action_space.shape[0]
        self.dim_parameters = par_ranges.shape[0]
        self.gamma = gamma
        self._noise = OURandomNoiseAction(mu = np.zeros(self.action_dim)) # zero mean noise 

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.dim_parameters,self.state_dim, self.action_dim)
        self.critic_target = Critic(self.dim_parameters,self.state_dim, self.action_dim)
        
#         self.actor_optimizer = optim.Adam(self.actor.parameters(),)
#         self.critic_optimizer = optim.Adam(self.critic.paramters(),)

    def action_noise(self):
        return self._noise()


#     def policy_update(self,batch_size):

#         state_batch, cur_action_batch, next_state_batch, env_batch = 
#         state_batch = Variable(torch.from_numpy(state_batch).float())
#         next_state_batch = Variable(torch.from_numpy(state_batch).float())
#         cur_action_batch = Variable(torch.from_numpy(state_batch).float())
#         reward_batch = Variable(torch.from_numpy(state_batch).float())
        
#         prev_action_batch = Variable(torch.from_numpy(state_batch).float())
        
        
#         value = self.critic.forward(state_batch,prev_action_batch,cur_action_batch)#parameters passed into this function need to be changed further)
#         target_action,(target_hx,target_cx) = self.actor_target.forward(next_state,action)
#         critic_target_val = self.critic_target.forward( x_dyn_par, next_state, target_action,hidden_state=None)
#         target_value = reward + self.gamma*critic_target_val
#         value_loss = nn.MSELoss(value,target_value)
#         value_loss = value_loss/len(trajectory)  # Back propagation through time, hence dividing by the length of the trajectory
#         value_loss_total = value_loss_total + value_loss 

#         action 