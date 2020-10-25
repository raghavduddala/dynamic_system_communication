import torch 
import torch.nn as nn
import torch.autograd 
from torch.autograd import Variable
import torch.optim as optim 
import numpy as np 
from ddpg_model import DDPGCritic, DDPGActor
from random_process_action import OURandomNoiseAction
from ddpg_memory_buffer import MemoryBuffer


class DDPGAgent:
    def __init__(self, env, memory_size, gamma, tau, actor_lr, critic_lr):
        self.dim_states = env.state_space.shape[0]
        self.dim_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau 
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # self.noise_action = OURandomNoiseAction(mu=np.zeros(self.dim_actions))


        self.actor = DDPGActor(self.dim_states, self.dim_actions)
        self.actor_target =  DDPGActor(self.dim_states, self.dim_actions)
        self.critic = DDPGCritic(self.dim_states, self.dim_actions)
        self.critic_target = DDPGCritic(self.dim_states, self.dim_actions)

        self.replay_buffer = MemoryBuffer(memory_size)

        for target_parameters, main_parameters in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(main_parameters.data)

        for target_parameters, main_parameters in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(main_parameters.data)


    def action_input(self,state):
        # State needs to be converted to Variable and the action we get from the output of the actor network needs to be changed back to 
        # numpy I guess using the .detach() method from torch 
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state)
        action = action.detach().numpy()
        return action

    # def action_noise(self):
        # return self.noise_action()
    def save_model(self, output):
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(output))
    
    def load_model(self, output):
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))
        
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def sample_random_action(self):
        action = np.random.uniform(-10,10,self.dim_actions)
        # print(action)
        return action

    def policy_update(self,batch_size):
        states_batch, actions_batch , rewards_batch, obss_batch, _ = self.replay_buffer.sample_experience_batch(batch_size)
        states_batch = torch.Tensor(states_batch)           # can also use torch.FloatTensor #
        actions_batch = torch.Tensor(actions_batch)
        rewards_batch = torch.Tensor(rewards_batch)
        obss_batch = torch.Tensor(obss_batch)
        # print(rewards_batch.shape)

        critic_value = self.critic.forward(states_batch,actions_batch)
        actions_next_batch = self.actor_target.forward(obss_batch)
        critic_target_value = self.critic_target.forward(obss_batch,actions_next_batch.detach())
        y_value = torch.reshape(rewards_batch,(batch_size,1)) + self.gamma*critic_target_value
        # print(critic_target_value.shape)
        # print(y_value.shape)
        # print(critic_value)
        loss_criterion = nn.MSELoss()
        critic_loss = loss_criterion(critic_value,y_value)

        #The actor loss is calculated directly as critic loss w.r.t the policy as described in the paper 
        #The loss.backward takes care of the chain rule in terms of calculating  the gradient
        #Also, the loss is negative beacuse this results in gradient being negative i.e during the weight update of gradient descent 
        # we are essentially performing gradient ascent i.e maximizing the total sum of rewards w.r.t the policy parameters
        # Continuous Control with Deep RL 
        action_policy_batch = self.actor.forward(states_batch)
        policy_loss = -self.critic.forward(states_batch, action_policy_batch).mean()

        # Weight Updates of the main network
        optim.Adam(self.actor.parameters(), lr=self.actor_lr).zero_grad()
        policy_loss.backward()
        optim.Adam(self.actor.parameters(), lr=self.actor_lr).step()

        optim.Adam(self.critic.parameters(), lr=self.critic_lr).zero_grad()
        critic_loss.backward()
        optim.Adam(self.critic.parameters(), lr=self.critic_lr).step()

        #Weight Updates of Target Networks
        #Using Polyak Averaging from the paper 
        #for this we will be suing .copy_() method from pytorch 

        for target_parameters, main_parameters in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(main_parameters.data*self.tau + target_parameters.data*(1.0-self.tau))

        for target_parameters, main_parameters in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(main_parameters.data*self.tau + target_parameters.data*(1.0-self.tau))





