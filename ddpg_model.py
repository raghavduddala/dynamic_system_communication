# pylint: disable=no-value-for-parameter 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd 
from torch.autograd import Variable

class DDPGCritic(nn.Module):
    def __init__(self,num_states,num_actions):
        super(DDPGCritic, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        # self.fc1 = nn.Linear(self.num_states+self.num_actions,256)
        self.fc1 = nn.Linear(self.num_states,400)
        self.fc2 = nn.Linear(self.num_actions + 400,300)
        self.fc3 = nn.Linear(300,1)
        # self.relu = F.relu()


    def forward(self,state,action):
        # x = torch.cat([state,action],1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # out = self.fc3(x)
        x1 = state
        x2 = action
        x1_out = F.relu(self.fc1(x1))
        x2_inp = torch.cat([x2,x1_out],1)
        x2_out = F.relu(self.fc2(x2_inp))
        out = self.fc3(x2_out)
        return out


class DDPGActor(nn.Module):
    def __init__(self,num_states,num_actions):
        super(DDPGActor,self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.num_states,400)
        self.fc2 = nn.Linear(400,300)
        self.fc3 = nn.Linear(300,self.num_actions)
        # self.relu = F.relu()

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = torch.tanh(self.fc3(x))
        return out