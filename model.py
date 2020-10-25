import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable

''' Code Inspired from the PyTorch RDPG Implementation of :
https://github.com/fshamshirdar/pytorch-rdpg/blob/master/model.py
and Tensorflow Implementation: 
https://github.com/little-nem/dynamic_randomization/blob/master/critic.py
https://github.com/little-nem/dynamic_randomization/blob/master/actor.py

Also, the current state is only given to the ff_branch assuming the history(input memory) already contains the current_state
Weight Initialization : used xavier_initializations rather than the fanin_initialization
'''


class Actor(nn.Module):
  def __init__(self, dim_actions, dim_states):
    super(Actor, self).__init__()
    self.dim_actions = dim_actions
    self.dim_states = dim_states
    self.fc1 = nn.Linear(self.dim_states, 128)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1_rec = nn.Linear(self.dim_states+self.dim_actions, 128)
    nn.init.xavier_uniform_(self.fc1_rec.weight)
    self.lstm = nn.LSTMCell(128, 128)
    self.fc2 = nn.Linear(256,  128)
    nn.init.xavier_uniform_(self.fc2.weight)
    self.fc3 = nn.Linear(128, 128)
    nn.init.xavier_uniform_(self.fc3.weight)
    self.fc4 = nn.Linear(128, self.dim_actions)
    nn.init.xavier_uniform_(self.fc4.weight)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.lstm_cx = Variable(torch.zeros(1, 128)).type(FLOAT)
    self.lstm_hx = Variable(torch.zeros(1, 128)).type(FLOAT)

  def reset_actor_lstm(self, reset=True):
      if reset == True:
          self.lstm_hx = Variable(torch.zeros(1, 128)).type(FLOAT)
          self.lstm_cx = Variable(torch.zeros(1, 128)).type(FLOAT)
      else:
          self.lstm_hx = Variable(self.lstm_hx.data).type(FLOAT)
          self.lstm_cx = Variable(self.lstm_cx.data).type(FLOAT)

  def forward(self, x_hist, x_state, hidden_state=None):
    '''we need x_hist that has the dimensions (batch*(dim_actions+dim_states)) for System Identification
    '''
    x_inp1 = x_state
    x_inp2 = x_hist
    #it's either the above if the x_state is a (1,dim_state) as in [[1,1,,,,,50]]:
    # this requires below or just [1,,,,,,50]: this requires above
    #x_inp2 = torch.cat((x_state,x_act_prev),1)
    x_out1 = self.relu(self.fc1(x_inp1))
    x_out2 = self.relu(self.fc1_rec(x_inp2))

    if hidden_state == None:
        lstm_hx, lstm_cx = self.lstm(x_out2, (self.lstm_hx, self.lstm_cx))
        self.lstm_hx = lstm_hx
        self.lstm_cx = lstm_cx
    else: 
        lstm_hx, lstm_cx = self.lstm(x_out2, hidden_state)

    x_lstm_out = lstm_hx
    x_inp3 = torch.cat((x_out1, x_lstm_out))
    #same here regarding the dimensionalities
    #x_inp3 = torch.cat((x_out1,x_out2),1)
    x_out3 = self.relu(self.fc2(x_inp3))
    x_out4 = self.relu(self.fc3(x_out3))
    out = self.tanh(self.fc4(x_out4))
    return out, (lstm_hx, lstm_cx)


class Critic(nn.Module):
  def __init__(self, dim_dyn_par, dim_actions, dim_states):
    super(Critic, self).__init__()
    self.dim_actions = dim_actions
    self.dim_states = dim_states
    self.dim_dyn_par = dim_dyn_par
    self.fc1 = nn.Linear(self.dim_dyn_par+self.dim_states + self.dim_actions, 128)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1_rec = nn.Linear(self.dim_states+self.dim_actions, 128)
    nn.init.xavier_uniform_(self.fc1_rec.weight)
    self.lstm = nn.LSTMCell(128, 128)
    self.fc2 = nn.Linear(256, 128)
    nn.init.xavier_uniform_(self.fc2.weight)
    self.fc3 = nn.Linear(128, 128)
    nn.init.xavier_uniform_(self.fc3.weight)
    self.fc4 = nn.Linear(128, 1)
    nn.init.xavier_uniform_(self.fc4.weight)
    self.relu = nn.ReLU()
    self.lstm_hx = Variable(torch.zeros(1, 128)).type(FLOAT)
    self.lstm_cx = Variable(torch.zeros(1, 128)).type(FLOAT)

  
  def reset_critic_lstm(self, reset=True):
    if reset == True:
      self.lstm_hx = Variable(torch.zeros(1, 128)).type(FLOAT)
      self.lstm_cx = Variable(torch.zeros(1, 128)).type(FLOAT)
    else:
      self.lstm_hx = Variable(self.lstm_hx.data).type(FLOAT)
      self.lstm_cx = Variable(self.lstm_cx.data).type(FLOAT)
  
  def forward(self, x_dyn_par, x_hist, x_state, x_act_cur,hidden_state=None):
    x_inp1 = torch.cat(x_dyn_par, x_state, x_act_cur)
    x_inp2 = x_hist
    x_out1 = self.relu(self.fc1(x_inp1))
    x_out2 = self.relu(self.fc1_rec(x_inp2))

    if hidden_state == None:
      lstm_hx, lstm_cx = self.lstm(x_out2, (self.lstm_hx, self.lstm_cx))
      self.lstm_hx = lstm_hx
      self.lstm_cx = lstm_cx
    else:
      lstm_hx, lstm_cx = self.lstm(x_out2, hidden_state)

    x_lstm_out = lstm_hx
    x_inp3 = torch.cat((x_out1, x_lstm_out))

    x_out3 = self.relu(self.fc2(x_inp3))
    x_out4 = self.relu(self.fc3(x_out3))

    out = self.fc4(x_out4)
    return out, (lstm_hx, lstm_cx)
