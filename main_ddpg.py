import os
import random
import gym 
import numpy as np 
import cartpole_random
from cartpole_random_env import random_cartpole_env
from random_process_action import OURandomNoiseAction
import matplotlib.pyplot as plt 
from ddpg import DDPGAgent

directory = "checkpoints"
# env = gym.make("Cartpole-v0")
env  = gym.make("Cartpole_rand-v0")
env.seed(1234)
np.random.seed(1234)

BATCH_SIZE = 32
EPISODES = 3000
EPISODE_LENGTH = 500
MAX_BUFFER_SIZE = 500000
GAMMA = 0.99 # default value , can be changed and see how this affects
TAU = 1e-3
ACTOR_LR = 1e-4 #default value= 10^-4, let us change to another value after training
CRITIC_LR = 1e-3
WARMUP_TIME = 100000

agent = DDPGAgent(env, MAX_BUFFER_SIZE, GAMMA, TAU, ACTOR_LR, CRITIC_LR)
noise = OURandomNoiseAction(env.action_space)
# Hyper Parameters


rewards = []
avg_rewards = []

if not os.path.exists(directory):
    os.makedirs(directory)

####### To check the environment 

# env.reset()

# for _ in range(1000000):
#     env.render()
#     env.step(env.action_space.sample())

# env.close()

### Actual Training Starts

# time_step = 0
# episode = 0
# for ep in range(EPISODES):
#     state_array = env.reset()
#     noise.reset()
#     episode_reward = 0
#     episode_step = 0

#     for ep_step in range(EPISODE_LENGTH):

#         # env.render()
#         if time_step <= WARMUP_TIME:
#             action = agent.sample_random_action()
#             # print(time_step)
#         else:
#             action_frm_net = agent.action_input(state_array)
#             action = action_frm_net + noise.get_action(action_frm_net,ep_step)
#             # print("raghav_network")
#         # noise = noise()
#         # print(action)
#         obs, reward, done, _ = env.step(action)
#         # print(reward)
#         agent.replay_buffer.add_episode_step(state_array,action,reward,obs,done)
#         time_step += 1
#         episode_step += 1
#         episode += 1
#         # print(episode_step)
#         state_array = obs
#         episode_reward += reward

#         if time_step > WARMUP_TIME:
#             agent.policy_update(BATCH_SIZE)
#             # print("raghav")
        
#         if done:
#             break
        
        
#     # print(episode_reward)

  
#     # print(episode)
#     rewards.append(episode_reward)
#     avg_rewards.append(np.mean(rewards[-10:]))


# env.close()
# agent.save_model(directory)

# # print(rewards)
# plt.plot(rewards)
# plt.plot(avg_rewards)
# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# plt.show()





######### Evaluating the model parameters 

# TEST_EPISODES = 1000
# TEST_EPISODE_LEN = 500
# agent.load_model(directory)
# agent.eval()

# test_rewards = []
# avg_test_rewards = []
# observation_list = []
# test_time_step = 0

# for ep in range(TEST_EPISODES):
#     state_array = env.reset()
#     episode_reward = 0
#     episode_step = 0
#     for ep_step in range(TEST_EPISODE_LEN):
#         # env.render()
#         action = agent.action_input(state_array)
#         # print("from network", action)
#         observation, reward, done, info = env.step(action)
#         episode_reward += reward
#         state_array = observation 
#         test_time_step += 1
#         episode_step += 1
#         # print(episode_step)
#         # print(done)
        
#         # print(reward)
#         if done:
#             break
#         observation_list.append(observation)
    
#     test_rewards.append(episode_reward)
#     avg_test_rewards.append(np.mean(test_rewards[-10:]))

# env.close()

# observation_list_arr = np.array(observation_list)
# print(observation_list_arr.shape)

# plt.figure()
# plt.plot(test_rewards)
# plt.plot(avg_test_rewards)
# plt.xlabel("Episodes")
# plt.ylabel("Reward")

# plt.figure()
# # plt.plot(observation_list_arr[:,0])
# # plt.plot(observation_list_arr[:,1])
# plt.plot(observation_list_arr[:,2])
# plt.xlabel("Time Steps")
# plt.ylabel("Theta")
# plt.show()


######### Evaluating a Random Communication Message from the env based on the actions #########
###Loading the learned model weights 
agent.load_model(directory)
agent.eval()


# #### Some Hyper Parameters for evaluation ####
N = 2000     #Length of the Episode
epsilon = 0.08   # Epsilon value for filling the communication message box #Default value = 0.08 works really good!


###### Message array for communication #########
message_array = np.zeros(N, dtype = int)
for i in range(len(message_array)):
    if random.random()< epsilon:
        message_array[i] = 1
# print(message_array)

time_step = 0
observation_list = []
random_action_list = []
indices = []
state_array_list = []


state_array = env.reset()

for j in range(N):
    env.render()
    if message_array[j] == 1:
        # Best Results are obtained 
        action = 4
        action = np.array(action).reshape(1,)
        indices.append(j)
        random_action_list.append(action)
    else:
        action = agent.action_input(state_array)
    
    state_array_list.append(state_array)
    observation, reward, done, info = env.step(action)
    time_step += 1
    observation_list.append(observation)
    if done:
        break 
    state_array = observation

env.close()

# state_arr = np.array(state_array_list)
observation_array = np.array(observation_list)
random_action_array = np.array(random_action_list)

# print(observation_array)
# print(observation_array.shape)

print("Indices with ones in input message:",indices)

"""
#Select a proper threshold for velocity(cart's and pole) and for pole angle 
#whenever a force of 4N is applied we can see that cart's velocity and pole velocity increases highly and the 
# pole angle does not change much 

#One of the best result I got:
for xdot_threshold and thetadot_threshold
[0, 13, 27, 28, 33, 101, 102, 110, 121, 123, 156, 194]
[0, 13, 27, 28, 33, 46, 48, 50, 52, 54, 73, 79, 85, 87, 101, 102, 103, 104, 110, 121, 123, 156, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 194]

# Best Result : Indices of message_array and the indices of  the observation
[0, 20, 72, 87, 96, 99, 164, 178, 190, 191, 196]
[20, 72, 87, 96, 99, 164, 178, 190, 191, 196]

xdot_threshold = 0.1
thetadot_threshold = 0.1
x_threshold = 0.05
[1, 15, 55, 56, 76, 88, 110, 115, 117, 122, 132, 154, 164, 171, 187, 190]
[1, 15, 55, 56, 76, 88, 110, 115, 117, 122, 132, 154, 164, 171, 187, 190]
"""

#Should see if I can optimize these values for different force magnitudes and different random samples
# Ideal thresholds found after manuallty tuning: works for epsilon = 0.08/0.09 and Force: 4N
xdot_threshold = 0.1
thetadot_threshold = 0.1
x_threshold = 0.05
theta_threshold = 0.01

observation_indices = []
for i in range(len(observation_array)):
    if (abs(observation_array[i,1] - observation_array[i-1,1])>= xdot_threshold) and (abs(observation_array[i,3] - observation_array[i-1,3]) >= thetadot_threshold) and (abs(observation_array[i,0]-observation_array[i-1,0])<=x_threshold):
        if i==0:
            continue
        else:
            observation_indices.append(i)

print("Indices with ones in Observed messages", observation_indices)


observed_message_array = np.zeros(N,dtype = int)
for i in range(len(observation_indices)):
    observed_message_array[observation_indices[i]] = 1

print("The Communicated Message array :",message_array)
print("The Observed Message Array:", observed_message_array)


selected_observations = []
for k in range(len(indices)):
    selected_observations.append(observation_array[indices[k]])


print("The Message array and observation_array match:", (message_array == observed_message_array).all())

# print(np.array(selected_observations))

# print(random_action_array)


# print(state_arr)




# Can give action disturbance based on the time step as a input to the step function or the constructor of the environment 
# Or store the binary communication in a list and indexed according to the time step and can be observed some how.