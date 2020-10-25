import numpy as np 
import random
import gym
import cartpole_random
from cartpole_random_env import random_cartpole_env
from agent import RDPGAgent
from experience_replay import (EpisodeMemory, ReplayMemory)


"""
Need not store the randomized dynamic parameters as they can just be obtained by sampling the environment 
once for an entire episode and can be used
"""



experiment = "Cartpole_rand-v0"

# Hyper - Parameters
MAX_STEP_EP = 50 # Maximum number of timesteps in an episode/episode length 
MAX_NUM_EP = 100 # Maximum Number of Episodes
GAMMA = 0.9 # The Discount factor on rewards
PAR_RANGES = np.array([[0,1], [0,0.75], [1,2], [0,1]]) # Parameter ranges in the order: Pole mass, Pole Length, Cart Mass, Friction
BUFFER_SIZE = 128
EPISODES =  10  # Num. of Episodes


agent = RDPGAgent(experiment,GAMMA,PAR_RANGES)

random_env = random_cartpole_env(experiment,PAR_RANGES)

replay_buffer =  ReplayMemory(BUFFER_SIZE)

for i in range(MAX_NUM_EP):
    #each episode has different sampled dynamic parameters in the environment
    random_env.sample_env()
    env, env_parameters = random_env.get_sampled_env()
    state_array = env.reset()

    episode = EpisodeMemory(env,MAX_STEP_EP) # The parameters passed into this function are to be decided yet

    # First action is a random sample from the action space of the environment, since we need history from the next time step
    # in an episode to implement the policy(action) from the actor network
    episode_reward = 0

    first_action = env.action_space.sample()
    first_obs, reward, done, _ = env.step(first_action) #adding the step function since we need to put 
    episode.episode_step(first_obs, action, reward)             # first state, action (random action) after reset and reward = 0

    done = False


    while not done:
        obs = first_obs
        prev_action = first_action
        act_noise = agent.action_noise()
        
        # action from the network, actor network's forward should be changed
        # action = RDPGAgent.actor.forward(env_parameters) + act_noise
        action = env.action_space.sample() + act_noise

        new_obs, reward, done, _ = env.step(action)

        episode.episode_step(new_obs,action,reward)
        obs = new_obs
        prev_action = action # May this should work for the Sys-ID/internal memory for 

    #agent.actor.reset_actor_lstm(reset=True)      # This is to reset the lstm hidden and cell state after an episode/traj is completed
    replay_buffer.add_episode(episode)




