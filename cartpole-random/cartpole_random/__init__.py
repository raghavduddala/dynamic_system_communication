from gym.envs.registration import register

register(
    id='Cartpole_rand-v0',
    entry_point='cartpole_random.envs:CartpoleRandomEnv',
    max_episode_steps=2000,
)

# max number of steps in episode = max_episode_steps
# here we are creating custom gym environment by first creating a folder and adding the envs 
# each custom created environment must be registered here with a id and a entry point, implying we can create multiple custom environments
#i.e associating the id with the custom environment class that we created(here in our case) in the cartpole_foo_env.py file


