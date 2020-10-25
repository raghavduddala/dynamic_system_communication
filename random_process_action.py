import numpy as np

"""
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Taken all the default values as it is.
# The ornstein-uhlenbeck is first considered in the DDPG paper because it produces noise that is correlated in time.
# typically dynamical systems are said to have temporally correlated noise parameters
"""

class OURandomNoiseAction:
    # def __init__(self, mu, sigma=0.01, theta=.15, dt=1e-2, x0=None):
    #     self.theta = theta
    #     self.mu = mu
    #     self.sigma = sigma
    #     self.dt = dt
    #     self.x0 = x0
    #     self.reset()

    # def __call__(self):
    #     x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
    #             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    #     self.x_prev = x
    #     return x

    # def reset(self):
    #     self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    # def __repr__(self):
    #     return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)



