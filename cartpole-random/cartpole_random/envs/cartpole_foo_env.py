
import logging
import math
import os
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

# from  gym.envs.classic_control.cartpole import CartPoleEnv 
# Commented above line since creating custom env as a new parent class with the below given changes
# In the original enviroment the Action space is Discrete with two dimensions
"""
Inspired from :https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py

for the following changes:
Creating a Custom Cartpole Environment Class from Scratch
Action Space: Continuous 1-D[Force]= Real value from (-1,1)
State Space: Continuous 4-D [x,x_dot, theta, theta_dot]=[pos_cart,vel_cart,angle_pole,ang_vel_pole]
Rewards: Sparse Rewards(Discrete-Binary rewards
i.e -1 for all times if the and "0" if the pole is at self.threshold_angle

b - friction coefficient between the track and the cart is considered
neglecting friction in the cart-pole joint
"""

logger = logging.getLogger(__name__)

class CartpoleRandomEnv(gym.Env):
  metadata = {
    'render.modes': ['human','rgb_array'],
    'video.frames_per_second': 50}

  def __init__(self):
    self._gravity = 9.82
    self.mass_pole = 0.5  
    self.mass_cart = 1.0
    self._total_mass = self.mass_cart + self.mass_pole
    self.length_pole = 0.5
    self._mass_pole_length = self.length_pole*self.mass_pole
    self._force_mag = 10.0 #this is the amplitude of the force applied whose value can be changed.
    self._dt = 0.01 #time-step can be adjusted and checked how itaffects learning
    self.b = 0.1
    self._theta_threshold = 12*(2*math.pi/360) #in radians = 12 radians on either side of the vertical
    self._x_threshold = 2.4 
    self._t = 0
    self._t_max = 1000 #can be added if we need time constraint i.e total length of an episode is fixed
    # print(self.length_pole)


    # the maximum values for the states in the state space/Observation Space
    # changing the state/Observation space to five dimensions from theta to sin(theta) and cos(theta)
    #minimumvals = -maximum_vals
    maximum_vals = np.array([np.finfo(np.float32).max,
                            np.finfo(np.float32).max,
                            np.finfo(np.float32).max,
                            np.finfo(np.float32).max])
    self.action_space = spaces.Box(-self._force_mag, self._force_mag, shape=(1,))
    self.state_space = spaces.Box(-maximum_vals,maximum_vals)

    self._seed()
    self.state = None
    self.viewer = None
    self.steps_beyond_done= None

  
  def set_dynamic_parameters(self,p_mass,p_length,c_mass,frict):
    self.mass_pole = p_mass
    self.length_pole = p_length 
    self.mass_cart = c_mass
    self.b = frict
    # print(self.mass_pole)
    # print(self.length_pole)
    # print(self.mass_cart)

  def _seed(self,seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def step(self, action):
    """
    This function performs the euler integration for the next time step
    """
    action = np.clip(action, -self._force_mag, self._force_mag)[0]
    # print(action)
    action *= self._force_mag
    # print(action)
    state = self.state
    x, x_dot, theta, theta_dot = state
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_acc = (-2*self._mass_pole_length*(theta_dot**2)*sin_theta + 3*self.mass_pole*self._gravity*sin_theta*cos_theta + 
              4*action - 4*self.b*x_dot)/(4*self._total_mass - 3*self.mass_pole*cos_theta**2)

    theta_acc = (-3*self._mass_pole_length*(theta_dot**2)*sin_theta*cos_theta + 6*self._total_mass*self._gravity*sin_theta + 
                  6*(action - self.b*x_dot)*cos_theta)/(4*self.length_pole*self._total_mass - 3*self._mass_pole_length*cos_theta**2)

    x = x + x_dot*self._dt
    theta = theta + theta_dot*self._dt
    x_dot = x_dot + x_acc*self._dt
    theta_dot = theta_dot + theta_acc*self._dt
    # print(theta)
    self.state = (x,x_dot,theta,theta_dot)

    done = bool( x< -self._x_threshold or
                 x> self._x_threshold )
    # if not done:
    #   if theta>(2*math.pi-self._theta_threshold) and theta<self._theta_threshold:
    #     reward = 0.0
    #   else:
    #     reward = -1
    # else:
    #   reward = 0.0
    # if not done:
    #this reward is the default reward from the Open AI Gym environment
    
    # Reward from the PILCO paper, reference  to the paper:
    #PILCO: A Model-Based and Data-Efficient Approach to Policy Search
    # http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf
    goal = np.array([0.0, self.length_pole])
    pole_x = self.length_pole*sin_theta
    pole_y = self.length_pole*cos_theta
    position = np.array([self.state[0] + pole_x, pole_y])
    squared_distance = np.sum((position - goal)**2)
    squared_sigma = 0.25**2
    costs = 1 - np.exp(-0.5*squared_distance/squared_sigma)
    reward = -costs
    obs = np.array([x,x_dot,normalize_pole_angle(theta),theta_dot])
    return obs,reward,done,{} #don't know how many variables the function returns, have to change this.


  def reset(self):
    self.state = np.random.normal(loc=np.array([0.0, 0.0, 0.0, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
    # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    x,x_dot,theta,theta_dot = self.state
    # return np.array(self.state)
    obs = np.array([x,x_dot,normalize_pole_angle(theta), theta_dot])
    return obs


  def render(self, mode='human'):
    screen_height = 400
    screen_width = 600
    world_width = 2*self._x_threshold
    # scale = screen_width/screen_height
    #Scale Value rechanged to the original gym formula:
    scale = screen_width/world_width
    cart_y_pos = screen_height/2  
    length_pole = scale*(2*self.length_pole)
    cart_height = 30.0
    cart_width = 50.0
    pole_width = 6.0

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
      axleoffset = cart_height / 4.0
      cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      self.carttrans = rendering.Transform()
      cart.add_attr(self.carttrans)
      self.viewer.add_geom(cart)
      l, r, t, b = -pole_width / 2, pole_width / 2, length_pole - pole_width / 2, -pole_width / 2
      pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      pole.set_color(.8, .6, .4)
      self.poletrans = rendering.Transform(translation=(0, axleoffset))
      pole.add_attr(self.poletrans)
      pole.add_attr(self.carttrans)
      self.viewer.add_geom(pole)
      self.axle = rendering.make_circle(pole_width/2)
      self.axle.add_attr(self.poletrans)
      self.axle.add_attr(self.carttrans)
      self.axle.set_color(.5, .5, .8)
      self.viewer.add_geom(self.axle)
      self.wheel_l = rendering.make_circle(cart_height/4)
      self.wheel_r = rendering.make_circle(cart_height/4)
      self.wheeltrans_l = rendering.Transform(translation=(-cart_width/2, -cart_height/2))
      self.wheeltrans_r = rendering.Transform(translation=(cart_width/2, -cart_height/2))
      self.wheel_l.add_attr(self.wheeltrans_l)
      self.wheel_l.add_attr(self.carttrans)
      self.wheel_r.add_attr(self.wheeltrans_r)
      self.wheel_r.add_attr(self.carttrans)
      self.wheel_l.set_color(0, 0, 0) 
      self.wheel_r.set_color(0, 0, 0) 
      self.viewer.add_geom(self.wheel_l)
      self.viewer.add_geom(self.wheel_r)
      self.track = rendering.Line((0,cart_y_pos - cart_height/2 - cart_height/4),
              (screen_width, cart_y_pos - cart_height/2 - cart_height/4))
      self.track.set_color(0,0,0)
      self.viewer.add_geom(self.track)
      # self.track = rendering.Line((0, cart_y_pos), (screen_width, cart_y_pos))
      # self.track.set_color(0, 0, 0)
      # self.viewer.add_geom(self.track)

      self._pole_geom = pole

    if self.state is None:
      return None

        # Edit the pole polygon vertex
    # pole = self._pole_geom
    # l, r, t, b = -pole_width / 2, pole_width / 2,  - pole_width / 2, -pole_width / 2
    # pole.v = [(l, b), (l, t), (r, t), (r, b)]

    x = self.state
    cart_x_pos = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    self.carttrans.set_translation(cart_x_pos, cart_y_pos)
    self.poletrans.set_rotation(-x[2])

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

#normalized angle taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
#to change the angles from gretaer than to 2*pi to range between (-pi,pi)
def normalize_pole_angle(x):
  return (((x+np.pi)%(2*np.pi))-np.pi)