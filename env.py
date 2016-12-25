import gym
import random
import numpy as np
import scipy.misc

class Env():
  def __init__(self, env_name, screen_height, screen_width, no_op_max, is_display):
    self.env = gym.make(env_name)
    self.dims = (screen_height, screen_width)
    self.no_op_max = no_op_max
    self.is_display = is_display

    self._screen = None
    self.reward = 0
    self.done = True

  def new_random_game(self):
    self._screen = self.env.reset()
    for _ in range(random.randint(0, self.no_op_max - 1)):
      self._random_step()
    self.render()
    return self.screen

  def _step(self, action):
    self._screen, self.reward, self.done, _ = self.env.step(action)

  def _random_step(self):
    self._step(self.random_action)

  @ property
  def screen(self):
    gray = (0.2126 * self._screen[:,:,0] + 0.7152 * self._screen[:,:,1] + 0.0722 * self._screen[:,:,2])
    return scipy.misc.imresize(gray/255, self.dims)

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def random_action(self):
    return self.env.action_space.sample()

  @property
  def lives(self):
    return self.env.ale.lives()

  @property
  def state(self):
    return self.screen, self.reward, self.done

  def render(self):
    if self.is_display:
      self.env.render()

  def act(self, action, is_training=True):
    self._step(action)
    if is_training and self.done:
      self.reward -= 1
    self.render()
    return self.state
