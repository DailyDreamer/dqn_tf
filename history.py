import numpy as np

class History:
  def __init__(self, history_length, screen_height, screen_width):
    self.history = np.zeros([history_length, screen_height, screen_width], np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def get(self):
    return self.history
  
  def empty(self):
    self.history *= 0