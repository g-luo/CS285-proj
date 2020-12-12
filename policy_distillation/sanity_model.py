import random
import numpy as np
class SanityModel(object):
  def __init__(self, ac_dim):
    self.ac_dim = (1,ac_dim)
    print("sanity model ac dim")
    print(self.ac_dim)
    self.counter=0

  def predict(self, obs):
    r = self.counter%2
    if r==1:
      self.counter = self.counter+1
      return -1*np.ones(self.ac_dim), None
    else:
      self.counter = self.counter+1
      return 1*np.ones(self.ac_dim), None
    

  def save(self, *args):
    pass