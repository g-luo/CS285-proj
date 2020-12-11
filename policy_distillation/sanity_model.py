import random
import numpy as np
class SanityModel(object):
  def __init__(self, ac_dim):
    self.ac_dim = (1,ac_dim)
    print("sanity model ac dim")
    print(self.ac_dim)

  def predict(self, obs):
    r = random.randint(0,1)
    if r==1:
      return -1*np.ones(self.ac_dim), None
    else:
      return np.ones(self.ac_dim), None
  def save(self, *args):
    pass