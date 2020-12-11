class SanityModel(object):
  def __init__(self, ac_dim):
    self.ac_dim = ac_dim

  def predict(self, obs):
    