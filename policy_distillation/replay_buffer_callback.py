from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import pickle

class ReplayBufferCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.replay_buffer = ReplayBuffer()

    def get_buffer(self):
      return self.replay_buffer.get_buffer()

    def _on_training_end(self):
      """
      This event is triggered before exiting the `learn()` method.
      """
      self.replay_buffer.add_to_buffer(self.locals["mb_actions"], "actions")
      self.replay_buffer.add_to_buffer(self.locals["mb_obs"], "observations")
      self.replay_buffer.add_to_buffer(self.locals["mb_rewards"], "rewards")

class ReplayBuffer():
  def __init__(self):
    self.buffer = {}

  def load_from_file(self, path):
    file = open(path, "rb")
    self.buffer = pickle.load(file)
  
  def sample(self, batch_size):
    assert "actions" in self.buffer and "observations" in self.buffer and "rewards" in self.buffer
    assert self.buffer["actions"].shape[0] == self.buffer["observations"].shape[0] == self.buffer["rewards"].shape[0]
    idx = np.random.permutation(range(self.buffer["actions"].shape[0]))[:batch_size]
    return self.buffer["actions"][idx], self.buffer["observations"][idx], self.buffer["rewards"][idx]

  def add_to_buffer(self, item, item_type):
    self.buffer[item_type] = np.squeeze(np.array(item))

  def get_buffer(self):
    return self.buffer
  