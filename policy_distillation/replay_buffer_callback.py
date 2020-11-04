from stable_baselines.common.callbacks import BaseCallback
import numpy as np

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
  
  def sample(self):
    pass

  def add_to_buffer(self, item, item_type):
    self.buffer[item_type] = np.squeeze(np.array(item))

  def get_buffer(self):
    return self.buffer
  