# every episode, switch the replay buffer
# train a neural network 
# use a special loss function
# https://github.com/jetsnguns/realtime-policy-distillation 
# https://github.com/decisionforce/ESPD

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN

from torch import optim
from torch import nn
from torch import distributions
import torch
from preprocessing import utils
import numpy as np
import time 

class PolicyDistillation(object):

    def run_policy_distillation(self, teachers, envs):
      start = time.time()
      # just learn one by one
      student_network = StudentPolicy(
          ob_dim = envs[0].observation_space.shape[0],
          ac_dim = envs[0].action_space.shape[0],
          n_layers=2, 
          size=64, 
          learning_rate=1e-4
      )
      for i in range(len(teachers)):
        student_network.update(teachers[i], envs[i])
           
      end = time.time()
      print('Training time: ', (end - start) / 60, ' minutes')
      return student_network
      # somehow pickle and save the model

class StudentPolicy(object):
    def __init__(self,
                 ob_dim, 
                 ac_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4
                 ):
        super().__init__()
        # init vars
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.size = size
        print(self.ob_dim)
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.logits_na = utils.build_mlp(input_size=self.ob_dim,
                                        output_size=self.ac_dim,
                                        n_layers=self.n_layers,
                                        size=self.size)
        self.logits_na.to(utils.device)
        self.optimizer = optim.Adam(self.logits_na.parameters(),
                                    self.learning_rate)

    def predict(self, obs: np.ndarray) -> np.ndarray:
      if len(obs.shape) > 1:
        observation = obs
      else:
        observation = obs[None]
      observation = utils.from_numpy(observation)
      action_distribution = self(observation)
      action = action_distribution.sample()
      return utils.to_numpy(action)
    
    def forward(self, observation: torch.FloatTensor):
      """
        The action space is discrete.
      """
      logits = self.logits_na(observation)
      action_distribution = distributions.Categorical(logits=logits)
      return action_distribution
      
    def update(self, teacher_model, env):
      # set the batch size to the buffer size to train sequentially
      batch_size = teacher_model.replay_buffer.buffer_size
      obs, acs, rews, obs_tp1, dones = teacher_model.replay_buffer.sample(batch_size, env=env)
      teacher_actions = teacher_model.action_probability(obs, actions=acs)

      observations = utils.from_numpy(obs)
      actions = utils.from_numpy(acs)

      student_distribution = self.forward(observations).log_prob(actions)
      output = nn.KLDivLoss(student_distribution, teacher_actions, log_target=True)
      self.optimizer.zero_grad()
      output.backward()
      self.optimizer.step()
        



