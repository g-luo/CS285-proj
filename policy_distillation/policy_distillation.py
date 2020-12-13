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
          n_layers=3, 
          size=64, 
          learning_rate=0.005
      )
      num_epochs = 500
      for _ in range(num_epochs):
        for i in range(len(teachers)):
          student_network.update(teachers[i])
            
      end = time.time()
      print('Training time: ', (end - start) / 60, ' minutes')
      return student_network
      # somehow pickle and save the model

class StudentPolicy(nn.Module):
    def __init__(self,
                 ob_dim, 
                 ac_dim,
                 n_layers,
                 size,
                 learning_rate=0.005
                 ):
        super().__init__()
        # init vars
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.size = size
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
      actions = self(observation)
      
      return utils.to_numpy(actions), None
    
    def forward(self, observation: torch.FloatTensor):
      """
        The action space is discrete.
      """
      logits = self.logits_na(observation)
      
      return logits
      
    def update(self, teacher_model):
      # set the batch size to the buffer size to train sequentially
      batch_size = 32
      acs, obs, rews = teacher_model.custom_replay_buffer.sample(32)
      observations = utils.from_numpy(obs)
      actions = utils.from_numpy(acs)
      teacher_actions, _ = teacher_model.predict(observations)
      
      teacher_actions= utils.from_numpy(teacher_actions)
      student_actions = self.forward(observations)
      self.optimizer.zero_grad()
      loss = nn.functional.mse_loss(student_actions, teacher_actions, reduction='sum')
      
      loss.backward()
      self.optimizer.step()
        



