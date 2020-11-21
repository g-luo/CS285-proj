from policy_distillation.policy_distillation import PolicyDistillation
from stable_baselines import PPO2
import torch
import os

def train_PPO_models(stocks = ['./data/TSLA.csv', './data/FB.csv'],  
                      tickers = ['TSLA', 'FB'], start_date=20130102, 
                      end_date=20180101):
  
 
  teachers = []
  envs = []
  for i in range(len(stocks)):
    print(i)
    df = process_yahoo_finance(stocks[i], tickers[i])
    train = data_split(df, start=start_date, end=end_date)
    
    env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
    model = train_PPO(env_train, model_name='PPO_'+tickers[i])
    teachers.append(model)
    envs.append(env_train)
  return teachers, envs

def load_models(model_path="/content/DeepRL4Stocks/trained_models"):
  files = os.listdir(model_path)
  models = []
  for f in files:
    if f.split("_")[0] =="PPO":
       models.append(load_model(model_path+f, "PPO"))
  return models

def load_model(path, model_type):

  if model_type=="PPO":
    model = PPO2.load(path)
  else:
    raise NotImplementedError 
  return model

def train_policy_distillation(teacher_models, envs):
  for env in envs:
    env.normalize_obs = lambda x: x
    env.normalize_reward = lambda x: x
  policy_distillaion = PolicyDistillation()
  student_network = policy_distillaion.run_policy_distillation(teacher_models, envs)
  return student_network

def save_policy_distillation_network(student_network, path="/content/DeepRL4Stocks/trained_models/student_network")
  torch.save(student_network.state_dict(), path)

def load_policy_distillation_network(path="/content/DeepRL4Stocks/trained_models/student_network"):
  network = StudentPolicy(
          ob_dim = envs[0].observation_space.shape[0],
          ac_dim = envs[0].action_space.shape[0],
          n_layers=2, 
          size=64, 
          learning_rate=1e-4
          )
  network.load_state_dict(torch.load("/content/DeepRL4Stocks/trained_models/student_network"))
  return network