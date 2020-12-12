# common library
import pandas as pd
import numpy as np
import time
import gym
import pickle
import math

# RL models from stable-baselines
from stable_baselines import HER
from stable_baselines import SAC
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines import DQN
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

from policy_distillation.replay_buffer_callback import ReplayBufferCallback
from preprocessing.preprocessors import process_yahoo_finance, data_split

# customized env
from env.EnvMultipleStock_trade import StockEnvTrade

def train_SAC(env_train, model_name, timesteps=50000):
    start = time.time()
    model = SAC('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (SAC): ', (end - start) / 60, ' minutes')
    return model

def train_HER(env_train, model_name, timesteps=50000):
    start = time.time()
    n_sampled_goal = 4
    goal_selection_strategy = 'future'
    model = HER('MlpPolicy', env_train, model_class=SAC, verbose=0, n_sampled_goal = n_sampled_goal,
                goal_selection_strategy=goal_selection_strategy)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (HER): ', (end - start) / 60, ' minutes')
    return model


def train_DQN(env_train, model_name, timesteps=50000):
    start = time.time()
    model = DQN('MlpPolicy', env_train, verbose=1)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DQN): ', (end - start) / 60, ' minutes')
    return model


def train_A2C(env_train, model_name, timesteps=50000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_TD3(env_train, model_name, timesteps=50000):
    """TD3 model"""

    start = time.time()
    model = TD3('MlpPolicy', env_train)
    model.learn(total_timesteps=timesteps, log_interval=10)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train)
    replay_buffer_callback = ReplayBufferCallback()
    model.learn(callback=replay_buffer_callback, total_timesteps=timesteps)
    end = time.time()

    model.custom_replay_buffer = replay_buffer_callback.get_replay_buffer()
    # model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    # with open(f"{config.TRAINED_MODEL_DIR}/{model_name}"+"_buffer.pkl", "wb") as file:
    #   pickle.dump(replay_buffer_callback.get_buffer(), file)

    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_PPO_update(initial_model, env_train, timesteps=10, policy="MlpPolicy"):
    model = initial_model
    if initial_model == None:
        model = PPO2(policy, env_train)
        #model.callback = ReplayBufferCallback()
    model.set_env(env_train)
    model.learn(total_timesteps=timesteps)
    return model

def train_multitask(df, unique_trade_date, timesteps=10, policy="MlpPolicy", model_name="multitask"):
  # df of all intermixed values
  # get out the individual tickers and switch out the dates
  # timesteps = num training steps per date
  start = time.time()
  df = data_split(df, start=unique_trade_date[0], end=unique_trade_date[len(unique_trade_date)-1])
  last_state, initial = [], True
  model = None
  for i in range(len(unique_trade_date) - 2):
    for ticker in df["tic"].unique():
      # Interval is every two days so we can optimize on the change in account value
      start_date, end_date = unique_trade_date[i], unique_trade_date[i+2]
      quanta_df = data_split(df, start=start_date, end=end_date)
      quanta_df = quanta_df[quanta_df["tic"] == ticker]
      if len(quanta_df.index) < 2:
        continue
      quanta_df = quanta_df.reset_index()
      quanta_env = DummyVecEnv([lambda: StockEnvTrade(quanta_df, 
                                                        previous_state=last_state, 
                                                        initial=initial, 
                                                        log_interval=1
                                                      )])
      quanta_env.reset()
      model = train_PPO_update(model, quanta_env, timesteps, policy=policy)
      last_state = quanta_env.render()
    initial = False

  model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
  end = time.time()
  print('Training time (Multitask): ', (end - start) / 60, ' minutes')
  return model

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model### 

    ## trading env
    print("DRLPREDICTION")
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state

def DRL_prediction_no_rebalance(df,
                   model,
                   name,
                   unique_trade_date,
                   log_interval):
    ### make a prediction based on trained model### 

    ## trading env
    print("DRL PREDICTION NO REBALANCE")
    all_data = data_split(df, start=unique_trade_date[0], end=unique_trade_date[len(unique_trade_date)-1])
    for ticker in all_data["tic"].unique():      
        trade_data = all_data[all_data["tic"] == ticker]
        env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                      initial=True,
                                                      model_name=name+"_"+ticker, 
                                                      log_interval=log_interval)])
        obs_trade = env_trade.reset()
        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(obs_trade) 
            print("action: ", action)
            obs_trade, rewards, dones, info = env_trade.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = env_trade.render()

    # df_last_state = pd.DataFrame({'last_state': last_state})
    # df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state

def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    if df_total_value['daily_return'].mean() == 0 or df_total_value['daily_return'].std() == 0:
        return 0
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


