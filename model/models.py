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
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

def run_student(df, unique_trade_date, rebalance_window, validation_window, student_model, strategy='Student'):
    
    print("============Start " + strategy + " Strategy============")
    last_state = []
    sharpe_list = []
    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window-63]))]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)   

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile, 
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold 
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk 
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Training and Validation ends ##############    

        ############## Trading starts ##############    
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])

        last_state = DRL_prediction(df=df, model=student_model, name=strategy,
                                            last_state=last_state, iter_num=i,
                                            unique_trade_date=unique_trade_date,
                                            rebalance_window=rebalance_window,
                                            turbulence_threshold=turbulence_threshold,
                                            initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############    

    end = time.time()
    print(strategy + " Strategy took: ", (end - start) / 60, " minutes")

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

def my_DRL_prediction(df,
                   model,
                   name,
                   start_date,
                   end_date,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model### 

    ## trading env
    trade_data = data_split(df, start=start_date, end=end_date)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data)])
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


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window-63]))]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)   

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile, 
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold 
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk 
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_10k_dow_{}".format(i), timesteps=10000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=50000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        ############## Training and Validation ends ##############    

        ############## Trading starts ##############    
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)

        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############    

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

def run_strategy(df, unique_trade_date, rebalance_window, validation_window, strategy) -> None:
    if strategy == 'Ensemble':
        run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window)
        return
    
    print("============Start " + strategy + " Strategy============")
    last_state = []
    sharpe_list = []
    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    model_selected = None
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window-63]))]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)   

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile, 
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold 
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk 
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
#         validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
#                                 end=unique_trade_date[i - rebalance_window])
#         env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
#                                                           turbulence_threshold=turbulence_threshold,
#                                                           iteration=i)])
#         obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        if strategy == 'PPO':
            print("======PPO Training========")
            model_ppo = train_PPO(env_train, model_name="PPO_" + ticker + "_{}".format(i), timesteps=50000)
            # print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
            #     unique_trade_date[i - rebalance_window])
            # DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
            # sharpe_ppo = get_validation_sharpe(i)
            # print("PPO Sharpe Ratio: ", sharpe_ppo)
            # sharpe_list.append(sharpe_ppo)
            model_selected = model_ppo

        # elif strategy == 'A2C':
        #     print("======A2C Training========")
        #     model_a2c = train_A2C(env_train, model_name="A2C_10k_dow_{}".format(i), timesteps=10000)
        #     print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #         unique_trade_date[i - rebalance_window])
        #     DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        #     sharpe_a2c = get_validation_sharpe(i)
        #     print("A2C Sharpe Ratio: ", sharpe_a2c)
        #     sharpe_list.append(sharpe_a2c)
        #     model_selected = model_a2c

        # elif strategy == 'DDPG': 
        #     print("======DDPG Training========")
        #     model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        #     print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #         unique_trade_date[i - rebalance_window])
        #     DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        #     sharpe_ddpg = get_validation_sharpe(i)
        #     print("DDPG Sharpe Ratio: ", sharpe_ddpg)
        #     sharpe_list.append(sharpe_ddpg)
        #     model_selected = model_ddpg
        elif strategy == "multitask":
              print("======Mulitask Training========")
              name="Multitasks_{}".format(i)
              model_multitask = train_multitask(train, model_name=name, timesteps=20000)
              # model_multitask.save(f"{config.TRAINED_MODEL_DIR}/{name}")
              model_selected = model_multitask
        else:
            print("Model is not part of supported list. Please choose from following list for strategy [Ensemble, PPO, A2C, DDPG]")
            return

        model_use.append(strategy)
        ############## Training and Validation ends ##############    

        ############## Trading starts ##############    
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])

        if strategy != "multitask":
          last_state = DRL_prediction(df=df, model=model_selected, name=strategy,
                                              last_state=last_state, iter_num=i,
                                              unique_trade_date=unique_trade_date,
                                              rebalance_window=rebalance_window,
                                              turbulence_threshold=turbulence_threshold,
                                              initial=initial)
        else:
          for ticker in df["tic"].unique():
            print("======Trading for : " + ticker + "======")
            ticker_df = df[df["tic"] == ticker]
            last_state = DRL_prediction(df=ticker_df, model=model_selected, name=strategy + "_" +ticker,
                                              last_state=last_state, iter_num=i,
                                              unique_trade_date=unique_trade_date,
                                              rebalance_window=rebalance_window,
                                              turbulence_threshold=turbulence_threshold,
                                              initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ############## 
    model_name = strategy + "_" + ticker
    model_selected.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    if (hasattr(model_selected, "custom_replay_buffer")):
      with open(f"{config.TRAINED_MODEL_DIR}/{model_name}"+"_buffer.pkl", "wb") as file:
        pickle.dump(model_selected.custom_replay_buffer, file)
    end = time.time()
    print(strategy + " Strategy took: ", (end - start) / 60, " minutes")

def train_multitask(df, model_name, timesteps=50000):
  # df of all intermixed values
  # get out the individual tickers and switch out the dates
  # use all unique timesteps
  model = None
  for date in df["datadate"].unique():
    for ticker in df["tic"].unique():
      quanta_df = df[df["datadate"] == date]
      quanta_df = quanta_df[quanta_df["tic"] == ticker]
      if quanta_df.empty:
        continue
      quanta_df = quanta_df.reset_index()
      quanta_env = DummyVecEnv([lambda: StockEnvTrain(quanta_df)])
      model = train_PPO_multitask(model, quanta_env)
  return model

def train_PPO_multitask(initial_model, env_train, timesteps=10):
  start = time.time()
  model = initial_model
  if initial_model == None:
    model = PPO2('MlpPolicy', env_train)
    #model.callback = ReplayBufferCallback()
  model.set_env(env_train)
  model.learn(total_timesteps=timesteps)
  end = time.time()
  return model
  
# def run_multitask(stocks, tickers, quanta, start, end, model_name, timesteps_per_quanta=100):
#   stock_dfs = []
#   for i in range(len(stocks)):
#     df = process_yahoo_finance(stocks[i], tickers[i])
#     stock_dfs.append(data_split(df, start=start, end=end))  
#   merged_stock_df = pd.concat(stock_dfs)

#   num_quanta = math.ceil((end - start) / quanta)
#   model = None
#   for t in range(num_quanta):
#     train_start, train_end = start + quanta * t, start + quanta * (t + 1)
#     quanta_df = merged_stock_df[(merged_stock_df.datadate >= train_start) & (merged_stock_df.datadate < train_end)]
#     quanta_df = quanta_df.reset_index()
#     if quanta_df.empty:
#       break
#     quanta_env = DummyVecEnv([lambda: StockEnvTrain(quanta_df)])
#     model = train_PPO_multitask(model, quanta_env, t, timesteps_per_quanta)
  
#   #model.custom_replay_buffer = replay_buffer_callback.get_replay_buffer()
#   model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#   #with open(f"{config.TRAINED_MODEL_DIR}/{model_name}"+"_buffer.pkl", "wb") as file:
#   #  pickle.dump(replay_buffer_callback.get_buffer(), file)

#   return model
