import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from preprocessing.utils import series_to_list

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = 1
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001


class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0, turbulence_threshold=140,
                 initial=True, previous_state=[], model_name='', iteration=0, log_interval=-1):
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (6 * STOCK_DIM + 1,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False  
        self.turbulence_threshold = turbulence_threshold
        self.turbulence_threshold = float('inf')
        # initalize state
        self.reset_state([INITIAL_ACCOUNT_BALANCE], [0]*STOCK_DIM)

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.date_memory = [0]
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration
        self.log_interval=log_interval

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        # we can be in debt because there is no cashout (can't short more shares)
        if self.turbulence<self.turbulence_threshold:
            # case where we own stocks and can sell
            sell_stocks = min(abs(action), self.state[index+STOCK_DIM+1])
            if self.state[index+STOCK_DIM+1] > 0:
              self.state[0] += self.state[index+1] * sell_stocks * (1 - TRANSACTION_FEE_PERCENT)
              self.state[index+STOCK_DIM+1] -= sell_stocks
              self.cost += self.state[index+1] * sell_stocks * TRANSACTION_FEE_PERCENT
              self.trades +=1
            available_stocks = self.state[0] // self.state[index+1]
            short_stocks = min(abs(action) - sell_stocks, available_stocks)
            if short_stocks > 0:
                self.state[0] -= self.state[index+1] * short_stocks * (1 - TRANSACTION_FEE_PERCENT)
                self.state[index+STOCK_DIM+1] -= short_stocks
                self.cost += self.state[index+1] * short_stocks * TRANSACTION_FEE_PERCENT
                self.trades+=1
            else:
              pass
        else:
            # if turbulence goes over threshold, just clear out all positions
            # self.state[index + 1] is the price 
            # self.state[index + 1] is the number of stocks
            self.state[0] += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                          (1- TRANSACTION_FEE_PERCENT)
            self.state[index+STOCK_DIM+1] = 0
            self.cost += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                          TRANSACTION_FEE_PERCENT
            self.trades+=1
          
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            buy_stocks = min(abs(action), abs(self.state[index+STOCK_DIM+1]))
            if self.state[index+STOCK_DIM+1] < 0:
                self.state[0] += self.state[index+1] * buy_stocks * (1 + TRANSACTION_FEE_PERCENT)
                self.state[index+STOCK_DIM+1] += buy_stocks
                self.cost += self.state[index+1] * buy_stocks * TRANSACTION_FEE_PERCENT
                self.trades += 1

            available_stocks = self.state[0] // self.state[index+1]
            num_stocks = min(abs(action) - buy_stocks, available_stocks)
            if num_stocks > 0:
                #update balance
                self.state[0] -= self.state[index+1] * num_stocks * (1 + TRANSACTION_FEE_PERCENT)
                self.state[index+STOCK_DIM+1] += num_stocks
                self.cost += self.state[index+1] * num_stocks * TRANSACTION_FEE_PERCENT
                self.trades += 1
        else:
            # if turbulence goes over threshold, just stop buying
            pass
        
    # The inputted df must have at least two days to get the change
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal: 
            results_output = pd.DataFrame({"date":self.date_memory, "assets":self.asset_memory})
            results_output.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            results_output = pd.DataFrame({"date":self.date_memory[1:], "rewards":self.rewards_memory})
            results_output.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            return self.state, self.reward, self.terminal,{}
        else:
            actions = actions * HMAX_NORMALIZE

            # Clear all positions
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
                
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = series_to_list(self.data['turbulence'])[0]
            
            # set the state with the new positions
            self.reset_state([self.state[0]], list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            print(self.state)
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            if self.log_interval > 0 and self.iteration % self.log_interval == 0:
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                print("=======================Iteration {}================================".format(self.iteration))
                print("previous_total_asset:{}".format(self.asset_memory[-1]))
                print("end_total_asset:{}".format(end_total_asset))
                print("total_reward:{}".format(end_total_asset - self.asset_memory[0]))
                print("total_cost: ", self.cost)
                print("total trades: ", self.trades)

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.date_memory.append(self.data.loc["datadate"])
            self.asset_memory.append(end_total_asset)

        self.iteration += 1
        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.date_memory = [0]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            self.reset_state([INITIAL_ACCOUNT_BALANCE], [0]*STOCK_DIM)
          
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory = [previous_total_asset]
            self.date_memory = [0]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False 
            self.rewards_memory = []

            self.reset_state([self.previous_state[0]], self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)])
            
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state
    
    def _seed(self, seed=1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset_state(self, balance, stocks):
      self.state = balance + \
                      series_to_list(self.data.adjcp) + \
                      stocks + \
                      series_to_list(self.data.macd) + \
                      series_to_list(self.data.rsi) + \
                      series_to_list(self.data.cci) + \
                      series_to_list(self.data.adx)
                      