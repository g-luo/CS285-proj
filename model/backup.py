def run_strategy_no_rebalance(df, unique_trade_date, training_window, validation_window, strategy, ticker="", policy="MlpPolicy", model_selected=None) -> None:
    print("============Start " + strategy + " Strategy============")
    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df.copy()
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    last_state = []
    testing = model_selected is not None
    for i in range(0, len(unique_trade_date), training_window + validation_window):
        print("============================================")
        ## initial state is empty
        initial = i == 0 

        # set the time range to [training window, validation window]
        if i + training_window + validation_window >= len(unique_trade_date):
          break

        start_train_date = unique_trade_date[i]
        end_train_date = unique_trade_date[i + training_window]
        start_val_date = end_train_date
        end_val_date = unique_trade_date[i + training_window + validation_window]

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        prev_start_train = start_train_date-training_window-validation_window
        historical_turbulence = df[(df.datadate<prev_start_train) & (df.datadate>=prev_start_train-63)]
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
        train = data_split(df, start=start_train_date, end=end_train_date)
        if strategy != 'multitask':
          env_train = DummyVecEnv([lambda: StockEnvTrade(train, 
                                                    previous_state=last_state,
                                                    turbulence_threshold=turbulence_threshold,
                                                    initial=initial)])
          env_train.reset()
        else:
          train_args = {
                          "df": train, 
                          "previous_state": last_state, 
                          "turbulence_threshold": turbulence_threshold, 
                          "initial": initial,
                          "iter": i
                        }
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", start_train_date, "to ", end_train_date)
        if strategy == 'PPO':
            if not testing:
              print("======PPO Training========")
              model_selected = train_PPO_update(model_selected, env_train, timesteps=100, policy=policy)
            else:
              print("======PPO Testing========")
              pass
        elif strategy =='sanity':
            pass
        elif strategy == 'DQN':
            if not testing:
              print("======DQN Training========")
              model_selected = train_DQN_update(model_selected, env_train, timesteps=100, policy=policy)
            else:
              print("======DQN Testing========")
              pass
        elif strategy == "multitask":
            if not testing:
              print("======Mulitask Training========")
              model_selected, last_state = train_multitask(model_selected, train_args, timesteps=10, policy=policy)
            else:
              print("======Multitask Testing========")
              pass
        elif strategy =="policy distillation":
            pass
        else:
            print("Model is not part of supported list. Please choose from following list for strategy [Ensemble, PPO, A2C, DDPG]")
            return
        ############## Training and Validation ends ##############    

        ############## Trading starts ##############   
        print("======Trading from: ", start_train_date, "to ", end_val_date)

        if strategy != "multitask":
          last_state = DRL_prediction_no_rebalance(df=df, model=model_selected, name=strategy+"_"+ticker,
                                              last_state=last_state,
                                              iter_num=i,
                                              start_date=start_train_date,
                                              end_date=end_val_date,
                                              turbulence_threshold=turbulence_threshold,
                                              initial=initial)
        if strategy == 'multitask' and testing:
          for ticker in df["tic"].unique():
            print("======Trading for : " + ticker + "======")
            ticker_df = df[df["tic"] == ticker]
            last_state = DRL_prediction(df=ticker_df, model=model_selected, name=strategy + "_" +ticker,
                                              last_state=last_state,
                                              iter_num=i,
                                              start_date=start_train_date,
                                              end_date=end_val_date,
                                              turbulence_threshold=turbulence_threshold,
                                              initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ############## 
    
    model_name = strategy + "_" + ticker
    if strategy != "policy distillation" and testing:
      model_selected.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    end = time.time()
    print(strategy + " Strategy took: ", (end - start) / 60, " minutes")

def run_strategy(df, unique_trade_date, rebalance_window, validation_window, strategy, model_selected=None, ticker="", policy_distillation_network=None, eval_mode=True) -> None:
    if strategy == 'Ensemble':
        run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window)
        return
    
    print("============Start " + strategy + " Strategy============")
    last_state = []
    sharpe_list = []
    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df.copy()
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
        env_train = DummyVecEnv([lambda: StockEnvTrade(train)])

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
          
            model_selected = model_ppo
        elif strategy =='policy distillation':
            model_ppo = train_PPO(env_train, model_name="PPO_" + ticker + "_{}".format(i), timesteps=50000)
            model_selected.update(model_ppo)
        elif strategy == "multitask":
            print("======Mulitask Training========")
            name="Multitasks_{}".format(i)
            model_multitask = train_multitask(train, model_name=name, timesteps=10)
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
          last_state = DRL_prediction(df=df, model=model_selected, name=strategy+"_"+ticker,
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
    if strategy !='policy distillation':
      model_selected.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    if (hasattr(model_selected, "custom_replay_buffer")):
      with open(f"{config.TRAINED_MODEL_DIR}/{model_name}"+"_buffer.pkl", "wb") as file:
        pickle.dump(model_selected.custom_replay_buffer, file)
    end = time.time()
    print(strategy + " Strategy took: ", (end - start) / 60, " minutes")

