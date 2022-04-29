import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import warnings
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
import argparse
import pickle
import joblib
# import jaydebeapi
import pandas.io.sql as pd_sql
from Utils.logger import Logger
from Utils.utils import *
from Utils.dataloader import *
from ML_Model.train_model import *
from ML_preprocessing.preprocessing import *


##python3 Optuna.py

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    target_name = TARGET
    dataset = DATA_TYPE

    log = Logger("Model_log")

     #### data load ####

    #     conn = jaydebeapi.connect('com.sybase.jdbc4.jdbc.SybDriver',
    #                           'jdbc:sybase:Tds:10.150.1.75:',
    #                           {'user':'aaa','password':'aaa'},
    #                           "/app/ml/jconn4.jar")
    #     conn.jconn.setAutoCommit(True)
    #     curs = conn.cursor()
    
    try:
        if dataset == 'SQL':
            log.info(f"DATA {target_name} load")
            read_sql = SQL_TRAIN
            data = pd_sql.read_sql(read_sql, conn, index_col=None)
        else:
            log.info(f"{READ_TRAIN_INFO} load")
            data = pd.read_csv(f"{READ_TRAIN_INFO}")
            
        if MODEL_PREDICTION_TYPE=='classification':
            log.info(f'target {target_name} : {round(data[target_name].value_counts()[1]/data.shape[0],3)*100}%')
    except Exception as e:
        log.error(f'{target_name} opt data load error : {e}')
    else:
        log.info(f'load Dataset : {target_name} opt')
        log.info(f'data shape : {data.shape}')
        
    ########## preprocessing #################
    
    try:
        data, key_data, cat_vars = Preprocessing(data, log,target_name)
    except Exception as e:
        log.error(f'{e} : {target_name} opt data preprocessing error')
    else:
        log.info(f'finish preprocessing : {target_name} opt')
    ##### model train #####
    ### Data cleaning ###
    try:
        train_x, train_y, valid_x, valid_y = Data_cleaning(data, target_name, log,  MODEL_PREDICTION_TYPE, cat_vars)
    except Exception as e:
        log.error(f'{e} : {target_name} opt data cleaning error')
    else:
        log.info(f'finish data cleaning : {target_name} opt')  
        
    ### Optuna objective function  ###
    def objective(X_train, X_valid, y_train, y_valid, prediction_type, cat_vars, trial : Trial) -> float :
        model_type = trial.suggest_categorical('model_type', ['XGB', 'LGB', 'CAT'])
        
        if model_type == "XGB":
            X_train = X_train.drop(columns=cat_vars)
            X_valid = X_valid.drop(columns=cat_vars)
        
        if prediction_type=='classification':
            if model_type == 'XGB':
                params_xgb = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.0003, 0.1),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1)
                    }
                model = XGBClassifier( ** params_xgb)
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    eval_metric = 'auc',
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                xgb_pred = model.predict(X_valid)
                loss = f1_score(y_valid, xgb_pred)
                return loss
            elif model_type == 'LGB':
                params_lgbm = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.0003, 0.1),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1)
                    }
                model = LGBMClassifier( ** params_lgbm )
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    eval_metric = 'auc',
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                lgb_pred = model.predict(X_valid)
                loss = f1_score(y_valid, lgb_pred)
                return loss
            elif model_type == 'CAT':
                params_cat = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.0003, 0.1),
                    'n_estimators' : 5000,
                    'cat_features' : cat_vars,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1),
                    'eval_metric' : 'AUC'
                    }
                model = CatBoostClassifier( ** params_cat )
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                cat_pred = model.predict(X_valid)
                loss = f1_score(y_valid, cat_pred)
                return loss
        elif prediction_type=='regression':
            if model_type == 'XGB':
                params_xgb = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.000003, 0.001),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1)
                    }
                model = XGBRegressor( ** params_xgb)
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    eval_metric = 'rmse',
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                xgb_pred = model.predict(X_valid)
                loss = -(mean_squared_error(y_valid, xgb_pred)**(0.5))
                return loss
            elif model_type == 'LGB':
                params_lgbm = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.000003, 0.001),
                    'n_estimators' : 5000,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_alpha' : trial.suggest_float('reg_alpha', 1e-8, 3e-3),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'max_bin' : trial.suggest_int('max_bin',2,100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1)
                    }
                model = LGBMRegressor( ** params_lgbm )
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    eval_metric = 'rmse',
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                lgb_pred = model.predict(X_valid)
                loss = -(mean_squared_error(y_valid, lgb_pred)**(0.5))
                return loss
            elif model_type == 'CAT':
                params_cat = {
                    'random_state' : 420,
                    'learning_rate' : trial.suggest_float('learning_rate', 0.0003, 0.1),
                    'n_estimators' : 5000,
                    'cat_features' : cat_vars,
                    'max_depth' : trial.suggest_int('max_depth', 3, 16),
                    'reg_lambda' : trial.suggest_float('reg_lambda', 1e-8, 9e-2),
                    'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
                    'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
                    'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0, 1),
                    'eval_metric' : 'RMSE'
                    }
                model = CatBoostRegressor( ** params_cat )
                model.fit(
                    X_train,
                    y_train,
                    eval_set = [(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds = 100,
                    verbose = 100
                )
                cat_pred = model.predict(X_valid)
                loss = -(mean_squared_error(y_valid, cat_pred)**(0.5))
                return loss
    try:
        ### Optuna Setting ###
        sampler = TPESampler()
        study = optuna.create_study(
        study_name = 'parameter_opt',
        direction = 'maximize', 
        sampler = sampler
        )
        ### Optuna Run ###

        study.optimize(lambda trial : objective(train_x, valid_x, train_y, valid_y, MODEL_PREDICTION_TYPE, cat_vars, trial), n_trials = 50)
        ### final output log ###
        log.info(f'Best Score : {study.best_value}')
        log.info(f'Best Trial : {study.best_trial.params}')

        ### Saving Best Parameters ###
        pd.DataFrame(list(study.best_trial.params.items())).set_index(0).to_csv(f'ML_Tuning/Best_Params_{target_name}.csv')
    except Exception as e:
        log.error(f'{e} : {target_name} opt parameter tuning error')
    else:
        log.info(f'finish parameter tuning : {target_name} opt')            
        

