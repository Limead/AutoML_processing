import warnings
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
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

##python3 model_train.py

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    target_name = TARGET
    dataset = DATA_TYPE
    try:
        Params = pd.read_csv(f'ML_Tuning/Best_Params_{target_name}.csv')
        Params = Params.set_index('0').to_dict()['1']
        model_type = Params['model_type']
    except:
        model_type = 'LGB'
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
        data, key_data, cat_vars = Preprocessing(data, log, target_name)
    except Exception as e:
        log.error(f'{e} : {target_name} data preprocessing error')
    else:
        log.info(f'finish preprocessing : {target_name}')
    ##### model train #####

    ### Data cleaning ###
    try:
        train_x, train_y, valid_x, valid_y = Data_cleaning(data, target_name, log, MODEL_PREDICTION_TYPE, cat_vars) 
    except Exception as e:
        log.error(f'{e} : {target_name} data cleaning error')
    else:
        log.info(f'finish data cleaning : {target_name}')        
    ### training ###
    try:
        best_model = Model_train(train_x, train_y, valid_x, valid_y, log, model_type, MODEL_PREDICTION_TYPE, cat_vars, target_name) 

    except Exception as e:
        log.error(f'{e} : {target_name} model training error')
    else:
        log.info(f'finish model training : {target_name}')   
        
    date = datetime.today().strftime('%Y%m%d')
    ### model save ###
    joblib.dump(best_model, open(f'ML_Model/saved_model/{model_type}_{target_name}_model.pkl', 'wb'))
    joblib.dump(best_model, open(f'ML_Model/saved_model/backup/{model_type}_{target_name}_model_{date}.pkl', 'wb'))
