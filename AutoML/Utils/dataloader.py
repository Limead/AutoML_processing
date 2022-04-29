import pandas as pd
import numpy as np
import json
from datetime import datetime

date = datetime.today().strftime('%Y%m')

with open(f'Utils/setting.json','r') as f:
    json_data = json.load(f)
    train_cols = ','.join(json_data['SQL_TRAIN_COLS'])
    key_cols = ','.join(json_data['KEY_COLS'])
    train_table_name = json_data['TRAIN_TABLE']
    test_table_name = json_data['TEST_TABLE']
    TARGET = json_data['TARGET']
    DATA_TYPE = json_data['DATA_TYPE']
    MODEL_PREDICTION_TYPE = json_data['MODEL_TYPE']
    
    KEYS = json_data['KEY_COLS']
    
    ## SQL 
    if DATA_TYPE == 'SQL': 
        SQL_TRAIN = f""" 
                         SELECT {key_cols},{train_cols},{target}
                         FROM {train_table_name}
                     """


        SQL_PROCESS = f""" 
                            SELECT {key_cols},{train_cols}
                            FROM {test_table_name}
                       """
    ## file
    elif DATA_TYPE == 'file':
        READ_TRAIN_INFO = f'{train_table_name}'
        
        READ_TEST_INFO = f'{test_table_name}'
