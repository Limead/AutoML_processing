import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Utils.utils import *
from Utils.dataloader import *


def Preprocessing(data, log, target_name, processing=False):
  
    try:
        if processing:
            pass
        else:
            target = data[target_name]
            data = data.drop(columns=[target_name])
    except Exception as e:
        log.warning(f'not in target : {e}')
    else:
        log.info(f' target drop : {data.shape}')
        
    try:
        key_data = data[KEYS]
        drop_cols = KEYS
        data = data.drop(columns=drop_cols)
    except Exception as e:
        log.warning(f'not in key : {e}')
    else:
        log.info(f'key drop : {data.shape}')

    try:
        cat_vars = list(data.select_dtypes(include='object').columns)

        data[cat_vars] = data[cat_vars].astype('category')

    except Exception as e:
        log.warning(f'category feature error : {e}')
    else:
        log.info(f'processing category feature : {data.shape}')
    if processing:
        pass
    else:
        data[f'{target_name}'] = target
    
    return data, key_data, cat_vars
