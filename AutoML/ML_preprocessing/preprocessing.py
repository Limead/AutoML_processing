import sys
import os
import pandas as pd
import numpy as np
from kozip import KoZIP
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Utils.utils import *
from Utils.dataloader import *

## 전처리 함수
def Preprocessing(data, log, target_name, processing=False):
  
    ## 타겟 분리
    try:
        if processing:
            pass
        else:
            target = data[target_name]
            data = data.drop(columns=[target_name])
    except Exception as e:
        log.warning(f'not in target : {e}')
    else:
        log.info(f' target 정보 분리 : {data.shape}')
        
    ## key 정보 분리
    try:
        key_data = data[KEYS]
        drop_cols = KEYS
        data = data.drop(columns=drop_cols)
    except Exception as e:
        log.warning(f'not in key : {e}')
    else:
        log.info(f'key 정보 분리 : {data.shape}')
        
    ## 카테고리 변수 변환
    try:
        cat_vars = list(data.select_dtypes(include='object').columns)

        data[cat_vars] = data[cat_vars].astype('category')

    except Exception as e:
        log.warning(f'category feature error : {e}')
    else:
        log.info(f'카테고리 변수 처리 수행 : {data.shape}')

    ## 전처리 후 최종 target 재결합
    if processing:
        pass
    else:
        data[f'{target_name}'] = target
    
    return data, key_data, cat_vars
