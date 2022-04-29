import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import pickle
import joblib
#import jaydebeapi
import pandas.io.sql as pd_sql
from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, recall_score, precision_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split

from Utils.logger import Logger
from Utils.utils import *


def Model_processing(data, log, model_type, prediction_type, target_name, key_data):
    ### 모델 load
    loaded_model = joblib.load(open(f'ML_Model/saved_model/{model_type}_{target_name}_model.pkl', 'rb'))

    if prediction_type=='classification':
        ### 해촉확률 예측 
        pred_prob = loaded_model.predict_proba(data)[:, 1]

    elif prediction_type=='regression':
        pred_prob = loaded_model.predict(data)
        
    #### 최종 아웃풋 # 
    data['model_prediction'] = pred_prob
    
    data = pd.concat([key_data,data], axis=1)
    ## data 파일 저장 방식
    data.to_csv(f'ML_result/final_{target_name}.csv', index=False, encoding='euc-kr')
    
