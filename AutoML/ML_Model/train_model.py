import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import plot_importance
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, recall_score, precision_score, precision_recall_curve, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from Utils.logger import Logger
from Utils.utils import *

def Data_cleaning(data, target, log, prediction_type, cat_vars, processing=False):

    
    if processing:
        NA_cols = [col if data[col].isnull().sum() != 0 else 'N' for col in data]
        NA_cols = list(set(NA_cols) - set(cat_vars))
        NA_cols.remove('N')
        for cols in NA_cols:
            data[cols] = data[cols].fillna(0)
        for cat_cols in cat_vars:
            data[cat_cols] = data[cat_cols].cat.add_categories(['NAN']).fillna("NAN")
        log.info(f'processing_NA_cols : {data.shape}')
        
        return data
    else:
        NA_cols = [col if data[col].isnull().sum() != 0 else 'N' for col in data]
        NA_cols = list(set(NA_cols) - set(cat_vars))
        NA_cols.remove('N')
        for cols in NA_cols:
            data[cols] = data[cols].fillna(0)
        for cat_cols in cat_vars:
            data[cat_cols] = data[cat_cols].cat.add_categories(['NAN']).fillna("NAN")
        log.info(f'processing_NA_cols : {data.shape}')

        x_data = data.drop(columns=target)
        y_data = data[target]
        
        if prediction_type=='classification':
            train_x, valid_x, train_y, valid_y = train_test_split(x_data, y_data, test_size=0.3, stratify = y_data, random_state=123)
        elif prediction_type=='regression':
            train_x, valid_x, train_y, valid_y = train_test_split(x_data, y_data, test_size=0.3, random_state=123)


        return train_x, train_y, valid_x, valid_y
     
def Model_train(train_x, train_y, valid_x, valid_y, log, model_type, prediction_type, cat_vars, target_name):

    if model_type == "XGB":
        train_x = train_x.drop(columns=cat_vars)
        valid_x = valid_x.drop(columns=cat_vars)
    # Parameter set loading
    if prediction_type=='classification':
        try:
            Params = pd.read_csv(f'ML_Tuning/Best_Params_{target_name}.csv')
            Params = Params.set_index('0').to_dict()['1']
            del Params['model_type']
        except Exception as e:
            if model_type == 'LGB':
                best_model = LGBMClassifier(
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'auc', early_stopping_rounds=100)
            elif model_type == 'XGB':
                best_model = XGBClassifier(
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'auc', early_stopping_rounds=100)
            elif model_type == 'CAT':
                best_model = CatBoostClassifier(cat_features = cat_vars,
                                        n_estimators = 5000,
                                        random_state = 420,
                                        eval_metric = 'AUC')
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], early_stopping_rounds=100)
        else:
            if model_type == 'LGB':
                Params['max_bin'] = int(Params['max_bin'])
                Params['max_depth'] = int(Params['max_depth'])
                Params['min_child_samples'] = int(Params['min_child_samples'])
                Params['num_leaves'] = int(Params['num_leaves'])
                best_model = LGBMClassifier( ** Params,
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'auc', early_stopping_rounds=100)
            elif model_type == 'XGB':
                Params['max_bin'] = int(Params['max_bin'])
                Params['max_depth'] = int(Params['max_depth'])
                best_model = XGBClassifier( ** Params,
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'auc', early_stopping_rounds=100)
            elif model_type == 'CAT':
                Params['min_child_samples'] = int(Params['min_child_samples'])
                Params['max_depth'] = int(Params['max_depth'])
                best_model = CatBoostClassifier( ** Params,
                                        cat_features = cat_vars,
                                        n_estimators = 5000,
                                        random_state = 420,
                                        eval_metric = 'AUC')
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], early_stopping_rounds=100)
        val_pred = best_model.predict(valid_x)
        val_pred_proba = best_model.predict_proba(valid_x)
        ## Score log

        log.info(f'Accuracy : {accuracy_score(valid_y, val_pred)} / train : {accuracy_score(train_y, best_model.predict(train_x))}')
        log.info(f'AUC Score : {roc_auc_score(valid_y, val_pred_proba[:,1])} / train : {roc_auc_score(train_y, best_model.predict_proba(train_x)[:,1])}')
        log.info(f'F1 Score : {f1_score(valid_y, val_pred)} / train : {f1_score(train_y, best_model.predict(train_x))}')

        ## P-R Curve Visualization
        train_pre, train_rec, _ = precision_recall_curve(train_y, best_model.predict_proba(train_x)[:,1])
        train_aucpr = auc(train_rec, train_pre)
        precision, recall, _ = precision_recall_curve(valid_y, val_pred_proba[:,1])
        pr_auc = auc(recall, precision)
        PR = plt.figure()
        lw =2
        plt.plot(recall, precision, color ='darkorange', lw=lw, label= 'P-R curve (area = %0.3f)' % pr_auc)

        plt.xlim([0.24,1.005])
        plt.ylim([0,1.05])
        plt.title(f'{model_type} P-R curve count : {len(valid_y)}')
        plt.legend(loc='lower left')
        PR.savefig(f'ML_result/OUT_PR_curve_{model_type}_{target_name}.png')
        plt.close()
        log.info(f'AUC P-R : {pr_auc} / train : {train_aucpr}')
        ## ROC Curve Visualization
        ROC = plt.figure()
        fper, tper, thresholds = roc_curve(valid_y, val_pred_proba[:,1])
        plt.plot(fper, tper, color='blue', lw=lw, label = 'AUC curve (area = %0.3f)' % roc_auc_score(valid_y,val_pred_proba[:,1]))
        plt.plot([0,1],[0,1], color='green', linestyle='--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('Receiver Operating Curve')
        plt.legend(loc = 'lower right')
        ROC.savefig(f'ML_result/OUT_ROC_curve_{model_type}_{target_name}.png')
        plt.close()

        importance_plot = plt.figure(figsize=(15,6))
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[-28:]
        plt.barh(range(len(sorted_idx)),
                importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)),
                  np.array(valid_x.columns)[sorted_idx])
        plt.title('Feature Importance')

        importance_plot.savefig(f'ML_result/importance_{model_type}_{target_name}.png')
        plt.close()

        score_df = pd.DataFrame({'Accuracy' : [accuracy_score(valid_y, val_pred)],
                             'AUC' : [roc_auc_score(valid_y, val_pred_proba[:,1])],
                             'F1' : [f1_score(valid_y, val_pred)],
                             'PR-AUC' : [pr_auc]})
        score_df.to_csv(f'ML_result/score_{model_type}_{target_name}.csv', index=False,encoding='euc-kr')
    elif prediction_type=='regression':
        try:
            Params = pd.read_csv(f'ML_Tuning/Best_Params_{target_name}.csv')
            Params = Params.set_index('0').to_dict()['1']
            del Params['model_type']
        except Exception as e:
            if model_type == 'LGB':
                best_model = LGBMRegressor(
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'rmse', early_stopping_rounds=100)
            elif model_type == 'XGB':
                best_model = XGBRegressor(
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'rmse', early_stopping_rounds=100)
            elif model_type == 'CAT':
                best_model = CatBoostRegressor(cat_features = cat_vars,
                                        n_estimators = 5000,
                                        random_state = 420,
                                        eval_metric = 'RMSE')
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], early_stopping_rounds=100)
        else:
            if model_type == 'LGB':
                Params['max_bin'] = int(Params['max_bin'])
                Params['max_depth'] = int(Params['max_depth'])
                Params['min_child_samples'] = int(Params['min_child_samples'])
                Params['num_leaves'] = int(Params['num_leaves'])
                best_model = LGBMRegressor( ** Params,
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'rmse', early_stopping_rounds=100)
            elif model_type == 'XGB':
                Params['max_bin'] = int(Params['max_bin'])
                Params['max_depth'] = int(Params['max_depth'])
                best_model = XGBRegressor( ** Params,
                                        n_estimators = 5000,
                                        random_state = 420)
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], eval_metric = 'rmse', early_stopping_rounds=100)
            elif model_type == 'CAT':
                Params['min_child_samples'] = int(Params['min_child_samples'])
                Params['max_depth'] = int(Params['max_depth'])
                best_model = CatBoostRegressor( ** Params,
                                        cat_features = cat_vars,
                                        n_estimators = 5000,
                                        random_state = 420,
                                        eval_metric = 'RMSE')
                best_model.fit(train_x, train_y, verbose=50, eval_set = [(valid_x, valid_y)], early_stopping_rounds=100)
            
        val_pred = best_model.predict(valid_x)
        log.info(f'MSE : {mean_squared_error(valid_y, val_pred)} / train : {mean_squared_error(train_y, best_model.predict(train_x))}')
        log.info(f'MAE : {mean_absolute_error(valid_y, val_pred)} / train : {mean_absolute_error(train_y, best_model.predict(train_x))}')
        importance_plot = plt.figure(figsize=(20,6))
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[-28:]
        plt.barh(range(len(sorted_idx)),
                importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)),
                  np.array(valid_x.columns)[sorted_idx])
        plt.title('Feature Importance')

        importance_plot.savefig(f'ML_result/importance_{model_type}_{target_name}.png', bbox_inches='tight')
        plt.close()
        score_df = pd.DataFrame({'RMSE' : [mean_squared_error(valid_y, val_pred)**(0.5)],
                             'MAE' : [mean_absolute_error(valid_y, val_pred)]})
        score_df.to_csv(f'ML_result/score_{model_type}_{target_name}.csv', index=False,encoding='euc-kr')
        
    shap.initjs()
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(valid_x)
    
    shap_plot = plt.figure(figsize=(15,6))
    if model_type == 'LGB':
        try:
            shap.summary_plot(shap_values, valid_x, feature_names = valid_x.columns, show=False)
        except:
            shap.summary_plot(shap_values[1], valid_x, feature_names = valid_x.columns, show=False)
    else:
        shap.summary_plot(shap_values, valid_x, feature_names = valid_x.columns, show=False)

    shap_plot.savefig(f'ML_result/SHAP_plot_{model_type}_{target_name}.png', bbox_inches='tight')
    plt.close()
    ## return model
    return best_model
