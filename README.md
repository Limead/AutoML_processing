# AutoML_processing
Tree base ML processing
# OverView

Overview

Similar to the AutoML structure, the this structure allows the selection of model algorithms to calculate final output with minimal data settings.
I would like to share this data to help improve the efficiency of ML projects in the future.

# 📃 Contents

---
For detailed code, let's assume that you can check it out for yourself and review it, and we'll explain how to use it.

1. Data Definition

You must define the dataset to use.

It is configured to set and use various factors in json format inside the "Utiles/setting.json" file in the script.


- MODEL_TYPE : Value for setting classification or regression - [classification, regression ]
- DATA_TYPE : The value that sets the method for loading data - [SQL, file] 
    - When selecting the sql method, the db connect connection setting is required internally, and the variable list to select can be selected.
- TRAIN_TABLE : Training dataset,  If SQL is selected, enter a table name for the from clause, if you selected a file, enter a file name along with the path.
- TEST_TABLE : Testing dataset, If SQL is selected, enter a table name for the from clause, if you selected a file, enter a file name along with the path.
- TARGET : Target variable name.
- KEY_COLS : Enter a KEY variable, such as ID. These variables are excluded during learning.
- SQL_TRAIN_COLS : When select the sql type, you can specify the variables for the data to be selected. You do not enter the target and key_cols variables for that value.

2. Selecting and tuning algorithms

in the linux terminal, Run Optuna with the command "python Optuna.py" to select and tune algorithms.
XGBoost, LightGBM, and CatBoost selections are added to the Optuna factor to automatically select the right model among the three tree algorithms.

3. Model training

in the linux terminal, Run the command "python model_train.py" to train model
After learning, you can check the importance plot, ROC curve, PR curve, SHAP plot, and csv file containing model score values within the "ML_result" folder.

- Importance plot

![image](https://user-images.githubusercontent.com/97657857/165936059-6941a838-fa79-4705-84e0-018c6b14ad8e.png)


- ROC curve

![image](https://user-images.githubusercontent.com/97657857/165936122-3c304790-e812-4fed-a15e-b54d70df75db.png)


- SHAP plot

![image](https://user-images.githubusercontent.com/97657857/165936175-163ede4e-3e0f-4914-9495-0aeb4a5ec548.png)

4. Apply Model

in the linux terminal, Run the command "python model_test.py" to apply model prediction
When the prediction is complete, the final output file is created in the ML_result folder and the data with the model prediction added is generated.


# 🌈 Others

---

- env
    - linux python 3.6    
    - package list    
       
    alembic               1.7.7  
    attrs                 21.4.0  
    autopage              0.5.0  
    backcall              0.2.0  
    bayesian-optimization 1.2.0  
    catboost              1.0.4  
    certifi               2016.9.26  
    charset-normalizer    2.0.12  
    cliff                 3.10.1  
    cloudpickle           2.0.0  
    cmaes                 0.8.2  
    cmd2                  2.4.1  
    colorlog              6.6.0  
    cycler                0.11.0  
    dataclasses           0.8  
    decorator             4.4.2  
    graphviz              0.19.1  
    greenlet              1.1.2  
    idna                  3.3  
    imageio               2.15.0  
    importlib-metadata    4.8.3  
    importlib-resources   5.4.0  
    ipython               7.16.3  
    ipython-genutils      0.2.0  
    jedi                  0.17.2  
    joblib                1.1.0  
    keras                 2.8.0  
    kiwisolver            1.3.1  
    kozip                 1.1.4  
    lightgbm              3.3.2  
    lime                  0.2.0.1  
    llvmlite              0.36.0  
    Mako                  1.1.6  
    MarkupSafe            2.0.1  
    matplotlib            3.3.4  
    networkx              2.5.1  
    numba                 0.53.1  
    numpy                 1.19.5  
    optuna                2.10.0  
    packaging             21.3  
    pandas                1.1.5  
    parso                 0.7.1  
    patsy                 0.5.2  
    pbr                   5.8.1  
    pexpect               4.8.0  
    pickleshare           0.7.5  
    Pillow                8.4.0  
    pip                   21.3.1  
    plotly                5.6.0  
    prettytable           2.5.0  
    prompt-toolkit        3.0.29  
    ptyprocess            0.7.0  
    pyasn1                0.4.8  
    Pygments              2.12.0  
    pyparsing             3.0.7  
    pyperclip             1.8.2  
    python-dateutil       2.8.2  
    pytz                  2021.3  
    PyWavelets            1.1.1  
    PyYAML                6.0  
    scikit-image          0.17.2  
    scikit-learn          0.24.2  
    scipy                 1.5.4  
    seaborn               0.11.2  
    setuptools            49.6.0.post20210108  
    shap                  0.40.0  
    six                   1.16.0  
    slicer                0.0.7  
    SQLAlchemy            1.4.36  
    statsmodels           0.12.2  
    stevedore             3.5.0  
    tenacity              8.0.1  
    threadpoolctl         3.1.0  
    tifffile              2020.9.3  
    torch                 1.10.1  
    torchvision           0.11.2  
    tqdm                  4.63.0  
    traitlets             4.3.3  
    typing_extensions     4.1.1  
    urllib3               1.26.8  
    wcwidth               0.2.5  
    wheel                 0.37.1  
    xgboost               1.5.2  
    zipp                  3.6.0  
