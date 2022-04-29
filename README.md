# AutoML_processing
Tree base ML processing
# 개요

롯데손해보험 프로젝트를 진행하며 구성한 ML 스크립트 구조가

tree기반 한정으로 AutoML 구조와 유사하게 최소한의 데이터 설정 만으로 모델 알고리즘 선택부터 최종 output 산출까지 가능한 형태로 구성되었습니다.

해당 자료를 공유하여 이후 ML 프로젝트의 효율성 향상에 도움이 되면 좋겠습니다.

# 📃 Contents

---

아직 완성도가 높은 자료가 아니기 때문에 상세 코드는 직접 확인하셔서 검토해보는 것을 가정하여 여기서는 간단한 사용 방법에 대해 기술하겠습니다.

1. 데이터 정의

사용 데이터를 정의하는 항목입니다.

스크립트 내 Utiles > query.json 파일 내부에 json 형태로 각종 인자를 설정하여 사용할 수 있도록 구성되어 있습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c32f1b73-72de-47e2-80d5-83967eb62874/Untitled.png)

- MODEL_TYPE : 분류 or 회귀 문제를 설정하는 값으로 [ classification, regression ] 2가지 인자가 지원됩니다.
- DATA_TYPE : 데이터를 불러올 방식을 설정하는 값이며 [ SQL, file] 2가지 방법이 존재합니다.
    - SQL 방식 선택 시 내부에서 DB connect 연결 설정이 필요하며 SELECT 할 변수 리스트 선택이 가능해집니다.
- TRAIN_TABLE : 학습 데이터셋으로 SQL 선택 시 from 절의 테이블 명을, file 선택 시 경로 상의 file명을 입력합니다.
- TEST_TABLE : 테스트 데이터셋으로 SQL 선택 시 from 절의 테이블 명을, file 선택 시 경로 상의 file명을 입력합니다.
- TARGET : 예측할 타겟 변수명을 입력합니다.
- KEY_COLS : ID와 같은 KEY 변수를 입력합니다. 해당 변수는 학습 시 제외됩니다.
- SQL_TRAIN_COLS : SQL 타입 선택 시 불러올 데이터의 변수를 지정할 수 있습니다. 해당 값에서는 TARGET과 KEY_COLS 변수는 입력하지 않습니다.

1. 알고리즘 선택 및 튜닝


터미널에서 python [Optuna.py](http://Optuna.py) 명령어로 Optuna를 실행시켜 알고리즘 선택 및 튜닝을 진행합니다.

Optuna 인자에 XGBoost, LightGBM, CatBoost 선택 항목을 추가하여 3가지 Tree 알고리즘 중 적합한 모델 선택에 대해서도 자동으로 진행 가능합니다.

1. 모델 학습

터미널에서 python model_train[.py](http://Optuna.py) 명령어로 최종 모델 학습을 진행합니다.

학습 후 ML_result 폴더 내에서 importance plot, ROC curve, PR curve, SHAP plot 과 모델 스코어값을 저장한 csv 파일을 확인할 수 있습니다.

- Importance plot


- ROC curve


- SHAP plot


1. 모델 적용

터미널에서 python model_test[.py](http://Optuna.py) 명령어로 최종 모델로 test 데이터에 대한 예측을 진행합니다.

예측이 완료되면 ML_result 폴더에 최종 output 파일이 생성되며 test 데이터에 모델 예측값이 추가된 형태의 데이터가 생성됩니다.


# 🌈 Reference

---

- 사용 환경
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
