import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import warnings
from sklearn.metrics import accuracy_score
import sys
import logging
try:
    from core.lgbm.lkigb import LKiGB
    from experiments.classification.setting import *
except ImportError:
    sys.path.append('./')
    from core.lgbm.lkigb import LKiGB
    from experiments.classification.setting import *


warnings.filterwarnings("ignore")
logging.basicConfig(format='%(message)s', level=logging.INFO)

"""Run comparision of LIGHTBM, LightGBM with inbuilt monotonicity constraint and KiGB """

data_list = ['australia','car','cleveland','ljubljana']

def get_error(y_test, y_pred):
    return accuracy_score(y_test, (y_pred > 0.5).astype(int))


result =""

for dataset in data_list:
    dirName = './datasets/classification/' + dataset

    test_data = pd.read_csv(dirName + '/test.csv')
    target = data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)
    fold_size = 5

    lmc_score = np.zeros((fold_size), dtype=np.float64)
    kigb_score = np.zeros((fold_size), dtype=np.float64)


    for fold in range(0,fold_size):
        train_data = pd.read_csv(dirName + '/train_' + str(fold) + '.csv')
        X_train = train_data.drop(target, axis=1)
        y_train = train_data[target]



        #  LightGBM Monotonicity
        mc_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_error',
            'learning_rate': 0.1,
            'max_depth': 14,
            'monotone_constraints': data_advice[dataset],
            'verbose':-1,
            'verbosity': -1
        }

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        LMC = lgb.train(mc_params,
                        lgb_train,
                        num_boost_round=30)

        lmc_score[fold] = get_error(y_test, LMC.predict(X_test))




        # KiGB
        advice  = np.array(data_advice[dataset].split(','), dtype=int)
        lkigb = LKiGB(zeta=lgb_penalty[dataset], epsilon=lgb_margin[dataset], max_depth=14, advice=advice, objective='binary', trees=30)
        lkigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(y_test, lkigb.kigb.predict(X_test))



    lmc_accuracy = np.mean(lmc_score)
    kigb_accuracy = np.mean(kigb_score)
    lmc_std = np.std(lmc_score)
    kigb_std = np.std(kigb_score)


    mc_ttest = ttest_rel(lmc_score, kigb_score)

    result = result +"\nFor '" + dataset + "' dataset, LKiGB achieved accuracy of '" + str(
        round(kigb_accuracy, 3)) + "' and LMC achieved accuracy of '" + str(round(lmc_accuracy, 3)) + "'."


logging.info(result)