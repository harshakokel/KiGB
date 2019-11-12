import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import warnings
from sklearn.metrics import mean_squared_error
import logging
import sys
try:
    from core.lgbm.lkigb import LKiGB
    from experiments.regression.setting import *
except ImportError:
    sys.path.append('./')
    from core.lgbm.lkigb import LKiGB
    from experiments.regression.setting import *

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(message)s', level=logging.INFO)

data_list = ['abalone','autompg','autoprice','boston','california','cpu','crime','redwine','whitewine','windsor']


def get_error(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

result = ""

for dataset in data_list:
    dirName = './datasets/regression/' + dataset


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
            'objective': 'regression',
            'metric': 'l2',
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
        lkigb = LKiGB(lamda=lgb_data_penalty[dataset], epsilon=lgb_data_margin[dataset], max_depth=14, advice=advice, objective='regression', trees=30)
        lkigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(y_test, lkigb.predict(X_test))



    lmc_mse = np.mean(lmc_score)
    kigb_mse = np.mean(kigb_score)
    lmc_std = np.std(lmc_score)
    kigb_std = np.std(kigb_score)


    mc_test = ttest_rel(lmc_score, kigb_score)

    # logging.info( "DATASET, LKiGB MSE, LMC MSE, P-value" )
    # logging.info(dataset + "," + str(round(kigb_mse, 3)) + ',' + str(round(lmc_mse, 3)) + ',' + str(round(mc_test.pvalue, 3)))

    result = result +"\nFor '"+dataset+ "' dataset, LKiGB achieved mean-squared error of '"\
             + str(round(kigb_mse, 3)) + "' and LMC achieved mean-squared error of '"\
             +str(round(lmc_mse, 3))+"'."

logging.info(result)

