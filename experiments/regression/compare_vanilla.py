# Import libraries necessary for this project
import numpy as np
import pandas as pd
import logging
from scipy.stats import ttest_rel
from sklearn.metrics import mean_squared_error
import sys
try:
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from core.scikit.skigb import SKiGB
    from experiments.regression.setting import *
except ImportError:
    sys.path.append('./')
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from core.scikit.skigb import SKiGB
    from experiments.regression.setting import *

logging.basicConfig(format='%(message)s', level=logging.INFO)

data_list = ['abalone','autompg','autoprice','boston','california','cpu','crime','redwine','whitewine','windsor']

def get_error(reg_model, X, y):
    y_pred = reg_model.predict(X)
    return mean_squared_error(y,y_pred)



for dataset in data_list:
    dirName = './datasets/regression/' + dataset
    test_data = pd.read_csv(dirName+'/test.csv')
    target =  data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    # Setting Parameter
    fold_size = 5

    kigb_score = np.zeros((fold_size), dtype=np.float64)
    gb_score = np.zeros((fold_size), dtype=np.float64)



    for fold in range(0, fold_size):
        train_data = pd.read_csv(dirName+'/train_'+str(fold)+'.csv')
        X_train = train_data.drop(target, axis=1)
        y_train = train_data[target]

        # Learn KiGB
        skigb = SKiGB(criterion='mse',
                      n_estimators=30,
                      max_depth=10,
                      learning_rate=0.1,
                      loss='ls',
                      random_state=12,
                      advice=advice,
                      lamda=data_penalty[dataset],
                      epsilon=data_margin[dataset]
                      )

        skigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(skigb.kigb, X_test, y_test)



        # Learn GB
        reg = GradientBoostingRegressor(criterion='mse',
                                        n_estimators=30,
                                        max_depth=10,
                                        learning_rate=0.1,
                                        loss='ls',
                                        random_state=12
                                        )
        reg.fit(X_train, y_train)
        gb_score[fold] = get_error(reg, X_test, y_test)




    gb_mse = np.mean(gb_score)
    kigb_mse = np.mean(kigb_score)
    gb_std = np.std(gb_score)
    kigb_std = np.std(kigb_score)



    ttest = ttest_rel(gb_score, kigb_score)


    # logging.info( "DATASET, KiGB MSE, GB MSE, P-value" )
    # logging.info(dataset + "," + str(round(kgb_mse, 3)) + "," + str(round(gb_mse, 3)) + "," + str(round(ttest.pvalue, 3)))

    logging.info("For '" + dataset + "' dataset, SKiGB achieved mean-squared error of '" + str(
        round(kigb_mse, 3)) + "' and SGB achieved mean-squared error of '" + str(round(gb_mse, 3))+"'.")



