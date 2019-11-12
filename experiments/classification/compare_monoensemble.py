# Import libraries necessary for this project
import numpy as np
import pandas as pd
import logging
import sys
from monoensemble import MonoGradientBoostingClassifier
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
try:
    from core.scikit.gradient_boosting import GradientBoostingClassifier
    from core.scikit.skigb import SKiGB
    from experiments.classification.setting import *
except ImportError:
    sys.path.append('./')
    from core.scikit.gradient_boosting import GradientBoostingClassifier
    from core.scikit.skigb import SKiGB
    from experiments.classification.setting import *


data_list = ['australia','car','cleveland','ljubljana']


def get_error(clf_model, X, y):
    y_pred = clf_model.predict(X)
    return accuracy_score(y,y_pred)


logging.basicConfig(format='%(message)s', level=logging.INFO)
mono_coef_calc_type= 'boost'

for dataset in data_list:
    dirName = './datasets/classification/'+dataset
    test_data = pd.read_csv(dirName+'/test.csv')
    target =  data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    # Setting Parameter
    fold_size = 5

    kigb_score = np.zeros((fold_size), dtype=np.float64)
    mono_score = np.zeros((fold_size), dtype=np.float64)


    for fold in range(0, fold_size):
        train_data = pd.read_csv(dirName+'/train_'+str(fold)+'.csv')
        X_train = train_data.drop(target, axis=1)
        y_train = train_data[target]

        # Learn KiGB
        skigb = SKiGB(criterion='mse',
                      n_estimators=30,
                      max_depth=14,
                      learning_rate=0.1,
                      loss='deviance',
                      random_state=12,
                      advice=advice,
                      lamda=data_penalty[dataset],
                      epsilon=data_margin[dataset]
                      )

        skigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(skigb.kigb, X_test, y_test)


        # Learn Mono Ensemble
        incr_feats =  np.where(advice == +1)[0]
        decr_feats = np.where(advice == -1)[0]
        mono = MonoGradientBoostingClassifier(criterion='mse',
                                              n_estimators=30,
                                              max_depth=14,
                                              learning_rate=0.1,
                                              incr_feats=incr_feats,
                                              decr_feats=decr_feats,
                                              coef_calc_type=mono_coef_calc_type,
                                              random_state=12
                                              )
        mono.fit(X_train, y_train)
        mono_score[fold] =  get_error(mono, X_test, y_test)


    mono_accuracy = np.mean(mono_score)
    kigb_accuracy = np.mean(kigb_score)
    kigb_std = np.std(kigb_score)
    mono_std = np.std(mono_score)

    mono_ttest = ttest_rel(mono_score, kigb_score)


    # logging.info( "DATASET, KiGB Accuracy, Mono Accuracy, P-value" )
    # logging.info(dataset + "," + str(round(kigb_accuracy, 3)) + ',' + str(round(mono_accuracy, 3)) + ',' + str(round(mono_ttest.pvalue, 2)))

    logging.info("For '"+dataset+ "' dataset, SKiGB achieved accuracy of '"
                 + str(round(kigb_accuracy, 3)) + "' and Monoensemble achieved accuracy of '"+str(round(mono_accuracy, 3))+"'.")


