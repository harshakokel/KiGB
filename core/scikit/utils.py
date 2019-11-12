import numpy as np
# import pydotplus
import matplotlib.pyplot as plt
from collections import defaultdict

import pydotplus
import collections
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import logging

""" KiGB update of Scikit Gradient boosting Regressor"""
advice = None
epsilon = 0.0
lamda = 0.0


# Update terminal region with KiGB Penalty
def kigb_penalty_update(stage, gbr, fields):
    global trees_modified
    global node_violations
    y_pred = fields['y_pred']
    regressor = gbr.estimators_[stage][0]
    children_left = regressor.tree_.children_left
    children_right = regressor.tree_.children_right
    feature = regressor.tree_.feature
    samples = regressor.tree_.n_node_samples
    values = regressor.tree_.__getstate__()['values']
    delta_values = defaultdict(float)
    for feature_index in np.where(advice != 0)[0]:
        if feature_index in feature:
            node_idx_list = np.where(feature == feature_index)[0]
            for node_idx in node_idx_list:
                stack = [children_left[node_idx]]
                # Calculate Expected Value of left child
                lvalue = 0.0
                lsamples = 0.0
                while len(stack) > 0:
                    node_id = stack.pop()
                    # node is not leaf node
                    if children_left[node_id] != children_right[node_id]:
                        stack.append(children_left[node_id])
                        stack.append(children_right[node_id])
                    else:
                        lvalue += (values[node_id][0][0] * samples[node_id])
                        lsamples += (samples[node_id])
                lexpected = lvalue / lsamples
                # Calculate Expected Value of right child
                rvalue = 0.0
                rsamples = 0.0
                stack = [children_right[node_idx]]
                while len(stack) > 0:
                    node_id = stack.pop()
                    # node is not leaf node
                    if children_left[node_id] != children_right[node_id]:
                        stack.append(children_left[node_id])
                        stack.append(children_right[node_id])
                    else:
                        rvalue += (values[node_id][0][0] * samples[node_id])
                        rsamples += (samples[node_id])
                rexpected = rvalue / rsamples
                if advice[feature_index] > 0:
                    error = 'isotonic constraint not satisfied for tree ' + str(stage) + " node " + str(node_idx)
                    if lexpected > (rexpected + epsilon):
                        # Isotonic constraint violated, calculate bepsilon penalty
                        violation = lexpected - rexpected - epsilon
                        logging.debug(error)
                        node_violations = node_violations + 1
                        # left leaves penalty
                        l_samples = samples[children_left[node_idx]]
                        logging.debug(
                            'left penalty: ' + str(-(lamda * violation) / (2.0 * l_samples)) + ' sample: ' + str(
                                l_samples) + ' violation: ' + str(lexpected - rexpected))
                        stack = [children_left[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if children_left[node_id] != children_right[node_id]:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[node_id] = delta_values[node_id] - (lamda * violation) / (2.0 * l_samples)
                        # right leaves penalty
                        r_samples = samples[children_right[node_idx]]
                        logging.debug(
                            'right penalty: ' + str((lamda * violation) / (2.0 * r_samples)) + ' sample: ' + str(
                                r_samples) + ' violation: ' + str(lexpected - rexpected))
                        stack = [children_right[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if children_left[node_id] != children_right[node_id]:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[node_id] = delta_values[node_id] + (lamda * violation) / (2.0 * r_samples)
                else:
                    error = 'antitonic constraint not satisfied for tree ' + str(stage) + " node " + str(node_idx)
                    if (lexpected + epsilon) < rexpected:
                        # Antitonic constraint violated, calculate beta penalty
                        violation = rexpected - lexpected - epsilon
                        logging.debug(error)
                        node_violations = node_violations + 1
                        # left leaves penalty
                        l_samples = samples[children_left[node_idx]]
                        logging.debug(
                            'left penalty: ' + str((lamda * violation) / (2.0 * l_samples)) + ' sample: ' + str(
                                l_samples) + ' violation: ' + str(rexpected - lexpected))
                        stack = [children_left[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if children_left[node_id] != children_right[node_id]:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[node_id] = delta_values[node_id] + (lamda * violation) / (2.0 * l_samples)
                        # right leaves penalty
                        stack = [children_right[node_idx]]
                        r_samples = samples[children_right[node_idx]]
                        logging.debug(
                            'right penalty: ' + str(-(lamda * violation) / (2.0 * r_samples)) + ' sample: ' + str(
                                r_samples) + ' violation: ' + str(rexpected - lexpected))
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if children_left[node_id] != children_right[node_id]:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[node_id] = delta_values[node_id] - (lamda * violation) / (2.0 * r_samples)
    if len(delta_values.keys()) > 0:
        trees_modified = trees_modified + 1
        X_train = fields['X']
        # export_tree(gbr.estimators_.flatten()[stage], features_list, 'temp/before_tree_'+str(stage)+'.png')
        for idx in delta_values.keys():
            logging.debug("Updating node " + str(idx) + " prev: " + str(values[idx][0][0]) + " new: " + str(
                values[idx][0][0] + delta_values[idx]))
            values[idx][0][0] = values[idx][0][0] + delta_values[idx]
        decision = regressor.apply(X_train)
        y_updated_pred = map(lambda x: [x[1] - delta_values[decision[x[0]]]], enumerate(y_pred))
        # export_tree(gbr.estimators_.flatten()[stage], features_list, 'temp/after_tree_'+str(stage)+'.png')
        return np.reshape(list(y_updated_pred), (-1, 1))
    return y_pred

def mse_score(clf, X_test, y_test):
    """compute stepwise scores on ``X_test`` and ``y_test``. """
    score = np.zeros((clf.n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf._staged_decision_function(X_test)):
        score[i] = mean_squared_error(y_test, np.reshape(y_pred, (1, -1))[0])
    return score
