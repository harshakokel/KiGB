import re
from builtins import float
from collections import defaultdict
import logging
import numpy as np


""" KiGB updation of LIGHT GBM Trees"""
END_OF_TREES = "end of trees"
TREE_START = "Tree="
LEAF_COUNT = "leaf_count"
LEAF_VALUE = "leaf_value"
LEFT_CHILD = "left_child"
RIGHT_CHILD = "right_child"
DECISION_TYPE = "decision_type"
SPLIT_FEATURE_ = "split_feature*"


def check_decision_type(tree):
    """Check the decision type for all the nodes is `<=`

    :param gbm: lightgbm model
    :param tree:
    """
    r = re.compile(DECISION_TYPE)
    if not list(filter(r.match, tree))[0].split(sep="=")[1]:
        return False
    if set(list(filter(r.match, tree))[0].split(sep="=")[1].split()) != {'2'}:
        raise KiGBError("Decisions are not symmetric")
    return True

def get_split_feature(tree):
    r = re.compile(SPLIT_FEATURE_)
    return np.array(list(filter(r.match, tree))[0].split(sep="=")[1].split(), dtype=int)


def get_left_child(tree):
    r = re.compile(LEFT_CHILD)
    return np.array(list(filter(r.match, tree))[0].split(sep="=")[1].split(), dtype=int)


def get_right_child(tree):
    r = re.compile(RIGHT_CHILD)
    return np.array(list(filter(r.match, tree))[0].split(sep="=")[1].split(), dtype=int)


def get_leaf_values(tree):
    r = re.compile(LEAF_VALUE)
    return np.array(list(filter(r.match, tree))[0].split(sep="=")[1].split(), dtype=float)

def update_leaf_values(full_tree, values, lower, upper, str_model):
    tree= full_tree[lower:upper]
    r = re.compile(LEAF_VALUE)
    old_value = list(filter(r.match, tree))[0]
    new_value = LEAF_VALUE +'='+ np.array2string(values, separator=' ').replace('\n','').replace('[','').replace(']','')
    logging.debug("old leaves: "+ old_value)
    logging.debug("new leaves: " + new_value)
    return str_model.replace(old_value, new_value)


def get_leaf_samples(tree):
    r = re.compile(LEAF_COUNT)
    return np.array(list(filter(r.match, tree))[0].split(sep="=")[1].split(), dtype=float)


def get_boundaries(tree, iteration):
    lower = np.where(tree == TREE_START + str(iteration))[0][0]
    upper = np.where(tree == END_OF_TREES)[0][0]
    return lower, upper


def kigb_penalty_update(gbm, advice, iteration=0, epsilon=0, lamda=0.2):
    tree = np.array(gbm.model_to_string().splitlines())
    lower, upper = get_boundaries(tree, iteration)
    if lamda==0:
        return False
    # Check all decision types
    if not check_decision_type(tree[lower:upper]):
        logging.info("Tree is just a leaf or has different decision types")
        return False
    feature = get_split_feature(tree[lower:upper])
    children_left = get_left_child(tree[lower:upper])
    children_right = get_right_child(tree[lower:upper])
    samples = get_leaf_samples(tree[lower:upper])
    values = get_leaf_values(tree[lower:upper])
    delta_values = defaultdict(float)
    for feature_index in np.where(advice != 0)[0]: # Get all the features with constraint
        if feature_index in feature: # check if the tree has those features
            node_idx_list = np.where(feature == feature_index)[0]  # get indexes
            for node_idx in node_idx_list:
                stack = [children_left[node_idx]]
                # Calculate Expected Value of left child
                lvalue = 0.0
                lsamples = 0.0
                while len(stack) > 0:
                    node_id = stack.pop()
                    # node is not leaf node
                    if node_id >= 0:
                        stack.append(children_left[node_id])
                        stack.append(children_right[node_id])
                    else:
                        lvalue += (values[(1+node_id)*-1] * samples[(1+node_id)*-1])
                        lsamples += (samples[(1+node_id)*-1])
                lexpected = lvalue / lsamples
                # Calculate Expected Value of right child
                rvalue = 0.0
                rsamples = 0.0
                stack = [children_right[node_idx]]
                while len(stack) > 0:
                    node_id = stack.pop()
                    # node is not leaf node
                    if node_id >= 0:
                        stack.append(children_left[node_id])
                        stack.append(children_right[node_id])
                    else:
                        rvalue += (values[(1+node_id)*-1] * samples[(1+node_id)*-1])
                        rsamples += (samples[(1+node_id)*-1])
                rexpected = rvalue / rsamples
                if advice[feature_index] > 0: # Isontonic
                    error = 'isotonic constraint not satisfied for tree ' \
                            + str(iteration) + " node " + str(node_idx)
                    if lexpected > (rexpected + epsilon):
                        violation = lexpected-rexpected - epsilon
                        logging.debug( error )
                        #left leaves penalty
                        l_samples = samples[children_left[node_idx]]
                        logging.debug('left penalty: ' + str(-(lamda * violation) / (2.0 * l_samples)) +
                              ' sample: ' + str( l_samples) + ' violation: ' + str(lexpected-rexpected ))
                        stack = [children_left[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if node_id >= 0:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[(1+node_id)*-1] = delta_values[(1+node_id)*-1] \
                                                               - (lamda * violation) / (2.0 * l_samples)
                        # right leaves penalty
                        r_samples = samples[children_right[node_idx]]
                        logging.debug('right penalty: ' + str((lamda * violation) / (2.0 * r_samples)) + ' sample: ' + str(r_samples) +
                              ' violation: ' + str(lexpected - rexpected))
                        stack = [children_right[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if node_id >= 0:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[(1+node_id)*-1] = delta_values[(1+node_id)*-1] \
                                                               + (lamda * violation) / (2.0 * r_samples)
                else:
                    error = 'antitonic constraint not satisfied for tree ' \
                            + str(iteration) + " node " + str(node_idx)
                    if (lexpected + epsilon) < rexpected:
                        # Antitonic constraint violated, calculate beta penalty
                        violation = rexpected - lexpected - epsilon
                        logging.debug(error)
                        # left leaves penalty
                        l_samples = samples[children_left[node_idx]]
                        logging.debug('left penalty: ' + str((lamda * violation) / (2.0 * l_samples)) + ' sample: ' + str(l_samples) +
                              ' violation: ' + str( rexpected - lexpected))
                        stack = [children_left[node_idx]]
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if node_id >= 0:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[(1+node_id)*-1] = delta_values[(1+node_id)*-1] \
                                                               + (lamda * violation) / (2.0 * l_samples)
                        # right leaves penalty
                        stack = [children_right[node_idx]]
                        r_samples = samples[children_right[node_idx]]
                        logging.debug('right penalty: ' + str(-(lamda * violation) / (2.0 * r_samples)) +
                              ' sample: ' + str(r_samples) +
                              ' violation: ' + str(rexpected - lexpected))
                        while len(stack) > 0:
                            node_id = stack.pop()
                            # node is not leaf node
                            if node_id >= 0:
                                stack.append(children_left[node_id])
                                stack.append(children_right[node_id])
                            else:
                                delta_values[(1+node_id)*-1] = delta_values[(1+node_id)*-1] - (lamda * violation) / (
                                            2.0 * r_samples)
    if len(delta_values.keys()) > 0:
        # export_tree(gbr.estimators_.flatten()[stage], features_list, 'temp/before_tree_'+str(stage)+'.png')
        for idx in delta_values.keys():
            # logging.debug("Updating leaf "+ str(idx) +" prev: " + str(values[idx]) + " new: "+  str(values[idx] + delta_values[idx]))
            values[idx] = values[idx] + delta_values[idx]
        return update_leaf_values(tree,values, lower, upper,gbm.model_to_string())
    logging.debug("No Knowledge updates in tree "+str(iteration))
    return False

class KiGBError(Exception):
    """Error thrown by LKiGB ."""

    pass
