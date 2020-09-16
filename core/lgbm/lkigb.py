from lightgbm import LGBMModel
import lightgbm as lgb
from sklearn.base import RegressorMixin
import  logging

from core.lgbm.utils import kigb_penalty_update


class LKiGB(LGBMModel, RegressorMixin):

    def __init__(self, max_depth=-1,
                 learning_rate=0.1, n_estimators=1,
                 objective='regression', boosting_type='gbdt', lamda=1, epsilon=0, advice=None, trees=50, min_data=None, **kwargs):
        # self.num_leaves= num_leaves
        self.max_depth=max_depth
        self.learning_rate= learning_rate
        self.n_estimators=1
        self.objective= objective
        self.lamda=lamda
        self.epsilon=epsilon
        self.advice=advice
        self.kigb=None
        self.verbose=-1
        self.verbosity=-1
        self.trees=trees
        self.min_data=min_data
        self.boosting_type=boosting_type
        self._other_params = {}

    def fit(self, X, y=None):
        render = False
        zero_update=True
        logging.debug("Starting KiGB fit")
        lgb_train = lgb.Dataset(X, y, free_raw_data=False)
        param = self.get_params().copy()
        param.pop('trees')
        param.pop('lamda')
        param.pop('epsilon')
        param.pop('advice')
        # Learn first tree
        kigb_gbm = lgb.train(param,
                            lgb_train,
                            num_boost_round=1)
        if render: # Render tree in pdf for debugging
            graph = lgb.create_tree_digraph(kigb_gbm, tree_index=0, name='before_update_' + str(0))
            graph.render('./render/lgbm/before_update_' + str(0))
        # Update penalty values
        update = kigb_penalty_update(kigb_gbm, self.advice, lamda=self.lamda, epsilon=self.epsilon)
        if update:
            zero_update=False
            kigb_gbm.model_from_string(update, verbose=False)
            if render: # Rrender tree in pdf for debugging
                graph = lgb.create_tree_digraph(kigb_gbm, tree_index=0, name='after_update_' + str(0))
                graph.render('./render/lgbm/after_update_' + str(0))
        # iterate over trees.
        for h in range(1, self.trees + 1):
            lgb_train = lgb.Dataset(X, y, free_raw_data=False) # Bug in Lightgbm, need to initialize data
            # Learn next tree with initial model
            kigb_gbm = lgb.train(param,
                                lgb_train,
                                num_boost_round=1,
                                init_model=kigb_gbm)
            # If trees are not learnt further, break the loop
            if kigb_gbm.num_trees() <= h:
                logging.info("Trees are not learnt further")
                break
            if render: # Render tree for debugging
                graph = lgb.create_tree_digraph(kigb_gbm, tree_index=h, name='before_update_'+str(h))
                graph.render('./render/lgbm/before_update_'+str(h))
            # Update the penalty
            update = kigb_penalty_update(kigb_gbm, self.advice, h, lamda=self.lamda, epsilon=self.epsilon)
            if update:
                zero_update=False
                kigb_gbm.model_from_string(update, verbose=False)
                if render: # Render tree for debugging
                    graph = lgb.create_tree_digraph(kigb_gbm, tree_index=h, name='after_update_' + str(h))
                    graph.render('./render/lgbm/after_update_' + str(h))
        self.kigb = kigb_gbm
        if zero_update:
            logging.info("ZERO UPDATES")
        logging.debug("finished KiGB fit")
        return self

    def predict(self, X, y=None, num_iteration=-1):
        if self.objective == 'regression':
            return self.kigb.predict(X, num_iteration=num_iteration)
        else:
            return (self.kigb.predict(X, num_iteration=num_iteration) > 0.5).astype(int)

    def feature_importance(self):
        return self.kigb.feature_importance()