from core.scikit.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import RegressorMixin
import logging
from core.scikit import utils


class SKiGB(GradientBoostingRegressor, RegressorMixin):

    def _make_estimator(self, append=True):
        pass

    def __init__(self, criterion='mse',
                 n_estimators=35,
                 max_depth=10,
                 learning_rate=0.1,
                 loss='ls',
                 random_state=21,
                 advice=None,
                 lamda=1,
                 epsilon=0,
                 init=None, **kwargs):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.advice = advice
        self.epsilon = epsilon
        self.lamda = lamda
        self.init = init
        self.kigb = None

    def fit(self, X, y, sample_weight=None, monitor=None):
        logging.debug("Starting KiGB fit")
        utils.advice = self.advice
        utils.epsilon = self.epsilon
        utils.lamda = self.lamda
        utils.trees_modified = 0
        utils.node_violations = 0
        if self.loss == 'deviance':
            clf = GradientBoostingClassifier(criterion=self.criterion, n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             warm_start=True,
                                             learning_rate=self.learning_rate,
                                             loss=self.loss,
                                             random_state=self.random_state, init=self.init)
        else:
            clf = GradientBoostingRegressor(criterion=self.criterion, n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            warm_start=True,
                                            learning_rate=self.learning_rate,
                                            loss=self.loss,
                                            random_state=self.random_state, init=self.init)
        clf.fit(X, y, monitor=utils.kigb_penalty_update)
        self.kigb = clf
        if utils.trees_modified ==0:
            logging.info("No Trees Updated")
        logging.debug("Trees Modified: " + str(utils.trees_modified))
        logging.debug("Nodes Violation: " + str(utils.node_violations))
        logging.debug("finished KiGB fit")
        return self

    def predict(self, X, y=None):
        return self.kigb.predict(X)

    def feature_importance(self):
        return self.kigb.feature_importances_