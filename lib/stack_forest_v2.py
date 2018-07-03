import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split,cross_val_score
from sklearn.ensemble import BaseEnsemble
import logging
import uuid
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import gcforest.data_load as load
import gcforest.data_load_phy as load2
from mlxtend.classifier import StackingCVClassifier


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger

class CascadeForest():
    def __init__(self, estimators_config, folds=3,verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds
        self.logger = create_logger(self, verbose)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            estimators = [estimator_config['estimator_class'](**estimator_config['estimator_params'])
                          for estimator_config in self.estimators_config]

            predictions = []
            for estimator in estimators:
                estimator.fit(X, y)
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )
                predictions.append(prediction)
            # stacking
            lr = LogisticRegression(n_jobs=-1)
            clfs = [est for est in estimators]
            sclf = StackingCVClassifier(classifiers=clfs,
                                        meta_classifier=lr, use_probas=True, cv=self.folds,
                                        verbose=3)
            # sclf.fit(X, y)
            final_pred = cross_val_predict(sclf, X, y, cv=self.folds, method='predict_proba',
                              n_jobs=-1, )
            # final_pred = sclf.predict_proba(X)


            X = np.hstack([X] + predictions)
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )
            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]

            # for es in estimators:
            #     self.print_important_features(es)
            X = np.hstack([X] + predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )


if __name__ == "__main__":
    estimators_config = {
        'cascade': [{
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 200,
                'n_jobs': -1,
                'random_state': 0
            }
        },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 200,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            },
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 200,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 200,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            }
        ]
    }


    #
    # iris = load_iris()
    # X, Y = iris.data,iris.target
    X,Y = load.obesity_data()

    x_tr,x_te,y_tr,y_te = train_test_split(X,Y,stratify=Y,random_state=42)
    c_forest = CascadeForest(estimators_config['cascade'])
    c_forest.fit(x_tr,y_tr)
    y_pred = c_forest.predict(x_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(accuracy)

    rf = RandomForestClassifier(n_estimators=200,random_state=42)
    rf.fit(x_tr,y_tr)
    y_pred = rf.predict(x_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(accuracy)



