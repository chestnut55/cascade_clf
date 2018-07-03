import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split,cross_val_score
from sklearn.ensemble import BaseEnsemble
import logging
import uuid
from sklearn.datasets import load_iris

import gcforest.data_load as load
import gcforest.data_load_phy as load2


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger

class StackingCascadeForest():
    def __init__(self, estimators_config, folds=3,verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds
        self.logger = create_logger(self, verbose)

    def _model(self,model_name):
        if model_name =='RF':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1)
        elif model_name=='ET':
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(n_estimators=500,random_state=0,max_features=1,n_jobs=-1)
        else:
            pass
        return model

    def _get_oof(self, clf, n_folds, x_train, y_train):
        n_train = x_train.shape[0]
        n_classes = len(np.unique(y_train))
        cv = StratifiedKFold(n_splits=n_folds, random_state=0)
        oof_train = np.zeros((n_train, n_classes))

        for i, (tr, te) in enumerate(cv.split(x_train,y_train)):
            cv_x_train = x_train[tr]
            cv_y_train = y_train[tr]

            cv_x_test = x_train[te]

            clf.fit(cv_x_train, cv_y_train)
            oof_train[te] = clf.predict_proba(cv_x_test)
        return oof_train

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.sublevels = []
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

            ## stack learning
            new_feature = reduce(lambda x, y: np.concatenate((x, y), axis=1), predictions)

            meta_learners = []
            model_name = 'RF'
            meta_learner1 = self._model(model_name)
            # new_features2 = self._get_oof(meta_learner, 3, new_feature, y)
            meta_learner1.fit(new_feature, y)
            new_features2 = cross_val_predict(meta_learner1,new_feature,y,cv=self.folds,method='predict_proba',n_jobs=-1)
            predictions.append(new_features2)
            meta_learners.append(meta_learner1)



            model_name = 'ET'
            meta_learner2 = self._model(model_name)
            # new_features3 = self._get_oof(meta_learner, 3, new_feature, y)
            meta_learner2.fit(new_feature, y)
            new_features3 = cross_val_predict(meta_learner2, new_feature, y, cv=self.folds, method='predict_proba',
                                              n_jobs=-1)
            predictions.append(new_features3)
            meta_learners.append(meta_learner2)

            tmp = []
            tmp.append(new_features2)
            tmp.append(new_features3)

            stacking_new_feature2 = reduce(lambda x, y: np.concatenate((x, y), axis=1), tmp)
            model_name = 'RF'
            meta_learner3 = self._model(model_name)
            # new_features2 = self._get_oof(meta_learner, 3, new_feature, y)
            meta_learner3.fit(stacking_new_feature2, y)
            new_features4 = cross_val_predict(meta_learner3,stacking_new_feature2,y,cv=self.folds,method='predict_proba',n_jobs=-1)
            predictions.append(new_features4)
            meta_learners.append(meta_learner3)

            model_name = 'ET'
            meta_learner4 = self._model(model_name)
            # new_features3 = self._get_oof(meta_learner, 3, new_feature, y)
            meta_learner4.fit(stacking_new_feature2, y)
            new_features5 = cross_val_predict(meta_learner4, stacking_new_feature2, y, cv=self.folds, method='predict_proba',
                                              n_jobs=-1)
            predictions.append(new_features5)
            meta_learners.append(meta_learner4)

            # aa=(new_features2 + new_features3)/2
            # predictions.append(aa)


            # predictions = np.asarray(predictions)
            # predictions = np.mean(predictions,axis=0)
            # predictions = [predictions]


            X = np.hstack([X] + predictions)
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )
            # y_prediction = np.array(predictions).mean(axis=0).argmax(axis=1)
            # y_prediction = [0 if x % 2 == 0 else 1 for x in y_prediction]
            # y_prediction = [x % len(self.classes) for x in y_prediction]
            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
                self.sublevels.append(meta_learners)
            else:
                break

    def predict(self, X):
        for (i,estimators) in enumerate(self.levels):
            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]

            test_new_feature = reduce(lambda x, y: np.concatenate((x, y), axis=1), predictions)
            meta_learner = self.sublevels[i]
            pred0 = meta_learner[0].predict_proba(test_new_feature)
            pred1 = meta_learner[1].predict_proba(test_new_feature)

            predictions.append(pred0)
            predictions.append(pred1)

            tmp1 = []
            tmp1.append(pred0)
            tmp1.append(pred1)

            tmp1 = reduce(lambda x, y: np.concatenate((x, y), axis=1), tmp1)
            pred3=meta_learner[2].predict_proba(tmp1)
            pred4=meta_learner[3].predict_proba(tmp1)

            predictions.append(pred3)
            predictions.append(pred4)
            # bb = (pred0+pred1)/2
            # predictions.append(bb)

            # predictions = np.asarray(predictions)
            # predictions = np.mean(predictions,axis=0)
            # predictions = [predictions]

            X = np.hstack([X] + predictions)

        y_prediction = self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

        # _y = np.array(predictions).mean(axis=0).argmax(axis=1)

        # y_prediction = [x % len(self.classes) for x in _y]
        return y_prediction


if __name__ == "__main__":
    estimators_config = {
        'cascade': [{
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 500,
                'n_jobs': -1,
                'random_state': 0
            }
        },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'max_features':1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'max_features': 1,
                    'random_state': 0
                }
            }
        ]
    }


    #
    # iris = load_iris()
    # X, Y = iris.data,iris.target
    for i in range(10):
        X,Y = load2.yeast_data()

        x_tr,x_te,y_tr,y_te = train_test_split(X,Y,stratify=Y,test_size=0.3)
        c_forest = StackingCascadeForest(estimators_config['cascade'])
        c_forest.fit(x_tr,y_tr)
        y_pred = c_forest.predict(x_te)
        accuracy = accuracy_score(y_te, y_pred)
        print(accuracy)

        rf = RandomForestClassifier(n_estimators=500,random_state=0)
        rf.fit(x_tr,y_tr)
        y_pred = rf.predict(x_te)
        accuracy = accuracy_score(y_te, y_pred)
        print(accuracy)



